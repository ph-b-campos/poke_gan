"""Mix de domínio entre espécies, StyleGAN-like e adaptado a sprites: a parte grossa do Pikachu, a fina do Lugia.

Por que não basta o DCGAN condicional (pokemon_gan.py, modo cwgp): ele aprende cada espécie (FID 44,5), mas
decora cada uma como um ponto isolado, e misturar camadas entre espécies dá ruído. O dataset tem 40.597 imagens
e só ~3.900 sprites distintos (mediana de 3 por espécie). Daqui vêm as peças da StyleGAN (Karras et al. 2019; 2020):
    mapeamento  (z, espécie) -> w por uma MLP
    constante   a imagem nasce de um tensor 4x4 aprendido; (z, espécie) só entram como estilo
    modulação   cada convolução é escalada pelo estilo da sua resolução (modulação e demodulação)
    mistura     no treino, metade do lote usa o w de uma espécie nas posições grossas e o de outra nas finas;
                o discriminador julga essas só por "parece Pokémon?", sem espécie
E as adaptações ao domínio (sprite = regiões de cor chapada, contorno escuro, fundo branco):
    paleta      (modo "paleta") a saída não pinta RGB livre: cada pixel escolhe um de K slots de cor, e as K
                cores vêm de um estilo próprio, a posição mais fina. Pikachugia = regiões do Pikachu com a
                paleta do Lugia. Um slot fixo é o fundo branco.
    aumento     espelho horizontal nos reais; DiffAugment (Zhao et al. 2020) de cor, translação (borda branca,
                como o fundo) e recorte, iguais em reais e falsos
    classe      cabeça classificadora no D (AC-GAN, Odena et al. 2017), treinada só nos sprites reais; o G é
                cobrado para que suas amostras puras sejam da espécie pedida. Sem ela (rodada cor_v1, 17/11) a
                projeção sozinha não ensinou a espécie: o juiz acertava 0,7% das amostras em 297 kimg
    espécies    só as gerações 1 a 3 (Pokédex 1 a 386), que contêm todos os pares das figuras
    transição   poucos sprites por espécie fazem a StyleGAN2 condicional colapsar (Shahbazi et al. 2022, ICLR);
                a espécie entra aos poucos: peso lam de 0 a 1 no mapeamento e na projeção do D
O modo "estilo" é a mesma rede com saída RGB por resolução (StyleGAN2 "skip"): o teste de comparação.

    python estilo_gan.py teste                                # autoteste em CPU
    python estilo_gan.py treina --modo paleta --horas 8       # retoma do checkpoint, se existir
    python estilo_gan.py figuras --modo paleta
"""
import argparse
import copy
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pokemon_gan import CACHE, N_ESP, RES, Juiz, avalia, carrega, dispositivo, norm, referencia

Z_DIM = W_DIM = 256
RESOLUCOES = (4, 8, 16, 32, 64)
CH = {4: 256, 8: 256, 16: 256, 32: 128, 64: 64}
K_PALETA = 12                          # slots de cor, fora o fundo; sprites têm ~10 a 16 cores
CONFIG = {                             # StyleGAN2-ADA (Karras et al. 2020) adaptada a 64x64 e a uma GPU de 6 GB
    # gamma 0,1 e aumento de cor: com 0,05 e sem cor o D dominou em 12 kimg (teste de saúde de 17/11); o DiffAugment
    # aplica a cor igual em reais e falsos, sem vazar para o gerador (Zhao et al. 2020)
    "lote": 32, "lr": 0.0025, "betas": (0.0, 0.99), "gamma": 0.1, "r1_cada": 16, "mapa": 4,
    "mistura": 0.5, "ema_kimg": 10, "aumento": "cor,translacao,recorte", "aval_min": 15,
    "transicao_kimg": (128, 256),       # Shahbazi et al.: 2k a 4k iterações de lote 64; aqui em mil imagens
    "especies": 386,                    # gerações 1 a 3
    "classe": 1.0,                      # peso da entropia cruzada da cabeça classificadora
}
SAIDA = {"cor": "cor", "estilo": "rgb", "paleta": "paleta"}             # modo da linha de comando -> saída do G
SQ2 = math.sqrt(2)


def lrelu(x):
    return F.leaky_relu(x, 0.2) * SQ2


def norm2(x):
    return x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()


# ------------------------------------------------------------------ camadas com taxa equalizada (Karras et al. 2018)
class Linear(nn.Module):
    def __init__(self, i, o, lr_mul=1.0, bias_init=0.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i) / lr_mul)
        self.b = nn.Parameter(torch.full((o,), float(bias_init)))
        self.g, self.lr_mul = lr_mul / math.sqrt(i), lr_mul

    def forward(self, x):
        return F.linear(x, self.w * self.g, self.b * self.lr_mul)


class Conv(nn.Module):
    def __init__(self, i, o, k, bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i, k, k))
        self.b = nn.Parameter(torch.zeros(o)) if bias else None
        self.g, self.p = 1 / math.sqrt(i * k * k), k // 2

    def forward(self, x):
        return F.conv2d(x, self.w * self.g, self.b, padding=self.p)


class ModConv(nn.Module):
    """Convolução modulada (StyleGAN2): o estilo escala cada canal de entrada; a demodulação devolve a variância a 1.
    Feita modulando a ativação, não o peso: mesma conta, sem convolução agrupada."""
    def __init__(self, i, o, k, demod=True, up=False):
        super().__init__()
        self.afim = Linear(W_DIM, i, bias_init=1)
        self.w = nn.Parameter(torch.randn(o, i, k, k))
        self.g, self.p, self.demod, self.up = 1 / math.sqrt(i * k * k), k // 2, demod, up

    def forward(self, x, w):
        s = self.afim(w)                                                   # (B, i): o estilo desta camada
        peso = self.w * self.g
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.conv2d(x * s[:, :, None, None], peso, padding=self.p)
        if self.demod:
            d = ((peso[None] * s[:, None, :, None, None]).square().sum([2, 3, 4]) + 1e-8).rsqrt()
            x = x * d[:, :, None, None]
        return x


class Camada(nn.Module):
    def __init__(self, i, o, up=False):
        super().__init__()
        self.conv = ModConv(i, o, 3, up=up)
        self.ruido = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(o))

    def forward(self, x, w):
        x = self.conv(x, w)
        x = x + torch.randn(len(x), 1, *x.shape[2:], device=x.device) * self.ruido
        return lrelu(x + self.b[None, :, None, None])


# ------------------------------------------------------------------ gerador
class Mapeamento(nn.Module):
    def __init__(self, camadas):
        super().__init__()
        self.emb = nn.Embedding(N_ESP, Z_DIM)
        dims = [2 * Z_DIM] + [W_DIM] * camadas
        self.fc = nn.ModuleList([Linear(a, b, lr_mul=0.01) for a, b in zip(dims[:-1], dims[1:])])
        self.lam = 1.0                                                     # peso da espécie (treino transicional)

    def forward(self, z, y):
        x = torch.cat([norm2(z), self.lam * norm2(self.emb(y))], 1)
        for f in self.fc:
            x = lrelu(f(x))
        return x


class Gerador(nn.Module):
    """Posições de estilo: uma por resolução (4, 8, 16, 32, 64) e, nos modos paleta e cor, mais uma para a cor.
    saida = "rgb":    StyleGAN2 skip, cada toRGB usa o estilo da sua resolução
            "cor":    StyleGAN2 skip, todas as toRGB usam a posição de cor (no StyleSpace, Wu et al. 2021, os
                      estilos das toRGB só mudam cor, nunca forma): estrutura de A + cor de B
            "paleta": K slots de cor por softmax; FALHOU (16/11): o slot branco virou estado absorvente,
                      o gerador pintou tudo de fundo, o softmax saturou e o gradiente sumiu"""
    def __init__(self, saida="cor", mapa=4):
        super().__init__()
        self.saida = saida
        self.n_ws = len(RESOLUCOES) + (saida in ("paleta", "cor"))
        self.mapa = Mapeamento(mapa)
        self.const = nn.Parameter(torch.randn(CH[4], 4, 4))
        self.blocos, ant = nn.ModuleList(), CH[4]
        for r in RESOLUCOES:
            self.blocos.append(nn.ModuleList(([Camada(ant, CH[r], up=True)] if r > 4 else []) + [Camada(CH[r], CH[r])]))
            ant = CH[r]
        if saida != "paleta":
            self.rgb = nn.ModuleList([ModConv(CH[r], 3, 1, demod=False) for r in RESOLUCOES])
            self.rgb_b = nn.Parameter(torch.zeros(len(RESOLUCOES), 3))
        else:
            self.slots = ModConv(CH[64], K_PALETA + 1, 1, demod=False)     # +1: o fundo branco
            self.slots_b = nn.Parameter(torch.zeros(K_PALETA + 1))
            self.cor1, self.cor2 = Linear(W_DIM, W_DIM), Linear(W_DIM, 3 * K_PALETA)

    def ws(self, z, y):
        return self.mapa(z, y)[:, None].repeat(1, self.n_ws, 1)

    def paleta(self, w):
        """Slot 0: fundo branco. Slots 1 a K: as cores em ordem de luminância, para o slot i ter o mesmo papel em
        toda espécie (o claro do Pikachu recebe o claro do Lugia, o contorno escuro recebe o escuro)."""
        cores = torch.tanh(self.cor2(lrelu(self.cor1(w)))).view(len(w), K_PALETA, 3)
        lum = cores @ torch.tensor([0.299, 0.587, 0.114], device=w.device)
        cores = cores.gather(1, lum.argsort(1)[..., None].expand(-1, -1, 3))
        return torch.cat([torch.ones_like(cores[:, :1]), cores], 1)

    def regioes(self, x, w):
        return (self.slots(x, w) + self.slots_b[None, :, None, None]).softmax(1)

    def sintese(self, ws, devolve_regioes=False):
        x = self.const[None].expand(len(ws), -1, -1, -1)
        img = None
        for k, bloco in enumerate(self.blocos):
            for cam in bloco:
                x = cam(x, ws[:, k])
            if self.saida != "paleta":
                y = self.rgb[k](x, ws[:, -1] if self.saida == "cor" else ws[:, k]) + self.rgb_b[k][None, :, None, None]
                img = y if img is None else F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False) + y
        if self.saida != "paleta":
            return img
        a = self.regioes(x, ws[:, len(RESOLUCOES) - 1])                       # (B, K+1, 64, 64)
        img = torch.einsum("bkhw,bkc->bchw", a, self.paleta(ws[:, -1]))
        return (img, a) if devolve_regioes else img

    def forward(self, z, y):
        return self.sintese(self.ws(z, y))


def mistura(G, z, y, p):
    """Mixing regularization entre espécies: posições grossas de (z, y), finas de (z2, y2), corte aleatório.
    Devolve os ws e o rótulo para o discriminador (-1 nas misturadas: não têm espécie)."""
    ws = G.ws(z, y)
    mix = torch.rand(len(y), device=y.device) < p
    w2 = G.mapa(torch.randn_like(z), y[torch.randperm(len(y), device=y.device)])
    k = torch.randint(1, G.n_ws, (len(y),), device=y.device)
    fina = (torch.arange(G.n_ws, device=y.device)[None] >= k[:, None]) & mix[:, None]
    return torch.where(fina[..., None], w2[:, None], ws), torch.where(mix, -1, y)


def ponte(G, wa, wb, k):
    """k primeiras posições (as grossas) de A, o resto de B."""
    return G.sintese(torch.cat([wa[:, :k], wb[:, k:]], 1))


# ------------------------------------------------------------------ discriminador (residual, StyleGAN2)
def desvio_lote(x, g=4):
    """Desvio padrão do minilote como canal extra: dá ao D um jeito de ver falta de variedade."""
    B, C, H, W = x.shape
    g = min(g, B)
    s = x.reshape(g, -1, C, H, W)
    s = (s - s.mean(0)).square().mean(0).add(1e-8).sqrt().mean([1, 2, 3])
    return torch.cat([x, s.reshape(-1, 1, 1, 1).repeat(g, 1, H, W)], 1)


class BlocoD(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1, self.c2, self.atalho = Conv(i, i, 3), Conv(i, o, 3), Conv(i, o, 1, bias=False)

    def forward(self, x):
        a = self.atalho(F.avg_pool2d(x, 2))
        x = F.avg_pool2d(lrelu(self.c2(lrelu(self.c1(x)))), 2)
        return (x + a) / SQ2


class Discriminador(nn.Module):
    """Dois logits (Shahbazi et al. 2022): u = "parece Pokémon?", para toda imagem; c = u + projeção da espécie
    (Miyato e Koyama 2018), só para as que têm espécie. Misturadas (rótulo -1) só entram em u."""
    def __init__(self):
        super().__init__()
        self.de_rgb = Conv(3, CH[64], 1)
        self.blocos = nn.Sequential(*[BlocoD(CH[r], CH[r // 2]) for r in (64, 32, 16, 8)])
        self.conv = Conv(CH[4] + 1, CH[4], 3)
        self.fc = Linear(CH[4] * 16, CH[4])
        self.saida = Linear(CH[4], 1)
        self.proj = nn.Embedding(N_ESP, CH[4])
        self.cls = Linear(CH[4], N_ESP)                                    # que espécie é esta? (AC-GAN)

    def forward(self, x, y):
        h = self.blocos(lrelu(self.de_rgb(x)))
        h = lrelu(self.fc(lrelu(self.conv(desvio_lote(h))).flatten(1)))
        u = self.saida(h).squeeze(1)
        return u, u + (self.proj(y.clamp(min=0)) * h).sum(1) / math.sqrt(h.shape[1]), self.cls(h)


def media_em(v, m):
    return v[m].mean() if m.any() else v.sum() * 0


# ------------------------------------------------------------------ aumento diferenciável (DiffAugment)
def aumenta(x, politica):
    B, _, H, W = x.shape
    dev = x.device
    if "cor" in politica:
        x = x + (torch.rand(B, 1, 1, 1, device=dev) - 0.5)
        m = x.mean(1, keepdim=True)
        x = (x - m) * (torch.rand(B, 1, 1, 1, device=dev) * 2) + m
        m = x.mean([1, 2, 3], keepdim=True)
        x = (x - m) * (torch.rand(B, 1, 1, 1, device=dev) + 0.5) + m
    if "translacao" in politica:                                          # até 1/8 do lado, borda branca
        s = H // 8
        ti = torch.randint(-s, s + 1, (B, 1, 1), device=dev)
        tj = torch.randint(-s, s + 1, (B, 1, 1), device=dev)
        gb, gi, gj = torch.meshgrid(torch.arange(B, device=dev), torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        gi, gj = (gi + ti + 1).clamp(0, H + 1), (gj + tj + 1).clamp(0, W + 1)
        x = F.pad(x, [1, 1, 1, 1], value=1.0).permute(0, 2, 3, 1)[gb, gi, gj].permute(0, 3, 1, 2)
    if "recorte" in politica:                                             # um quadrado de metade do lado
        c = H // 2
        oi = torch.randint(0, H + (1 - c % 2), (B, 1, 1), device=dev)
        oj = torch.randint(0, W + (1 - c % 2), (B, 1, 1), device=dev)
        gb, gi, gj = torch.meshgrid(torch.arange(B, device=dev), torch.arange(c, device=dev), torch.arange(c, device=dev), indexing="ij")
        m = torch.ones(B, H, W, device=dev, dtype=x.dtype)
        m[gb, (gi + oi - c // 2).clamp(0, H - 1), (gj + oj - c // 2).clamp(0, W - 1)] = 0
        x = x * m[:, None]
    return x


# ------------------------------------------------------------------ medidas da ponte
def mascara(x):
    """Figura = tudo menos o branco ligado à borda (o corpo branco do Lugia é figura, não fundo)."""
    branco = (x > 0.8).all(1, keepdim=True).float()
    fundo = torch.zeros_like(branco)
    fundo[..., 0, :] = fundo[..., -1, :] = fundo[..., :, 0] = fundo[..., :, -1] = 1
    fundo = fundo * branco
    for _ in range(64):                                                    # inundação a partir da borda
        fundo = F.max_pool2d(fundo, 3, 1, 1) * branco
    return fundo[:, 0] < 0.5


def histograma(x, m, n=4):
    q = ((x.clamp(-1, 1) + 1) / 2 * (n - 1e-3)).long()
    idx = (q[:, 0] * n + q[:, 1]) * n + q[:, 2]
    h = torch.zeros(len(x), n ** 3, device=x.device).scatter_add_(1, idx.flatten(1), m.flatten(1).float())
    return h / h.sum(1, keepdim=True).clamp_min(1)


@torch.no_grad()
def medidas_ponte(G, wa, wb):
    """Para cada corte k (k posições grossas de A, o resto de B):
       forma = IoU da silhueta com a de A puro; cor = d(A) / (d(A) + d(B)), distâncias L1 entre histogramas de cor
       da figura (0 = cores de A, 1 = cores de B). Controle: média de pixels de A e B."""
    A, B = G.sintese(wa), G.sintese(wb)
    ma, ha, hb = mascara(A), histograma(A, mascara(A)), histograma(B, mascara(B))

    def mede(X):
        m = mascara(X)
        iou = ((m & ma).sum([1, 2]) / (m | ma).sum([1, 2]).clamp_min(1)).float()
        h = histograma(X, m)
        da, db = (h - ha).abs().sum(1), (h - hb).abs().sum(1)
        return float(iou.mean()), float((da / (da + db + 1e-8)).mean())

    r = {str(k): mede(ponte(G, wa, wb, k)) for k in range(G.n_ws + 1)}
    r["pixels"] = mede((A + B) / 2)
    return r


# ------------------------------------------------------------------ treino
def treina(modo, horas, dev=None):
    dev = dev or dispositivo()
    cfg = CONFIG
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    if dev == "cuda":   # acima disso o driver do Windows joga memória na RAM e o treino fica 7x mais lento (17/11)
        torch.cuda.set_per_process_memory_fraction(0.7)
    X, y = carrega(dev)
    m = y < cfg["especies"]
    X, y = X[m], y[m]
    if dev == "cuda":
        torch.cuda.empty_cache()                                               # devolve o dataset inteiro, já filtrado
    G = Gerador(SAIDA[modo], cfg["mapa"]).to(dev)
    D = Discriminador().to(dev)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    c = cfg["r1_cada"] / (cfg["r1_cada"] + 1)                              # regularização preguiçosa (StyleGAN2)
    oG = torch.optim.Adam(G.parameters(), cfg["lr"], betas=cfg["betas"], eps=1e-8)
    oD = torch.optim.Adam(D.parameters(), cfg["lr"] * c, betas=(cfg["betas"][0] ** c, cfg["betas"][1] ** c), eps=1e-8)
    est = dict(it=0, minutos=0.0, historico=[], avaliacoes=[])
    ck = RES / f"{modo}_ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        for m, nome in ((G, "G"), (G_ema, "G_ema"), (D, "D"), (oG, "oG"), (oD, "oD")):
            m.load_state_dict(s[nome])
        est = s["est"]
        print(f"{modo}: retomando do passo {est['it']} ({est['minutos']:.0f} min)", flush=True)
    J = Juiz().to(dev).eval()
    J.load_state_dict(torch.load(RES / "juiz.pt", map_location=dev))
    gfix = torch.Generator(device=dev).manual_seed(7)
    with torch.no_grad():
        ref, reais = referencia(J, X, dev)
        base = avalia(J, reais, ref)
    z_fixo = torch.randn(2000, Z_DIM, device=dev, generator=gfix)
    y_fixo = y[torch.randint(0, len(y), (2000,), device=dev, generator=gfix)]       # espécies na proporção do dataset
    pa = torch.randint(0, cfg["especies"], (2, 128), device=dev, generator=gfix)               # 128 pares fixos para medir a ponte
    za, zb = torch.randn(2, 128, Z_DIM, device=dev, generator=gfix)
    lote, ema = cfg["lote"], 0.5 ** (cfg["lote"] / (cfg["ema_kimg"] * 1000))
    print(f"{modo}: {sum(p.numel() for p in G.parameters()) / 1e6:.2f} M em G, {sum(p.numel() for p in D.parameters()) / 1e6:.2f} M em D; "
          f"reais: FID {base['fid']:.1f}", flush=True)
    t0, m0 = time.time(), est["minutos"]
    prox_aval = time.time() + 60 * cfg["aval_min"]
    r1 = torch.zeros(())

    def salva():
        est["minutos"] = m0 + (time.time() - t0) / 60
        tmp = ck.with_suffix(".tmp")
        torch.save(dict(G=G.state_dict(), G_ema=G_ema.state_dict(), D=D.state_dict(), oG=oG.state_dict(), oD=oD.state_dict(), est=est), tmp)
        tmp.replace(ck)
        torch.save(G_ema.state_dict(), RES / f"{modo}_G.pt")
        snaps = RES / f"{modo}_snaps"                                      # um G por avaliação, para escolher depois
        snaps.mkdir(exist_ok=True)
        torch.save(G_ema.state_dict(), snaps / f"G_{est['it'] * lote // 1000:05d}kimg.pt")
        (RES / f"{modo}.json").write_text(json.dumps(dict(modo=modo, config=cfg, **est), indent=1), encoding="utf-8")

    while time.time() < t0 + 3600 * horas:
        t_ini, t_fim = cfg["transicao_kimg"]
        lam = min(max((est["it"] * lote / 1000 - t_ini) / (t_fim - t_ini), 0.0), 1.0)
        G.mapa.lam = G_ema.mapa.lam = lam
        idx = torch.randint(0, len(X), (lote,), device=dev)
        real, yr = norm(X[idx]), y[idx]
        real = torch.where(torch.rand(lote, 1, 1, 1, device=dev) < 0.5, real.flip(3), real)
        # --- discriminador: perda incondicional em todas as imagens + lam vezes a condicional nas que têm espécie
        with torch.no_grad():
            ws, yd = mistura(G, torch.randn(lote, Z_DIM, device=dev), yr, cfg["mistura"])
            falso = G.sintese(ws)
        (u_f, c_f, _), (u_r, c_r, k_r) = D(aumenta(falso, cfg["aumento"]), yd), D(aumenta(real, cfg["aumento"]), yr)
        puro = yd >= 0
        ce_d = F.cross_entropy(k_r, yr)                                    # classificador só nos reais, desde o início
        lD = (F.softplus(u_f).mean() + F.softplus(-u_r).mean() + lam * (media_em(F.softplus(c_f), puro) + F.softplus(-c_r).mean())
              + cfg["classe"] * ce_d)
        oD.zero_grad(set_to_none=True)
        lD.backward()
        oD.step()
        if est["it"] % cfg["r1_cada"] == 0:                                # r1 guarda o último valor calculado
            rr = real.detach().requires_grad_(True)
            u, c, _ = D(aumenta(rr, cfg["aumento"]), yr)
            g, = torch.autograd.grad((u + lam * c).sum(), rr, create_graph=True)
            r1 = g.square().sum([1, 2, 3]).mean()
            oD.zero_grad(set_to_none=True)
            (cfg["gamma"] / 2 * r1 * cfg["r1_cada"]).backward()
            oD.step()
        # --- gerador
        D.requires_grad_(False)
        yg = y[torch.randint(0, len(y), (lote,), device=dev)]
        ws, ydg = mistura(G, torch.randn(lote, Z_DIM, device=dev), yg, cfg["mistura"])
        u_g, c_g, k_g = D(aumenta(G.sintese(ws), cfg["aumento"]), ydg)
        ce_g = media_em(F.cross_entropy(k_g, ydg.clamp(min=0), reduction="none"), ydg >= 0)     # só as puras têm espécie
        lG = F.softplus(-u_g).mean() + lam * (media_em(F.softplus(-c_g), ydg >= 0) + cfg["classe"] * ce_g)
        oG.zero_grad(set_to_none=True)
        lG.backward()
        oG.step()
        D.requires_grad_(True)
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1 - ema)
        est["it"] += 1
        if est["it"] % 100 == 0:
            mix = yd < 0
            est["historico"].append(dict(it=est["it"], kimg=est["it"] * lote / 1000, minutos=m0 + (time.time() - t0) / 60,
                                         lam=lam, lD=float(lD), lG=float(lG), r1=float(r1), d_real=float(u_r.mean()),
                                         ce_d=float(ce_d), ce_g=float(ce_g), acc_d=float((k_r.argmax(1) == yr).float().mean()),
                                         d_puro=float(u_f[~mix].mean()) if (~mix).any() else None,      # logit "parece Pokémon?"
                                         d_mist=float(u_f[mix].mean()) if mix.any() else None))
        if time.time() >= prox_aval:
            prox_aval = time.time() + 60 * cfg["aval_min"]
            with torch.no_grad():
                imgs = torch.cat([G_ema(z_fixo[k:k + 250], y_fixo[k:k + 250]).clamp(-1, 1) for k in range(0, 2000, 250)])
                m = avalia(J, imgs, ref)
                m["acerto_especie"] = float((torch.cat([J(imgs[k:k + 500])[0].argmax(1) for k in range(0, 2000, 500)]) == y_fixo).float().mean())
                m["ponte"] = medidas_ponte(G_ema, G_ema.ws(za, pa[0]), G_ema.ws(zb, pa[1]))
            est["avaliacoes"].append(dict(it=est["it"], kimg=est["it"] * lote / 1000, minutos=m0 + (time.time() - t0) / 60, **m))
            k_meio = str(G.n_ws - 2)
            print(f"  it {est['it']:6d} [{m0 + (time.time() - t0) / 60:5.1f} min, {est['it'] * lote / 1000:6.0f} kimg] FID {m['fid']:6.1f} "
                  f"espécies {m['especies']:3d} (efetivas {m['especies_efetivas']:5.1f}) | ponte k={k_meio}: forma {m['ponte'][k_meio][0]:.2f} "
                  f"cor de B {m['ponte'][k_meio][1]:.2f} | juiz acerta a espécie {m['acerto_especie']:.1%} | lD {float(lD):.3f} "
                  f"lG {float(lG):.3f} ce_g {float(ce_g):.2f} | GPU {torch.cuda.max_memory_reserved() / 2 ** 30 if dev == 'cuda' else 0:.1f} GB",
                  flush=True)
            salva()
    salva()
    print(f"{modo}: pausa em {est['it']} passos, {est['minutos']:.0f} min no total", flush=True)


# ------------------------------------------------------------------ autoteste
def teste():
    torch.manual_seed(0)
    for saida in ("paleta", "rgb", "cor"):
        G, D = Gerador(saida, mapa=2), Discriminador()
        z, y = torch.randn(8, Z_DIM), torch.randint(0, N_ESP, (8,))
        x = G(z, y)
        assert x.shape == (8, 3, 64, 64) and torch.isfinite(x).all()
        ws, yd = mistura(G, z, y, 1.0)
        assert (yd == -1).all() and torch.equal(ws[:, 0], G.ws(z, y)[:, 0])     # posição mais grossa: sempre de A
        ws0, yd0 = mistura(G, z, y, 0.0)
        assert torch.equal(ws0, G.ws(z, y)) and torch.equal(yd0, y)
        u, c, k = D(aumenta(x, "translacao,recorte,cor"), yd)
        assert u.shape == c.shape == (8,) and k.shape == (8, N_ESP)
        (F.softplus(-u).mean() + media_em(F.softplus(-c), yd >= 0)).backward()
        assert all(torch.isfinite(p.grad).all() for p in G.parameters() if p.grad is not None)
        xd = x.detach()                                                       # u não depende da espécie; c depende
        h = lrelu(D.fc(lrelu(D.conv(desvio_lote(D.blocos(lrelu(D.de_rgb(xd)))))).flatten(1)))
        u1, c1, k1 = D(xd, y)
        u2, c2, k2 = D(xd, torch.full((8,), -1))
        assert torch.allclose(k1, k2)                                         # a classificação não vê o rótulo
        assert torch.allclose(u1, D.saida(h).squeeze(1)) and torch.allclose(u1, u2) and not torch.allclose(c1, u1)
        assert float(media_em(c1, torch.zeros(8, dtype=torch.bool))) == 0.0   # sem amostras com espécie: termo zero
        wa, wb = G.ws(z, y), G.ws(torch.randn(8, Z_DIM), torch.randint(0, N_ESP, (8,)))
        torch.manual_seed(1)
        a = ponte(G, wa, wb, G.n_ws)
        torch.manual_seed(1)
        assert torch.allclose(a, G.sintese(wa))                               # k = tudo de A: é A
        r = medidas_ponte(G, wa, wb)
        assert set(r) == {str(k) for k in range(G.n_ws + 1)} | {"pixels"}
    G = Gerador("paleta", mapa=2)
    img, a = G.sintese(G.ws(torch.randn(4, Z_DIM), torch.randint(0, N_ESP, (4,))), devolve_regioes=True)
    assert torch.allclose(a.sum(1), torch.ones(4, 64, 64)) and img.abs().max() <= 1 + 1e-5   # regiões somam 1; cores em [-1, 1]
    # paleta: slots 1 a K em ordem de luminância
    pal = G.paleta(G.ws(torch.randn(4, Z_DIM), torch.randint(0, N_ESP, (4,)))[:, -1])
    lum = pal[:, 1:] @ torch.tensor([0.299, 0.587, 0.114])
    assert (lum[:, 1:] >= lum[:, :-1]).all() and torch.equal(pal[:, 0], torch.ones(4, 3))
    # transição: com lam = 0 a espécie não muda o mapeamento
    G.mapa.lam = 0.0
    z = torch.randn(4, Z_DIM)
    assert torch.allclose(G.mapa(z, torch.tensor([1, 2, 3, 4])), G.mapa(z, torch.tensor([9, 8, 7, 6])))
    # máscara: branco ligado à borda é fundo; branco cercado pela figura é figura
    x = torch.ones(1, 3, 64, 64)
    x[..., 10:50, 10:50] = -1
    x[..., 20:40, 20:40] = 1
    m = mascara(x)
    assert not m[0, 0, 0] and m[0, 30, 30] and m[0, 15, 15]
    # aumento: translação com borda branca não suja uma imagem toda branca
    assert torch.allclose(aumenta(torch.ones(2, 3, 64, 64), "translacao"), torch.ones(2, 3, 64, 64))
    # medidas: silhueta idêntica dá IoU 1; mesma cor dá distância 0
    x = -torch.ones(1, 3, 64, 64)
    assert mascara(x).all() and float(histograma(x, mascara(x)).sum()) == 1.0
    # desvio do minilote: lote repetido tem desvio ~0
    h = desvio_lote(torch.randn(1, 5, 4, 4).repeat(8, 1, 1, 1))
    assert h[:, -1].abs().max() < 1e-3
    print("teste ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["teste", "treina", "figuras"])
    ap.add_argument("--modo", choices=list(SAIDA), default="cor")
    ap.add_argument("--horas", type=float, default=8)
    a = ap.parse_args()
    if a.acao == "teste":
        teste()
    elif a.acao == "treina":
        treina(a.modo, a.horas)
    else:
        from figuras_estilo import figuras
        figuras(a.modo)
    sys.exit(0)
