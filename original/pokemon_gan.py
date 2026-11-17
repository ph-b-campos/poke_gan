"""DCGAN no Pokémon do D2L (Zhang et al.): treino ruim contra treino bom, e a ponte entre espécies.

Mesma arquitetura DCGAN em todos os treinos; só muda a técnica:
    minimax  GAN original, perda saturante log(1 - D(G(z)))        -> esperado: vanishing gradient
    ns       GAN não saturante, -log D(G(z)) (o padrão da prática)  -> ver se colapsa
    colapso  a mesma, com D encolhido (Metz et al. 2017)            -> esperado: colapso de modos
    wclip    WGAN com clipping de pesos (Arjovsky et al. 2017)
    wgp      WGAN-GP, penalidade de gradiente (Gulrajani et al. 2017)
    cwgp     WGAN-GP condicional na espécie, BatchNorm condicional por camada: a "ponte"

    python pokemon_gan.py teste                              # autoteste em CPU, segundos
    python pokemon_gan.py prepara --raiz .../pokemon         # 40.597 sprites -> cache 64x64
    python pokemon_gan.py classificador                      # juiz de espécies (o "Inception" do FID)
    python pokemon_gan.py treina --modo wgp --passos 4000

Dados: http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip (sha1 c065c0e2593b8b161a2d7873e42418bf6a21106c).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

AQUI = Path(__file__).resolve().parent
RES = AQUI / "resultados"
CACHE_WIN = Path(r"D:\Arquivos\Documents\Mestrado_PEB\dados_externos\pokemon_d2l\pokemon64.npz")
CACHE = Path(os.environ.get("POKEMON_CACHE") or (AQUI / "dados" / "pokemon64.npz" if (AQUI / "dados").is_dir() else CACHE_WIN))
NZ, N_ESP, EMB = 100, 721, 128
CAMADAS_G = 4                         # camadas com normalização no gerador: 4x4, 8x8, 16x16, 32x32
CONFIG = {                            # fixada antes de rodar (Radford 2015; Arjovsky 2017; Gulrajani 2017)
    "minimax": dict(perda="minimax", opt="adam", lr=2e-4, betas=(0.5, 0.999), n_critico=1),
    "ns": dict(perda="ns", opt="adam", lr=2e-4, betas=(0.5, 0.999), n_critico=1),
    "colapso": dict(perda="ns", opt="adam", lr=2e-4, betas=(0.5, 0.999), n_critico=1, nd=32),   # D com ~1/5 dos parâmetros de G (Metz et al. 2017)
    "wclip": dict(perda="wgan", opt="rmsprop", lr=5e-5, clip=0.01, n_critico=5),
    "wgp": dict(perda="wgan", opt="adam", lr=1e-4, betas=(0.0, 0.9), n_critico=5, gp=10.0),
    "cwgp": dict(perda="wgan", opt="adam", lr=1e-4, betas=(0.0, 0.9), n_critico=5, gp=10.0, condicional=True, especies=251),   # gerações 1 e 2: Pikachu #25 e Lugia #249
    # o mesmo + mixing regularization (Karras et al. 2019): metade do lote já sai misturada no treino (0,5 e não 0,9: G é condicional)
    "cwgp_mix": dict(perda="wgan", opt="adam", lr=1e-4, betas=(0.0, 0.9), n_critico=5, gp=10.0, condicional=True, especies=251, mistura=0.5),
}



def dispositivo():
    """GPU NVIDIA (cuda), GPU do Mac (mps) ou CPU; a variável DEVICE força um deles."""
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------ dados
def prepara(raiz, saida=CACHE):
    """Sprites com transparência sobre fundo branco, centralizados num quadrado, 64x64."""
    from PIL import Image
    arqs = sorted(Path(raiz).glob("*/*.png"), key=lambda f: (int(f.parent.name), f.name))
    X = np.empty((len(arqs), 64, 64, 3), np.uint8)
    y = np.empty(len(arqs), np.int16)
    for i, f in enumerate(arqs):
        im = Image.open(f).convert("RGBA")
        fundo = Image.new("RGBA", im.size, (255, 255, 255, 255))
        fundo.alpha_composite(im)
        w, h = im.size
        q = Image.new("RGB", (max(w, h),) * 2, "white")
        q.paste(fundo.convert("RGB"), ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
        X[i] = np.asarray(q.resize((64, 64), Image.LANCZOS))
        y[i] = int(f.parent.name) - 1                     # número da Pokédex - 1
    np.savez(saida, X=X, y=y)
    print(f"{len(X)} sprites, {len(np.unique(y))} espécies -> {saida}")


def carrega(dev):
    d = np.load(CACHE)
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).contiguous().to(dev)   # uint8 na GPU: ~500 MB
    return X, torch.from_numpy(d["y"].astype(np.int64)).to(dev)


def norm(xb):
    return xb.float() / 127.5 - 1


# ------------------------------------------------------------------ modelos
class BNCond(nn.Module):
    """BatchNorm sem afim próprio; ganho e viés vêm do embedding da espécie (condicional por camada)."""
    def __init__(self, c, cond):
        super().__init__()
        self.bn = nn.BatchNorm2d(c, affine=not cond)
        self.af = nn.Linear(EMB, 2 * c) if cond else None
        if cond:
            nn.init.zeros_(self.af.weight)
            nn.init.zeros_(self.af.bias)

    def forward(self, h, e):
        h = self.bn(h)
        if self.af is None:
            return h
        g, b = self.af(e).chunk(2, 1)
        return h * (1 + g[..., None, None]) + b[..., None, None]


class Gerador(nn.Module):
    """DCGAN: z (100) -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x3."""
    def __init__(self, cond=False, ng=64):
        super().__init__()
        cs = [ng * 8, ng * 4, ng * 2, ng]
        self.cond = cond
        self.emb = nn.Embedding(N_ESP, EMB) if cond else None
        self.convs = nn.ModuleList([nn.ConvTranspose2d(NZ, cs[0], 4, 1, 0, bias=False)] +
                                   [nn.ConvTranspose2d(cs[i], cs[i + 1], 4, 2, 1, bias=False) for i in range(3)])
        self.norms = nn.ModuleList([BNCond(c, cond) for c in cs])
        self.saida = nn.ConvTranspose2d(cs[-1], 3, 4, 2, 1)

    def embeddings(self, y):
        """Um embedding por camada; misturar camadas = ponte entre espécies."""
        return self.emb(y)[:, None].expand(-1, CAMADAS_G, -1)

    def forward(self, z, y=None, e=None):
        if self.cond and e is None:
            e = self.embeddings(y)
        h = z[..., None, None]
        for k, (cv, nm) in enumerate(zip(self.convs, self.norms)):
            h = F.relu(nm(cv(h), e[:, k] if self.cond else None))
        return torch.tanh(self.saida(h))


class Discriminador(nn.Module):
    """Mesma rede para D e para o crítico: sem BatchNorm (a penalidade de gradiente é por amostra),
    GroupNorm de um grupo nas camadas internas; projeção da espécie no modo condicional (Miyato e Koyama 2018)."""
    def __init__(self, cond=False, nd=64):
        super().__init__()
        cs = [nd, nd * 2, nd * 4, nd * 8]
        camadas, ant = [], 3
        for i, c in enumerate(cs):
            camadas += [nn.Conv2d(ant, c, 4, 2, 1)] + ([nn.GroupNorm(1, c)] if i else []) + [nn.LeakyReLU(0.2)]
            ant = c
        self.corpo = nn.Sequential(*camadas)                        # 64 -> 4x4x512
        self.lin = nn.Linear(cs[-1] * 16, 1)
        self.proj = nn.Embedding(N_ESP, cs[-1] * 16) if cond else None

    def forward(self, x, y=None):
        h = self.corpo(x).flatten(1)
        out = self.lin(h).squeeze(1)
        if self.proj is None:
            return out
        return out + (self.proj(y.clamp(min=0)) * h).sum(1) * (y >= 0)     # y = -1: amostra misturada, só "parece Pokémon?"


def mistura(G, y, p, n_esp):
    """Mixing regularization: numa fração p do lote, as camadas finas vêm de outra espécie, com corte aleatório.
    Devolve os embeddings por camada e o rótulo para o crítico (-1 nas misturadas: elas não têm espécie)."""
    y2 = torch.randint(0, n_esp, y.shape, device=y.device)
    mix = torch.rand(y.shape, device=y.device) < p
    k = torch.where(mix, torch.randint(1, CAMADAS_G, y.shape, device=y.device), CAMADAS_G)   # 1 a 3 camadas grossas de y
    de_y = (torch.arange(CAMADAS_G, device=y.device)[None] < k[:, None])[..., None]
    return torch.where(de_y, G.embeddings(y), G.embeddings(y2)), torch.where(mix, -1, y)


# ------------------------------------------------------------------ perdas
def perda_d(cfg, l_real, l_falso):
    if cfg["perda"] == "wgan":
        return l_falso.mean() - l_real.mean()
    return F.softplus(-l_real).mean() + F.softplus(l_falso).mean()          # BCE: -log D(x) - log(1 - D(G(z)))


def perda_g(cfg, l_falso):
    if cfg["perda"] == "minimax":
        return -F.softplus(l_falso).mean()                                  # = log(1 - D(G(z))), minimizada
    if cfg["perda"] == "ns":
        return F.softplus(-l_falso).mean()                                  # = -log D(G(z))
    return -l_falso.mean()


def penalidade(D, real, falso, y=None):
    a = torch.rand(len(real), 1, 1, 1, device=real.device)
    xh = (a * real + (1 - a) * falso).requires_grad_(True)
    g, = torch.autograd.grad(D(xh, y).sum(), xh, create_graph=True)
    return ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()


def norma_grad(params):
    return float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)))


# ------------------------------------------------------------------ juiz de espécies (para FID, IS e contagem de espécies)
class Juiz(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 32, 64, 128, 256]
        self.f = nn.Sequential(*[m for i in range(4) for m in
                                 (nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.BatchNorm2d(c[i + 1]), nn.ReLU(), nn.MaxPool2d(2))],
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cls = nn.Linear(256, N_ESP)

    def forward(self, x):
        h = self.f(x)
        return self.cls(h), h


def treina_juiz(epocas=6, dev=None):
    dev = dev or dispositivo()
    torch.manual_seed(0)
    X, y = carrega(dev)
    idx = torch.randperm(len(X), device=dev)
    te, tr = idx[:4000], idx[4000:]
    J = Juiz().to(dev)
    opt = torch.optim.Adam(J.parameters(), 1e-3)
    for ep in range(epocas):
        J.train()
        for i in range(0, len(tr), 256):
            b = tr[i:i + 256]
            xb = norm(X[b])
            xb = torch.where(torch.rand(len(b), 1, 1, 1, device=dev) < 0.5, xb.flip(3), xb)
            loss = F.cross_entropy(J(xb)[0], y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = tr[torch.randperm(len(tr), device=dev)]
        J.eval()
        with torch.no_grad():
            acc = float((torch.cat([J(norm(X[te[i:i + 500]]))[0].argmax(1) for i in range(0, len(te), 500)]) == y[te]).float().mean())
        print(f"juiz época {ep + 1}: acurácia {acc:.3f} em 4.000 sprites separados", flush=True)
    RES.mkdir(exist_ok=True)
    torch.save(J.state_dict(), RES / "juiz.pt")
    (RES / "juiz.json").write_text(json.dumps({"acuracia_teste": acc, "epocas": epocas}), encoding="utf-8")


@torch.no_grad()
def avalia(J, imgs, ref):
    """Métricas no espaço do juiz: FID, IS, espécies distintas e diversidade em pixels."""
    logit, h = zip(*(J(imgs[i:i + 500]) for i in range(0, len(imgs), 500)))
    p, h = torch.cat(logit).cpu().softmax(1), torch.cat(h).cpu().double()   # float64 e eigvals na CPU (o MPS não tem)
    mu, C = h.mean(0), torch.cov(h.T)
    v = torch.linalg.eigvals(C @ ref["C"]).real.clamp(min=0).sqrt().sum()
    fid = float(((mu - ref["mu"]) ** 2).sum() + torch.trace(C) + torch.trace(ref["C"]) - 2 * v)
    py = p.mean(0)
    is_ = float(torch.exp((p * (p.clamp_min(1e-12).log() - py.clamp_min(1e-12).log())).sum(1).mean()))
    contagem = torch.bincount(p.argmax(1), minlength=N_ESP).double()
    q = contagem / contagem.sum()
    efetivas = float(torch.exp(-(q[q > 0] * q[q > 0].log()).sum()))
    a, b = imgs[: len(imgs) // 2].flatten(1), imgs[len(imgs) // 2:].flatten(1)
    div = float((a - b).norm(dim=1).mean())
    return dict(fid=fid, is_=is_, especies=int((contagem > 0).sum()), especies_efetivas=efetivas, diversidade=div)


def referencia(J, X, dev, n=2000):
    g = torch.Generator(device="cpu").manual_seed(1)
    imgs = norm(X[torch.randperm(len(X), generator=g)[:n].to(dev)])
    logit, h = zip(*(J(imgs[i:i + 500]) for i in range(0, n, 500)))
    h = torch.cat(h).cpu().double()
    ref = {"mu": h.mean(0), "C": torch.cov(h.T)}
    return ref, imgs


# ------------------------------------------------------------------ treino
def treina(modo, passos, minutos=240, dev=None, lote=128, cada_aval=250):
    dev = dev or dispositivo()
    """Orçamento em passos do gerador, o mesmo para todos os modos (critério de Gulrajani et al. 2017)."""
    cfg = CONFIG[modo]
    cond = cfg.get("condicional", False)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    X, y = carrega(dev)
    if cfg.get("especies"):                                       # só as primeiras espécies da Pokédex
        m = y < cfg["especies"]
        X, y = X[m], y[m]
    G, D = Gerador(cond).to(dev), Discriminador(cond, nd=cfg.get("nd", 64)).to(dev)
    if cfg["opt"] == "adam":
        oG = torch.optim.Adam(G.parameters(), cfg["lr"], betas=cfg["betas"])
        oD = torch.optim.Adam(D.parameters(), cfg["lr"], betas=cfg["betas"])
    else:
        oG = torch.optim.RMSprop(G.parameters(), cfg["lr"])
        oD = torch.optim.RMSprop(D.parameters(), cfg["lr"])
    J = Juiz().to(dev).eval()
    J.load_state_dict(torch.load(RES / "juiz.pt", map_location=dev))
    with torch.no_grad():
        ref, reais = referencia(J, X, dev)
        base = avalia(J, reais, ref)
    z_fixo = torch.randn(2000, NZ, device=dev, generator=torch.Generator(device=dev).manual_seed(7))
    y_fixo = torch.randint(0, cfg.get("especies", N_ESP), (2000,), device=dev, generator=torch.Generator(device=dev).manual_seed(8)) if cond else None
    hist, aval, grades = [], [{"it_g": 0, "minutos": 0.0, **base, "real": True}], {}
    t0, it_g, it_d = time.time(), 0, 0
    print(f"{modo}: {sum(p.numel() for p in G.parameters()) / 1e6:.2f} M em G, {sum(p.numel() for p in D.parameters()) / 1e6:.2f} M em D; "
          f"reais: FID {base['fid']:.1f}, espécies {base['especies']}", flush=True)
    fim = t0 + 60 * minutos

    def salva():
        """Grava tudo até aqui: chamada a cada avaliação para um treino interrompido não se perder."""
        RES.mkdir(exist_ok=True)
        (RES / f"{modo}.json").write_text(json.dumps(dict(modo=modo, config=cfg, lote=lote, it_g=it_g, it_d=it_d, passos=passos,
                                                          minutos=(time.time() - t0) / 60, historico=hist, avaliacoes=aval), indent=1), encoding="utf-8")
        if grades:
            np.savez_compressed(RES / f"{modo}_grades.npz", its=np.array(sorted(grades)), grades=np.stack([grades[k] for k in sorted(grades)]))
        torch.save(G.state_dict(), RES / f"{modo}_G.pt")
    while it_g < passos and time.time() < fim:
        ordem = torch.randperm(len(X), device=dev)
        for i in range(0, len(X) - lote + 1, lote):
            b = ordem[i:i + lote]
            real, yb = norm(X[b]), (y[b] if cond else None)
            z = torch.randn(lote, NZ, device=dev)
            with torch.no_grad():
                e, yd = mistura(G, yb, cfg["mistura"], cfg["especies"]) if "mistura" in cfg else (None, yb)
                falso = G(z, yb, e=e)
            l_r, l_f = D(real, yb), D(falso, yd)
            lD = perda_d(cfg, l_r, l_f)
            gp = penalidade(D, real, falso, yd) if "gp" in cfg else torch.zeros((), device=dev)
            oD.zero_grad(set_to_none=True)
            (lD + cfg.get("gp", 0) * gp).backward()
            oD.step()
            if "clip" in cfg:
                for p in D.parameters():
                    p.data.clamp_(-cfg["clip"], cfg["clip"])
            it_d += 1
            if it_d % cfg["n_critico"]:
                continue
            z = torch.randn(lote, NZ, device=dev)
            e, yd = mistura(G, yb, cfg["mistura"], cfg["especies"]) if "mistura" in cfg else (None, yb)
            falso = G(z, yb, e=e)
            falso.retain_grad()
            lG = perda_g(cfg, D(falso, yd))
            oG.zero_grad(set_to_none=True)
            lG.backward()
            gnorm = norma_grad(G.parameters())
            gx = float(falso.grad.flatten(1).norm(dim=1).mean()) * lote     # sinal que cada imagem recebe
            oG.step()
            it_g += 1
            if it_g % 10 == 0:
                with torch.no_grad():
                    rec = dict(it_g=it_g, minutos=(time.time() - t0) / 60, lD=float(lD), lG=float(lG), gp=float(gp),
                               grad_G=gnorm, grad_x=gx, w=float(l_r.mean() - l_f.mean()),
                               d_real=float(torch.sigmoid(l_r).mean()), d_falso=float(torch.sigmoid(l_f).mean()))
                hist.append(rec)
            if it_g % cada_aval == 0:
                G.eval()
                with torch.no_grad():
                    imgs = torch.cat([G(z_fixo[k:k + 500], None if y_fixo is None else y_fixo[k:k + 500]) for k in range(0, 2000, 500)])
                    m = avalia(J, imgs, ref)
                    grades[it_g] = ((imgs[:64].permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                G.train()
                aval.append(dict(it_g=it_g, minutos=(time.time() - t0) / 60, **m))
                print(f"  it_G {it_g:5d} [{(time.time() - t0) / 60:4.1f} min] FID {m['fid']:7.1f} IS {m['is_']:6.1f} "
                      f"espécies {m['especies']:3d} (efetivas {m['especies_efetivas']:5.1f}) grad_G {gnorm:.2e} "
                      f"lD {float(lD):+.3f} lG {float(lG):+.3f}", flush=True)
                salva()
            if it_g >= passos or time.time() >= fim:
                break
    salva()
    print(f"{modo} pronto: {it_g} passos de G, {it_d} de D, {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------ autoteste
def teste():
    torch.manual_seed(0)
    for cond in (False, True):
        G, D = Gerador(cond, ng=8), Discriminador(cond, nd=8)
        y = torch.randint(0, N_ESP, (4,)) if cond else None
        x = G(torch.randn(4, NZ), y)
        assert x.shape == (4, 3, 64, 64) and x.abs().max() <= 1
        assert D(x, y).shape == (4,)
        for modo in ("minimax", "ns", "wgp"):
            cfg = CONFIG[modo]
            G.zero_grad()
            perda_g(cfg, D(G(torch.randn(4, NZ), y), y)).backward()
            assert math.isfinite(norma_grad(G.parameters()))
        assert penalidade(D, x.detach(), x.detach().flip(0), y) >= 0
    # Ponte: camadas iniciais de uma espécie e finais de outra mudam a imagem; só uma espécie = a própria espécie
    G = Gerador(True, ng=8).eval()
    for nm in G.norms:
        nn.init.normal_(nm.af.weight, std=0.5)
    z, a, b = torch.randn(1, NZ), torch.tensor([24]), torch.tensor([248])
    e = torch.cat([G.embeddings(a)[:, :2], G.embeddings(b)[:, 2:]], 1)
    assert not torch.allclose(G(z, e=e), G(z, a)) and torch.allclose(G(z, e=G.embeddings(a)), G(z, a))
    # Vanishing gradient, analítico: com D(G(z)) perto de 0, a minimax quase não empurra G; a não saturante empurra
    l = torch.tensor([-8.0], requires_grad=True)
    g_mm, = torch.autograd.grad(perda_g(CONFIG["minimax"], l), l)
    g_ns, = torch.autograd.grad(perda_g(CONFIG["ns"], l), l)
    assert abs(g_mm) < 1e-3 < 0.9 < abs(g_ns), (g_mm, g_ns)
    # Métricas: imagens repetidas têm menos espécies efetivas que imagens variadas
    J = Juiz().eval()
    ref = {"mu": torch.zeros(256, dtype=torch.float64), "C": torch.eye(256, dtype=torch.float64)}
    var, rep = torch.randn(40, 3, 64, 64), torch.randn(1, 3, 64, 64).repeat(40, 1, 1, 1)
    assert avalia(J, rep, ref)["diversidade"] == 0 and avalia(J, var, ref)["especies_efetivas"] > avalia(J, rep, ref)["especies_efetivas"]
    # Mixing regularization: p = 0 não mistura nada; p = 1 mistura tudo, o crítico só vê o termo incondicional
    y = torch.randint(0, 251, (16,))
    e0, yd0 = mistura(G, y, 0.0, 251)
    e1, yd1 = mistura(G, y, 1.0, 251)
    assert torch.equal(e0, G.embeddings(y)) and torch.equal(yd0, y) and (yd1 == -1).all()
    assert torch.equal(e1[:, 0], G.embeddings(y)[:, 0])                  # a camada 4x4 é sempre da primeira espécie
    Dc = Discriminador(True, nd=8)
    xm = torch.randn(16, 3, 64, 64)
    assert torch.allclose(Dc(xm, yd1), Dc.lin(Dc.corpo(xm).flatten(1)).squeeze(1))
    print("teste ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["teste", "prepara", "classificador", "treina"])
    ap.add_argument("--raiz", default=r"D:\Arquivos\Documents\Mestrado_PEB\dados_externos\pokemon_d2l\pokemon")
    ap.add_argument("--modo", choices=list(CONFIG))
    ap.add_argument("--passos", type=int, default=4000)
    ap.add_argument("--minutos", type=float, default=240)
    a = ap.parse_args()
    if a.acao == "teste":
        teste()
    elif a.acao == "prepara":
        prepara(a.raiz)
    elif a.acao == "classificador":
        treina_juiz()
    else:
        treina(a.modo, a.passos, a.minutos)
    sys.exit(0)
