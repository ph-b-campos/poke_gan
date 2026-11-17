"""Treino do gerador StyleGAN-like (models/stylegan): mistura entre espécies, DiffAugment e o laço de treino.

    python -m training.stylegan --modo cor --horas 8       # retoma do checkpoint, se existir

Configuração em models/stylegan/config/treino.py. Requer o cache (python -m data_handler) e o juiz
(python -m training.juiz). Salva em resultados/: <modo>_ckpt.pt (retomada), <modo>_G.pt (G com EMA),
<modo>_snaps/ (um G por avaliação) e <modo>.json (config, histórico, avaliações).
"""
import argparse
import copy
import json
import time

import torch
import torch.nn.functional as F

from analise.metricas import avalia, medidas_ponte, referencia
from config import RES
from data_handler import carrega, norm
from models.juiz import Juiz
from models.stylegan import Discriminador, Gerador
from models.stylegan.config import CONFIG, HORAS, MODO, SAIDA, Z_DIM
from utils import dispositivo


# ------------------------------------------------------------------ mixing regularization
def mistura(G, z, y, p):
    """Mixing regularization entre espécies: posições grossas de (z, y), finas de (z2, y2), corte aleatório.
    Devolve os ws e o rótulo para o discriminador (-1 nas misturadas: não têm espécie)."""
    ws = G.ws(z, y)
    mix = torch.rand(len(y), device=y.device) < p
    w2 = G.mapa(torch.randn_like(z), y[torch.randperm(len(y), device=y.device)])
    k = torch.randint(1, G.n_ws, (len(y),), device=y.device)
    fina = (torch.arange(G.n_ws, device=y.device)[None] >= k[:, None]) & mix[:, None]
    return torch.where(fina[..., None], w2[:, None], ws), torch.where(mix, -1, y)


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


# ------------------------------------------------------------------ treino
def treina(modo=MODO, horas=HORAS, dev=None):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=list(SAIDA), default=MODO)
    ap.add_argument("--horas", type=float, default=HORAS)
    a = ap.parse_args()
    treina(a.modo, a.horas)
