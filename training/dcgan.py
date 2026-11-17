"""Treino da DCGAN no Pokémon do D2L: perdas de cada técnica, mixing regularization e o laço de treino.

    python -m training.dcgan --modo wgp
    python -m training.dcgan --modo minimax --passos 4000 --minutos 240

Mesma arquitetura em todos os modos (models/dcgan); só muda a técnica, configurada em
models/dcgan/config/treino.py. Requer o cache (python -m data_handler) e o juiz (python -m training.juiz).
Salva em resultados/: <modo>.json (config, histórico, avaliações), <modo>_grades.npz e <modo>_G.pt.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from analise.metricas import avalia, referencia
from config import N_ESP, RES
from data_handler import carrega, norm
from models.dcgan import Discriminador, Gerador
from models.dcgan.config import CADA_AVAL, CAMADAS_G, CONFIG, LOTE, MINUTOS, NZ, PASSOS
from models.juiz import Juiz
from utils import dispositivo


# ------------------------------------------------------------------ mixing regularization
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


# ------------------------------------------------------------------ treino
def treina(modo, passos=PASSOS, minutos=MINUTOS, dev=None, lote=LOTE, cada_aval=CADA_AVAL):
    """Orçamento em passos do gerador, o mesmo para todos os modos (critério de Gulrajani et al. 2017)."""
    dev = dev or dispositivo()
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=list(CONFIG), required=True)
    ap.add_argument("--passos", type=int, default=PASSOS)
    ap.add_argument("--minutos", type=float, default=MINUTOS)
    a = ap.parse_args()
    treina(a.modo, a.passos, a.minutos)
