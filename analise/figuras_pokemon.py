"""Figuras de avaliação da DCGAN (models/dcgan): provas com curvas e amostras, e a ponte entre espécies.

    python -m analise.figuras_pokemon entrada                  # o que os modelos veem
    python -m analise.figuras_pokemon curvas                   # vanishing gradient, FID, espécies, Wasserstein
    python -m analise.figuras_pokemon amostras                 # 32 amostras finais de cada modo
    python -m analise.figuras_pokemon ponte --a 25 --b 249     # Pikachu com camadas do Lugia (números da Pokédex)
    python -m analise.figuras_pokemon condicional --modo cwgp
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from analise.historico import le, suave
from analise.metricas import avalia, referencia
from analise.plot_config import COR, MODOS, NOME, REAL, plt
from config import CACHE, FIG, RES
from data_handler import norm
from models.dcgan import Gerador
from models.dcgan.config import CAMADAS_G, NZ
from models.juiz import Juiz
from utils import dispositivo


def piso_fid():
    """FID entre duas amostras disjuntas de sprites reais: o melhor valor possível com 2.000 imagens."""
    dev = dispositivo()
    d = np.load(CACHE)
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).contiguous().to(dev)
    J = Juiz().to(dev).eval()
    J.load_state_dict(torch.load(RES / "juiz.pt", map_location=dev))
    with torch.no_grad():
        ref, reais = referencia(J, X, dev)
        g = torch.Generator(device="cpu").manual_seed(99)
        outros = norm(X[torch.randperm(len(X), generator=g)[:2000].to(dev)])
        return avalia(J, outros, ref)


def curvas():
    FIG.mkdir(parents=True, exist_ok=True)
    R = {m: le(m) for m in MODOS if le(m)}
    piso = piso_fid()
    print("piso real:", {k: round(v, 2) for k, v in piso.items()})

    # 1. Vanishing gradient: o sinal que chega a cada imagem gerada, e o que D pensa dos falsos
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in ("minimax", "ns"):
        if m not in R:
            continue
        h = R[m]["historico"]
        it = np.array([r["it_g"] for r in h])
        ax[0].plot(it[9:], suave([r["grad_x"] for r in h]), color=COR[m], label=NOME[m])
        ax[1].plot(it[9:], suave([r["d_falso"] for r in h]), color=COR[m], label=f"D(G(z)), {NOME[m]}")
        ax[1].plot(it[9:], suave([r["d_real"] for r in h]), color=COR[m], ls="--", lw=1.5, label=f"D(x real), {NOME[m]}")
    ax[0].set_yscale("log")
    ax[0].set(title="Gradiente que chega a cada imagem gerada", xlabel="passos do gerador", ylabel="‖∂L_G/∂G(z)‖ (escala log)")
    ax[1].set(title="O que o discriminador responde", xlabel="passos do gerador", ylabel="probabilidade de ser real", ylim=(0, 1))
    for a in ax:
        a.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "1_vanishing_gradient.png", dpi=150)

    # 2. Qualidade (FID) e cobertura (espécies efetivas), todos os modos
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m, r in R.items():
        if m == "cwgp":                                   # 251 espécies e referência própria: gráfico separado (8_)
            continue
        av = [a for a in r["avaliacoes"] if not a.get("real")]
        it = [a["it_g"] for a in av]
        ax[0].plot(it, [a["fid"] for a in av], color=COR[m], label=NOME[m], marker="o", ms=3)
        ax[1].plot(it, [a["especies_efetivas"] for a in av], color=COR[m], label=NOME[m], marker="o", ms=3)
    ax[0].axhline(piso["fid"], color=REAL, ls=":", lw=1.5, label="piso: duas amostras de sprites reais")
    ax[1].axhline(piso["especies_efetivas"], color=REAL, ls=":", lw=1.5, label="sprites reais")
    ax[0].set_yscale("log")
    ax[0].set(title="FID no espaço do juiz de espécies (menor é melhor)", xlabel="passos do gerador", ylabel="FID (escala log)")
    ax[1].set(title="Espécies efetivas em 2.000 amostras (colapso = poucas)", xlabel="passos do gerador", ylabel="exp(entropia das espécies)")
    for a in ax:
        a.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "2_fid_e_especies.png", dpi=150)

    # 3. WGAN: a estimativa de Wasserstein do crítico acompanha a qualidade
    ws = [m for m in ("wclip", "wgp", "cwgp") if m in R]
    if ws:
        fig, ax = plt.subplots(1, len(ws), figsize=(4.4 * len(ws), 4), squeeze=False)
        for a, m in zip(ax[0], ws):
            h = R[m]["historico"]
            it = np.array([r["it_g"] for r in h])
            a.plot(it[9:], suave([r["w"] for r in h]), color=COR[m])
            a.set(title=f"{NOME[m]}: E[D(real)] − E[D(falso)]", xlabel="passos do gerador", ylabel="estimativa de Wasserstein")
        fig.tight_layout()
        fig.savefig(FIG / "3_wasserstein.png", dpi=150)

    resumo = {m: {k: r["avaliacoes"][-1][k] for k in ("it_g", "fid", "is_", "especies", "especies_efetivas", "diversidade")}
              | {"minutos": round(r["minutos"], 1)} for m, r in R.items()}
    resumo["piso_real"] = piso
    (FIG / "resumo.json").write_text(json.dumps(resumo, indent=1), encoding="utf-8")
    print(json.dumps(resumo, indent=1))


def grade(imgs, cols):
    n = len(imgs)
    W = Image.new("RGB", (cols * 64, -(-n // cols) * 64), "white")
    for i, im in enumerate(imgs):
        W.paste(Image.fromarray(im), ((i % cols) * 64, (i // cols) * 64))
    return W


def _painel(ax, imgs, titulo, cols=6):
    ax.imshow(grade(imgs, cols).resize((cols * 128, -(-len(imgs) // cols) * 128), Image.NEAREST))
    ax.set_title(titulo, fontsize=12)
    ax.axis("off")


def entrada():
    """O que os modelos veem: sprites 64x64 de espécies sorteadas, e os quadros do Pikachu e do Lugia."""
    FIG.mkdir(parents=True, exist_ok=True)
    d = np.load(CACHE)
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(0)
    sorteio = [X[rng.choice(np.flatnonzero(y == e))] for e in rng.choice(721, 48, replace=False)]
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.4), gridspec_kw={"width_ratios": [8, 4, 4]})
    _painel(ax[0], sorteio, "48 espécies sorteadas, como entram no treino (64 x 64)", cols=8)
    _painel(ax[1], X[y == 24][:16], "Pikachu (#25): 16 dos 33 quadros", cols=4)
    _painel(ax[2], X[y == 248][:16], "Lugia (#249): 16 dos 70 quadros", cols=4)
    fig.tight_layout()
    fig.savefig(FIG / "0_entrada.png", dpi=110)


def amostras():
    FIG.mkdir(parents=True, exist_ok=True)
    ms = [m for m in MODOS if (RES / f"{m}_grades.npz").exists() and m != "cwgp"]
    d = np.load(CACHE)
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12.6))
    _painel(ax[0, 0], d["X"][rng.choice(len(d["X"]), 36, replace=False)], "Sprites reais (referência)")
    for a, m in zip(ax.flat[1:], ms):
        g = np.load(RES / f"{m}_grades.npz")
        _painel(a, g["grades"][-1][:36], f"{NOME[m]}, {g['its'][-1]} passos")
    fig.tight_layout()
    fig.savefig(FIG / "4_amostras_finais.png", dpi=100)
    for m in ms:                                          # evolução de cada modo
        d = np.load(RES / f"{m}_grades.npz")
        sel = np.unique(np.linspace(0, len(d["its"]) - 1, 5).round().astype(int))
        fig, ax = plt.subplots(1, len(sel), figsize=(3.2 * len(sel), 1.9))
        for a, k in zip(np.atleast_1d(ax), sel):
            a.imshow(grade(d["grades"][k][:16], 8))
            a.set_title(f"{d['its'][k]} passos", fontsize=9)
            a.axis("off")
        fig.suptitle(NOME[m], fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / f"5_evolucao_{m}.png", dpi=150)


@torch.no_grad()
def ponte(a, b, n_z=4, semente=3, modo="cwgp"):
    """Linhas: ruídos z diferentes. Colunas: quantas camadas iniciais vêm de A (o resto vem de B)."""
    FIG.mkdir(parents=True, exist_ok=True)
    G = Gerador(True).eval()
    G.load_state_dict(torch.load(RES / f"{modo}_G.pt", map_location="cpu"))
    suf = "" if modo == "cwgp" else f"_{modo}"
    ya, yb = torch.tensor([a - 1]), torch.tensor([b - 1])
    z = torch.randn(n_z, NZ, generator=torch.Generator().manual_seed(semente))
    ea, eb = G.embeddings(ya).expand(n_z, -1, -1), G.embeddings(yb).expand(n_z, -1, -1)
    cols, titulos = [], []
    for k in range(CAMADAS_G, -1, -1):
        e = torch.cat([ea[:, :k], eb[:, k:]], 1)
        cols.append(G(z, e=e))
        titulos.append(f"{k} de A, {CAMADAS_G - k} de B")
    d = np.load(CACHE)
    reais = [d["X"][np.flatnonzero(d["y"] == c - 1)[0]] for c in (a, b)]
    pixels = ((reais[0].astype(float) + reais[1]) / 2).astype(np.uint8)    # controle: só média dos pixels
    fig, ax = plt.subplots(n_z, len(cols) + 3, figsize=(1.6 * (len(cols) + 3), 1.7 * n_z))
    for i in range(n_z):
        ax[i, 0].imshow(reais[0])
        ax[i, -2].imshow(reais[1])
        ax[i, -1].imshow(pixels)
        for j, c in enumerate(cols):
            ax[i, j + 1].imshow(((c[i].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1))
    for j, t in enumerate(["A real (#%d)" % a] + titulos + ["B real (#%d)" % b, "média de pixels"]):
        ax[0, j].set_title(t, fontsize=8)
    for x in ax.flat:
        x.axis("off")
    fig.suptitle(f"{NOME[modo]}, ponte #{a} → #{b}: camadas iniciais (forma) de A, finais (cor e textura) de B", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"6_ponte_{a}_{b}{suf}.png", dpi=150)
    print("salvo", FIG / f"6_ponte_{a}_{b}{suf}.png")


@torch.no_grad()
def condicional(especies=(25, 249, 1, 6, 130, 150), n_z=8, modo="cwgp"):
    """WGAN-GP condicional: amostras por espécie (linhas) e ruído z (colunas), e as curvas dos treinos condicionais."""
    FIG.mkdir(parents=True, exist_ok=True)
    G = Gerador(True).eval()
    G.load_state_dict(torch.load(RES / f"{modo}_G.pt", map_location="cpu"))
    suf = "" if modo == "cwgp" else f"_{modo}"
    z = torch.randn(n_z, NZ, generator=torch.Generator().manual_seed(5))
    d = np.load(CACHE)
    fig, ax = plt.subplots(len(especies), n_z + 1, figsize=(1.5 * (n_z + 1), 1.6 * len(especies)))
    for i, e in enumerate(especies):
        ax[i, 0].imshow(d["X"][np.flatnonzero(d["y"] == e - 1)[0]])
        ax[i, 0].set_title(f"#{e} real", fontsize=8)
        x = G(z, torch.full((n_z,), e - 1))
        for j in range(n_z):
            ax[i, j + 1].imshow(((x[j].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1))
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"{NOME[modo]}: cada linha é uma espécie, cada coluna um ruído z", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"7_condicional_especies{suf}.png", dpi=130)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for m in [m for m in ("cwgp", "cwgp_mix") if (RES / f"{m}.json").exists()]:
        r = le(m)
        av = [a for a in r["avaliacoes"] if not a.get("real")]
        h = r["historico"]
        it = [a["it_g"] for a in av]
        ax[0].plot(it, [a["fid"] for a in av], color=COR[m], marker="o", ms=3, label=NOME[m])
        ax[1].plot(it, [a["especies_efetivas"] for a in av], color=COR[m], marker="o", ms=3, label=NOME[m])
        ax[2].plot([q["it_g"] for q in h][9:], suave([q["w"] for q in h]), color=COR[m], label=NOME[m])
    ax[0].set(yscale="log", title="FID no espaço do juiz (251 espécies)", xlabel="passos do gerador", ylabel="FID (escala log)")
    ax[1].set(title="Espécies efetivas em 2.000 amostras", xlabel="passos do gerador", ylabel="exp(entropia das espécies)")
    ax[2].set(title="Estimativa de Wasserstein do crítico", xlabel="passos do gerador", ylabel="E[D(real)] − E[D(falso)]")
    ax[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "8_condicional_curvas.png", dpi=150)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["entrada", "curvas", "amostras", "ponte", "condicional"])
    ap.add_argument("--a", type=int, default=25)
    ap.add_argument("--b", type=int, default=249)
    ap.add_argument("--modo", default="cwgp", choices=["cwgp", "cwgp_mix"])
    x = ap.parse_args()
    {"entrada": entrada, "curvas": curvas, "amostras": amostras, "ponte": lambda: ponte(x.a, x.b, modo=x.modo),
     "condicional": lambda: condicional(modo=x.modo)}[x.acao]()
