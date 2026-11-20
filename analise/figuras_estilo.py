"""Figuras de avaliação do mix de domínio StyleGAN-like (models/stylegan): curvas, grade de misturas, ponte,
como a paleta funciona e medidas por ponto de corte.

    python -m analise.figuras_estilo --modo cor                                   # todas as figuras, pares padrão
    python -m analise.figuras_estilo --modo cor --pares 25:249 4:131 --so-misturas  # só as misturas desses pares
    python -m analise.figuras_estilo --forma 25 1 133 --estilo 249 6 --so-misturas  # grade: forma x estilo

As espécies são números da Pokédex (1 a CONFIG["especies"], gerações 1 a 3). Em --pares, A:B leva a forma
(posições grossas) de A e o estilo (posições finas) de B.
"""
import argparse
import json

import numpy as np
import torch

from analise.historico import le, suave
from analise.metricas import avalia, medidas_ponte, referencia
from analise.plot_config import COR_LINHA, REAL, plt
from config import CACHE, FIG, RES
from data_handler import carrega
from models.juiz import Juiz
from models.stylegan import Gerador, ponte
from models.stylegan.config import CONFIG, MODO, RESOLUCOES, SAIDA, Z_DIM
from utils import dispositivo

A_ESP = (25, 1, 133, 143, 39, 54)        # forma: Pikachu, Bulbasaur, Eevee, Snorlax, Jigglypuff, Psyduck
B_ESP = (249, 6, 130, 94, 7, 150)        # estilo: Lugia, Charizard, Gyarados, Gengar, Squirtle, Mewtwo
PARES = ((25, 249), (249, 25), (1, 6), (6, 1), (150, 25), (133, 94))      # Pikachugia e Lugachu, e outros pares
PARES_PALETA = ((25, 249), (1, 6), (133, 94))                             # figura "como a paleta funciona"


def posicoes(G):
    return [f"{r}×{r}" for r in RESOLUCOES] + ([G.saida] if G.n_ws > len(RESOLUCOES) else [])


def rotulo(G, k):
    p = posicoes(G)
    return "só A" if k == G.n_ws else "só B" if k == 0 else f"A: até {p[k - 1]}\nB: {p[k]} em diante"


def img(x):
    return ((x.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)


def carrega_G(modo, dev):
    G = Gerador(SAIDA[modo], CONFIG["mapa"]).to(dev).eval().requires_grad_(False)
    G.load_state_dict(torch.load(RES / f"{modo}_G.pt", map_location=dev))
    return G


def ws_de(G, especies, semente, dev):
    z = torch.randn(len(especies), Z_DIM, generator=torch.Generator().manual_seed(semente)).to(dev)
    return G.ws(z, torch.tensor([e - 1 for e in especies], device=dev))


def sintese(G, ws, semente=0):
    torch.manual_seed(semente)                                    # mesmo ruído por pixel em todas as colunas
    return G.sintese(ws)


@torch.no_grad()
def reais(dev):
    """Juiz, referência do FID e métricas dos sprites reais, só nas espécies do treino (gerações 1 a 3)."""
    X, y = carrega(dev)
    m = y < CONFIG["especies"]
    X, y = X[m], y[m]
    J = Juiz().to(dev).eval()
    J.load_state_dict(torch.load(RES / "juiz.pt", map_location=dev))
    ref, r = referencia(J, X, dev)
    base = avalia(J, r, ref)
    i = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:2000].to(dev)
    base["acerto_especie"] = float((J(X[i].float() / 127.5 - 1)[0].argmax(1) == y[i]).float().mean())
    return J, ref, base, y


def real(e):
    d = np.load(CACHE)
    return d["X"][np.flatnonzero(d["y"] == e - 1)[0]]


# ------------------------------------------------------------------ 1. curvas
def curvas(modo, base):
    r = le(modo)
    av, h = r["avaliacoes"], r["historico"]
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    kimg = [a["kimg"] for a in av]
    ax[0, 0].plot(kimg, [a["fid"] for a in av], color="#000000", marker="o", ms=3, label="este modelo")
    for m, cor, nome in (("wgp", "#0072B2", "WGAN-GP (DCGAN)"), ("ns", "#E69F00", "GAN não saturante (DCGAN)")):
        o = le(m)
        if o:
            a = [q for q in o["avaliacoes"] if not q.get("real")]
            ax[0, 0].plot([q["it_g"] * o["lote"] / 1000 for q in a], [q["fid"] for q in a], color=cor, lw=1.5, alpha=0.8,
                          label=nome + ", 721 espécies")
    ax[0, 0].set(yscale="log", title="FID no espaço do juiz (este modelo: gerações 1 a 3)", xlabel="mil imagens vistas pelo gerador",
                 ylabel="FID (escala log)")
    ax[0, 0].legend(fontsize=9)
    ax[0, 1].plot(kimg, [a["especies_efetivas"] for a in av], color="#000000", marker="o", ms=3, label="este modelo")
    ax[0, 1].axhline(base["especies_efetivas"], color=REAL, ls="--", lw=1.5, label="sprites reais")
    ax[0, 1].set(title="Espécies efetivas em 2.000 amostras", xlabel="mil imagens vistas pelo gerador", ylabel="exp(entropia das espécies)")
    ax[0, 1].legend(fontsize=9)
    ax[0, 2].plot(kimg, [a.get("acerto_especie", np.nan) for a in av], color="#000000", marker="o", ms=3, label="amostras puras do modelo")
    ax[0, 2].axhline(base["acerto_especie"], color=REAL, ls="--", lw=1.5, label="sprites reais")
    ax[0, 2].set(ylim=(0, 1.02), title="O juiz reconhece a espécie pedida?", xlabel="mil imagens vistas pelo gerador",
                 ylabel="fração de 2.000 amostras")
    ax[0, 2].legend(fontsize=9)
    for chave, cor, nome in (("ce_d", REAL, "sprites reais (treina o D)"), ("ce_g", "#009E73", "amostras puras (cobra o G)")):
        s = suave([q.get(chave, np.nan) for q in h], 20)
        ax[1, 2].plot([q["kimg"] for q in h][len(h) - len(s):], s, color=cor, label=nome)
    ax[1, 2].set(title="Classificador de espécies do D: entropia cruzada", xlabel="mil imagens vistas pelo gerador",
                 ylabel="entropia cruzada (média móvel)")
    ax[1, 2].legend(fontsize=9)
    kh = [q["kimg"] for q in h]
    for chave, cor, nome in (("d_real", REAL, "reais"), ("d_puro", "#009E73", "falsos de uma espécie"), ("d_mist", "#CC79A7", "falsos misturados")):
        v = [q[chave] if q[chave] is not None else np.nan for q in h]
        s = suave(v, 20)
        ax[1, 0].plot(kh[len(kh) - len(s):], s, color=cor, label=nome)
    ax[1, 0].set(title="O que o discriminador dá a cada tipo de imagem", xlabel="mil imagens vistas pelo gerador", ylabel="logit de D (média móvel)")
    ax[1, 0].legend(fontsize=9)
    G_nws = len(av[-1]["ponte"]) - 2
    for k, ls in ((G_nws - 1, "-"), (2, "--")):
        r_k = RESOLUCOES[k - 1]
        ax[1, 1].plot(kimg, [a["ponte"][str(k)][0] for a in av], color=COR_LINHA["forma"], ls=ls, label=f"forma de A (A até {r_k}×{r_k})")
        ax[1, 1].plot(kimg, [a["ponte"][str(k)][1] for a in av], color=COR_LINHA["cor"], ls=ls, label=f"cor de B (A até {r_k}×{r_k})")
    ax[1, 1].set(ylim=(0, 1.02), title="A ponte durante o treino (128 pares fixos)", xlabel="mil imagens vistas pelo gerador",
                 ylabel="0 a 1 (1 = forma toda de A / cor toda de B)")
    ax[1, 1].legend(fontsize=8)
    t0, t1 = r["config"]["transicao_kimg"]
    for a in ax.flat:                                                      # faixa em que a espécie entra (lam de 0 a 1)
        a.axvspan(t0, t1, color="#999999", alpha=0.15, lw=0)
    fig.suptitle("Faixa cinza: a espécie entra aos poucos (lam de 0 a 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"9_{modo}_curvas.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 2. grade de misturas, como na figura 3 da StyleGAN
@torch.no_grad()
def grade(G, modo, k, dev, forma=A_ESP, estilo=B_ESP):
    wa, wb = ws_de(G, forma, 11, dev), ws_de(G, estilo, 12, dev)
    pa, pb = sintese(G, wa), sintese(G, wb)
    n, m = len(forma), len(estilo)
    fig, ax = plt.subplots(n + 1, m + 1, figsize=(1.7 * (m + 1), 1.75 * (n + 1)), squeeze=False)
    for j in range(m):
        ax[0, j + 1].imshow(img(pb[j]))
        ax[0, j + 1].set_title(f"#{estilo[j]}", fontsize=9)
    for i in range(n):
        ax[i + 1, 0].imshow(img(pa[i]))
        ax[i + 1, 0].set_title(f"#{forma[i]}", fontsize=9)
        x = sintese(G, torch.cat([wa[i:i + 1, :k].expand(m, -1, -1), wb[:, k:]], 1))
        for j in range(m):
            ax[i + 1, j + 1].imshow(img(x[j]))
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"Linhas: forma de A (coluna da esquerda). Colunas: estilo de B (linha de cima).\n{rotulo(G, k).replace(chr(10), '; ')}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"10_{modo}_mistura_k{k}.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ 3. a ponte de um par, corte a corte
@torch.no_grad()
def ponte_par(G, modo, a, b, dev, n_z=4):
    wa, wb = ws_de(G, [a] * n_z, 21, dev), ws_de(G, [b] * n_z, 22, dev)
    ks = list(range(G.n_ws, -1, -1))
    cols = [sintese(G, torch.cat([wa[:, :k], wb[:, k:]], 1)) for k in ks]
    ra, rb = real(a), real(b)
    media = ((ra.astype(float) + rb) / 2).astype(np.uint8)
    fig, ax = plt.subplots(n_z, len(ks) + 3, figsize=(1.55 * (len(ks) + 3), 1.75 * n_z + 0.6))
    for i in range(n_z):
        ax[i, 0].imshow(ra)
        ax[i, -2].imshow(rb)
        ax[i, -1].imshow(media)
        for j, c in enumerate(cols):
            ax[i, j + 1].imshow(img(c[i]))
    for j, t in enumerate([f"A real (#{a})"] + [rotulo(G, k) for k in ks] + [f"B real (#{b})", "controle:\nmédia de pixels"]):
        ax[0, j].set_title(t, fontsize=7)
    for x in ax.flat:
        x.axis("off")
    fig.suptitle(f"Mix de domínio #{a} → #{b}: posições grossas de A, finas de B (cada linha, um ruído z)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"11_{modo}_ponte_{a}_{b}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 4. como a paleta funciona (só no modo paleta)
@torch.no_grad()
def como_funciona(G, modo, pares, dev):
    fig, ax = plt.subplots(len(pares), 6, figsize=(13, 2.3 * len(pares)), squeeze=False)
    cmap = plt.get_cmap("tab20")
    for i, (a, b) in enumerate(pares):
        wa, wb = ws_de(G, [a], 31, dev), ws_de(G, [b], 32, dev)
        torch.manual_seed(0)
        xa, ra = G.sintese(wa, devolve_regioes=True)
        xb = sintese(G, wb)
        mix = sintese(G, torch.cat([wa[:, :-1], wb[:, -1:]], 1))
        rot = ra[0].argmax(0).cpu().numpy()
        mapa = np.where(rot[..., None] == 0, 1.0, np.array([cmap(int(s) % 20)[:3] for s in rot.flat]).reshape(64, 64, 3))
        pal_a = ((G.paleta(wa[:, -1])[0].cpu().numpy() + 1) / 2).clip(0, 1)[None]
        pal_b = ((G.paleta(wb[:, -1])[0].cpu().numpy() + 1) / 2).clip(0, 1)[None]
        for j, (im, t) in enumerate([(img(xa[0]), f"A gerado (#{a})"), (mapa, "regiões de A\n(um tom por slot)"),
                                     (pal_a, "paleta de A"), (pal_b, f"paleta de B (#{b})"),
                                     (img(mix[0]), "regiões de A,\npaleta de B"), (img(xb[0]), f"B gerado (#{b})")]):
            ax[i, j].imshow(im, aspect="auto" if j in (2, 3) else "equal", interpolation="nearest")
            ax[i, j].set_title(t, fontsize=9)
            ax[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / f"12_{modo}_como_funciona.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 5. medidas por corte, com controle
@torch.no_grad()
def medidas(G, modo, dev, J, ref, y, n_pares=1000, n_fid=2000):
    g = torch.Generator(device=dev).manual_seed(41)
    pares = torch.randint(0, CONFIG["especies"], (2, n_pares), device=dev, generator=g)
    soma = {}
    for s in range(0, n_pares, 250):
        torch.manual_seed(s)
        wa = G.ws(torch.randn(250, Z_DIM, device=dev, generator=g), pares[0, s:s + 250])
        wb = G.ws(torch.randn(250, Z_DIM, device=dev, generator=g), pares[1, s:s + 250])
        for k, v in medidas_ponte(G, wa, wb).items():
            soma.setdefault(k, []).append(v)
    med = {k: [float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs]))] for k, vs in soma.items()}
    # veredito do juiz: em quantas misturas o classificador de espécies vê A, e em quantas vê B (identidade, não só cor)
    vered = {}
    for k in range(G.n_ws + 1):
        va = vb = 0
        for s in range(0, n_pares, 250):
            torch.manual_seed(s)
            ya, yb = pares[0, s:s + 250], pares[1, s:s + 250]
            wa = G.ws(torch.randn(len(ya), Z_DIM, device=dev, generator=g), ya)
            wb = G.ws(torch.randn(len(yb), Z_DIM, device=dev, generator=g), yb)
            p = J(ponte(G, wa, wb, k).clamp(-1, 1))[0].argmax(1)
            va, vb = va + int((p == ya).sum()), vb + int((p == yb).sum())
        vered[str(k)] = [va / n_pares, vb / n_pares]
    fid = {}
    for k in range(G.n_ws + 1):                       # FID das misturas: elas parecem Pokémon?
        imgs = []
        for s in range(0, n_fid, 250):
            ya = y[torch.randint(0, len(y), (250,), device=dev, generator=g)]
            yb = y[torch.randint(0, len(y), (250,), device=dev, generator=g)]
            wa = G.ws(torch.randn(250, Z_DIM, device=dev, generator=g), ya)
            wb = G.ws(torch.randn(250, Z_DIM, device=dev, generator=g), yb)
            imgs.append(ponte(G, wa, wb, k).clamp(-1, 1))
        fid[str(k)] = avalia(J, torch.cat(imgs), ref)["fid"]
    (FIG / f"{modo}_medidas.json").write_text(json.dumps(dict(forma_cor=med, veredito_juiz=vered, fid_misturas=fid, n_pares=n_pares),
                                                          indent=1), encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(20, 5.5))
    for k in range(G.n_ws + 1):
        f, c = med[str(k)]
        ax[0].scatter(f, c, s=60, color="#000000", zorder=3)
        ax[0].annotate(rotulo(G, k).replace("\n", " | "), (f, c), fontsize=7, xytext=(5, 4), textcoords="offset points")
    f, c = med["pixels"]
    ax[0].scatter(f, c, s=80, marker="X", color=REAL, zorder=3)
    ax[0].annotate("controle: média de pixels", (f, c), fontsize=8, color=REAL, xytext=(5, -10), textcoords="offset points")
    ax[0].axhline(0.5, color=REAL, lw=0.8, ls=":")
    ax[0].set(xlim=(0, 1.05), ylim=(-0.02, 1.05), xlabel="forma de A (IoU da silhueta com A)", ylabel="cor de B (0 = cores de A, 1 = cores de B)",
              title=f"O que cada corte leva de A e de B ({n_pares} pares de espécies)")
    ks = list(range(G.n_ws, -1, -1))
    ax[1].bar(range(len(ks)), [fid[str(k)] for k in ks], color=["#009E73" if k in (0, G.n_ws) else "#000000" for k in ks])
    ax[1].set_xticks(range(len(ks)), [rotulo(G, k) for k in ks], fontsize=7, rotation=35, ha="right")
    ax[1].set(ylabel="FID no espaço do juiz", title=f"As misturas parecem Pokémon? ({n_fid} imagens por corte; verde = sem mistura)")
    x = np.arange(len(ks))
    ax[2].bar(x - 0.2, [vered[str(k)][0] for k in ks], 0.4, color="#0072B2", label="juiz vê A (quem deu o grosso)")
    ax[2].bar(x + 0.2, [vered[str(k)][1] for k in ks], 0.4, color="#D55E00", label="juiz vê B (quem deu o fino)")
    ax[2].set_xticks(x, [rotulo(G, k) for k in ks], fontsize=7, rotation=35, ha="right")
    ax[2].set(ylim=(0, 1), ylabel=f"fração de {n_pares} misturas", title="Que espécie o juiz enxerga em cada corte?")
    ax[2].legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / f"13_{modo}_medidas_corte.png", dpi=150)
    plt.close(fig)
    print(json.dumps(dict(forma_cor=med, veredito_juiz=vered, fid_misturas=fid), indent=1))


# ------------------------------------------------------------------ 6. espécies e ruído z
@torch.no_grad()
def especies(G, modo, dev, lista=(25, 249, 1, 6, 130, 150, 94, 133), n_z=8):
    fig, ax = plt.subplots(len(lista), n_z + 1, figsize=(1.5 * (n_z + 1), 1.6 * len(lista)))
    for i, e in enumerate(lista):
        ax[i, 0].imshow(real(e))
        ax[i, 0].set_title(f"#{e} real", fontsize=8)
        x = sintese(G, ws_de(G, [e] * n_z, 51, dev))
        for j in range(n_z):
            ax[i, j + 1].imshow(img(x[j]))
    for a in ax.flat:
        a.axis("off")
    fig.suptitle("Cada linha é uma espécie, cada coluna um ruído z", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"14_{modo}_especies.png", dpi=130)
    plt.close(fig)


def figuras(modo, pares=None, forma=A_ESP, estilo=B_ESP, so_misturas=False):
    """pares=None usa PARES (e PARES_PALETA no modo paleta). so_misturas=True gera só a grade, as pontes e,
    no modo paleta, "como funciona": pula curvas, espécies e medidas, que carregam o juiz e são as partes lentas."""
    FIG.mkdir(parents=True, exist_ok=True)
    dev = dispositivo()
    G = carrega_G(modo, dev)
    if not so_misturas:
        J, ref, base, y = reais(dev)
        print(f"sprites reais (gerações 1 a 3): {json.dumps(base)}")
        curvas(modo, base)
    for k in ((2, 3, G.n_ws - 1) if G.n_ws > len(RESOLUCOES) else (2, 3)):       # grosso até 8, até 16, só a cor
        grade(G, modo, k, dev, forma, estilo)
    for a, b in pares or PARES:
        ponte_par(G, modo, a, b, dev)
    if modo == "paleta":
        como_funciona(G, modo, pares or PARES_PALETA, dev)
    if not so_misturas:
        especies(G, modo, dev)
        medidas(G, modo, dev, J, ref, y)
    print("figuras em", FIG)


def _especie(s):
    """Número da Pokédex dentro das espécies do treino."""
    e = int(s)
    if not 1 <= e <= CONFIG["especies"]:
        raise argparse.ArgumentTypeError(f"{e} fora do treino: use 1 a {CONFIG['especies']}")
    return e


def _par(s):
    """'A:B' -> (A, B): forma de A, estilo de B."""
    try:
        a, b = s.split(":")
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{s}' não é um par A:B (ex.: 25:249)") from None
    return _especie(a), _especie(b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=list(SAIDA), default=MODO)
    ap.add_argument("--pares", type=_par, nargs="+", metavar="A:B",
                    help="pares da ponte, forma de A e estilo de B (padrão: %s)" % " ".join(f"{a}:{b}" for a, b in PARES))
    ap.add_argument("--forma", type=_especie, nargs="+", default=A_ESP, metavar="N",
                    help="linhas da grade: espécies que dão a forma (padrão: %s)" % " ".join(map(str, A_ESP)))
    ap.add_argument("--estilo", type=_especie, nargs="+", default=B_ESP, metavar="N",
                    help="colunas da grade: espécies que dão o estilo (padrão: %s)" % " ".join(map(str, B_ESP)))
    ap.add_argument("--so-misturas", action="store_true",
                    help="só a grade e as pontes; pula curvas, espécies e medidas (não precisa do juiz)")
    x = ap.parse_args()
    figuras(x.modo, x.pares, tuple(x.forma), tuple(x.estilo), x.so_misturas)
