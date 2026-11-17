"""Estilo comum das figuras: backend, rcParams, cores e nomes de cada modo."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 2})

# ---------------------------------------------------------------------------
# DCGAN
# ---------------------------------------------------------------------------
MODOS = ("minimax", "ns", "colapso", "wclip", "wgp", "cwgp")
NOME = {"minimax": "GAN minimax (saturante)", "ns": "GAN não saturante", "colapso": "GAN não saturante, D pequeno", "wclip": "WGAN com clipping",
        "wgp": "WGAN-GP", "cwgp": "WGAN-GP condicional", "cwgp_mix": "WGAN-GP condicional + mixing"}
COR = {"minimax": "#D55E00", "ns": "#E69F00", "colapso": "#CC79A7", "wclip": "#56B4E9", "wgp": "#0072B2", "cwgp": "#009E73",
       "cwgp_mix": "#000000"}  # Okabe-Ito, ordem fixa
REAL = "#555555"

# ---------------------------------------------------------------------------
# StyleGAN
# ---------------------------------------------------------------------------
COR_LINHA = {"forma": "#0072B2", "cor": "#D55E00"}
