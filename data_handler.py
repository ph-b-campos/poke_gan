"""Dados: os 40.597 sprites do Pokémon do D2L, em cache 64x64.

    python -m data_handler                       # usa RAIZ_SPRITES do config.py
    python -m data_handler --raiz .../pokemon
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import CACHE, RAIZ_SPRITES


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=RAIZ_SPRITES)
    prepara(ap.parse_args().raiz)
