"""Dados: os 40.597 sprites do Pokémon do D2L, em cache 64x64.

    python -m data_handler --baixar              # baixa o pokemon.zip, confere o sha1 e monta o cache
    python -m data_handler                       # só monta o cache, a partir de dados/pokemon
    python -m data_handler --raiz .../pokemon    # idem, a partir de outra pasta
"""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from config import CACHE, DADOS_DIR, RAIZ_SPRITES, SHA1_SPRITES, URL_SPRITES


def baixa(destino=DADOS_DIR):
    """Baixa o pokemon.zip do D2L, confere o sha1 e extrai em destino/pokemon."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    arq = destino / "pokemon.zip"
    if not arq.exists():
        print(f"baixando {URL_SPRITES} ...", flush=True)
        urllib.request.urlretrieve(URL_SPRITES, arq)
    h = hashlib.sha1()
    with open(arq, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    if h.hexdigest() != SHA1_SPRITES:
        raise RuntimeError(f"sha1 de {arq} não confere ({h.hexdigest()}); apague o arquivo e baixe de novo")
    with zipfile.ZipFile(arq) as z:
        z.extractall(destino)
    print(f"sprites em {destino / 'pokemon'}")
    return destino / "pokemon"


def prepara(raiz=RAIZ_SPRITES, saida=CACHE):
    """Sprites com transparência sobre fundo branco, centralizados num quadrado, 64x64."""
    from PIL import Image
    arqs = sorted(Path(raiz).glob("*/*.png"), key=lambda f: (int(f.parent.name), f.name))
    if not arqs:
        raise FileNotFoundError(f"nenhum sprite em {raiz}/<número>/*.png; use --baixar ou --raiz")
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
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    np.savez(saida, X=X, y=y)
    print(f"{len(X)} sprites, {len(np.unique(y))} espécies -> {saida}")


def carrega(dev):
    if not Path(CACHE).exists():
        raise FileNotFoundError(f"cache {CACHE} não existe; rode `python -m data_handler --baixar`")
    d = np.load(CACHE)
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).contiguous().to(dev)   # uint8 na GPU: ~500 MB
    return X, torch.from_numpy(d["y"].astype(np.int64)).to(dev)


def norm(xb):
    return xb.float() / 127.5 - 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baixar", action="store_true", help="baixa e extrai o pokemon.zip do D2L antes de montar o cache")
    ap.add_argument("--raiz", default=RAIZ_SPRITES, help="pasta com uma subpasta por número da Pokédex")
    a = ap.parse_args()
    prepara(baixa() if a.baixar else a.raiz)
