"""Leitura dos históricos de treino gravados em resultados/<modo>.json."""
import json

import numpy as np

from config import RES


def le(modo):
    f = RES / f"{modo}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def suave(v, k=10):
    v = np.asarray(v, float)
    return np.convolve(v, np.ones(k) / k, mode="valid") if len(v) > k else v
