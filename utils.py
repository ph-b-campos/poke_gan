"""Utilidades comuns a treino e avaliação."""
import os

import torch


def dispositivo():
    """GPU NVIDIA (cuda), GPU do Mac (mps) ou CPU; a variável DEVICE força um deles."""
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
