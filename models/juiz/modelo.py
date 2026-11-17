"""Juiz de espécies: classificador usado para FID, IS e contagem de espécies (o "Inception" deste projeto)."""
from torch import nn

from config import N_ESP

from .config import CANAIS


class Juiz(nn.Module):
    """Devolve (logits das espécies, features da penúltima camada)."""
    def __init__(self):
        super().__init__()
        c = CANAIS
        self.f = nn.Sequential(*[m for i in range(len(c) - 1) for m in
                                 (nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.BatchNorm2d(c[i + 1]), nn.ReLU(), nn.MaxPool2d(2))],
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cls = nn.Linear(c[-1], N_ESP)

    def forward(self, x):
        h = self.f(x)
        return self.cls(h), h
