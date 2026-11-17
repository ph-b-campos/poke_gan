"""Camadas com taxa de aprendizado equalizada (Karras et al. 2018) e a convolução modulada da StyleGAN2."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import W_DIM

SQ2 = math.sqrt(2)


def lrelu(x):
    return F.leaky_relu(x, 0.2) * SQ2


def norm2(x):
    return x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()


class Linear(nn.Module):
    def __init__(self, i, o, lr_mul=1.0, bias_init=0.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i) / lr_mul)
        self.b = nn.Parameter(torch.full((o,), float(bias_init)))
        self.g, self.lr_mul = lr_mul / math.sqrt(i), lr_mul

    def forward(self, x):
        return F.linear(x, self.w * self.g, self.b * self.lr_mul)


class Conv(nn.Module):
    def __init__(self, i, o, k, bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i, k, k))
        self.b = nn.Parameter(torch.zeros(o)) if bias else None
        self.g, self.p = 1 / math.sqrt(i * k * k), k // 2

    def forward(self, x):
        return F.conv2d(x, self.w * self.g, self.b, padding=self.p)


class ModConv(nn.Module):
    """Convolução modulada (StyleGAN2): o estilo escala cada canal de entrada; a demodulação devolve a variância a 1.
    Feita modulando a ativação, não o peso: mesma conta, sem convolução agrupada."""
    def __init__(self, i, o, k, demod=True, up=False):
        super().__init__()
        self.afim = Linear(W_DIM, i, bias_init=1)
        self.w = nn.Parameter(torch.randn(o, i, k, k))
        self.g, self.p, self.demod, self.up = 1 / math.sqrt(i * k * k), k // 2, demod, up

    def forward(self, x, w):
        s = self.afim(w)                                                   # (B, i): o estilo desta camada
        peso = self.w * self.g
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.conv2d(x * s[:, :, None, None], peso, padding=self.p)
        if self.demod:
            d = ((peso[None] * s[:, None, :, None, None]).square().sum([2, 3, 4]) + 1e-8).rsqrt()
            x = x * d[:, :, None, None]
        return x


class Camada(nn.Module):
    def __init__(self, i, o, up=False):
        super().__init__()
        self.conv = ModConv(i, o, 3, up=up)
        self.ruido = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(o))

    def forward(self, x, w):
        x = self.conv(x, w)
        x = x + torch.randn(len(x), 1, *x.shape[2:], device=x.device) * self.ruido
        return lrelu(x + self.b[None, :, None, None])
