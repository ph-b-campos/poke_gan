"""DCGAN do experimento Pokémon: gerador e discriminador/crítico.

A mesma arquitetura serve a todos os modos; o modo só muda a perda e o otimizador (training/dcgan.py).
No modo condicional a espécie entra por BatchNorm condicional no gerador e por projeção no discriminador.
"""
import torch
import torch.nn.functional as F
from torch import nn

from config import N_ESP

from .config import CAMADAS_G, EMB, ND, NG, NZ


class BNCond(nn.Module):
    """BatchNorm sem afim próprio; ganho e viés vêm do embedding da espécie (condicional por camada)."""
    def __init__(self, c, cond):
        super().__init__()
        self.bn = nn.BatchNorm2d(c, affine=not cond)
        self.af = nn.Linear(EMB, 2 * c) if cond else None
        if cond:
            nn.init.zeros_(self.af.weight)
            nn.init.zeros_(self.af.bias)

    def forward(self, h, e):
        h = self.bn(h)
        if self.af is None:
            return h
        g, b = self.af(e).chunk(2, 1)
        return h * (1 + g[..., None, None]) + b[..., None, None]


class Gerador(nn.Module):
    """DCGAN: z (100) -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x3."""
    def __init__(self, cond=False, ng=NG):
        super().__init__()
        cs = [ng * 8, ng * 4, ng * 2, ng]
        self.cond = cond
        self.emb = nn.Embedding(N_ESP, EMB) if cond else None
        self.convs = nn.ModuleList([nn.ConvTranspose2d(NZ, cs[0], 4, 1, 0, bias=False)] +
                                   [nn.ConvTranspose2d(cs[i], cs[i + 1], 4, 2, 1, bias=False) for i in range(3)])
        self.norms = nn.ModuleList([BNCond(c, cond) for c in cs])
        self.saida = nn.ConvTranspose2d(cs[-1], 3, 4, 2, 1)

    def embeddings(self, y):
        """Um embedding por camada; misturar camadas = ponte entre espécies."""
        return self.emb(y)[:, None].expand(-1, CAMADAS_G, -1)

    def forward(self, z, y=None, e=None):
        if self.cond and e is None:
            e = self.embeddings(y)
        h = z[..., None, None]
        for k, (cv, nm) in enumerate(zip(self.convs, self.norms)):
            h = F.relu(nm(cv(h), e[:, k] if self.cond else None))
        return torch.tanh(self.saida(h))


class Discriminador(nn.Module):
    """Mesma rede para D e para o crítico: sem BatchNorm (a penalidade de gradiente é por amostra),
    GroupNorm de um grupo nas camadas internas; projeção da espécie no modo condicional (Miyato e Koyama 2018)."""
    def __init__(self, cond=False, nd=ND):
        super().__init__()
        cs = [nd, nd * 2, nd * 4, nd * 8]
        camadas, ant = [], 3
        for i, c in enumerate(cs):
            camadas += [nn.Conv2d(ant, c, 4, 2, 1)] + ([nn.GroupNorm(1, c)] if i else []) + [nn.LeakyReLU(0.2)]
            ant = c
        self.corpo = nn.Sequential(*camadas)                        # 64 -> 4x4x512
        self.lin = nn.Linear(cs[-1] * 16, 1)
        self.proj = nn.Embedding(N_ESP, cs[-1] * 16) if cond else None

    def forward(self, x, y=None):
        h = self.corpo(x).flatten(1)
        out = self.lin(h).squeeze(1)
        if self.proj is None:
            return out
        return out + (self.proj(y.clamp(min=0)) * h).sum(1) * (y >= 0)     # y = -1: amostra misturada, só "parece Pokémon?"
