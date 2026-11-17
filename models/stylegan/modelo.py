"""Gerador StyleGAN-like (mapeamento + síntese) e discriminador residual de dois logits."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from config import N_ESP

from .camadas import SQ2, Camada, Conv, Linear, ModConv, lrelu, norm2
from .config import CH, K_PALETA, RESOLUCOES, W_DIM, Z_DIM


# ------------------------------------------------------------------ gerador
class Mapeamento(nn.Module):
    def __init__(self, camadas):
        super().__init__()
        self.emb = nn.Embedding(N_ESP, Z_DIM)
        dims = [2 * Z_DIM] + [W_DIM] * camadas
        self.fc = nn.ModuleList([Linear(a, b, lr_mul=0.01) for a, b in zip(dims[:-1], dims[1:])])
        self.lam = 1.0                                                     # peso da espécie (treino transicional)

    def forward(self, z, y):
        x = torch.cat([norm2(z), self.lam * norm2(self.emb(y))], 1)
        for f in self.fc:
            x = lrelu(f(x))
        return x


class Gerador(nn.Module):
    """Posições de estilo: uma por resolução (4, 8, 16, 32, 64) e, nos modos paleta e cor, mais uma para a cor.
    saida = "rgb":    StyleGAN2 skip, cada toRGB usa o estilo da sua resolução
            "cor":    StyleGAN2 skip, todas as toRGB usam a posição de cor (no StyleSpace, Wu et al. 2021, os
                      estilos das toRGB só mudam cor, nunca forma): estrutura de A + cor de B
            "paleta": K slots de cor por softmax; FALHOU (16/11): o slot branco virou estado absorvente,
                      o gerador pintou tudo de fundo, o softmax saturou e o gradiente sumiu"""
    def __init__(self, saida="cor", mapa=4):
        super().__init__()
        self.saida = saida
        self.n_ws = len(RESOLUCOES) + (saida in ("paleta", "cor"))
        self.mapa = Mapeamento(mapa)
        self.const = nn.Parameter(torch.randn(CH[4], 4, 4))
        self.blocos, ant = nn.ModuleList(), CH[4]
        for r in RESOLUCOES:
            self.blocos.append(nn.ModuleList(([Camada(ant, CH[r], up=True)] if r > 4 else []) + [Camada(CH[r], CH[r])]))
            ant = CH[r]
        if saida != "paleta":
            self.rgb = nn.ModuleList([ModConv(CH[r], 3, 1, demod=False) for r in RESOLUCOES])
            self.rgb_b = nn.Parameter(torch.zeros(len(RESOLUCOES), 3))
        else:
            self.slots = ModConv(CH[64], K_PALETA + 1, 1, demod=False)     # +1: o fundo branco
            self.slots_b = nn.Parameter(torch.zeros(K_PALETA + 1))
            self.cor1, self.cor2 = Linear(W_DIM, W_DIM), Linear(W_DIM, 3 * K_PALETA)

    def ws(self, z, y):
        return self.mapa(z, y)[:, None].repeat(1, self.n_ws, 1)

    def paleta(self, w):
        """Slot 0: fundo branco. Slots 1 a K: as cores em ordem de luminância, para o slot i ter o mesmo papel em
        toda espécie (o claro do Pikachu recebe o claro do Lugia, o contorno escuro recebe o escuro)."""
        cores = torch.tanh(self.cor2(lrelu(self.cor1(w)))).view(len(w), K_PALETA, 3)
        lum = cores @ torch.tensor([0.299, 0.587, 0.114], device=w.device)
        cores = cores.gather(1, lum.argsort(1)[..., None].expand(-1, -1, 3))
        return torch.cat([torch.ones_like(cores[:, :1]), cores], 1)

    def regioes(self, x, w):
        return (self.slots(x, w) + self.slots_b[None, :, None, None]).softmax(1)

    def sintese(self, ws, devolve_regioes=False):
        x = self.const[None].expand(len(ws), -1, -1, -1)
        img = None
        for k, bloco in enumerate(self.blocos):
            for cam in bloco:
                x = cam(x, ws[:, k])
            if self.saida != "paleta":
                y = self.rgb[k](x, ws[:, -1] if self.saida == "cor" else ws[:, k]) + self.rgb_b[k][None, :, None, None]
                img = y if img is None else F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False) + y
        if self.saida != "paleta":
            return img
        a = self.regioes(x, ws[:, len(RESOLUCOES) - 1])                       # (B, K+1, 64, 64)
        img = torch.einsum("bkhw,bkc->bchw", a, self.paleta(ws[:, -1]))
        return (img, a) if devolve_regioes else img

    def forward(self, z, y):
        return self.sintese(self.ws(z, y))


def ponte(G, wa, wb, k):
    """k primeiras posições (as grossas) de A, o resto de B."""
    return G.sintese(torch.cat([wa[:, :k], wb[:, k:]], 1))


# ------------------------------------------------------------------ discriminador (residual, StyleGAN2)
def desvio_lote(x, g=4):
    """Desvio padrão do minilote como canal extra: dá ao D um jeito de ver falta de variedade."""
    B, C, H, W = x.shape
    g = min(g, B)
    s = x.reshape(g, -1, C, H, W)
    s = (s - s.mean(0)).square().mean(0).add(1e-8).sqrt().mean([1, 2, 3])
    return torch.cat([x, s.reshape(-1, 1, 1, 1).repeat(g, 1, H, W)], 1)


class BlocoD(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1, self.c2, self.atalho = Conv(i, i, 3), Conv(i, o, 3), Conv(i, o, 1, bias=False)

    def forward(self, x):
        a = self.atalho(F.avg_pool2d(x, 2))
        x = F.avg_pool2d(lrelu(self.c2(lrelu(self.c1(x)))), 2)
        return (x + a) / SQ2


class Discriminador(nn.Module):
    """Dois logits (Shahbazi et al. 2022): u = "parece Pokémon?", para toda imagem; c = u + projeção da espécie
    (Miyato e Koyama 2018), só para as que têm espécie. Misturadas (rótulo -1) só entram em u."""
    def __init__(self):
        super().__init__()
        self.de_rgb = Conv(3, CH[64], 1)
        self.blocos = nn.Sequential(*[BlocoD(CH[r], CH[r // 2]) for r in (64, 32, 16, 8)])
        self.conv = Conv(CH[4] + 1, CH[4], 3)
        self.fc = Linear(CH[4] * 16, CH[4])
        self.saida = Linear(CH[4], 1)
        self.proj = nn.Embedding(N_ESP, CH[4])
        self.cls = Linear(CH[4], N_ESP)                                    # que espécie é esta? (AC-GAN)

    def forward(self, x, y):
        h = self.blocos(lrelu(self.de_rgb(x)))
        h = lrelu(self.fc(lrelu(self.conv(desvio_lote(h))).flatten(1)))
        u = self.saida(h).squeeze(1)
        return u, u + (self.proj(y.clamp(min=0)) * h).sum(1) / math.sqrt(h.shape[1]), self.cls(h)
