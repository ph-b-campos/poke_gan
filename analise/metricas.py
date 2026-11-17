"""Métricas de avaliação.

    avalia, referencia   no espaço do juiz de espécies: FID, IS, espécies distintas e efetivas, diversidade em pixels
    medidas_ponte        o que cada corte da ponte StyleGAN leva de A (forma) e de B (cor), com controle
"""
import torch
import torch.nn.functional as F

from config import N_ESP
from data_handler import norm
from models.stylegan import ponte


# ------------------------------------------------------------------ espaço do juiz
@torch.no_grad()
def avalia(J, imgs, ref):
    """Métricas no espaço do juiz: FID, IS, espécies distintas e diversidade em pixels."""
    logit, h = zip(*(J(imgs[i:i + 500]) for i in range(0, len(imgs), 500)))
    p, h = torch.cat(logit).cpu().softmax(1), torch.cat(h).cpu().double()   # float64 e eigvals na CPU (o MPS não tem)
    mu, C = h.mean(0), torch.cov(h.T)
    v = torch.linalg.eigvals(C @ ref["C"]).real.clamp(min=0).sqrt().sum()
    fid = float(((mu - ref["mu"]) ** 2).sum() + torch.trace(C) + torch.trace(ref["C"]) - 2 * v)
    py = p.mean(0)
    is_ = float(torch.exp((p * (p.clamp_min(1e-12).log() - py.clamp_min(1e-12).log())).sum(1).mean()))
    contagem = torch.bincount(p.argmax(1), minlength=N_ESP).double()
    q = contagem / contagem.sum()
    efetivas = float(torch.exp(-(q[q > 0] * q[q > 0].log()).sum()))
    a, b = imgs[: len(imgs) // 2].flatten(1), imgs[len(imgs) // 2:].flatten(1)
    div = float((a - b).norm(dim=1).mean())
    return dict(fid=fid, is_=is_, especies=int((contagem > 0).sum()), especies_efetivas=efetivas, diversidade=div)


def referencia(J, X, dev, n=2000):
    g = torch.Generator(device="cpu").manual_seed(1)
    imgs = norm(X[torch.randperm(len(X), generator=g)[:n].to(dev)])
    logit, h = zip(*(J(imgs[i:i + 500]) for i in range(0, n, 500)))
    h = torch.cat(h).cpu().double()
    ref = {"mu": h.mean(0), "C": torch.cov(h.T)}
    return ref, imgs


# ------------------------------------------------------------------ medidas da ponte (StyleGAN)
def mascara(x):
    """Figura = tudo menos o branco ligado à borda (o corpo branco do Lugia é figura, não fundo)."""
    branco = (x > 0.8).all(1, keepdim=True).float()
    fundo = torch.zeros_like(branco)
    fundo[..., 0, :] = fundo[..., -1, :] = fundo[..., :, 0] = fundo[..., :, -1] = 1
    fundo = fundo * branco
    for _ in range(64):                                                    # inundação a partir da borda
        fundo = F.max_pool2d(fundo, 3, 1, 1) * branco
    return fundo[:, 0] < 0.5


def histograma(x, m, n=4):
    q = ((x.clamp(-1, 1) + 1) / 2 * (n - 1e-3)).long()
    idx = (q[:, 0] * n + q[:, 1]) * n + q[:, 2]
    h = torch.zeros(len(x), n ** 3, device=x.device).scatter_add_(1, idx.flatten(1), m.flatten(1).float())
    return h / h.sum(1, keepdim=True).clamp_min(1)


@torch.no_grad()
def medidas_ponte(G, wa, wb):
    """Para cada corte k (k posições grossas de A, o resto de B):
       forma = IoU da silhueta com a de A puro; cor = d(A) / (d(A) + d(B)), distâncias L1 entre histogramas de cor
       da figura (0 = cores de A, 1 = cores de B). Controle: média de pixels de A e B."""
    A, B = G.sintese(wa), G.sintese(wb)
    ma, ha, hb = mascara(A), histograma(A, mascara(A)), histograma(B, mascara(B))

    def mede(X):
        m = mascara(X)
        iou = ((m & ma).sum([1, 2]) / (m | ma).sum([1, 2]).clamp_min(1)).float()
        h = histograma(X, m)
        da, db = (h - ha).abs().sum(1), (h - hb).abs().sum(1)
        return float(iou.mean()), float((da / (da + db + 1e-8)).mean())

    r = {str(k): mede(ponte(G, wa, wb, k)) for k in range(G.n_ws + 1)}
    r["pixels"] = mede((A + B) / 2)
    return r
