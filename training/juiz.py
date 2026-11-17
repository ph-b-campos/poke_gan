"""Treino do juiz de espécies (o "Inception" do FID). Rodar uma vez, antes de qualquer GAN.

    python -m training.juiz

Configuração em models/juiz/config/. Salva resultados/juiz.pt e resultados/juiz.json.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from config import RES
from data_handler import carrega, norm
from models.juiz import Juiz
from models.juiz.config import EPOCAS, LOTE, LOTE_AVAL, LR, N_TESTE
from utils import dispositivo


def treina_juiz(epocas=EPOCAS, dev=None):
    dev = dev or dispositivo()
    torch.manual_seed(0)
    X, y = carrega(dev)
    idx = torch.randperm(len(X), device=dev)
    te, tr = idx[:N_TESTE], idx[N_TESTE:]
    J = Juiz().to(dev)
    opt = torch.optim.Adam(J.parameters(), LR)
    for ep in range(epocas):
        J.train()
        for i in range(0, len(tr), LOTE):
            b = tr[i:i + LOTE]
            xb = norm(X[b])
            xb = torch.where(torch.rand(len(b), 1, 1, 1, device=dev) < 0.5, xb.flip(3), xb)
            loss = F.cross_entropy(J(xb)[0], y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = tr[torch.randperm(len(tr), device=dev)]
        J.eval()
        with torch.no_grad():
            acc = float((torch.cat([J(norm(X[te[i:i + LOTE_AVAL]]))[0].argmax(1) for i in range(0, len(te), LOTE_AVAL)]) == y[te]).float().mean())
        print(f"juiz época {ep + 1}: acurácia {acc:.3f} em {N_TESTE} sprites separados", flush=True)
    RES.mkdir(exist_ok=True)
    torch.save(J.state_dict(), RES / "juiz.pt")
    (RES / "juiz.json").write_text(json.dumps({"acuracia_teste": acc, "epocas": epocas}), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=EPOCAS)
    treina_juiz(ap.parse_args().epocas)
