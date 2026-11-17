"""Treino: um módulo por modelo, executados a partir da raiz com `python -m training.<modulo>`.

    juiz.py      classificador de espécies (rodar primeiro: as GANs o usam nas avaliações)
    dcgan.py     DCGAN, todos os modos (perdas, penalidade de gradiente, mixing regularization)
    stylegan.py  gerador StyleGAN-like (mistura entre espécies, DiffAugment, regularização R1)

As configurações de cada treino ficam em models/<modelo>/config/.
"""
