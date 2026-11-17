"""
Treino do gerador StyleGAN-like: StyleGAN2-ADA (Karras et al. 2020) adaptada a 64x64 e a uma GPU de 6 GB.

O dicionário CONFIG é gravado em resultados/<modo>.json junto com o histórico.
"""

CONFIG = {
    # gamma 0,1 e aumento de cor: com 0,05 e sem cor o D dominou em 12 kimg (teste de saúde de 17/11); o DiffAugment
    # aplica a cor igual em reais e falsos, sem vazar para o gerador (Zhao et al. 2020)
    "lote": 32, "lr": 0.0025, "betas": (0.0, 0.99), "gamma": 0.1, "r1_cada": 16, "mapa": 4,
    "mistura": 0.5, "ema_kimg": 10, "aumento": "cor,translacao,recorte", "aval_min": 15,
    "transicao_kimg": (128, 256),       # Shahbazi et al.: 2k a 4k iterações de lote 64; aqui em mil imagens
    "especies": 386,                    # gerações 1 a 3
    "classe": 1.0,                      # peso da entropia cruzada da cabeça classificadora
}

# ---------------------------------------------------------------------------
# Duração
# ---------------------------------------------------------------------------
MODO = "cor"            # modo padrão da linha de comando (ver SAIDA em arquitetura.py)
HORAS = 8               # cada chamada treina por tantas horas e retoma do checkpoint, se existir
