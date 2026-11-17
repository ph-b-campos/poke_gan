"""
Treino da DCGAN: uma configuração por modo, fixada antes de rodar (Radford 2015; Arjovsky 2017; Gulrajani 2017).

    minimax   GAN original, perda saturante log(1 - D(G(z)))        -> esperado: vanishing gradient
    ns        GAN não saturante, -log D(G(z)) (o padrão da prática)  -> ver se colapsa
    colapso   a mesma, com D encolhido (Metz et al. 2017)            -> esperado: colapso de modos
    wclip     WGAN com clipping de pesos (Arjovsky et al. 2017)
    wgp       WGAN-GP, penalidade de gradiente (Gulrajani et al. 2017)
    cwgp      WGAN-GP condicional na espécie, BatchNorm condicional por camada: a "ponte"
    cwgp_mix  o mesmo + mixing regularization (Karras et al. 2019)

O dicionário do modo é gravado em resultados/<modo>.json junto com o histórico.
"""

CONFIG = {
    "minimax": dict(perda="minimax", opt="adam", lr=2e-4, betas=(0.5, 0.999), n_critico=1),
    "ns": dict(perda="ns", opt="adam", lr=2e-4, betas=(0.5, 0.999), n_critico=1),
    "colapso": dict(perda="ns", opt="adam", lr=2e-4, betas=(0.5, 0.999), n_critico=1, nd=32),   # D com ~1/5 dos parâmetros de G (Metz et al. 2017)
    "wclip": dict(perda="wgan", opt="rmsprop", lr=5e-5, clip=0.01, n_critico=5),
    "wgp": dict(perda="wgan", opt="adam", lr=1e-4, betas=(0.0, 0.9), n_critico=5, gp=10.0),
    "cwgp": dict(perda="wgan", opt="adam", lr=1e-4, betas=(0.0, 0.9), n_critico=5, gp=10.0, condicional=True, especies=251),   # gerações 1 e 2: Pikachu #25 e Lugia #249
    # o mesmo + mixing regularization (Karras et al. 2019): metade do lote já sai misturada no treino (0,5 e não 0,9: G é condicional)
    "cwgp_mix": dict(perda="wgan", opt="adam", lr=1e-4, betas=(0.0, 0.9), n_critico=5, gp=10.0, condicional=True, especies=251, mistura=0.5),
}

# ---------------------------------------------------------------------------
# Duração e avaliação (iguais para todos os modos)
# ---------------------------------------------------------------------------
PASSOS = 4000           # orçamento em passos do gerador, o mesmo para todos os modos (critério de Gulrajani et al. 2017)
MINUTOS = 240           # limite de tempo de parede
LOTE = 128
CADA_AVAL = 250         # avalia (FID, IS, espécies) e salva a cada tantos passos do gerador
