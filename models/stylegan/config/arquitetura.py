"""Arquitetura do gerador StyleGAN-like (StyleGAN2-ADA, Karras et al. 2020, adaptada a 64x64)."""

Z_DIM = W_DIM = 256
RESOLUCOES = (4, 8, 16, 32, 64)
CH = {4: 256, 8: 256, 16: 256, 32: 128, 64: 64}     # canais por resolução
K_PALETA = 12                                        # slots de cor, fora o fundo; sprites têm ~10 a 16 cores

# Modo da linha de comando -> saída do gerador
SAIDA = {"cor": "cor", "estilo": "rgb", "paleta": "paleta"}
