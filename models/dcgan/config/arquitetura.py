"""Arquitetura da DCGAN."""

NZ = 100                # dimensão do ruído z
EMB = 128               # dimensão do embedding da espécie (modo condicional)
CAMADAS_G = 4           # camadas com normalização no gerador: 4x4, 8x8, 16x16, 32x32
NG = 64                 # canais-base do gerador
ND = 64                 # canais-base do discriminador/crítico (o modo "colapso" sobrescreve com "nd")
