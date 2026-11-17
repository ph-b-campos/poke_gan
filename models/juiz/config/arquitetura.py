"""Arquitetura do juiz: 4 blocos conv-BN-ReLU-maxpool, média global e uma camada linear para as espécies."""

CANAIS = [3, 32, 64, 128, 256]      # o último é a dimensão das features usadas no FID
