"""
Configuração global do projeto: caminhos e constantes do dataset, comuns a todos os modelos.

As configurações de cada modelo (arquitetura, hiperparâmetros, duração do treino) ficam em
models/<modelo>/config/, não aqui.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos (todos relativos à raiz do projeto)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RES = PROJECT_ROOT / "resultados"          # checkpoints, históricos (.json), grades de amostras e o juiz
FIG = RES / "figs"                         # figuras de avaliação
DADOS_DIR = PROJECT_ROOT / "dados"         # fora do git: criada por `python -m data_handler --baixar`

# Cache 64x64 dos sprites, gerado por `python -m data_handler`. A variável de ambiente POKEMON_CACHE
# aponta para outro arquivo, se preciso.
CACHE = Path(os.environ.get("POKEMON_CACHE") or DADOS_DIR / "pokemon64.npz")

# Sprites originais do D2L: uma subpasta por número da Pokédex, usada só para montar o cache.
URL_SPRITES = "http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip"
SHA1_SPRITES = "c065c0e2593b8b161a2d7873e42418bf6a21106c"
RAIZ_SPRITES = DADOS_DIR / "pokemon"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
N_ESP = 721                                # espécies na Pokédex do D2L
