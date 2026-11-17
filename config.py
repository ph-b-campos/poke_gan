"""
Configuração global do projeto: caminhos e constantes do dataset, comuns a todos os modelos.

As configurações de cada modelo (arquitetura, hiperparâmetros, duração do treino) ficam em
models/<modelo>/config/, não aqui.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RES = PROJECT_ROOT / "resultados"          # checkpoints, históricos (.json), grades de amostras e o juiz
FIG = RES / "figs"                         # figuras de avaliação
DADOS_DIR = PROJECT_ROOT / "dados"

# Cache 64x64 dos sprites (gerado por `python -m data_handler`). Ordem de prioridade: variável de
# ambiente POKEMON_CACHE; dados/pokemon64.npz, se a pasta dados/ existir; senão o caminho da máquina Windows.
CACHE_WIN = Path(r"D:\Arquivos\Documents\Mestrado_PEB\dados_externos\pokemon_d2l\pokemon64.npz")
CACHE = Path(os.environ.get("POKEMON_CACHE") or (DADOS_DIR / "pokemon64.npz" if DADOS_DIR.is_dir() else CACHE_WIN))

# Pasta com os sprites originais (uma subpasta por número da Pokédex), usada só para montar o cache.
# Dados: http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip (sha1 c065c0e2593b8b161a2d7873e42418bf6a21106c).
RAIZ_SPRITES = r"D:\Arquivos\Documents\Mestrado_PEB\dados_externos\pokemon_d2l\pokemon"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
N_ESP = 721                                # espécies na Pokédex do D2L
