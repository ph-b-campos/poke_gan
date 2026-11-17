# Apresentação Deep Learning: GANs no Pokémon do D2L

DCGAN em vários modos (minimax, não saturante, colapso, WGAN com clipping, WGAN-GP e condicional) e um
gerador StyleGAN-like para o mix de domínio entre espécies (a parte grossa do Pikachu, a fina do Lugia).

Dados: <http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip>
(sha1 `c065c0e2593b8b161a2d7873e42418bf6a21106c`).

## Estrutura

| Pasta / arquivo | Papel |
|---|---|
| [config.py](config.py) | Configuração global: caminhos (`RES`, `FIG`, `CACHE`, `RAIZ_SPRITES`) e `N_ESP` |
| [data_handler.py](data_handler.py) | Cache 64x64 dos sprites (`prepara`), carga (`carrega`) e normalização (`norm`) |
| [utils.py](utils.py) | `dispositivo()`: cuda, mps ou cpu (a variável `DEVICE` força um deles) |
| [models/](models) | Só os modelos, um submódulo por modelo, cada um com sua pasta `config/` |
| [training/](training) | Funções de treino: perdas, regularizações, aumento e os laços de treino |
| [analise/](analise) | Avaliação: métricas (FID, IS, espécies, medidas da ponte) e figuras |
| [tests/](tests) | Autotestes em CPU |
| [original/](original) | Os scripts originais, antes da separação |

```
models/
  dcgan/      modelo.py (BNCond, Gerador, Discriminador)
              config/arquitetura.py  NZ, EMB, CAMADAS_G, NG, ND
              config/treino.py       CONFIG por modo, PASSOS, MINUTOS, LOTE, CADA_AVAL
  stylegan/   camadas.py (Linear, Conv, ModConv, Camada), modelo.py (Mapeamento, Gerador, ponte, Discriminador)
              config/arquitetura.py  Z_DIM, W_DIM, RESOLUCOES, CH, K_PALETA, SAIDA
              config/treino.py       CONFIG, MODO, HORAS
  juiz/       modelo.py (Juiz: classificador de espécies, o "Inception" do FID)
              config/arquitetura.py  CANAIS
              config/treino.py       EPOCAS, LR, LOTE, LOTE_AVAL, N_TESTE
training/     juiz.py, dcgan.py, stylegan.py
analise/      metricas.py, historico.py, plot_config.py, figuras_pokemon.py, figuras_estilo.py
tests/        teste_dcgan.py, teste_stylegan.py
```

Para mudar épocas, passos, taxa de aprendizado etc. de um modelo, edite `models/<modelo>/config/treino.py`.
Caminhos e constantes do dataset ficam no [config.py](config.py) da raiz.

## Uso

Tudo roda como módulo, a partir da raiz do repositório.

```bash
python -m tests.teste_dcgan
```

```bash
python -m tests.teste_stylegan
```

```bash
python -m data_handler --raiz .../pokemon
```

```bash
python -m training.juiz
```

```bash
python -m training.dcgan --modo wgp --passos 4000
```

```bash
python -m training.stylegan --modo cor --horas 8
```

```bash
python -m analise.figuras_pokemon curvas
```

```bash
python -m analise.figuras_estilo --modo cor
```

Ordem: autotestes, cache dos sprites, juiz (as GANs o usam nas avaliações), treinos e figuras.
Os resultados vão para `resultados/` e as figuras para `resultados/figs/`.

### Equivalência com os comandos antigos

| Antes (`original/`) | Agora |
|---|---|
| `python pokemon_gan.py teste` | `python -m tests.teste_dcgan` |
| `python pokemon_gan.py prepara --raiz ...` | `python -m data_handler --raiz ...` |
| `python pokemon_gan.py classificador` | `python -m training.juiz` |
| `python pokemon_gan.py treina --modo wgp --passos 4000` | `python -m training.dcgan --modo wgp --passos 4000` |
| `python estilo_gan.py teste` | `python -m tests.teste_stylegan` |
| `python estilo_gan.py treina --modo paleta --horas 8` | `python -m training.stylegan --modo paleta --horas 8` |
| `python estilo_gan.py figuras --modo paleta` | `python -m analise.figuras_estilo --modo paleta` |
| `python figuras_pokemon.py <acao>` | `python -m analise.figuras_pokemon <acao>` |

Os nomes dos arquivos em `resultados/` não mudaram, mas as pastas `resultados/` e `dados/` agora ficam na
raiz do projeto (antes ficavam ao lado dos scripts, em `original/`).
