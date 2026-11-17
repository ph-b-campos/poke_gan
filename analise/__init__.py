"""Avaliação: métricas e figuras.

    metricas.py         FID, IS, espécies (espaço do juiz) e medidas da ponte
    historico.py        leitura dos históricos de treino (resultados/<modo>.json)
    plot_config.py      estilo, cores e nomes das figuras
    figuras_pokemon.py  figuras da DCGAN     (python -m analise.figuras_pokemon <acao>)
    figuras_estilo.py   figuras da StyleGAN  (python -m analise.figuras_estilo --modo <modo>)

Este __init__ não importa nada de propósito: o treino usa analise.metricas sem carregar o matplotlib.
"""
