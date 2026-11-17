"""Mix de domínio entre espécies, StyleGAN-like e adaptado a sprites: a parte grossa do Pikachu, a fina do Lugia.

Por que não basta o DCGAN condicional (models/dcgan, modo cwgp): ele aprende cada espécie (FID 44,5), mas
decora cada uma como um ponto isolado, e misturar camadas entre espécies dá ruído. O dataset tem 40.597 imagens
e só ~3.900 sprites distintos (mediana de 3 por espécie). Daqui vêm as peças da StyleGAN (Karras et al. 2019; 2020):
    mapeamento  (z, espécie) -> w por uma MLP
    constante   a imagem nasce de um tensor 4x4 aprendido; (z, espécie) só entram como estilo
    modulação   cada convolução é escalada pelo estilo da sua resolução (modulação e demodulação)
    mistura     no treino, metade do lote usa o w de uma espécie nas posições grossas e o de outra nas finas;
                o discriminador julga essas só por "parece Pokémon?", sem espécie
E as adaptações ao domínio (sprite = regiões de cor chapada, contorno escuro, fundo branco):
    paleta      (modo "paleta") a saída não pinta RGB livre: cada pixel escolhe um de K slots de cor, e as K
                cores vêm de um estilo próprio, a posição mais fina. Pikachugia = regiões do Pikachu com a
                paleta do Lugia. Um slot fixo é o fundo branco.
    aumento     espelho horizontal nos reais; DiffAugment (Zhao et al. 2020) de cor, translação (borda branca,
                como o fundo) e recorte, iguais em reais e falsos
    classe      cabeça classificadora no D (AC-GAN, Odena et al. 2017), treinada só nos sprites reais; o G é
                cobrado para que suas amostras puras sejam da espécie pedida. Sem ela (rodada cor_v1, 17/11) a
                projeção sozinha não ensinou a espécie: o juiz acertava 0,7% das amostras em 297 kimg
    espécies    só as gerações 1 a 3 (Pokédex 1 a 386), que contêm todos os pares das figuras
    transição   poucos sprites por espécie fazem a StyleGAN2 condicional colapsar (Shahbazi et al. 2022, ICLR);
                a espécie entra aos poucos: peso lam de 0 a 1 no mapeamento e na projeção do D
O modo "estilo" é a mesma rede com saída RGB por resolução (StyleGAN2 "skip"): o teste de comparação.

O aumento, a mistura e o laço de treino ficam em training/stylegan.py; as medidas da ponte em analise/metricas.py.
"""
from .camadas import SQ2, Camada, Conv, Linear, ModConv, lrelu, norm2
from .modelo import BlocoD, Discriminador, Gerador, Mapeamento, desvio_lote, ponte

__all__ = [
    "SQ2",
    "lrelu",
    "norm2",
    "Linear",
    "Conv",
    "ModConv",
    "Camada",
    "Mapeamento",
    "Gerador",
    "ponte",
    "desvio_lote",
    "BlocoD",
    "Discriminador",
]
