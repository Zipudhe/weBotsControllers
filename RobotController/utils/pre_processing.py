from typing import List
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def range_img_to_img(range_img: list[float], width: int, height: int, max_range: float):
    """Converte uma imagem de profundidade em uma imagem 2D.

    A imagem de profundidade  uma lista de floats com valores de profundidade
    para cada p xel. A fun o retorna um array 2D com a profundidade
    convertida para uma cor.

    Par metros:
        range_img (list[float]): Imagem de profundidade a ser convertida.
        width (int): Largura da imagem.
        height (int): Altura da imagem.
        maxRange (float): Valor m ximo de profundidade.

    Retorna:
        Uma imagem de profundidade em RGB.
    """
    img = np.array(range_img).reshape(height, width)
    
    # normalizar
    normalized = np.nan_to_num(img / max_range, posinf=1, neginf=0)
    
    return normalized