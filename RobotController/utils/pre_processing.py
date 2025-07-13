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

# Transformações para RGB
rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transformações para Depth
depth_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Converter para 3 canais
    transforms.Normalize([0.5], [0.5])  # Normalização simples para depth
])

def preprocess_data(rgb, depth):
    """Pré-processa um lote de imagens RGB e Depth"""
    processed_rgb = torch.stack([rgb_transform(img) for img in rgb])
    processed_depth = torch.stack([depth_transform(img) for img in depth])
    return processed_rgb, processed_depth
