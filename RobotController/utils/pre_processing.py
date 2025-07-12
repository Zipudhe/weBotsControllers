from typing import List
import numpy as np
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

# Transformações específicas para DINOv2
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def prepare_inputs(rgb, depth, device):
    # Converter depth para 3 canais
    depth_rgb = depth.repeat(3,1,1) if depth.shape[0]==1 else depth
    
    return {
        'rgb': preprocess(rgb).unsqueeze(0).to(device),
        'depth': preprocess(depth_rgb).unsqueeze(0).to(device)
    }
