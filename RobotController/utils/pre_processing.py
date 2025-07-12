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

def get_rgb_tensor(camera_data: List[List[List[int]]]):
    """
    Converte dados da câmera em um tensor RGB normalizado.

    Esta função aplica uma série de transformações aos dados de imagem
    fornecidos pela câmera, resultando em um tensor RGB que pode ser usado
    em modelos de visão computacional.

    Parâmetros:
        camera_data: Dados da imagem capturada pela câmera, representados
        como um array.

    Retorna:
        Um tensor RGB normalizado, com tamanho 224x224, pronto para uso
        em redes neurais.
    """

    rgb_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    rgb_tensor = rgb_transform(Image.fromarray(camera_data))
    return rgb_tensor

def get_depth_tensor(depth_img: np.ndarray):
    """
    Converte uma imagem de profundidade em um tensor RGB normalizado.

    Esta fun o aplica uma s rie de transforma es  imagem de profundidade
    fornecida, resultando em um tensor RGB que pode ser usado
    em modelos de vis o computacional.

    Par metros:
        depth_img: Imagem de profundidade, representada como um array 2D.

    Retorna:
        Um tensor RGB normalizado, com tamanho 224x224, pronto para uso
        em redes neurais.
    """
    depth_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    depth_tensor = depth_transform(depth_img)
    return depth_tensor