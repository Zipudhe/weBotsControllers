import math
import numpy as np
from controller import LidarPoint


def point_cloud_to_image(point_cloud: list[LidarPoint], resolution = 64): 
    """
    Converte um point cloud em uma imagem de profundidade.

    Args:
        point_cloud (list[LidarPoint]): O point cloud a ser convertido.
        resolution (int, optional): A resoluo da imagem de profundidade. Defaults to 64.

    Returns:
        depth_img (np.ndarray): A imagem de profundidade.
    """

    depth_img = np.zeros((resolution, resolution))
     
    for point in point_cloud:
        # normalizar as coordenadas
        if math.isinf(point.x):
            if point.x < 0:
                x = -1
            else:
                x = 1
        else:
            x = point.x
            
        if math.isinf(point.y):
            if point.y < 0:
                y = -1
            else:
                y = 1
        else:
            y = point.y
            
        if math.isinf(point.z):
            if point.z < 0:
                z = -1
            else:
                z = 1
        else:
            z = point.z
        
        
        # fazer "flatten" nas coordenadas
        u = int((np.arctan2(y, x) + np.pi) / (2 * np.pi) * resolution)
        v = int((np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2)) * resolution) / np.pi + resolution / 2)  
        if 0 <= u < resolution and 0 <= v < resolution:
            distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            depth_img[v, u] = distance
    return depth_img