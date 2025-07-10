import numpy as np
from controller import LidarPoint


def point_cloud_to_image(point_cloud: list[LidarPoint], resolution = 64): 
    depth_img = np.zeros((resolution, resolution))
    
    for point in point_cloud:
        x, y, z = point.x, point.y, point.z
        
        # fazer "flatten" nas coordenadas
        u = int((np.arctan2(y, x) + np.pi) / (2 * np.pi) * resolution)
        v = int((np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2)) * resolution) / np.pi + resolution / 2)  
        if 0 <= u < resolution and 0 <= v < resolution:
            distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            depth_img[v, u] = distance
    return depth_img