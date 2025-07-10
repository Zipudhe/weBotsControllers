import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils.pre_processing as pp


def write_matrix_data(matrix):
    fout = open("../matrix_data.txt", "a")
    fout.write(str(matrix) + "\n")
    fout.close()


def write_lidar_object_data(lidar_data):
    flat_lidar_points = [
        coord for point in lidar_data for coord in (point.x, point.y, point.z)
    ]

    with open("../lidar_data.csv", "a") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(flat_lidar_points)
        
def show_lidar_img(range_img: list[float], width: int, height: int, max_range: float):
    """
    Função para visualizar imagem de profundidade LiDAR.

    A imagem de profundidade  uma lista de floats com valores de profundidade
    para cada pixel. A função retorna uma imagem de profundidade em RGB.

    Parâmetros:
        range_img (list[float]): Imagem de profundidade a ser visualizada.
        width (int): Largura da imagem.
        height (int): Altura da imagem.
        maxRange (float): Valor máximo de profundidade.

    Retorna:
        Uma imagem de profundidade em RGB.
    """
    depth_img = pp.range_img_to_img(range_img, width, height, max_range)
    visualize_depth_map(depth_img, max_range)
        

def visualize_depth_map(depth_map: np.ndarray, max_range: float):
    """
    Função para visualizar mapas de profundidade.

    O mapa de profundidade  um array 2D com valores de profundidade
    para cada píxel. A função retorna um array 3D com a profundidade
    convertida para uma cor.

    Parâmetros:
        depth_map (np.ndarray): Mapa de profundidade a ser visualizado.

    Retorna:
        Uma imagem de profundidade em RGB.
    """
    plt.figure(figsize=(10, 8))
    
    # Mostrar imagem com colormap
    plt.imshow(depth_map, 
               cmap='jet_r', 
               vmin=0, 
               vmax=max_range)
    
    # Adicionar barra de cores com valores em metros
    cbar = plt.colorbar()
    cbar.set_label('Distância (metros)')
    
    plt.title('Mapa de Profundidade LiDAR')
    plt.axis('off')
    plt.show()