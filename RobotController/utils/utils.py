import math
import numpy as np

target_pos = np.array([1.80988, -1.73972])
obstaculos = [
    np.array([0.708464, -1.38034]),
    np.array([-1.24925, 0.460737]),
    np.array([-1.57095, -1.4525]),
    np.array([1.13961, 1.38342]),
    np.array([-0.867456, 0.199629]),
    np.array([0.495359, 0.103811]),
    np.array([-0.590771, -0.859123]),
    np.array([0.287378, 1.49634]),
    np.array([1.51884, 0.751898]),
    np.array([1.90315, -0.72552])
]


def generate_position():
    rng = np.random.default_rng()
    
    obstacles = obstaculos.copy()
    obstacles.append(target_pos)

    x, y = float('inf'), float('inf')
    while (x > 2 or x < -2) or (y > 2 or y < -2): # garante que o robo nao esteja fora da arena
        obs = rng.choice(obstacles)
        interval_choice_x = rng.choice([0, 1])
        interval_choice_y = rng.choice([0, 1])
    
        if interval_choice_x == 0: # garante que o robo nao esteja dentro do obstaculo
            x = obs[0] + rng.uniform(-1.75, -1.45) 
        else:
            x = obs[0] + rng.uniform(1.45, 1.75)
            
        if interval_choice_y == 0:
            y = obs[1] + rng.uniform(-1.75, -1.45)
        else:
            y = obs[1] + rng.uniform(1.45, 1.75)

    return x, y

def closest_visible_object(camera_pos: np.ndarray, camera_rot, obstacles, fov=1.5708, near=0.01, far=1.0):
    """
    Encontra o objeto mais próximo dentro do campo de visão da câmera.
    
    Args:
        camera_pos: Tupla (x, y) da posição da câmera.
        camera_rot: Ângulo de rotação da câmera em radianos.
        obstacles: Lista de tuplas [(x1, y1), (x2, y2), ...] com os obstáculos.
        fov: Campo de visão em radianos (padrão: 1.5708 = 90°).
        near: Distância mínima de visão (padrão: 0.01).
        far: Distância máxima de visão (padrão: 1.0).
    
    Returns:
        O ponto mais próximo (x, y) ou None se nenhum estiver visível.
    """
    # Constantes pré-calculadas
    half_fov = fov / 2.0
    cos_half_fov = math.cos(half_fov)
    cos_half_fov_sq = cos_half_fov ** 2
    near_sq = near ** 2
    far_sq = far ** 2
    
    # Vetor de direção da câmera (unitário)
    dir_x = math.cos(camera_rot)
    dir_y = math.sin(camera_rot)
    
    closest = None
    min_distance = float('inf')
    cam_x, cam_y = camera_pos + (np.array([dir_x, dir_y]) * 0.25)
    
    for obj_x, obj_y in obstacles:
        # Vetor do ponto à câmera
        dx = obj_x - cam_x
        dy = obj_y - cam_y
        dist_sq = dx**2 + dy**2
        
        # Verificar distância (near/far)
        if dist_sq < near_sq or dist_sq > far_sq:
            continue
        
        # Produto escalar com a direção da câmera
        dot = dx * dir_x + dy * dir_y
        
        # Se ponto está atrás da câmera (produto escalar negativo)
        if dot <= 0:
            continue
        
        # Verificar ângulo (usando quadrados para otimizar)
        if dot**2 < dist_sq * cos_half_fov_sq:
            continue
        
        # Calcular distância real e atualizar mais próximo
        distance = math.sqrt(dist_sq)
        if distance < min_distance:
            min_distance = distance
            closest = (obj_x, obj_y)
    
    if min_distance == float('inf'):
        closest = None
        min_distance = 1
    
    return closest, min_distance

def calculate_targets(x, y, theta):
        obstacles = obstaculos.copy()
        obstacles.append(target_pos)
        
        dist_to_obstacle = closest_visible_object(np.array([x, y]), theta, obstacles)[1]
        
        # angulo ao target
        vec_to_target = target_pos - np.array([x, y])
        global_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
        angle = global_angle - theta # angulo relativo ao robo
        
        # normalizar angulo
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        
        return dist_to_obstacle, angle