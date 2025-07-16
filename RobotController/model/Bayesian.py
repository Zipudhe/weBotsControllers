from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

model = DiscreteBayesianNetwork([('ObstacleDetected', 'TargetVisible'),
                                ('ObstacleDetected', 'Success'),
                                ('ObstacleDetected', 'Action'),
                                ('Direction', 'TargetVisible'),
                                ('Direction', 'Success'),
                                ('Direction', 'Action'),
                                ('TargetVisible', 'Success'),
                                ('Success', 'Action')])


def sigmoid(x, a=1.0):
    """Converte distância em probabilidade para ObstacleDetected"""
    return 1 / (1 + np.exp(-a * x))

def gaussian_angle(x, sigma=30.0):
    """Converte ângulo em probabilidade para Direction"""
    angle_norm = x % 2*np.pi
    if angle_norm > np.pi:
        angle_norm -= 2*np.pi
    
    # Distribuição gaussiana para frente (centrada em 0°)
    p_meio = np.exp(-0.5 * (angle_norm / sigma) ** 2)
    
    # Distribuição para esquerda (centrada em -90°)
    angle_left = angle_norm + 90
    p_esquerda = np.exp(-0.5 * (angle_left / sigma) ** 2)
    
    # Distribuição para direita (centrada em +90°)
    angle_right = angle_norm - 90
    p_direita = np.exp(-0.5 * (angle_right / sigma) ** 2)
    
    # Normalizar para soma = 1
    total = p_esquerda + p_meio + p_direita
    if total < 1e-10:  # Evitar divisão por zero
        return 1/3, 1/3, 1/3
    
    p_esquerda /= total
    p_meio /= total
    p_direita /= total
    
    return p_esquerda, p_meio, p_direita


def bayesian_inference(dist_output, angle_output):
    """
    Processa as saídas da RNA e executa a inferência bayesiana
    
    Args:
        dist_output: Distância estimada para o obstáculo (float)
        angle_output: Ângulo estimado para o alvo (float)
    
    Returns:
        str: Ação recomendada ('go', 'turn_left', 'turn_right', 'stop')
    """
    # Converter saídas contínuas em probabilidades
    p_obstacle = sigmoid(1/dist_output)  # Probabilidade de haver obstáculo
    p_esquerda, p_frente, p_direita = gaussian_angle(angle_output)  # Probabilidade de alvo visível
    
    
    cpd_obstacle = TabularCPD(
        variable='ObstacleDetected',
        variable_card=2,
        values=[[p_obstacle], [1 - p_obstacle]],
        state_names={'ObstacleDetected': ['sim', 'não']}
    )

    cpd_direction = TabularCPD(
        variable='Direction',
        variable_card=3,
        values=[[p_esquerda], [p_frente], [p_direita]],
        state_names={'Direction': ['Esquerda', 'Frente', 'Direita']}
    )

    cpd_target = TabularCPD(
        variable='TargetVisible',
        variable_card=2,
        values=[
            # sim      não      sim    não    sim     não
            [0.1,      0.3,     0.8,   0.7,   0.1,    0.3], # visible yes
            [0.9,      0.7,     0.2,   0.3,   0.9,    0.7]  # visible no
            # Esquerda Esquerda Frente Frente Direita Direita
        ],
        evidence=['ObstacleDetected', 'Direction'],
        evidence_card=[2, 3],
        state_names={
            'TargetVisible': ['sim', 'não'],
            'ObstacleDetected': ['sim', 'não'],
            'Direction': ['Esquerda', 'Frente', 'Direita']
        }
    )

    cpd_success = TabularCPD(
        variable='Success',
        variable_card=2,
        values=[
            # sim      não      sim    não    sim     não     sim      não      sim    não    sim     não
            [0.05,     0.05,    (p_obstacle * 0.95),  0.05,  0.05,   0.05,   0.05,    0.05,   0.05,  0.05,  0.05,   0.05], # success yes
            [0.95,     0.95,    (1 - p_obstacle * 0.95),  0.95,  0.95,   0.95,   0.95,    0.95,   0.95,  0.95,  0.95,   0.95]  # success no
            # Esquerda Esquerda Frente Frente Direita Direita Esquerda Esquerda Frente Frente Direita Direita
            # sim      sim      sim    sim    sim     sim     não      não      não    não    não     não
        ],
        evidence=['ObstacleDetected', 'Direction', 'TargetVisible'],
        evidence_card=[2, 3, 2],
        state_names={
            'Success': ['sim', 'não'],
            'ObstacleDetected': ['sim', 'não'],
            'Direction': ['Esquerda', 'Frente', 'Direita'],
            'TargetVisible': ['sim', 'não']
        }
    )

    cpd_action = TabularCPD(
        variable='Action',
        variable_card=4,
        values=[
            # sim      não      sim    não    sim     não     sim      não      sim    não    sim     não
            [0.15,     0.25,    (1 - p_obstacle * 0.95),  0.90,  0.15,   0.25,   0.15,     0.25,    0.05,  0.90,  0.15,   0.25], # action seguir
            [0.65,     0.65,    0.00,  0.05,  0.20,   0.10,   0.65,     0.65,    0.45,  0.05,  0.20,   0.10], # action virar esquerda
            [0.20,     0.10,    0.00,  0.05,  0.65,   0.65,   0.20,     0.10,    0.45,  0.05,  0.65,   0.65], # action virar direita
            [0.00,     0.00,    (p_obstacle * 0.95),  0.00,  0.00,   0.00,   0.00,     0.00,    0.05,  0.00,  0.00,   0.00]  # action parar
            # Esquerda Esquerda Frente Frente Direita Direita Esquerda Esquerda Frente Frente Direita Direita
            # sim      sim      sim    sim    sim     sim     não      não      não    não    não     não
        ],
        evidence=['ObstacleDetected', 'Direction', 'Success'],
        evidence_card=[2, 3, 2],
        state_names={
            'Action': ['seguir', 'virar esquerda', 'virar direita', 'parar'],
            'ObstacleDetected': ['sim', 'não'],
            'Direction': ['Esquerda', 'Frente', 'Direita'],
            'Success': ['sim', 'não']
        }
    )
    
    model.add_cpds(cpd_obstacle, cpd_direction, cpd_target, cpd_success, cpd_action)
    model.check_model()

    infer = VariableElimination(model)
    
    obstacle_detected = infer.query(
        variables=['ObstacleDetected'],
        show_progress=False
    )
    
    direction = infer.query(
        variables=['Direction'],
        show_progress=False
    )
    
    target_visible = infer.query(
        variables=['TargetVisible'],
        evidence={
            'ObstacleDetected': ['sim', 'não'][obstacle_detected.values.argmax()],
            'Direction': ['Esquerda', 'Frente', 'Direita'][direction.values.argmax()],
        },
        show_progress=False
    )
    
    success = infer.query(
        variables=['Success'],
        evidence={
            'ObstacleDetected': ['sim', 'não'][obstacle_detected.values.argmax()],
            'Direction': ['Esquerda', 'Frente', 'Direita'][direction.values.argmax()],
            'TargetVisible': ['sim', 'não'][target_visible.values.argmax()]
        },
        show_progress=False
    )
    
    # Realizar inferência
    result = infer.query(
        variables=['Action'],
        evidence={
            'ObstacleDetected': ['sim', 'não'][obstacle_detected.values.argmax()],
            'Direction': ['Esquerda', 'Frente', 'Direita'][direction.values.argmax()],
            'Success': ['sim', 'não'][success.values.argmax()]
        },
        show_progress=False
    )
    
    # Selecionar ação com maior probabilidade
    action = result.values.argmax()
    return ['go', 'turn_left', 'turn_right', 'stop'][action]

if __name__ == "__main__":
    # Valores hipotéticos da RNA (distância em metros, ângulo em graus)
    dist_output = 0.8  # Distância pequena -> alta probabilidade de obstáculo
    angle_output = 15  # Ângulo pequeno -> alta probabilidade de alvo visível
    
    ação = bayesian_inference(dist_output, angle_output)
    print(f"Ação recomendada: {ação}")  # Provavelmente "go"