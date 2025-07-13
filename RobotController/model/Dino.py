import numpy as np
import torch
import torch.nn as nn
from transformers import  AutoImageProcessor, AutoModel

from ..utils.pre_processing import preprocess_data


data = np.load('RobotController/training_data.npz')

rgb_data = data['rgb']
depth_data = data['depth']
dist_data = data['dist']
angle_data = data['angle']

targets = np.column_stack((dist_data, angle_data))

# Aplicar pré-processamento a todos os dados
rgb_tensor, depth_tensor = preprocess_data(rgb_data.astype(np.uint8), depth_data.astype(np.uint8))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True)
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
model.eval()

def extract_features(rgb, depth):
    """Extrai features DINOv2 para um lote de imagens"""
    with torch.no_grad():
        # Processar RGB
        rgb_inputs = processor(images=rgb, return_tensors="pt", do_convert_rgb=False, do_resize=False, do_normalize=False).to(device)
        rgb_features = model(**rgb_inputs).last_hidden_state.mean(dim=1)
        
        # Processar Depth
        depth_inputs = processor(images=depth, return_tensors="pt", do_convert_rgb=False, do_resize=False, do_normalize=False).to(device)
        depth_features = model(**depth_inputs).last_hidden_state.mean(dim=1)
        
        return torch.cat([rgb_features, depth_features], dim=1).cpu()

# Extrair features em lotes (para evitar estouro de memória)
batch_size = 8
features_list = []

for i in range(0, len(rgb_tensor), batch_size):
    rgb_batch = rgb_tensor[i:i+batch_size]
    depth_batch = depth_tensor[i:i+batch_size]
    
    batch_features = extract_features(rgb_batch, depth_batch)
    
    features_list.append(batch_features)
    
# Liberar memória após extração de features
del rgb_data, depth_data, rgb_tensor, depth_tensor
torch.cuda.empty_cache()

# Concatenar todas as features
all_features = torch.cat(features_list, dim=0)

# Salvar para uso futuro
torch.save({
    'features': all_features,
    'targets': targets
}, 'dino_features.pt')