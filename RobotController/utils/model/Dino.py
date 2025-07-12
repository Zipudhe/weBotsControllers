import torch
import torch.nn as nn
from transformers import  AutoModel


class DINORegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Backbone DINOv2 para imagens
        self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        
        # Congelar DINOv2
        for param in self.dino.parameters():
            param.requires_grad = False
            
        # Camadas de regressão
        self.regressor = nn.Sequential(
            nn.Linear(1536, 512),  # 768*2 = 1536 features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # dist + angle
        )
    
    def forward(self, rgb, depth):
        # Processar imagem RGB
        rgb_features = self.dino(pixel_values=rgb).last_hidden_state
        rgb_features = rgb_features.mean(dim=1)
        
        # Processar depth image
        depth_features = self.dino(pixel_values=depth).last_hidden_state
        depth_features = depth_features.mean(dim=1)
        
        # Concatenar e regredir
        combined = torch.cat([rgb_features, depth_features], dim=1)
        return self.regressor(combined)

# Inicialização
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DINORegressor().to(device)