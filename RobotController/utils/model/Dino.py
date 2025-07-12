import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_small_patch14_reg4_dinov2

class DINOv2Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_small_patch14_reg4_dinov2(pretrained=True)
        
        # congela pesos
        for param in self.backbone.parameters():
            param.requires_grad = False

        # cria regressor
        self.regressor = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, rgb, depth):
        # Extrai features
        rgb_features = self.backbone(rgb)
        depth_features = self.backbone(depth)
        
        fused = torch.cat([rgb_features, depth_features], dim=1)
        
        #faz a regressao
        output = self.regressor(fused)
        return output[:, 0], output[:, 1] # dist, angle
        