from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoImageProcessor


class DinoRegressor(nn.Module): 
    def __init__(self, dino_hidden_size=768, top_k=4):
        super(DinoRegressor, self).__init__()
        self.dino_hidden_size = dino_hidden_size
        self.top_k = top_k
        
        self.imgProcessor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base", device_map="auto", output_attentions=True, attn_implementation="eager")
        
        self.regressor = nn.Sequential( # 1 camada oculta
            nn.Linear(self.dino_hidden_size*(self.top_k + 1)*2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 2) # dist e angle
        )
        
        # freeze dino
        for params in self.dino.parameters():
            params.require_grad = False
        
    def extract_features(self, images, top_k=4):
        type_img, image, _, _, _ = images.shape
        
        bundle_features = []
        for i in range(image):
            features = []
            for j in range(type_img):
                inputs = self.imgProcessor(images=images[j, i, :, :, :], return_tensors="pt", device='cuda' if torch.cuda.is_available() else 'cpu')
                outputs = self.dino(**inputs)
                
                embeddings = outputs.last_hidden_state[0, :, :].detach().numpy()
                cls_token = embeddings[0, :]
                    
                # pega os patches com maiores attention
                attentions = outputs.attentions[-1].detach().numpy()
                attention = attentions[0, :, 0, 1:].mean(axis=0)
                top_patches = np.argsort(attention)[-top_k:]
                top_patch_emb = embeddings[1:][top_patches].flatten()
                    
                features.append(np.concatenate([cls_token, top_patch_emb]))
            
            bundle_features.append(np.concatenate(features))
        
        return np.vstack(bundle_features)
        
        

    def forward(self, x):
        features = torch.from_numpy(self.extract_features(x))
        
        return self.regressor(features)
    
class RegressionDataset(Dataset):
    def __init__(self):
        super(RegressionDataset, self).__init__()
        data = np.load('RobotController/training_data.npz')
        
        self.rgb_data = torch.from_numpy(data['rgb'])
        depth = np.zeros((len(self.rgb_data), 4, 512, 3))
        depth[:, :, :, 0] = data['depth']
        self.depth_data = torch.from_numpy(depth)
        
        self.dist_data = torch.from_numpy(data['dist'])
        self.angle_data = torch.from_numpy(data['angle'])


    def __getitem__(self, index):
        return self.rgb_data[index], self.depth_data[index], self.dist_data[index], self.angle_data[index]

    def __len__(self):
        return len(self.rgb_data)