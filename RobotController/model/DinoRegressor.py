from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoImageProcessor


class DinoRegressor(nn.Module): 
    def __init__(self, dino_hidden_size=768, top_k=2):
        super(DinoRegressor, self).__init__()
        self.dino_hidden_size = dino_hidden_size
        self.top_k = top_k
        
        self.imgProcessor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base", device_map="auto", output_attentions=True, attn_implementation="eager")
        
        self.regressor = nn.Sequential( # 4 camadas ocultas
            nn.Linear(self.dino_hidden_size*(self.top_k + 1)*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2) # dist e angle
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
                
                embeddings = outputs.last_hidden_state[0, :, :]
                cls_token = embeddings[0, :]
                    
                # pega os patches com maiores attention
                attentions = outputs.attentions[-1]
                attention = attentions[0, :, 0, 1:].mean(axis=0)
                top_patches = torch.argsort(attention)[-top_k:]
                top_patch_emb = embeddings[1:][top_patches].flatten()
                    
                features.append(torch.cat([cls_token, top_patch_emb]))
            
            bundle_features.append(torch.cat(features))
        
        bundle_features_cpu = [tensor.cpu().detach().numpy() for tensor in bundle_features]
        return np.vstack(bundle_features_cpu)
        
        

    def forward(self, x):
        features = torch.from_numpy(self.extract_features(x, self.top_k)).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        
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