from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from .DinoRegressor import DinoRegressor, RegressionDataset

if __name__ == '__main__':
    dataset = RegressionDataset()
    
    train_size = int(0.8*len(dataset.rgb_data)) # 80%
    val_size = len(dataset.rgb_data) - train_size # 20%
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_epochs = 50
    total_samples = len(dataset)
    
    
    class CosineLoss(nn.Module):
        def __init__(self):
            super(CosineLoss, self).__init__()

        def forward(self, pred, target):
            # pred e target são ângulos em radianos (apenas a coluna do ângulo)
            cos_sim = torch.cos(pred - target).mean()  # Similaridade cosseno
            return 1 - cos_sim  # Loss = 1 - cos(similaridade)
        
    class CombinedLoss(nn.Module):
        def __init__(self, w1, w2):
            super(CombinedLoss, self).__init__()
            self.w1 = w1
            self.w2 = w2
            self.cosine_loss = CosineLoss()

        def forward(self, pred, target):
            mse_dist = torch.mean((pred[:, 0] - target[:, 0]) ** 2)  # MSE para distância
            cos_angle = self.cosine_loss(pred[:, 1], target[:, 1])  # Cosine loss para ângulo
            return self.w1 * mse_dist + self.w2 * cos_angle
    
    model = DinoRegressor(top_k=2).to(device)
    criterion_dist = nn.MSELoss()
    criterion_angle = CosineLoss()
    criterion = CombinedLoss(w1=0.5, w2=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    losses_train = []
    losses_val = []
    
    # loop over the dataset multiple times
    for epoch in range(total_epochs):
        model.train()
        running_loss_dist = 0.0
        running_loss_angle = 0.0
        running_loss_total = 0.0
        
        print(f'Epoch {epoch+1}/{total_epochs}')
        
        # Treinamento
        for i, data in enumerate(trainloader, 0):
            rgb, depth, dist, angle = data
            depth_resized = torch.nn.functional.interpolate(depth.permute(0, 3, 1, 2), size=(64, 64), mode='bilinear', align_corners=False)
            depth_resized = depth_resized.permute(0, 2, 3, 1)
            
            inputs, labels = torch.stack((rgb, depth_resized)).to(device), torch.stack((dist, angle)).permute(1, 0).float().to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            
            loss_dist = criterion_dist(outputs[:, 0], labels[:, 0])  # Loss para dist
            loss_angle = criterion_angle(outputs[:, 1], labels[:, 1])  # Loss para angle
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
    
            running_loss_dist += loss_dist.item()
            running_loss_angle += loss_angle.item()
            running_loss_total += loss.item()
            
        val_running_loss_dist = 0.0
        val_running_loss_angle = 0.0
        val_running_loss_total = 0.0    
            
        # Validação
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                rgb, depth, dist, angle = data
                depth_resized = torch.nn.functional.interpolate(depth.permute(0, 3, 1, 2), size=(64, 64), mode='bilinear', align_corners=False)
                depth_resized = depth_resized.permute(0, 2, 3, 1)
                
                inputs, labels = torch.stack((rgb, depth_resized)).to(device), torch.stack((dist, angle)).permute(1, 0).float().to(device)
                
                outputs = model(inputs)
                
                loss_dist = criterion_dist(outputs[:, 0], labels[:, 0])  # Loss para dist
                loss_angle = criterion_angle(outputs[:, 1], labels[:, 1])  # Loss para angle
                loss = criterion(outputs, labels)
                
                val_running_loss_dist += loss_dist.item()
                val_running_loss_angle += loss_angle.item()
                val_running_loss_total += loss.item()
    
        print(f'Loss Dist: {running_loss_dist:.4f}')
        print(f'Loss Angle: {running_loss_angle:.4f}')
        print(f'Loss Total: {running_loss_total:.4f}')
        print("----------------------- VALIDATION -----------------------")
        print(f'Loss Dist: {val_running_loss_dist:.4f}')
        print(f'Loss Angle: {val_running_loss_angle:.4f}')
        print(f'Loss Total: {val_running_loss_total:.4f}')
        
        # salva as losses de cada epoch
        losses_train.append((running_loss_dist, running_loss_angle, running_loss_total))
        losses_val.append((val_running_loss_dist, val_running_loss_angle, val_running_loss_total))
        
    plt.figure(figsize=(15, 5))
    
    print('Finished Training')
    torch.save(model.parameters(), 'dino_regressor.pth')
    
    plt.plot(range(1, len(losses_train) + 1), losses_train[2], label='Training Total Loss', color='blue')
    plt.plot(range(1, len(losses_val) + 1), losses_val[2], label='Validation Total Loss', color='orange')
    plt.plot(range(1, len(losses_train) + 1), losses_train[0], label='DistToObject Training Loss', color='green')
    plt.plot(range(1, len(losses_val) + 1), losses_val[2], label='DistToObject Validation Loss', color='red')
    plt.plot(range(1, len(losses_train) + 1), losses_train[0], label='AngTarget Training Loss', color='purple')
    plt.plot(range(1, len(losses_val) + 1), losses_val[2], label='AngTarget Validation Loss', color='cyan')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
        
    plt.tight_layout()
    plt.show()