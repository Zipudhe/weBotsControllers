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

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    total_epochs = 50
    total_samples = len(dataset)
    
    model = DinoRegressor(top_k=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    
    
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
            
            loss_dist = criterion(outputs[:, 0], labels[:, 0])  # Loss para dist
            loss_angle = criterion(outputs[:, 1], labels[:, 1])  # Loss para angle
            loss = loss_dist + loss_angle
            
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
                
                loss_dist = criterion(outputs[:, 0], labels[:, 0])  # Loss para dist
                loss_angle = criterion(outputs[:, 1], labels[:, 1])  # Loss para angle
                loss = loss_dist + loss_angle
                
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
    
    print('Finished Training')
    torch.save(model.parameters(), 'dino_regressor.pth')