import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.Dino import model, device


class SensorDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# config de treino
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# DataLoader
dataset = SensorDataset(features, targets) # WIP
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

#Loop de treino
for epoch in range(50):
    model.train()
    total_loss = 0
    
    for inputs in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    
    print(f"Epoch [{epoch+1}/{50}], Loss: {avg_loss:.4f}")