import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from .Dino import model, device, all_features, targets


class SensorDataset(Dataset):
    def __init__(self, features, targets):
        """
        Args:
            features: Tensor de features extraídas (N, feature_dim)
            targets: Array de targets (N, 2)
        """
        self.features = features
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DINORegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # dist e angle
        )

    def forward(self, x):
        return self.regressor(x)


# Criar dataset final
dataset = SensorDataset(all_features, targets)

# Dividir em treino e validação (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Inicializar modelo
input_dim = all_features.shape[1]
dino_model = DINORegressor(input_dim).to(device)

# Configurar treinamento
criterion = nn.MSELoss()
optimizer = optim.AdamW(dino_model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=3, factor=0.5
)


def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detectado em {name}")
        return True
    return False


for features, targets in train_loader:
    if check_nan(features, "Features") or check_nan(targets, "Targets"):
        continue  # Pula batch problemático

# Loop de treinamento
for epoch in range(50):
    dino_model.train()
    train_loss = 0.0

    # Treino
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

        train_loss += loss.item()

    # Validação
    dino_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            val_loss += criterion(outputs, targets).item()

    # Estatísticas
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}/{50}")
    print(
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

