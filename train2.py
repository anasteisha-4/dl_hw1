import torch.nn as nn
from utils import show_graphs, train, process_data

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        return self.block(x)

class BiggerModel(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_blocks=3):
        super(BiggerModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc2(x)
        return self.sigmoid(x)

train_loader, val_loader, input_dim = process_data()

model = BiggerModel(input_dim)
num_epochs = 17
train_losses, val_losses, train_roc_auc, val_roc_auc = train(model, train_loader, val_loader, num_epochs)
show_graphs(train_losses, val_losses, train_roc_auc, val_roc_auc, num_epochs)