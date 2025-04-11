import torch.nn as nn
from utils import show_graphs, train, process_data

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_size=32):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.block(x)
        x = self.fc2(x)
        return self.sigmoid(x)

train_loader, val_loader, input_dim = process_data()
model = SimpleModel(input_dim)

train_losses, val_losses, train_roc_auc, val_roc_auc = train(model, train_loader, val_loader)
show_graphs(train_losses, val_losses, train_roc_auc, val_roc_auc)

# подбираем оптимальное число эпох
model = SimpleModel(input_dim)
num_epochs = 20
train_losses, val_losses, train_roc_auc, val_roc_auc = train(model, train_loader, val_loader, num_epochs)
show_graphs(train_losses, val_losses, train_roc_auc, val_roc_auc, num_epochs)
