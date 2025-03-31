import torch.nn as nn
from utils import show_graphs, train, process_data

class ResidualBlockWithDropout(nn.Module):
    def __init__(self, hidden_size, dropout_p):
        super(ResidualBlockWithDropout, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.BatchNorm1d(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
    
    def forward(self, x):
        return x + self.block(x)

class ModelWithDropout(nn.Module):
    def __init__(self, input_dim, dropout_p, hidden_size=128, num_blocks=3):
        super(ModelWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlockWithDropout(hidden_size, dropout_p) for _ in range(num_blocks)])
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc2(x)
        return self.sigmoid(x)

train_loader, val_loader, input_dim = process_data()

for lr in [0.01, 0.05, 0.1]:
    for weight_decay in [0.1, 0.01, 0.001]:
        print(f'lr = {lr}, weight_decay = {weight_decay}')
        num_epochs = 15
        model = ModelWithDropout(input_dim, 0.1)

        train_losses, val_losses, train_roc_auc, val_roc_auc = train(model, train_loader, val_loader, num_epochs, lr=lr, weight_decay=weight_decay)
        show_graphs(train_losses, val_losses, train_roc_auc, val_roc_auc, num_epochs)