import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision
from torchvision import transforms

class MLP(nn.Module):
    def __init__(self, n_in, n_intr, n_out):
        super().__init__()

        self.n_in = n_in
        self.n_intr = n_intr
        self.n_out = n_out

        self.fc1 = nn.Linear(n_in, n_intr)
        self.fc2 = nn.Linear(n_intr, n_out)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.softmax(self.fc2(y))

        return y
