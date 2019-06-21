import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, d_in, d_intr):
        super().__init__()

        self.fc1 = nn.Linear(d_in, d_intr)
        self.fc2 = nn.Linear(d_intr, d_in)

    def forward(self, x):
        x = F.relu(self.encode(x))
        x = self.fc2(x)

        return x

    def encode(self, x):
        # flatten inputs to vector
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        dim = 1
        for s in size:
            dim *= s

        return dim
