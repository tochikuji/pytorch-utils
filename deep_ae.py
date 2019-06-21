import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepAE(nn.Module):
    def __init__(self, *dims):
        super().__init__()

        self.depth = len(dims)
        self.units = dims

        for i in range(self.depth - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            setattr(self, f'encoder{i + 1}', layer)

        for i in range(self.depth - 1):
            layer = nn.Linear(dims[self.depth - i - 1], dims[self.depth - i - 2])
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            setattr(self, f'decoder{i + 1}', layer)

    def forward(self, x):
        x = F.relu(self.encode(x))

        decode_layers = [f'decoder{i + 1}' for i in range(self.depth - 1)]

        for layer in decode_layers[:-1]:
            x = getattr(self, layer)(x)
            x = F.relu(x)

        x = getattr(self, decode_layers[-1])(x)

        return x

    def encode(self, x, layer=None):
        if layer is None:
            mark_pos = self.depth - 1
        else:
            mark_pos = layer

        encode_layers = [f'encoder{i + 1}' for i in range(mark_pos)]

        # flatten inputs to vector
        x = x.view(-1, self.num_flat_features(x))

        for layer in encode_layers[:-1]:
            x = getattr(self, layer)(x)
            x = F.relu(x)

        x = getattr(self, encode_layers[-1])(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        dim = 1
        for s in size:
            dim *= s

        return dim


if __name__ == '__main__':

    model = DeepAE(100, 50, 20, 10)
    print(model)

    x = torch.rand(10, 100)
    t = model.encode(x)
    y = model(x)

    print(f'x: {x.shape}, t: {t.shape}, y: {y.shape}')

    t1 = model.encode(x, layer=1)
    t2 = model.encode(x, layer=2)
    print(f't1: {t1.shape}, t2: {t2.shape}')
