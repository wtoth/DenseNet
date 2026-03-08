import torch 
from torch import nn
import torch.nn.functional as F

class DenseNetwork(nn.Module):
    def __init__(self, channels, classes=10):
        super().__init__()
        
        # creates a consistent number of channels to be used in the highway layers
        self.input = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.MaxPool2d()
        )
        self.denseblock1 = DenseBlock(num_layers=5, channels=3)

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(channels, classes)
        ) 

    def forward(self, x):
        x = self.input(x)
        x = self.denseblock1(x)
        x = self.output(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, channels, kernel_size=3):
        super().__init__()

        self.denseblock = self._build_dense_block(num_layers, channels, kernel_size)

    def forward(self, x):
        x = self.denseblock(x)
        return x

    def _build_dense_block(self, num_layers, channels, kernel_size=3):
        layers = []

        for i in range(num_layers):
            layers.append(nn.Conv2d(channels=(i+1)*channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)

class BottleneckedDenseBlock(nn.Module):
    def __init__(self, depth, channels, kernel_size=3):
        super().__init__()
        
    def forward(self, x):
        pass

    def _build_bottlenecked_dense_block(self, num_layers, channels, kernel_size=3):
        pass

class TransitionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
