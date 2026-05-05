import torch 
from torch import nn
import torch.nn.functional as F

class BottleneckedDenseNetwork(nn.Module):
    def __init__(self, channels=3, k=3, classes=10):
        super().__init__()
        
        # creates a consistent number of channels to be used in the highway layers
        # creates a consistent number of channels to be used in the highway layers
        self.input = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # block 1
        block_1_layers = 6
        block_1_in_channels = 3
        block_1_out_channels = 12
        self.block_1 = nn.Sequential(
            BottleneckedDenseBlock(num_layers=block_1_layers, channels=block_1_in_channels, k=k),
            TransitionLayer(in_channels=block_1_in_channels + block_1_layers * k, out_channels=block_1_out_channels) 
        )

        # block 2
        block_2_layers = 6
        block_2_out_channels = 12
        self.block_2 = nn.Sequential(
            BottleneckedDenseBlock(num_layers=block_2_layers, channels=block_1_out_channels, k=k),
            TransitionLayer(in_channels=block_1_out_channels + block_2_layers * k, out_channels=block_2_out_channels) 
        )

        # block 3
        block_3_layers = 10
        block_3_out_channels = 12
        self.block_3 = nn.Sequential(
            BottleneckedDenseBlock(num_layers=block_3_layers, channels=block_2_out_channels, k=k),
            TransitionLayer(in_channels=block_2_out_channels + block_3_layers * k, out_channels=block_3_out_channels) 
        )

        # block 4
        block_4_layers = 10
        self.block_4 = nn.Sequential(
            BottleneckedDenseBlock(num_layers=block_4_layers, channels=block_3_out_channels, k=k),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(block_3_out_channels + block_4_layers * k, classes)
        )
    def forward(self, x):
        x = self.input(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.output(x)
        return x


class BottleneckedDenseBlock(nn.Module):
    def __init__(self, num_layers, channels, k=3, kernel_size=3):
        super().__init__()
        self.bottlenecked_dense_block = self._build_bottlenecked_dense_block(num_layers, channels, k, kernel_size)
        
    def forward(self, x):
        features = x
        for bottlenecked_dense_layer in self.bottlenecked_dense_block:
            x = bottlenecked_dense_layer(features)
            features = torch.cat([features, x], dim=1)
        return features

    def _build_bottlenecked_dense_block(self, num_layers, channels, k, kernel_size=3):
        layers = []
        bottleneck_channels = 4 * k  # standard DenseNet-B choice

        for i in range(num_layers):
            bottleneck_sequence = nn.Sequential(
                # BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=channels, out_channels=bottleneck_channels, kernel_size=1),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=bottleneck_channels, out_channels=k, kernel_size=kernel_size, padding=1),
            )
            layers.append(bottleneck_sequence)
            channels += k  # grow by k per layer # double number of channels every iteration 
        return nn.ModuleList(layers)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.transition_layer(x)
        return x
