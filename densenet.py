import torch 
from torch import nn
import torch.nn.functional as F

class DenseNetwork(nn.Module):
    def __init__(self, channels=3, classes=10):
        super().__init__()
        
        # creates a consistent number of channels to be used in the highway layers
        self.input = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=1, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # block 1
        block_1_layers = 6
        block_1_in_channels = 3
        block_1_out_channels = 12
        self.block_1 = nn.Sequential(
            DenseBlock(num_layers=block_1_layers, channels=block_1_in_channels),
            # transition_layer in_channels will be # of channels the denseblock returns
            TransitionLayer(in_channels=block_1_layers*block_1_in_channels, out_channels=block_1_out_channels) 
        )

        # block 2
        block_2_layers = 12
        block_2_out_channels = 12
        self.block_2 = nn.Sequential(
            DenseBlock(num_layers=block_2_layers, channels=block_1_out_channels),
            # transition_layer in_channels will be # of channels the denseblock returns
            TransitionLayer(in_channels=block_2_layers*block_1_out_channels, out_channels=block_2_out_channels) 
        )

        # block 3
        block_3_layers = 24
        block_3_out_channels = 12
        self.block_3 = nn.Sequential(
            DenseBlock(num_layers=block_3_layers, channels=block_2_out_channels),
            # transition_layer in_channels will be # of channels the denseblock returns
            TransitionLayer(in_channels=block_3_layers*block_2_out_channels, out_channels=block_3_out_channels) 
        )
        # block 4
        block_4_layers = 16
        self.block_4 = nn.Sequential(
            DenseBlock(num_layers=block_4_layers, channels=block_3_out_channels),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(block_3_out_channels, classes)
        ) 

    def forward(self, x):
        x = self.input(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.output(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, channels, kernel_size=3):
        super().__init__()
        self.dense_block = self._build_dense_block(num_layers, channels, kernel_size)

    def forward(self, x):
        for dense_layer in self.dense_block:
            x = dense_layer(x)
        return x

    def _build_dense_block(self, num_layers, channels, kernel_size=3):
        layers = []

        for i in range(num_layers):
            bottleneck_sequence = nn.Sequential(
                # BN-ReLU-Conv
                nn.BatchNorm2d((i+1)*channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=(i+1)*channels, out_channels=(i+1)*channels, kernel_size=kernel_size),
            )
            layers.append(bottleneck_sequence)
        return nn.ModuleList(layers)

class BottleneckedDenseBlock(nn.Module):
    def __init__(self, num_layers, channels, kernel_size=3):
        super().__init__()
        self.bottlenecked_dense_block = self._build_dense_block(num_layers, channels, kernel_size)
        
    def forward(self, x):
        for bottlenecked_dense_layer in self.bottlenecked_dense_block:
            x = bottlenecked_dense_layer(x)
        return x

    def _build_bottlenecked_dense_block(self, num_layers, channels, kernel_size=3):
        layers = []
        for i in range(num_layers):
            bottleneck_sequence = nn.Sequential(
                # BN-ReLU-Conv
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=(i+1)*channels, out_channels=(i+1)*channels, kernel_size=1),

                # BN-ReLU-Conv
                nn.BatchNorm2d((i+1)*channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=(i+1)*channels, out_channels=(i+1)*channels, kernel_size=kernel_size),
            )
            layers.append(bottleneck_sequence)
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
