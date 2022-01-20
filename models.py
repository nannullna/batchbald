from turtle import forward
import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),
        )

    def forward(self, x):
        return self.module(x)

class MNISTCNN(nn.Module):
    def __init__(self, channels, num_classes:int=10):
        super().__init__()
        self.channels = channels
        self.layers = [CNNBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)]
        self.layers = nn.ModuleList(self.layers)
        self.classifier = nn.Sequential(
            nn.Linear(5*5*channels[-1], 5*5*channels[-1]),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Linear(5*5*channels[-1], num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)