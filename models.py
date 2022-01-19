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
            nn.Conv2d(out_channels, out_channels, 3, 1, 3),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.module(x)

class MNISTCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = [CNNBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)]
        self.layers = nn.ModuleList(self.layers)
        self.classifier = nn.Sequential(
            nn.Linear(4*4*channels[-1], 4*4*channels[-1]),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Linear(4*4*channels[-1], 10)
        )

    def forward(self, x):
        B = x.size(0)
        for layer in self.layers:
            x = layer(x)
        x = x.view(B, -1)
        return self.classifier(x)