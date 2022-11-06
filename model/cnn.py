import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, kernel_size=(7, 7), out_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, kernel_size=(5, 5), out_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, kernel_size=(3, 3), out_channels=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, kernel_size=(3, 3), out_channels=output_dim),
            nn.ReLU()
        )

    def forward(self, h):
        return self.model(h)
