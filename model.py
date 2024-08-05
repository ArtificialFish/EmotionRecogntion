import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Model(nn.Module):
    def __init__(self):
        """Define the architecture, i.e. what layers our network contains."""
        super().__init__()

        # Input (3x48x48)
        # Output (16x24x24)
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)

        self.pool = nn.MaxPool2d(2, stride=2)

        # Input (16x12x12)
        # Output (64x6x6)
        self.conv2 = nn.Conv2d(16, 64, 5, stride=2, padding=2)

        # Input (64x3x3)
        # Output (8x2x2)
        self.conv3 = nn.Conv2d(64, 8, 5, stride=2, padding=2)

        self.fc_1 = nn.Linear(32, 2)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        nn.init.normal_(
            self.fc_1.weight, 0.0, 1 / sqrt(self.fc_1.weight.size(1))
        )

        nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x):
        """
        Pass the output of the previous layer as the input into the next layer (after applying
        activation functions). Returns the final output as a torch.Tensor object.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc_1(x)

        return x
