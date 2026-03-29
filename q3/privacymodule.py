import torch
import torch.nn as nn

class PrivacyObfuscator(nn.Module):
    def __init__(self, input_dim=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        # x: [B, 40, T]
        x = x.transpose(1, 2)   # [B, T, 40]
        x = self.encoder(x)
        return x.transpose(1, 2)