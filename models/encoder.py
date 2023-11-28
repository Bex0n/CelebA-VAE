import torch
import torch.nn as nn
import torch.nn.functional as F
from .types_ import List, Tensor

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 z_dim: int,
                 hidden_dims: List = None,
                 **kwargs):
        super(Encoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128, 128]
        self.hidden_dims = [in_channels] + hidden_dims

        modules = []

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.hidden_dims[i],
                              out_channels=self.hidden_dims[i + 1],
                              kernel_size=3,
                              padding=1,
                              stride=2),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.ReLU()
                )
            )

        self.encoder = nn.Sequential(*modules)

        self.fc_mean = nn.Linear(self.hidden_dims[-1] * 16, z_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * 16, z_dim)

    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_var(x)
        return z_mean, z_log_var
