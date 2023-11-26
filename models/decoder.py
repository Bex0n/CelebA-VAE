import torch
import torch.nn as nn
import torch.nn.functional as F
from .types_ import List, Tensor

class Decoder(nn.Module):
    def __init__(self,
                 z_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128, 128, 128]
        self.hidden_dims = hidden_dims

        modules = []

        self.decoder_input = nn.Sequential(
            nn.Linear(z_dim, self.hidden_dims[0] * 4 * 4),
            nn.ReLU()
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.ReLU()
                )
            )

        # Final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, x: Tensor):
        x = self.decoder_input(x)
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.decoder(x)
        return x
