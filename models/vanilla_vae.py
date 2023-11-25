import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from .types_ import Tensor

from .decoder import Decoder
from .encoder import Encoder


class VanillaVAE(L.LightningModule):
    def __init__(self, 
                 lr: int,
                 beta: int,
                 encoder: Encoder, 
                 decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.beta = beta

    def training_step(self, x: Tensor, batch_idx: int) -> int:
        z_mean, z_log_var = self.encoder(x)
        z = self.sample_z(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        kl_loss = self.beta * torch.mean(
            torch.sum(
                0.5 * (torch.exp(z_log_var)**2 + z_mean**2 - 1 - 2 * z_log_var),
                dim=1
            ))
        return reconstruction_loss + kl_loss

    def sample_z(self, z_mean: Tensor, z_log_var: Tensor) -> Tensor:
        eps = torch.normal(0, 1, size=z_mean.size())
        return z_mean + torch.exp(0.5 * z_log_var) * eps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
