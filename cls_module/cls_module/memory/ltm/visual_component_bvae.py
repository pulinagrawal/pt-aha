"""VisualComponent class."""
# 13 oct
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cerenaut_pt_core import utils

from cls_module.memory.interface import MemoryInterface

import numpy as np
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :52, :52]


class beta_VAE(nn.Module):
    def __init__(self, input_shape, config, output_shape=None):
        super().__init__()

        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.build()

    def build(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01), 
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.Flatten(),
        )

        self.z_mean = torch.nn.Linear(3136, self.config['latent_size'])
        self.z_log_var = torch.nn.Linear(3136, self.config['latent_size'])

        self.decoder = nn.Sequential(
            torch.nn.Linear(self.config['latent_size'], 3136),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
            Trim(),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoding = self.reparameterize(z_mean, z_log_var)
        return encoding

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.device)
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        self.z_m, self.z_lv = self.z_mean(x), self.z_log_var(x)
        encoding = self.reparameterize(self.z_m, self.z_lv)
        decoding = self.decoder(encoding)
        return encoding, decoding

    def get_distribution(self):
        return self.z_m, self.z_lv


class VisualComponentBVAE(MemoryInterface):
    """An implementation of a long-term memory module using beta variational autoencoder (beta-VAE)."""

    global_key = 'ltm'
    local_key = 'vc'

    def build(self):
        """Build Visual Component as long-term memory module."""
        vc = beta_VAE(self.input_shape, self.config).to(self.device)
        vc_optimizer = torch.optim.Adam(vc.parameters(), lr=self.config['learning_rate'])

        self.add_module(self.local_key, vc)
        self.add_optimizer(self.local_key, vc_optimizer)

        # Compute expected output shape
        with torch.no_grad():
            sample_input = torch.rand(1, *(self.input_shape[1:])).to(self.device)
            sample_output = vc.encode(sample_input)
            self.output_shape = list(sample_output.data.shape)
            self.output_shape[0] = -1

    def forward_memory(self, inputs, targets, labels):
        """Perform an optimization step using the memory module."""
        del labels

        if self.vc.training:
            self.vc_optimizer.zero_grad()

        encoding, decoding = self.vc(inputs)
        mu, logvar = self.vc.get_distribution()

        kl_div_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=1)
        kl_div_loss = kl_div_loss.mean()

        # reconstruction_loss = F.binary_cross_entropy(decoding, targets, reduction='sum')
        reconstruction_loss = F.mse_loss(decoding, targets, reduction='sum')

        k = self.config['recon_loss_const']
        b = self.config['beta']
        loss = k * reconstruction_loss + b * kl_div_loss

        if self.vc.training:
            loss.backward()
            self.vc_optimizer.step()

        output_encoding = self.prepare_encoding(encoding)

        outputs = {
            'encoding': encoding,
            'decoding': decoding,
            'output': output_encoding  # Designated output for linked modules
        }

        self.features = {
            'vc': output_encoding.detach().cpu(),
            'recon': decoding.detach().cpu()
        }

        return loss, outputs

    def prepare_encoding(self, encoding):
        return encoding.detach()

    def unprepare_encoding(self, prepared_encoding):
        return prepared_encoding
