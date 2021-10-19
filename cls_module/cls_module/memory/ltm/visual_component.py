"""VisualComponent class."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cerenaut_pt_core import utils

from cls_module.memory.interface import MemoryInterface
# from cerenaut_pt_core.components.sparse_autoencoder import SparseAutoencoder

import numpy as np
import torch.nn.functional as F

# Device - cuda vs. cpu
# cuda = torch.cuda.is_available()
# DEVICE = torch.device("cuda:0" if cuda else "cpu")

# # Hyperparameters
# RANDOM_SEED = 123

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
        # return x[:, :, :28, :28]
        return x[:, :, :52, :52]


class beta_VAE(nn.Module):
    def __init__(self, input_shape, config, output_shape=None):
        super().__init__()

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
            # nn.Flatten(),
        ) 

        self.z_mean = nn.Sequential(
            nn.Flatten(), 
            torch.nn.Linear(3136, 30))
        self.z_log_var = nn.Sequential(
            nn.Flatten(), 
            torch.nn.Linear(3136, 30))

        self.decoder = nn.Sequential(
            torch.nn.Linear(30, 3136),
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

    def encode(self, x, stride):
        encoding = self.encoder(x)
        return encoding
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x, stride):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoding = self.reparameterize(z_mean, z_log_var)
        # print(encoding.size())
        decoding = self.decoder(encoding)
        # print("\nhi\n")
        return encoding, decoding


class VisualComponent(MemoryInterface):
    """An implementation of a long-term memory module using sparse convolutional autoencoder."""

    global_key = 'ltm'
    local_key = 'vc'

    def build(self):

        LEARNING_RATE = 0.0005

        """Build Visual Component as long-term memory module."""
        # print("\n\n\nprint", self.input_shape)
        vc = beta_VAE(self.input_shape, self.config).to(self.device)
        vc_optimizer = torch.optim.Adam(vc.parameters(), lr=LEARNING_RATE) 

        self.add_module(self.local_key, vc)
        self.add_optimizer(self.local_key, vc_optimizer)

        # Compute expected output shape
        with torch.no_grad():
            stride = self.config.get('eval_stride', self.config.get('stride'))
            sample_input = torch.rand(1, *(self.input_shape[1:])).to(self.device)
            print ("\nsample_input check in my visual_comp: ", sample_input)
            print ("\nsample_input.shape check in my visual_comp: ", sample_input.shape)
            sample_output = vc.encode(sample_input, stride=stride)
            print ("\n sample_output.shape check in my visual_comp after vc.encode: ", sample_output.shape)
            print ("\n\n ")
            # sample_output = self.prepare_encoding(sample_output)
            # print ("\n sample_output.shape check in my visual_comp after prepare_encoding: ", sample_output.shape)
            self.output_shape = list(sample_output.data.shape)
            print ("\n self.output_shape check in my visual_comp: ", list(sample_output.data.shape))
            self.output_shape[0] = -1

        # if 'classifier' in self.config:
        #     self.build_classifier(input_shape=self.output_shape)

    # def forward_decode(self, encoding): 
    #     # Optionally use different stride at test time
    #     stride = self.config['stride']
    #     if not self.vc.training and 'eval_stride' in self.config:
    #         stride = self.config['eval_stride']

    #     encoding = self.unprepare_encoding(encoding)

    #     with torch.no_grad():
    #         return self.vc.decode(encoding, stride)

    def forward_memory(self, inputs, targets, labels):    

        """Perform an optimization step using the memory module."""
        del labels

        if self.vc.training:
            self.vc_optimizer.zero_grad()

        # Optionally use different stride at test time
        stride = self.config['stride']
        if not self.vc.training and 'eval_stride' in self.config:
            stride = self.config['eval_stride']

        encoding, decoding = self.vc(inputs, stride)
        loss = F.mse_loss(decoding, targets)

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
        
        """Postprocessing for the VC encoding."""
        # encoding = encoding.detach()

        # if self.config['output_pool_size'] > 1:
        #     encoding, self.pool_indices = F.max_pool2d(
        #         encoding,
        #         kernel_size=self.config['output_pool_size'],
        #         stride=self.config['output_pool_stride'],
        #         padding=self.config.get('output_pool_padding', 0), return_indices=True)

        return encoding

    # def unprepare_encoding(self, prepared_encoding):

    #     """Undo any postprocessing for the VC encoding."""
    #     encoding = prepared_encoding
    #     encoding = prepared_encoding.detach()

    #     # if self.config['output_pool_size'] > 1:
    #     #     encoding = F.max_unpool2d(
    #     #         encoding,
    #     #         kernel_size=self.config['output_pool_size'],
    #     #         stride=self.config['output_pool_stride'],
    #     #         padding=self.config.get('output_pool_padding', 0),
    #     #         indices=self.pool_indices)

    #     return encoding
