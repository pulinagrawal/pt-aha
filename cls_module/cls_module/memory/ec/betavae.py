

import torch
import torch.nn as nn

import numpy as np

import cerenaut_pt_core.utils as utils


class BetaVAE(nn.Module):

  def __init__(self, input_shape, config, device=None, writer=None):
    super(BetaVAE, self).__init__()

    self.config = config

    self.input_shape = list(input_shape)
    self.input_size = np.prod(self.input_shape[1:])
    self.output_shape = None

    self.build()

  def build(self):
    self.output_shape = self.input_shape

  def forward(self, inputs):
    return inputs
