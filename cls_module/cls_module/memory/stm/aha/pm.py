"""PatternMapper class."""

from collections import defaultdict

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder


class PatternMapper(nn.Module):
  """
  A simple pattern mapping module based on a fully-connected simple autoencoder. It can map arbitrary inputs to
  outputs, and can be easily configured.
  """

  def __init__(self, input_shape, target_shape, config):
    super(PatternMapper, self).__init__()

    self.config = config
    self.reset_params = self.config.get('reset_params', True)
    self.reset_optim = self.config.get('reset_optim', True)

    self.model = SimpleAutoencoder(input_shape, config, output_shape=target_shape)
    self.optimizer = optim.Adam(self.model.parameters(),
                                lr=self.config['learning_rate'],
                                weight_decay=self.config['weight_decay'])

  def reset(self):
    if self.reset_params:
      self.model.reset_parameters()

    if self.reset_optim:
      self.optimizer.state = defaultdict(dict)

  def forward(self, inputs, targets):
    if self.training:
      self.optimizer.zero_grad()

    encoding, decoding = self.model(inputs)

    loss = F.mse_loss(decoding, targets)

    outputs = {
        'encoding': encoding.detach(),
        'decoding': decoding.detach(),

        'output': encoding.detach()  # Designated output for linked modules
    }

    if self.training:
      loss.backward()
      self.optimizer.step()

    return loss, outputs
