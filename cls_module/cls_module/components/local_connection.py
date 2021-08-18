"""LocalConnection class."""

import torch.nn as nn

class LocalConnection(nn.Linear):
  def __init__(self, input_features, output_features, bias=False):
    super(LocalConnection, self).__init__(input_features, output_features, bias)

    self.weight.requires_grad = False

    if bias:
      self.bias.requires_grad = False

    self.reset_parameters()

  def reset_parameters(self):
    def custom_weight_init(m):
      if not isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        return

      if m.weight is not None:
        nn.init.uniform_(m.weight)

      if m.bias is not None:
        nn.init.zeros_(m.bias)

    self.apply(custom_weight_init)
