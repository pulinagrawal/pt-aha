"""LocalConnection class."""

import torch.nn as nn

class LocalConnection(nn.Linear):
  """
  LocalConnection class is based on the PyTorch module, nn.Linear, which is a standard
  fully connected layer. The forward method performs a typical weight multiplication with optional bias
    output = matmul(weight, inputs) + bias

  The only difference between LocalConnection and nn.Linear is that LocalConnection disables
  gradients, making it an untrainable layer with PyTorch backpropagation.

  This is because it will be predominantly used with local Hebbian learning, where weight updates
  will be applied without backprop optimization.

  The `weight` variable is a standard matrix shaped (input_features, output_features).
  The `bias` variable is a standard vector shaped (output_features).

  The bias is not necessarily useful at the moment as it will always be zero, unless we make the
  initialisation parameterisable, allowing you to bias the output by either random values or constants.
  """
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
