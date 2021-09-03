"""KNNBuffer class."""

import torch
import torch.nn as nn


class KNNBuffer(nn.Module):
  """
  A simple buffer implementation with K-nearest neighbour lookup.
  """

  def __init__(self, input_shape, target_shape, config):
    super(KNNBuffer, self).__init__()

    self.config = config

    self.reset()

  def set_buffer_mode(self, mode='override'):
    self.buffer_mode = mode

  def reset(self):
    self.buffer = None
    self.buffer_batch = None
    self.buffer_mode = 'override'

  def shift_inputs(self, tensor):
    """From sparse r[0,1] to b[-1,1]"""
    tensor = (tensor > 0).float()
    tensor[tensor == 0] = -1
    return tensor

  def forward(self, inputs):
    """
    During training, store the training pattern inputs into the buffer.
    Depending on the buffer_mode, the buffer can be either overriden every step
    or continously appended.

    At test time, use the test pattern inputs to lookup the matching patterns
    from the buffer using K-nearest neighbour with K=1.
    """
    if self.training:
      # Range shift from unit to signed unit
      if self.config['shift_range']:
        inputs = self.shift_inputs(inputs)

      self.buffer_batch = inputs

      # Memorise inputs in buffer
      if self.buffer is None or self.buffer_mode == 'override':
        self.buffer = inputs
      elif self.buffer_mode == 'append':
        self.buffer = torch.cat((self.buffer, inputs))

      return self.buffer_batch

    recalled = torch.zeros_like(inputs)

    for i, test_input in enumerate(inputs):
      dist = torch.norm(self.buffer - test_input, dim=1, p=None)
      knn = dist.topk(k=1, largest=False)
      recalled[i] = self.buffer[knn.indices]

    return recalled
