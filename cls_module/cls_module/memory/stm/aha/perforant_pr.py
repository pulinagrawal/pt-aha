"""PerforantPR class."""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cerenaut_pt_core.utils import build_topk_mask
from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder


class PerforantPR(nn.Module):
  """
  An error-driven perforant pathway implementation.
  """

  def __init__(self, input_shape, target_shape, config):
    super(PerforantPR, self).__init__()

    self.config = config
    self.reset_params = self.config.get('reset_params', True)
    self.reset_optim = self.config.get('reset_optim', False)

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
    """Perform one step using the PR module to learn to replicate the patterns generated using the PS."""
    if self.training:
      self.optimizer.zero_grad()

    _, logits = self.model(inputs)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    if self.training:
      loss.backward()
      self.optimizer.step()

    y = torch.sigmoid(logits)
    y = y.detach()

    # Clip
    y = torch.clamp(y, 0.0, 1.0)

    # Sparsen
    if self.config['sparsen']:
      k_pr = int(self.config['sparsity'] * self.config['sparsity_boost'])
      mask = build_topk_mask(y, dim=1, k=k_pr)
      y = y * mask

    # Sum norm (all input is positive)
    # We expect a lot of zeros, or near zeros, and a few larger values.
    if self.config['sum_norm'] > 0.0:
      eps = 1e-13
      y_sum = torch.sum(y, dim=1, keepdim=True)
      reciprocal = 1.0 / y_sum + eps
      y = y * reciprocal * self.config['sum_norm']

    # Softmax norm
    if self.config['softmax']:
      y = F.softmax(y)

    # After norm
    if self.config['gain'] != 1.0:
      y = y * self.config['gain']

    # Normalize to [0, 1]
    # y = (y - y.min()) / (y.max() - y.min())

    # This output will get used for the matching accuracy, similar to TF-AHA
    pr_out = y  # Unit range

    z_cue_in = pr_out

    # Range shift from unit to signed unit
    if self.config['shift_range']:
      z_cue_in = self.unit_to_pc_linear(pr_out)  # Theoretical range limits [-1, 1]

    z_cue_shift = z_cue_in

    # Shift until k bits are > 0, i.e. min *masked* value should become equal to zero.
    if self.config['shift_bits']:
      shift = self.get_pc_topk_shift(pr_out, self.config['sparsity'])
      z_cue_shift = z_cue_in + shift

    outputs = {
        'pr_out': pr_out.detach(),
        'z_cue_in': z_cue_in.detach(),
        'z_cue': z_cue_shift.detach()
    }

    return loss, outputs

  def unit_to_pc_linear(self, tensor):
    """Input assumed to be unit range. Linearly scales to -1 <= x <= 1"""
    return (tensor * 2.0) - 1.0  # Theoretical range limits -1 : 1


  def get_pc_topk_shift(self, tensor, sparsity):
    """Input tensor must be batch of vectors.
    Returns a vector per batch sample of the shift required to make Hopfield converge.
    Assumes knowledge of Hopfield fixed sparsity."""

    # Intuition: The output distribution must straddle the zero point to make hopfield work.
    # These are the values that should be positive.
    cue_top_k_mask = build_topk_mask(tensor, dim=1, k=int(sparsity + 1))

    y = tensor
    y_inv = 1.0 - y
    y_inv_masked = y_inv * cue_top_k_mask
    y_inv_masked_max, _ = torch.max(y_inv_masked, dim=1)  # max per batch sample
    y_masked_min = 1.0 - y_inv_masked_max

    cue_tanh_masked_min = (y_masked_min * 2.0) - 1.0  # scale these values
    shift = (0.0 - cue_tanh_masked_min).unsqueeze(1)

    return shift
