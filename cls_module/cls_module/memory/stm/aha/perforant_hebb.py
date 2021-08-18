"""PerforantHebb class."""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from cls_module.components.local_optimizer import LocalOptim
from cls_module.components.learning_rules import OjaLeabraRule
from cls_module.components.local_connection import LocalConnection


class PerforantHebb(nn.Module):
  """
  A non-trainable module based on Dentate Gyrus (DG), produces sparse outputs and inhibits neurons after firing.
  """

  def __init__(self, ec_shape, dg_shape, ca3_shape, config):
    super(PerforantHebb, self).__init__()

    self.config = config
    self.reset_params = self.config.get('reset_params', True)
    self.reset_optim = self.config.get('reset_optim', False)

    ec_size = np.prod(ec_shape[1:])
    dg_size = np.prod(dg_shape[1:])
    ca3_size = np.prod(ca3_shape[1:])

    self.dg_ca3 = LocalConnection(dg_size, ca3_size, bias=False).to(self.device)
    self.dg_ca3_optimizer = LocalOptim(self.dg_ca3.named_parameters(), lr=self.config['learning_rate'])

    self.ec_ca3 = LocalConnection(ec_size, ca3_size, bias=False).to(self.device)
    self.ec_ca3_optimizer = LocalOptim(self.ec_ca3.named_parameters(), lr=self.config['learning_rate'])

    self.learning_rule = OjaLeabraRule()

  def reset(self):
    for name, module in self.named_children():
      # Reset the module parameters
      if hasattr(module, 'reset_parameters') and self.reset_params:
        print(name, '=>', 'resetting parameters')
        module.reset_parameters()

    # Reset the module optimizer
    if self.reset_optim:
      print('PerforantHebb', '=>', 'resetting optimizer')
      self.dg_ca3_optimizer.state = defaultdict(dict)
      self.ec_ca3_optimizer.state = defaultdict(dict)

  def forward(self, ec_inputs, dg_inputs):
    with torch.no_grad():
      dg_ca3_in = dg_inputs
      pre_dg_ca3_out = self.dg_ca3(dg_ca3_in)

      ec_ca3_in = torch.flatten(ec_inputs, 1)
      pre_ec_ca3_out = self.ec_ca3(ec_ca3_in)

      pre_pc_cue = pre_dg_ca3_out + pre_ec_ca3_out
      pc_cue = pre_pc_cue

      if self.training:
        # Update DG:CA3 with respect to dg_ca3_in (i.e. outputs['ps'])
        d_dg_ca3 = self.learning_rule.compute_dw(dg_ca3_in, pre_pc_cue, self.dg_ca3.weight)
        d_dg_ca3 = d_dg_ca3.view(*self.dg_ca3.weight.size())
        self.dg_ca3_optimizer.local_step(d_dg_ca3)

        # Update EC:CA3 with respect to ec_ca3_in (i.e. inputs)
        d_ec_ca3 = self.learning_rule.compute_dw(ec_ca3_in, pre_pc_cue, self.ec_ca3.weight)
        d_ec_ca3 = d_ec_ca3.view(*self.ec_ca3.weight.size())
        self.ec_ca3_optimizer.local_step(d_ec_ca3)

      # Compute the post synaptic activity for loss calculation
      post_dg_ca3_out = self.dg_ca3(dg_ca3_in)
      post_ec_ca3_out = self.ec_ca3(ec_ca3_in)
      post_pc_cue = post_dg_ca3_out + post_ec_ca3_out

      dg_ca3_loss = F.mse_loss(pre_dg_ca3_out, post_dg_ca3_out)
      ec_ca3_loss = F.mse_loss(pre_ec_ca3_out, post_ec_ca3_out)
      pc_cue_loss = F.mse_loss(pre_pc_cue, post_pc_cue)

    return pc_cue, dg_ca3_loss, ec_ca3_loss, pc_cue_loss
