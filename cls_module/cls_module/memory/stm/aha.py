"""AHA class."""

import logging
from collections import defaultdict
from os import pathconf

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cls_module.memory.interface import MemoryInterface
from cls_module.components.dg import DG
from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder
from cerenaut_pt_core.utils import build_topk_mask
from cls_module.components.local_optimizer import LocalOptim
from cls_module.components.learning_rules import OjaLeabraRule

def pc_to_unit(tensor):
  """
  From b[-1,1] to [0,1]
  This implementation only works assuming tensor is binary
  """
  return (tensor > 0).float()


def unit_to_pc_linear(tensor):
  """Input assumed to be unit range. Linearly scales to -1 <= x <= 1"""
  return (tensor * 2.0) - 1.0  # Theoretical range limits -1 : 1


def unit_to_pc_sparse(tensor):
  """ From b[0,1] to b[-1,1] or [0,1] to r[-1,1] """
  tensor[tensor == 0] = -1
  return tensor


def get_pc_topk_shift(tensor, sparsity):
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


def dg_to_pc(tensor):
  """ From sparse r[0,1] to b[-1,1]"""
  tensor = (tensor > 0).float()
  tensor[tensor == 0] = -1
  return tensor


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


class AHA(MemoryInterface):
  """
  An implementation of a short-term memory module using a AHA.

    There are two major modes. 
    1) Standard CLS, with Hebbian learning in the perforant path
  
    Build:
    DG -> CA3: set of weights (randomly initialised)
    EC -> CA3: set of weights (randomly initialised)
    For CA3 neurons, it's a weighted sum from two sets of weights.

    Forward:
    Propagate
    Update weights given pre and post synaptic activity

    2) Our approach, with error driven learning in perforant pathconf
    Build:
    DG -> CA3: 1 to 1 clamp
    EC -> CA3 = PR network

    Forward:
    Propagate
    PR: BP Optimization step over
  """

  global_key = 'stm'
  local_key = 'aha'

  def is_hebbian_perforant(self):
    k = 'hebbian_perforant'
    if k in self.config and self.config[k]:
      return True

    return False

  def reset(self):
    """Reset modules and optimizers."""
    self.pc_buffer = None
    self.pc_buffer_batch = None
    self.pc_buffer_mode = 'override'

    # TF-AHA currently only resets PM optimizer, so avoid resetting the PR
    # No point resetting the PS as it's not trainable anyway, keep it consistent between runs
    resets = {
        'pr': {
            'params': True,
            'optim': False,
        },
        'ps': {
            'params': False,
            'optim': False
        },
        'pm': {
            'params': True,
            'optim': True
        },
        'dg_ca3': {
            'params': True,
            'optim': True
        },
        'ec_ca3': {
            'params': True,
            'optim': True
        }
    }

    for name, module in self.named_children():
      if name not in resets.keys():
        continue

      # Reset the module parameters
      if hasattr(module, 'reset_parameters') and resets[name]['params']:
        # print(name, '=>', 'resetting parameters')
        module.reset_parameters()

      # Reset the module optimizer
      optimizer_name = name + '_optimizer'
      if hasattr(self, optimizer_name) and resets[name]['optim']:
        # print(name, '=>', 'resetting optimizer')
        module_optimizer = getattr(self, optimizer_name)
        module_optimizer.state = defaultdict(dict)

  def build(self):
    """Build AHA as short-term memory module."""

    # Build the Pattern Separation (PS) module
    ps_output_shape = self.build_ps(ps_config=self.config['ps'], ps_input_shape=self.input_shape)

    # Build the Pattern Retrieval (PR) module
    self.build_pr(pr_config=self.config['pr'], pr_input_shape=self.input_shape, pr_target_shape=ps_output_shape)

    # Build the Pattern Completion (PC) module
    pc_output_shape = self.build_pc(pc_config=self.config['pc'], pc_input_shape=ps_output_shape)

    # Build the Pattern Mapping (PM) module - to the EC (as per biology)
    self.build_ae(name='pm_ec', config=self.config['pm_ec'],
                  input_shape=pc_output_shape,
                  target_shape=self.input_shape)

    # Build the Pattern Mapping (PM) module - to an arbitrary 'target' (main use-case is the original input image)
    self.build_ae(name='pm', config=self.config['pm'],
                  input_shape=pc_output_shape,
                  target_shape=self.target_shape)

    # Build the Label Learner module
    if 'classifier' in self.config:
      self.build_classifier(input_shape=pc_output_shape)

    self.output_shape = pc_output_shape

  def build_ps(self, ps_config, ps_input_shape):
    """Build the Pattern Separation (PS) module."""
    ps = DG(ps_input_shape, ps_config).to(self.device)
    ps_output_shape = [1, ps_config['num_units']]

    self.add_module('ps', ps)

    return ps_output_shape

  def forward_ps(self, inputs):
    return self.ps(inputs)

  def build_pr(self, pr_config, pr_input_shape, pr_target_shape):
    """Builds the Pattern Retrieval (PR) module."""
    pr = SimpleAutoencoder(pr_input_shape, pr_config, output_shape=pr_target_shape).to(self.device)
    pr_optimizer = optim.Adam(pr.parameters(),
                              lr=pr_config['learning_rate'],
                              weight_decay=pr_config['weight_decay'])

    self.add_module('pr', pr)
    self.add_optimizer('pr', pr_optimizer)

  def forward_pr(self, inputs, targets):
    """Perform one step using the PR module to learn to replicate the patterns generated using the PS."""
    pr_config = self.config['pr']

    if self.pr.training:
      self.pr_optimizer.zero_grad()

    _, logits = self.pr(inputs)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    if self.pr.training:
      loss.backward()
      self.pr_optimizer.step()

    y = torch.sigmoid(logits)
    y = y.detach()

    # Clip
    y = torch.clamp(y, 0.0, 1.0)

    # Sparsen
    if pr_config['sparsen']:
      k_pr = int(pr_config['sparsity'] * pr_config['sparsity_boost'])
      logging.info('PR Sparsen enabled k = %s', str(k_pr))
      mask = build_topk_mask(y, dim=1, k=k_pr)
      y = y * mask

    # Sum norm (all input is positive)
    # We expect a lot of zeros, or near zeros, and a few larger values.
    if pr_config['sum_norm'] > 0.0:
      logging.info('PR Sum-norm enabled')
      eps = 1e-13
      y_sum = torch.sum(y, dim=1, keepdim=True)
      reciprocal = 1.0 / y_sum + eps
      y = y * reciprocal * pr_config['sum_norm']

    # Softmax norm
    if pr_config['softmax']:
      logging.info('PR Softmax enabled')
      y = F.softmax(y)

    # After norm
    if pr_config['gain'] != 1.0:
      logging.info('PR Gain enabled')
      y = y * pr_config['gain']

    # Normalize to [0, 1]
    # y = (y - y.min()) / (y.max() - y.min())

    # This output will get used for the matching accuracy, similar to TF-AHA
    pr_out = y  # Unit range

    z_cue_in = pr_out

    # Range shift from unit to signed unit
    if pr_config['shift_range']:
      z_cue_in = unit_to_pc_linear(pr_out)  # Theoretical range limits [-1, 1]

    z_cue_shift = z_cue_in

    # Shift until k bits are > 0, i.e. min *masked* value should become equal to zero.
    if pr_config['shift_bits']:
      logging.info('PR Shift enabled')
      shift = get_pc_topk_shift(pr_out, pr_config['sparsity'])
      z_cue_shift = z_cue_in + shift

    outputs = {
        'pr_out': pr_out.detach(),
        'z_cue_in': z_cue_in.detach(),
        'z_cue': z_cue_shift.detach()
    }

    return loss, outputs

  def build_pc(self, pc_config, pc_input_shape):
    """Builds the Pattern Completion (PC) module."""

    if self.is_hebbian_perforant():
      # Builds single set of weights (doesn't currently exist, because activations are copied across)
      # PC must be initialised with random weights
      ec_size = np.prod(self.input_shape[1:])
      ps_size = np.prod(pc_input_shape[1:])

      dg_ca3 = LocalConnection(ps_size, ps_size, bias=False).to(self.device)
      dg_ca3_optimizer = LocalOptim(dg_ca3.named_parameters(), lr=pc_config['learning_rate'])

      self.add_module('dg_ca3', dg_ca3)
      self.add_optimizer('dg_ca3', dg_ca3_optimizer)

      ec_ca3 = LocalConnection(ec_size, ps_size, bias=False).to(self.device)
      ec_ca3_optimizer = LocalOptim(ec_ca3.named_parameters(), lr=pc_config['learning_rate'])

      self.add_module('ec_ca3', ec_ca3)
      self.add_optimizer('ec_ca3', ec_ca3_optimizer)

      self.learning_rule = OjaLeabraRule()

    # Initialise the buffer
    self.pc_buffer = None

    return pc_input_shape

  def set_pc_buffer_mode(self, mode='override'):
    self.pc_buffer_mode = mode

  def forward_pc(self, inputs):
    """
    During training, store the inputs from PS into the buffer. At test time, use the inputs from the PR to lookup the
    matching PS pattern from the buffer using K-nearest neighbour with K=1.
    """
    pc_config = self.config['pc']

    if self.training:
      # Range shift from unit to signed unit
      if pc_config['shift_range']:
        inputs = dg_to_pc(inputs)

      self.pc_buffer_batch = inputs

      # Memorise inputs in buffer
      if self.pc_buffer is None or self.pc_buffer_mode == 'override':
        self.pc_buffer = inputs
      elif self.pc_buffer_mode == 'append':
        self.pc_buffer = torch.cat((self.pc_buffer, inputs))

      return self.pc_buffer_batch

    recalled = torch.zeros_like(inputs)

    for i, test_input in enumerate(inputs):
      dist = torch.norm(self.pc_buffer - test_input, dim=1, p=None)
      knn = dist.topk(k=1, largest=False)
      recalled[i] = self.pc_buffer[knn.indices]

    return recalled

  def build_ae(self, name, config, input_shape, target_shape):
    ae = SimpleAutoencoder(input_shape, config, output_shape=target_shape).to(self.device)
    ae_optimizer = optim.Adam(ae.parameters(),
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])

    self.add_module(name, ae)
    self.add_optimizer(name, ae_optimizer)

  def forward_ae(self, name, inputs, targets):
    """
    Perform one step using the PM module to learn to reconstruct input image using the patterns from the PC.
    """

    ae = getattr(self, name)
    ae_optimizer = getattr(self, name + '_optimizer')

    if ae.training:
      ae_optimizer.zero_grad()

    encoding, decoding = self.pm(inputs)

    loss = F.mse_loss(decoding, targets)

    outputs = {
        'encoding': encoding.detach(),
        'decoding': decoding.detach(),

        'output': encoding.detach()  # Designated output for linked modules
    }

    if ae.training:
      loss.backward()
      ae_optimizer.step()

    return loss, outputs

  def forward_memory(self, inputs, targets, labels):
    """Perform one step using the entire system (i.e. all sub-modules of AHA)."""
    del labels

    losses = {}
    outputs = {}
    features = {}

    outputs['ps'] = self.forward_ps(inputs)
    features['ps'] = outputs['ps'].detach().cpu()

    # Compute DG Overlap
    overlap = self.ps.compute_overlap(outputs['ps'])
    losses['ps_overlap'] = overlap.sum()

    # Perforant Pathway: Hebbian Learning
    if self.is_hebbian_perforant():
      with torch.no_grad():
        dg_ca3_in = outputs['ps']
        pre_dg_ca3_out = self.dg_ca3(dg_ca3_in)

        ec_ca3_in = torch.flatten(inputs, 1)
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
        post_pc_cue = post_ec_ca3_out + post_ec_ca3_out

        losses['dg_ca3'] = F.mse_loss(pre_dg_ca3_out, post_dg_ca3_out)
        losses['ec_ca3'] = F.mse_loss(pre_ec_ca3_out, post_ec_ca3_out)
        losses['pc_cue'] = F.mse_loss(pre_pc_cue, post_pc_cue)

    # Perforant Pathway: Error-Driven Learning
    if not self.is_hebbian_perforant():
      pr_targets = outputs['ps'] if self.training else self.pc_buffer_batch
      losses['pr'], outputs['pr'] = self.forward_pr(inputs=inputs, targets=pr_targets)
      features['pr'] = outputs['pr']['pr_out'].detach().cpu()

      # Compute PR Mismatch
      pr_out = outputs['pr']['pr_out']
      pr_batch_size = pr_out.shape[0]
      losses['pr_mismatch'] = torch.sum(torch.abs(pr_targets - pr_out)) / pr_batch_size

      pc_cue = outputs['ps'] if self.training else outputs['pr']['z_cue']

    outputs['pc'] = self.forward_pc(inputs=pc_cue)

    losses['pm'], outputs['pm'] = self.forward_ae(name='pm', inputs=outputs['pc'], targets=targets)
    losses['pm_ec'], outputs['pm_ec'] = self.forward_ae(name='pm_ec', inputs=outputs['pc'], targets=targets)

    outputs['encoding'] = outputs['pc'].detach()
    outputs['decoding'] = outputs['pm']['decoding'].detach()
    outputs['decoding_ec'] = outputs['pm_ec']['decoding'].detach()
    outputs['output'] = outputs['pc'].detach()

    features['pc'] = outputs['pc'].detach().cpu()
    features['recon'] = outputs['pm']['decoding'].detach().cpu()
    features['recon_ec'] =  outputs['pm_ec']['decoding'].detach().cpu()

    self.features = features

    return losses, outputs
