"""AHA class."""

import logging
from collections import defaultdict
from os import pathconf
from cls_module.memory.stm.aha.msp import MonosynapticPathway
from cls_module.memory.stm.aha.perforant_hebb import PerforantHebb
from cls_module.memory.stm.aha.pm import PatternMapper

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cls_module.memory.interface import MemoryInterface

from cls_module.components.dg import DG
from cls_module.components.knn_buffer import KNNBuffer
from cls_module.memory.stm.aha.perforant_pr import PerforantPR
from cls_module.memory.stm.aha.perforant_hebb import PerforantHebb

from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder


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
    return self.config.get('hebbian_perforant', False)

  def reset(self):
    """Reset modules and optimizers."""
    for _, module in self.named_children():
      if hasattr(module, 'reset'):
        module.reset()

  def build(self):
    """Build AHA as short-term memory module."""

    # Build the Dentae Gyrus
    self.dg = DG(self.input_shape, self.config['dg']).to(self.device)
    dg_output_shape = [1, self.config['dg']['num_units']]

    # Build the Perforant Pathway
    if self.is_hebbian_perforant():
      self.perforant = PerforantHebb(ec_shape=self.input_shape,
                                          dg_shape=dg_output_shape,
                                          ca3_shape=dg_output_shape,
                                          config=self.config['perforant_hebb'])
    else:
      self.perforant = PerforantPR(self.input_shape, dg_output_shape, self.config['perforant_pr'])

    # Build the CA3
    self.ca3 = KNNBuffer(input_shape=dg_output_shape, target_shape=dg_output_shape, config=self.config['ca3'])
    self.stored_ca3_cue = None
    ca3_output_shape = dg_output_shape

    # Build the Monosynaptic Pathway
    # Optionally between a bioligically plausible MSP (CA1, CA3 => CA1 pathways) or a simple pattern mapper
    if self.config.get('msp_type', None) == 'ca1':
      self.msp = MonosynapticPathway(ca3_shape=ca3_output_shape, ec_shape=self.input_shape, config=self.config['msp'])
    else:
      self.pm = PatternMapper(ca3_output_shape, self.target_shape, self.config['pm'])
      self.pm_ec = PatternMapper(ca3_output_shape, self.input_shape, self.config['pm_ec'])

    # Build the Label Learner module
    if 'classifier' in self.config:
      self.build_classifier(input_shape=ca3_output_shape)

    self.output_shape = ca3_output_shape

  def forward_memory(self, inputs, targets, labels):
    """Perform one step using the entire system (i.e. all sub-modules of AHA)."""
    del labels

    losses = {}
    outputs = {}
    features = {}

    norm_dims = list(range(inputs.dim()))
    norm_dims = norm_dims[1:]
    normed_inputs = inputs
    frobenius_norm = torch.sqrt(
        torch.sum(torch.square(inputs),
                  dim=norm_dims,
                  keepdim=True)
    )

    normed_inputs = normed_inputs / frobenius_norm

    outputs['ec_in'] = inputs

    outputs['dg'] = self.dg(normed_inputs)
    features['dg'] = outputs['dg'].detach().cpu()

    # Compute DG Overlap
    overlap = self.dg.compute_overlap(outputs['dg'])
    losses['dg_overlap'] = overlap.sum()

    unique_overlap = self.dg.compute_unique_overlap(normed_inputs, outputs['dg'])
    losses['dg_unique_overlap'] = unique_overlap.sum()

    # Perforant Pathway: Hebbian Learning
    if self.is_hebbian_perforant():
      ca3_cue, losses['dg_ca3'], losses['ec_ca3'], losses['ca3_cue'] = self.perforant(ec_inputs=normed_inputs, dg_inputs=outputs['dg'])

      outputs['ec_ca3'] = {
        'ca3_cue': ca3_cue.detach().cpu()
      }

      features['ca3_cue'] = ca3_cue.detach().cpu()

    # Perforant Pathway: Error-Driven Learning
    else:
      pr_targets = outputs['dg'] if self.training else self.stored_ca3_cue
      losses['pr'], outputs['pr'] = self.perforant(inputs=normed_inputs, targets=pr_targets)
      features['pr'] = outputs['pr']['pr_out'].detach().cpu()

      # Compute PR Mismatch
      pr_out = outputs['pr']['pr_out']
      pr_batch_size = pr_out.shape[0]
      losses['pr_mismatch'] = torch.sum(torch.abs(pr_targets - pr_out)) / pr_batch_size

      # Store the cue to the CA3 to compare during recall
      if self.training:
        self.stored_ca3_cue = outputs['dg']

      ca3_cue = outputs['dg'] if self.training else outputs['pr']['z_cue']

    # CA3
    outputs['ca3'] = self.ca3(inputs=ca3_cue)
    features['ca3'] = outputs['ca3'].detach().cpu()

    outputs['encoding'] = outputs['ca3'].detach()
    outputs['output'] = outputs['ca3'].detach()

    # Monosynaptic Pathway
    if self.config.get('msp_type', None) == 'ca1':
      # During study, EC will drive the CA1
      losses['ca1'], outputs['ca1'], losses['ca3_ca1'], outputs['ca3_ca1'] = self.msp(ec_inputs=inputs,
                                                                                      ca3_inputs=outputs['ca3'])

      outputs['decoding'] = None
      features['recon'] = None

      outputs['decoding_ec'] = outputs['ca1']['decoding'].detach()
      features['recon_ec'] =  outputs['ca1']['decoding'].detach().cpu()
    else:
      losses['pm'], outputs['pm'] = self.pm(inputs=outputs['ca3'], targets=targets)
      losses['pm_ec'], outputs['pm_ec'] = self.pm_ec(inputs=outputs['ca3'], targets=inputs)

      outputs['decoding'] = outputs['pm']['decoding'].detach()
      features['recon'] = outputs['pm']['decoding'].detach().cpu()

      outputs['decoding_ec'] = outputs['pm_ec']['decoding'].detach()
      features['recon_ec'] =  outputs['pm_ec']['decoding'].detach().cpu()

    self.features = features

    return losses, outputs
