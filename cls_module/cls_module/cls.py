"""Complementary Learning System module."""

import os
import collections
import logging
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import numpy as np

import cls_module
from cerenaut_pt_core.utils import square_image_shape_from_1d
from .memory import ltm, stm, ec


class CLS(nn.Module):
  """Complementary Learning System module."""

  ltm_key = 'ltm'
  ec_key = 'ec'
  stm_key = 'stm'
  decoder_key = 'decoder'

  def __init__(self, input_shape, config, device=None, writer=None, output_shape=None):
    super(CLS, self).__init__()
    print ("\n\nself.writer = writer: ", writer)
    self.config = config
    self.writer = writer
    self.device = device
    self.input_shape = input_shape
    self.output_shape = output_shape

    self.step = {}
    self.previous_mode = None

    if self.device is None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if output_shape is None:
      self.output_shape = self.input_shape

    self.build()

    self.features = {}
    self.stm_modes = ['study', 'study_validate', 'recall']

    for mode in self.stm_modes:
      self.features[mode] = {}

    self.initial_state = self.state_dict()

  def build(self):
    """Build and initialize the long-term and short-term memory modules."""

    # 1) _________ ltm ____________________________________
    ltm_config = self.config[self.ltm_key]
    ltm_type = self.config['ltm_type']

    if ltm_type == 'vc':
      ltm_class = ltm.VisualComponent
    elif ltm_type == 'vc_bvae':
      ltm_class = ltm.VisualComponentBVAE
    elif ltm_type == 'vgg':
      ltm_class = ltm.VGG
    else:
      raise NotImplementedError('LTM type not supported: ' + ltm_type)

    ltm_ = ltm_class(config=ltm_config,
                     input_shape=self.input_shape,
                     target_shape=None,
                     device=self.device,
                     writer=self.writer)

    print ("\n\nltm_.output_shape in cls.py ltm", ltm_.output_shape)
    next_input = ltm_.output_shape
    self.add_module(self.ltm_key, ltm_)

    # 2) _________ ec ____________________________________

    ec_type = self.config.get('ec_type', 'none')
    if ec_type != 'none':
      ec_config = self.config[self.ec_key]

      if ec_type == 'bvae':
        ec_class = ec.BetaVAE
        logging.info("Creating a BetaVAE Entorhinal Cortex")
      else:
        raise NotImplementedError('EC type not supported: ' + ec_type)

      ec_ = ec_class(config=ec_config,
                     input_shape=ltm_.output_shape,
                     device=self.device,
                     writer=self.writer)

      next_input = ec_.output_shape
      self.add_module(self.ec_key, ec_)

    # 3) _________ stm ____________________________________
    stm_type = self.config['stm_type']
    stm_config = self.config[self.stm_key]

    if stm_type == 'fastnn':
      stm_class = stm.FastNN
    elif stm_type == 'aha':
      stm_class = stm.AHA
    else:
      raise NotImplementedError('STM type not supported: ' + stm_type)

    stm_ = stm_class(config=stm_config,
                     input_shape=next_input,
                     target_shape=self.input_shape,
                     device=self.device,
                     writer=self.writer)

    self.add_module(self.stm_key, stm_)

    # 4) _________ decoder ____________________________________
    if 'decoder' in self.config:
      decoder_config = self.config['decoder']
      decoder_layers = []
      decoder_input_size = np.prod(list(next_input)[1:])
      decoder_hidden_size = decoder_config['units'][0]
      decoder_output_size = np.prod(list(self.output_shape)[1:])

      for i in range(decoder_config['num_layers']):
        decoder_layers.extend([
          ('layer_' + str(i), nn.Linear(decoder_input_size, decoder_hidden_size)),
          ('act_' + str(i), nn.LeakyReLU())
        ])

        if i < (decoder_config['num_layers'] - 1):
          decoder_input_size = decoder_hidden_size
          decoder_hidden_size = decoder_config['units'][i + 1]

      decoder_layers.extend([
        ('output_layer', nn.Linear(decoder_hidden_size, decoder_output_size)),
        ('output_act', nn.LeakyReLU())
      ])

      print(decoder_layers)

      self.decoder = nn.Sequential(OrderedDict(decoder_layers))
      self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),
                                                lr=decoder_config['learning_rate'],
                                                weight_decay=decoder_config['weight_decay'])

  def reset(self, names=None):
    """Reset relevant sub-modules."""
    if names is None:
      names = ['ltm.classifier', 'stm']

    if 'ltm.classifier' in names:
      ltm = self._modules[self.ltm_key]
      if hasattr(ltm, 'classifier'):  # if classifier is a separate module, it can be reset
        ltm.classifier.reset()

    if 'stm' in names:
      stm = self._modules[self.stm_key]
      stm.reset()

      if hasattr(stm, 'classifier'):
        stm.classifier.reset()

  def freeze(self, names):
    """Selectively freeze sub-modules."""
    depth = 2

    for name in names:
      name = name.split('.', maxsplit=depth)
      parent_module = getattr(self, name[0], None)

      if parent_module is None:
        continue

      if len(name) == 1:
        parent_module.train(False)
        continue

      if len(name) > 1:
        for child_name, child_module in parent_module.named_modules():
          if child_name == name[1]:
            child_module.train(False)

  def forward_decoder(self, inputs, targets, training=None):
    if training:
      self.decoder_optimizer.zero_grad()

    output_shape = targets.shape

    inputs = torch.flatten(inputs, start_dim=1)
    targets = torch.flatten(targets, start_dim=1)

    if self.config['decoder'].get('norm_inputs'):
      inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())

    decoding = self.decoder(inputs)
    loss = F.mse_loss(decoding, targets)

    decoding_reshape = decoding.reshape(output_shape)

    outputs = {
      'ltm': {
        'decoding': decoding_reshape.detach(),
        'output': decoding_reshape.detach()  # Designated output for linked modules
      }
    }

    if training:
      loss.backward()
      self.decoder_optimizer.step()

    losses = {
      'ltm': {
        'loss': {
          'decoder': loss,
        }
      }
    }

    return losses, outputs

  def load_state_dict(self, state_dict, strict=False):
    # TODO(@abdel): Why did we do this? This means that we can never re-load
    # any other sub modules, other than something that starts with LTM.

    # modified_state_dict = {}

    # for state_key in state_dict:
    #   if state_key.startswith(self.ltm_key) and state_key in state_dict:
    #     modified_state_dict[state_key] = state_dict[state_key]
    #     continue

    #   modified_state_dict[state_key] = self.initial_state[state_key]

    # super().load_state_dict(modified_state_dict, strict)

    super().load_state_dict(state_dict, strict)

    # Load pre-trained VC weights from TensorFlow implementation
    load_tf_vc_weights = False

    if load_tf_vc_weights:
      module_dirpath = os.path.dirname(cls_module.__file__)
      weights_filepath = os.path.join(module_dirpath, '..', 'vc_weights.npz')

      vc_params = np.load(weights_filepath)
      vc_encoder_weight = torch.from_numpy(vc_params['weights']).permute(3, 2, 0, 1)
      vc_encoder_bias = torch.from_numpy(vc_params['encoding_bias'])

      self.ltm.vc.encoder.weight.data = vc_encoder_weight.to(self.device)
      self.ltm.vc.encoder.bias.data = vc_encoder_bias.to(self.device)

  def pretrain(self, inputs, labels):
    return self.forward(inputs, labels, mode='pretrain')

  def validate(self, inputs, labels):
    return self.forward(inputs, labels, mode='validate')

  def memorise(self, inputs, labels):
    return self.forward(inputs, labels, mode='study')

  def memorise_test(self, inputs, labels):
    return self.forward(inputs, labels, mode='study_validate')

  def recall(self, inputs, labels):
    return self.forward(inputs, labels, mode='recall')

  def consolidate(self, inputs, labels):
    return self.forward(inputs, labels, mode='consolidate')

  def forward(self, inputs, labels=None, mode=None, ec_inputs=None, paired_inputs=None):  # pylint: disable=arguments-differ
    """Perform an optimisation step with the entire CLS module."""
    if labels is not None and not isinstance(labels, torch.Tensor):
      labels = torch.from_numpy(labels).to(self.device)

    if mode not in self.step:
      self.step[mode] = 0

    losses = {}
    outputs = {}
    accuracies = {}

    # Freeze ALL except LTM feature extractor for pretraining
    if mode == 'pretrain':
      self.train()
      self.freeze([self.ltm_key + '.classifier', self.stm_key])

    # Freeze ALL except LTM feature extractor during validation
    elif mode == 'validate' or mode == 'recall':
      self.eval()

    # Freeze LTM during memorisation
    elif mode == 'study':
      self.train()
      self.freeze([self.ltm_key])

    # Freeze ALL during recall
    elif mode == 'recall':
      self.eval()

    # Freeze ALL except LTM classifier for consolidation
    elif mode == 'consolidate':
      self.train()
      self.freeze([self.ltm_key + '.vc', self.stm_key])

    else:
      raise NotImplementedError('Mode not supported.')

    if ec_inputs == None:
      losses[self.ltm_key], outputs[self.ltm_key] = self.ltm(inputs=inputs, targets=inputs, labels=labels)
      preds = outputs[self.ltm_key]['classifier']['predictions']

      if preds is not None:
        softmax_preds = F.softmax(preds, dim=1).argmax(dim=1)
        accuracies[self.ltm_key] = torch.eq(softmax_preds, labels).data.cpu().float().mean()

    if mode in ['study', 'recall']:

      if ec_inputs == None:
        next_input = outputs[self.ltm_key]['memory']['output'].detach()  # Ensures no gradients pass through modules
      else:
        next_input = ec_inputs

      # iterate EC
      if self.ec_key in self._modules:
        outputs[self.ec_key] = self.ec(inputs=next_input)
        next_input = outputs[self.ec_key].detach()  # Ensures no gradients pass through modules

      # iterate STM
      losses[self.stm_key], outputs[self.stm_key] = self.stm(inputs=next_input, targets=inputs, labels=labels)

      # iterate decoder
      if self.decoder_key in self._modules:
        decoder_targets = inputs

        if ec_inputs is not None and paired_inputs is not None:
          decoder_targets = paired_inputs

        losses['decoder'], outputs['decoder'] = self.forward_decoder(inputs=next_input, targets=decoder_targets,
                                                                      training=self.training)

        # Decode output from STM via the EC <=> LTM
        if outputs[self.stm_key]['memory']['decoding'] is None:
          output_decoding = outputs[self.stm_key]['memory']['decoding_ec']

          with torch.no_grad():
            _, decoder_outputs = self.forward_decoder(inputs=output_decoding, targets=decoder_targets, training=False)
            output_decoding = decoder_outputs['ltm']['decoding']

          outputs[self.stm_key]['memory']['decoding'] = output_decoding.detach()
          self.stm.features['recon'] = output_decoding.detach().cpu()

      preds = outputs[self.stm_key]['classifier']['predictions']

      if preds is not None:
        softmax_preds = F.softmax(preds, dim=1).argmax(dim=1)
        accuracies[self.stm_key] = torch.eq(softmax_preds, labels).data.cpu().float().mean()

      self.features[mode]['inputs'] = inputs.detach().cpu()
      self.features[mode]['labels'] = labels.detach().cpu()

      if paired_inputs is not None:
        self.features[mode]['paired_inputs'] = paired_inputs.detach().cpu()

      for key, value in self.stm.features.items():
        self.features[mode][self.stm_key + '_' + key] = value

      for key, value in self.ltm.features.items():
        self.features[mode][self.ltm_key + '_' + key] = value

    # Add summaries to TensorBoard
    if self.writer:
      summary_step = self.step[mode]

      self.write_loss_summary(self.writer, losses, mode, summary_step)
      self.write_accuracy_summary(self.writer, accuracies, mode, summary_step)
      self.write_output_summaries(self.writer, outputs, mode, summary_step)

      if mode != "study" and mode != "recall":
        self.writer.add_image(mode + '/inputs', torchvision.utils.make_grid(inputs), summary_step)

      if paired_inputs is not None:
        self.writer.add_image(mode + '/paired_inputs', torchvision.utils.make_grid(paired_inputs), summary_step)

      self.writer.flush()

    self.step[mode] += 1

    return losses, outputs

  def is_ltm_supervised(self):
    ltm_type = self.config['ltm_type']
    if ltm_type == 'vc':
      return False
    if ltm_type == 'vc_bvae':
      return False  
    elif ltm_type == 'vgg':
      return True
    else:
      return False

  def write_loss_summary(self, writer, losses, mode, summary_step):
    for module_name in losses:
      for submodule_name in losses[module_name]:
        for metric_key, metric_value in losses[module_name][submodule_name].items():
          scope = mode + '/' + module_name + '/' + submodule_name + '/' + metric_key

          if isinstance(metric_value, dict):
            for submetric_key, submetric_value in metric_value.items():
              writer.add_scalar(scope + '/' + submetric_key, submetric_value, summary_step)
          else:
            writer.add_scalar(scope, metric_value, summary_step)

  def write_accuracy_summary(self, writer, accuracies, mode, summary_step):
    for module_name, accuracy_value in accuracies.items():
      writer.add_scalar(mode + '/' + module_name + '/classifier/accuracy', accuracy_value, summary_step)

  def write_output_summaries(self, writer, outputs, mode, summary_step):
    for module_name in outputs:
      for submodule_name in outputs[module_name]:
        for metric_key, metric_value in outputs[module_name][submodule_name].items():
          summary_image = None

          if metric_key == 'decoding':
            summary_image = torchvision.utils.make_grid(metric_value, normalize=True, scale_each=True)

          elif metric_key == 'encoding' or metric_key == 'decoding_ec':
            square_image_shape, _ = square_image_shape_from_1d(np.prod(metric_value.shape[1:]))
            summary_image = torch.reshape(metric_value, [-1, 1, square_image_shape[1], square_image_shape[2]])
            summary_image = summary_image[0]

          if summary_image is not None:
            writer.add_image(mode + '/' + module_name + '/' + submodule_name + '/' + metric_key,
                             summary_image,
                             summary_step)
