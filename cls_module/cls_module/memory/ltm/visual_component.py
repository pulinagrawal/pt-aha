"""VisualComponent class."""

import torch
import torch.optim as optim
import torch.nn.functional as F

from cerenaut_pt_core import utils

from cls_module.memory.interface import MemoryInterface
from cerenaut_pt_core.components.sparse_autoencoder import SparseAutoencoder
# from torchsummary import summary

class VisualComponent(MemoryInterface):
  """An implementation of a long-term memory module using sparse convolutional autoencoder."""

  global_key = 'ltm'
  local_key = 'vc'

  def build(self):
    """Build Visual Component as long-term memory module."""
    vc = SparseAutoencoder(self.input_shape, self.config).to(self.device)
    vc_optimizer = optim.AdamW(vc.parameters(), lr=self.config['learning_rate'])

    # print("\n Model Summary : ", summary(vc))
    self.add_module(self.local_key, vc)
    self.add_optimizer(self.local_key, vc_optimizer)

    # Compute expected output shape
    with torch.no_grad():
      print("\n\nhi I am in VC code\n\n")
      stride = self.config.get('eval_stride', self.config.get('stride'))
      sample_input = torch.rand(1, *(self.input_shape[1:])).to(self.device)
      print ("\nsample_input check in org visual_comp: ", sample_input)
      print ("\nsample_input.shape check in org visual_comp: ", sample_input.shape)
      sample_output = vc.encode(sample_input, stride=stride)
      print ("\n sample_output.shape check in org visual_comp after vc.encode: ", sample_output.shape)
      print ("\n\n ")
      sample_output = self.prepare_encoding(sample_output)
      print ("\n sample_output.shape check in org visual_comp after prepare_encoding: ", sample_output.shape)
      self.output_shape = list(sample_output.data.shape)
      print ("\n self.output_shape check in org visual_comp: ", list(sample_output.data.shape))
      self.output_shape[0] = -1

    if 'classifier' in self.config:
      self.build_classifier(input_shape=self.output_shape)

  def forward_decode(self, encoding):
   # Optionally use different stride at test time
    stride = self.config['stride']
    if not self.vc.training and 'eval_stride' in self.config:
      stride = self.config['eval_stride']

    encoding = self.unprepare_encoding(encoding)

    with torch.no_grad():
      return self.vc.decode(encoding, stride)

  def forward_memory(self, inputs, targets, labels):
    """Perform an optimization step using the memory module."""
    del labels

    if self.vc.training:
      self.vc_optimizer.zero_grad()

    # Optionally use different stride at test time
    stride = self.config['stride']
    if not self.vc.training and 'eval_stride' in self.config:
      stride = self.config['eval_stride']

    encoding, decoding = self.vc(inputs, stride)
    loss = F.mse_loss(decoding, targets)

    if self.vc.training:
      loss.backward()
      self.vc_optimizer.step()

    output_encoding = self.prepare_encoding(encoding)

    outputs = {
        'encoding': encoding,
        'decoding': decoding,

        'output': output_encoding  # Designated output for linked modules
    }

    self.features = {
        'vc': output_encoding.detach().cpu(),
        'recon': decoding.detach().cpu()
    }

    return loss, outputs

  def prepare_encoding(self, encoding):
    """Postprocessing for the VC encoding."""
    encoding = encoding.detach()

    if self.config['output_pool_size'] > 1:
      encoding, self.pool_indices = F.max_pool2d(
          encoding,
          kernel_size=self.config['output_pool_size'],
          stride=self.config['output_pool_stride'],
          padding=self.config.get('output_pool_padding', 0), return_indices=True)

    return encoding

  def unprepare_encoding(self, prepared_encoding):
    """Undo any postprocessing for the VC encoding."""
    encoding = prepared_encoding.detach()

    if self.config['output_pool_size'] > 1:
      encoding = F.max_unpool2d(
          encoding,
          kernel_size=self.config['output_pool_size'],
          stride=self.config['output_pool_stride'],
          padding=self.config.get('output_pool_padding', 0),
          indices=self.pool_indices)

    return encoding