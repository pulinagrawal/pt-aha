"""DentateGyrus class."""

import torch
import torch.nn as nn

import numpy as np

import cerenaut_pt_core.utils as utils


class DG(nn.Module):
  """
  A non-trainable module based on Dentate Gyrus (DG), produces sparse outputs and inhibits neurons after firing.
  """

  def __init__(self, input_shape, config):
    super(DG, self).__init__()

    self.config = config

    self.input_shape = list(input_shape)
    self.input_size = np.prod(self.input_shape[1:])

    self.layer = nn.Linear(self.input_size, self.config['num_units'], bias=False)
    self.layer.weight.requires_grad = False

    self.reset_parameters()

  def reset_parameters(self):
    """Custom initialization *does* make a big difference to orthogonality, even with inhibition"""
    input_area = self.input_size
    hidden_size = self.config['num_units']

    num_weights = hidden_size * input_area
    random_values = torch.rand(num_weights, dtype=torch.float32)

    knockout_rate = self.config['knockout_rate']
    keep_rate = 1.0 - knockout_rate

    initial_mask = np.random.choice(np.array([0, 1], dtype=np.float32),
                                    size=(num_weights),
                                    p=[knockout_rate, keep_rate])

    initial_values = random_values * initial_mask * self.config['init_scale']
    initial_values = torch.reshape(initial_values, shape=(hidden_size, input_area))

    abs_sum = torch.sum(torch.abs(initial_values), dim=1, keepdim=True)
    norm = 1.0 / abs_sum

    initial_values = initial_values * norm

    self.layer.weight.data = initial_values

  def apply_sparse_filter_uniuqe(self, encoding):
    """Sparse filtering with inhibition."""
    hidden_size = self.config['num_units']

    unique_encoding, unique_idxs = self.find_unique_samples(encoding)

    batch_size = encoding.shape[0]
    unique_batch_size = unique_encoding.shape[0]

    k = int(self.config['sparsity'])
    inhibition_decay = self.config['inhibition_decay']

    cells_shape = [hidden_size]
    batch_cells_shape = [batch_size, hidden_size]
    unique_batch_cells_shape = [unique_batch_size, hidden_size]

    device = unique_encoding.device

    inhibition = torch.zeros(cells_shape, device=device)
    filtered_unique = torch.zeros(unique_batch_cells_shape, device=device)

    # Inhibit over time within a batch (because we don't bother having repeats for this).
    for i in range(0, unique_batch_size):
      # Create a mask with a 1 for this batch only
      this_batch_mask = torch.zeros([unique_batch_size, 1], device=device)
      this_batch_mask[i][0] = 1.0

      refraction = 1.0 - inhibition
      refraction_2d = refraction.unsqueeze(0)  # add batch dim
      refracted = torch.abs(unique_encoding) * refraction_2d

      # Find the "winners". The top k elements in each batch sample
      # ---------------------------------------------------------------------
      top_k_mask = utils.build_topk_mask(refracted, dim=-1, k=k)

      # Retrospectively add batch-sparsity per cell: pick the top-k
      # ---------------------------------------------------------------------
      batch_filtered = unique_encoding * top_k_mask  # apply mask 3 to output 2
      this_batch_filtered = batch_filtered * this_batch_mask
      this_batch_topk = top_k_mask * this_batch_mask
      fired, _ = torch.max(this_batch_topk, dim=0)  # reduce over batch

      inhibition = inhibition * inhibition_decay + fired  # set to 1

      filtered_unique = filtered_unique + this_batch_filtered

    filtered = torch.zeros(batch_cells_shape, device=device)

    for i in range(0, batch_size):
      for j in range(0, unique_batch_size):
        if not torch.equal(encoding[i], unique_encoding[j]):
          continue

        filtered[i] = filtered_unique[j]

    return filtered, inhibition

  def apply_sparse_filter(self, encoding):
    """Sparse filtering with inhibition."""
    hidden_size = self.config['num_units']
    batch_size = encoding.shape[0]

    k = int(self.config['sparsity'])
    inhibition_decay = self.config['inhibition_decay']

    cells_shape = [hidden_size]
    batch_cells_shape = [batch_size, hidden_size]

    device = encoding.device

    inhibition = torch.zeros(cells_shape, device=device)
    filtered = torch.zeros(batch_cells_shape, device=device)

    # Inhibit over time within a batch (because we don't bother having repeats for this).
    for i in range(0, batch_size):
      # Create a mask with a 1 for this batch only
      this_batch_mask = torch.zeros([batch_size, 1], device=device)
      this_batch_mask[i][0] = 1.0

      refraction = 1.0 - inhibition
      refraction_2d = refraction.unsqueeze(0)  # add batch dim
      refracted = torch.abs(encoding) * refraction_2d

      # Find the "winners". The top k elements in each batch sample
      # ---------------------------------------------------------------------
      top_k_mask = utils.build_topk_mask(refracted, dim=-1, k=k)

      # Retrospectively add batch-sparsity per cell: pick the top-k
      # ---------------------------------------------------------------------
      batch_filtered = encoding * top_k_mask  # apply mask 3 to output 2
      this_batch_filtered = batch_filtered * this_batch_mask
      this_batch_topk = top_k_mask * this_batch_mask
      fired, _ = torch.max(this_batch_topk, dim=0)  # reduce over batch

      inhibition = inhibition * inhibition_decay + fired  # set to 1

      filtered = filtered + this_batch_filtered

    return filtered, inhibition

  def stub(self, x):
    """Returns a batch of non overlapping n-hot samples in range [0, 1]."""
    batch_size = x.shape[0]
    hidden_size = self.config['num_units']
    n = self.config['sparsity']

    assert ((batch_size * n - 1) + n) < hidden_size, "Can't produce batch_size {0} non-overlapping samples, " \
           "reduce n {1} or increase sample_size {2}".format(batch_size, n, hidden_size)

    batch = torch.zeros(batch_size, hidden_size).to(x.device)

    # Return the sample at given index
    for idx in range(batch_size):
      start_idx = idx * n
      end_idx = start_idx + n
      batch[idx][start_idx:end_idx] = 1

    return batch

  def compute_overlap(self, encoding):
    """ a, b = (0,1)
    Overlap is where tensors have 1's in the same position.
    Return number of bits that overlap """

    num_samples = encoding.shape[0]
    batch_overlap = torch.zeros(num_samples).to(encoding.device)

    for i in range(num_samples):
      a = encoding[i]

      for j in range(num_samples):
        # Avoid matching the same samples
        if i == j:
          continue

        b = encoding[j]
        overlap = torch.sum(a * b)
        batch_overlap[i] += overlap

    return batch_overlap

  def find_unique_samples(self, x):
    unique_samples = torch.unique(x, dim=0)

    uniuqe_idxs = torch.zeros(unique_samples.shape[0], dtype=torch.long)
    for i, sample in enumerate(unique_samples):
      idx = (sample == x).all(1).nonzero(as_tuple=True)
      uniuqe_idxs[i] = idx[0][0]

    return unique_samples, uniuqe_idxs

  def compute_unique_overlap(self, inputs, encoding):
    """ a, b = (0,1)
    Overlap is where tensors have 1's in the same position.
    Return number of bits that overlap """

    unique_samples, uniuqe_idxs = self.find_unique_samples(inputs)
    # print('==== Found', unique_samples.shape[0], 'uniques out of', inputs.shape[0], 'samples')

    unique_encoding = encoding[uniuqe_idxs]
    unique_overlap = self.compute_overlap(unique_encoding)

    return unique_overlap

  def forward(self, inputs):  # pylint: disable=arguments-differ
    inputs = torch.flatten(inputs, start_dim=1)

    # Ensure inputs are in [0.. 1] range
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())

    if self.config['use_stub']:
      top_k_mask = self.stub(inputs)

    else:
      with torch.no_grad():
        encoding = self.layer(inputs)

        if self.config.get('unique_mode'):
          filtered_encoding, _ = self.apply_sparse_filter_uniuqe(encoding)
        else:
          filtered_encoding, _ = self.apply_sparse_filter(encoding)

      # Override encoding to become binary mask
      top_k_mask = utils.build_topk_mask(filtered_encoding, dim=-1, k=self.config['sparsity'])

    # overlap = self.compute_overlap(top_k_mask)
    # overlap_sum = overlap.sum().item()

    # assert overlap_sum == 0.0, 'Found overlap between samples in batch'

    return top_k_mask.detach()
