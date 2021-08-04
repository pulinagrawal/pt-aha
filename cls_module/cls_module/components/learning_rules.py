import logging
from abc import ABC, abstractmethod

import torch

class LearningRule(ABC):
  def __init__(self):
    self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

  @abstractmethod
  def compute_dw(self, inputs, targets, weights):
    """
      Compute the delta of the weights (dw), based on some rule,
      using the given inputs and targets.
    """
    pass


class OjaLeabraRule(LearningRule):
  def compute_dw(self, inputs, targets, weights):
    inputs_tiled = inputs.unsqueeze(-1)
    inputs_tiled = inputs_tiled.repeat((1, 1, targets.size(1))).transpose(2, 1)

    targets_tiled = targets.unsqueeze(-1)
    targets_tiled = targets_tiled.repeat((1, 1, inputs.size(1)))

    # Expand dimension of weights and tile to match corresponding samples in
    # the given batch of inputs/targets
    weights_tiled = weights.unsqueeze(0)
    weights_tiled = weights_tiled.repeat((inputs.size(0), 1, 1))

    # This rule from Leabra is a variation on Oja's rule
    d_ws = targets_tiled * (inputs_tiled - weights_tiled)

    return torch.mean(d_ws, dim=0)


class LeabraRule(LearningRule):
  def compute_dw(self, inputs, targets, weights):
    # Two-stage learning
    #   Contrastive Hebbian Learning (CHL)
    #     O’Reilly RC (1996) Biologically plausible error-driven learning using local activation differences:
    #     The generalized recirculation algorithm. Neural Compu- tation 8: 895–938.
    #   Pure hebbian learning (see OjaLeabraRule)
    #   proportionally weighted (lmix), controlled by eps
    # dCHL_ij = [x_i+ * y_j+ - x_i- * y_j]
    # dHebb_ij = y_i+[x_i+ * y_j+ - x_i- * y_j]
    # dw_ij = eps[ lmix(dHebb) + (1-lmix)(dCHL) ]
    pass
