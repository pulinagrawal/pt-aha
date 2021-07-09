import logging
from abc import ABC, abstractmethod

import torch

class LearningRule(ABC):
  def __init__(self):
    self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

  def init_layers(self, model):
    pass

  @abstractmethod
  def update(self, x, w):
    pass


class OjasRule(LearningRule):
  def __init__(self, c=0.1):
    super().__init__()
    self.c = c

  def update(self, inputs, w):
    d_ws = torch.zeros(inputs.size(0), *w.shape)
    for idx, x in enumerate(inputs):
        y = torch.mm(w, x.unsqueeze(1))

        d_w = torch.zeros(w.shape)
        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                d_w[i, j] = self.c * y[i] * (x[j] - y[i] * w[i, j])

        d_ws[idx] = d_w

    return torch.mean(d_ws, dim=0)

class PureHebbRule(LearningRule):
  def __init__(self):
    super().__init__()

  def update(self, inputs, targets, w):
    inputs_tiled = inputs.unsqueeze(-1)
    inputs_tiled = inputs_tiled.repeat((1, 1, targets.size(1))).transpose(2, 1)

    targets_tiled = targets.unsqueeze(-1)
    targets_tiled = targets_tiled.repeat((1, 1, inputs.size(1)))

    w_tiled = w.unsqueeze(0)
    w_tiled = w_tiled.repeat((inputs.size(0), 1, 1))

    d_ws = targets_tiled * (inputs_tiled - w_tiled)

    return torch.mean(d_ws, dim=0)


class LeabraRule(LearningRule):
  def __init__(self, eps=0.1, lmix=0.1):
    super().__init__()
    self.eps = eps
    self.lmix = lmix

  def update(self, inputs, w):
    pass
    # two-stage learning
    #   Contrastive Hebbian Learning (CHL)
    #     O’Reilly RC (1996) Biologically plausible error-driven learning using local activation differences:
    #     The generalized recirculation algorithm. Neural Compu- tation 8: 895–938.
    #   Pure hebbian learning
    #   proportionally weighted (lmix), controlled by eps
    # dCHL_ij = [x_i+ * y_j+ - x_i- * y_j]
    # dHebb_ij = y_i+[x_i+ * y_j+ - x_i- * y_j]
    # dw_ij = eps[ lmix(dHebb) + (1-lmix)(dCHL) ]
