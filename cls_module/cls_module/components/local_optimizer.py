from torch.optim.optimizer import Optimizer, required

class LocalOptim(Optimizer):
  def __init__(self, named_params, lr=required):
    self.param_names, params = zip(*named_params)

    if lr is not required and lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))

    defaults = dict(lr=lr)
    super(LocalOptim, self).__init__(params, defaults)

  def local_step(self, param_delta):
    """Performs a single local optimization step."""
    for group in self.param_groups:
      layer_index = self.param_names.index('weight')
      layer = group['params'][layer_index]

      layer.data.add_(group['lr'] * param_delta)

    try:
      self._step_count += 1
    except AttributeError:
      pass
