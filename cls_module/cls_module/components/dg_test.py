"""dg_test.py"""

import torch

from cls_module.components.dg import DG

config = {
    "inhibition_decay": 0.95,
    "knockout_rate": 0.25,
    "init_scale": 10.0,
    "num_units": 200,
    "sparsity": 10,
    "use_stub": False
}

# rule of thumb
# num_units = batch_size * sparsity
# control using sparsity and inhibition decay

x = torch.rand(20, 121, 5, 5)

torch.set_printoptions(threshold=5000)

dg = DG(x.shape, config)
out = dg(x)

# Verify sparsity
test_out = torch.sum(out, dim=1)
expected_out = torch.ones_like(test_out) * config['sparsity']
torch.testing.assert_allclose(test_out, expected_out)

# Verify min per sample
test_out, _ = torch.min(out, dim=1)
expected_out = torch.zeros_like(test_out)
torch.testing.assert_allclose(test_out, expected_out)

# Verify max per sample
test_out, _ = torch.max(out, dim=1)
expected_out = torch.ones_like(test_out)
torch.testing.assert_allclose(test_out, expected_out)

dg_overlap = dg.compute_overlap(out)

print('Overlap (per sample) =', dg_overlap)
print('Overlap (sum) =', dg_overlap.sum().item())
