"""dg_test.py"""

import torch

from cls_module.components.dg import DG

torch.set_printoptions(threshold=5000)

config = {
    "inhibition_decay": 0.95,
    "knockout_rate": 0.25,
    "init_scale": 10.0,
    "num_units": 200,
    "sparsity": 10,
    "use_stub": False
}

# Rule of thumb
# ==========================================================
# * num_units = batch_size * sparsity
# * Control overlap using `sparsity` and `inhibition_decay``

# Notes
# ==========================================================
# Overlap (lower bound) = 0.0
# Overlap (upper bound) = (num_samples * num_samples - 1) * sparsity

x = torch.rand(20, 121, 5, 5)

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

# Compute overlaps
dg_overlap = dg.compute_overlap(out)
dg_overlap_unique = dg.compute_unique_overlap(x, out)

print('Per-sample Overlap =', dg_overlap)
print('Per-sample Overlap (sum) =', dg_overlap.sum().item())

print('Per-unique Overlap =', dg_overlap_unique)
print('Per-unique Overlap (sum) =', dg_overlap_unique.sum().item())
