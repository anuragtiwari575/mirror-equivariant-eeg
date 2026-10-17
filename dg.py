"""Domain-generalization loss: multi-source Deep CORAL (Sun & Saenko, 2016).

Aligns the second-order feature statistics (covariances) across the source
subjects present in a batch, pushing the backbone toward subject-invariant
features so an unseen target subject decodes without calibration.

Two implementations:
  coral_loss_np : NumPy reference, unit-tested in test_dg_math.py.
  coral_loss    : torch version that mirrors the reference line for line.
Their equivalence on real tensors is checked on the target machine in
test_dg.py.

Definition: for domains present in the batch, take each domain's feature
covariance C_d (d x d); the loss is the mean over all domain pairs (a, b) of
||C_a - C_b||_F^2 / (4 d^2). Zero when all source covariances match.
"""
from __future__ import annotations
import numpy as np


def _pairwise_mean(covs, d):
    total, n = 0.0, 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            diff = covs[i] - covs[j]
            total = total + (diff * diff).sum() / (4.0 * d * d)
            n += 1
    return total / max(n, 1)


def coral_loss_np(features, domains):
    """features: (N, d) array; domains: (N,) domain ids. Returns a scalar."""
    features = np.asarray(features, dtype=np.float64)
    domains = np.asarray(domains)
    d = features.shape[1]
    covs = []
    for dv in np.unique(domains):
        F = features[domains == dv]
        Fc = F - F.mean(axis=0, keepdims=True)
        covs.append(Fc.T @ Fc / max(len(F) - 1, 1))
    return float(_pairwise_mean(covs, d))


def coral_loss(features, domains):
    """Torch version of coral_loss_np. features: (N, d) tensor; domains: (N,)
    tensor of ids. Returns a 0-dim tensor with gradient to `features`."""
    import torch
    d = features.shape[1]
    covs = []
    for dv in torch.unique(domains):
        F = features[domains == dv]
        Fc = F - F.mean(dim=0, keepdim=True)
        covs.append(Fc.t() @ Fc / max(F.shape[0] - 1, 1))
    total = features.new_zeros(())
    n = 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            diff = covs[i] - covs[j]
            total = total + (diff * diff).sum() / (4.0 * d * d)
            n += 1
    return total / max(n, 1)
