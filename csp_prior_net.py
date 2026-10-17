"""
CSPPriorNet: a compact spatio-temporal network whose spatial-projection layer is
INITIALIZED from CSP spatial filters (instead of random), then refined
end-to-end. This injects the proven discriminative subspace that plain
convolutional nets fail to learn from small MI training sets, and lets
backpropagation adapt it further.

Pipeline:
  input (B, 1, C, T)
    -> spatial Conv2d(1, K, (C,1))   [weights set to top-K CSP filters]
    -> BN, ELU
    -> temporal Conv2d(K, F, (1,kt)) + BN + ELU + pool
    -> separable temporal conv + pool
    -> LazyLinear head

build_cspprior(...) constructs the model and copies CSP filters into the spatial
layer. Torch is guarded so the file imports without torch; smoke-tested in
test_cspnet.py.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


if _HAS_TORCH:
  class CSPPriorNet(nn.Module):
    def __init__(self, n_classes, n_channels, K=8, F=16, kt=32, p_drop=0.4):
        super().__init__()
        K = min(K, n_channels)
        self.K = K
        self.spatial = nn.Conv2d(1, K, (n_channels, 1), bias=False)   # CSP-init
        self.bn1 = nn.BatchNorm2d(K)
        self.temporal = nn.Sequential(
            nn.Conv2d(K, F, (1, kt), padding=(0, kt // 2), bias=False),
            nn.BatchNorm2d(F), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(p_drop),
            nn.Conv2d(F, F, (1, 16), padding=(0, 8), groups=F, bias=False),
            nn.Conv2d(F, F, (1, 1), bias=False),
            nn.BatchNorm2d(F), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p_drop),
        )
        self.head = nn.LazyLinear(n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.elu(self.bn1(self.spatial(x)))       # (B, K, 1, T)
        x = self.temporal(x)
        return self.head(torch.flatten(x, 1))

  def build_cspprior(n_classes, n_channels, csp_filters, K=8, init="csp",
                     F=16, kt=32, p_drop=0.4):
    """init='csp' copies the top-K CSP filters into the spatial layer;
    init='random' leaves it random (ablation to show the CSP prior matters)."""
    K = min(K, n_channels)
    model = CSPPriorNet(n_classes, n_channels, K=K, F=F, kt=kt, p_drop=p_drop)
    if init == "csp" and csp_filters is not None:
        W = np.asarray(csp_filters, dtype=np.float32)[:K]            # (K, C)
        with torch.no_grad():
            model.spatial.weight.copy_(torch.tensor(W).view(K, 1, n_channels, 1))
    return model
