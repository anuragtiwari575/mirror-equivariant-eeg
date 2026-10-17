"""
FBCSP-Net: a multi-band extension of CSPPriorNet.

The signal is pre-filtered into several sub-bands. Each band has its own spatial
projection layer, INITIALIZED from that band's CSP filters, and the band
components are concatenated before shared temporal processing. This combines the
proven filter-bank + CSP spatial power (which beat single-band CSP by a clear
margin in our experiments) with end-to-end deep refinement.

Input:  (B, n_bands, C, T)   [band-filtered signal stacked on a band axis]
Output: (B, n_classes) logits

Torch is guarded so the file imports without torch; smoke-tested in
test_fbcspnet.py.
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
  class FBCSPNet(nn.Module):
    def __init__(self, n_classes, n_channels, n_bands, K=4, Fdim=16, kt=32, p_drop=0.4):
        super().__init__()
        K = min(K, n_channels)
        self.n_bands, self.K = n_bands, K
        self.spatials = nn.ModuleList(
            [nn.Conv2d(1, K, (n_channels, 1), bias=False) for _ in range(n_bands)])
        self.bn1 = nn.BatchNorm2d(K * n_bands)
        self.temporal = nn.Sequential(
            nn.Conv2d(K * n_bands, Fdim, (1, kt), padding=(0, kt // 2), bias=False),
            nn.BatchNorm2d(Fdim), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p_drop),
            nn.Conv2d(Fdim, Fdim, (1, 16), padding=(0, 8), groups=Fdim, bias=False),
            nn.Conv2d(Fdim, Fdim, (1, 1), bias=False),
            nn.BatchNorm2d(Fdim), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p_drop),
        )
        self.head = nn.LazyLinear(n_classes)

    def forward(self, x):                         # (B, n_bands, C, T)
        outs = [self.spatials[b](x[:, b:b + 1]) for b in range(self.n_bands)]  # (B,K,1,T)
        z = F.elu(self.bn1(torch.cat(outs, dim=1)))
        z = self.temporal(z)
        return self.head(torch.flatten(z, 1))

  def build_fbcspnet(n_classes, n_channels, band_filters, K=4, **kw):
    """band_filters: list of (C, C) CSP filter matrices, one per band. Copies the
    top-K filters of each band into the matching spatial layer."""
    nb = len(band_filters)
    K = min(K, n_channels)
    model = FBCSPNet(n_classes, n_channels, nb, K=K, **kw)
    with torch.no_grad():
        for b, filt in enumerate(band_filters):
            W = np.asarray(filt, dtype=np.float32)[:K]
            model.spatials[b].weight.copy_(torch.tensor(W).view(K, 1, n_channels, 1))
    return model
