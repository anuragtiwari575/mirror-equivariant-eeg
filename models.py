"""Self-contained EEGNet (Lawhern et al., 2018) for MI decoding.

Input:  (batch, n_channels, n_times)  or  (batch, 1, n_channels, n_times)
Output: (batch, n_classes) logits

The classifier head uses nn.LazyLinear, so the flatten dimension is inferred at
the first forward pass. This removes the single most common EEGNet bug (a wrong
hardcoded in_features after the pooling stack).

LazyLinear gotcha: the head's weights do not exist until one forward pass has
run. Therefore ALWAYS do a dummy forward on a correctly shaped batch BEFORE
constructing the optimizer, otherwise the optimizer will not see the head
parameters. train_fold in deep_baseline.py handles this; test_deep.py checks it.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_classes, n_channels, n_times=None,
                 F1=8, D=2, F2=16, kern_length=64, p_drop=0.5, align_dim=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(p_drop),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p_drop),
        )
        self.head = nn.LazyLinear(n_classes)
        self.proj = nn.LazyLinear(align_dim)     # low-dim features for CORAL

    def embed(self, x):
        """Flattened pre-classifier embedding (batch, feat_dim), used by the
        classifier head."""
        if x.dim() == 3:                    # (B, C, T) -> (B, 1, C, T)
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return torch.flatten(x, 1)

    def align_embed(self, x):
        """Low-dim projection of the embedding for the CORAL alignment loss;
        its covariance is far better conditioned than the full embedding's."""
        return self.proj(self.embed(x))

    def forward(self, x):
        return self.head(self.embed(x))
