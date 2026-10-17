"""
Segmentation-and-recombination augmentation for motor imagery (Ma et al. 2024;
used by DESAnet for a large within-subject gain).

Each trial is split into s equal temporal segments. New synthetic trials are
built, per class, by concatenating segment j taken from a randomly chosen trial
of the SAME class, for j = 0..s-1, preserving temporal order. This expands the
training set while keeping class-consistent temporal structure. Applied to the
training split only.

Pure NumPy; unit-tested in test_augment.py.
"""
from __future__ import annotations
import numpy as np


def segment_recombine(X, y, n_aug_per_class, s=8, seed=0):
    """X: (n, C, T), y: (n,). Returns (X_aug, y_aug) with n_aug_per_class new
    trials per class (segment count s)."""
    X = np.asarray(X)
    y = np.asarray(y)
    n, C, T = X.shape
    seg = T // s
    usable = seg * s
    rng = np.random.default_rng(seed)

    Xa, ya = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        for _ in range(n_aug_per_class):
            new = np.empty((C, usable), dtype=X.dtype)
            for j in range(s):
                src = idx[rng.integers(len(idx))]
                new[:, j * seg:(j + 1) * seg] = X[src, :, j * seg:(j + 1) * seg]
            Xa.append(new)
            ya.append(cls)
    if not Xa:
        return X[:, :, :usable].copy(), y.copy()
    return np.asarray(Xa, dtype=X.dtype), np.asarray(ya)


def augment_train(X, y, factor=1.0, s=8, seed=0):
    """Return original trials (cropped to s*seg) concatenated with `factor` x
    (per-class count) recombined trials. factor=1 roughly doubles the set."""
    X = np.asarray(X)
    y = np.asarray(y)
    T = X.shape[2]
    usable = (T // s) * s
    per_class = int(round(factor * (len(y) / len(np.unique(y)))))
    Xa, ya = segment_recombine(X, y, per_class, s=s, seed=seed)
    X0 = X[:, :, :usable]
    return np.concatenate([X0, Xa], 0), np.concatenate([y, ya], 0)
