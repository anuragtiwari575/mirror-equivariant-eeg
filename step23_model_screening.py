"""
step23_model_screening.py

Three new network components for decoding left- against right-hand motor imagery across datasets,
screened in ONE run under the protocol of step 16: a network is trained on the labeled trials of one
source dataset and applied to every subject of another dataset; the unlabeled trials of that subject
may be used, its labels never. A subject's score is the mean over source datasets (and seeds).

Run from the Codes folder, after steps 15 and 16:

    python step23_model_screening.py --selftest    toy data, a few minutes: checks the installation, writes nothing
    python step23_model_screening.py               screening: 17 channels, one seed (1 to 2 hours with a GPU,
                                                   plus 30 to 40 minutes at the first run for the pre-cue segments)
    python step23_model_screening.py --full        both montages, seeds 0 and 1 (finished trainings are reused)
    python step23_model_screening.py "D:\\other\\folder\\BCI Database"       (if the fifth dataset is elsewhere)

    other options: --methods a,b,c   --montage 17ch|3ch|both   --seeds 0,1   --epochs N   --no-rest   --redo

How a user enters the networks. Every user is described by the lagged covariances of his or her
unlabeled trials, G(l)[c, d] = mean of x_c(t) x_d(t + l) for l = 0 ... 124 samples. The covariance of
that user's data after ANY temporal kernel w then follows in closed form,
R(w) = r(0) G(0) + sum over l > 0 of r(l) [G(l) + G(l)'], with r the autocorrelation of the kernel,
so that a layer can use it while the kernel is still being learned.

Direction 1: beamforming convolution (body of EEGNet)
  The spatial stage learns PATTERNS a (how a source projects to the channels) instead of filters. The
  filter is computed for every user from his or her covariance R after the temporal kernel of the same
  branch: w = R^-1 a, the beamformer for that pattern (R shrunk towards the identity by a learned amount
  gamma; gamma = 1 gives an ordinary spatial filter). The filter is then scaled in one of two ways:
  "gain": a'w = 1 (unit gain for the pattern), or "power": w'Rw = 1 (the output has unit power in the
  data of that user, a normalization per user in closed form).
     eegnet_bf           beamformer and unit power per user; one covariance per learned temporal kernel
     eegnet_bf_gain      beamformer only (unit gain)                                 } the two ingredients
     eegnet_bf_scale     unit power per user only (gamma fixed at 1: no beamformer)  } of eegnet_bf, apart
     eegnet_bf_off       neither: the same network without any covariance (control)
     eegnet_bf_broad     as eegnet_bf with one broadband covariance (the statistic of Euclidean alignment)
     eegnet_bf_rest      as eegnet_bf with covariances from the pre-cue segments

Direction 2: reference-contrast layer (body of ShallowConvNet, without batch normalization)
  Feature = log of [power of a learned spatio-temporal filter in the task segment / REFERENCE power of the
  same filter]: event-related desynchronization as a layer.
     shallow_erd_trial   reference: the pre-cue segment of the same trial (no other trial of the user needed)
     shallow_erd_user    reference: mean pre-cue power of the user (closed form from the lagged covariances)
     shallow_ref_task    reference: mean task power of the user (closed form; control without pre-cue data)

Direction 3: mirror-equivariant network
  One network g with one output: evidence for the right hand = g(x), for the left hand = g(mirror image
  of x, left and right hemisphere exchanged). Exchanging the hemispheres then exchanges the classes exactly.
     eegnet_mirror       EEGNet, no alignment (uses no other trial of the user)
     eegnet_ea_mirror    EEGNet with Euclidean alignment
     eegnet_bf_mirror    beamforming convolution: every pattern is used together with its mirror image

Direction 4: mode convolution (motivated by multivariate variational mode decomposition, MVMD: ur Rehman
and Aftab 2019; used before a graph network by Du et al. 2026, J. Neural Eng.)
  MVMD splits the signals into rhythmic modes that share one centre frequency over the channels: each mode is
  the signal after a band-pass around that frequency, and the frequency is moved to the centroid of the power
  inside the mode. Here the temporal kernels of the beamforming network ARE such modes, exp(-|t|/tau)
  cos(2 pi f t), with learned centre frequency f and width (2 numbers per kernel instead of 125). The update
  of the centre frequency is done for every user from his or her own spectrum (which the lagged covariances
  contain), in two steps of learned size eta: every user gets the kernels at his or her own rhythms.
     eegnet_bf_mode        mode kernels, the same for all users
     eegnet_bf_mode_user   mode kernels, centre frequencies moved for every user

Direction 5: patterns from scalp positions (one model for any montage)
  A spatial pattern is usually one weight per channel, so a trained model belongs to one montage. Here a pattern
  is a smooth function of the position of the electrode on the scalp, a weighted sum of Gaussians on a fixed grid;
  the weights are learned and the positions come from the electrode names. The number of parameters no longer
  depends on the number of channels, and the same trained model can be applied to any montage (the beamforming
  convolution already removes the channel dimension, so nothing else in the network depends on it).
     eegnet_bf_geo         beamforming convolution with patterns from positions
     eegnet_bf_geo_mirror  the same, with the mirror construction

Pooled sources: every method above is trained on ONE source dataset. The methods ending in _pool are trained on
all source datasets at once (for a given target, every other dataset), which is the same protocol with more
training data, so the gain of pooling and the gain of a component can be separated.
     eegnet_ea_pool, eegnet_ea_mirror_pool, eegnet_bf_mirror_pool

Baselines, trained again here as in step 16: eegnet_none, eegnet_ea, shallow_none, shallow_ea.

Data. Task segments: the cached epochs of steps 11 and 16 (8-30 Hz, 0.5-2.5 s, 250 Hz). Pre-cue segments
(-1.5 to 0 s): read once from the recordings (30 to 40 minutes) and stored in
corrected_results/cache_step23_rest (about 1.2 GB). Trials without a pre-cue segment are left out for
all methods. With --no-rest, or if the recordings cannot be read, the three methods that need pre-cue
segments are skipped and everything else runs from the caches.

Outputs in corrected_results/: step23_summary.txt, step23_models.csv (one row per method and subject),
step23_raw.csv (one row per training and subject; an interrupted run resumes from it), step23_info.csv
(learned centre frequencies of direction 4).
"""
from __future__ import annotations

import os
import re
import sys
import glob
import time
import argparse
import datetime

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
except Exception as _e:                                    # pragma: no cover
    print(f"PyTorch could not be imported in this environment ({_e!r}).")
    sys.exit(1)

try:
    import step5_benchmark as S5
    import step8_new_analyses as S8
    import step9_reviewer_checks as S9
    import step11_directions as S11
    import step15_external_validation as S15
    import step16_baselines as S16
except Exception as _e:                                    # pragma: no cover
    print(f"could not import the earlier step scripts (5, 8 to 16); run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
out = S5.out
EXT = S15.NAME
REST_CACHE = os.path.join(OUTDIR, "cache_step23_rest")
LAGS = 125                     # lags of the stored covariances = length of the temporal kernels of EEGNet
REST = (125, 500)              # pre-cue segment inside an epoch of -2.0 to 2.5 s: -1.5 to 0 s
TASK = (625, 1126)             # task segment inside the same epoch: 0.5 to 2.5 s
GAMMA_MIN, GAMMA_INIT = 0.01, 0.1
SM17 = ["FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4"]

SPECS = {
    "eegnet_none":       dict(net="eegnet", inp="raw", label="EEGNet, no alignment"),
    "eegnet_ea":         dict(net="eegnet", inp="ea", label="EEGNet, Euclidean alignment"),
    "shallow_none":      dict(net="shallow", inp="raw", label="ShallowConvNet, no alignment"),
    "shallow_ea":        dict(net="shallow", inp="ea", label="ShallowConvNet, Euclidean alignment"),
    "eegnet_bf_off":     dict(net="bf", gamma="one", norm="gain", label="1  no covariance (control)"),
    "eegnet_bf_scale":   dict(net="bf", gamma="one", norm="power", stat="task", label="1  unit power per user only"),
    "eegnet_bf_gain":    dict(net="bf", gamma="learn", norm="gain", stat="task", label="1  beamformer only"),
    "eegnet_bf":         dict(net="bf", gamma="learn", norm="power", stat="task", label="1  beamforming convolution"),
    "eegnet_bf_broad":   dict(net="bf", gamma="learn", norm="power", stat="task", mode="broad", label="1  same, broadband covariance"),
    "eegnet_bf_rest":    dict(net="bf", gamma="learn", norm="power", stat="rest", label="1  same, pre-cue covariances"),
    "shallow_erd_trial": dict(net="ref", ref="trial", label="2  reference: same trial, pre-cue"),
    "shallow_erd_user":  dict(net="ref", ref="user", stat="rest", label="2  reference: user, pre-cue"),
    "shallow_ref_task":  dict(net="ref", ref="user", stat="task", label="2  reference: user, task"),
    "eegnet_mirror":     dict(net="eegnet", inp="raw", mirror=True, label="3  mirror, EEGNet"),
    "eegnet_ea_mirror":  dict(net="eegnet", inp="ea", mirror=True, label="3  mirror, EEGNet with EA"),
    "eegnet_bf_mirror":  dict(net="bf", gamma="learn", norm="power", stat="task", mirror=True, label="1+3  mirror, beamforming conv."),
    "eegnet_bf_mode":    dict(net="bf", gamma="learn", norm="power", stat="task", kernel="mode", label="1+4  mode kernels"),
    "eegnet_bf_mode_user": dict(net="bf", gamma="learn", norm="power", stat="task", kernel="mode_user", label="1+4  mode kernels, per user"),
    "eegnet_bf_geo":     dict(net="bf", gamma="learn", norm="power", stat="task", geo=True, label="5  patterns from scalp positions"),
    "eegnet_bf_geo_mirror": dict(net="bf", gamma="learn", norm="power", stat="task", geo=True, mirror=True, label="3+5  same, with mirror"),
    "eegnet_ea_pool":    dict(net="eegnet", inp="ea", pool=True, label="6  EEGNet with EA, sources pooled"),
    "eegnet_ea_mirror_pool": dict(net="eegnet", inp="ea", mirror=True, pool=True, label="3+6  mirror with EA, pooled"),
    "eegnet_bf_mirror_pool": dict(net="bf", gamma="learn", norm="power", stat="task", mirror=True, pool=True, label="1+3+6  mirror, beamforming, pooled"),
}
BASELINES = ["eegnet_none", "eegnet_ea", "shallow_none", "shallow_ea"]
CONTRASTS = [
    ("Direction 1, beamforming convolution", [
        ("eegnet_bf", "eegnet_ea", "against the standard (EEGNet with EA)"),
        ("eegnet_bf", "eegnet_bf_off", "covariance used or not, same network"),
        ("eegnet_bf_scale", "eegnet_bf_off", "unit power per user alone"),
        ("eegnet_bf_gain", "eegnet_bf_off", "beamformer alone"),
        ("eegnet_bf", "eegnet_bf_scale", "what the beamformer adds to unit power"),
        ("eegnet_bf_broad", "eegnet_ea", "the covariance of EA, used in this layer"),
        ("eegnet_bf", "eegnet_bf_broad", "one covariance per kernel or one broadband"),
        ("eegnet_bf_rest", "eegnet_bf", "covariances of pre-cue or of task segments"),
        ("eegnet_bf_off", "eegnet_none", "the network without covariance against EEGNet")]),
    ("Direction 2, reference-contrast layer", [
        ("shallow_erd_trial", "shallow_none", "neither uses another trial of the user"),
        ("shallow_erd_trial", "shallow_ea", ""),
        ("shallow_erd_user", "shallow_ea", ""),
        ("shallow_erd_user", "shallow_erd_trial", "reference of the user or of the trial"),
        ("shallow_ref_task", "shallow_ea", ""),
        ("shallow_erd_user", "shallow_ref_task", "pre-cue or task reference")]),
    ("Direction 3, mirror-equivariant network", [
        ("eegnet_mirror", "eegnet_none", "neither uses another trial of the user"),
        ("eegnet_mirror", "eegnet_ea", ""),
        ("eegnet_ea_mirror", "eegnet_ea", ""),
        ("eegnet_bf_mirror", "eegnet_bf", ""),
        ("eegnet_bf_mirror", "eegnet_ea", "directions 1 and 3 against the standard")]),
    ("Direction 4, mode convolution", [
        ("eegnet_bf_mode", "eegnet_bf", "modes with two numbers or free kernels"),
        ("eegnet_bf_mode_user", "eegnet_bf_mode", "centre frequencies moved for every user"),
        ("eegnet_bf_mode_user", "eegnet_bf", ""),
        ("eegnet_bf_mode_user", "eegnet_ea", "directions 1 and 4 against the standard")]),
    ("Direction 5, patterns from scalp positions", [
        ("eegnet_bf_geo", "eegnet_bf", "positions instead of one weight per channel"),
        ("eegnet_bf_geo_mirror", "eegnet_bf_mirror", "the same, with mirror"),
        ("eegnet_bf_geo_mirror", "eegnet_ea", "against the standard")]),
    ("Pooled sources (trained on all source datasets at once)", [
        ("eegnet_ea_pool", "eegnet_ea", "pooling alone"),
        ("eegnet_ea_mirror_pool", "eegnet_ea_pool", "mirror, with pooled sources"),
        ("eegnet_ea_mirror_pool", "eegnet_ea_mirror", "pooling, with mirror"),
        ("eegnet_bf_mirror_pool", "eegnet_ea_mirror_pool", "beamformer, with pooling and mirror"),
        ("eegnet_ea_mirror_pool", "eegnet_ea", "everything together against the standard")]),
]


def needs_rest(name):
    sp = SPECS[name]
    return sp.get("stat") == "rest" or sp.get("ref") == "trial"


# =========================== statistics of a user ===========================
def lagcov(x, L=LAGS):
    """Lagged covariances of the trials x (n, C, T): G[l, c, d] = sum over trials and time of x_c(t) x_d(t + l),
    divided by n T, for l = 0 ... L-1. Dividing by n T (and not by the number of products) makes the covariance
    after any temporal kernel equal to the power of the zero-padded filtered trials, hence positive semi-definite."""
    n, C, T = x.shape
    N = 1 << int(np.ceil(np.log2(T + L)))
    F = np.fft.rfft(np.asarray(x, dtype=np.float64), N, axis=2).transpose(2, 0, 1)        # (frequency, trial, channel)
    S = np.matmul(F.conj().transpose(0, 2, 1), F)                                          # cross-spectra, summed over trials
    return np.ascontiguousarray(np.fft.irfft(S, N, axis=0)[:L]) / (n * T)


def whiten(P, x):
    return np.einsum("cd,ndt->nct", P, x).astype(np.float32)


def mirror_perm(names):
    """For every channel the index of its mirror image (left and right hemisphere exchanged; midline stays)."""
    up = [n.upper() for n in names]
    perm = []
    for n in up:
        m = re.match(r"^([A-Z]+?)(\d+|Z)$", n)
        if m.group(2) == "Z":
            perm.append(up.index(n))
        else:
            k = int(m.group(2))
            perm.append(up.index(f"{m.group(1)}{k + 1 if k % 2 else k - 1}"))
    return perm


ROW_Y = {"FP": 4.0, "AF": 3.0, "F": 2.0, "FT": 1.5, "FC": 1.0, "T": 0.0, "C": 0.0, "TP": -1.5, "CP": -1.0,
         "P": -2.0, "PO": -3.0, "O": -4.0}
GRID_X = [-3.0, -1.5, 0.0, 1.5, 3.0]
GRID_Y = [-1.0, 0.0, 1.0]


def positions(names):
    """Position of every electrode on the scalp (x to the right, y to the front), read from its 10-20 name.
    Odd numbers are left, even numbers right, Z is the midline; the letters give the row."""
    P = []
    for n in names:
        m = re.match(r"^([A-Z]+?)(\d+|Z)$", n.upper())
        if m is None or m.group(1) not in ROW_Y:
            raise RuntimeError(f"no position known for the electrode {n}")
        k = m.group(2)
        P.append([0.0 if k == "Z" else (int(k) + 1) // 2 * (1.0 if int(k) % 2 == 0 else -1.0), ROW_Y[m.group(1)]])
    return np.asarray(P, dtype=np.float64)


def basis_matrix(names, width=1.5):
    """Smooth functions of the position (B, C): Gaussians on a fixed grid, each scaled to unit norm over the
    electrodes at hand. A pattern is a weighted sum of these, so its parameters do not depend on the montage."""
    P = positions(names)
    ctr = np.array([[x, y] for y in GRID_Y for x in GRID_X], dtype=np.float64)
    Phi = np.exp(-((P[None, :, :] - ctr[:, None, :]) ** 2).sum(-1) / (2.0 * width ** 2))
    return Phi / np.sqrt((Phi ** 2).sum(axis=1, keepdims=True) + 1e-12)


class Set:
    """All subjects of one dataset in one montage."""

    def __init__(self, ids, x, y, rest):
        self.ids, self.x, self.y, self.rest = list(ids), x, y, rest
        self.ea = [S11.ea_matrix(v) for v in x]
        self.g = {"task": np.stack([lagcov(v) for v in x]),
                  "rest": None if rest is None else np.stack([lagcov(v) for v in rest])}

    @staticmethod
    def pooled(parts):
        """Several datasets as one source set; the statistics of the users are reused, not computed again."""
        S = Set.__new__(Set)
        S.ids = [i for p in parts for i in p.ids]
        S.x = [v for p in parts for v in p.x]
        S.y = [v for p in parts for v in p.y]
        S.rest = None if any(p.rest is None for p in parts) else [v for p in parts for v in p.rest]
        S.ea = [v for p in parts for v in p.ea]
        S.g = {k: (None if any(p.g[k] is None for p in parts) else np.concatenate([p.g[k] for p in parts]))
               for k in ("task", "rest")}
        return S


def prepare(D, montage, tnames):
    sets = {}
    for d in tnames:
        pick = (lambda v: v) if (montage == "17ch" or d == "2b") else (lambda v: np.ascontiguousarray(v[:, S5.TRIPLET_IN_SM, :]))
        sets[d] = Set(D[d]["ids"], [pick(v) for v in D[d]["x"]], D[d]["y"],
                      None if D[d]["rest"] is None else [pick(v) for v in D[d]["rest"]])
    return sets


# =========================== networks ===========================
class EEGNet(nn.Module):
    """EEGNet exactly as in step 16; n_out = 1 gives the one-output network of the mirror models."""

    def __init__(self, C, T, n_out=2, F1=8, D=2, F2=16, k1=125, p=0.5):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.b2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            n = self.b2(self.b1(torch.zeros(1, 1, C, T))).numel()
        self.fc = nn.Linear(n, n_out)

    def forward(self, x, uid=None, G=None, rest=None):
        return self.fc(self.b2(self.b1(x.unsqueeze(1))).flatten(1))


class Shallow(nn.Module):
    """ShallowConvNet exactly as in step 16."""

    def __init__(self, C, T, n_out=2, F=40, k=25, pool=75, stride=15, p=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, F, (1, k))
        self.spatial = nn.Conv2d(F, F, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(F)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.drop = nn.Dropout(p)
        with torch.no_grad():
            n = self._features(torch.zeros(1, 1, C, T)).numel()
        self.fc = nn.Linear(n, n_out)

    def _features(self, x):
        x = self.bn(self.spatial(self.temporal(x)))
        x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
        return self.drop(x).flatten(1)

    def forward(self, x, uid=None, G=None, rest=None):
        return self.fc(self._features(x.unsqueeze(1)))


class BeamConv(nn.Module):
    """Beamforming convolution. Learned: D patterns per temporal kernel and one shrinkage per kernel.
    Computed for every user: the spatial filters w = R^-1 a, scaled to unit gain or to unit output power."""

    def __init__(self, C, F1, D, mode="band", gamma="learn", norm="power", basis=None):
        super().__init__()
        self.mode, self.gamma_mode, self.norm = mode, gamma, norm            # "band" | "broad"; "learn" | "one"; "gain" | "power"
        self.uses_cov = not (gamma == "one" and norm == "gain")
        if basis is None:                                                    # one weight per channel
            self.register_buffer("Phi", None)
            self.a = nn.Parameter(torch.empty(F1, D, C).uniform_(-1.0, 1.0) / C ** 0.5)
        else:                                                                # weights of the position functions
            Phi = torch.as_tensor(np.asarray(basis), dtype=torch.float32)
            self.register_buffer("Phi", Phi)
            B = Phi.shape[0]
            self.a = nn.Parameter(torch.empty(F1, D, B).uniform_(-1.0, 1.0) / B ** 0.5)
        t = (GAMMA_INIT - GAMMA_MIN) / (1.0 - GAMMA_MIN)
        self.theta = nn.Parameter(torch.full((F1,), float(np.log(t / (1.0 - t)))), requires_grad=gamma == "learn")

    def gamma(self):
        if self.gamma_mode == "one":
            return torch.ones_like(self.theta)
        return GAMMA_MIN + (1.0 - GAMMA_MIN) * torch.sigmoid(self.theta)

    def patterns(self):
        """The spatial patterns (F1, D, C)."""
        return self.a if self.Phi is None else torch.einsum("fdb,bc->fdc", self.a, self.Phi)

    def plain(self):
        """No covariance: the pattern itself as filter, with unit gain (F1, D, C)."""
        A = self.patterns()
        return A / (A * A).sum(-1, keepdim=True)

    def covariances(self, w, G):
        """Covariance of every user after every temporal kernel (U, F1, C, C), float64.
        w: kernels (F1, k), or (U, F1, k) if every user has his or her own; G: lagged covariances (U, L, C, C), float64."""
        U, C, F1 = G.shape[0], G.shape[-1], self.a.shape[0]
        if self.mode == "broad":
            return G[:, :1].expand(U, F1, C, C)
        k = w.shape[-1]
        wd = w.double().reshape(1, -1, k)
        r = Fn.conv1d(wd, wd.transpose(0, 1), padding=k - 1, groups=wd.shape[1])[0, :, k - 1:]         # autocorrelation, lags 0 ... k-1
        r = torch.cat([0.5 * r[:, :1], r[:, 1:]], dim=1)
        H = torch.einsum("ufl,ulcd->ufcd", r.reshape(U, F1, k), G[:, :k]) if w.dim() == 3 else torch.einsum("fl,ulcd->ufcd", r, G[:, :k])
        return H + H.transpose(-1, -2)

    def weights(self, w, G):
        """Spatial filters of every user (U, F1, D, C)."""
        R = self.covariances(w, G)
        s = R.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)[..., None, None]
        Rn = R / s                                                                                      # unit mean variance
        g = self.gamma().double()[None, :, None, None]
        Rg = (1.0 - g) * Rn + g * torch.eye(R.shape[-1], dtype=R.dtype, device=R.device)
        At = self.patterns().double().transpose(1, 2).unsqueeze(0).expand(R.shape[0], -1, -1, -1)       # (U, F1, C, D)
        Z = torch.linalg.solve(Rg, At)                                                                  # R^-1 a
        if self.norm == "gain":
            Z = Z / (At * Z).sum(dim=2, keepdim=True)                                                   # a'w = 1
        else:
            Z = Z / torch.sqrt(((Z * torch.matmul(Rn, Z)).sum(dim=2, keepdim=True) * s).clamp_min(1e-20))   # w'Rw = 1
        return Z.transpose(2, 3).float()


class ModeKernels(nn.Module):
    """Temporal kernels as the modes of a variational mode decomposition: every kernel is a band-pass around a centre
    frequency, exp(-|t| / tau) cos(2 pi f t); in frequency this is the filter of the decomposition, 1 / (1 + ((w - w_f) tau)^2).
    Learned: centre frequency and width of every mode. With adapt=True the centre frequency is moved, for every user,
    towards the centroid of that user's power inside the mode (the update rule of the decomposition), in `steps` steps of
    learned size eta; the spectrum of the user comes from the lagged covariances."""

    def __init__(self, F1, k, adapt, fs=250.0, f_lo=9.0, f_hi=30.0, steps=2):
        super().__init__()
        self.adapt, self.steps, self.fs = adapt, steps, fs
        self.f = nn.Parameter(torch.linspace(f_lo, f_hi, F1))                                  # centre frequencies, Hz
        self.rho = nn.Parameter(torch.full((F1,), float(np.log(0.375 / 0.625))))                # tau = 5 + 40 sigmoid(rho): start 20 samples
        self.eta_raw = nn.Parameter(torch.zeros(()), requires_grad=adapt)                       # eta = sigmoid: start 0.5
        self.register_buffer("t", torch.arange(k, dtype=torch.float32) - k // 2)
        grid = torch.arange(5.0, 35.01, 0.25)
        self.register_buffer("grid", grid)
        self.register_buffer("kcos", torch.cos(2 * np.pi * self.t[:, None] * grid[None, :] / fs))               # (k, grid)
        lag, m = torch.arange(LAGS, dtype=torch.float64), torch.full((LAGS,), 2.0, dtype=torch.float64)
        m[0] = 1.0
        self.register_buffer("lcos", m[:, None] * torch.cos(2 * np.pi * lag[:, None] * grid.double()[None, :] / fs))   # (lags, grid)

    def tau(self):
        return 5.0 + 40.0 * torch.sigmoid(self.rho)

    def eta(self):
        return torch.sigmoid(self.eta_raw)

    def freq(self):
        return self.f.clamp(5.0, 35.0)

    def kernels(self, f):
        """Kernels (..., F1, k) with unit norm for the centre frequencies f (..., F1) in Hz."""
        w = torch.exp(-self.t.abs() / self.tau()[:, None]) * torch.cos(2 * np.pi * f[..., None] * self.t / self.fs)
        return w / w.norm(dim=-1, keepdim=True)

    def spectrum(self, G):
        """Power spectrum of every user on the frequency grid (U, grid), from the lagged covariances (U, L, C, C)."""
        return (G.diagonal(dim1=-2, dim2=-1).sum(-1) @ self.lcos).clamp_min(0.0).float()

    def user_freq(self, G):
        """Centre frequencies of every user (U, F1)."""
        P = self.spectrum(G)
        f = self.freq().unsqueeze(0).expand(G.shape[0], -1)
        for _ in range(self.steps):
            H2 = torch.matmul(self.kernels(f), self.kcos) ** 2 * P[:, None, :]                  # the user's power inside every mode
            f = f + self.eta() * ((H2 * self.grid).sum(-1) / H2.sum(-1).clamp_min(1e-20) - f)
        return f


class EEGNetBF(nn.Module):
    """EEGNet whose spatial stage is a beamforming convolution (no batch normalization before it: the scaling
    of the beamformer fixes the scale). kernel: "free" (ordinary temporal kernels) | "mode" | "mode_user"."""

    def __init__(self, C, T, mode, gamma, norm, n_out=2, kernel="free", basis=None, F1=8, D=2, F2=16, k1=125, p=0.5):
        super().__init__()
        self.k1 = k1
        if kernel == "free":
            self.temporal, self.modes = nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), None
        else:
            self.temporal, self.modes = None, ModeKernels(F1, k1, kernel == "mode_user")
        self.beam = BeamConv(C, F1, D, mode, gamma, norm, basis)
        self.post = nn.Sequential(nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.b2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            n = self.b2(self.post(torch.zeros(1, F1 * D, 1, T))).numel()
        self.fc = nn.Linear(n, n_out)

    def forward(self, x, uid=None, G=None, rest=None):
        B, pad = x.shape[0], (0, self.k1 // 2)
        us = inv = None
        if self.beam.uses_cov:
            us, inv = torch.unique(uid, return_inverse=True)                # only the users of this batch
        if self.modes is None:
            w, u = self.temporal.weight[:, 0, 0, :], self.temporal(x.unsqueeze(1))             # u: (B, F1, C, T)
        elif not self.modes.adapt:
            w = self.modes.kernels(self.modes.freq())
            u = Fn.conv2d(x.unsqueeze(1), w[:, None, None, :], padding=pad)
        else:
            w = self.modes.kernels(self.modes.user_freq(G[us]))             # (U, F1, k): every user has his or her own kernels
            u = Fn.conv2d(x.unsqueeze(0), w[inv].reshape(-1, 1, 1, self.k1), padding=pad, groups=B).reshape(B, -1, x.shape[1], x.shape[2])
        if not self.beam.uses_cov:
            z = torch.einsum("fdc,bfct->bfdt", self.beam.plain(), u)
        else:
            z = torch.einsum("bfdc,bfct->bfdt", self.beam.weights(w, G[us])[inv], u)
        z = z.reshape(B, -1, 1, z.shape[-1])
        return self.fc(self.b2(self.post(z)).flatten(1))


class ShallowRef(nn.Module):
    """ShallowConvNet whose log-power features are taken relative to a reference power of the same filters
    (no batch normalization and no bias: the ratio fixes the scale)."""

    def __init__(self, C, T, ref, n_out=2, F=40, k=25, pool=75, stride=15, p=0.5):
        super().__init__()
        self.ref = ref                                                      # "trial" | "user"
        self.temporal = nn.Conv2d(1, F, (1, k), bias=False)
        self.spatial = nn.Conv2d(F, F, (C, 1), bias=False)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(F * ((T - k + 1 - pool) // stride + 1), n_out)

    def power(self, x):
        v = self.spatial(self.temporal(x.unsqueeze(1)))                     # (B, F, 1, T - k + 1)
        return v * v

    def user_power(self, G):
        """Mean power of every filter in the data of every user (U, F), from the lagged covariances (U, L, C, C)."""
        K = torch.einsum("gfc,fi->gci", self.spatial.weight[:, :, :, 0], self.temporal.weight[:, 0, 0, :]).double()
        k = K.shape[-1]                                                     # K: the spatio-temporal kernel of each filter
        Kl = Fn.pad(K, (0, k - 1)).unfold(-1, k, 1)                         # Kl[g, d, l, i] = K[g, d, i + l]
        A = torch.einsum("gci,gdli->gcdl", K, Kl)
        m = torch.full((k,), 2.0, dtype=A.dtype, device=A.device)
        m[0] = 1.0
        return torch.einsum("gcdl,ulcd->ug", A * m, G[:, :k])

    def forward(self, x, uid=None, G=None, rest=None):
        p = self.pool(self.power(x))                                        # (B, F, 1, bins)
        if self.ref == "trial":
            q = self.power(rest).mean(-1, keepdim=True)
        else:
            us, inv = torch.unique(uid, return_inverse=True)
            q = self.user_power(G[us])[inv].float()[:, :, None, None]
        z = torch.log(torch.clamp(p / q.clamp_min(1e-12), min=1e-6))
        return self.fc(self.drop(z).flatten(1))


class Mirror(nn.Module):
    """Class 1 (right hand) gets g(x), class 0 (left hand) gets g(mirror image of x); g has one output."""

    def __init__(self, core, perm):
        super().__init__()
        self.core = core
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def forward(self, x, uid=None, G=None, rest=None):
        B, p = x.shape[0], self.perm
        x2 = torch.cat([x, x[:, p]])
        r2 = None if rest is None else torch.cat([rest, rest[:, p]])
        u2 = None if uid is None else torch.cat([uid, uid + (0 if G is None else G.shape[0] // 2)])   # G holds users, then mirrored users
        o = self.core(x2, u2, G, r2)[:, 0]
        return torch.stack([o[B:], o[:B]], dim=1)


def build(name, C, T, perm, names=None):
    sp = SPECS[name]
    n_out = 1 if sp.get("mirror") else 2
    basis = None
    if sp.get("geo"):
        if names is None or len(names) != C:
            raise RuntimeError(f"{name}: the names of the {C} electrodes are needed for the positions")
        basis = basis_matrix(names)
    if sp["net"] == "eegnet":
        core = EEGNet(C, T, n_out)
    elif sp["net"] == "shallow":
        core = Shallow(C, T, n_out)
    elif sp["net"] == "bf":
        core = EEGNetBF(C, T, sp.get("mode", "band"), sp["gamma"], sp["norm"], n_out, sp.get("kernel", "free"), basis)
    else:
        core = ShallowRef(C, T, sp["ref"], n_out)
    return Mirror(core, perm) if sp.get("mirror") else core


# =========================== training and testing ===========================
def tensors(S, name, device, perm, js=None):
    """Inputs of a method for the listed subjects of a set (all subjects if js is None), on the device."""
    sp = SPECS[name]
    js = list(range(len(S.y))) if js is None else list(js)
    x = [whiten(S.ea[j], S.x[j]) if sp.get("inp") == "ea" else S.x[j] for j in js]
    t = dict(x=torch.as_tensor(x[0] if len(x) == 1 else np.concatenate(x), dtype=torch.float32, device=device),
             uid=torch.as_tensor(np.concatenate([np.full(len(S.y[j]), i) for i, j in enumerate(js)]), dtype=torch.long, device=device),
             G=None, rest=None)
    if sp.get("stat"):
        G = torch.as_tensor(S.g[sp["stat"]][js], dtype=torch.float64, device=device)
        if sp.get("mirror"):
            p = torch.as_tensor(perm, dtype=torch.long, device=device)
            G = torch.cat([G, G[:, :, p][:, :, :, p]])
        t["G"] = G
    if sp.get("ref") == "trial":
        t["rest"] = torch.as_tensor(np.concatenate([S.rest[j] for j in js]), dtype=torch.float32, device=device)
    return t


def fit(name, S, seed, epochs, device, perm, names=None):
    """Training as in step 16: Adam, learning rate 0.001, batches of 64, all trials of the source dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build(name, S.x[0].shape[1], S.x[0].shape[2], perm, names).to(device)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    t = tensors(S, name, device, perm)
    yt = torch.as_tensor(np.concatenate(S.y), dtype=torch.long, device=device)
    lossf = nn.CrossEntropyLoss()
    n = len(yt)
    for _ in range(epochs):
        net.train()
        order = torch.randperm(n, device=device)
        for i in range(0, n, 64):
            idx = order[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(net(t["x"][idx], t["uid"][idx], t["G"], None if t["rest"] is None else t["rest"][idx]), yt[idx])
            if not torch.isfinite(loss):
                raise RuntimeError("the loss is not finite")
            loss.backward()
            opt.step()
    net.eval()
    return net


def score(net, t):
    res = []
    with torch.no_grad():
        for i in range(0, len(t["x"]), 1024):
            o = net(t["x"][i:i + 1024], t["uid"][i:i + 1024], t["G"], None if t["rest"] is None else t["rest"][i:i + 1024])
            res.append((o[:, 1] - o[:, 0]).float().cpu().numpy())
    return np.concatenate(res)


def run_method(name, src, targets, seed, epochs, device, perm, names=None):
    net = fit(name, src, seed, epochs, device, perm, names)
    rows = {}
    for tname, S in targets:
        for j, y in enumerate(S.y):
            o = score(net, tensors(S, name, device, perm, [j]))
            rows[(tname, str(S.ids[j]))] = (len(y), float(np.mean((o > 0) == (np.asarray(y) == 1))))
    beam = [m for m in net.modules() if isinstance(m, BeamConv)]
    extra = float(beam[0].gamma().mean().item()) if beam else float("nan")
    info = {}
    for m in net.modules():
        if isinstance(m, ModeKernels):
            info = {f"f{i + 1}": float(v) for i, v in enumerate(m.freq().detach().cpu().numpy())}
            info["width"] = float((m.fs / (np.pi * m.tau())).mean().item())                    # full width at half maximum of the gain, Hz
            if m.adapt:
                info["eta"] = float(m.eta().item())
    return rows, extra, info


class Store:
    """Finished trainings, (montage, method, source, seed) -> {(target, subject): accuracy}, mirrored in a CSV file."""
    HEAD = "data,epochs,montage,method,source,seed,target,subject,n,accuracy,extra"

    def __init__(self, path, data, epochs, redo=False):
        self.path, self.data, self.epochs, self.redo = path, data, int(epochs), redo
        self.res, self.extra, self.info, self.fresh = {}, {}, {}, set()
        self.path_info = path[:-len("raw.csv")] + "info.csv" if path else None
        if path and os.path.exists(self.path_info):
            with open(self.path_info, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.rstrip("\n").split(",")
                    if len(p) == 8 and p[0] == data and p[1] == str(self.epochs):
                        self.info.setdefault((p[2], p[3], p[4], int(p[5])), {})[p[6]] = float(p[7])
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.rstrip("\n").split(",")
                    if len(p) != 11 or p[0] != data or p[1] != str(self.epochs):
                        continue
                    key = (p[2], p[3], p[4], int(p[5]))
                    self.res.setdefault(key, {})[(p[6], p[7])] = float(p[9])
                    self.extra[key] = float(p[10])

    def has(self, key, n):
        return len(self.res.get(key, {})) == n and (not self.redo or key in self.fresh)

    def put(self, key, rows, extra, info=None):
        self.res[key] = {k: v[1] for k, v in rows.items()}
        self.extra[key] = extra
        self.fresh.add(key)
        if info:
            self.info[key] = dict(info)
        if not self.path:
            return
        if info:
            new = not os.path.exists(self.path_info)
            with open(self.path_info, "a", encoding="utf-8") as f:
                f.write(("data,epochs,montage,method,source,seed,name,value\n" if new else "")
                        + "".join(f"{self.data},{self.epochs},{key[0]},{key[1]},{key[2]},{key[3]},{n},{v:.4f}\n" for n, v in info.items()))
        new = not os.path.exists(self.path)
        text = "".join(f"{self.data},{self.epochs},{key[0]},{key[1]},{key[2]},{key[3]},{t},{s},{n},{a:.6f},{extra:.4f}\n"
                       for (t, s), (n, a) in rows.items())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write((self.HEAD + "\n" if new else "") + text)


def jobs(method, names, tnames):
    """The trainings one method needs: [(key, source datasets, target datasets), ...]. Without pooling this is one
    training per source dataset, as in step 16; with pooling one training per target, on every other dataset."""
    if not SPECS[method].get("pool"):
        return [(s, [s], [t for t in tnames if t != s]) for s in names]
    res = []
    for t in tnames:
        src = [d for d in names if d != t]
        if len(src) >= 2:
            res.append(("not_" + t, src, [t]))
    return res


def screen(D, montage_sets, ch_names, methods, seeds, epochs, device, store):
    t0, n_done, failed = time.time(), 0, {}
    for montage, members in montage_sets.items():
        names = [d for d in members if d in D and len(D[d]["y"]) >= 3]
        tnames = names + ([EXT] if EXT in D else [])
        print(f"    {montage}: statistics of the users ...", flush=True)
        sets = prepare(D, montage, tnames)
        perm = mirror_perm(ch_names[montage])
        todo, n_all = [], 0
        for seed in seeds:
            for m in methods:
                for key, src, tg in jobs(m, names, tnames):
                    n_all += 1
                    if not store.has((montage, m, key, seed), sum(len(sets[t].y) for t in tg)):
                        todo.append((seed, key, src, tg, m))
        print(f"    {montage}: {len(todo)} trainings to do ({n_all - len(todo)} found in {os.path.basename(store.path or 'memory')})",
              flush=True)
        pools = {}
        for i, (seed, key, src, tg, m) in enumerate(todo):
            if m in failed:
                continue
            t1 = time.time()
            try:
                if len(src) > 1 and key not in pools:
                    pools[key] = Set.pooled([sets[d] for d in src])
                S = sets[src[0]] if len(src) == 1 else pools[key]
                rows, extra, info = run_method(m, S, [(t, sets[t]) for t in tg], seed, epochs, device, perm, ch_names[montage])
            except Exception as e:                                          # one method must not stop the others
                failed[m] = f"{montage}, source {key}, seed {seed}: {e!r}"
                out(f"    {m} FAILED and is skipped from here on ({failed[m]})")
                continue
            store.put((montage, m, key, seed), rows, extra, info)
            n_done += 1
            print(f"    {montage}: seed {seed}, source {key}, {m:<22} {time.time() - t1:5.0f} s | {i + 1}/{len(todo)} of this montage,"
                  f" {(time.time() - t0) / 60:.0f} min so far, about {(time.time() - t0) / n_done * (len(todo) - i - 1) / 60:.0f} min left",
                  flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
        del sets, pools
    return failed


# =========================== report ===========================
def per_subject(store, montage, method, seeds, combos=None):
    """{(target, subject): (mean accuracy over trainings, number of trainings)} and the (source, seed) combinations used."""
    have = sorted(k[2:] for k in store.res if k[0] == montage and k[1] == method and k[3] in seeds)
    use = have if combos is None else [c for c in have if c in combos]
    acc = {}
    for c in use:
        for key, a in store.res[(montage, method) + c].items():
            acc.setdefault(key, []).append(a)
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}, use


def contrast(store, montage, seeds, a, b):
    """Paired difference a - b over subjects, on the trainings both methods have: (benchmark, fifth dataset, note)
    or None. A pooled method has no training in common with a method trained on single sources; the difference is
    then taken between what each of them was trained on, and the line is marked."""
    ca, cb = per_subject(store, montage, a, seeds)[1], per_subject(store, montage, b, seeds)[1]
    common = [c for c in ca if c in cb]
    if common:
        va, vb, mark = per_subject(store, montage, a, seeds, common)[0], per_subject(store, montage, b, seeds, common)[0], ""
    elif ca and cb:
        va, vb, mark = per_subject(store, montage, a, seeds)[0], per_subject(store, montage, b, seeds)[0], "[other sources]"
    else:
        return None
    res = []
    for ext in (False, True):
        keys = sorted(k for k in va if k in vb and (k[0] == EXT) == ext)
        res.append(S5.fmt(S5.paired([va[k][0] for k in keys], [vb[k][0] for k in keys])) if len(keys) >= 3 else f"{'n/a':<45}")
    return res + [mark]


def step16_reference():
    ref = {}
    with open(os.path.join(OUTDIR, "step16_baselines.csv"), encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(head, line.strip().split(",")))
            if r["setting"] == "all":
                ref.setdefault((r["montage"], r["method"]), {})[(r["target"], r["subject"])] = (float(r["accuracy"]), 1)
    return ref


def report(store, montage_sets, methods, seeds, failed, with_step16=True):
    rows_csv = []
    for montage, members in montage_sets.items():
        acc = {m: per_subject(store, montage, m, seeds) for m in methods}
        acc = {m: v for m, v in acc.items() if v[0]}
        if not acc:
            continue
        targets = sorted({k[0] for v in acc.values() for k in v[0]})
        bench = [d for d in members if d in targets]
        have_ext = EXT in targets
        full = {}                                       # a pooled method has one training per target, the others one per source
        for m, (v, used) in acc.items():
            p = bool(SPECS[m].get("pool"))
            full[p] = max(full.get(p, 0), len(used))
        mean = lambda v, sel: 100 * float(np.mean([a[0] for k, a in v.items() if sel(k)])) if any(sel(k) for k in v) else float("nan")
        out(f"[M1] {montage}: accuracy % with all trials of the target subject, mean over subjects")
        out(f"      {'method':<24}{'component':<36}" + "".join(f"{d:>11}" for d in bench) + f"{'benchmark':>11}" + (f"{EXT:>12}" if have_ext else ""))
        for m, (v, used) in acc.items():
            n_full = full[bool(SPECS[m].get("pool"))]
            out(f"      {m:<24}{SPECS[m]['label']:<36}" + "".join(f"{mean(v, lambda k: k[0] == d):>11.2f}" for d in bench)
                + f"{mean(v, lambda k: k[0] != EXT):>11.2f}" + (f"{mean(v, lambda k: k[0] == EXT):>12.2f}" if have_ext else "")
                + ("" if len(used) == n_full else f"   ({len(used)} of {n_full} trainings)"))
            rows_csv += [(montage, m, k[0], k[1], a[0], a[1]) for k, a in sorted(v.items())]
        out()
        out(f"[M2] {montage}: differences in points [95% CI over subjects] dz p:   benchmark" + (f"   |   {EXT}" if have_ext else ""))
        for title, pairs in CONTRASTS:
            lines = []
            for a, b, note in pairs:
                res = contrast(store, montage, seeds, a, b) if a in acc and b in acc else None
                if res:
                    tail = " ".join(s for s in (note, res[2]) if s)
                    lines.append(f"      {a + ' - ' + b:<46}{res[0]}" + (f"   |  {res[1]}" if have_ext else "") + (f"   {tail}" if tail else ""))
            if lines:
                out(f"    {title}")
                for line in lines:
                    out(line)
        gam = {}
        for (mo, m, s, seed), g in store.extra.items():
            if mo == montage and m in acc and seed in seeds and np.isfinite(g) and SPECS[m].get("gamma") == "learn":
                gam.setdefault(m, []).append(g)
        if gam:
            out("    learned shrinkage gamma of the beamforming convolution (start 0.10; 1 = covariance not used), mean [range] over trainings")
            for m, g in gam.items():
                out(f"      {m:<38}{np.mean(g):.2f} [{min(g):.2f}, {max(g):.2f}]")
        for m in acc:
            vals = [v for k, v in store.info.items() if k[0] == montage and k[1] == m and k[3] in seeds]
            if vals:
                fs = sorted((n for n in vals[0] if n[0] == "f" and n[1:].isdigit()), key=lambda n: int(n[1:]))
                out(f"    {m}: learned centre frequencies in Hz (start 9, 12, ... 30), mean over trainings: "
                    + " ".join(f"{np.mean([v[n] for v in vals]):.1f}" for n in fs)
                    + f" | width {np.mean([v['width'] for v in vals]):.1f} Hz (start 4.0)"
                    + (f" | step eta {np.mean([v['eta'] for v in vals]):.2f} (start 0.50)" if "eta" in vals[0] else ""))
        out()
        if not with_step16:
            continue
        try:
            ref = step16_reference()
        except Exception as e:
            out(f"[M3] {montage}: step16_baselines.csv could not be read ({e!r}); no comparison with step 16")
            out()
            continue
        out(f"[M3] {montage}: against the run of step 16 (same protocol and subjects). Small differences are expected: other")
        out("      hardware and seeds, and a few trials without pre-cue segment.")
        out(f"      {'baseline':<38}{'benchmark: step 16 | here':>28}" + (f"{EXT + ': step 16 | here':>32}" if have_ext else ""))
        for m in BASELINES:
            r = ref.get((montage, m))
            if m not in acc or not r:
                continue
            out(f"      {m:<38}{mean(r, lambda k: k[0] != EXT):>18.2f} |{mean(acc[m][0], lambda k: k[0] != EXT):>6.2f}"
                + (f"{mean(r, lambda k: k[0] == EXT):>23.2f} |{mean(acc[m][0], lambda k: k[0] == EXT):>6.2f}" if have_ext else ""))
        prim = ref.get((montage, "ra_tslr"), {})
        if prim:
            out("      step 16, for orientation: " + ", ".join(
                f"{lab} {mean(ref[(montage, k)], lambda q: q[0] != EXT):.2f}" + (f" | {mean(ref[(montage, k)], lambda q: q[0] == EXT):.2f}" if have_ext else "")
                for k, lab in (("ra_tslr", "recentering + TS-LR"), ("mekt_r", "MEKT-R"), ("ea_csp", "EA + CSP + LDA")) if (montage, k) in ref))
            out("      new method minus recentering + TS-LR of step 16, in points [95% CI] dz p:   benchmark" + (f"   |   {EXT}" if have_ext else ""))
            for m, (v, _) in acc.items():
                if m in BASELINES:
                    continue
                res = []
                for ext in (False, True):
                    keys = sorted(k for k in v if k in prim and (k[0] == EXT) == ext)
                    res.append(S5.fmt(S5.paired([v[k][0] for k in keys], [prim[k][0] for k in keys])) if len(keys) >= 3 else "n/a")
                out(f"      {m:<38}{res[0]}" + (f"   |  {res[1]}" if have_ext else ""))
        out()
    if failed:
        out("methods that failed: " + "; ".join(f"{m} ({why})" for m, why in failed.items()))
        out()
    out("How to read this. A direction is worth building on if its method beats the standard of its family (eegnet_ea,")
    out("shallow_ea) in the benchmark AND in the fifth dataset by clearly more than the intervals, and if the control of the")
    out("same direction (eegnet_bf_off; the task reference; the network without mirror; modes that are not moved) does not")
    out("explain the gain.")
    out("A line marked [other sources] holds a pooled method against one trained on single source datasets: the subjects")
    out("behind the two numbers are the same, the trainings are not.")
    out(f"Seeds used here: {len(seeds)}. With one seed, differences below about one point should not be interpreted: the")
    out("intervals cover the variation over subjects, not the variation between trainings.")
    return rows_csv


# =========================== data ===========================
def match_trials(task, y_long, x_short, y_short):
    """Position of every long epoch among the cached task epochs (same recording order), or None if they do not correspond."""
    if len(task) == 0 or task.shape[1:] != x_short.shape[1:]:
        return None
    a = task.reshape(len(task), -1).astype(np.float64)
    b = x_short.reshape(len(x_short), -1).astype(np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-300
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-300
    sim, ys, idx, j0 = a @ b.T, np.asarray(y_short), [], 0
    for i, yi in enumerate(np.asarray(y_long)):
        cand = np.nonzero((sim[i, j0:] > 0.98) & (ys[j0:] == yi))[0]
        if len(cand) == 0:
            return None
        idx.append(j0 + int(cand[0]))
        j0 = idx[-1] + 1
    return np.array(idx)


def rest_of(ds, key, x, y, read_long):
    """Pre-cue segments of the cached task epochs x of one subject: (segments, positions of their trials in x)."""
    path = os.path.join(REST_CACHE, f"{ds}_{key}.npz")
    if os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if int(z["n_cached"]) == len(y):
                return z["rest"], z["idx"]
        except Exception:
            pass
    X, yl = read_long()
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[2] < TASK[1]:
        raise RuntimeError(f"{ds} {key}: epochs of {X.shape[-1]} samples; {TASK[1]} expected for -2.0 to 2.5 s at 250 Hz")
    ok = (np.isfinite(X).all(axis=(1, 2)) & (X[..., :500].var(axis=2).min(axis=1) > 0)
          & (X[..., REST[0]:REST[1]].var(axis=2).min(axis=1) > 0) & (X[..., TASK[0]:TASK[1]].var(axis=2).min(axis=1) > 0))
    X, yl = X[ok], np.asarray(yl)[ok]
    idx = match_trials(X[..., TASK[0]:TASK[1]], yl, x, y)
    if idx is None:
        raise RuntimeError(f"{ds} {key}: the epochs of -2.0 to 2.5 s do not correspond to the cached task epochs")
    rest = np.ascontiguousarray(X[..., REST[0]:REST[1]], dtype=np.float32)
    os.makedirs(REST_CACHE, exist_ok=True)
    tmp = path[:-4] + "_tmp.npz"
    np.savez(tmp, rest=rest, idx=idx, n_cached=np.array(len(y)))
    os.replace(tmp, path)
    return rest, idx


def read_ext_long(sid, folder, mods):
    """Epochs of -2.0 to 2.5 s of one participant of the fifth dataset, read as in step 16."""
    if folder is None:
        raise RuntimeError(f"recordings of {EXT} not found (give the folder of the database as first argument)")
    mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
    files = sorted(glob.glob(os.path.join(folder, f"{sid}_R*_*.gdf")),
                   key=lambda f: int(re.search(r"_R(\d+)_", os.path.basename(f)).group(1)))
    Xs, ys = [], []
    for f in files:
        try:
            raw = mne.io.read_raw_gdf(f, preload=True, verbose="ERROR")
            X, y = CD._epoch(CD._prep(raw, SM, 8.0, 30.0, 250), {"769": 0, "770": 1}, -2.0, 2.5)
            del raw
        except Exception:
            continue
        if X is None or X.shape[2] < TASK[1]:
            continue
        Xs.append(X[..., :TASK[1]])
        ys.append(np.asarray(y))
    if not Xs:
        raise RuntimeError(f"{EXT} {sid}: no recording could be read")
    return np.concatenate(Xs), np.concatenate(ys)


def load_real(root, want_rest):
    libs = S5._libs()
    mods = S9.project_modules()
    main = S8.load_cache()
    kept, folders = [], {}
    if os.path.isdir(S15.CACHE):
        for num, sid, folder in S15.list_subjects(root):
            rec = S15.load_subject(sid, folder, mods, libs)
            if int(rec["n_runs"]) >= 4 and min(int((rec["y"] == 0).sum()), int((rec["y"] == 1).sum())) >= 40:
                kept.append(dict(sid=sid, y=rec["y"]))
                folders[sid] = folder
    t0 = time.time()
    short = S9.make_loaders((8.0, 30.0), (0.5, 2.5))
    D = {}
    for ds in S5.DATASETS:
        D[ds] = dict(ids=[], key=[], x=[], y=[], rest=None)
        for r in main[ds]:
            x, y = S11.get_epochs(ds, r["subj"], mods, short)
            if not np.array_equal(y, r["y"]):
                raise RuntimeError(f"{ds} subject {r['subj']}: epochs do not match the covariance cache")
            D[ds]["ids"].append(r["subj"])
            D[ds]["key"].append(f"{r['subj']:03d}")
            D[ds]["x"].append(np.asarray(x, dtype=np.float32))
            D[ds]["y"].append(np.asarray(y))
    if kept:
        D[EXT] = dict(ids=[], key=[], x=[], y=[], rest=None)
        for r in kept:
            x, y = S16.ext_epochs(r["sid"], folders.get(r["sid"]), mods)
            if not np.array_equal(y, r["y"]):
                raise RuntimeError(f"{EXT} participant {r['sid']}: epochs do not match the covariance cache")
            D[EXT]["ids"].append(r["sid"])
            D[EXT]["key"].append(str(r["sid"]))
            D[EXT]["x"].append(np.asarray(x, dtype=np.float32))
            D[EXT]["y"].append(np.asarray(y))
    print(f"    task epochs ready ({time.time() - t0:.0f} s)", flush=True)
    note = "not used (--no-rest)"
    if want_rest:
        try:
            long = S9.make_loaders((8.0, 30.0), (-2.0, 2.5))
            R, n_all, n_kept = {}, 0, 0
            for ds in D:
                R[ds] = []
                for j, sid in enumerate(D[ds]["ids"]):
                    reader = (lambda: read_ext_long(sid, folders.get(sid), mods)) if ds == EXT else (lambda: long[ds](sid, mods)[:2])
                    R[ds].append(rest_of(ds, D[ds]["key"][j], D[ds]["x"][j], D[ds]["y"][j], reader))
                    n_all += len(D[ds]["y"][j])
                    n_kept += len(R[ds][-1][1])
                    if (j + 1) % 10 == 0 or j + 1 == len(D[ds]["ids"]):
                        print(f"    pre-cue segments: {ds} {j + 1}/{len(D[ds]['ids'])} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            for ds in D:
                D[ds]["x"] = [x[idx] for x, (_, idx) in zip(D[ds]["x"], R[ds])]
                D[ds]["y"] = [y[idx] for y, (_, idx) in zip(D[ds]["y"], R[ds])]
                D[ds]["rest"] = [rest for rest, _ in R[ds]]
            note = f"available; {n_all - n_kept} of {n_all} trials have none and are left out for all methods"
        except Exception as e:
            note = f"NOT available ({e!r}); the methods that need them are skipped"
    for ds in D:                                                            # amplitude as in step 16; the pre-cue segment keeps its ratio to the task segment
        for j, x in enumerate(D[ds]["x"]):
            s = np.sqrt(x.var(axis=2).mean(axis=1))[:, None, None]
            D[ds]["x"][j] = (x / s).astype(np.float32)
            if D[ds]["rest"] is not None:
                D[ds]["rest"][j] = (D[ds]["rest"][j] / s).astype(np.float32)
    return D, note


# =========================== self-test ===========================
def toy_data(seed=0, n_tr=40):
    """Toy sets with the structure of the real ones: two mirror-image sources; the one opposite to the imagined hand loses power."""
    rng = np.random.default_rng(seed)
    pos = np.array([[{"5": -3, "3": -2, "1": -1, "Z": 0, "2": 1, "4": 2, "6": 3}[re.match(r"^(FC|CP|C)(\d|Z)$", n).group(2)],
                     {"FC": 1, "C": 0, "CP": -1}[re.match(r"^(FC|CP|C)(\d|Z)$", n).group(1)]] for n in SM17], dtype=float)
    aL = np.exp(-((pos[:, 0] + 2) ** 2 + pos[:, 1] ** 2) / 2.0)
    aR = np.exp(-((pos[:, 0] - 2) ** 2 + pos[:, 1] ** 2) / 2.0)

    def band(n, T, lo, hi):
        f = np.fft.rfftfreq(T, 1 / 250.0)
        Z = np.fft.rfft(rng.standard_normal((n, T)), axis=1)
        Z[:, (f < lo) | (f > hi)] = 0
        z = np.fft.irfft(Z, T, axis=1)
        return z / z.std(axis=1, keepdims=True)

    D = {}
    for d, n_sub, gain in (("A", 4, 1.0), ("B", 4, 3.0), ("C", 4, 0.5), (EXT, 3, 2.0)):
        D[d] = dict(ids=[], x=[], y=[], rest=[])
        for s in range(n_sub):
            y = rng.permutation(np.arange(n_tr) % 2)
            asym = np.exp(0.3 * rng.standard_normal())
            Pn = rng.standard_normal((17, 3)) * np.exp(0.5 * rng.standard_normal(3))
            f0 = rng.uniform(9.5, 12.5)                                     # the rhythm of this subject

            def segment(T, task):
                sL, sR = band(n_tr, T, f0 - 1, f0 + 1) * asym, band(n_tr, T, f0 - 1, f0 + 1)
                if task:
                    sL = sL * np.where(y == 1, 0.5, 1.0)[:, None]          # right hand: the left hemisphere loses power
                    sR = sR * np.where(y == 0, 0.5, 1.0)[:, None]
                noise = np.einsum("ck,nkt->nct", Pn, np.stack([band(n_tr, T, 8, 30) for _ in range(3)], axis=1))
                return gain * (aL[None, :, None] * sL[:, None, :] + aR[None, :, None] * sR[:, None, :] + noise
                               + 0.3 * rng.standard_normal((n_tr, 17, T)))

            x, rest = segment(TASK[1] - TASK[0], True), segment(REST[1] - REST[0], False)
            sc = np.sqrt(x.var(axis=2).mean(axis=1))[:, None, None]
            D[d]["ids"].append(s + 1)
            D[d]["x"].append((x / sc).astype(np.float32))
            D[d]["rest"].append((rest / sc).astype(np.float32))
            D[d]["y"].append(y)
    return D


def selftest(device):
    print("self-test (toy data; nothing is written)")
    rng = np.random.default_rng(1)
    n, C, T, k = 6, 5, 200, 31
    x = rng.standard_normal((n, C, T))
    G = lagcov(x, k)
    direct = np.stack([np.einsum("nct,ndt->cd", x[:, :, :T - l], x[:, :, l:]) for l in range(k)]) / (n * T)
    assert np.allclose(G, direct, atol=1e-10), "lagged covariances"
    w = rng.standard_normal((2, k))
    wt, Gt = torch.as_tensor(w, dtype=torch.float32), torch.as_tensor(G[None], dtype=torch.float64)
    for norm in ("gain", "power"):
        beam = BeamConv(C, 2, 2, "band", "learn", norm)
        with torch.no_grad():
            Rc = beam.covariances(wt, Gt)[0].numpy()
            W = beam.weights(wt, Gt)[0].double().numpy()
            A, gam = beam.patterns().double().numpy(), float(beam.gamma()[0])
        for f in range(2):
            u = np.stack([[np.convolve(x[i, c], w[f], mode="full") for c in range(C)] for i in range(n)])
            R = np.einsum("nct,ndt->cd", u, u) / (n * T)                    # covariance of the filtered, zero-padded trials
            assert np.allclose(Rc[f], R, rtol=1e-5, atol=1e-8), "covariance after a temporal kernel"
            Rg = (1 - gam) * R / np.trace(R) * C + gam * np.eye(C)
            for d in range(2):
                ref = np.linalg.solve(Rg, A[f, d])
                assert np.allclose(W[f, d] / np.linalg.norm(W[f, d]), ref / np.linalg.norm(ref), atol=1e-4), "direction of the beamformer"
                if norm == "gain":
                    plain = A[f, d] / (A[f, d] @ A[f, d])
                    assert abs(W[f, d] @ A[f, d] - 1) < 1e-4 and W[f, d] @ Rg @ W[f, d] <= plain @ Rg @ plain + 1e-6, "unit gain, minimum power"
                else:
                    assert abs(W[f, d] @ R @ W[f, d] - 1) < 1e-3, "unit power for the user"
    print("  ok  lagged covariances; covariance after a learned kernel in closed form; beamformer with unit gain or unit power")
    net = ShallowRef(C, T, "user").double()
    with torch.no_grad():
        xp = Fn.pad(torch.as_tensor(x), (24, 24))
        direct = net.power(xp).sum(dim=(0, 2, 3)) / (n * T)
        closed = net.user_power(torch.as_tensor(lagcov(x, LAGS)[None]))[0]
    assert torch.allclose(direct, closed, rtol=1e-6, atol=1e-10), "reference power in closed form"
    print("  ok  reference power of the user in closed form")
    T2, perm = TASK[1] - TASK[0], mirror_perm(SM17)
    assert perm == [4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 16, 15, 14, 13, 12] and mirror_perm(["C3", "Cz", "C4"]) == [2, 1, 0]
    Pm = positions(SM17)
    assert np.allclose(Pm[perm, 0], -Pm[:, 0]) and np.allclose(Pm[perm, 1], Pm[:, 1]), "positions against the mirror image"
    assert np.allclose(positions(["C3", "CZ", "C4"]), [[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]), "positions of the three channels"
    assert basis_matrix(SM17).shape == (len(GRID_X) * len(GRID_Y), 17) and np.isfinite(basis_matrix(SM17)).all(), "position functions"
    g17 = build("eegnet_bf_geo", 17, T2, perm, SM17)
    g3 = build("eegnet_bf_geo", 3, T2, [2, 1, 0], ["C3", "CZ", "C4"])
    assert g17.beam.a.shape == g3.beam.a.shape, "patterns from positions still depend on the montage"
    z4 = rng.standard_normal((4, 17, T2)).astype(np.float32)
    two = Set.pooled([Set([1], [z4], [np.zeros(4, int)], None), Set([2], [z4], [np.ones(4, int)], None)])
    assert two.g["task"].shape[0] == 2 and len(two.x) == 2 and two.g["rest"] is None, "pooled sources"
    print("  ok  positions and their functions; patterns of one size for any montage; pooled sources")
    xs = rng.standard_normal((8, 17, T2)).astype(np.float32)
    rs = rng.standard_normal((8, 17, REST[1] - REST[0])).astype(np.float32)
    S1, S2 = Set([1], [xs], [np.zeros(8, int)], [rs]), Set([1], [xs[:, perm]], [np.zeros(8, int)], [rs[:, perm]])
    for name in ("eegnet_mirror", "eegnet_ea_mirror", "eegnet_bf_mirror"):
        torch.manual_seed(0)
        net = build(name, 17, T2, perm).eval()
        a, b = score(net, tensors(S1, name, "cpu", perm)), score(net, tensors(S2, name, "cpu", perm))
        assert np.allclose(a, -b, atol=1e-4 * (1 + np.abs(a).max())), f"mirror: {name}"
    print("  ok  mirror networks: exchanging the hemispheres exchanges the classes")
    tt = np.arange(T2) / 250.0
    tone = lambda f: np.stack([np.cos(2 * np.pi * (f + 0.3 * rng.standard_normal()) * tt + 6.28 * rng.random((17, 1))) for _ in range(8)])
    modes = ModeKernels(8, LAGS, True)
    with torch.no_grad():
        modes.eta_raw.fill_(20.0)                                           # full steps
        Gt = torch.as_tensor(np.stack([lagcov(tone(11.0)), lagcov(tone(13.5))]))
        fu = modes.user_freq(Gt).numpy()
    assert abs(fu[0, 1] - 11.0) < 0.5 and abs(fu[1, 1] - 13.5) < abs(12.0 - 13.5) and fu[1, 2] < 15.0, f"centre frequencies of the users: {fu[:, :3]}"
    torch.manual_seed(0)
    m1, m2 = build("eegnet_bf_mode", 17, T2, perm).eval(), build("eegnet_bf_mode_user", 17, T2, perm).eval()
    m2.load_state_dict(m1.state_dict())
    S3 = Set([1, 2], [xs[:4], xs[4:] * 2.0], [np.zeros(4, int)] * 2, None)
    with torch.no_grad():
        m2.modes.eta_raw.fill_(-50.0)                                       # steps of size zero: every user keeps the common kernels
        a, b = score(m1, tensors(S3, "eegnet_bf_mode", "cpu", perm)), score(m2, tensors(S3, "eegnet_bf_mode_user", "cpu", perm))
        m2.modes.eta_raw.fill_(0.0)
        c = score(m2, tensors(S3, "eegnet_bf_mode_user", "cpu", perm))
    assert np.allclose(a, b, atol=1e-4 * (1 + np.abs(a).max())) and np.isfinite(c).all(), "kernels of every user"
    print("  ok  mode kernels: centre frequencies follow the rhythm of the user; kernels of every user agree with common kernels")
    net = build("shallow_erd_trial", 17, T2, perm).eval()
    t = tensors(S1, "shallow_erd_trial", "cpu", perm)
    with torch.no_grad():
        a, b = net(t["x"], t["uid"], None, t["rest"]), net(7.0 * t["x"], t["uid"], None, 7.0 * t["rest"])
    assert torch.allclose(a, b, atol=1e-4), "reference from the same trial: gain"
    long = rng.standard_normal((12, 3, TASK[1])).astype(np.float32)
    cached = long[..., TASK[0]:TASK[1]] * (1 + 1e-6)
    yl = np.arange(12) % 2
    keep = np.array([0, 1, 2, 4, 5, 7, 8, 9, 11])
    assert np.array_equal(match_trials(long[keep][..., TASK[0]:TASK[1]], yl[keep], cached, yl), keep), "matching of the trials"
    assert match_trials(long[keep][..., TASK[0]:TASK[1]], 1 - yl[keep], cached, yl) is None
    print("  ok  reference of the same trial is free of the gain; pre-cue segments are matched to the cached trials")
    D = toy_data()
    store = Store(None, "toy", 8)
    members = {"17ch": ["A", "B", "C"], "3ch": ["A", "B", "C"]}
    failed = screen(D, members, {"17ch": SM17, "3ch": ["C3", "CZ", "C4"]}, list(SPECS), [0], 8, device, store)
    S5._LINES.clear()
    report(store, members, list(SPECS), [0], failed, with_step16=False)
    n_ok = len({k[1] for k in store.res})
    if failed or n_ok != len(SPECS):
        print(f"SELF-TEST FAILED: {len(SPECS) - n_ok} of {len(SPECS)} methods did not run")
        return 1
    print(f"self-test passed: all {len(SPECS)} methods trained and were tested on {device} (the toy accuracies mean nothing)")
    return 0


# =========================== main ===========================
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=S15.ROOT)
    ap.add_argument("--full", action="store_true", help="both montages, seeds 0 and 1")
    ap.add_argument("--montage", choices=["17ch", "3ch", "both"])
    ap.add_argument("--seeds", help="for example 0,1")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--methods", help="comma-separated; default: all")
    ap.add_argument("--no-rest", action="store_true", help="do not read pre-cue segments")
    ap.add_argument("--redo", action="store_true", help="train again what step23_raw.csv already holds")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if args.selftest:
        return selftest(device)
    methods = list(SPECS) if not args.methods else [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in SPECS]
    if unknown:
        print(f"unknown method(s): {', '.join(unknown)}. Known: {', '.join(SPECS)}")
        return 1
    epochs = args.epochs or (60 if cuda else 20)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else ([0, 1] if args.full else [0])
    mont = args.montage or ("both" if args.full else "17ch")
    montages = ["17ch", "3ch"] if mont == "both" else [mont]
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    t_start = time.time()
    S5._LINES.clear()
    out("Step 23 model screening  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    out(f"PyTorch {torch.__version__}, device: {device}" + ("" if cuda else "  (no GPU visible to PyTorch: this will be slow)"))
    D, note = load_real(args.root, not args.no_rest)
    have_rest = all(D[d]["rest"] is not None for d in D)
    out("subjects: " + ", ".join(f"{d} {len(D[d]['y'])}" for d in D) + ("" if EXT in D else f" | {EXT}: not available (run step 15 first)"))
    out(f"pre-cue segments: {note}")
    skipped = [m for m in methods if needs_rest(m) and not have_rest]
    methods = [m for m in methods if m not in skipped]
    out(f"settings: {epochs} epochs, seeds {seeds}, montages {montages}")
    out("methods: " + ", ".join(methods) + (f" | skipped (no pre-cue segments): {', '.join(skipped)}" if skipped else ""))
    out()
    store = Store(os.path.join(OUTDIR, "step23_raw.csv"), "pre" if have_rest else "task", epochs, args.redo)
    members = {m: S5.MONTAGES[m] for m in montages}
    try:
        import variants as V
        sm = list(V.SM)
    except Exception:
        sm = list(SM17)
    failed = screen(D, members, {"17ch": sm, "3ch": [sm[i] for i in S5.TRIPLET_IN_SM]}, methods, seeds, epochs, device, store)
    rows = report(store, members, list(SPECS), seeds, failed)
    S9.write_csv("step23_models.csv", "montage,method,target,subject,accuracy,trainings", rows)
    out()
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step23_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    print(f"written: {os.path.abspath(OUTDIR)} (step23_summary.txt, step23_models.csv, step23_raw.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
