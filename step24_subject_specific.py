"""
step24_subject_specific.py

Subject-specific decoding of left against right hand motor imagery: every model is trained and tested on the
trials of ONE subject, which is the protocol under which the published accuracies of EEGNet, ShallowConvNet,
ATCNet, EEG-Conformer and the classical pipelines are reported. Nothing is transferred between subjects here.

Run from the Codes folder:

    python step24_subject_specific.py --selftest      synthetic data, a few minutes; writes nothing
    python step24_subject_specific.py                 2a and 2b (18 subjects), both protocols
    python step24_subject_specific.py --datasets all  all five datasets (260 subjects; hours)

    other options: --models a,b,c   --protocol session|cv|both   --folds 5   --epochs N   --seeds 0,1
                   --band 4,38   --window 0,3   --redo   "D:\\other\\folder\\BCI Database"

Two protocols, reported side by side.
  session   the last recorded block of the subject is the test set, everything before it is the training set
            (2b: the third session; PhysioNet: the third run; the fifth dataset: the last run). Where the
            recordings hold one block only, or the last block holds one class only, the test set is the last
            third of the trials of each class in recording order; Cho2017 records the two classes in blocks, so
            the last third of all trials would hold one class. No trial of the test set is seen in training, and
            every test trial was recorded after the training trials of its own class.
  cv        stratified k-fold over the trials of the subject, shuffled (k = 5 by default). This is the protocol
            behind most of the published accuracies above 85 %; it is reported here for comparability, but
            trials recorded next to each other end up in the training and the test set, so it is optimistic.

Preprocessing, the same for every model and fitted inside the training set only.
  1. all EEG channels of the recording (EOG, ECG and EMG dropped), band-pass 4-38 Hz, resampled to 250 Hz
  2. epochs 0 to 3 s after the cue, no baseline correction; the trials keep their recording order and carry
     the index of the file (session or run) they come from
  3. the trials are split FIRST; the per-channel mean and standard deviation are then computed on the training
     trials of that subject alone and applied to both sets
  4. window-wise: every trial gives several windows of 2 s, moved by 0.2 s (6 windows per trial). The networks
     are trained on the windows; a test trial is decided by the mean of the outputs of its own windows. No
     window ever crosses the boundary between two trials, and all windows of a trial stay in the same set.

Models. The baselines are re-implementations from the descriptions in the papers named next to them, not the
authors' own code; every network is given the same training budget (optimizer, batch size and number of epochs),
so the comparison is between architectures and not between training schedules.
  csp_lda        common spatial patterns (4 components) and linear discriminant analysis, on 0.5 to 2.5 s
  fbcsp          filter-bank CSP (Ang et al. 2008): eight 4 Hz bands from 4 to 36 Hz, 4 CSP components in each,
                 the four features with the highest mutual information, LDA. The sub-bands are taken from the
                 epochs rather than from the continuous recording.
  ts_lr          OAS covariance, tangent space at the Riemannian mean of the training trials, logistic regression
  eegnet         EEGNet (Lawhern et al. 2018), trained on the windows
  shallow        ShallowConvNet (Schirrmeister et al. 2017), trained on the windows
  deepnet        DeepConvNet (Schirrmeister et al. 2017), four convolution blocks
  tcnet          EEG-TCNet (Ingolfsson et al. 2020): the EEGNet block followed by a temporal convolutional network
  conformer      EEG-Conformer (Song et al. 2023): convolutional embedding followed by a transformer encoder
  eegnet_cr      EEGNet trained with channel reflection (Wang et al. 2024): with probability one half a training
                 trial is replaced by its mirror image and its label is exchanged. The same prior knowledge as the
                 mirror construction, but on the data instead of in the network; the network itself is unchanged
                 and the symmetry holds only as far as the training reaches it. Train time only.
  shallow_cr     the same augmentation on the ShallowConvNet backbone
  shallow_mirror the mirror construction on the other backbone, to see whether it is tied to EEGNet
  eegnet_mirror  EEGNet with the mirror construction: the evidence for the right hand is g(x), for the left hand
                 g(x with the left and right hemisphere exchanged). Exchanging the hemispheres exchanges the two
                 classes exactly, so the network cannot prefer one side; with few trials per subject this prior
                 is worth more than it is in a large training set.

Outputs in corrected_results/: step24_summary.txt, step24_subject_specific.csv (one row per subject, protocol,
fold and model; an interrupted run resumes from it).
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
    import step9_reviewer_checks as S9
    import step15_external_validation as S15
except Exception as _e:                                    # pragma: no cover
    print(f"could not import the earlier step scripts (5, 9, 15); run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
out = S5.out
EXT = S15.NAME
FS = 250.0
DATASETS = ["2a", "2b", "PhysioNet", "Cho2017", EXT]
CROP, STRIDE = 500, 50                      # windows of 2 s, moved by 0.2 s
CLASSIC_WIN = (125, 625)                    # 0.5 to 2.5 s inside the epoch, for the two classical pipelines
CH_2A = ["FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
         "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"]      # the documented order of dataset 2a
MODELS = {
    "csp_lda":       "CSP (4 components) and LDA",
    "fbcsp":         "filter-bank CSP and LDA (Ang 2008)",
    "ts_lr":         "tangent space and logistic regression",
    "eegnet":        "EEGNet (Lawhern 2018)",
    "shallow":       "ShallowConvNet (Schirrmeister 2017)",
    "deepnet":       "DeepConvNet (Schirrmeister 2017)",
    "tcnet":         "EEG-TCNet (Ingolfsson 2020)",
    "conformer":     "EEG-Conformer (Song 2023)",
    "eegnet_cr":     "EEGNet with channel reflection (Wang 2024)",
    "shallow_cr":    "ShallowConvNet with channel reflection",
    "eegnet_mirror": "EEGNet with the mirror construction",
    "shallow_mirror": "ShallowConvNet with the mirror construction",
    "eegnet_rand":    "EEGNet, the construction with a random involution",
    "shallow_rand":   "ShallowConvNet, the construction with a random involution",
    "eegnet_two":     "EEGNet, two independent branches (the same compute, no symmetry)",
    "shallow_two":    "ShallowConvNet, two independent branches",
    "conformer_cr":   "EEG-Conformer with channel reflection",
    "conformer_mirror": "EEG-Conformer with the mirror construction",
}
DEEP = {"eegnet", "shallow", "deepnet", "tcnet", "conformer",
        "eegnet_cr", "shallow_cr", "eegnet_mirror", "shallow_mirror",
        "eegnet_rand", "shallow_rand", "eegnet_two", "shallow_two",
        "conformer_cr", "conformer_mirror"}
CONTRASTS = [("eegnet_mirror", "eegnet", "the mirror construction, same network"),
             ("shallow_mirror", "shallow", "the same construction on the other backbone"),
             ("shallow_mirror", "eegnet_mirror", "which backbone carries the construction better"),
             ("eegnet_cr", "eegnet", "the same symmetry as augmentation instead"),
             ("shallow_cr", "shallow", "the same, on the other backbone"),
             ("eegnet_mirror", "eegnet_cr", "built into the network or used as augmentation"),
             ("shallow_mirror", "shallow_cr", "the same question on the other backbone"),
             ("shallow_mirror", "ts_lr", "the best new model against the best classical one"),
             ("eegnet_mirror", "eegnet_rand", "the anatomical exchange against a random involution"),
             ("shallow_mirror", "shallow_rand", "the same control on the other backbone"),
             ("eegnet_mirror", "eegnet_two", "the construction against two branches of the same cost"),
             ("shallow_mirror", "shallow_two", "the same control on the other backbone"),
             ("shallow", "eegnet", ""),
             ("deepnet", "eegnet", ""),
             ("tcnet", "eegnet", ""),
             ("conformer", "eegnet", ""),
             ("fbcsp", "csp_lda", "filter bank against one band"),
             ("eegnet", "ts_lr", "network against the Riemannian pipeline"),
             ("eegnet_mirror", "conformer", "the new model against the most recent baseline"),
             ("eegnet_mirror", "ts_lr", "the new model against the best classical one")]


def cache_dir(band, win):
    return os.path.join(OUTDIR, f"cache_step24_{band[0]:g}-{band[1]:g}Hz_{win[0]:g}-{win[1]:g}s")


# =========================== channels ===========================
def norm_name(n):
    return re.sub(r"[^A-Z0-9]", "", n.upper().replace("EEG", ""))


def clean_names(names, fallback=None):
    """Channel names without the prefixes of the recording systems; `fallback` is used when the file numbers
    its channels instead of naming them (the GDF files of dataset 2a)."""
    res = [norm_name(n) for n in names]
    if fallback is not None and len(res) == len(fallback) and not all(re.match(r"^[A-Z]+(\d+|Z)$", s) for s in res):
        return list(fallback)
    return res


def mirror_index(names):
    """For every channel the index of its mirror image, or None if the montage has no complete mirror image."""
    up = [n.upper() for n in names]
    perm = []
    for n in up:
        m = re.match(r"^([A-Z]+?)(\d+|Z)$", n)
        if m is None:
            return None
        if m.group(2) == "Z":
            perm.append(up.index(n))
        else:
            k = int(m.group(2))
            twin = f"{m.group(1)}{k + 1 if k % 2 else k - 1}"
            if twin not in up:
                return None
            perm.append(up.index(twin))
    return perm


# =========================== reading the recordings ===========================
def labels_2a(cfg, key):
    """The class labels of the evaluation session of dataset 2a. The GDF of that session marks every cue with
    the same code and keeps the class to itself; the competition distributes the classes separately, as
    A01E.mat ... A09E.mat inside true_labels.zip. Returns the labels (1 = left hand, 2 = right hand, 3 = feet,
    4 = tongue) or None, in which case only the training session of that subject is used."""
    seen = []
    for folder in (getattr(cfg, "DATA_2A_LABELS", None), "true_labels",
                   os.path.join(cfg.DATA_2A, "true_labels"), cfg.DATA_2A):
        if not folder:
            continue
        p = os.path.join(folder, f"A{key:02d}E.mat")
        seen.append(p)
        if os.path.exists(p):
            try:
                from scipy.io import loadmat
                v = np.asarray(loadmat(p)["classlabel"]).ravel().astype(np.int64)
                return v if v.size else None
            except Exception as e:
                out(f"    2a {key}: {os.path.basename(p)} could not be read ({e!r}); the evaluation session is"
                    f" left out")
                return None
    return None


def _epochs_labelled(raw, mods, code, labels, win, n_t):
    """Epochs of a session whose cues all carry the same event code, with the classes supplied separately.
    Only the two hand classes are returned. None if the recording and the label file do not describe the same
    session, so that a mismatched label file can never silently produce wrong labels."""
    mne = mods["mne"]
    events, eid = mne.events_from_annotations(raw)
    if code not in eid:
        return None, None
    cue = events[events[:, 2] == eid[code]]
    if len(cue) == 0 or len(cue) != len(labels):
        return None, None
    ep = mne.Epochs(raw, cue, event_id=None, tmin=win[0], tmax=win[1], baseline=None, preload=True,
                    verbose="ERROR")
    if len(ep) == 0:
        return None, None
    y = np.asarray(labels, dtype=np.int64)[ep.selection]
    keep = (y == 1) | (y == 2)                                  # 1 = left hand, 2 = right hand
    if not keep.any():
        return None, None
    X = ep.get_data(copy=True).astype(np.float32)[keep]
    if X.shape[2] < n_t:
        return None, None
    return X[..., :n_t], y[keep] - 1


def _epochs(raw, mods, event_map, win, n_t):
    mne = mods["mne"]
    events, eid = mne.events_from_annotations(raw)
    want = {k: eid[k] for k in event_map if k in eid}
    if not want:
        return None, None
    ep = mne.Epochs(raw, events, event_id=want, tmin=win[0], tmax=win[1], baseline=None, preload=True, verbose="ERROR")
    if len(ep) == 0:
        return None, None
    id2c = {eid[k]: event_map[k] for k in want}
    y = np.array([id2c[e] for e in ep.events[:, 2]], dtype=np.int64)
    X = ep.get_data(copy=True).astype(np.float32)
    if X.shape[2] < n_t:
        return None, None
    return X[..., :n_t], y


def _prep(raw, mods, band, average_reference=False):
    drop = [c for c in raw.ch_names if any(t in c.upper() for t in ("EOG", "ECG", "EMG", "STI", "STATUS", "TRIGGER"))]
    if drop:
        raw.drop_channels(drop)
    if average_reference:
        keep = [c for c in raw.ch_names if norm_name(c) in mods["std_names"]]
        if keep:
            raw.pick(keep)
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.filter(band[0], band[1], verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - FS) > 1e-6:
        raw.resample(FS, verbose="ERROR")
    return raw


def subject_files(ds, key, mods, root):
    """The recordings of one subject, in recording order; every file is one block (session or run)."""
    cfg = mods["config"]
    if ds == "2a":
        files = [f for f in [os.path.join(cfg.DATA_2A, f"A{key:02d}T.gdf")] if os.path.exists(f)]
        ev = os.path.join(cfg.DATA_2A, f"A{key:02d}E.gdf")      # the evaluation session, only with its labels
        if files and os.path.exists(ev) and labels_2a(cfg, key) is not None:
            files.append(ev)
        return files
    if ds == "2b":
        return [f for f in [os.path.join(cfg.DATA_2B, f"B{key:02d}0{s}T.gdf") for s in (1, 2, 3)] if os.path.exists(f)]
    if ds == "PhysioNet":
        return [f for f in [os.path.join(cfg.DATA_PHYSIONET, "files", f"S{key:03d}", f"S{key:03d}R{r:02d}.edf")
                            for r in S5.PN_RUNS] if os.path.exists(f)]
    if ds == "Cho2017":
        return sorted(glob.glob(os.path.join(cfg.DATA_HGD, f"sub-{key}", "**", f"sub-{key}_*task-imagery*_eeg.bdf"),
                                recursive=True))
    folder = root.get(key) if isinstance(root, dict) else None
    if folder is None:
        return []
    return sorted(glob.glob(os.path.join(folder, f"{key}_R*_*.gdf")),
                  key=lambda f: int(re.search(r"_R(\d+)_", os.path.basename(f)).group(1)))


def read_subject(ds, key, mods, band, win, root):
    """(X, y, block, names) of one subject: trials in recording order, with the index of the file they come from."""
    path = os.path.join(cache_dir(band, win), f"{ds}_{key}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return z["X"], z["y"], z["block"], [str(s) for s in z["names"]]
    files = subject_files(ds, key, mods, root)
    if not files:
        raise RuntimeError(f"{ds} {key}: no recording found (check the paths in config.py)")
    n_t = int(round((win[1] - win[0]) * FS)) + 1
    emap = {"T1": 0, "T2": 1} if ds == "PhysioNet" else (S5.CHO_EVENTS if ds == "Cho2017" else {"769": 0, "770": 1})
    reader = {"2a": "gdf", "2b": "gdf", EXT: "gdf", "PhysioNet": "edf", "Cho2017": "bdf"}[ds]
    Xs, ys, bl, names = [], [], [], None
    for b, f in enumerate(files):
        try:
            raw = getattr(mods["mne"].io, f"read_raw_{reader}")(f, preload=True, verbose="ERROR")
            raw = _prep(raw, mods, band, average_reference=(ds == "Cho2017"))
            nm = clean_names(raw.ch_names, CH_2A if ds == "2a" else None)
            if ds == "2a" and os.path.basename(f).upper().endswith("E.GDF"):
                X, y = _epochs_labelled(raw, mods, "783", labels_2a(mods["config"], key), win, n_t)
            else:
                X, y = _epochs(raw, mods, emap, win, n_t)
            del raw
        except Exception:
            continue
        if X is None:
            continue
        if names is None:
            names = nm
        if nm != names:                                     # keep only files with the same channels
            continue
        ok = np.isfinite(X).all(axis=(1, 2)) & (X.var(axis=2).min(axis=1) > 0)
        if not ok.any():
            continue
        Xs.append(X[ok])
        ys.append(y[ok])
        bl.append(np.full(int(ok.sum()), b, dtype=np.int64))
    if not Xs:
        raise RuntimeError(f"{ds} {key}: no usable trials")
    X, y, block = np.concatenate(Xs), np.concatenate(ys), np.concatenate(bl)
    os.makedirs(cache_dir(band, win), exist_ok=True)
    tmp = path[:-4] + "_tmp.npz"
    np.savez(tmp, X=X.astype(np.float32), y=y, block=block, names=np.array(names))
    os.replace(tmp, path)
    return X.astype(np.float32), y, block, names


def subject_keys(ds, mods, root):
    if ds == "2a" or ds == "2b":
        return [s for s in range(1, 10) if subject_files(ds, s, mods, root)]
    if ds == "PhysioNet":
        return [s for s in range(1, 110) if len(subject_files(ds, s, mods, root)) == len(S5.PN_RUNS)]
    if ds == "Cho2017":
        return [s for s in range(1, 53) if subject_files(ds, s, mods, root)]
    return sorted(root.keys()) if isinstance(root, dict) else []


# =========================== splits and windows ===========================
def split_session(y, block, min_test=10, min_class=4):
    """Index of the test trials: the last block of the subject, or the last third of the trials in recording
    order when the recordings hold one block only. Returns None if the split would be too small or one-sided.
    The limits are low because a run of the PhysioNet database holds only about 15 trials of the two classes."""
    y, b = np.asarray(y), np.asarray(block)
    test = None
    if b.max() > 0:                                     # more than one file: the last one, if it holds both classes
        cand = np.nonzero(b == b.max())[0]
        if min(int((y[cand] == c).sum()) for c in (0, 1)) >= min_class:
            test = cand
    if test is None:                                    # one file: the last third of the trials OF EACH CLASS, in
        test = np.sort(np.concatenate(                  # recording order. Taking the last third of all trials would
            [np.nonzero(y == c)[0][int(int((y == c).sum()) * 2 / 3):] for c in (0, 1)]))   # give one class only in a
    train = np.setdiff1d(np.arange(len(y)), test)       # dataset whose classes are recorded in blocks (Cho2017).
    if len(test) < min_test or len(train) < 2 * min_test:
        return None
    for idx in (train, test):
        if min(int((y[idx] == c).sum()) for c in (0, 1)) < min_class:
            return None
    return train, test


def subsample(train, test, y, frac, min_class=4, seed=0):
    """A fixed draw of a fraction of the training trials, the same fraction from each class. The test set is
    returned untouched, so every fraction is scored on exactly the same trials and the curve is comparable.
    None if a class would be left with fewer than `min_class` training trials."""
    if frac >= 1.0:
        return train, test
    rng = np.random.default_rng(seed)
    train, y = np.asarray(train), np.asarray(y)
    keep = []
    for c in (0, 1):
        idx = train[y[train] == c]
        n = int(round(len(idx) * frac))
        if n < min_class:
            return None
        keep.append(rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(keep)), test


def split_cv(y, folds, seed=0):
    """Stratified folds over the trials, shuffled: [(train, test), ...]."""
    rng = np.random.default_rng([24, seed])
    part = [[] for _ in range(folds)]
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            part[i % folds].append(j)
    res = []
    for f in range(folds):
        test = np.sort(np.array(part[f], dtype=int))
        train = np.setdiff1d(np.arange(len(y)), test)
        if len(test) >= 4 and len(np.unique(y[test])) == 2 and len(np.unique(y[train])) == 2:
            res.append((train, test))
    return res


def standardize(X, train):
    """Per channel, from the training trials only."""
    mu = X[train].mean(axis=(0, 2), keepdims=True)
    sd = X[train].std(axis=(0, 2), keepdims=True) + 1e-7
    return ((X - mu) / sd).astype(np.float32)


def windows(X, idx):
    """All windows of the listed trials: (n_windows, C, CROP) and, for every window, the trial it belongs to."""
    starts = list(range(0, X.shape[2] - CROP + 1, STRIDE)) or [0]
    w = np.stack([X[i][:, s:s + CROP] for i in idx for s in starts])
    owner = np.repeat(np.arange(len(idx)), len(starts))
    return np.ascontiguousarray(w), owner


# =========================== models ===========================
class EEGNet(nn.Module):
    """EEGNet exactly as in step 16; n_out = 1 gives the one-output network of the mirror model."""

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

    def forward(self, x):
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

    def forward(self, x):
        return self.fc(self._features(x.unsqueeze(1)))


class DeepConvNet(nn.Module):
    """DeepConvNet (Schirrmeister et al. 2017): four convolution blocks, max pooling by 3."""

    def __init__(self, C, T, n_out=2, p=0.5):
        super().__init__()
        blocks = [nn.Conv2d(1, 25, (1, 10)), nn.Conv2d(25, 25, (C, 1), bias=False),
                  nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)), nn.Dropout(p)]
        for cin, cout in ((25, 50), (50, 100), (100, 200)):
            blocks += [nn.Conv2d(cin, cout, (1, 10), bias=False), nn.BatchNorm2d(cout), nn.ELU(),
                       nn.MaxPool2d((1, 3), stride=(1, 3)), nn.Dropout(p)]
        self.net = nn.Sequential(*blocks)
        with torch.no_grad():
            n = self.net(torch.zeros(1, 1, C, T)).numel()
        self.fc = nn.Linear(n, n_out)

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1)).flatten(1))


class TCNBlock(nn.Module):
    """One residual block of a temporal convolutional network: two causal dilated convolutions."""

    def __init__(self, cin, cout, k, d, p):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(cin, cout, k, dilation=d)
        self.c2 = nn.Conv1d(cout, cout, k, dilation=d)
        self.drop = nn.Dropout(p)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        z = self.drop(Fn.elu(self.c1(Fn.pad(x, (self.pad, 0)))))
        z = self.drop(Fn.elu(self.c2(Fn.pad(z, (self.pad, 0)))))
        return Fn.elu(z + (x if self.down is None else self.down(x)))


class EEGTCNet(nn.Module):
    """EEG-TCNet (Ingolfsson et al. 2020): the EEGNet block, then a temporal convolutional network."""

    def __init__(self, C, T, n_out=2, F1=8, D=2, F2=16, k1=64, p=0.3, f=12, k=4, pt=0.3):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p))
        self.b2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p))
        self.t1, self.t2 = TCNBlock(F2, f, k, 1, pt), TCNBlock(f, f, k, 2, pt)
        self.fc = nn.Linear(f, n_out)

    def forward(self, x):
        z = self.b2(self.b1(x.unsqueeze(1)))[:, :, 0, :]                # (B, F2, time)
        return self.fc(self.t2(self.t1(z))[:, :, -1])


class Conformer(nn.Module):
    """EEG-Conformer (Song et al. 2023): the convolutional embedding of ShallowConvNet turned into a sequence of
    tokens, a transformer encoder over the tokens, and a small fully connected head."""

    def __init__(self, C, T, n_out=2, k=40, depth=6, heads=10, p=0.5):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Conv2d(1, k, (1, 25)), nn.Conv2d(k, k, (C, 1), bias=False), nn.BatchNorm2d(k), nn.ELU(),
            nn.AvgPool2d((1, 75), stride=(1, 15)), nn.Dropout(p))
        layer = nn.TransformerEncoderLayer(d_model=k, nhead=heads, dim_feedforward=4 * k, dropout=p,
                                           activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        with torch.no_grad():
            n = self.emb(torch.zeros(1, 1, C, T)).shape[-1] * k
        self.head = nn.Sequential(nn.Linear(n, 256), nn.ELU(), nn.Dropout(0.3),
                                  nn.Linear(256, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, n_out))

    def forward(self, x):
        z = self.emb(x.unsqueeze(1))[:, :, 0, :].transpose(1, 2)        # (B, tokens, k)
        return self.head(self.enc(z).flatten(1))


class Mirror(nn.Module):
    """Right hand gets g(x), left hand g(mirror image of x); g has one output."""

    def __init__(self, core, perm):
        super().__init__()
        self.core = core
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def forward(self, x):
        B = x.shape[0]
        o = self.core(torch.cat([x, x[:, self.perm]]))[:, 0]
        return torch.stack([o[B:], o[:B]], dim=1)


class TwoHeads(nn.Module):
    """Two independent copies of the backbone, one per class. It costs exactly what the mirror
    construction costs -- two evaluations of the backbone per step -- and carries no symmetry, so it
    separates the effect of the constraint from the effect of the extra compute."""

    def __init__(self, core_left, core_right):
        super().__init__()
        self.left, self.right = core_left, core_right

    def forward(self, x):
        return torch.cat([self.left(x), self.right(x)], dim=1)


def random_involution(C, seed):
    """A permutation of the channels that is its own inverse but has nothing to do with the head: the
    channels are paired at random (one is left fixed when C is odd). The control for the claim that the
    gain comes from the anatomical exchange and not from the shape of the computation."""
    rng = np.random.default_rng(1000 + seed)
    order = rng.permutation(C)
    perm = np.arange(C)
    for i in range(0, C - 1, 2):
        a, b = int(order[i]), int(order[i + 1])
        perm[a], perm[b] = b, a
    return perm.tolist()


def fit_deep(model, Xw, yw, C, T, perm, seed, epochs, device, lr=1e-3, batch=64, curve=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cr = model.endswith("_cr")                                          # channel reflection, applied to the batches
    base = model[:-3] if cr else model
    if base == "shallow":
        net = Shallow(C, T, 2)
    elif base == "deepnet":
        net = DeepConvNet(C, T, 2)
    elif base == "tcnet":
        net = EEGTCNet(C, T, 2)
    elif base == "conformer":
        net = Conformer(C, T, 2)
    elif base == "eegnet_mirror":
        net = Mirror(EEGNet(C, T, 1), perm)
    elif base == "shallow_mirror":
        net = Mirror(Shallow(C, T, 1), perm)
    elif base == "conformer_mirror":          # a third backbone, architecturally unlike the other two
        net = Mirror(Conformer(C, T, 1), perm)
    elif base == "eegnet_rand":
        net = Mirror(EEGNet(C, T, 1), random_involution(C, seed))
    elif base == "shallow_rand":
        net = Mirror(Shallow(C, T, 1), random_involution(C, seed))
    elif base == "eegnet_two":
        net = TwoHeads(EEGNet(C, T, 1), EEGNet(C, T, 1))
    elif base == "shallow_two":
        net = TwoHeads(Shallow(C, T, 1), Shallow(C, T, 1))
    else:
        net = EEGNet(C, T, 2)
    net = net.to(device)
    pt = torch.as_tensor(perm, dtype=torch.long, device=device) if cr else None
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(Xw, dtype=torch.float32, device=device)
    yt = torch.as_tensor(yw, dtype=torch.long, device=device)
    n = len(yt)
    for ep in range(epochs):
        net.train()
        run_loss, run_n = 0.0, 0
        order = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            if len(idx) < 2:
                continue
            xb, yb = Xt[idx], yt[idx]
            if pt is not None:                                          # mirror half of the batch, exchange its labels
                flip = torch.rand(len(idx), device=device) < 0.5
                xb = torch.where(flip[:, None, None], xb[:, pt], xb)
                yb = torch.where(flip, 1 - yb, yb)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("the loss is not finite")
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * len(idx)
            run_n += len(idx)
        if curve is not None and run_n:
            curve.append((ep, run_loss / run_n))
    net.eval()
    del Xt, yt

    def predict(Xq, owner, n_trials):
        """One score per trial: the sum of the outputs of its own windows. The class is score > 0."""
        s = np.zeros(n_trials)
        with torch.no_grad():
            for i in range(0, len(Xq), 1024):
                o = net(torch.as_tensor(Xq[i:i + 1024], dtype=torch.float32, device=device))
                np.add.at(s, owner[i:i + 1024], (o[:, 1] - o[:, 0]).float().cpu().numpy())
        return s
    return predict


def run_fbcsp(X, y, train, test, perm=None, n_comp=4, n_sel=4):
    """Filter-bank CSP (Ang et al. 2008): CSP in every 4 Hz band, the features with the highest mutual
    information, LDA. Everything after the band-pass is fitted on the training trials only."""
    from scipy.signal import butter, sosfiltfilt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import mutual_info_classif
    from mne.decoding import CSP
    Ftr, Fte, Fmi = [], [], []
    for lo in range(4, 36, 4):
        sos = butter(4, [lo / (FS / 2), (lo + 4) / (FS / 2)], btype="band", output="sos")
        Z = np.ascontiguousarray(sosfiltfilt(sos, X, axis=2)[:, :, CLASSIC_WIN[0]:CLASSIC_WIN[1]]).astype(np.float64)
        csp = CSP(n_components=min(n_comp, X.shape[1]), log=True, norm_trace=False, reg="oas")
        Ftr.append(csp.fit_transform(Z[train], y[train]))
        Fte.append(csp.transform(Z[test]))
        if perm is not None:
            Fmi.append(csp.transform(np.ascontiguousarray(Z[test][:, perm])))
    Ftr, Fte = np.concatenate(Ftr, axis=1), np.concatenate(Fte, axis=1)
    mi = mutual_info_classif(Ftr, y[train], random_state=0)
    keep = np.argsort(mi)[::-1][:min(n_sel, Ftr.shape[1])]
    clf = LinearDiscriminantAnalysis().fit(Ftr[:, keep], y[train])
    pm = clf.predict(np.concatenate(Fmi, axis=1)[:, keep]).astype(int) if perm is not None else None
    return clf.predict(Fte[:, keep]).astype(int), pm, clf.decision_function(Fte[:, keep]).ravel()


def run_classic(model, X, y, train, test, libs, perm=None):
    """CSP + LDA, filter-bank CSP, or tangent space + logistic regression; all on 0.5 to 2.5 s and fitted on the
    training trials. Returns the predictions for the test trials, the predictions of the same fitted model for
    those trials with the hemispheres exchanged (if perm is given), and the real-valued decision function,
    which is what an ROC curve needs."""
    if model == "fbcsp":
        return run_fbcsp(X, y, train, test, perm)
    Covariances, _, mean_riemann, tangent_space, LogisticRegression = libs
    Z = np.ascontiguousarray(X[:, :, CLASSIC_WIN[0]:CLASSIC_WIN[1]]).astype(np.float64)
    Zm = np.ascontiguousarray(Z[test][:, perm]) if perm is not None else None
    if model == "ts_lr":
        Cv = Covariances(estimator="oas").fit_transform(Z)
        Cv = Cv / np.trace(Cv, axis1=1, axis2=2)[:, None, None]
        M = mean_riemann(Cv[train])
        F = tangent_space(Cv, M)
        clf = LogisticRegression(max_iter=1000).fit(F[train], y[train])
        pm = None
        if Zm is not None:
            Cm = Covariances(estimator="oas").fit_transform(Zm)
            Cm = Cm / np.trace(Cm, axis1=1, axis2=2)[:, None, None]
            pm = clf.predict(tangent_space(Cm, M)).astype(int)
        return clf.predict(F[test]).astype(int), pm, clf.decision_function(F[test]).ravel()
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    csp = CSP(n_components=min(4, Z.shape[1]), log=True, norm_trace=False, reg="oas")
    Ftr = csp.fit_transform(Z[train], y[train])
    clf = LinearDiscriminantAnalysis().fit(Ftr, y[train])
    pm = clf.predict(csp.transform(Zm)).astype(int) if Zm is not None else None
    Fte = csp.transform(Z[test])
    return clf.predict(Fte).astype(int), pm, clf.decision_function(Fte).ravel()


# =========================== bookkeeping ===========================
def kappa(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    p0 = float(np.mean(pred == y))
    pe = sum(float(np.mean(pred == c)) * float(np.mean(y == c)) for c in (0, 1))
    return (p0 - pe) / (1.0 - pe) if pe < 1.0 else 0.0


def consistency(pred, pred_mirror):
    """Share of test trials on which exchanging the two hemispheres of the input exchanges the predicted class.
    Read it together with `asymmetry`: a trial whose score sits at zero flips on numerical noise alone, so this
    share is below one even for a model that is equivariant in exact arithmetic."""
    if pred_mirror is None:
        return float("nan")
    return float(np.mean(np.asarray(pred_mirror) == 1 - np.asarray(pred)))


def asymmetry(score, score_mirror):
    """How far the decision function is from exchanging its two classes when the hemispheres are exchanged:
    mean |s + s'| / mean (|s| + |s'|). Zero, down to the precision of the arithmetic, for the mirror construction,
    which satisfies s' = -s for every input by construction; a sizeable number for anything that reaches the same
    symmetry only through training. Unlike the share of exchanged predictions, this does not depend on how many
    trials happen to sit close to the decision threshold."""
    if score_mirror is None:
        return float("nan")
    s, sm = np.asarray(score, dtype=np.float64), np.asarray(score_mirror, dtype=np.float64)
    den = float(np.mean(np.abs(s) + np.abs(sm)))
    return float(np.mean(np.abs(s + sm)) / den) if den > 0 else float("nan")


def lean(pred, y):
    """Recall of the right hand minus recall of the left hand: how far the decoder leans to one side."""
    r = []
    for c in (0, 1):
        m = np.asarray(y) == c
        r.append(float(np.mean(np.asarray(pred)[m] == c)) if m.any() else float("nan"))
    return r[1] - r[0]


class Store:
    HEAD = "band,window,dataset,subject,protocol,fold,model,seed,n_test,accuracy,kappa,consistency,lean,asymmetry"

    def __init__(self, path, tag, redo=False):
        self.path, self.tag, self.redo = path, tag, redo
        self.res, self.fresh = {}, set()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.rstrip("\n").split(",")
                    if len(p) < 11 or (p[0], p[1]) != self.tag:        # lines of the earlier format have 11 fields
                        continue
                    nan = float("nan")

                    def num(i, p=p, nan=nan):                          # columns added later may be absent
                        return float(p[i]) if len(p) > i and p[i] != "" else nan
                    self.res[(p[2], p[3], p[4], int(p[5]), p[6], int(p[7]))] = (
                        int(p[8]), float(p[9]), float(p[10]), num(11), num(12), num(13))

    def has(self, key):
        return key in self.res and (not self.redo or key in self.fresh)

    def put(self, key, n, acc, kap, cons=float("nan"), ln=float("nan"), asym=float("nan")):
        self.res[key] = (n, acc, kap, cons, ln, asym)
        self.fresh.add(key)
        if not self.path:
            return
        new = not os.path.exists(self.path)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write((self.HEAD + "\n" if new else "")
                    + f"{self.tag[0]},{self.tag[1]},{key[0]},{key[1]},{key[2]},{key[3]},{key[4]},{key[5]},"
                      f"{n},{acc:.6f},{kap:.6f},{cons:.6f},{ln:.6f},{asym:.3e}\n")

    def per_subject(self, dataset, protocol, model, col=1):
        """{subject: mean over folds and seeds} for one dataset, protocol and model; rows whose column was not
        measured (the earlier format has no consistency) are left out."""
        acc = {}
        for k, v in self.res.items():
            if k[0] == dataset and k[2] == protocol and k[4] == model and np.isfinite(v[col]):
                acc.setdefault(k[1], []).append(v[col])
        return {s: float(np.mean(v)) for s, v in acc.items()}


# =========================== one subject ===========================
def do_subject(ds, key, mods, band, win, root, models, protocols, folds, seeds, epochs, device, libs, store,
               scores_path=None, curves_path=None):
    X, y, block, names = read_subject(ds, key, mods, band, win, root)
    perm = mirror_index(names)
    todo, n_parts = [], 0
    for protocol in protocols:
        base, _, frac = protocol.partition("@")          # "session@0.25": a quarter of the training trials
        parts = []
        if base == "session":
            sp = split_session(y, block)
            if sp is not None:
                parts = [(0, sp)]
        else:
            parts = list(enumerate(split_cv(y, folds)))
        if frac:
            parts = [(f, sp) for f, sp in ((f, subsample(tr, te, y, float(frac))) for f, (tr, te) in parts)
                     if sp is not None]
        n_parts += len(parts)
        for fold, (train, test) in parts:
            for model in models:
                for seed in (seeds if model in DEEP else [0]):
                    k = (ds, str(key), protocol, fold, model, seed)
                    if not store.has(k):
                        todo.append((k, train, test))
    if not todo:
        return 0, 0.0, n_parts
    t0 = time.time()
    done = 0
    cache = {}
    for k, train, test in todo:
        model, seed = k[4], k[5]
        t1 = time.time()
        curve = None
        try:
            if perm is None and (model.endswith("_mirror") or model.endswith("_cr")):
                continue                                                # this montage has no complete mirror image
            ZZ = cache.get(k[2:4])
            if ZZ is None:
                Z = standardize(X, train)
                ZZ = (Z, np.ascontiguousarray(Z[:, perm]) if perm is not None else None)
                cache[k[2:4]] = ZZ
            Z, Zm = ZZ
            asym = float("nan")
            if model in DEEP:
                Xw, ow = windows(Z, train)
                curve = [] if curves_path else None
                pred_fn = fit_deep(model, Xw, y[train][ow], Z.shape[1], CROP, perm, seed, epochs, device,
                                   curve=curve)
                Xq, oq = windows(Z, test)
                sc = pred_fn(Xq, oq, len(test))                          # one real-valued score per test trial
                pred, pm, scm = (sc > 0).astype(int), None, None
                if Zm is not None:
                    Xi, oi = windows(Zm, test)
                    scm = pred_fn(Xi, oi, len(test))                     # the same model, hemispheres exchanged
                    pm, asym = (scm > 0).astype(int), asymmetry(sc, scm)
            else:
                pred, pm, sc = run_classic(model, Z, y, train, test, libs, perm)
                scm = None
            store.put(k, len(test), float(np.mean(pred == y[test])), kappa(pred, y[test]),
                      consistency(pred, pm), lean(pred, y[test]), asym)
            if scores_path:                                              # one line per test trial: for ROC
                with open(scores_path, "a", encoding="utf-8") as f:
                    mv = np.asarray(scm, dtype=np.float64) if scm is not None else None
                    for j, (yt, sv) in enumerate(zip(y[test], np.asarray(sc, dtype=np.float64))):
                        m = f"{mv[j]:.6f}" if mv is not None else ""
                        f.write(f"{ds},{key},{k[2]},{k[3]},{model},{seed},{j},{int(yt)},{sv:.6f},{m}\n")
            if curve:                                                    # one line per epoch: for the loss curve
                with open(curves_path, "a", encoding="utf-8") as f:
                    for ep, lv in curve:
                        f.write(f"{ds},{key},{k[2]},{k[3]},{model},{seed},{ep},{lv:.6f}\n")
            done += 1
            if model in DEEP:                       # free the network of this fit before the next one is built
                pred_fn = None
                if device == "cuda":
                    torch.cuda.empty_cache()
            mem = (f", {torch.cuda.memory_reserved() / 2 ** 20:.0f} MiB held" if device == "cuda" else "")
            print(f"      {ds} {key} {model} {k[2]}/{k[3]} seed {seed}: {time.time() - t1:.0f} s{mem}", flush=True)
        except Exception as e:                                          # one model must not stop the others
            out(f"    {ds} {key}, {model}, {k[2]} fold {k[3]}: FAILED ({e!r})")
    return done, time.time() - t0, n_parts


# =========================== report ===========================
def report(store, datasets, models, protocols):
    for protocol in protocols:
        base, _, frac = protocol.partition("@")
        name = {"session": "the last block held out", "cv": "stratified k-fold over shuffled trials"}[base]
        if frac:
            name += f", trained on {100 * float(frac):g} % of the training trials (the test set is unchanged)"
        out(f"[S1] {protocol}: {name}. Accuracy %, mean over subjects (kappa in brackets, then the number of")
        out("      subjects the mean is taken over -- every subject in the file, not only the ones fitted today)")
        out(f"      {'model':<16}{'component':<42}" + "".join(f"{d:>21}" for d in datasets))
        for m in models:
            cells = ""
            for d in datasets:
                a, kp = store.per_subject(d, protocol, m, 1), store.per_subject(d, protocol, m, 2)
                cells += (f"{100 * np.mean(list(a.values())):>10.2f} ({np.mean(list(kp.values())):.2f}) {len(a):>3}"
                          if a else f"{'-':>21}")
            out(f"      {m:<16}{MODELS[m]:<42}{cells}")
        out()
        out(f"[S3] {protocol}: mirror consistency % (share of test trials on which exchanging the hemispheres of")
        out("      the input exchanges the predicted class) and, in brackets, the lean to the right hand in points")
        out("      (recall of the right minus recall of the left). Below 100 even for the mirror models: a trial")
        out("      whose score is zero to within float32 rounding is counted as not exchanged. [S4] is the measure")
        out("      that does not depend on that.")
        out(f"      {'model':<16}" + "".join(f"{d:>18}" for d in datasets))
        any_cons = False
        for m in models:
            cells = ""
            for d in datasets:
                c, b = store.per_subject(d, protocol, m, 3), store.per_subject(d, protocol, m, 4)
                if c:
                    any_cons = True
                    cells += f"{100 * np.mean(list(c.values())):>11.1f} ({100 * np.mean(list(b.values())):>+5.1f})"
                else:
                    cells += f"{'-':>18}"
            out(f"      {m:<16}{cells}")
        if not any_cons:
            out("      (not measured in these runs; it is written only for fits made after this was added)")
        out()
        out(f"[S4] {protocol}: asymmetry of the decision function, mean |s + s'| / mean (|s| + |s'|), where s is")
        out("      the score of a test trial and s' the score of the same trial with its hemispheres exchanged.")
        out("      Zero for the mirror construction, down to the precision of float32 arithmetic (expect 1e-7 to")
        out("      1e-5); a model that learns the symmetry from the data, by augmentation or otherwise, cannot")
        out("      reach zero. Read this, not the share of exchanged predictions, as the test of the symmetry:")
        out("      a trial whose score sits at zero changes its predicted class on rounding alone.")
        out(f"      {'model':<16}" + "".join(f"{d:>18}" for d in datasets))
        any_asym = False
        for m in models:
            cells = ""
            for d in datasets:
                a = store.per_subject(d, protocol, m, 5)
                if a:
                    any_asym = True
                    cells += f"{np.mean(list(a.values())):>18.2e}"
                else:
                    cells += f"{'-':>18}"
            out(f"      {m:<16}{cells}")
        if not any_asym:
            out("      (not measured in these runs; it is written only for fits made after this was added)")
        out()
        out(f"[S2] {protocol}: differences in points [95% CI over subjects] dz p")
        for a, b, note in CONTRASTS:
            if a not in models or b not in models:
                continue
            for d in datasets:
                va, vb = store.per_subject(d, protocol, a), store.per_subject(d, protocol, b)
                keys = sorted(set(va) & set(vb))
                if len(keys) >= 3:
                    out(f"      {d:>12}  {a + ' - ' + b:<34}{S5.fmt(S5.paired([va[s] for s in keys], [vb[s] for s in keys]))}"
                        + f"  n={len(keys)}" + (f"   {note}" if note else ""))
        out()
    out("How to read this. The numbers of the 'session' protocol are the ones to put next to published")
    out("subject-specific results; the 'cv' numbers are higher because trials recorded next to each other are")
    out("split between training and test, which is what most papers reporting very high accuracies do.")
    out("Everything here is left hand against right hand, so chance is 50 %.")


# =========================== self-test ===========================
def selftest(device):
    print("self-test (synthetic data; nothing is written)")
    names = ["C3", "CZ", "C4", "CP3", "CP4"]
    assert mirror_index(names) == [2, 1, 0, 4, 3], "mirror of the montage"
    assert mirror_index(CH_2A)[CH_2A.index("C3")] == CH_2A.index("C4"), "mirror of dataset 2a"
    assert mirror_index(["C3", "CZ"]) is None, "a montage without a complete mirror image"
    assert clean_names(["EEG-Fz"] + [f"EEG-{i}" for i in range(21)], CH_2A) == CH_2A, "numbered channels of 2a"
    assert clean_names(["EEG:C3", "EEG:Cz", "EEG:C4"]) == ["C3", "CZ", "C4"], "channel names"
    y = np.array([0, 1] * 30)
    block = np.array([0] * 40 + [1] * 20)
    tr, te = split_session(y, block)
    assert sorted(te) == list(range(40, 60)) and len(tr) == 40, "the last block as test set"
    tr, te = split_session(y, np.zeros(60, int))
    assert sorted(te) == list(range(40, 60)), "the last third as test set"
    assert split_session(y[:10], np.zeros(10, int)) is None, "too few trials"
    tr0, te0 = split_session(y, np.zeros(60, int))
    s1, s2 = subsample(tr0, te0, y, 0.5), subsample(tr0, te0, y, 0.5)
    assert np.array_equal(s1[0], s2[0]), "the draw of the training trials is fixed"
    assert len(s1[0]) == len(tr0) // 2 and np.array_equal(s1[1], te0), "half the training trials, the same test set"
    assert sum(y[s1[0]] == 0) == sum(y[s1[0]] == 1), "the same fraction of each class"
    assert set(s1[0]) <= set(tr0) and not set(s1[0]) & set(te0), "no test trial enters the training set"
    assert subsample(tr0, te0, y, 0.05) is None, "too few training trials left"
    full = subsample(tr0, te0, y, 1.0)
    assert np.array_equal(full[0], tr0) and np.array_equal(full[1], te0), "the full training set is unchanged"
    parts = split_cv(y, 5)
    seen = np.concatenate([te for _, te in parts])
    assert len(parts) == 5 and sorted(seen) == list(range(60)), "every trial is tested once"
    for tr, te in parts:
        assert len(set(tr) & set(te)) == 0 and abs(y[te].mean() - 0.5) < 1e-9, "folds are disjoint and balanced"
    X = np.random.default_rng(0).standard_normal((60, 5, 751)).astype(np.float32) * 3.0 + 7.0
    Z = standardize(X, np.arange(40))
    assert abs(Z[:40].mean()) < 1e-4 and abs(Z[:40].std() - 1.0) < 0.05, "standardized on the training trials"
    w, ow = windows(Z, np.arange(3))
    assert w.shape == (18, 5, CROP) and list(ow[:7]) == [0, 0, 0, 0, 0, 0, 1], "windows of a trial"
    assert np.allclose(w[1], Z[0][:, STRIDE:STRIDE + CROP]), "position of the windows"
    print("  ok  mirror, channel names, splits, standardization inside the training set, windows")
    rng = np.random.default_rng(1)
    t = np.arange(751) / FS
    y = rng.permutation(np.arange(80) % 2)
    sig = np.sin(2 * np.pi * 11 * t) * rng.standard_normal((80, 1))
    X = rng.standard_normal((80, 5, 751)).astype(np.float32)
    X[:, 0] += (sig * np.where(y == 1, 0.4, 1.6)[:, None]).astype(np.float32)        # C3 loses power for the right hand
    X[:, 2] += (sig * np.where(y == 0, 0.4, 1.6)[:, None]).astype(np.float32)
    block = np.array([0] * 56 + [1] * 24)
    tr, te = split_session(y, block)
    Z = standardize(X, tr)
    libs = None
    perm5 = mirror_index(names)
    Zm = np.ascontiguousarray(Z[:, perm5])
    for m in MODELS:
        note = ""
        if m in DEEP:
            Xw, ow = windows(Z, tr)
            f = fit_deep(m, Xw, y[tr][ow], 5, CROP, perm5, 0, 8, device)
            Xq, oq = windows(Z, te)
            sc = f(Xq, oq, len(te))
            Xi, oi = windows(Zm, te)
            scm = f(Xi, oi, len(te))
            pred, a = (sc > 0).astype(int), asymmetry(sc, scm)
            note = f", asymmetry {a:.1e}"
            if m.endswith("_mirror"):                   # s' = -s must hold for every trial, not on average
                assert a < 1e-4, f"{m}: the construction is not equivariant (asymmetry {a:.2e})"
                assert np.allclose(sc, -scm, atol=1e-3 * max(1e-9, float(np.max(np.abs(sc))))), \
                    f"{m}: the mirrored scores are not the negated scores"
        else:
            if libs is None:
                libs = S5._libs()
            pred, _pm, scv = run_classic(m, Z, y, tr, te, libs)
            assert scv.shape == (len(te),), f"{m}: decision function"
        assert pred.shape == (len(te),) and set(np.unique(pred)) <= {0, 1}, f"{m}: predictions"
        print(f"  ok  {m:<16} ran on synthetic data (accuracy {100 * np.mean(pred == y[te]):.0f} %, meaningless{note})")
    s = np.array([1.0, -2.0, 0.5])
    assert asymmetry(s, -s) == 0.0 and abs(asymmetry(s, s) - 1.0) < 1e-12, "the asymmetry measure"
    assert np.isnan(asymmetry(s, None)), "no mirrored scores, no asymmetry"
    assert abs(consistency(np.array([0, 1, 1]), np.array([1, 0, 1])) - 2 / 3) < 1e-12, "the consistency measure"
    print("  ok  asymmetry and consistency measures")
    print("self-test passed")
    return 0


# =========================== main ===========================
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=S15.ROOT)
    ap.add_argument("--datasets", default="2a,2b", help="comma-separated, or all")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--protocol", default="both", choices=["session", "cv", "both"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--band", default="4,38")
    ap.add_argument("--window", default="0,3")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N subjects of each dataset (for spot checks)")
    ap.add_argument("--train-fraction", default="",
                    help="e.g. 0.25,0.5: run the protocol again with that share of the training trials, the test"
                         " set unchanged. Those fits are stored under 'session@0.25' and so on, so the rows"
                         " already in the file are not touched; the 100 %% point of the curve is those rows.")
    ap.add_argument("--scores", action="store_true",
                    help="write one line per test trial to corrected_results/step24_scores.csv"
                         " (what an ROC curve is computed from)")
    ap.add_argument("--curves", action="store_true",
                    help="write the training loss of every epoch to corrected_results/step24_curves.csv"
                         " (slower to write, so use it on a few subjects)")
    ap.add_argument("--cpu-ok", action="store_true",
                    help="run even though no GPU is visible. The rows it writes are trained for fewer epochs"
                         " than the rows already in the file, so they do not belong in the same table.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.selftest:
        return selftest(device)
    if device == "cpu" and not args.cpu_ok and not args.epochs:
        print(f"No GPU is visible to PyTorch ({torch.__version__}), so this run would train every network for 60"
              f" epochs instead of the 200 used for every row already in the file. Those rows would not be"
              f" comparable with the rest of the table and the difference would not show in the table itself.\n"
              f"Nothing was run. Most likely the environment with the CUDA build of PyTorch is not active: in that"
              f" environment the first line of a run reads 'device: cuda'.\n"
              f"To run on the processor anyway, pass --cpu-ok (or --epochs 200 to keep the budget).")
        return 1
    band = tuple(float(v) for v in args.band.split(","))
    win = tuple(float(v) for v in args.window.split(","))
    datasets = DATASETS if args.datasets == "all" else [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    protocols = ["session", "cv"] if args.protocol == "both" else [args.protocol]
    fracs = [f.strip() for f in args.train_fraction.split(",") if f.strip()]
    if fracs:
        protocols = [p if float(f) >= 1.0 else f"{p}@{f}" for p in protocols for f in fracs]
    bad = [d for d in datasets if d not in DATASETS] + [m for m in models if m not in MODELS]
    if bad:
        print(f"unknown dataset or model: {', '.join(bad)}. Known: {', '.join(DATASETS)} | {', '.join(MODELS)}")
        return 1
    epochs = args.epochs or (200 if device == "cuda" else 60)
    t_start = time.time()
    S5._LINES.clear()
    out("Step 24 subject-specific  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    out(f"PyTorch {torch.__version__}, device: {device}" + ("" if device == "cuda" else "  (no GPU visible to PyTorch: this will be slow)"))
    out(f"{band[0]:g}-{band[1]:g} Hz, {win[0]:g} to {win[1]:g} s after the cue, {FS:g} Hz, windows of {CROP / FS:g} s"
        f" every {STRIDE / FS:g} s; {epochs} epochs, seeds {seeds}, {args.folds}-fold")
    out("protocols: " + ", ".join(protocols) + " | models: " + ", ".join(models))
    mods = S9.project_modules()
    libs = S5._libs()
    root = {}
    if EXT in datasets:
        for num, sid, folder in S15.list_subjects(args.root):
            if folder is not None:
                root[sid] = folder
    store = Store(os.path.join(OUTDIR, "step24_subject_specific.csv"),
                  (f"{band[0]:g}-{band[1]:g}", f"{win[0]:g}-{win[1]:g}"), args.redo)
    scores_path = curves_path = None
    if args.scores:
        scores_path = os.path.join(OUTDIR, "step24_scores.csv")
        if not os.path.exists(scores_path):
            with open(scores_path, "w", encoding="utf-8") as f:
                f.write("dataset,subject,protocol,fold,model,seed,trial,y_true,score,score_mirror\n")
    if args.curves:
        curves_path = os.path.join(OUTDIR, "step24_curves.csv")
        if not os.path.exists(curves_path):
            with open(curves_path, "w", encoding="utf-8") as f:
                f.write("dataset,subject,protocol,fold,model,seed,epoch,loss\n")
    for ds in datasets:
        keys = subject_keys(ds, mods, root if ds == EXT else None)
        if args.limit:
            keys = keys[:args.limit]
        out(f"{ds}: {len(keys)} subjects" + (f"  (first {args.limit} only)" if args.limit else ""))
        if not keys:
            continue
        t0, n_done, n_bad, n_nosplit = time.time(), 0, 0, 0
        for i, key in enumerate(keys):
            try:
                done, dt, n_parts = do_subject(ds, key, mods, band, win, root, models, protocols,
                                               args.folds, seeds, epochs, device, libs, store,
                                               scores_path, curves_path)
            except Exception as e:
                n_bad += 1
                out(f"    {ds} {key}: could not be read ({e!r})")
                continue
            if n_parts == 0:
                n_nosplit += 1
            n_done += done
            print(f"    {ds} {key}: {done} fits in {dt:.0f} s | {i + 1}/{len(keys)} subjects,"
                  f" {(time.time() - t0) / 60:.0f} min so far", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
        out(f"    {n_done} fits, {(time.time() - t0) / 60:.1f} min"
            + (f"  |  {n_bad} of {len(keys)} subjects could not be read" if n_bad else "")
            + (f"  |  {n_nosplit} of {len(keys)} subjects gave no usable split" if n_nosplit else ""))
        if n_bad + n_nosplit == len(keys):
            out(f"    WARNING: {ds} contributed nothing at all; the lines above say why")
    out()
    report(store, datasets, models, protocols)
    out()
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step24_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    print(f"written: {os.path.abspath(OUTDIR)} (step24_summary.txt, step24_subject_specific.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
