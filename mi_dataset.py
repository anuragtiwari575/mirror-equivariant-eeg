"""
Dataset assembly, preprocessing, and evaluation splits for BCI IV-2a / IV-2b.

Sits on top of mi_data.load_gdf. Produces per-subject epoched tensors with
band-pass, optional resample, and trial/channel normalization, then builds the
two protocols this project needs:

  * within_subject : the standard competition split used by the SOTA baselines
                     (2a: T-train / E-test; 2b: sessions 1-3 train / 4-5 test)
  * loso_splits    : leave-one-subject-out, for the cross-subject and
                     calibration-free study

Leakage rule: any normalization statistic that is fit on data is fit on the
TRAIN partition of that split only and reused on TEST. The 'trial' mode fits
nothing (each trial is standardized against itself) and is leakage-free by
construction. Per-subject epoched arrays are cached UNNORMALIZED under
config.CACHE; normalization is re-applied per split (cheap) so the expensive
GDF parse/filter/epoch runs once.

Logic is unit tested in test_mi_dataset.py against a stubbed loader (no real
GDF required).
"""
from __future__ import annotations
import os
import json
import hashlib
import numpy as np
import mi_data

try:
    import config
    _CACHE = config.CACHE
    _PREPROC = dict(config.PREPROC)
    _ROOT = {"2a": config.DATA_2A, "2b": config.DATA_2B}
except Exception:  # config absent (e.g. isolated test run)
    config = None
    _CACHE = "."
    _PREPROC = {"tmin": 0.0, "tmax": 4.0, "l_freq": 4.0, "h_freq": 40.0,
                "resample": None, "normalize": "trial"}
    _ROOT = {"2a": "", "2b": ""}

DATASETS = {
    "2a": {"root": _ROOT["2a"], "code_map": mi_data.CODE_MAP_2A,
           "subjects": tuple(range(1, 10)), "n_classes": 4},
    "2b": {"root": _ROOT["2b"], "code_map": mi_data.CODE_MAP_2B,
           "subjects": tuple(range(1, 10)), "n_classes": 2},
}


class MissingLabels(Exception):
    """Raised when a required session has no usable labels (E-session .mat absent)."""


def subject_files(dataset, subj):
    """Return {'train': [...], 'test': [...]} GDF basenames for one subject."""
    if dataset == "2a":
        return {"train": [f"A{subj:02d}T.gdf"], "test": [f"A{subj:02d}E.gdf"]}
    if dataset == "2b":
        return {"train": [f"B{subj:02d}0{s}T.gdf" for s in (1, 2, 3)],
                "test":  [f"B{subj:02d}0{s}E.gdf" for s in (4, 5)]}
    raise ValueError(f"unknown dataset {dataset!r}")


# ---------- loading ----------

def _load_files(dataset, files, cfg):
    root = DATASETS[dataset]["root"]
    code_map = DATASETS[dataset]["code_map"]
    Xs, ys = [], []
    for fn in files:
        X, y, _ = mi_data.load_gdf(
            os.path.join(root, fn), code_map,
            tmin=cfg["tmin"], tmax=cfg["tmax"],
            l_freq=cfg["l_freq"], h_freq=cfg["h_freq"], resample=cfg["resample"])
        if y is None:
            raise MissingLabels(f"{fn}: evaluation labels missing (.mat classlabel not found)")
        Xs.append(np.asarray(X, dtype=np.float32))
        ys.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def load_partition(dataset, subj, which, cfg=None):
    """Load one subject's 'train' or 'test' partition as (X, y)."""
    cfg = cfg or _PREPROC
    return _load_files(dataset, subject_files(dataset, subj)[which], cfg)


# ---------- normalization (fit on train only) ----------

def fit_normalizer(X, mode):
    if mode == "channel":
        mu = X.mean(axis=(0, 2), keepdims=True)
        sd = X.std(axis=(0, 2), keepdims=True) + 1e-7
        return (mu, sd)
    return None  # 'trial' and None fit nothing


def apply_normalizer(X, mode, stats=None):
    if mode == "trial":
        mu = X.mean(axis=2, keepdims=True)
        sd = X.std(axis=2, keepdims=True) + 1e-7
        return (X - mu) / sd
    if mode == "channel":
        mu, sd = stats
        return (X - mu) / sd
    return X


# ---------- per-subject cache (unnormalized) ----------

def _cfg_key(cfg):
    payload = {k: cfg[k] for k in ("tmin", "tmax", "l_freq", "h_freq", "resample")}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]


def build_subject_cached(dataset, subj, cfg=None, use_cache=True):
    """Return {'X_train','y_train'[, 'X_test','y_test']} epoched but UNNORMALIZED.

    X_test/y_test are omitted when the evaluation session has no labels yet.
    """
    cfg = cfg or _PREPROC
    key = _cfg_key(cfg)
    path = os.path.join(_CACHE, f"{dataset}_S{subj:02d}_{key}.npz")
    if use_cache and os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}

    out = {}
    Xtr, ytr = load_partition(dataset, subj, "train", cfg)
    out["X_train"], out["y_train"] = Xtr, ytr
    try:
        Xte, yte = load_partition(dataset, subj, "test", cfg)
        out["X_test"], out["y_test"] = Xte, yte
    except MissingLabels:
        pass  # E-session labels not present yet; train-only for now

    if use_cache:
        os.makedirs(_CACHE, exist_ok=True)
        np.savez_compressed(path, **out)
    return out


# ---------- protocols ----------

def within_subject(dataset, subj, cfg=None, use_cache=True):
    """Standard competition split for one subject. Raises MissingLabels if the
    evaluation session is unlabeled."""
    cfg = cfg or _PREPROC
    d = build_subject_cached(dataset, subj, cfg, use_cache)
    if "X_test" not in d:
        raise MissingLabels(f"{dataset} S{subj:02d}: evaluation session labels missing")
    stats = fit_normalizer(d["X_train"], cfg["normalize"])
    return dict(
        X_train=apply_normalizer(d["X_train"], cfg["normalize"], stats), y_train=d["y_train"],
        X_test=apply_normalizer(d["X_test"], cfg["normalize"], stats), y_test=d["y_test"],
    )


def loso_splits(dataset, cfg=None, include_test_sessions=False, use_cache=True):
    """Yield leave-one-subject-out folds.

    Each fold: dict(test_subject, X_train, y_train, X_test, y_test), normalized
    with statistics fit on the pooled source subjects only. When
    include_test_sessions=True, a subject's labeled E-session (if present) is
    folded into its pool; unlabeled E-sessions are silently skipped.
    """
    cfg = cfg or _PREPROC
    subjects = list(DATASETS[dataset]["subjects"])
    pool = {}
    for s in subjects:
        d = build_subject_cached(dataset, s, cfg, use_cache)
        X, y = d["X_train"], d["y_train"]
        if include_test_sessions and "X_test" in d:
            X = np.concatenate([X, d["X_test"]], axis=0)
            y = np.concatenate([y, d["y_test"]], axis=0)
        pool[s] = (X, y)

    for test_s in subjects:
        src = [s for s in subjects if s != test_s]
        Xtr = np.concatenate([pool[s][0] for s in src], axis=0)
        ytr = np.concatenate([pool[s][1] for s in src], axis=0)
        Xte, yte = pool[test_s]
        stats = fit_normalizer(Xtr, cfg["normalize"])
        yield dict(
            test_subject=test_s,
            X_train=apply_normalizer(Xtr, cfg["normalize"], stats), y_train=ytr,
            X_test=apply_normalizer(Xte, cfg["normalize"], stats), y_test=yte,
        )
