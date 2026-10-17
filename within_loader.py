"""
Self-contained within-subject loader for BCI IV-2a / IV-2b, using the exact
epoching path that gave a correct CSP+LDA reference (~77% on 2a S01). It reads
GDF directly, band-passes, epochs on the cue codes, and z-scores per trial. No
project cache and no Riemannian estimator are involved (those were the source of
earlier confusion).

load_subject(dataset, subj) -> (X, y):
  2a: 4 classes (769-772), 22 EEG channels
  2b: 2 classes (769-770), 3 EEG channels
"""
from __future__ import annotations
import os
import glob
import numpy as np
import mne

mne.set_log_level("ERROR")

try:
    import config
    ROOT = {"2a": config.DATA_2A, "2b": config.DATA_2B}
except Exception:
    config = None
    ROOT = {"2a": "", "2b": ""}

CODES = {"2a": ["769", "770", "771", "772"], "2b": ["769", "770"]}
N_CLASSES = {"2a": 4, "2b": 2}


def _epoch_file(path, codes, tmin, tmax, l_freq, h_freq, sfreq):
    raw = mne.io.read_raw_gdf(path, preload=True)
    raw.drop_channels([c for c in raw.ch_names if "EOG" in c.upper()])
    raw.filter(l_freq, h_freq, verbose="ERROR")
    if sfreq and float(sfreq) != float(raw.info["sfreq"]):
        raw.resample(float(sfreq), verbose="ERROR")
    events, eid = mne.events_from_annotations(raw)
    want = {k: eid[k] for k in codes if k in eid}
    if not want:
        return None, None
    ep = mne.Epochs(raw, events, event_id=want, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True)
    X = ep.get_data(copy=True).astype(np.float32)
    id2c = {eid[k]: i for i, k in enumerate(codes) if k in eid}
    y = np.array([id2c[e] for e in ep.events[:, 2]], dtype=np.int64)
    return X, y


def _zscore(X):
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-7
    return (X - mu) / sd


def load_subject(dataset, subj, tmin=0.5, tmax=2.5, l_freq=8.0, h_freq=30.0,
                 sfreq=None, zscore=True):
    codes = CODES[dataset]
    if dataset == "2a":
        files = [os.path.join(ROOT["2a"], f"A{subj:02d}T.gdf")]
    else:
        files = [os.path.join(ROOT["2b"], f"B{subj:02d}0{s}T.gdf") for s in (1, 2, 3)]
    Xs, ys = [], []
    for f in files:
        if not os.path.exists(f):
            continue
        X, y = _epoch_file(f, codes, tmin, tmax, l_freq, h_freq, sfreq)
        if X is not None:
            Xs.append(X); ys.append(y)
    if not Xs:
        raise RuntimeError(f"{dataset} S{subj:02d}: no trials (check config paths)")
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    if zscore:
        X = _zscore(X)
    return X, y


def quick_check():
    """Sanity: CSP+LDA on 2a S01 through THIS loader (expect ~75%+)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score
    from mne.decoding import CSP
    X, y = load_subject("2a", 1, zscore=False)
    print(f"loaded 2a S01: X={X.shape} balance={np.bincount(y).tolist()}")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        csp = CSP(n_components=8, reg="ledoit_wolf", log=True)
        lda = LinearDiscriminantAnalysis().fit(csp.fit_transform(X[tr], y[tr]), y[tr])
        accs.append(accuracy_score(y[te], lda.predict(csp.transform(X[te]))))
    print(f"CSP+LDA 5-fold via this loader: {np.mean(accs)*100:.2f}%  (expect ~75%+)")


if __name__ == "__main__":
    quick_check()
