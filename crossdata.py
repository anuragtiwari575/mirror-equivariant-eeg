"""
Cross-dataset LH/RH loaders and channel harmonization for the R3 study
(train on one dataset, test on another, zero calibration).

Datasets and their left/right-hand sources:
  PhysioNet EEGMMIDB : EDF, 64 ch @160 Hz; imagined-fist runs R04/R08/R12,
                       events T1=left, T2=right (T0=rest).
  High Gamma (BIDS)  : BDF, 64 ch @512 Hz; task-imagery runs, events
                       'left_hand' / 'right_hand'.
  BCI IV-2a / 2b     : handled by mi_data (classes 769=left, 770=right).

Everything is harmonized to a common montage (C3, Cz, C4 by default, the only
channels shared with 2b), a common rate, band, and window, so trials from any
dataset are directly comparable.

Channel-name normalization and montage selection are unit-tested on synthetic
Raw objects in test_crossdata.py. Real EDF/BDF reads run on the target machine.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import mne

mne.set_log_level("ERROR")

COMMON3 = ["C3", "Cz", "C4"]                     # shared across all four datasets
PN_RUNS_LR = [4, 8, 12]                          # PhysioNet imagined left/right fist
# PhysioNet subjects with known annotation/rate issues (commonly excluded).
PN_BAD = {88, 89, 92, 100}


def _norm(name):
    """Canonical uppercase electrode label: strip dots, spaces, EEG prefixes."""
    n = name.strip().upper().replace(".", "").replace(" ", "")
    for p in ("EEG-", "EEG:", "EEG"):
        if n.startswith(p):
            n = n[len(p):]
    return n


def pick_common(raw, common=COMMON3):
    """Keep only the common channels, renamed and ordered to `common`."""
    target = [c.upper() for c in common]
    keep = [ch for ch in raw.ch_names if _norm(ch) in target]
    missing = [c for c in target if c not in {_norm(ch) for ch in keep}]
    if missing:
        raise ValueError(f"missing common channels {missing} in {raw.ch_names[:6]}...")
    raw.pick(keep)
    ordered = sorted(keep, key=lambda ch: target.index(_norm(ch)))
    raw.reorder_channels(ordered)
    raw.rename_channels({ch: common[target.index(_norm(ch))] for ch in ordered})
    return raw


def _epoch(raw, event_map, tmin, tmax):
    """event_map: {annotation_description: class_int}. Returns (X, y)."""
    events, eid = mne.events_from_annotations(raw)
    sel = {k: eid[k] for k in event_map if k in eid}
    if not sel:
        return None, None
    ep = mne.Epochs(raw, events, event_id=sel, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True)
    id2cls = {eid[k]: event_map[k] for k in sel}
    y = np.array([id2cls[e] for e in ep.events[:, 2]], dtype=np.int64)
    return ep.get_data(copy=True).astype(np.float32), y


def _prep(raw, common, l_freq, h_freq, sfreq):
    raw = pick_common(raw, common)
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, verbose="ERROR")
    if sfreq is not None and float(sfreq) != float(raw.info["sfreq"]):
        raw.resample(float(sfreq), verbose="ERROR")
    return raw


def load_physionet_subject(root, subj, *, common=COMMON3, sfreq=250, tmin=0.5,
                           tmax=2.5, l_freq=8.0, h_freq=30.0):
    Xs, ys = [], []
    for r in PN_RUNS_LR:
        f = os.path.join(root, "files", f"S{subj:03d}", f"S{subj:03d}R{r:02d}.edf")
        if not os.path.exists(f):
            continue
        raw = _prep(mne.io.read_raw_edf(f, preload=True), common, l_freq, h_freq, sfreq)
        X, y = _epoch(raw, {"T1": 0, "T2": 1}, tmin, tmax)     # T1=left, T2=right
        if X is not None:
            Xs.append(X); ys.append(y)
    if not Xs:
        raise RuntimeError(f"PhysioNet S{subj:03d}: no LH/RH trials found")
    return np.concatenate(Xs), np.concatenate(ys)


def load_hgd_subject(root, subj, *, common=COMMON3, sfreq=250, tmin=0.5,
                     tmax=2.5, l_freq=8.0, h_freq=30.0):
    pat = os.path.join(root, f"sub-{subj}", "**",
                       f"sub-{subj}_*task-imagery*_eeg.bdf")
    files = sorted(glob.glob(pat, recursive=True))
    Xs, ys = [], []
    for f in files:
        raw = _prep(mne.io.read_raw_bdf(f, preload=True), common, l_freq, h_freq, sfreq)
        X, y = _epoch(raw, {"left_hand": 0, "right_hand": 1}, tmin, tmax)
        if X is not None:
            Xs.append(X); ys.append(y)
    if not Xs:
        raise RuntimeError(f"HGD sub-{subj:02d}: no LH/RH trials found")
    return np.concatenate(Xs), np.concatenate(ys)


def load_2a_subject(root, subj, *, common=COMMON3, sfreq=250, tmin=0.5,
                    tmax=2.5, l_freq=8.0, h_freq=30.0):
    """BCI IV-2a training session, reduced to LH/RH on the common montage."""
    f = os.path.join(root, f"A{subj:02d}T.gdf")
    raw = _prep(mne.io.read_raw_gdf(f, preload=True), common, l_freq, h_freq, sfreq)
    X, y = _epoch(raw, {"769": 0, "770": 1}, tmin, tmax)   # 769=LH, 770=RH
    if X is None:
        raise RuntimeError(f"2a S{subj:02d}: no LH/RH trials")
    return X, y


def load_2b_subject(root, subj, *, common=COMMON3, sfreq=250, tmin=0.5,
                    tmax=2.5, l_freq=8.0, h_freq=30.0):
    """BCI IV-2b training sessions (1-3), LH/RH on the common montage."""
    Xs, ys = [], []
    for s in (1, 2, 3):
        f = os.path.join(root, f"B{subj:02d}0{s}T.gdf")
        if not os.path.exists(f):
            continue
        raw = _prep(mne.io.read_raw_gdf(f, preload=True), common, l_freq, h_freq, sfreq)
        X, y = _epoch(raw, {"769": 0, "770": 1}, tmin, tmax)
        if X is not None:
            Xs.append(X); ys.append(y)
    if not Xs:
        raise RuntimeError(f"2b S{subj:02d}: no LH/RH trials")
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    try:
        import config
        pn, hg = config.DATA_PHYSIONET, config.DATA_HGD
        d2a, d2b = config.DATA_2A, config.DATA_2B
    except Exception:
        base = r"C:\Users\Anurag Tiwari\OneDrive\Desktop\Motor Imagery Classification-Work"
        pn = os.path.join(base, "eeg-motor-movementimagery-dataset-1.0.0")
        hg = os.path.join(base, "v1.0.2")
        d2a = r"E:\Motor Imagery Classification\BCICIV_2a_gdf"
        d2b = r"E:\Motor Imagery Classification\BCICIV_2b_gdf"

    print("Cross-dataset LH/RH smoke (subject 1 each), C3/Cz/C4 @250 Hz, [0.5,2.5]s\n")
    jobs = [("2a", load_2a_subject, d2a, 1), ("2b", load_2b_subject, d2b, 1),
            ("PhysioNet", load_physionet_subject, pn, 1), ("HGD", load_hgd_subject, hg, 1)]
    for name, fn, root, sid in jobs:
        try:
            X, y = fn(root, sid)
            vals, cnts = np.unique(y, return_counts=True)
            bal = " ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts))
            print(f"  {name} S{sid}: X={X.shape}  balance {bal}")
        except Exception as e:
            print(f"  {name} S{sid}: ERROR {e}")


if __name__ == "__main__":
    main()
