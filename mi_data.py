"""
BCI Competition IV-2a / IV-2b loader and dataset diagnostics.

Reads raw GDF files via MNE, extracts motor-imagery trials, and resolves the
evaluation-session label problem: E-session cues are marked 783 ("unknown") in
the GDF, and the true labels live in the companion .mat file's 'classlabel'
field. Training (T) sessions carry class cues (769-772) directly.

Verified against mne 1.13.2. The core epoching logic (epoch_from_raw) is unit
tested on synthetic annotated data in test_mi_data.py.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import mne
from scipy.io import loadmat

mne.set_log_level("ERROR")

# GDF event codes -> zero-indexed MI class
CODE_MAP_2A = {"769": 0, "770": 1, "771": 2, "772": 3}   # LH, RH, feet, tongue
CODE_MAP_2B = {"769": 0, "770": 1}                        # LH, RH
UNKNOWN_CODE = "783"
CLASS_NAMES = {
    "2a": ["left_hand", "right_hand", "feet", "tongue"],
    "2b": ["left_hand", "right_hand"],
}


def _prep_eeg(raw):
    """Mark EOG channels by name and keep EEG only. Returns (raw, eog_names)."""
    eog = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    if eog:
        raw.set_channel_types({ch: "eog" for ch in eog})
    raw.pick("eeg")
    return raw, eog


def epoch_from_raw(raw, code_map, tmin, tmax, classlabel=None):
    """Extract MI epochs from a prepared Raw.

    Returns (X, y, meta):
      X : (n_trials, n_ch, n_times) float64
      y : (n_trials,) int, or None for an unknown session with no classlabel
      meta : dict with session type and shapes
    """
    events, event_id = mne.events_from_annotations(raw)
    present = set(event_id)
    labeled = [c for c in code_map if c in present]

    if labeled:
        sel = {c: event_id[c] for c in labeled}
        ep = mne.Epochs(raw, events, event_id=sel, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
        id2code = {event_id[c]: c for c in labeled}
        y = np.array([code_map[id2code[e]] for e in ep.events[:, 2]], dtype=int)
        session = "labeled"
    elif UNKNOWN_CODE in present:
        sel = {UNKNOWN_CODE: event_id[UNKNOWN_CODE]}
        matched = np.where(events[:, 2] == event_id[UNKNOWN_CODE])[0]
        ep = mne.Epochs(raw, events, event_id=sel, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
        session = "unknown"
        if classlabel is not None:
            classlabel = np.asarray(classlabel).ravel()
            if len(classlabel) != len(matched):
                raise ValueError(
                    f"classlabel length {len(classlabel)} != number of unknown "
                    f"cues {len(matched)} (session/label file mismatch)")
            row2lab = {row: int(classlabel[i]) - 1 for i, row in enumerate(matched)}
            y = np.array([row2lab[s] for s in ep.selection], dtype=int)
        else:
            y = None
    else:
        return (np.empty((0,)), None,
                {"session": "no_mi_events", "n_trials": 0,
                 "present_codes": sorted(present)})

    X = ep.get_data(copy=True)
    meta = {"session": session, "n_trials": len(ep),
            "sfreq": float(raw.info["sfreq"]), "n_ch": X.shape[1],
            "n_times": X.shape[2], "present_codes": sorted(present)}
    return X, y, meta


def _find_classlabel(gdf_path):
    """Look for a companion .mat holding the true labels (E-session case)."""
    mat_path = os.path.splitext(gdf_path)[0] + ".mat"
    if os.path.exists(mat_path):
        m = loadmat(mat_path)
        if "classlabel" in m:
            return np.asarray(m["classlabel"]).ravel(), mat_path
    return None, mat_path


def load_gdf(gdf_path, code_map, tmin=0.0, tmax=4.0,
             l_freq=None, h_freq=None, resample=None):
    """Load one GDF session into (X, y, meta). y is None if labels are missing.

    Optional preprocessing is applied to the continuous EEG BEFORE epoching:
      l_freq / h_freq : band-pass edges in Hz (either may be None)
      resample        : target sampling rate in Hz (None keeps native rate)
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    raw, eog = _prep_eeg(raw)
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, verbose="ERROR")
    if resample is not None and float(resample) != float(raw.info["sfreq"]):
        raw.resample(float(resample), verbose="ERROR")
    classlabel, mat_path = _find_classlabel(gdf_path)
    X, y, meta = epoch_from_raw(raw, code_map, tmin, tmax, classlabel=classlabel)
    meta.update({"file": os.path.basename(gdf_path), "n_eog": len(eog),
                 "mat_path": mat_path if os.path.exists(mat_path) else None})
    return X, y, meta


def diagnose_folder(root, code_map, tmin=0.0, tmax=4.0):
    """Scan a folder of GDFs and report what each contains."""
    # Dedup case-insensitively: on Windows *.gdf and *.GDF match the same file.
    seen = {}
    for pat in ("*.gdf", "*.GDF"):
        for f in glob.glob(os.path.join(root, pat)):
            seen[os.path.normcase(os.path.abspath(f))] = f
    gdfs = sorted(seen.values(), key=lambda p: os.path.basename(p).lower())
    rows = []
    for f in gdfs:
        try:
            X, y, meta = load_gdf(f, code_map, tmin, tmax)
            balance = ""
            if y is not None and len(y):
                vals, cnts = np.unique(y, return_counts=True)
                balance = " ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts))
            rows.append({
                "file": meta["file"], "sess": meta["session"],
                "eeg": meta.get("n_ch", 0), "eog": meta.get("n_eog", 0),
                "fs": meta.get("sfreq", 0), "trials": meta.get("n_trials", 0),
                "labels": "yes" if y is not None else "MISSING",
                "mat": "found" if meta.get("mat_path") else "-",
                "balance": balance,
            })
        except Exception as e:  # report, do not crash the whole scan
            rows.append({"file": os.path.basename(f), "sess": "ERROR",
                         "error": str(e)})
    return rows


def _format_table(rows, title):
    out = [f"\n=== {title} ==="]
    if not rows:
        out.append("  (no .gdf files found at this path)")
        return "\n".join(out)
    hdr = ["file", "sess", "eeg", "eog", "fs", "trials", "labels", "mat", "balance"]
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in hdr}
    out.append("  " + "  ".join(h.ljust(widths[h]) for h in hdr))
    for r in rows:
        if r.get("sess") == "ERROR":
            out.append("  " + r["file"].ljust(widths["file"]) + "  ERROR: " + r["error"])
        else:
            out.append("  " + "  ".join(str(r.get(h, "")).ljust(widths[h]) for h in hdr))
    return "\n".join(out)


def main():
    import argparse
    try:
        import config
        config.ensure_dirs()
        default_2a, default_2b, out_dir = config.DATA_2A, config.DATA_2B, config.OUTPUTS
    except Exception:
        default_2a = r"E:\Motor Imagery Classification\BCICIV_2a_gdf"
        default_2b = r"E:\Motor Imagery Classification\BCICIV_2b_gdf"
        out_dir = "."

    ap = argparse.ArgumentParser(description="Diagnose BCI IV-2a/2b GDF folders")
    ap.add_argument("--root2a", default=default_2a)
    ap.add_argument("--root2b", default=default_2b)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=4.0)
    ap.add_argument("--out", default=out_dir)
    args = ap.parse_args()

    lines = [f"mne {mne.__version__} | window [{args.tmin}, {args.tmax}] s relative to cue"]
    r2a = diagnose_folder(args.root2a, CODE_MAP_2A, args.tmin, args.tmax)
    lines.append(_format_table(r2a, f"BCI IV-2a  ({args.root2a})"))
    r2b = diagnose_folder(args.root2b, CODE_MAP_2B, args.tmin, args.tmax)
    lines.append(_format_table(r2b, f"BCI IV-2b  ({args.root2b})"))

    n_missing = sum(1 for r in (r2a + r2b) if r.get("labels") == "MISSING")
    if n_missing:
        lines.append(f"\n[!] {n_missing} session(s) have NO usable labels. For "
                     f"E-sessions you need the companion .mat ('classlabel') next "
                     f"to each GDF.")
    else:
        lines.append("\n[ok] Every scanned session resolved to real labels.")

    report = "\n".join(lines)
    print(report)
    try:
        os.makedirs(args.out, exist_ok=True)
        report_path = os.path.join(args.out, "dataset_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nReport written to {report_path}")
    except Exception as e:
        print(f"\n[warn] could not write report: {e}")


if __name__ == "__main__":
    main()
