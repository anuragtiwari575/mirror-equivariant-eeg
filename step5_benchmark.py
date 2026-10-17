"""
step5_benchmark.py

The expanded, corrected benchmark. Run from the Codes folder:

    python step5_benchmark.py

The first run reads every recording (about 20 to 40 minutes) and stores trial
covariances per subject in corrected_results/cache_<recipe>/. Later runs reuse
that cache and take a few minutes.

Data (all motor imagery, left hand versus right hand)
  2a         BCI-IV-2a, 9 subjects, training session, reference as distributed
  2b         BCI-IV-2b, 9 subjects, training sessions 1-3, three bipolar channels
  PhysioNet  every subject on disk, imagined left/right fist, runs 4, 8 and 12
  Cho2017    every subject on disk (the folder config.DATA_HGD; this is the
             dataset the old code called "HGD"), re-referenced to the common
             average of its 64 scalp channels before channel selection, because
             the BioSemi recording has no reference (CMS/DRL)

  Every recording: band-pass 8-30 Hz (MNE default zero-phase FIR), resampled
  to 250 Hz where the native rate differs, epochs 0.5 to 2.5 s after cue onset.
  No trial or subject is removed on the basis of decoding performance.

Montages
  3ch    C3, Cz, C4                                  four datasets, 12 directions
  17ch   the 17 sensorimotor channels of variants.SM  2a, PhysioNet, Cho2017, 6 directions

Pipeline ladder, per montage
  covariance   raw = OAS covariance | tn = OAS covariance divided by its trace
  alignment    none
               ea, ra        source pooled under one reference, each target subject on its own
               ea_ps, ra_ps  every source subject on its own reference as well
  classifier   lr = tangent-space logistic regression | mdm = minimum distance to mean

Primary configuration, fixed before this run: tn + ra_ps + lr.
Target labels are used only to score predictions.

Outputs (new files only)
  corrected_results/benchmark_subjects.csv    one row per source, target subject and condition
  corrected_results/benchmark_inclusion.csv   every subject on disk, included or not, and why
  corrected_results/benchmark_summary.txt     the tables printed on screen
"""
from __future__ import annotations

import os
import sys
import glob
import json
import time
import datetime
import itertools

import numpy as np

OUTDIR = "corrected_results"
RECIPE = "bp8-30fir_fs250_win0.5-2.5_oas_choCAR64_v1"
CACHE = os.path.join(OUTDIR, "cache_" + RECIPE)
DATASETS = ["2a", "2b", "PhysioNet", "Cho2017"]
MONTAGES = {"3ch": ["2a", "2b", "PhysioNet", "Cho2017"], "17ch": ["2a", "PhysioNet", "Cho2017"]}
TRIPLET_IN_SM = [6, 8, 10]                 # C3, Cz, C4 inside the 17-channel list
IDX_2A_SM = list(range(1, 18))             # the 17 channels inside the 22 of BCI-IV-2a
IDX_2A_TRIPLET = [7, 9, 11]
PN_RUNS = [4, 8, 12]
# PhysioNet subjects excluded in earlier studies for recording or annotation irregularities.
PN_EXCLUDE = {38, 88, 89, 92, 100, 104}
PN_FS = 160.0
CHO_EVENTS = {"left_hand": 0, "right_hand": 1}

COVS = ["raw", "tn"]
ALIGNS = ["none", "ea", "ra", "ea_ps", "ra_ps"]
CLFS = ["lr", "mdm"]
PRIMARY = ("tn", "ra_ps")

# Triplet numbers from the earlier runs (steps 2 and 3), TS-LR, for the two directions whose
# subjects are unchanged: (raw none, tn none, tn ra).
REGRESSION = {("2a", "2b"): (62.54, 66.50, 67.77), ("2b", "2a"): (59.49, 62.27, 65.66)}

_LINES = []


def out(msg=""):
    msg = str(msg)
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
    _LINES.append(msg)


def _libs():
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    try:                                     # pyriemann >= 0.12
        from pyriemann.geometry.mean import mean_riemann
        from pyriemann.geometry.tangentspace import tangent_space
    except Exception:                        # older releases
        from pyriemann.utils.mean import mean_riemann
        from pyriemann.utils.tangentspace import tangent_space
    from sklearn.linear_model import LogisticRegression
    return Covariances, MDM, mean_riemann, tangent_space, LogisticRegression


# =========================== loading ===========================
def _binary(X, y):
    """Keep the first two classes (left, right) and code them 0 / 1."""
    y = np.asarray(y).ravel()
    cl = np.unique(y)
    if len(cl) > 2:
        keep = np.isin(y, cl[:2])
        X, y = X[keep], y[keep]
    return X, (y == np.unique(y)[-1]).astype(int)


def load_2a(subj, mods):
    X, y = mods["WL"].load_subject("2a", subj, zscore=False)
    if X.shape[1] != 22:
        raise ValueError(f"expected 22 EEG channels, got {X.shape[1]}")
    X, y = _binary(np.asarray(X), y)
    return X[:, IDX_2A_SM, :], y, {"fs_native": 250.0, "reference": "as distributed"}


def load_2b(subj, mods):
    X, y = mods["WL"].load_subject("2b", subj, zscore=False)
    if X.shape[1] != 3:
        raise ValueError(f"expected 3 EEG channels, got {X.shape[1]}")
    X, y = _binary(np.asarray(X), y)
    return X, y, {"fs_native": 250.0, "reference": "bipolar, as distributed"}


def load_physionet(subj, mods):
    mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
    root = mods["config"].DATA_PHYSIONET
    Xs, ys, fs, per_run, dur = [], [], [], [], []
    for r in PN_RUNS:
        f = os.path.join(root, "files", f"S{subj:03d}", f"S{subj:03d}R{r:02d}.edf")
        if not os.path.exists(f):
            continue
        raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        fs.append(float(raw.info["sfreq"]))
        d = [float(a["duration"]) for a in raw.annotations if a["description"] in ("T1", "T2")]
        dur.extend(d)
        X, y = CD._epoch(CD._prep(raw, SM, 8.0, 30.0, 250), {"T1": 0, "T2": 1}, 0.5, 2.5)
        if X is not None:
            Xs.append(X)
            ys.append(y)
            per_run.append(int(len(y)))
    if not Xs:
        raise RuntimeError("no left/right trials found")
    meta = {"fs_native": fs, "runs_found": len(fs), "trials_per_run": per_run,
            "cue_duration_s": float(np.median(dur)) if dur else None, "reference": "as distributed"}
    return np.concatenate(Xs), np.concatenate(ys), meta


def load_cho(subj, mods):
    mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
    root = mods["config"].DATA_HGD
    files = sorted(glob.glob(os.path.join(root, f"sub-{subj}", "**",
                                          f"sub-{subj}_*task-imagery*_eeg.bdf"), recursive=True))
    Xs, ys, fs, n_scalp = [], [], [], []
    for f in files:
        raw = mne.io.read_raw_bdf(f, preload=True, verbose="ERROR")
        fs.append(float(raw.info["sfreq"]))
        scalp = [ch for ch in raw.ch_names if CD._norm(ch) in mods["std_names"]]
        n_scalp.append(len(scalp))
        if len(scalp) < 32:
            raise ValueError(f"only {len(scalp)} scalp channels recognized in {os.path.basename(f)}")
        raw.pick(scalp)
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        X, y = CD._epoch(CD._prep(raw, SM, 8.0, 30.0, 250), CHO_EVENTS, 0.5, 2.5)
        del raw
        if X is not None:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        raise RuntimeError("no left_hand / right_hand trials found")
    meta = {"fs_native": fs, "files": len(files), "scalp_channels": n_scalp,
            "reference": "common average of the scalp channels"}
    return np.concatenate(Xs), np.concatenate(ys), meta


LOADERS = {"2a": load_2a, "2b": load_2b, "PhysioNet": load_physionet, "Cho2017": load_cho}


def subjects_on_disk(ds, mods):
    if ds in ("2a", "2b"):
        return list(range(1, 10))
    cfg = mods["config"]
    if ds == "PhysioNet":
        root = os.path.join(cfg.DATA_PHYSIONET, "files")
        names = os.listdir(root) if os.path.isdir(root) else []
        return sorted(int(d[1:]) for d in names if d.upper().startswith("S") and d[1:].isdigit())
    names = os.listdir(cfg.DATA_HGD) if os.path.isdir(cfg.DATA_HGD) else []
    return sorted(int(d[4:]) for d in names if d.lower().startswith("sub-") and d[4:].isdigit())


def _oas(X, Covariances):
    return np.asarray(Covariances("oas").fit_transform(X.astype(np.float64) * 1e6), dtype=np.float64)


def get_subject(ds, subj, mods, libs):
    """Covariances of one subject, from the cache or from the recordings."""
    path = os.path.join(CACHE, f"{ds}_{subj:03d}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        C17 = z["C17"] if z["C17"].ndim == 3 else None
        return dict(subj=subj, y=z["y"], C3=z["C3"], C17=C17, meta=json.loads(str(z["meta"])))
    X, y, meta = LOADERS[ds](subj, mods)
    X = np.asarray(X)
    ok = np.isfinite(X).all(axis=(1, 2)) & (X.var(axis=2).min(axis=1) > 0)
    meta["trials_dropped_nonfinite_or_flat"] = int((~ok).sum())
    X, y = X[ok], np.asarray(y)[ok]
    if len(np.unique(y)) != 2:
        raise RuntimeError("fewer than two classes")
    X3 = X if ds == "2b" else X[:, TRIPLET_IN_SM, :]
    meta["n_trials"] = int(len(y))
    meta["balance"] = np.bincount(y, minlength=2).tolist()
    meta["amplitude_uV_triplet"] = float(1e6 * np.median(X3.std(axis=2)))
    meta["samples_per_trial"] = int(X.shape[2])
    C3 = _oas(X3, libs[0])
    C17 = None if ds == "2b" else _oas(X, libs[0])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, y=y, C3=C3, C17=(C17 if C17 is not None else np.zeros(0)),
                        meta=json.dumps(meta))
    return dict(subj=subj, y=y, C3=C3, C17=C17, meta=meta)


def exclusion_reason(ds, rec):
    m = rec["meta"]
    if ds == "PhysioNet":
        if rec["subj"] in PN_EXCLUDE:
            return "listed as irregular in earlier studies"
        if m.get("runs_found", 3) < 3:
            return "fewer than three runs"
        if any(abs(f - PN_FS) > 1e-6 for f in m.get("fs_native", [PN_FS])):
            return "native sampling rate is not 160 Hz"
    return ""


def load_everything(mods, libs):
    data, table = {}, []
    for ds in DATASETS:
        subs = subjects_on_disk(ds, mods)
        out(f"  {ds}: {len(subs)} subjects on disk")
        kept, t0 = [], time.time()
        for i, s in enumerate(subs):
            try:
                rec = get_subject(ds, s, mods, libs)
            except Exception as e:
                table.append((ds, s, "no", f"could not be loaded: {e!r}", {}))
                out(f"    {ds} {s}: NOT loaded ({e!r})")
                continue
            why = exclusion_reason(ds, rec)
            table.append((ds, s, "no" if why else "yes", why, rec["meta"]))
            if not why:
                kept.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(subs):
                print(f"    {ds}: {i + 1}/{len(subs)} done ({time.time() - t0:.0f} s)", flush=True)
        data[ds] = kept
    return data, table


def report_inclusion(data, table):
    out("[1] Subjects")
    for ds in DATASETS:
        rows = [r for r in table if r[0] == ds]
        used = data.get(ds, [])
        n_tr = [r["meta"]["n_trials"] for r in used]
        amp = [r["meta"]["amplitude_uV_triplet"] for r in used]
        line = f"    {ds:<10} on disk {len(rows):>3} | used {len(used):>3}"
        if used:
            line += (f" | trials per subject: median {int(np.median(n_tr))} (min {min(n_tr)}, max {max(n_tr)})"
                     f" | total {sum(n_tr)} | amplitude {np.median(amp):.1f} uV")
        out(line)
        for _, s, inc, why, meta in rows:
            if inc == "no":
                extra = ""
                if meta:
                    extra = f" [native rate {meta.get('fs_native')}, trials {meta.get('n_trials')}]"
                out(f"        excluded {ds} {s}: {why}{extra}")
        odd = [r["subj"] for r in used if ds == "PhysioNet" and r["meta"]["n_trials"] != 45]
        if odd:
            out(f"        note: used PhysioNet subjects without exactly 45 trials: {odd}")
        odd = [r["subj"] for r in used if ds == "Cho2017" and r["meta"]["n_trials"] not in (200, 240)]
        if odd:
            out(f"        note: used Cho2017 subjects without 200 or 240 trials: {odd}")
    out(f"    total subjects used: {sum(len(v) for v in data.values())}")
    out()


def save_inclusion(table, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("dataset,subject,included,reason,n_trials,balance,native_rate,amplitude_uV_triplet,"
                "trials_dropped,details\n")
        for ds, s, inc, why, m in table:
            details = json.dumps({k: v for k, v in m.items() if k in
                                  ("trials_per_run", "cue_duration_s", "files", "scalp_channels",
                                   "reference", "samples_per_trial")}).replace(",", ";")
            rate = str(m.get("fs_native", "")).replace(",", ";")
            f.write(f"{ds},{s},{inc},{why.replace(',', ';')},{m.get('n_trials', '')},"
                    f"{str(m.get('balance', '')).replace(',', ';')},{rate},"
                    f"{m.get('amplitude_uV_triplet', '')},"
                    f"{m.get('trials_dropped_nonfinite_or_flat', '')},{details}\n")


# =========================== alignment and classification ===========================
def invsqrtm(M):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    if w.min() <= 0:
        raise ValueError("reference covariance is not positive definite")
    return (V * (1.0 / np.sqrt(w))) @ V.T


def recenter(C, kind, mean_riemann):
    M = C.mean(0) if kind == "ea" else mean_riemann(C)
    P = invsqrtm(M)
    return P @ C @ P.T


def align_source(subj_covs, align, mean_riemann):
    if align == "none":
        return np.concatenate(subj_covs)
    if align.endswith("_ps"):
        return np.concatenate([recenter(C, align[:2], mean_riemann) for C in subj_covs])
    return recenter(np.concatenate(subj_covs), align, mean_riemann)


class Model:
    """Source-trained classifier, frozen before any target trial is seen."""

    def __init__(self, Cs, ys, clf, libs):
        _, MDM, mean_riemann, tangent_space, LogisticRegression = libs
        self.ts = tangent_space
        if clf == "mdm":
            self.ref, self.m = None, MDM().fit(Cs, ys)
        else:
            self.ref = mean_riemann(Cs)
            self.m = LogisticRegression(max_iter=1000).fit(tangent_space(Cs, self.ref), ys)

    def predict(self, Ct):
        return self.m.predict(Ct if self.ref is None else self.ts(Ct, self.ref))


def run_montage(data, montage, libs):
    """Returns ({(src, tgt, subj, cov, align, clf): accuracy}, dataset names)."""
    mean_riemann = libs[2]
    key = "C3" if montage == "3ch" else "C17"
    names = [d for d in MONTAGES[montage] if len(data.get(d, [])) >= 3]
    covs = {}
    for d in names:
        raw = [r[key] for r in data[d]]
        covs[d] = {"raw": raw, "tn": [C / np.trace(C, axis1=1, axis2=2)[:, None, None] for C in raw]}
    acc, cache = {}, {}

    def target_cov(tgt, j, cov, align):
        kind = "none" if align == "none" else align[:2]
        k = (tgt, j, cov, kind)
        if k not in cache:
            C = covs[tgt][cov][j]
            cache[k] = C if kind == "none" else recenter(C, kind, mean_riemann)
        return cache[k]

    for src in names:
        t0 = time.time()
        ys = np.concatenate([r["y"] for r in data[src]])
        for cov in COVS:
            for align in ALIGNS:
                Cs = align_source(covs[src][cov], align, mean_riemann)
                for clf in CLFS:
                    model = Model(Cs, ys, clf, libs)
                    for tgt in names:
                        if tgt == src:
                            continue
                        for j, r in enumerate(data[tgt]):
                            pred = model.predict(target_cov(tgt, j, cov, align))
                            acc[(src, tgt, r["subj"], cov, align, clf)] = float(np.mean(pred == r["y"]))
        print(f"    {montage}: source {src} done ({time.time() - t0:.0f} s)", flush=True)
    return acc, names


# =========================== statistics ===========================
def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(lo), float(hi)


def signflip_p(d, two_sided=True, n_mc=100000, seed=0):
    """Sign-flip permutation p-value for mean(d) = 0: exact enumeration of the 2^n
    sign patterns when n <= 16, otherwise n_mc random patterns and p = (b + 1) / (n_mc + 1)."""
    d = np.asarray(d, float)
    n, obs = len(d), float(np.mean(d))
    if n <= 16:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        perm = (signs * d).mean(1)
        b = (np.abs(perm) >= abs(obs) - 1e-12).sum() if two_sided else (perm >= obs - 1e-12).sum()
        return float(b / len(perm))
    rng = np.random.default_rng(seed)
    b, chunk = 0, max(1, 4000000 // n)
    done = 0
    while done < n_mc:
        k = min(chunk, n_mc - done)
        perm = (rng.choice([-1.0, 1.0], size=(k, n)) * d).mean(1)
        b += int((np.abs(perm) >= abs(obs) - 1e-12).sum() if two_sided else (perm >= obs - 1e-12).sum())
        done += k
    return float((b + 1) / (n_mc + 1))


def paired(a, b):
    d = 100.0 * (np.asarray(a, float) - np.asarray(b, float))
    lo, hi = boot_ci(d)
    sd = d.std(ddof=1)
    return float(d.mean()), lo, hi, (float(d.mean() / sd) if sd > 0 else float("nan")), signflip_p(d)


def holm(pvals):
    p = np.asarray(pvals, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def ptxt(p):
    return "p<0.0001" if p < 1e-4 else f"p={p:.4f}"


def fmt(res):
    d, lo, hi, dz, p = res
    dzs = f"{dz:+.2f}" if np.isfinite(dz) else "n/a"
    return f"{d:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] dz={dzs} {ptxt(p)}"


# =========================== summary ===========================
class View:
    """Accuracies of one montage with a fixed subject order per target dataset."""

    def __init__(self, acc, data, names):
        self.acc, self.names = acc, names
        self.subs = {d: [r["subj"] for r in data[d]] for d in names}

    def vec(self, src, tgt, cov, align, clf):
        return np.array([self.acc[(src, tgt, s, cov, align, clf)] for s in self.subs[tgt]])

    def over_sources(self, tgt, cov, align, clf, sources=None):
        sources = [s for s in (sources or self.names) if s != tgt]
        return np.mean([self.vec(s, tgt, cov, align, clf) for s in sources], axis=0)

    def all_targets(self, cov, align, clf):
        return np.concatenate([self.over_sources(t, cov, align, clf) for t in self.names])


def report_matrix(v, title, cov, align, clf):
    out(f"    {title}")
    out("      " + f"{'':<11}" + "".join(f"{t:>11}" for t in v.names))
    vals = []
    for s in v.names:
        line = f"      {s:<11}"
        for t in v.names:
            if s == t:
                line += f"{'-':>11}"
            else:
                a = 100.0 * v.vec(s, t, cov, align, clf).mean()
                vals.append(a)
                line += f"{a:>11.2f}"
        out(line)
    out(f"      mean over the {len(vals)} directions: {np.mean(vals):.2f}"
        f" (lowest {min(vals):.2f}, highest {max(vals):.2f})")


def report_montage(v, montage):
    n_t = sum(len(v.subs[d]) for d in v.names)
    n_dir = len(v.names) * (len(v.names) - 1)
    out(f"===== Montage {montage}: {', '.join(v.names)} | {n_dir} directions | {n_t} target subjects =====")
    out()
    out("[2] Transfer matrices (rows: source, columns: target; mean accuracy %)")
    report_matrix(v, "primary: tn + ra_ps + TS-LR", "tn", "ra_ps", "lr")
    report_matrix(v, "tn + ra_ps + MDM", "tn", "ra_ps", "mdm")
    report_matrix(v, "tn, no alignment, TS-LR", "tn", "none", "lr")
    out()

    out("[3] All directions together (each target subject averaged over its sources)")
    for clf in CLFS:
        cells = [("raw", "none"), ("tn", "none"), ("tn", "ea"), ("tn", "ra"), ("tn", "ea_ps"),
                 ("tn", "ra_ps"), ("raw", "ra_ps")]
        out(f"    {clf.upper():<4} means: "
            + " | ".join(f"{c} {a} {100.0 * v.all_targets(c, a, clf).mean():.2f}" for c, a in cells))
        g = lambda c, a: v.all_targets(c, a, clf)
        out(f"         amplitude alone (tn none - raw none)                 {fmt(paired(g('tn', 'none'), g('raw', 'none')))}")
        out(f"         recentering beyond amplitude (tn ra_ps - tn none)    {fmt(paired(g('tn', 'ra_ps'), g('tn', 'none')))}")
        out(f"         per-subject versus pooled source (tn ra_ps - tn ra)  {fmt(paired(g('tn', 'ra_ps'), g('tn', 'ra')))}")
        out(f"         Riemannian versus Euclidean (tn ra_ps - tn ea_ps)    {fmt(paired(g('tn', 'ra_ps'), g('tn', 'ea_ps')))}")
        out(f"         without normalizing amplitude (raw ra_ps - raw none) {fmt(paired(g('raw', 'ra_ps'), g('raw', 'none')))}")
    out(f"    TS-LR versus MDM at tn + ra_ps                            "
        f"{fmt(paired(v.all_targets('tn', 'ra_ps', 'lr'), v.all_targets('tn', 'ra_ps', 'mdm')))}")
    out()

    out("[4] Per source: gain from recentering once amplitude is removed (tn ra_ps - tn none)")
    for clf in CLFS:
        for s in v.names:
            tg = [t for t in v.names if t != s]
            cat = lambda c, a: np.concatenate([v.vec(s, t, c, a, clf) for t in tg])
            n = sum(len(v.subs[t]) for t in tg)
            out(f"    {clf.upper():<4} source {s:<10} n={n:<4} {fmt(paired(cat('tn', 'ra_ps'), cat('tn', 'none')))}")
    out()

    out("[5] Does that gain (TS-LR) depend on the source? Paired over the target subjects both sources share")
    for s1, s2 in itertools.combinations(v.names, 2):
        tg = [t for t in v.names if t not in (s1, s2)]
        if not tg:
            continue
        gain = lambda s: np.concatenate([v.vec(s, t, "tn", "ra_ps", "lr")
                                         - v.vec(s, t, "tn", "none", "lr") for t in tg])
        g1, g2 = gain(s1), gain(s2)
        n = sum(len(v.subs[t]) for t in tg)
        out(f"    {s1 + ' vs ' + s2:<22} n={n:<4} gain {100 * g1.mean():+6.2f} vs {100 * g2.mean():+6.2f}"
            f"   difference {fmt(paired(g1, g2))}")
    out()

    out("[6] Per direction, TS-LR: accuracy, gain from recentering, and the primary configuration against chance")
    out(f"    {'direction':<24}{'n':>4}{'raw none':>10}{'tn none':>9}{'tn ra_ps':>10}"
        f"   {'gain [95% CI] dz p':<44}{'chance p':>10}{'Holm p':>9}")
    dirs = [(s, t) for s in v.names for t in v.names if s != t]
    chance = [signflip_p(v.vec(s, t, "tn", "ra_ps", "lr") - 0.5, two_sided=False) for s, t in dirs]
    adj = holm(chance)
    for (s, t), pc, pa in zip(dirs, chance, adj):
        m = {k: 100.0 * v.vec(s, t, k[0], k[1], "lr").mean()
             for k in (("raw", "none"), ("tn", "none"), ("tn", "ra_ps"))}
        res = paired(v.vec(s, t, "tn", "ra_ps", "lr"), v.vec(s, t, "tn", "none", "lr"))
        out(f"    {s + ' -> ' + t:<24}{len(v.subs[t]):>4}{m[('raw', 'none')]:>10.2f}{m[('tn', 'none')]:>9.2f}"
            f"{m[('tn', 'ra_ps')]:>10.2f}   {fmt(res):<44}{pc:>10.4f}{pa:>9.4f}")
    out(f"    primary configuration above chance after Holm: {int((adj < 0.05).sum())}/{len(dirs)} directions")
    out()


def report_montage_comparison(v3, v17):
    shared = [d for d in v17.names if d in v3.names]
    if len(shared) < 2:
        return
    n = sum(len(v17.subs[d]) for d in shared)
    out(f"===== 17 channels versus 3 channels, same {n} target subjects, directions among {', '.join(shared)} =====")
    for clf in CLFS:
        for cov, align in (("tn", "ra_ps"), ("tn", "none")):
            a17 = np.concatenate([v17.over_sources(t, cov, align, clf, shared) for t in shared])
            a3 = np.concatenate([v3.over_sources(t, cov, align, clf, shared) for t in shared])
            out(f"    {clf.upper():<4} {cov} {align:<6} 3ch {100 * a3.mean():.2f} | 17ch {100 * a17.mean():.2f}"
                f" | difference {fmt(paired(a17, a3))}")
    out()


def report_regression(v3):
    if not all(d in v3.names for d in ("2a", "2b")):
        return
    out("[0] Regression check against the earlier runs (triplet, TS-LR, directions whose subjects are unchanged)")
    worst = 0.0
    for (s, t), ref in REGRESSION.items():
        now = [100.0 * v3.vec(s, t, c, a, "lr").mean() for c, a in (("raw", "none"), ("tn", "none"), ("tn", "ra"))]
        worst = max(worst, max(abs(x - r) for x, r in zip(now, ref)))
        out(f"    {s} -> {t}: raw none {ref[0]:.2f} / {now[0]:.2f} | tn none {ref[1]:.2f} / {now[1]:.2f}"
            f" | tn ra {ref[2]:.2f} / {now[2]:.2f}   (earlier / now)")
    out(f"    largest difference: {worst:.2f} points "
        + ("(matches)" if worst <= 0.15 else "<-- does NOT match the earlier runs: stop and report this"))
    out()


def save_csv(results, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("montage,source,target,subject,covariance,alignment,classifier,acc\n")
        for montage, acc in results.items():
            for (src, tgt, s, cov, al, clf), a in acc.items():
                f.write(f"{montage},{src},{tgt},{s},{cov},{al},{clf},{a:.6f}\n")


def check_2a_channels(mods):
    """Print the channel names of BCI-IV-2a as stored, to document the 17-channel block."""
    try:
        f = os.path.join(mods["config"].DATA_2A, "A01T.gdf")
        raw = mods["mne"].io.read_raw_gdf(f, preload=False, verbose="ERROR")
        names = [c for c in raw.ch_names if "EOG" not in c.upper()]
        ok = (len(names) == 22 and "FZ" in names[0].upper() and "C3" in names[7].upper()
              and "CZ" in names[9].upper() and "C4" in names[11].upper() and "PZ" in names[19].upper())
        out("  BCI-IV-2a channel names as stored: " + ", ".join(names))
        out("  named anchors Fz, C3, Cz, C4, Pz at positions 0, 7, 9, 11, 19: "
            + ("yes" if ok else "NO <-- the 17-channel block for 2a must be checked"))
    except Exception as e:
        out(f"  (2a header not read: {e!r})")


def run():
    t_start = time.time()
    try:
        import mne
        import config
        import crossdata as CD
        import variants as V
        import within_loader as WL
    except Exception as e:
        print(f"could not import the project modules; run this from the Codes folder. ({e!r})")
        return 1
    libs = _libs()
    import pyriemann
    import sklearn
    mods = dict(mne=mne, config=config, CD=CD, WL=WL, SM=list(V.SM),
                std_names={n.upper() for n in mne.channels.make_standard_montage("standard_1005").ch_names})
    out("Step 5 benchmark  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    out(f"python {sys.version.split()[0]} | numpy {np.__version__} | mne {mne.__version__} | "
        f"pyriemann {getattr(pyriemann, '__version__', '?')} | scikit-learn {sklearn.__version__}")
    out(f"recipe: {RECIPE}")
    out(f"17 channels: {', '.join(mods['SM'])}")
    check_2a_channels(mods)
    out()
    out("Loading (cached subjects are read from " + CACHE + ")")
    data, table = load_everything(mods, libs)
    out()
    os.makedirs(OUTDIR, exist_ok=True)
    save_inclusion(table, os.path.join(OUTDIR, "benchmark_inclusion.csv"))
    report_inclusion(data, table)

    results, views = {}, {}
    out("Running the transfer experiments")
    for montage in MONTAGES:
        acc, names = run_montage(data, montage, libs)
        if len(names) >= 2:
            results[montage] = acc
            views[montage] = View(acc, data, names)
    out()
    save_csv(results, os.path.join(OUTDIR, "benchmark_subjects.csv"))
    if "3ch" in views:
        report_regression(views["3ch"])
    for montage, v in views.items():
        report_montage(v, montage)
    if "3ch" in views and "17ch" in views:
        report_montage_comparison(views["3ch"], views["17ch"])
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "benchmark_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")
    print(f"written: {os.path.abspath(OUTDIR)} (benchmark_summary.txt, benchmark_subjects.csv, "
          "benchmark_inclusion.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
