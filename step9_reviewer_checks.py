"""
step9_reviewer_checks.py

Run from the Codes folder, after steps 5 and 8 (about 30 to 45 minutes; parts
[L], [M] and [N] read the recordings again):

    python step9_reviewer_checks.py

Five analyses that answer reviewer-type concerns. Unless stated otherwise the
pipeline is the primary one of step 5 (trace-normalized OAS covariance, every
subject recentered on its own Riemannian mean, tangent-space logistic
regression fitted on the source and frozen). Target labels only score.

[J] A classical decoder that is not Riemannian: CSP + LDA, with and without
    recentering, in every direction (from the cache).
[K] Pseudo-online use: each trial is classified with a reference built only
    from the EARLIER trials of the same subject (targets that preserve the
    recording order: 2a, 2b, PhysioNet). Plain running mean, and the shrinkage
    estimate of step 8 that starts from the site prior.
[L] Sessions: transfer between recording days of the same subject in
    BCI-IV-2b, with and without recentering each session, next to
    cross-dataset transfer scored on the same trials.
[M] Three classes across datasets: left hand, right hand and feet, between
    BCI-IV-2a and PhysioNet (chance 33.3%).
[N] Sensitivity of the main contrast to the frequency band and to the time
    window (the recordings are filtered and cut again).

Each part is independent: if one fails, the failure is reported and the others
still run. Nothing existing is modified. Outputs in corrected_results/:
    step9_summary.txt, step9_csp.csv, step9_online.csv, step9_sessions.csv,
    step9_threeclass.csv, step9_sensitivity.csv, and a refreshed results_bundle.zip
"""
from __future__ import annotations

import os
import sys
import glob
import json
import time
import datetime
import traceback

import numpy as np

try:
    import step5_benchmark as S5
    import step8_new_analyses as S8
except Exception as _e:                                    # pragma: no cover
    print(f"could not import step5_benchmark.py / step8_new_analyses.py; run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
out = S5.out
tn = S8.tn
ORDERED = ("2a", "2b", "PhysioNet")
BLOCKS = [("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16), ("17-32", 17, 32), ("33+", 33, 10 ** 9),
          ("1-8", 1, 8), ("all", 1, 10 ** 9)]
SENS = [("band 4-8 Hz", (4.0, 8.0), (0.5, 2.5)),
        ("band 8-13 Hz", (8.0, 13.0), (0.5, 2.5)),
        ("band 13-30 Hz", (13.0, 30.0), (0.5, 2.5)),
        ("window 0-2 s", (8.0, 30.0), (0.0, 2.0)),
        ("window 1-3 s", (8.0, 30.0), (1.0, 3.0))]
PN_LR_RUNS, PN_FEET_RUNS = (4, 8, 12), (6, 10, 14)


def project_modules():
    import mne
    import config
    import crossdata as CD
    import variants as V
    import within_loader as WL
    return dict(mne=mne, config=config, CD=CD, WL=WL, SM=list(V.SM),
                std_names={n.upper() for n in mne.channels.make_standard_montage("standard_1005").ch_names})


def read_benchmark():
    """{(montage, align): {(target, subject): accuracy averaged over sources}} for tn + TS-LR."""
    path = os.path.join(OUTDIR, "benchmark_subjects.csv")
    acc = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            m, s, t, sub, cov, al, clf, a = line.strip().split(",")
            if cov == "tn" and clf == "lr" and al in ("none", "ra_ps"):
                acc.setdefault((m, al), {}).setdefault((t, int(sub)), []).append(float(a))
    return {k: {kk: float(np.mean(v)) for kk, v in d.items()} for k, d in acc.items()}


def write_csv(name, header, rows):
    with open(os.path.join(OUTDIR, name), "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")


def mean_ci(x):
    lo, hi = S5.boot_ci(100.0 * np.asarray(x, float))
    return f"{100 * np.mean(x):6.2f} [{lo:5.1f}, {hi:5.1f}]"


# =========================== [J] CSP + LDA ===========================
class CspLda:
    """Common spatial patterns on class-mean covariances, log-variance features, LDA."""

    def __init__(self, Cs, ys, n_pairs):
        from scipy.linalg import eigh
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        S0, S1 = Cs[ys == 0].mean(0), Cs[ys == 1].mean(0)
        w, V = eigh(S1, S0 + S1)                       # ascending eigenvalues
        pick = np.r_[np.arange(n_pairs), np.arange(len(w) - n_pairs, len(w))]
        self.W = V[:, pick].T
        self.lda = LinearDiscriminantAnalysis().fit(self._features(Cs), ys)

    def _features(self, C):
        v = np.einsum("fi,nij,fj->nf", self.W, C, self.W)
        return np.log(v / v.sum(axis=1, keepdims=True))

    def predict(self, C):
        return self.lda.predict(self._features(C))


def section_j(mont, bench):
    rows = []
    for M in mont:
        t0 = time.time()
        n_pairs = 1 if M.name == "3ch" else 3
        variants = {"none": {d: M.C[d] for d in M.names},
                    "ea": {d: [S8.apply_ref(C.mean(0), C) for C in M.C[d]] for d in M.names},
                    "ra": {d: [S8.apply_ref(R, C) for R, C in zip(M.R[d], M.C[d])] for d in M.names}}
        for cond, V in variants.items():
            for s in M.names:
                model = CspLda(np.concatenate(V[s]), np.concatenate(M.y[s]), n_pairs)
                for t in M.names:
                    if t == s:
                        continue
                    for j, C in enumerate(V[t]):
                        rows.append((M.name, cond, s, t, M.subj[t][j], float(np.mean(model.predict(C) == M.y[t][j]))))
        print(f"    [J] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[J] CSP + LDA (log-variance of 2 filters on three channels, 6 filters on 17), trace-normalized covariances")
    out("    none: no alignment | EA / RA: every subject recentered on its arithmetic / Riemannian mean")
    for M in mont:
        sel = [r for r in rows if r[0] == M.name]
        keys = sorted({(r[3], r[4]) for r in sel})
        v = {c: np.array([np.mean([r[5] for r in sel if r[1] == c and (r[3], r[4]) == k]) for k in keys])
             for c in ("none", "ea", "ra")}
        out(f"    {M.name}: {len(keys)} target subjects, mean accuracy % [95% CI]")
        out(f"      CSP+LDA  none {mean_ci(v['none'])} | EA {mean_ci(v['ea'])} | RA {mean_ci(v['ra'])}")
        out(f"      EA - none          {S5.fmt(S5.paired(v['ea'], v['none']))}")
        out(f"      RA - EA            {S5.fmt(S5.paired(v['ra'], v['ea']))}")
        ts_none = np.array([bench[(M.name, "none")][k] for k in keys])
        ts_prim = np.array([bench[(M.name, "ra_ps")][k] for k in keys])
        out(f"      TS-LR    none {mean_ci(ts_none)} | primary {mean_ci(ts_prim)}   (step 5, same subjects)")
        out(f"      TS-LR primary - CSP+LDA with RA   {S5.fmt(S5.paired(ts_prim, v['ra']))}")
        out(f"      TS-LR none - CSP+LDA none         {S5.fmt(S5.paired(ts_none, v['none']))}")
    out()
    write_csv("step9_csp.csv", "montage,alignment,source,target,subject,accuracy", rows)


# =========================== [K] pseudo-online ===========================
def _running_mean(mean_riemann, C, init):
    try:
        return mean_riemann(C, init=init)
    except TypeError:
        return mean_riemann(C)


def section_k(mont, libs):
    mean_riemann = libs[2]
    rows = []
    for M in mont:
        t0 = time.time()
        for tgt in [d for d in M.names if d in ORDERED]:
            srcs = M.sources(tgt)
            for j, C in enumerate(M.C[tgt]):
                y, n = M.y[tgt][j], len(C)
                site, k0 = M.site[tgt][j], float(M.k0_site[tgt][j])
                plain, shrunk = np.empty_like(C), np.empty_like(C)
                plain[0] = C[0]                                    # nothing is known yet
                shrunk[0] = S8.apply_ref(site, C[:1])[0]           # site prior only
                R = C[0]
                for t in range(1, n):
                    R = C[0] if t == 1 else _running_mean(mean_riemann, C[:t], R)
                    plain[t] = S8.apply_ref(R, C[t:t + 1])[0]
                    shrunk[t] = S8.apply_ref(S8.geodesic(site, R, t / (t + k0)), C[t:t + 1])[0]
                offline = S8.apply_ref(M.R[tgt][j], C)
                ok = {name: np.mean([M.models[s].predict(X) == y for s in srcs], axis=0)
                      for name, X in (("plain", plain), ("shrunk", shrunk), ("offline", offline))}
                idx = np.arange(1, n + 1)
                for label, lo, hi in BLOCKS:
                    sel = (idx >= lo) & (idx <= hi)
                    if sel.sum():
                        rows.append((M.name, tgt, M.subj[tgt][j], label, int(sel.sum()),
                                     float(ok["plain"][sel].mean()), float(ok["shrunk"][sel].mean()),
                                     float(ok["offline"][sel].mean())))
        print(f"    [K] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[K] Pseudo-online use: trial t is recentered with a reference from trials 1 .. t-1 of the same subject only")
    out("    plain: running Riemannian mean (trial 1 is not recentered) | shrunk: shrinkage from the site prior (step 8)")
    out("    offline: reference from all trials of the subject (main analysis). Accuracy %, mean over subjects.")
    for M in mont:
        sel = [r for r in rows if r[0] == M.name]
        if not sel:
            continue
        tg = [d for d in M.names if d in ORDERED]
        nsub = len({(r[1], r[2]) for r in sel})
        out(f"    {M.name}: targets {', '.join(tg)} ({nsub} subjects)")
        out(f"      {'trials':>8}{'subjects':>10}{'plain':>9}{'shrunk':>9}{'offline':>9}   shrunk - plain  [95% CI]  dz  p")
        for label, _, _ in BLOCKS:
            x = np.array([r[5:] for r in sel if r[3] == label])
            if len(x) < 3:
                continue
            out(f"      {label:>8}{len(x):>10}{100 * x[:, 0].mean():>9.2f}{100 * x[:, 1].mean():>9.2f}"
                f"{100 * x[:, 2].mean():>9.2f}   {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}")
        x = np.array([r[5:] for r in sel if r[3] == "all"])
        out(f"      whole recording: offline - shrunk  {S5.fmt(S5.paired(x[:, 2], x[:, 1]))}")
        out(f"      whole recording: offline - plain   {S5.fmt(S5.paired(x[:, 2], x[:, 0]))}")
    out()
    write_csv("step9_online.csv", "montage,target,subject,trial_block,n_trials,acc_plain,acc_shrunk,acc_offline", rows)


# =========================== [L] sessions of BCI-IV-2b ===========================
def cv_accuracy(C, y, libs, folds=5, repeats=5):
    from sklearn.model_selection import StratifiedKFold
    acc = []
    for rep in range(repeats):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=rep).split(C, y):
            acc.append(np.mean(S5.Model(C[tr], y[tr], "lr", libs).predict(C[te]) == y[te]))
    return float(np.mean(acc))


def section_l(mods, libs, mont):
    WL, mean_riemann = mods["WL"], libs[2]
    M3 = [M for M in mont if M.name == "3ch"][0]
    srcs = M3.sources("2b")

    def rc(C):
        return S8.apply_ref(mean_riemann(C), C)

    rows = []
    for subj in range(1, 10):
        sess = {}
        for s in (1, 2, 3):
            path = os.path.join(WL.ROOT["2b"], f"B{subj:02d}0{s}T.gdf")
            X, y = WL._epoch_file(path, WL.CODES["2b"], 0.5, 2.5, 8.0, 30.0, None)
            if X is None:
                raise RuntimeError(f"no trials in {path}")
            X, y = S5._binary(np.asarray(X), y)
            sess[s] = (tn(S5._oas(X, libs[0])), y)
        for label, train, test in (("day 1 -> day 2", (1,), 2), ("day 2 -> day 1", (2,), 1),
                                   ("days 1+2 -> day 3", (1, 2), 3)):
            Ct, yt = sess[test]
            ytr = np.concatenate([sess[s][1] for s in train])
            a_none = np.mean(S5.Model(np.concatenate([sess[s][0] for s in train]), ytr, "lr", libs).predict(Ct) == yt)
            a_rec = np.mean(S5.Model(np.concatenate([rc(sess[s][0]) for s in train]), ytr, "lr", libs)
                            .predict(rc(Ct)) == yt)
            a_cv = cv_accuracy(Ct, yt, libs)
            a_xd = np.mean([np.mean(M3.models[s].predict(rc(Ct)) == yt) for s in srcs])
            rows.append((subj, label, int(len(ytr)), int(len(yt)), float(a_none), float(a_rec), float(a_cv), float(a_xd)))
        print(f"    [L] 2b subject {subj}: done", flush=True)
    out("[L] BCI-IV-2b, transfer between recording days of the same subject (three bipolar channels, TS-LR)")
    out("    none: classifier of the training day(s) applied as is | recentered: every session recentered on its own")
    out("    unlabeled mean | within-day: 5-fold cross-validation inside the test day | cross-dataset: frozen models of")
    out("    the other datasets on the same test trials (test day recentered). Mean accuracy % over 9 subjects.")
    out(f"      {'transfer':<20}{'none':>8}{'recentered':>12}{'within-day':>12}{'cross-dataset':>15}")
    groups = (("between days 1 and 2", ("day 1 -> day 2", "day 2 -> day 1")), ("days 1+2 -> day 3", ("days 1+2 -> day 3",)))
    summary = {}
    for name, labels in groups:
        x = np.array([[np.mean([r[c] for r in rows if r[0] == s and r[1] in labels]) for c in (4, 5, 6, 7)]
                      for s in range(1, 10)])
        summary[name] = x
        m = 100 * x.mean(0)
        out(f"      {name:<20}{m[0]:>8.2f}{m[1]:>12.2f}{m[2]:>12.2f}{m[3]:>15.2f}")
    for name, x in summary.items():
        out(f"    {name}")
        out(f"      recentered - none                 {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}")
        out(f"      recentered - cross-dataset        {S5.fmt(S5.paired(x[:, 1], x[:, 3]))}")
        out(f"      within-day - recentered           {S5.fmt(S5.paired(x[:, 2], x[:, 1]))}")
    out()
    write_csv("step9_sessions.csv",
              "subject,transfer,n_train,n_test,acc_none,acc_recentered,acc_within_day_cv,acc_cross_dataset", rows)


# =========================== [M] three classes ===========================
def load_2a_three(subj, mods):
    X, y = mods["WL"].load_subject("2a", subj, zscore=False)
    X, y = np.asarray(X), np.asarray(y).ravel()
    cl = np.unique(y)                                   # left hand, right hand, feet, tongue
    if len(cl) < 3:
        raise RuntimeError("fewer than three classes in BCI-IV-2a")
    keep = np.isin(y, cl[:3])
    return X[keep][:, S5.IDX_2A_SM, :], np.searchsorted(cl[:3], y[keep]), {}


def load_pn_three(subj, mods):
    mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
    root = mods["config"].DATA_PHYSIONET
    Xs, ys, fs = [], [], []
    for runs, emap in ((PN_LR_RUNS, {"T1": 0, "T2": 1}), (PN_FEET_RUNS, {"T2": 2})):
        for r in runs:
            f = os.path.join(root, "files", f"S{subj:03d}", f"S{subj:03d}R{r:02d}.edf")
            if not os.path.exists(f):
                continue
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            fs.append(float(raw.info["sfreq"]))
            X, y = CD._epoch(CD._prep(raw, SM, 8.0, 30.0, 250), emap, 0.5, 2.5)
            if X is not None:
                Xs.append(X)
                ys.append(y)
    if not Xs:
        raise RuntimeError("no trials found")
    return np.concatenate(Xs), np.concatenate(ys), {"fs_native": fs, "runs_found": len(fs)}


def get_three(ds, subj, mods, libs, cache):
    path = os.path.join(cache, f"{ds}_{subj:03d}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return dict(subj=subj, y=z["y"], C3=z["C3"], C17=z["C17"], meta=json.loads(str(z["meta"])))
    X, y, meta = (load_2a_three if ds == "2a" else load_pn_three)(subj, mods)
    ok = np.isfinite(X).all(axis=(1, 2)) & (X.var(axis=2).min(axis=1) > 0)
    X, y = X[ok], np.asarray(y)[ok]
    meta["balance"] = np.bincount(y, minlength=3).tolist()
    C3, C17 = S5._oas(X[:, S5.TRIPLET_IN_SM, :], libs[0]), S5._oas(X, libs[0])
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(path, y=y, C3=C3, C17=C17, meta=json.dumps(meta))
    return dict(subj=subj, y=y, C3=C3, C17=C17, meta=meta)


def section_m(mods, libs, main):
    mean_riemann = libs[2]
    cache = S5.CACHE + "_3class"
    data, t0 = {"2a": [], "PhysioNet": []}, time.time()
    for ds in data:
        for r in main[ds]:                              # the subjects of the main analysis
            rec = get_three(ds, r["subj"], mods, libs, cache)
            bal = rec["meta"]["balance"]
            if ds == "PhysioNet" and (rec["meta"].get("runs_found", 6) < 6 or min(bal) < 10
                                      or any(abs(f - S5.PN_FS) > 1e-6 for f in rec["meta"].get("fs_native", []))):
                continue
            data[ds].append(rec)
    print(f"    [M] three-class data ready ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    for montage, key in (("3ch", "C3"), ("17ch", "C17")):
        C = {d: [tn(r[key]) for r in data[d]] for d in data}
        Cr = {d: [S8.apply_ref(mean_riemann(c), c) for c in C[d]] for d in data}
        for s, t in (("2a", "PhysioNet"), ("PhysioNet", "2a")):
            ys = np.concatenate([r["y"] for r in data[s]])
            for clf in ("lr", "mdm"):
                m0 = S5.Model(np.concatenate(C[s]), ys, clf, libs)
                m1 = S5.Model(np.concatenate(Cr[s]), ys, clf, libs)
                for j, r in enumerate(data[t]):
                    rows.append((montage, s, t, r["subj"], clf, float(np.mean(m0.predict(C[t][j]) == r["y"])),
                                 float(np.mean(m1.predict(Cr[t][j]) == r["y"]))))
    out("[M] Three classes (left hand, right hand, feet) between BCI-IV-2a and PhysioNet; chance is 33.3%")
    out(f"    subjects: 2a {len(data['2a'])}, PhysioNet {len(data['PhysioNet'])} | trials per class, median: "
        + ", ".join(f"{d} {np.median([min(r['meta']['balance']) for r in data[d]]):.0f}" for d in data))
    out(f"      {'montage':<8}{'direction':<20}{'clf':<6}{'n':>4}{'none':>8}{'recentered':>12}   gain [95% CI] dz p"
        f"{'':<22}p vs chance")
    for montage in ("3ch", "17ch"):
        for s, t in (("2a", "PhysioNet"), ("PhysioNet", "2a")):
            for clf in ("lr", "mdm"):
                x = np.array([r[5:] for r in rows if r[:3] == (montage, s, t) and r[4] == clf])
                pc = S5.signflip_p(100 * (x[:, 1] - 1 / 3), two_sided=False)
                out(f"      {montage:<8}{s + ' -> ' + t:<20}{clf:<6}{len(x):>4}{100 * x[:, 0].mean():>8.2f}"
                    f"{100 * x[:, 1].mean():>12.2f}   {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}   {S5.ptxt(pc)}")
    out()
    write_csv("step9_threeclass.csv", "montage,source,target,subject,classifier,acc_none,acc_recentered", rows)


# =========================== [N] band and window ===========================
def make_loaders(band, win):
    lo, hi = band
    t0, t1 = win

    def l2a(subj, mods):
        X, y = mods["WL"].load_subject("2a", subj, tmin=t0, tmax=t1, l_freq=lo, h_freq=hi, zscore=False)
        X, y = S5._binary(np.asarray(X), y)
        return X[:, S5.IDX_2A_SM, :], y, {"fs_native": 250.0}

    def l2b(subj, mods):
        X, y = mods["WL"].load_subject("2b", subj, tmin=t0, tmax=t1, l_freq=lo, h_freq=hi, zscore=False)
        X, y = S5._binary(np.asarray(X), y)
        return X, y, {"fs_native": 250.0}

    def lpn(subj, mods):
        mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
        Xs, ys = [], []
        for r in S5.PN_RUNS:
            f = os.path.join(mods["config"].DATA_PHYSIONET, "files", f"S{subj:03d}", f"S{subj:03d}R{r:02d}.edf")
            if not os.path.exists(f):
                continue
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            X, y = CD._epoch(CD._prep(raw, SM, lo, hi, 250), {"T1": 0, "T2": 1}, t0, t1)
            if X is not None:
                Xs.append(X)
                ys.append(y)
        if not Xs:
            raise RuntimeError("no trials")
        return np.concatenate(Xs), np.concatenate(ys), {}

    def lcho(subj, mods):
        mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
        files = sorted(glob.glob(os.path.join(mods["config"].DATA_HGD, f"sub-{subj}", "**",
                                              f"sub-{subj}_*task-imagery*_eeg.bdf"), recursive=True))
        Xs, ys = [], []
        for f in files:
            raw = mne.io.read_raw_bdf(f, preload=True, verbose="ERROR")
            raw.pick([ch for ch in raw.ch_names if CD._norm(ch) in mods["std_names"]])
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            X, y = CD._epoch(CD._prep(raw, SM, lo, hi, 250), S5.CHO_EVENTS, t0, t1)
            del raw
            if X is not None:
                Xs.append(X)
                ys.append(y)
        if not Xs:
            raise RuntimeError("no trials")
        return np.concatenate(Xs), np.concatenate(ys), {}

    return {"2a": l2a, "2b": l2b, "PhysioNet": lpn, "Cho2017": lcho}


def lean_run(data, montage, libs):
    """No alignment and primary configuration, TS-LR only: per-target-subject accuracy over sources."""
    mean_riemann = libs[2]
    key = "C3" if montage == "3ch" else "C17"
    names = [d for d in S5.MONTAGES[montage] if len(data.get(d, [])) >= 3]
    C = {d: [tn(r[key]) for r in data[d]] for d in names}
    Cr = {d: [S8.apply_ref(mean_riemann(c), c) for c in C[d]] for d in names}
    acc = {}
    for s in names:
        ys = np.concatenate([r["y"] for r in data[s]])
        m0 = S5.Model(np.concatenate(C[s]), ys, "lr", libs)
        m1 = S5.Model(np.concatenate(Cr[s]), ys, "lr", libs)
        for t in names:
            if t == s:
                continue
            for j, r in enumerate(data[t]):
                a = acc.setdefault((t, r["subj"]), ([], []))
                a[0].append(np.mean(m0.predict(C[t][j]) == r["y"]))
                a[1].append(np.mean(m1.predict(Cr[t][j]) == r["y"]))
    keys = sorted(acc)
    return keys, np.array([np.mean(acc[k][0]) for k in keys]), np.array([np.mean(acc[k][1]) for k in keys])


def section_n(mods, libs, main):
    rows, table = [], []
    configs = [("main: 8-30 Hz, 0.5-2.5 s", None, None)] + SENS
    saved_cache, saved_loaders = S5.CACHE, S5.LOADERS
    try:
        for name, band, win in configs:
            t0 = time.time()
            if band is None:
                data = main
            else:
                tag = f"bp{band[0]:g}-{band[1]:g}_win{win[0]:g}-{win[1]:g}"
                S5.CACHE = os.path.join(OUTDIR, "cache_sens_" + tag)
                S5.LOADERS = make_loaders(band, win)
                data = {d: [S5.get_subject(d, r["subj"], mods, libs) for r in main[d]] for d in S5.DATASETS}
            for montage in S5.MONTAGES:
                keys, a0, a1 = lean_run(data, montage, libs)
                table.append((name, montage, len(keys), a0, a1))
                rows += [(name.replace(",", ";"), montage, k[0], k[1], float(x0), float(x1))
                         for k, x0, x1 in zip(keys, a0, a1)]
            print(f"    [N] {name}: done ({time.time() - t0:.0f} s)", flush=True)
    finally:
        S5.CACHE, S5.LOADERS = saved_cache, saved_loaders
    out("[N] Sensitivity to the frequency band and to the time window (TS-LR, same subjects as the main analysis)")
    out(f"      {'configuration':<28}{'montage':<8}{'n':>4}{'none':>8}{'recentered':>12}   gain [95% CI] dz p")
    for name, montage, n, a0, a1 in table:
        out(f"      {name:<28}{montage:<8}{n:>4}{100 * a0.mean():>8.2f}{100 * a1.mean():>12.2f}   {S5.fmt(S5.paired(a1, a0))}")
    out()
    write_csv("step9_sensitivity.csv", "configuration,montage,target,subject,acc_none,acc_recentered", rows)


# =========================== driver ===========================
def guard(label, fn, *args):
    t0 = time.time()
    try:
        fn(*args)
    except Exception as e:                                # one part must not stop the others
        out(f"[{label}] FAILED and was skipped: {e!r}")
        out("    " + traceback.format_exc().strip().splitlines()[-3].strip())
        out()
    print(f"    part [{label}] finished ({(time.time() - t0) / 60:.1f} min)", flush=True)


def run():
    t_start = time.time()
    libs = S5._libs()
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 9 reviewer checks  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    main = S8.load_cache()
    out("subjects: " + ", ".join(f"{d} {len(main[d])}" for d in S5.DATASETS))
    out()
    mont = [S8.Montage(main, m, libs) for m in S5.MONTAGES]
    bench = read_benchmark()
    guard("J", section_j, mont, bench)
    guard("K", section_k, mont, libs)
    try:
        mods = project_modules()
    except Exception as e:
        mods = None
        out(f"[L] [M] [N] skipped: the project modules could not be imported ({e!r})")
    if mods is not None:
        guard("L", section_l, mods, libs, mont)
        guard("M", section_m, mods, libs, main)
        guard("N", section_n, mods, libs, main)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step9_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    try:
        import step7_final_checks as S7
        names = S7.make_bundle()
        print(f"bundle refreshed: {os.path.abspath(S7.BUNDLE)} ({len(names)} files) -- attach this one file in the chat")
    except Exception as e:
        print(f"written: {os.path.abspath(OUTDIR)} (step9_summary.txt and CSV files); bundle not refreshed ({e!r})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
