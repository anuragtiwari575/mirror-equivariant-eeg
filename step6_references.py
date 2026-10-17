"""
step6_references.py

Run from the Codes folder, after step5_benchmark.py:

    python step6_references.py

It reads only the cache and the CSV written by step 5 (no recording is opened,
about 30 minutes) and adds what a reader needs in order to judge the transfer
numbers:

[A] Within-dataset references for every subject, both montages
      within-subject  stratified 5-fold cross-validation, repeated 5 times
      LOSO            leave-one-subject-out inside the dataset, every subject
                      recentered on its own trials
    and, subject by subject, how zero-label cross-dataset transfer compares.
[B] How many unlabeled target trials recentering needs. The target reference
    is estimated from k trials (labels are never used) and the frozen source
    model is scored on the remaining trials:
      random  k trials drawn at random, 20 draws
      first   the first k trials in recording order (2a, 2b, PhysioNet; the
              Cho2017 file is not stored in recording order)
[C] Sensitivity: the six PhysioNet subjects excluded on precedent put back in.

Everything uses the pipeline of step 5: OAS covariance divided by its trace,
Riemannian recentering of every subject, tangent-space logistic regression
(MDM is reported next to it in [A]).

Outputs (new files only)
  corrected_results/references_subjects.csv   per subject: within-subject, LOSO, transfer
  corrected_results/calibration_curve.csv     per subject and k
  corrected_results/step6_summary.txt         the tables printed on screen
"""
from __future__ import annotations

import os
import sys
import glob
import time
import datetime

import numpy as np

try:
    import step5_benchmark as S5
except Exception as _e:                                    # pragma: no cover
    print(f"could not import step5_benchmark.py; run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
K_LIST = [1, 2, 4, 8, 16, 32]
N_DRAWS = 20
N_REP = 5
MIN_REST = 5                      # trials that must remain for scoring
NO_ORDER = {"Cho2017"}            # trials are not stored in recording order

out = S5.out


def tn(C):
    return C / np.trace(C, axis1=1, axis2=2)[:, None, None]


# =========================== data ===========================
def load_cache():
    """Every cached subject, and the subset used in the main analysis."""
    everything = {}
    for ds in S5.DATASETS:
        recs = []
        for f in sorted(glob.glob(os.path.join(S5.CACHE, f"{ds}_*.npz"))):
            subj = int(os.path.basename(f)[len(ds) + 1:-4])
            recs.append(S5.get_subject(ds, subj, None, None))
        everything[ds] = sorted(recs, key=lambda r: r["subj"])
    main = {ds: [r for r in recs if not S5.exclusion_reason(ds, r)] for ds, recs in everything.items()}
    return main, everything


def read_transfer(path):
    """{(montage, target, subject, cov, align, clf): accuracy averaged over sources}."""
    acc = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if header != ["montage", "source", "target", "subject", "covariance", "alignment",
                      "classifier", "acc"]:
            raise ValueError(f"unexpected header in {path}: {header}")
        for line in f:
            if line.strip():
                m, s, t, sub, cov, al, clf, a = line.strip().split(",")
                acc.setdefault((m, t, int(sub), cov, al, clf), []).append(float(a))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# =========================== [A] references ===========================
def within_subject(C, y, clf, libs):
    from sklearn.model_selection import StratifiedKFold
    accs = []
    for rep in range(N_REP):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(C, y):
            accs.append(float(np.mean(S5.Model(C[tr], y[tr], clf, libs).predict(C[te]) == y[te])))
    return float(np.mean(accs))


def loso(recs, key, clf, libs):
    mean_riemann = libs[2]
    aligned = [S5.recenter(tn(r[key]), "ra", mean_riemann) for r in recs]
    ys = [r["y"] for r in recs]
    accs = []
    for i in range(len(recs)):
        Cs = np.concatenate([a for j, a in enumerate(aligned) if j != i])
        ytr = np.concatenate([v for j, v in enumerate(ys) if j != i])
        accs.append(float(np.mean(S5.Model(Cs, ytr, clf, libs).predict(aligned[i]) == ys[i])))
    return accs


def section_a(data, transfer, libs):
    rows = []                                 # (montage, dataset, subject, clf, within, loso, transfer, none)
    for montage, members in S5.MONTAGES.items():
        key = "C3" if montage == "3ch" else "C17"
        names = [d for d in members if len(data.get(d, [])) >= 3]
        for ds in names:
            t0 = time.time()
            recs = data[ds]
            for clf in S5.CLFS:
                lo = loso(recs, key, clf, libs)
                for r, a_loso in zip(recs, lo):
                    w = within_subject(tn(r[key]), r["y"], clf, libs)
                    k = (montage, ds, r["subj"])
                    rows.append((montage, ds, r["subj"], clf, w, a_loso,
                                 transfer[k + ("tn", "ra_ps", clf)], transfer[k + ("tn", "none", clf)]))
            print(f"    [A] {montage} {ds}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[A] Within-dataset references and zero-label cross-dataset transfer (mean accuracy %, subjects as units)")
    out("    transfer = tn + ra_ps, each target subject averaged over its source datasets; no target label is used")
    for montage in S5.MONTAGES:
        for clf in S5.CLFS:
            sel = [r for r in rows if r[0] == montage and r[3] == clf]
            if not sel:
                continue
            out(f"    {montage}, {clf.upper()}")
            out(f"      {'target':<11}{'n':>4}{'within-subj':>13}{'LOSO':>8}{'transfer':>10}{'no align':>10}")
            groups = [d for d in S5.DATASETS if any(r[1] == d for r in sel)] + ["all"]
            stats = []
            for g in groups:
                x = [r for r in sel if g == "all" or r[1] == g]
                w, lo, tr, no = (np.array([r[i] for r in x]) for i in (4, 5, 6, 7))
                out(f"      {g:<11}{len(x):>4}{100 * w.mean():>13.2f}{100 * lo.mean():>8.2f}{100 * tr.mean():>10.2f}"
                    f"{100 * no.mean():>10.2f}")
                stats.append((g, S5.paired(tr, w), S5.paired(tr, lo)))
            out(f"      {'':<11}{'transfer - within-subject':<46}{'transfer - LOSO'}")
            for g, a, b in stats:
                out(f"      {g:<11}{S5.fmt(a):<46}{S5.fmt(b)}")
            tr = np.array([r[6] for r in sel])
            w = np.array([r[4] for r in sel])
            out(f"      subjects at or above 70%: transfer {100 * np.mean(tr >= 0.70):.1f}% | "
                f"within-subject {100 * np.mean(w >= 0.70):.1f}% | "
                f"transfer beats within-subject in {100 * np.mean(tr > w):.1f}% of subjects")
    out()
    with open(os.path.join(OUTDIR, "references_subjects.csv"), "w", encoding="utf-8") as f:
        f.write("montage,dataset,subject,classifier,within_subject,loso,transfer_primary,transfer_no_alignment\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f},{r[7]:.6f}\n")


# =========================== [B] unlabeled trials needed ===========================
def _recenter_with(C_ref, C_apply, mean_riemann):
    M = C_ref[0] if len(C_ref) == 1 else mean_riemann(C_ref)
    P = S5.invsqrtm(M)
    return P @ C_apply @ P.T


def section_b(data, transfer, libs):
    mean_riemann = libs[2]
    rows = []                                  # (montage, target, subject, mode, k, acc)
    for mi, (montage, members) in enumerate(S5.MONTAGES.items()):
        key = "C3" if montage == "3ch" else "C17"
        names = [d for d in members if len(data.get(d, [])) >= 3]
        t0 = time.time()
        models = {}
        for src in names:
            Cs = S5.align_source([tn(r[key]) for r in data[src]], "ra_ps", mean_riemann)
            models[src] = S5.Model(Cs, np.concatenate([r["y"] for r in data[src]]), "lr", libs)
        for ti, tgt in enumerate(names):
            for r in data[tgt]:
                C, y, n = tn(r[key]), r["y"], len(r["y"])
                rng = np.random.default_rng([mi, ti, int(r["subj"])])
                res = {}
                res[("random", 0)] = [float(np.mean(m.predict(C) == y)) for s, m in models.items() if s != tgt]
                for k in K_LIST:
                    if n - k < MIN_REST:
                        continue
                    draws = []
                    for _ in range(N_DRAWS):
                        idx = rng.choice(n, size=k, replace=False)
                        rest = np.setdiff1d(np.arange(n), idx)
                        Ct = _recenter_with(C[idx], C[rest], mean_riemann)
                        draws.append(np.mean([np.mean(m.predict(Ct) == y[rest])
                                              for s, m in models.items() if s != tgt]))
                    res[("random", k)] = draws
                    if tgt not in NO_ORDER:
                        Ct = _recenter_with(C[:k], C[k:], mean_riemann)
                        res[("first", k)] = [np.mean([np.mean(m.predict(Ct) == y[k:])
                                                      for s, m in models.items() if s != tgt])]
                for (mode, k), v in res.items():
                    rows.append((montage, tgt, r["subj"], mode, k, float(np.mean(v))))
        print(f"    [B] {montage}: done ({time.time() - t0:.0f} s)", flush=True)

    out("[B] Unlabeled target trials used for the reference, TS-LR, scored on the other trials of the subject")
    out("    k = 0: the target is not recentered | all: every trial, as in the main analysis (from step 5)")
    for montage in S5.MONTAGES:
        sel = [r for r in rows if r[0] == montage]
        if not sel:
            continue
        subs = sorted({(r[1], r[2]) for r in sel})
        full = np.array([transfer[(montage, t, s, "tn", "ra_ps", "lr")] for t, s in subs])
        zero = {(r[1], r[2]): r[5] for r in sel if r[3] == "random" and r[4] == 0}
        base = np.array([zero[k] for k in subs])
        out(f"    {montage}: random draws, all {len(subs)} target subjects")
        out(f"      {'k':>5}{'subjects':>10}{'accuracy':>10}{'vs k = 0  [95% CI]  p':>40}{'share of full gain':>21}")
        out(f"      {0:>5}{len(subs):>10}{100 * base.mean():>10.2f}")
        for k in K_LIST:
            d = {(r[1], r[2]): r[5] for r in sel if r[3] == "random" and r[4] == k}
            keys = [s for s in subs if s in d]
            if len(keys) < 3:
                continue
            a = np.array([d[s] for s in keys])
            b = np.array([zero[s] for s in keys])
            f = np.array([transfer[(montage, t, s, "tn", "ra_ps", "lr")] for t, s in keys])
            share = (a.mean() - b.mean()) / (f.mean() - b.mean()) if f.mean() != b.mean() else float("nan")
            out(f"      {k:>5}{len(keys):>10}{100 * a.mean():>10.2f}   {S5.fmt(S5.paired(a, b)):<42}{100 * share:>12.0f}%")
        out(f"      {'all':>5}{len(subs):>10}{100 * full.mean():>10.2f}   {S5.fmt(S5.paired(full, base))}")
        ordered = sorted({r[1] for r in sel if r[3] == "first"})
        if ordered:
            osubs = [s for s in subs if s[0] in ordered]
            out(f"    {montage}: first k trials in recording order, targets {', '.join(ordered)} ({len(osubs)} subjects);"
                " random draws on the same subjects for comparison")
            out(f"      {'k':>5}{'subjects':>10}{'first k':>10}{'random k':>10}")
            for k in K_LIST:
                fk = {(r[1], r[2]): r[5] for r in sel if r[3] == "first" and r[4] == k}
                rk = {(r[1], r[2]): r[5] for r in sel if r[3] == "random" and r[4] == k}
                keys = [s for s in osubs if s in fk and s in rk]
                if len(keys) < 3:
                    continue
                out(f"      {k:>5}{len(keys):>10}{100 * np.mean([fk[s] for s in keys]):>10.2f}"
                    f"{100 * np.mean([rk[s] for s in keys]):>10.2f}")
    out()
    with open(os.path.join(OUTDIR, "calibration_curve.csv"), "w", encoding="utf-8") as f:
        f.write("montage,target,subject,mode,k,acc\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.6f}\n")


# =========================== [C] sensitivity ===========================
def section_c(main, everything, libs):
    n_main, n_all = len(main.get("PhysioNet", [])), len(everything.get("PhysioNet", []))
    out(f"[C] Sensitivity to the PhysioNet exclusions: {n_main} subjects (main analysis) versus all {n_all} on disk")
    if n_all == n_main:
        out("    nothing was excluded; section skipped")
        out()
        return
    saved = (S5.COVS, S5.ALIGNS, S5.CLFS)
    S5.COVS, S5.ALIGNS, S5.CLFS = ["tn"], ["none", "ra_ps"], ["lr"]
    try:
        for montage in S5.MONTAGES:
            line = []
            for label, data in (("main", main), ("all", everything)):
                acc, names = S5.run_montage(data, montage, libs)
                v = S5.View(acc, data, names)
                a, b = v.all_targets("tn", "ra_ps", "lr"), v.all_targets("tn", "none", "lr")
                d, lo, hi, dz, p = S5.paired(a, b)
                line.append(f"{label}: n={len(a)}, tn ra_ps {100 * a.mean():.2f}, gain {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            out(f"    {montage}  " + "  |  ".join(line))
    finally:
        S5.COVS, S5.ALIGNS, S5.CLFS = saved
    out()


def run():
    t_start = time.time()
    libs = S5._libs()
    csv = os.path.join(OUTDIR, "benchmark_subjects.csv")
    if not os.path.isdir(S5.CACHE) or not os.path.exists(csv):
        print(f"{S5.CACHE} or {csv} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 6 references  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    out(f"recipe: {S5.RECIPE}")
    main, everything = load_cache()
    transfer = read_transfer(csv)
    out("subjects in the main analysis: " + ", ".join(f"{d} {len(main[d])}" for d in S5.DATASETS)
        + f" (total {sum(len(v) for v in main.values())})")
    missing = [(m, d, r["subj"]) for m, members in S5.MONTAGES.items() for d in members
               for r in main.get(d, []) if (m, d, r["subj"], "tn", "ra_ps", "lr") not in transfer]
    if missing:
        out(f"<-- {len(missing)} subjects of the cache are not in {csv} (first: {missing[0]}); "
            "run step5_benchmark.py again before this script")
        return 1
    out()
    section_a(main, transfer, libs)
    section_b(main, transfer, libs)
    section_c(main, everything, libs)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step6_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    print(f"written: {os.path.abspath(OUTDIR)} (step6_summary.txt, references_subjects.csv, calibration_curve.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
