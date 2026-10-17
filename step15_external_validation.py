"""
step15_external_validation.py

External validation on a fifth dataset that no earlier analysis has touched:
Dreyer et al. 2023 (Scientific Data 10:580), 87 participants, left- versus
right-hand motor imagery, 27 scalp electrodes, 512 Hz.

Run from the Codes folder, after steps 5, 10, 12 and 13 (about 25 to 45
minutes the first time; the recordings are read once and cached):

    python step15_external_validation.py
    python step15_external_validation.py "D:\\other\\folder\\BCI Database"     (if the data are elsewhere)

=====================================================================
PLAN, FIXED BEFORE THE DATA ARE ANALYSED (only folder names, channel
names and event codes have been looked at)
=====================================================================
Data.   All motor-imagery runs of every participant (R1, R2 acquisition; R3 to
        R6 online), in recording order. Cue codes 769 (left) and 770 (right).
        Same preprocessing as the benchmark: 17 sensorimotor channels, 8-30 Hz,
        250 Hz, task window 0.5-2.5 s after the cue, rest window the 2 s before
        the cue, reference as distributed, OAS covariance, trace normalization.
Exclusion. A participant is kept if at least four runs can be read and each
        class has at least 40 valid trials. Nothing else is excluded, and no
        exclusion is based on decoding performance.
Role.   The new dataset is a TARGET only. The frozen classifiers are those of
        the four benchmark datasets (tangent-space logistic regression, every
        source subject recentered). Nothing is fitted or tuned on the new data.

Hypotheses and criteria (both montages unless stated)
  H1  Recentering: accuracy with every participant recentered on its own
      unlabeled task trials is higher than without alignment
      (paired over participants, sign-flip p < 0.05).
  H2  Law: for a reference from k random trials (k = 1, 2, 4, 8, 16, 32) and
      for a reference of 16 trials with class composition 8+8, 12+4, 14+2,
      16+0, the mean accuracy over participants lies within 0.5 points of the
      prediction  Phi( a / sqrt(1 + (1 + a^2)/k) )  and of its fixed-composition
      form, a = Phi^-1(accuracy with all trials). No parameter is fitted.
  H3  Dual reference (geodesic midpoint of the task mean and the rest mean):
      (a) composition 8+8: not more than 0.5 points below the task reference;
      (b) composition 16+0: above the task reference, p < 0.05;
      (c) over the four compositions its largest shortfall from the best of
          the three references is smaller than that of the task reference and
          than that of the rest reference;
      (d) pseudo-online: in the blocked stream above the running task mean,
          p < 0.05; in the recorded stream not more than 0.5 points below it.
Every outcome is reported, passed or failed.
=====================================================================

Outputs in corrected_results/: step15_summary.txt, step15_*.csv, refreshed
results_bundle.zip. Cache: corrected_results/cache_dreyer_pre2s/.
"""
from __future__ import annotations

import os
import re
import sys
import glob
import time
import datetime
from collections import Counter

import numpy as np

try:
    import step5_benchmark as S5
    import step8_new_analyses as S8
    import step9_reviewer_checks as S9
    import step10_headroom as S10
    import step11_directions as S11
    import step12_rest_anchored as S12
    import step13_adaptive_reference as S13
except Exception as _e:                                    # pragma: no cover
    print(f"could not import the earlier step scripts (5, 8 to 13); run this from the Codes folder. ({_e!r})")
    sys.exit(1)

ROOT = r"E:\September Manuscript\Motor Imagery Transfer- JNE\BCI Database\BCI Database"
OUTDIR = S5.OUTDIR
CACHE = os.path.join(OUTDIR, "cache_dreyer_pre2s")
out = S5.out
tn = S8.tn
NAME = "Dreyer2023"
K_CURVE = [1, 2, 4, 8, 16, 32]
COMPOSITIONS = [(8, 8), (12, 4), (4, 12), (14, 2), (2, 14), (16, 0), (0, 16)]
GROUPS = (("8 + 8", [(8, 8)], 0.5), ("12 + 4", [(12, 4), (4, 12)], 0.75), ("14 + 2", [(14, 2), (2, 14)], 0.875),
          ("16 + 0", [(16, 0), (0, 16)], 1.0))
N_DRAW_K, N_DRAW_C = 10, 6
TOL = 0.5
VERDICT = []


def verdict(name, passed, detail):
    VERDICT.append((name, bool(passed), detail))


# =========================== loading ===========================
def list_subjects(root):
    res = []
    for grp in sorted(glob.glob(os.path.join(root, "Signals", "DATA *"))):
        for d in os.listdir(grp):
            m = re.match(r"^([A-Za-z]+)(\d+)$", d)
            if m and os.path.isdir(os.path.join(grp, d)):
                res.append((int(m.group(2)), d, os.path.join(grp, d)))
    if not res:                                 # recordings not at hand: use what has been cached
        for f in glob.glob(os.path.join(CACHE, "*.npz")):
            d = os.path.splitext(os.path.basename(f))[0]
            m = re.match(r"^([A-Za-z]+)(\d+)$", d)
            if m:
                res.append((int(m.group(2)), d, None))
    return sorted(res)


def load_subject(sid, folder, mods, libs):
    path = os.path.join(CACHE, f"{sid}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
    files = sorted(glob.glob(os.path.join(folder, f"{sid}_R*_*.gdf")),
                   key=lambda f: int(re.search(r"_R(\d+)_", os.path.basename(f)).group(1)))
    Xs, ys, runs, bad = [], [], [], []
    for f in files:
        r = int(re.search(r"_R(\d+)_", os.path.basename(f)).group(1))
        try:
            raw = mne.io.read_raw_gdf(f, preload=True, verbose="ERROR")
            X, y = CD._epoch(CD._prep(raw, SM, 8.0, 30.0, 250), {"769": 0, "770": 1}, -2.0, 2.5)
            del raw
        except Exception as e:
            bad.append(f"{os.path.basename(f)}: {e!r}")
            continue
        if X is None or X.shape[2] < 1126:
            bad.append(f"{os.path.basename(f)}: no usable epochs")
            continue
        ok = (np.isfinite(X).all(axis=(1, 2)) & (X[..., :500].var(axis=2).min(axis=1) > 0)
              & (X[..., 625:1126].var(axis=2).min(axis=1) > 0))
        Xs.append(X[ok])
        ys.append(np.asarray(y)[ok])
        runs.append(np.full(int(ok.sum()), r))
    if not Xs:
        rec = dict(y=np.zeros(0, int), run=np.zeros(0, int), n_runs=np.array(0), note=np.array("; ".join(bad)[:500]))
    else:
        X, y = np.concatenate(Xs), np.concatenate(ys)
        X3 = X[:, S5.TRIPLET_IN_SM, :]
        rec = dict(y=y, run=np.concatenate(runs), n_runs=np.array(len(Xs)), note=np.array("; ".join(bad)[:500]),
                   R3=S5._oas(X3[..., :500], libs[0]), T3=S5._oas(X3[..., 625:1126], libs[0]),
                   R17=S5._oas(X[..., :500], libs[0]), T17=S5._oas(X[..., 625:1126], libs[0]))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **rec)
    return rec


def load_all(root, mods, libs):
    subs = list_subjects(root)
    if not subs:
        raise RuntimeError(f"no participant folders under {os.path.join(root, 'Signals')}")
    kept, dropped, t0 = [], [], time.time()
    for i, (num, sid, folder) in enumerate(subs):
        rec = load_subject(sid, folder, mods, libs)
        y = rec["y"]
        n_runs = int(rec["n_runs"])
        if n_runs < 4 or min(int((y == 0).sum()), int((y == 1).sum())) < 40:
            dropped.append(f"{sid} ({n_runs} runs, {int((y == 0).sum())}/{int((y == 1).sum())} trials)")
        else:
            kept.append(dict(sid=sid, num=num, **rec))
        if (i + 1) % 10 == 0 or i + 1 == len(subs):
            print(f"    {NAME}: {i + 1}/{len(subs)} participants read ({time.time() - t0:.0f} s)", flush=True)
    out(f"[E0] {NAME}: {len(subs)} participant folders, {len(kept)} kept, {len(dropped)} excluded by the rule of the plan")
    if dropped:
        out("     excluded: " + "; ".join(dropped))
    notes = [f"{r['sid']}: {str(r['note'])}" for r in kept if str(r["note"])]
    if notes:
        out("     runs that could not be used: " + " | ".join(notes)[:600])
    n = np.array([len(r["y"]) for r in kept])
    nr = Counter(int(r["n_runs"]) for r in kept)
    out(f"     trials per participant: median {int(np.median(n))} (min {n.min()}, max {n.max()}); total {int(n.sum())}; "
        "runs per participant: " + ", ".join(f"{c} with {k} runs" for k, c in sorted(nr.items())))
    for txt in sorted(glob.glob(os.path.join(root, "Signals", "DATA *", "*", "*.txt"))):
        try:
            with open(txt, encoding="utf-8", errors="replace") as f:
                out(f"     note in the database ({os.path.relpath(txt, root)}): " + " ".join(f.read().split())[:300])
        except OSError:
            pass
    out()
    return kept


class Target:
    """Task and rest covariances of the new dataset for one montage."""

    def __init__(self, kept, montage, libs):
        kr, kt = ("R3", "T3") if montage == "3ch" else ("R17", "T17")
        self.T = [tn(r[kt]) for r in kept]
        self.B = [tn(r[kr]) for r in kept]
        self.y = [r["y"] for r in kept]
        self.sid = [r["sid"] for r in kept]
        self.Rt = [libs[2](c) for c in self.T]
        self.Rb = [libs[2](c) for c in self.B]


def lg(M, s, R, C):
    return S13.logits(M, s, R, C)


# =========================== [E1] main contrast ===========================
def section_e1(st, tg, libs):
    rows = []
    for M, G in zip(st, tg):
        none = {s: S5.Model(np.concatenate(M.T[s]), np.concatenate(M.y[s]), "lr", libs) for s in M.names}
        for j, T in enumerate(G.T):
            y = G.y[j]
            a0 = [float(np.mean(none[s].predict(T) == y)) for s in M.names]
            a1 = [float(np.mean((lg(M, s, G.Rt[j], T) > 0) == (y == 1))) for s in M.names]
            rows.append((M.name, G.sid[j], float(np.mean(a0)), float(np.mean(a1)), *a0, *a1))
    out(f"[E1] Transfer into {NAME} from the frozen models of the benchmark (TS-LR, accuracy %, no target label)")
    for M in st:
        x = np.array([r[2:] for r in rows if r[0] == M.name], dtype=float)
        ns = len(M.names)
        res = S5.paired(x[:, 1], x[:, 0])
        pc = S5.signflip_p(100 * (x[:, 1] - 0.5), two_sided=False)
        out(f"     {M.name}: {len(x)} participants | no alignment {100 * x[:, 0].mean():.2f} | recentered {100 * x[:, 1].mean():.2f}"
            f" | at or above 70%: {100 * np.mean(x[:, 1] >= 0.70):.0f}% of participants")
        out(f"       gain                 {S5.fmt(res)}     recentered against chance: {S5.ptxt(pc)}")
        for i, s in enumerate(M.names):
            out(f"       source {s:<10} none {100 * x[:, 2 + i].mean():6.2f} | recentered {100 * x[:, 2 + ns + i].mean():6.2f} | gain "
                f"{S5.fmt(S5.paired(x[:, 2 + ns + i], x[:, 2 + i]))}")
        verdict(f"H1 recentering gain, {M.name}", res[0] > 0 and res[4] < 0.05, f"gain {res[0]:+.2f} points, {S5.ptxt(res[4])}")
    out()
    S9.write_csv("step15_main.csv", "montage,participant,acc_none,acc_recentered", [r[:4] for r in rows])
    long = []
    for M in st:
        ns = len(M.names)
        for r in rows:
            if r[0] == M.name:
                long += [(r[0], r[1], s, r[4 + i], r[4 + ns + i]) for i, s in enumerate(M.names)]
    S9.write_csv("step15_main_by_source.csv", "montage,participant,source,acc_none,acc_recentered", long)
    return {(r[0], r[1]): r[3] for r in rows}


# =========================== [E2] the law: k random trials ===========================
def section_e2(st, tg, libs, acc_all):
    mean_riemann = libs[2]
    rows = []
    for mi, (M, G) in enumerate(zip(st, tg)):
        t0 = time.time()
        for j, T in enumerate(G.T):
            y, n = G.y[j] == 1, len(G.y[j])
            rng = np.random.default_rng([18, mi, j])
            for k in K_CURVE:
                acc = 0.0
                for _ in range(N_DRAW_K):
                    idx = rng.choice(n, k, replace=False)
                    rest = np.setdiff1d(np.arange(n), idx)
                    R = T[idx][0] if k == 1 else mean_riemann(T[idx])
                    acc += float(np.mean([np.mean((lg(M, s, R, T[rest]) > 0) == y[rest]) for s in M.names]))
                rows.append((M.name, G.sid[j], k, acc / N_DRAW_K, float(S11.law(acc_all[(M.name, G.sid[j])], k))))
        print(f"    [E2] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[E2] The law on the new dataset: reference from k random unlabeled trials, scored on the other trials")
    for M in st:
        sel = [r for r in rows if r[0] == M.name]
        out(f"     {M.name}")
        out(f"       {'k':>4}{'observed':>10}{'predicted':>11}{'difference':>12}")
        worst = 0.0
        for k in K_CURVE:
            x = np.array([r[3:] for r in sel if r[2] == k])
            d = 100 * (x[:, 0].mean() - x[:, 1].mean())
            worst = max(worst, abs(d))
            out(f"       {k:>4}{100 * x[:, 0].mean():>10.2f}{100 * x[:, 1].mean():>11.2f}{d:>+12.2f}")
        verdict(f"H2 law, k random trials, {M.name}", worst <= TOL, f"largest deviation {worst:.2f} points (limit {TOL})")
    out()
    S9.write_csv("step15_law_curve.csv", "montage,participant,k,acc_observed,acc_predicted", rows)


# =========================== [E3] class composition: law and dual reference ===========================
def section_e3(st, tg, libs):
    mean_riemann = libs[2]
    rows = []
    for mi, (M, G) in enumerate(zip(st, tg)):
        t0 = time.time()
        for j, T in enumerate(G.T):
            y, B = G.y[j], G.B[j]
            i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
            full = float(np.mean([S11.bal_acc((lg(M, s, G.Rt[j], T) > 0).astype(int), y) for s in M.names]))
            rng = np.random.default_rng([19, mi, j])
            for n1, n0 in COMPOSITIONS:
                acc = np.zeros(3)                                   # task, rest, dual
                for _ in range(N_DRAW_C):
                    idx = np.r_[rng.choice(i1, n1, replace=False), rng.choice(i0, n0, replace=False)].astype(int)
                    rest = np.setdiff1d(np.arange(len(y)), idx)
                    Rt, Rb = mean_riemann(T[idx]), mean_riemann(B[idx])
                    Rd = S8.geodesic(Rt, Rb, 0.5)
                    for s in M.names:
                        acc += np.array([S11.bal_acc((lg(M, s, R, T[rest]) > 0).astype(int), y[rest])
                                         for R in (Rt, Rb, Rd)]) / len(M.names)
                rows.append((M.name, G.sid[j], n1, n0, *(acc / N_DRAW_C).tolist(), full))
        print(f"    [E3] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[E3] Reference from 16 unlabeled trials with a fixed class composition (balanced accuracy % on the other trials)")
    out("     task: mean of the task trials | rest: mean of the baselines | dual: geodesic midpoint of the two")
    for M in st:
        sel = [r for r in rows if r[0] == M.name]
        top = np.quantile([r[7] for r in sel if (r[2], r[3]) == (8, 8)], 2 / 3)
        for title, keep in (("all participants", lambda r: True), ("best third", lambda r: r[7] >= top)):
            out(f"     {M.name}, {title}")
            out(f"       {'composition':<12}{'n':>5}{'task':>9}{'law':>9}{'rest':>9}{'dual':>9}   dual - task  [95% CI]  dz  p")
            worst, law_dev, stats = np.zeros(3), 0.0, {}
            for lab, comps, pi in GROUPS:
                by = {}
                for r in sel:
                    if (r[2], r[3]) in comps and keep(r):
                        by.setdefault(r[1], []).append(r[4:])
                v = np.array([np.mean(x, axis=0) for x in by.values() if len(x) == len(comps)])
                m = v[:, :3].mean(0)
                pred = float(S11.law(v[:, 3], 16, pi).mean())
                law_dev = max(law_dev, abs(100 * (m[0] - pred)))
                worst = np.maximum(worst, m.max() - m)
                stats[lab] = S5.paired(v[:, 2], v[:, 0])
                out(f"       {lab:<12}{len(v):>5}{100 * m[0]:>9.2f}{100 * pred:>9.2f}{100 * m[1]:>9.2f}{100 * m[2]:>9.2f}   {S5.fmt(stats[lab])}")
            out(f"       largest shortfall from the best reference: task {100 * worst[0]:.2f} | rest {100 * worst[1]:.2f} | dual {100 * worst[2]:.2f}")
            if title == "all participants":
                verdict(f"H2 law, class composition, {M.name}", law_dev <= TOL, f"largest deviation {law_dev:.2f} points (limit {TOL})")
                a, b = stats["8 + 8"], stats["16 + 0"]
                verdict(f"H3a dual reference, balanced, {M.name}", a[0] >= -TOL, f"dual - task {a[0]:+.2f} points (limit -{TOL})")
                verdict(f"H3b dual reference, one class, {M.name}", b[0] > 0 and b[4] < 0.05, f"dual - task {b[0]:+.2f} points, {S5.ptxt(b[4])}")
                verdict(f"H3c dual reference, worst case, {M.name}", worst[2] < worst[0] and worst[2] < worst[1],
                        f"shortfall dual {100 * worst[2]:.2f}, task {100 * worst[0]:.2f}, rest {100 * worst[1]:.2f}")
    out()
    S9.write_csv("step15_composition.csv", "montage,participant,n_class1,n_class0,bacc_task,bacc_rest,bacc_dual,bacc_all_trials", rows)


# =========================== [E4] pseudo-online streams ===========================
def _stream(M, G, j, order):
    T, B, y = G.T[j][order], G.B[j][order], G.y[j][order] == 1
    n, ok = len(y), np.zeros((3, len(y)))
    Rt, Rb = T[0], B[0]
    for t in range(n):
        if t > 0:
            Rb = S8.geodesic(Rb, B[t], 1.0 / (t + 1))
        if t > 1:
            Rt = S8.geodesic(Rt, T[t - 1], 1.0 / t)
        cur = T[t:t + 1]
        refs = (np.eye(T.shape[1]), Rb, Rb) if t == 0 else (Rt, Rb, S8.geodesic(Rt, Rb, 0.5))
        for s in M.names:
            ok[:, t] += np.array([float((lg(M, s, R, cur) > 0)[0] == y[t]) for R in refs]) / len(M.names)
    return ok.mean(1)


def section_e4(st, tg):
    rows = []
    for M, G in zip(st, tg):
        t0 = time.time()
        for j in range(len(G.T)):
            y = G.y[j]
            i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
            rec = _stream(M, G, j, np.arange(len(y)))
            blk = np.mean([_stream(M, G, j, np.r_[i1, i0]), _stream(M, G, j, np.r_[i0, i1])], axis=0)
            rows.append((M.name, G.sid[j], *rec.tolist(), *blk.tolist()))
        print(f"    [E4] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[E4] Pseudo-online streams on the new dataset (accuracy % over the stream; earlier trials only)")
    out("     recorded: runs and trials as recorded | blocked: all trials of one class, then all trials of the other")
    for M in st:
        x = np.array([r[2:] for r in rows if r[0] == M.name], dtype=float)
        out(f"     {M.name}")
        out(f"       {'stream':<10}{'n':>5}{'task':>9}{'rest':>9}{'dual':>9}   dual - task  [95% CI]  dz  p")
        res = {}
        for i, name in enumerate(("recorded", "blocked")):
            v = x[:, 3 * i:3 * i + 3]
            res[name] = S5.paired(v[:, 2], v[:, 0])
            out(f"       {name:<10}{len(v):>5}" + "".join(f"{100 * a:>9.2f}" for a in v.mean(0)) + "   " + S5.fmt(res[name]))
        r, b = res["recorded"], res["blocked"]
        verdict(f"H3d dual reference, streams, {M.name}", b[0] > 0 and b[4] < 0.05 and r[0] >= -TOL,
                f"blocked {b[0]:+.2f} points ({S5.ptxt(b[4])}); recorded {r[0]:+.2f} points (limit -{TOL})")
    out()
    S9.write_csv("step15_streams.csv", "montage,participant,acc_recorded_task,acc_recorded_rest,acc_recorded_dual,"
                 "acc_blocked_task,acc_blocked_rest,acc_blocked_dual", rows)


# =========================== [E5] reference inside the dataset (descriptive) ===========================
def section_e5(st, tg, libs, acc_all):
    rows = []
    for M, G in zip(st, tg):
        t0 = time.time()
        pooled = S5.Model(np.concatenate([S8.apply_ref(R, C) for s in M.names for R, C in zip(M.Rt[s], M.T[s])]),
                          np.concatenate([yy for s in M.names for yy in M.y[s]]), "lr", libs)
        for j, T in enumerate(G.T):
            y = G.y[j]
            rows.append((M.name, G.sid[j], S9.cv_accuracy(T, y, libs), acc_all[(M.name, G.sid[j])],
                         float(np.mean(pooled.predict(S8.apply_ref(G.Rt[j], T)) == y))))
        print(f"    [E5] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[E5] For context, not part of the plan: decoder trained on the participant's own labels, and pooled sources")
    for M in st:
        x = np.array([r[2:] for r in rows if r[0] == M.name], dtype=float)
        out(f"     {M.name}: within-subject 5-fold {100 * x[:, 0].mean():.2f} | transfer, one source at a time {100 * x[:, 1].mean():.2f}"
            f" | transfer, four benchmark datasets pooled {100 * x[:, 2].mean():.2f} | r(within, transfer) = "
            f"{np.corrcoef(x[:, 0], x[:, 1])[0, 1]:.2f}")
        out(f"       transfer - within-subject   {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}")
        out(f"       pooled - one source         {S5.fmt(S5.paired(x[:, 2], x[:, 1]))}")
    out()
    S9.write_csv("step15_within.csv", "montage,participant,acc_within_cv,acc_transfer,acc_transfer_pooled", rows)


def run():
    t_start = time.time()
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    libs = S5._libs()
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    if not os.path.isdir(root) and not os.path.isdir(CACHE):
        print(f"folder not found: {root}\ngive the folder of the database as an argument, in quotes")
        return 1
    S5._LINES.clear()
    out("Step 15 external validation  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    out("plan: see the header of this script (fixed before the new data were analysed)")
    out()
    mods = S9.project_modules()
    main = S8.load_cache()
    old = S12.load_states(main, libs)
    st = [S12.States(old, m, libs) for m in S5.MONTAGES]
    kept = load_all(root, mods, libs)
    tg = [Target(kept, m, libs) for m in S5.MONTAGES]
    acc_all = section_e1(st, tg, libs)
    S9.guard("E2", section_e2, st, tg, libs, acc_all)
    S9.guard("E3", section_e3, st, tg, libs)
    S9.guard("E4", section_e4, st, tg)
    S9.guard("E5", section_e5, st, tg, libs, acc_all)
    out("[E6] Outcome against the criteria of the plan")
    for name, passed, detail in VERDICT:
        out(f"     {'PASS' if passed else 'FAIL'}   {name:<46} {detail}")
    out(f"     {sum(p for _, p, _ in VERDICT)} of {len(VERDICT)} criteria met")
    out()
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step15_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    try:
        import step7_final_checks as S7
        names = S7.make_bundle()
        print(f"bundle refreshed: {os.path.abspath(S7.BUNDLE)} ({len(names)} files) -- attach this one file in the chat")
    except Exception as e:
        print(f"written: {os.path.abspath(OUTDIR)} (step15_summary.txt and CSV files); bundle not refreshed ({e!r})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
