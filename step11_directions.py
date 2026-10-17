"""
step11_directions.py

One script for all candidate directions, ending with one scoreboard.

Run from the Codes folder (steps 5, 6, 8 and 9 must have been run; the file
step10_headroom.py must be in the folder, it does not have to be run first):

    python step11_directions.py             everything (the neural part needs PyTorch)
    python step11_directions.py --no-deep   without the neural part
    python step11_directions.py --redo      recompute parts whose result files already exist

A part whose result file already exists is skipped, so the script can be
restarted after an interruption. Expect 45 to 75 minutes for a first full run.

Parts
  [P] [Q] [R] [S] [T]  the headroom tests of step 10 (dynamic recentering, value of
        labels and self-training, choice of source, label-free accuracy estimate,
        rest-referenced alignment)
  [P2] controls for dynamic recentering: is the gain of a short memory local in
        time, or does it come from the balanced order of the cues?
  [L] the recentering law  acc(k) = Phi( a / sqrt(1 + (1 + a^2)/k) ),  a = Phi^-1(acc with all trials):
        L1 prediction of the k-trial curve of step 6;  L2 prediction for a reference
        with a fixed class composition;  L3 a reference balanced with pseudo-labels,
        the estimator that the law suggests;  L4 prediction of pseudo-online accuracy
  [U] how far the covariance framework goes: filter bank (8-13, 13-30, 8-30 Hz),
        all other datasets pooled, CSP + LDA
  [V] selective decoding: accuracy on the most confident trials
  [D] EEGNet trained on the source with and without Euclidean alignment, and its
        k-trial curve against the law (GPU if PyTorch sees one)
  [Z] scoreboard over all parts

Outputs in corrected_results/: step11_summary.txt, step11_*.csv (and the step10
files), and a refreshed results_bundle.zip.
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import datetime

import numpy as np

try:
    import step5_benchmark as S5
    import step8_new_analyses as S8
    import step9_reviewer_checks as S9
    import step10_headroom as S10
except Exception as _e:                                    # pragma: no cover
    print(f"could not import the earlier step scripts (5, 8, 9, 10); run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
out = S5.out
tn = S8.tn
K_CURVE = [1, 2, 4, 8, 16, 32]
K_BAL = [4, 8, 16]
COMPOSITIONS = [(8, 8), (12, 4), (4, 12), (16, 0), (0, 16)]
N_DRAW = 10
BANDS = {"mu": "bp8-13_win0.5-2.5", "beta": "bp13-30_win0.5-2.5"}
EPOCH_CACHE = os.path.join(OUTDIR, "cache_epochs_bp8-30_win0.5-2.5")
COVERAGE = [1.0, 0.75, 0.5, 0.25]


def have(name):
    return os.path.exists(os.path.join(OUTDIR, name))


def read_rows(name):
    with open(os.path.join(OUTDIR, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(rows, cols):
    return np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)


def law(acc_all, k, pi=None):
    """Predicted accuracy when the reference of a subject comes from k unlabeled trials.
    pi = None: k trials drawn at random.  pi given: a reference in which a fraction pi of the
    trials belongs to class 1 (balanced accuracy is predicted)."""
    from scipy.stats import norm
    a = norm.ppf(np.clip(np.asarray(acc_all, float), 1e-4, 1 - 1e-4))
    if pi is None:
        return norm.cdf(a / np.sqrt(1.0 + (1.0 + a ** 2) / k))
    q = np.sqrt(1.0 + 1.0 / k)
    return 0.5 * (norm.cdf(2.0 * (1.0 - pi) * a / q) + norm.cdf(2.0 * pi * a / q))


def bal_acc(pred, y):
    return 0.5 * (np.mean(pred[y == 1] == 1) + np.mean(pred[y == 0] == 0))


def rmean(mean_riemann, X):
    return X[0] if len(X) == 1 else mean_riemann(X)


# =========================== [L] the recentering law ===========================
def section_l1(bench):
    if not have("calibration_curve.csv"):
        out("[L1] skipped: calibration_curve.csv (step 6) not found")
        out()
        return
    rows = []
    obs = {}
    for r in read_rows("calibration_curve.csv"):
        if r["mode"] == "random":
            obs[(r["montage"], r["target"], int(r["subject"]), int(r["k"]))] = float(r["acc"])
    out("[L1] The law against the k-trial curve of step 6 (reference from k random trials, scored on the other trials)")
    out("     no parameter is fitted: the prediction for a subject uses only its accuracy with all trials")
    for montage in S5.MONTAGES:
        keys = sorted(bench[(montage, "ra_ps")])
        inf = np.array([bench[(montage, "ra_ps")][k] for k in keys])
        out(f"     {montage}: {len(keys)} subjects, accuracy with all trials {100 * inf.mean():.2f}")
        out(f"       {'k':>4}{'observed':>10}{'predicted':>11}{'difference':>12}{'r of the drops':>16}{'mean |error|':>14}")
        for k in K_CURVE:
            o = np.array([obs.get((montage, t, s, k), np.nan) for t, s in keys])
            p = law(inf, k)
            ok = np.isfinite(o)
            if ok.sum() < 3:
                continue
            r = np.corrcoef(inf[ok] - o[ok], inf[ok] - p[ok])[0, 1]
            out(f"       {k:>4}{100 * o[ok].mean():>10.2f}{100 * p[ok].mean():>11.2f}{100 * (o[ok].mean() - p[ok].mean()):>+12.2f}"
                f"{r:>16.2f}{100 * np.mean(np.abs(o[ok] - p[ok])):>14.2f}")
            rows += [(montage, t, s, k, float(oo), float(pp)) for (t, s), oo, pp in zip(keys, o, p) if np.isfinite(oo)]
    out()
    S9.write_csv("step11_law_curve.csv", "montage,target,subject,k,acc_observed,acc_predicted", rows)


def section_l2(mont, libs):
    mean_riemann = libs[2]
    rows = []
    for mi, M in enumerate(mont):
        t0 = time.time()
        for ti, tgt in enumerate(M.names):
            srcs = M.sources(tgt)
            for j, C in enumerate(M.C[tgt]):
                y = M.y[tgt][j]
                i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
                full_c = S8.apply_ref(M.R[tgt][j], C)
                full = float(np.mean([bal_acc(M.models[s].predict(full_c), y) for s in srcs]))
                rng = np.random.default_rng([12, mi, ti, int(M.subj[tgt][j])])
                for n1, n0 in COMPOSITIONS:
                    if len(i1) < n1 + 3 or len(i0) < n0 + 3:
                        continue
                    acc = 0.0
                    for _ in range(N_DRAW):
                        idx = np.r_[rng.choice(i1, n1, replace=False), rng.choice(i0, n0, replace=False)].astype(int)
                        rest = np.setdiff1d(np.arange(len(y)), idx)
                        Ct = S8.apply_ref(mean_riemann(C[idx]), C[rest])
                        acc += float(np.mean([bal_acc(M.models[s].predict(Ct), y[rest]) for s in srcs]))
                    rows.append((M.name, tgt, M.subj[tgt][j], n1, n0, acc / N_DRAW, full))
        print(f"    [L2] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[L2] The law for a reference with a fixed class composition (16 trials; balanced accuracy on the other trials)")
    out("     a reference made of one class only is the extreme case of label shift")
    for M in mont:
        sel = [r for r in rows if r[0] == M.name]
        out(f"     {M.name}")
        out(f"       {'reference':<22}{'subjects':>9}{'observed':>10}{'predicted':>11}{'difference':>12}")
        for lab, comps, pi in (("8 + 8 (balanced)", [(8, 8)], 0.5), ("12 + 4", [(12, 4), (4, 12)], 0.75),
                               ("16 + 0 (one class)", [(16, 0), (0, 16)], 1.0)):
            by = {}
            for r in sel:
                if (r[3], r[4]) in comps:
                    by.setdefault((r[1], r[2]), []).append((r[5], r[6]))
            v = np.array([np.mean(x, axis=0) for x in by.values() if len(x) == len(comps)])
            if len(v) < 3:
                continue
            p = law(v[:, 1], 16, pi)
            out(f"       {lab:<22}{len(v):>9}{100 * v[:, 0].mean():>10.2f}{100 * p.mean():>11.2f}{100 * (v[:, 0].mean() - p.mean()):>+12.2f}")
    out()
    S9.write_csv("step11_law_composition.csv", "montage,target,subject,n_class1,n_class0,balanced_acc,balanced_acc_all_trials", rows)


def section_l3(mont, libs):
    mean_riemann = libs[2]
    rows = []
    for mi, M in enumerate(mont):
        t0 = time.time()
        for ti, tgt in enumerate(M.names):
            srcs = M.sources(tgt)
            for j, C in enumerate(M.C[tgt]):
                y, n = M.y[tgt][j], len(C)
                rng = np.random.default_rng([13, mi, ti, int(M.subj[tgt][j])])
                for k in K_BAL:
                    if n - k < 6:
                        continue
                    acc = np.zeros(3)                        # plain, balanced with pseudo-labels, balanced with labels
                    for _ in range(N_DRAW):
                        idx = rng.choice(n, k, replace=False)
                        rest = np.setdiff1d(np.arange(n), idx)
                        R0 = mean_riemann(C[idx])
                        part = np.zeros(3)
                        lab = y[idx]
                        if 0 < lab.sum() < k:
                            Ro = S8.geodesic(rmean(mean_riemann, C[idx][lab == 0]), rmean(mean_riemann, C[idx][lab == 1]), 0.5)
                        else:
                            Ro = R0
                        for s in srcs:
                            m = M.models[s]
                            part[0] += np.mean(m.predict(S8.apply_ref(R0, C[rest])) == y[rest])
                            R = R0
                            for _it in range(2):
                                pl = m.predict(S8.apply_ref(R, C[idx]))
                                if 0 < pl.sum() < k:
                                    R = S8.geodesic(rmean(mean_riemann, C[idx][pl == 0]), rmean(mean_riemann, C[idx][pl == 1]), 0.5)
                            part[1] += np.mean(m.predict(S8.apply_ref(R, C[rest])) == y[rest])
                            part[2] += np.mean(m.predict(S8.apply_ref(Ro, C[rest])) == y[rest])
                        acc += part / len(srcs)
                    full = float(np.mean([np.mean(M.models[s].predict(S8.apply_ref(M.R[tgt][j], C)) == y) for s in srcs]))
                    rows.append((M.name, tgt, M.subj[tgt][j], k, *(acc / N_DRAW).tolist(), full))
        print(f"    [L3] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[L3] Reference balanced between the two classes, from k random trials (accuracy % on the other trials)")
    out("     plain: mean of the k trials | pseudo: midpoint of the two class means, classes guessed by the frozen model")
    out("     (no label) | labels: the same with the true classes of the k trials (what balancing can give at most)")
    for M in mont:
        sel = [r for r in rows if r[0] == M.name]
        full = np.array([r[7] for r in sel if r[3] == K_BAL[0]])
        top = np.quantile(full, 2 / 3) if len(full) else 1.0
        out(f"     {M.name}")
        out(f"       {'k':>4}{'subjects':>10}{'plain':>8}{'pseudo':>8}{'labels':>8}   pseudo - plain  [95% CI]  dz  p"
            f"{'':<16}best third: plain / pseudo / labels")
        for k in K_BAL:
            x = np.array([r[4:8] for r in sel if r[3] == k])
            if len(x) < 3:
                continue
            t3 = x[:, 3] >= top
            out(f"       {k:>4}{len(x):>10}{100 * x[:, 0].mean():>8.2f}{100 * x[:, 1].mean():>8.2f}{100 * x[:, 2].mean():>8.2f}"
                f"   {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}   {100 * x[t3, 0].mean():6.2f} / {100 * x[t3, 1].mean():.2f} / {100 * x[t3, 2].mean():.2f}")
    out()
    S9.write_csv("step11_balanced.csv", "montage,target,subject,k,acc_plain,acc_pseudo_balanced,acc_label_balanced,acc_all_trials", rows)


def section_l4():
    if not (have("step9_online.csv") and have("step8_levels.csv")):
        out("[L4] skipped: step9_online.csv or step8_levels.csv not found")
        out()
        return
    lev = {(r["montage"], r["target"], int(r["subject"])): (float(r["acc_none"]), float(r["acc_own"])) for r in read_rows("step8_levels.csv")}
    onl = read_rows("step9_online.csv")
    n_all = {(r["montage"], r["target"], int(r["subject"])): int(r["n_trials"]) for r in onl if r["trial_block"] == "all"}
    blocks = [("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16), ("17-32", 17, 32), ("33+", 33, 10 ** 9), ("all", 1, 10 ** 9)]
    rows = []
    out("[L4] The law against pseudo-online use (running mean of the earlier trials; step 9)")
    out("     predicted: trial 1 at the level without recentering, trial t at the law with k = t - 1")
    out("     observed above predicted: recent trials make a better reference than random ones, or the balanced order")
    out("     of the cues helps a causal reference (part P2 separates the two)")
    for montage in S5.MONTAGES:
        out(f"     {montage}")
        out(f"       {'trials':>8}{'subjects':>10}{'observed':>10}{'predicted':>11}{'difference':>12}{'offline':>9}")
        for lab, lo, hi in blocks:
            o, p, f = [], [], []
            for r in onl:
                key = (r["montage"], r["target"], int(r["subject"]))
                if r["montage"] != montage or r["trial_block"] != lab or key not in lev or key not in n_all:
                    continue
                none, own = lev[key]
                t = np.arange(lo, min(hi, n_all[key]) + 1)
                if len(t) == 0:
                    continue
                pred = np.where(t == 1, none, law(own, np.maximum(t - 1, 1)))
                o.append(float(r["acc_plain"])); p.append(float(pred.mean())); f.append(float(r["acc_offline"]))
                rows.append((montage, r["target"], int(r["subject"]), lab, o[-1], p[-1]))
            if len(o) >= 3:
                out(f"       {lab:>8}{len(o):>10}{100 * np.mean(o):>10.2f}{100 * np.mean(p):>11.2f}{100 * (np.mean(o) - np.mean(p)):>+12.2f}{100 * np.mean(f):>9.2f}")
    out()
    S9.write_csv("step11_law_online.csv", "montage,target,subject,trial_block,acc_observed,acc_predicted", rows)


# =========================== [P2] controls for dynamic recentering ===========================
HL_CTRL = [2, 4, None]


def _prequential(M, tgt, C, y, half_lives):
    """Accuracy over the recording when trial t is recentered on the (forgetting) mean of trials before t."""
    srcs, n, res = M.sources(tgt), len(C), []
    for h in half_lives:
        a = 0.0 if h is None else 1.0 - 2.0 ** (-1.0 / h)
        X = np.empty_like(C)
        X[0] = C[0]
        R = C[0]
        for t in range(1, n):
            if t > 1:
                R = S8.geodesic(R, C[t - 1], max(1.0 / t, a))
            X[t] = S8.apply_ref(R, C[t:t + 1])[0]
        res.append(float(np.mean([np.mean(M.models[s].predict(X) == y) for s in srcs])))
    return res


def section_p2(mont, libs, main, mods):
    mean_riemann = libs[2]
    rows = []
    for mi, M in enumerate(mont):
        t0 = time.time()
        for ti, tgt in enumerate([d for d in M.names if d in S10.ORDERED]):
            for j, C in enumerate(M.C[tgt]):
                y, n = M.y[tgt][j], len(C)
                rng = np.random.default_rng([15, mi, ti, int(M.subj[tgt][j])])
                rec = _prequential(M, tgt, C, y, HL_CTRL)
                within, rand = np.zeros(len(HL_CTRL)), np.zeros(len(HL_CTRL))
                for _ in range(5):
                    perm = np.arange(n)
                    for c in (0, 1):                      # same label sequence, trials exchanged inside each class
                        idx = np.where(y == c)[0]
                        perm[idx] = rng.permutation(idx)
                    within += np.array(_prequential(M, tgt, C[perm], y, HL_CTRL)) / 5
                    q = rng.permutation(n)                # trials and labels in random order
                    rand += np.array(_prequential(M, tgt, C[q], y[q], HL_CTRL)) / 5
                rows.append((M.name, tgt, M.subj[tgt][j], *rec, *within.tolist(), *rand.tolist()))
        print(f"    [P2] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    labels = [f"half-life {h}" if h else "no forgetting" for h in HL_CTRL]
    out("[P2] Is the gain of a short memory real? Pseudo-online accuracy % with the trials in three orders")
    out("     recorded: as recorded | within class: the sequence of labels is kept and the trials are exchanged inside")
    out("     each class, which removes what is local in time and keeps what the label sequence gives | random: both removed")
    out("     A gain that survives 'within class' comes from the balanced order of the cues, not from tracking the signal.")
    nh = len(HL_CTRL)
    for M in mont:
        x = S10.by_subject(rows, M.name, range(3, 3 + 3 * nh))
        if len(x) == 0:
            continue
        out(f"     {M.name}: {len(x)} subjects")
        out(f"       {'order':<14}" + "".join(f"{lab:>15}" for lab in labels) + "   half-life 4 against no forgetting  [95% CI]  dz  p")
        for oi, name in enumerate(("recorded", "within class", "random")):
            v = x[:, oi * nh:(oi + 1) * nh]
            out(f"       {name:<14}" + "".join(f"{100 * m:>15.2f}" for m in v.mean(0)) + "   " + S5.fmt(S5.paired(v[:, 1], v[:, 2])))
        loc = (x[:, 1] - x[:, 2]) - (x[:, nh + 1] - x[:, nh + 2])
        out(f"       gain in recorded order minus gain within class (what is local in time)   "
            + S5.fmt(S5.paired(loc + 0.5, np.full(len(loc), 0.5))))
    out()
    S9.write_csv("step11_dynamic_controls.csv", "montage,target,subject," + ",".join(
        f"acc_{o}_{'hl' + str(h) if h else 'none'}" for o in ("recorded", "withinclass", "random") for h in HL_CTRL), rows)

    # a reference that contains no class information at all: the pre-cue baselines of part T
    cache = os.path.join(OUTDIR, "cache_states_pre2s")
    if mods is None and not os.path.isdir(cache):
        out("[P2] rest reference with forgetting: skipped (no cache of part T and no project modules)")
        out()
        return
    loaders = S9.make_loaders((8.0, 30.0), (-2.0, 2.5))
    data = {ds: [dict(subj=r["subj"], **S10.get_states(ds, r["subj"], mods, libs, loaders, cache)) for r in main[ds]]
            for ds in S5.DATASETS}
    rows2 = []
    for montage, kr, kt in (("3ch", "R3", "T3"), ("17ch", "R17", "T17")):
        names = [d for d in S5.MONTAGES[montage] if len(data[d]) >= 3]
        T = {d: [tn(r[kt]) for r in data[d]] for d in names}
        B = {d: [tn(r[kr]) for r in data[d]] for d in names}
        models = {}
        for s in names:
            Cs = np.concatenate([S8.apply_ref(mean_riemann(b), c) for b, c in zip(B[s], T[s])])
            models[s] = S5.Model(Cs, np.concatenate([r["y"] for r in data[s]]), "lr", libs)
        for tgt in [d for d in names if d in S10.ORDERED]:
            for j, r in enumerate(data[tgt]):
                acc = []
                for h in HL_CTRL:
                    a = 0.0 if h is None else 1.0 - 2.0 ** (-1.0 / h)
                    X = np.empty_like(T[tgt][j])
                    R = B[tgt][j][0]
                    for t in range(len(X)):
                        if t > 0:                          # baselines of trials 0 .. t (the baseline precedes the trial)
                            R = S8.geodesic(R, B[tgt][j][t], max(1.0 / (t + 1), a))
                        X[t] = S8.apply_ref(R, T[tgt][j][t:t + 1])[0]
                    acc.append(float(np.mean([np.mean(models[s].predict(X) == r["y"]) for s in names if s != tgt])))
                rows2.append((montage, tgt, r["subj"], *acc))
    out("[P2] Rest reference with forgetting (baselines before the cues carry no class information)")
    for montage in S5.MONTAGES:
        x = S10.by_subject(rows2, montage, range(3, 3 + nh))
        if len(x) == 0:
            continue
        out(f"     {montage}: {len(x)} subjects   " + " | ".join(f"{lab} {100 * m:.2f}" for lab, m in zip(labels, x.mean(0))))
        out(f"       half-life 4 against no forgetting   {S5.fmt(S5.paired(x[:, 1], x[:, 2]))}")
    out()
    S9.write_csv("step11_dynamic_rest.csv", "montage,target,subject," + ",".join(
        f"acc_rest_{'hl' + str(h) if h else 'none'}" for h in HL_CTRL), rows2)


# =========================== [U] covariance stack ===========================
def load_band(tag, main):
    path = os.path.join(OUTDIR, "cache_sens_" + tag)
    if not os.path.isdir(path):
        raise RuntimeError(f"{path} not found (it is written by part N of step 9)")
    data = {}
    for ds in S5.DATASETS:
        recs = []
        for r in main[ds]:
            z = np.load(os.path.join(path, f"{ds}_{r['subj']:03d}.npz"), allow_pickle=False)
            if len(z["y"]) != len(r["y"]) or not np.array_equal(z["y"], r["y"]):
                raise RuntimeError(f"{ds} subject {r['subj']}: trials of band {tag} do not match the main cache")
            recs.append(dict(subj=r["subj"], y=z["y"], C3=z["C3"], C17=(z["C17"] if z["C17"].ndim == 3 else None)))
        data[ds] = recs
    return data


def section_u(mont, libs, main):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    mean_riemann, tangent, LogReg = libs[2], libs[3], libs[4]
    band_data = {"full": main}
    for b, tag in BANDS.items():
        band_data[b] = load_band(tag, main)
    rows = []
    for M in mont:
        t0 = time.time()
        key = "C3" if M.name == "3ch" else "C17"
        n_pairs = 1 if M.name == "3ch" else 3
        F = {}                                                # recentered covariances per band, dataset, subject
        for b, data in band_data.items():
            F[b] = {}
            for d in M.names:
                F[b][d] = []
                for r in data[d]:
                    c = tn(r[key])
                    F[b][d].append(S8.apply_ref(mean_riemann(c), c))
        Y = {d: M.y[d] for d in M.names}

        def fit_ts(src_list, bands):
            ys = np.concatenate([yy for s in src_list for yy in Y[s]])
            refs, Z = {}, []
            for b in bands:
                Cs = np.concatenate([c for s in src_list for c in F[b][s]])
                refs[b] = mean_riemann(Cs)
                Z.append(tangent(Cs, refs[b]))
            clf = LogReg(max_iter=2000).fit(np.hstack(Z), ys)
            return lambda d, j: clf.predict(np.hstack([tangent(F[b][d][j], refs[b]) for b in bands]))

        def fit_csp(src_list, bands):
            ys = np.concatenate([yy for s in src_list for yy in Y[s]])
            filt = {b: S9.CspLda(np.concatenate([c for s in src_list for c in F[b][s]]), ys, n_pairs) for b in bands}
            feats = np.hstack([filt[b]._features(np.concatenate([c for s in src_list for c in F[b][s]])) for b in bands])
            lda = LinearDiscriminantAnalysis().fit(feats, ys)
            return lambda d, j: lda.predict(np.hstack([filt[b]._features(F[b][d][j]) for b in bands]))

        fb = ("mu", "beta", "full")
        for tgt in M.names:
            srcs = M.sources(tgt)
            single_fb = [fit_ts([s], fb) for s in srcs]
            pooled = {"ts_full": fit_ts(srcs, ("full",)), "ts_fb": fit_ts(srcs, fb),
                      "csp_full": fit_csp(srcs, ("full",)), "csp_fb": fit_csp(srcs, fb)}
            for j in range(len(F["full"][tgt])):
                y = Y[tgt][j]
                prim = float(np.mean([np.mean(M.models[s].predict(F["full"][tgt][j]) == y) for s in srcs]))
                a_fb = float(np.mean([np.mean(f(tgt, j) == y) for f in single_fb]))
                rows.append((M.name, tgt, M.subj[tgt][j], prim, float(np.mean(pooled["ts_full"](tgt, j) == y)), a_fb,
                             float(np.mean(pooled["ts_fb"](tgt, j) == y)), float(np.mean(pooled["csp_full"](tgt, j) == y)),
                             float(np.mean(pooled["csp_fb"](tgt, j) == y))))
        print(f"    [U] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    names = ["primary (TS-LR, 8-30 Hz, one source at a time)", "TS-LR, 8-30 Hz, other datasets pooled",
             "TS-LR, filter bank, one source at a time", "TS-LR, filter bank, pooled",
             "CSP + LDA, 8-30 Hz, pooled", "CSP + LDA, filter bank, pooled"]
    out("[U] How far the covariance framework goes (every subject recentered in every band; no target label)")
    out("    filter bank: 8-13, 13-30 and 8-30 Hz, features concatenated")
    for M in mont:
        x = S10.by_subject(rows, M.name, range(3, 9))
        out(f"    {M.name}: {len(x)} subjects")
        for i, nm in enumerate(names):
            tail = "" if i == 0 else "   " + S5.fmt(S5.paired(x[:, i], x[:, 0]))
            out(f"      {nm:<48}{100 * x[:, i].mean():>7.2f}{tail}")
    out()
    S9.write_csv("step11_stack.csv", "montage,target,subject,acc_primary,acc_ts_pooled,acc_ts_fb,acc_ts_fb_pooled,"
                 "acc_csp_pooled,acc_csp_fb_pooled", rows)


# =========================== [V] selective decoding ===========================
def section_v(mont):
    rows = []
    for M in mont:
        for tgt in M.names:
            for j, C in enumerate(M.C[tgt]):
                y = M.y[tgt][j]
                Ct = S8.apply_ref(M.R[tgt][j], C)
                acc = np.zeros(len(COVERAGE))
                for s in M.sources(tgt):
                    w, b = S10.weights(M, s)
                    d = S10.vectors(M, s, Ct) @ w + b
                    order = np.argsort(-np.abs(d))
                    ok = ((d > 0) == (y == 1))[order]
                    acc += np.array([ok[: max(1, int(round(c * len(y))))].mean() for c in COVERAGE])
                rows.append((M.name, tgt, M.subj[tgt][j], *(acc / len(M.sources(tgt))).tolist()))
    out("[V] Selective decoding: accuracy % when only the most confident trials of a subject are decided (no label)")
    for M in mont:
        x = S10.by_subject(rows, M.name, range(3, 3 + len(COVERAGE)))
        top = x[:, 0] >= np.quantile(x[:, 0], 2 / 3)
        out(f"    {M.name}: {len(x)} subjects")
        out("      share of trials decided " + "".join(f"{int(100 * c):>8}%" for c in COVERAGE))
        out("      all subjects            " + "".join(f"{100 * v:>9.2f}" for v in x.mean(0)))
        out("      best third of subjects  " + "".join(f"{100 * v:>9.2f}" for v in x[top].mean(0)))
        out(f"      half of the trials against all trials   {S5.fmt(S5.paired(x[:, 2], x[:, 0]))}")
    out()
    S9.write_csv("step11_selective.csv", "montage,target,subject," + ",".join(f"acc_coverage_{int(100 * c)}" for c in COVERAGE), rows)


# =========================== [D] EEGNet with alignment ===========================
def get_epochs(ds, subj, mods, loaders):
    path = os.path.join(EPOCH_CACHE, f"{ds}_{subj:03d}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return z["X"], z["y"]
    X, y, _ = loaders[ds](subj, mods)
    X = np.asarray(X, dtype=np.float32)
    os.makedirs(EPOCH_CACHE, exist_ok=True)
    np.savez(path, X=X, y=np.asarray(y))
    return X, np.asarray(y)


def amp_norm(X):
    """Every trial scaled to unit mean channel variance (the counterpart of trace normalization)."""
    s = np.sqrt(X.var(axis=2).mean(axis=1))
    return (X / s[:, None, None]).astype(np.float32)


def ea_matrix(X):
    """Inverse square root of the arithmetic mean covariance of the trials in X."""
    R = np.einsum("nct,ndt->cd", X, X, dtype=np.float64) / (X.shape[0] * X.shape[2])
    return S8.invsqrtm(R)


def make_trainer(mock, epochs, device):
    """Returns train(X, y, seed) -> scorer(X) giving the logit difference class 1 minus class 0."""
    if mock:                                   # numpy stand-in used only to test the plumbing without PyTorch
        from sklearn.linear_model import LogisticRegression

        def train(X, y, seed):
            f = lambda Z: np.log(Z.var(axis=2))
            clf = LogisticRegression(max_iter=500).fit(f(X), y)
            return lambda Z: clf.decision_function(f(Z))
        return train
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self, C, T, F1=8, D=2, F2=16, k1=125, p=0.5):
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
            self.fc = nn.Linear(n, 2)

        def forward(self, x):
            return self.fc(self.b2(self.b1(x.unsqueeze(1))).flatten(1))

    def train(X, y, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = EEGNet(X.shape[1], X.shape[2]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
        lossf = nn.CrossEntropyLoss()
        n = len(yt)
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, 64):
                idx = perm[i:i + 64]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = lossf(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        net.eval()
        del Xt, yt

        def score(Z):
            res = []
            with torch.no_grad():
                for i in range(0, len(Z), 1024):
                    o = net(torch.as_tensor(Z[i:i + 1024], dtype=torch.float32, device=device))
                    res.append((o[:, 1] - o[:, 0]).float().cpu().numpy())
            return np.concatenate(res)
        return score
    return train


def section_d(mods, main, mock):
    if mock:
        device, epochs, seeds, montages = "mock", 1, [0], ["17ch", "3ch"]
    else:
        import torch
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        epochs, seeds = (60, [0, 1]) if cuda else (20, [0])
        montages = ["17ch", "3ch"] if cuda else ["17ch"]
        out(f"[D] PyTorch {torch.__version__}, device: {device}"
            + ("" if cuda else "  (no GPU visible to PyTorch: reduced run, 17 channels, one seed, 20 epochs)"))
    train = make_trainer(mock, epochs, device)
    loaders = S9.make_loaders((8.0, 30.0), (0.5, 2.5))
    t0 = time.time()
    X, Y = {}, {}
    for ds in S5.DATASETS:
        X[ds], Y[ds] = [], []
        for r in main[ds]:
            x, y = get_epochs(ds, r["subj"], mods, loaders)
            if len(y) != len(r["y"]) or not np.array_equal(y, r["y"]):
                raise RuntimeError(f"{ds} subject {r['subj']}: epochs do not match the covariance cache")
            X[ds].append(amp_norm(x))
            Y[ds].append(y)
        print(f"    [D] {ds}: epochs ready ({time.time() - t0:.0f} s)", flush=True)
    rows, curve = [], []
    for mi, montage in enumerate(montages):
        names = [d for d in S5.MONTAGES[montage] if len(X[d]) >= 3]
        pick = (lambda d, x: x) if montage == "17ch" else (lambda d, x: x if d == "2b" else x[:, S5.TRIPLET_IN_SM, :])
        Xm = {d: [np.ascontiguousarray(pick(d, x)) for x in X[d]] for d in names}
        Xe = {d: [np.einsum("cd,ndt->nct", ea_matrix(x), x).astype(np.float32) for x in Xm[d]] for d in names}
        acc, kc = {}, {}
        for s in names:
            ys = np.concatenate(Y[s])
            for seed in seeds:
                f_none = train(np.concatenate(Xm[s]), ys, seed)
                f_ea = train(np.concatenate(Xe[s]), ys, seed)
                for ti, t in enumerate(names):
                    if t == s:
                        continue
                    for j, y in enumerate(Y[t]):
                        key = (t, main[t][j]["subj"])
                        a = acc.setdefault(key, ([], []))
                        a[0].append(np.mean((f_none(Xm[t][j]) > 0) == (y == 1)))
                        a[1].append(np.mean((f_ea(Xe[t][j]) > 0) == (y == 1)))
                        rng = np.random.default_rng([14, mi, ti, int(key[1]), seed])
                        x, n = Xm[t][j], len(y)
                        for k in K_CURVE:
                            if n - k < 6:
                                continue
                            v = []
                            for _ in range(5):
                                idx = rng.choice(n, k, replace=False)
                                rest = np.setdiff1d(np.arange(n), idx)
                                z = np.einsum("cd,ndt->nct", ea_matrix(x[idx]), x[rest]).astype(np.float32)
                                v.append(np.mean((f_ea(z) > 0) == (y[rest] == 1)))
                            kc.setdefault(key + (k,), []).append(float(np.mean(v)))
                print(f"    [D] {montage}: source {s}, seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
        rows += [(montage, t, sub, float(np.mean(a[0])), float(np.mean(a[1]))) for (t, sub), a in sorted(acc.items())]
        curve += [(montage, t, sub, k, float(np.mean(v))) for (t, sub, k), v in sorted(kc.items())]
    bench = S9.read_benchmark()
    out("[D] EEGNet trained on the source dataset, applied to every target subject (8-30 Hz, 0.5-2.5 s, no target label)")
    out("    none: trials scaled to unit variance | EA: every subject whitened with its mean covariance (Euclidean alignment)")
    for montage in montages:
        sel = [r for r in rows if r[0] == montage]
        x = np.array([r[3:] for r in sel])
        ts = np.array([bench[(montage, "ra_ps")][(r[1], r[2])] for r in sel])
        ts0 = np.array([bench[(montage, "none")][(r[1], r[2])] for r in sel])
        out(f"    {montage}: {len(x)} subjects | EEGNet none {100 * x[:, 0].mean():.2f}, EA {100 * x[:, 1].mean():.2f}"
            f" | TS-LR none {100 * ts0.mean():.2f}, recentered {100 * ts.mean():.2f}")
        out(f"      EEGNet: EA - none                  {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}")
        out(f"      EEGNet with EA - TS-LR recentered  {S5.fmt(S5.paired(x[:, 1], ts))}")
        inf = {(r[1], r[2]): r[4] for r in sel}
        out(f"      {'k':>4}{'observed':>10}{'law':>8}{'difference':>12}   (reference from k random trials, EEGNet with EA)")
        for k in K_CURVE:
            c = [r for r in curve if r[0] == montage and r[3] == k]
            if len(c) < 3:
                continue
            o = np.array([r[4] for r in c])
            p = law(np.array([inf[(r[1], r[2])] for r in c]), k)
            out(f"      {k:>4}{100 * o.mean():>10.2f}{100 * p.mean():>8.2f}{100 * (o.mean() - p.mean()):>+12.2f}")
    out()
    S9.write_csv("step11_deep.csv", "montage,target,subject,acc_eegnet_none,acc_eegnet_ea", rows)
    S9.write_csv("step11_deep_curve.csv", "montage,target,subject,k,acc_eegnet_ea_k_trials", curve)


# =========================== [Z] scoreboard ===========================
def scoreboard(bench):
    lines = []

    def add(name, labels, montage, a, b, ref):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d, lo, hi, dz, p = S5.paired(a, b)
        lines.append((name, labels, montage, len(a), ref, 100 * b.mean(), 100 * a.mean(), d, lo, hi, p))

    def block(fn):
        try:
            fn()
        except Exception as e:                           # a missing file must not hide the other rows
            lines.append((f"(row group not available: {e!r})"[:70], "", "", 0, "", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

    def rows_of(name, montage):
        return [r for r in read_rows(name) if r["montage"] == montage]

    for montage in S5.MONTAGES:
        def dyn():
            r = rows_of("step10_dynamic.csv", montage)
            cols = [c for c in r[0] if c.startswith("acc_halflife_")]
            x = fl(r, cols + ["acc_no_forgetting"])
            i = int(np.argmax(x[:, :-1].mean(0)))
            add(f"Dynamic recentering, best {cols[i][4:].replace('_', ' ')}", "no", montage, x[:, i], x[:, -1], "no forgetting")

        def dyn_controls():
            x = fl(rows_of("step11_dynamic_controls.csv", montage), ["acc_recorded_hl4", "acc_recorded_none",
                                                                      "acc_withinclass_hl4", "acc_withinclass_none"])
            add("Short memory (half-life 4), recorded order", "no", montage, x[:, 0], x[:, 1], "no forgetting")
            add("  same, trials exchanged inside each class", "no", montage, x[:, 2], x[:, 3], "no forgetting")
            y = fl(rows_of("step11_dynamic_rest.csv", montage), ["acc_rest_hl4", "acc_rest_none"])
            add("Rest reference with short memory", "no", montage, y[:, 0], y[:, 1], "rest, all history")

        def states():
            x = fl(rows_of("step10_states.csv", montage), ["acc_task", "acc_rest", "acc_rest_running", "acc_per_trial"])
            add("Rest reference, one per subject", "no", montage, x[:, 1], x[:, 0], "task reference")
            add("Rest reference, causal running", "no", montage, x[:, 2], x[:, 0], "task reference")
            add("Every trial against its own baseline", "no", montage, x[:, 3], x[:, 0], "task reference")

        def selftrain():
            r = rows_of("step10_selftraining.csv", montage)
            cols = [c for c in r[0] if c.startswith("acc_self_")]
            x = fl(r, cols + ["acc_frozen"])
            i = int(np.argmax(x[:, :-1].mean(0)))
            add("Self-training on pseudo-labels, best setting", "no", montage, x[:, i], x[:, -1], "frozen model")

        def labels():
            r = [q for q in rows_of("step10_labels.csv", montage) if q["k_per_class"] == "16"]
            cols = [c for c in r[0] if c.startswith("acc_finetuned_")]
            x = fl(r, cols + ["acc_zero"])
            i = int(np.argmax(x[:, :-1].mean(0)))
            add("Fine-tuned with 16 true labels per class", "YES", montage, x[:, i], x[:, -1], "frozen model")

        def site():
            x = fl(rows_of("step8_levels.csv", montage), ["acc_site_prior", "acc_own"])
            add("Site prior only (no trial of the user)", "no", montage, x[:, 0], x[:, 1], "own trials")

        def sources():
            x = fl(rows_of("step10_sources.csv", montage), ["acc_mean_of_sources", "acc_pooled", "acc_chosen_by_confidence",
                                                             "acc_oracle_split", "acc_mean_same_halves"])
            add("Source chosen by confidence", "no", montage, x[:, 2], x[:, 0], "primary")
            add("Source chosen with labels (oracle)", "YES", montage, x[:, 3], x[:, 4], "primary")

        def balanced():
            for k in ("4", "8"):
                x = fl([q for q in rows_of("step11_balanced.csv", montage) if q["k"] == k], ["acc_plain", "acc_pseudo_balanced"])
                add(f"Balanced reference from pseudo-labels, k = {k}", "no", montage, x[:, 1], x[:, 0], f"plain mean, k = {k}")

        def stack():
            cols = ["acc_primary", "acc_ts_pooled", "acc_ts_fb", "acc_ts_fb_pooled", "acc_csp_pooled", "acc_csp_fb_pooled"]
            x = fl(rows_of("step11_stack.csv", montage), cols)
            add("Other datasets pooled (TS-LR)", "no", montage, x[:, 1], x[:, 0], "primary")
            i = 1 + int(np.argmax(x[:, 1:].mean(0)))
            add(f"Best covariance stack ({cols[i][4:]})", "no", montage, x[:, i], x[:, 0], "primary")

        def deep():
            r = rows_of("step11_deep.csv", montage)
            x = fl(r, ["acc_eegnet_none", "acc_eegnet_ea"])
            ts = np.array([bench[(montage, "ra_ps")][(q["target"], int(q["subject"]))] for q in r])
            add("EEGNet without alignment", "no", montage, x[:, 0], ts, "primary")
            add("EEGNet with Euclidean alignment", "no", montage, x[:, 1], ts, "primary")

        def selective():
            x = fl(rows_of("step11_selective.csv", montage), ["acc_coverage_100", "acc_coverage_50"])
            add("Most confident half of the trials only", "no", montage, x[:, 1], x[:, 0], "all trials")

        for fn in (dyn, dyn_controls, states, selftrain, labels, site, sources, balanced, stack, deep, selective):
            block(fn)
    out("[Z] Scoreboard. Accuracy % (mean over subjects); gain = result minus reference, in points, with 95% CI and p")
    out("    'best ...' rows pick the best of several settings after seeing the results, so they are optimistic.")
    out("    Rows marked * use labels of the target subject: they are ceilings, not methods.")
    for montage in S5.MONTAGES:
        prim = np.array(list(bench[(montage, "ra_ps")].values()))
        out(f"    {montage}: primary pipeline of step 5 = {100 * prim.mean():.2f} ({len(prim)} subjects)")
        out(f"      {'direction (* = uses target labels)':<47}{'n':>4}{'reference':>18}{'ref.':>7}{'result':>8}   gain [95% CI]  p")
        for name, lab, m, n, ref, b, a, d, lo, hi, p in lines:
            if m == montage:
                out(f"      {('* ' if lab == 'YES' else '') + name:<47}{n:>4}{ref:>18}{b:>7.2f}{a:>8.2f}   {d:+.2f} [{lo:+.2f}, {hi:+.2f}] {S5.ptxt(p)}")
        for name, lab, m, n, ref, b, a, d, lo, hi, p in lines:
            if m == "" and montage == "3ch":
                out(f"      {name}")
    out()


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-deep", action="store_true", help="skip the neural part")
    ap.add_argument("--redo", action="store_true", help="recompute parts whose result files exist")
    ap.add_argument("--mock-deep", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    t_start = time.time()
    libs = S5._libs()
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 11 directions  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    main = S8.load_cache()
    out("subjects: " + ", ".join(f"{d} {len(main[d])}" for d in S5.DATASETS))
    out()
    mont = [S8.Montage(main, m, libs) for m in S5.MONTAGES]
    bench = S9.read_benchmark()
    try:
        mods = S9.project_modules()
    except Exception as e:
        mods = None
        out(f"the project modules could not be imported ({e!r}): parts T and D are skipped")

    def part(label, files, fn, *a):
        if not args.redo and all(have(f) for f in files):
            out(f"[{label}] already done ({', '.join(files)}); use --redo to recompute")
            out()
            return
        S9.guard(label, fn, *a)

    part("P", ["step10_dynamic.csv"], S10.section_p, mont)
    part("Q", ["step10_labels.csv", "step10_selftraining.csv"], S10.section_q, mont, libs)
    part("R", ["step10_sources.csv"], S10.section_r, mont, libs)
    part("S", ["step10_estimate.csv"], S10.section_s, mont)
    if mods is not None:
        part("T", ["step10_states.csv"], S10.section_t, mods, libs, main)
    part("L1", ["step11_law_curve.csv"], section_l1, bench)
    part("L2", ["step11_law_composition.csv"], section_l2, mont, libs)
    part("L3", ["step11_balanced.csv"], section_l3, mont, libs)
    part("L4", ["step11_law_online.csv"], section_l4)
    part("P2", ["step11_dynamic_controls.csv", "step11_dynamic_rest.csv"], section_p2, mont, libs, main, mods)
    part("U", ["step11_stack.csv"], section_u, mont, libs, main)
    part("V", ["step11_selective.csv"], section_v, mont)
    if args.no_deep:
        out("[D] skipped (--no-deep)")
        out()
    elif mods is not None:
        ok = args.mock_deep
        if not ok:
            try:
                import torch  # noqa: F401
                ok = True
            except Exception as e:
                out(f"[D] skipped: PyTorch could not be imported in this environment ({e!r})")
                out()
        if ok:
            part("D", ["step11_deep.csv", "step11_deep_curve.csv"], section_d, mods, main, args.mock_deep)
    S9.guard("Z", scoreboard, bench)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step11_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    try:
        import step7_final_checks as S7
        names = S7.make_bundle()
        print(f"bundle refreshed: {os.path.abspath(S7.BUNDLE)} ({len(names)} files) -- attach this one file in the chat")
    except Exception as e:
        print(f"written: {os.path.abspath(OUTDIR)} (step11_summary.txt and CSV files); bundle not refreshed ({e!r})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
