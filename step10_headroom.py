"""
step10_headroom.py

Run from the Codes folder, after steps 5, 8 and 9 (about 15 to 25 minutes; part
[T] reads the recordings again):

    python step10_headroom.py

Before a new method is built, this script measures how much room each candidate
direction has on top of the primary pipeline of step 5 (trace-normalized OAS
covariance, every subject recentered on its own mean, tangent-space logistic
regression frozen on the source). Parts [P] to [S] use the cache only.

[P] Dynamic recentering. Pseudo-online reference from earlier trials with
    exponential forgetting (half-life 4, 8, 16, 32 trials) against no
    forgetting. Room for "dynamic recentering".
[Q] Value of target labels. k labeled trials per class of the target subject:
    threshold recalibration, fine-tuning of the source classifier (penalty on
    the distance to the source weights), and a classifier from those labels
    alone, scored on the other trials. No method that guesses labels
    (class-conditional alignment, confidence-guided adaptation) can be expected
    to beat true labels, so this is their ceiling. Self-training on pseudo-labels
    is run as the simplest such method.
[R] Choice of source. Pooling all other datasets, choosing the source by
    confidence (no labels), and an oracle that chooses the source per subject
    from labels on half of the trials. Room for "transferability-aware" decoding.
[S] Label-free estimate of accuracy. A two-component mixture fitted to the
    decision values of a subject's unlabeled trials against mean confidence.
[T] States. The reference is taken from the two seconds BEFORE each cue (rest)
    instead of from the task trials: one reference per subject, a causal running
    reference that exists from the first trial on, and the baseline of each trial
    on its own. Room for "multi-state" alignment.

Each part is independent. Outputs in corrected_results/: step10_summary.txt,
step10_*.csv and a refreshed results_bundle.zip.
"""
from __future__ import annotations

import os
import sys
import json
import time
import datetime

import numpy as np

try:
    import step5_benchmark as S5
    import step8_new_analyses as S8
    import step9_reviewer_checks as S9
except Exception as _e:                                    # pragma: no cover
    print(f"could not import the earlier step scripts; run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
out = S5.out
tn = S8.tn
ORDERED = ("2a", "2b", "PhysioNet")
HALF_LIVES = [4, 8, 16, 32, None]
K_LABELED = [2, 4, 8, 16]
LAMBDAS = [1.0, 10.0, 100.0]
N_DRAWS = 5
MIN_REST = 6
FS = 250


def by_subject(rows, montage, cols):
    """Rows of one montage as an array, in the order they were produced."""
    return np.array([[r[c] for c in cols] for r in rows if r[0] == montage], dtype=float)


def vectors(M, s, Ct):
    m = M.models[s]
    return m.ts(Ct, m.ref)


def weights(M, s):
    lr = M.models[s].m
    return lr.coef_.ravel().copy(), float(lr.intercept_[0])


# =========================== [P] dynamic recentering ===========================
def section_p(mont):
    rows = []
    for M in mont:
        t0 = time.time()
        for tgt in [d for d in M.names if d in ORDERED]:
            srcs = M.sources(tgt)
            for j, C in enumerate(M.C[tgt]):
                y, n = M.y[tgt][j], len(C)
                acc = []
                for h in HALF_LIVES:
                    a = 0.0 if h is None else 1.0 - 2.0 ** (-1.0 / h)
                    X = np.empty_like(C)
                    X[0] = C[0]
                    R = C[0]
                    for t in range(1, n):
                        if t > 1:                       # R: recursive mean of trials 0 .. t-1
                            R = S8.geodesic(R, C[t - 1], max(1.0 / t, a))
                        X[t] = S8.apply_ref(R, C[t:t + 1])[0]
                    acc.append(float(np.mean([np.mean(M.models[s].predict(X) == y) for s in srcs])))
                off = float(np.mean([np.mean(M.models[s].predict(S8.apply_ref(M.R[tgt][j], C)) == y) for s in srcs]))
                rows.append((M.name, tgt, M.subj[tgt][j], *acc, off))
        print(f"    [P] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[P] Dynamic recentering: pseudo-online reference from earlier trials with exponential forgetting")
    out("    accuracy % over the whole recording; 'none' = no forgetting (recursive mean of all earlier trials)")
    labels = [f"half-life {h}" if h else "none" for h in HALF_LIVES]
    for M in mont:
        x = by_subject(rows, M.name, range(3, 3 + len(HALF_LIVES) + 1))
        if len(x) == 0:
            continue
        tg = np.array([r[1] for r in rows if r[0] == M.name])
        out(f"    {M.name}: {len(x)} subjects")
        out(f"      {'forgetting':<16}" + "".join(f"{d:>11}" for d in sorted(set(tg), key=list(ORDERED).index))
            + f"{'all':>9}   against no forgetting  [95% CI]  dz  p")
        for i, lab in enumerate(labels):
            per = "".join(f"{100 * x[tg == d, i].mean():>11.2f}" for d in sorted(set(tg), key=list(ORDERED).index))
            tail = "" if HALF_LIVES[i] is None else "   " + S5.fmt(S5.paired(x[:, i], x[:, -2]))
            out(f"      {lab:<16}{per}{100 * x[:, i].mean():>9.2f}{tail}")
        out(f"      {'offline':<16}" + "".join(f"{100 * x[tg == d, -1].mean():>11.2f}"
                                               for d in sorted(set(tg), key=list(ORDERED).index))
            + f"{100 * x[:, -1].mean():>9.2f}")
    out()
    S9.write_csv("step10_dynamic.csv", "montage,target,subject," + ",".join(
        f"acc_halflife_{h}" if h else "acc_no_forgetting" for h in HALF_LIVES) + ",acc_offline", rows)


# =========================== [Q] value of labels ===========================
def finetune(z, s, w0, b0, lam):
    """Logistic regression on (z, s = +-1) with penalty lam/2 * ||w - w0||^2, started at the source solution."""
    from scipy.optimize import minimize
    from scipy.special import expit

    def f(p):
        w, b = p[:-1], p[-1]
        m = s * (z @ w + b)
        loss = np.logaddexp(0.0, -m).sum() + 0.5 * lam * np.sum((w - w0) ** 2)
        g = -s * expit(-m)
        return loss, np.r_[z.T @ g + lam * (w - w0), g.sum()]

    res = minimize(f, np.r_[w0, b0], jac=True, method="L-BFGS-B", options=dict(maxiter=200))
    return res.x[:-1], float(res.x[-1])


def section_q(mont, libs):
    LogReg = libs[4]
    rows_lab, rows_self = [], []
    for mi, M in enumerate(mont):
        t0 = time.time()
        for ti, tgt in enumerate(M.names):
            srcs = M.sources(tgt)
            for j, C in enumerate(M.C[tgt]):
                y, n = M.y[tgt][j], len(C)
                Ct = S8.apply_ref(M.R[tgt][j], C)
                Z = {s: vectors(M, s, Ct) for s in srcs}
                W = {s: weights(M, s) for s in srcs}
                L0 = {s: Z[s] @ W[s][0] + W[s][1] for s in srcs}
                sgn = 2.0 * y - 1.0
                rng = np.random.default_rng([10, mi, ti, int(M.subj[tgt][j])])
                i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
                for k in K_LABELED:
                    if min(len(i0), len(i1)) < k or n - 2 * k < MIN_REST:
                        continue
                    acc = np.zeros(3 + len(LAMBDAS))                # zero, threshold, scratch, fine-tuned ...
                    for _ in range(N_DRAWS):
                        lab = np.r_[rng.choice(i0, k, replace=False), rng.choice(i1, k, replace=False)]
                        rest = np.setdiff1d(np.arange(n), lab)
                        part = np.zeros_like(acc)
                        for s in srcs:
                            w0, b0 = W[s]
                            l0 = L0[s]
                            part[0] += np.mean((l0[rest] > 0) == (y[rest] == 1))
                            shift = -0.5 * (l0[lab][y[lab] == 1].mean() + l0[lab][y[lab] == 0].mean())
                            part[1] += np.mean((l0[rest] + shift > 0) == (y[rest] == 1))
                            sc = LogReg(max_iter=1000).fit(Z[s][lab], y[lab])
                            part[2] += np.mean(sc.predict(Z[s][rest]) == y[rest])
                            for li, lam in enumerate(LAMBDAS):
                                w, b = finetune(Z[s][lab], sgn[lab], w0, b0, lam)
                                part[3 + li] += np.mean((Z[s][rest] @ w + b > 0) == (y[rest] == 1))
                        acc += part / len(srcs)
                    rows_lab.append((M.name, tgt, M.subj[tgt][j], k, *(acc / N_DRAWS).tolist()))
                # self-training on pseudo-labels, all trials, no label
                base = float(np.mean([np.mean((L0[s] > 0) == (y == 1)) for s in srcs]))
                res = []
                for confident in (False, True):
                    for lam in LAMBDAS:
                        a = []
                        for s in srcs:
                            w, b = W[s]
                            w0, b0 = W[s]
                            for _ in range(2):
                                logit = Z[s] @ w + b
                                use = np.abs(logit) >= np.median(np.abs(logit)) if confident else np.ones(n, bool)
                                w, b = finetune(Z[s][use], np.where(logit[use] > 0, 1.0, -1.0), w0, b0, lam)
                            a.append(np.mean((Z[s] @ w + b > 0) == (y == 1)))
                        res.append(float(np.mean(a)))
                rows_self.append((M.name, tgt, M.subj[tgt][j], base, *res))
        print(f"    [Q] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[Q] Value of target labels: k labeled trials per class of the target subject, scored on its other trials")
    out("    zero: frozen source model, no label | threshold: intercept moved to the midpoint of the labeled class means |")
    out("    fine-tuned: source weights adapted with penalty lam * ||w - w_source||^2 / 2 | scratch: labeled trials only")
    for M in mont:
        sel = [r for r in rows_lab if r[0] == M.name]
        out(f"    {M.name}")
        out(f"      {'k':>4}{'subjects':>10}{'zero':>8}{'threshold':>11}{'scratch':>9}"
            + "".join(f"{'lam=' + format(l, 'g'):>10}" for l in LAMBDAS) + "   best fine-tuned - zero  [95% CI]  dz  p")
        for k in K_LABELED:
            x = np.array([r[4:] for r in sel if r[3] == k])
            if len(x) < 3:
                continue
            best = 3 + int(np.argmax(x[:, 3:].mean(0)))
            out(f"      {k:>4}{len(x):>10}" + "".join(f"{100 * v:>{w}.2f}" for v, w in zip(x.mean(0), [8, 11, 9] + [10] * len(LAMBDAS)))
                + "   " + S5.fmt(S5.paired(x[:, best], x[:, 0])))
        x = by_subject(rows_self, M.name, range(3, 4 + 2 * len(LAMBDAS)))
        out(f"      self-training on pseudo-labels (no label), all {len(x)} subjects: frozen model {100 * x[:, 0].mean():.2f}")
        for ci, name in enumerate(("all trials", "confident half")):
            for li, lam in enumerate(LAMBDAS):
                col = 1 + ci * len(LAMBDAS) + li
                out(f"        {name:<15} lam={lam:<6g} {100 * x[:, col].mean():6.2f}   {S5.fmt(S5.paired(x[:, col], x[:, 0]))}")
    out()
    S9.write_csv("step10_labels.csv", "montage,target,subject,k_per_class,acc_zero,acc_threshold,acc_scratch,"
                 + ",".join(f"acc_finetuned_lam{l:g}" for l in LAMBDAS), rows_lab)
    S9.write_csv("step10_selftraining.csv", "montage,target,subject,acc_frozen,"
                 + ",".join(f"acc_self_{n}_lam{l:g}" for n in ("all", "confident") for l in LAMBDAS), rows_self)


# =========================== [R] choice of source ===========================
def section_r(mont, libs):
    rows = []
    for mi, M in enumerate(mont):
        t0 = time.time()
        for ti, tgt in enumerate(M.names):
            srcs = M.sources(tgt)
            pooled = S5.Model(np.concatenate([S8.apply_ref(R, C) for s in srcs for R, C in zip(M.R[s], M.C[s])]),
                              np.concatenate([yy for s in srcs for yy in M.y[s]]), "lr", libs)
            per = []
            for j, C in enumerate(M.C[tgt]):
                y, n = M.y[tgt][j], len(C)
                Ct = S8.apply_ref(M.R[tgt][j], C)
                ok, conf = [], []
                for s in srcs:
                    p = M.models[s].m.predict_proba(vectors(M, s, Ct))[:, 1]
                    ok.append((p >= 0.5) == (y == 1))
                    conf.append(np.mean(np.maximum(p, 1 - p)))
                ok = np.array(ok, dtype=float)
                rng = np.random.default_rng([11, mi, ti, int(M.subj[tgt][j])])
                orc, base = [], []
                for _ in range(20):
                    perm = rng.permutation(n)
                    a, b = perm[: n // 2], perm[n // 2:]
                    best = int(np.argmax(ok[:, a].mean(1)))
                    orc.append(ok[best, b].mean())
                    base.append(ok[:, b].mean())
                per.append(ok.mean(1))
                rows.append([M.name, tgt, M.subj[tgt][j], float(ok.mean()), float(np.mean(pooled.predict(Ct) == y)),
                             float(ok[int(np.argmax(conf))].mean()), float(np.mean(orc)), float(np.mean(base)), 0.0])
            fixed = int(np.argmax(np.mean(per, axis=0)))               # best single source for this target dataset
            for r, p in zip(rows[-len(per):], per):
                r[8] = float(p[fixed])
        print(f"    [R] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    rows = [tuple(r) for r in rows]
    out("[R] Choice of source for a target subject (TS-LR, every subject recentered), mean accuracy %")
    out("    mean: average over single-source models (main analysis) | pooled: one model on all other datasets |")
    out("    by confidence: source with the highest mean confidence on the unlabeled trials | best fixed: best single")
    out("    source for the target dataset (chosen with labels) | oracle: source chosen per subject on half of the trials")
    out("    with labels, scored on the other half ('mean, same halves' is its baseline)")
    for M in mont:
        x = by_subject(rows, M.name, range(3, 9))
        out(f"    {M.name}: {len(x)} subjects")
        out(f"      mean {100 * x[:, 0].mean():.2f} | pooled {100 * x[:, 1].mean():.2f} | by confidence {100 * x[:, 2].mean():.2f}"
            f" | best fixed {100 * x[:, 5].mean():.2f} | oracle {100 * x[:, 3].mean():.2f} (mean, same halves {100 * x[:, 4].mean():.2f})")
        out(f"      pooled - mean          {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}")
        out(f"      by confidence - mean   {S5.fmt(S5.paired(x[:, 2], x[:, 0]))}")
        out(f"      best fixed - mean      {S5.fmt(S5.paired(x[:, 5], x[:, 0]))}")
        out(f"      oracle - mean          {S5.fmt(S5.paired(x[:, 3], x[:, 4]))}")
    out()
    S9.write_csv("step10_sources.csv", "montage,target,subject,acc_mean_of_sources,acc_pooled,acc_chosen_by_confidence,"
                 "acc_oracle_split,acc_mean_same_halves,acc_best_fixed_source", rows)


# =========================== [S] label-free estimate of accuracy ===========================
def mixture_accuracy(d, iters=200):
    """Two Gaussians with a common variance fitted to decision values; returns the accuracy implied
    by the fit if the upper component is the positive class."""
    from scipy.stats import norm
    d = np.asarray(d, float)
    med = np.median(d)
    lo, hi = d[d < med], d[d >= med]
    if len(lo) < 2 or len(hi) < 2 or d.std() == 0:
        return 0.5
    mu = np.array([lo.mean(), hi.mean()])
    sd = max(d.std() / 2.0, 1e-6)
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        r = pi * np.exp(-0.5 * ((d[:, None] - mu) / sd) ** 2) + 1e-300
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0)
        mu = (r * d[:, None]).sum(0) / nk
        sd = max(float(np.sqrt((r * (d[:, None] - mu) ** 2).sum() / len(d))), 1e-6)
        pi = nk / len(d)
    a, b = int(np.argmin(mu)), int(np.argmax(mu))
    return float(pi[b] * norm.cdf(mu[b] / sd) + pi[a] * norm.cdf(-mu[a] / sd))


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def section_s(mont):
    from scipy.stats import pearsonr, spearmanr
    rows = []
    for M in mont:
        for tgt in M.names:
            for j, C in enumerate(M.C[tgt]):
                y = M.y[tgt][j]
                Ct = S8.apply_ref(M.R[tgt][j], C)
                acc, conf, mix = [], [], []
                for s in M.sources(tgt):
                    w, b = weights(M, s)
                    d = vectors(M, s, Ct) @ w + b
                    p = 1.0 / (1.0 + np.exp(-np.clip(d, -50, 50)))
                    acc.append(np.mean((d > 0) == (y == 1)))
                    conf.append(np.mean(np.maximum(p, 1 - p)))
                    mix.append(mixture_accuracy(d))
                rows.append((M.name, tgt, M.subj[tgt][j], float(np.mean(acc)), float(np.mean(conf)), float(np.mean(mix)),
                             len(y)))
    out("[S] Label-free estimate of the accuracy of a subject from its unlabeled trials (averaged over sources)")
    out("    confidence: mean of max(p, 1 - p) | mixture: two-Gaussian fit to the decision values")
    for M in mont:
        x = by_subject(rows, M.name, (3, 4, 5, 6))
        acc = x[:, 0]
        out(f"    {M.name}: n = {len(x)}, mean accuracy {100 * acc.mean():.1f}%")
        for name, col in (("confidence", 1), ("mixture", 2)):
            e = x[:, col]
            out(f"      {name:<11} mean {100 * e.mean():5.1f}% | Pearson r {pearsonr(e, acc)[0]:+.2f} | Spearman {spearmanr(e, acc)[0]:+.2f}"
                f" | mean absolute error {100 * np.mean(np.abs(e - acc)):4.1f} | AUC (acc >= 70%) {S8._auc(e, acc >= 0.70):.2f}")
        many = x[:, 3] >= 100
        if many.sum() >= 10 and (~many).sum() >= 10:
            for lab, sel in (("subjects with >= 100 trials", many), ("subjects with < 100 trials", ~many)):
                out(f"      {lab} (n = {int(sel.sum())}): mixture r {_corr(x[sel, 2], acc[sel]):+.2f}, error "
                    f"{100 * np.mean(np.abs(x[sel, 2] - acc[sel])):.1f} | confidence r {_corr(x[sel, 1], acc[sel]):+.2f}, error "
                    f"{100 * np.mean(np.abs(x[sel, 1] - acc[sel])):.1f}")
    out()
    S9.write_csv("step10_estimate.csv", "montage,target,subject,accuracy,mean_confidence,mixture_estimate,n_trials", rows)


# =========================== [T] states: rest-referenced alignment ===========================
def get_states(ds, subj, mods, libs, loaders, cache):
    path = os.path.join(cache, f"{ds}_{subj:03d}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    X, y, _ = loaders[ds](subj, mods)
    X = np.asarray(X)
    if X.shape[2] < 1126:
        raise RuntimeError(f"epochs of {X.shape[2]} samples; 1126 expected for -2.0 to 2.5 s at 250 Hz")
    ok = np.isfinite(X).all(axis=(1, 2)) & (X[..., :500].var(axis=2).min(axis=1) > 0) & (X[..., 625:1126].var(axis=2).min(axis=1) > 0)
    X, y = X[ok], np.asarray(y)[ok]
    X3 = X if ds == "2b" else X[:, S5.TRIPLET_IN_SM, :]
    rec = dict(y=y, R3=S5._oas(X3[..., :500], libs[0]), T3=S5._oas(X3[..., 625:1126], libs[0]))
    if ds != "2b":
        rec.update(R17=S5._oas(X[..., :500], libs[0]), T17=S5._oas(X[..., 625:1126], libs[0]))
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(path, **rec)
    return rec


def section_t(mods, libs, main):
    mean_riemann = libs[2]
    cache = os.path.join(OUTDIR, "cache_states_pre2s")
    loaders = S9.make_loaders((8.0, 30.0), (-2.0, 2.5))
    data, t0 = {}, time.time()
    for ds in S5.DATASETS:
        data[ds] = [dict(subj=r["subj"], **get_states(ds, r["subj"], mods, libs, loaders, cache)) for r in main[ds]]
        print(f"    [T] {ds}: rest and task covariances ready ({time.time() - t0:.0f} s)", flush=True)
    # the task window must reproduce the main cache
    worst = 0.0
    for ds in S5.DATASETS:
        for r, m in zip(data[ds], main[ds]):
            if len(r["y"]) == len(m["y"]):
                worst = max(worst, float(np.max(np.abs(tn(r["T3"]) - tn(m["C3"])))))
    rows = []
    for montage, kr, kt in (("3ch", "R3", "T3"), ("17ch", "R17", "T17")):
        names = [d for d in S5.MONTAGES[montage] if len(data[d]) >= 3]
        T = {d: [tn(r[kt]) for r in data[d]] for d in names}
        B = {d: [tn(r[kr]) for r in data[d]] for d in names}
        Rt = {d: [mean_riemann(c) for c in T[d]] for d in names}
        Rb = {d: [mean_riemann(c) for c in B[d]] for d in names}

        def running(Bj, Tj):
            X = np.empty_like(Tj)
            R = Bj[0]
            for t in range(len(Tj)):
                if t > 0:                                   # mean of the baselines of trials 0 .. t
                    R = S8.geodesic(R, Bj[t], 1.0 / (t + 1))
                X[t] = S8.apply_ref(R, Tj[t:t + 1])[0]
            return X

        def per_trial(Bj, Tj):
            return np.array([S8.apply_ref(b, t[None])[0] for b, t in zip(Bj, Tj)])

        feats = {"none": T,
                 "task": {d: [S8.apply_ref(R, c) for R, c in zip(Rt[d], T[d])] for d in names},
                 "rest": {d: [S8.apply_ref(R, c) for R, c in zip(Rb[d], T[d])] for d in names},
                 "trial": {d: [per_trial(b, c) for b, c in zip(B[d], T[d])] for d in names}}
        feats["rest_run"] = {d: [running(b, c) for b, c in zip(B[d], T[d])] for d in names}
        train = {"none": "none", "task": "task", "rest": "rest", "rest_run": "rest", "trial": "trial"}
        acc = {}
        for s in names:
            ys = np.concatenate([r["y"] for r in data[s]])
            models = {v: S5.Model(np.concatenate(feats[v][s]), ys, "lr", libs) for v in ("none", "task", "rest", "trial")}
            for t in names:
                if t == s:
                    continue
                for j, r in enumerate(data[t]):
                    for v, tr in train.items():
                        acc.setdefault((t, r["subj"]), {}).setdefault(v, []).append(
                            np.mean(models[tr].predict(feats[v][t][j]) == r["y"]))
        for (t, sub), d in acc.items():
            rows.append((montage, t, sub, *(float(np.mean(d[v])) for v in ("none", "task", "rest", "rest_run", "trial"))))
        print(f"    [T] {montage}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[T] States: reference from the 2 s before each cue (rest) instead of from the task trials (TS-LR, accuracy %)")
    out(f"    check: the task window reproduces the main cache to {worst:.1e} (trace-normalized covariances, C3/Cz/C4)")
    out("    none: no alignment | task: own task trials (primary) | rest: one rest reference per subject |")
    out("    rest, running: causal mean of the baselines up to the current trial (target; source as 'rest') |")
    out("    per trial: every trial referenced to its own baseline (source and target)")
    out("    (in Cho2017 the stored order of the trials is not the recording order, so 'running' is not chronological there)")
    for montage in ("3ch", "17ch"):
        sel = [r for r in rows if r[0] == montage]
        x = np.array([r[3:] for r in sel])
        tg = np.array([r[1] for r in sel])
        out(f"    {montage}: {len(x)} subjects")
        out(f"      {'target':<11}{'n':>4}{'none':>8}{'task':>8}{'rest':>8}{'running':>9}{'per trial':>11}")
        for g in [d for d in S5.MONTAGES[montage] if (tg == d).any()] + ["all"]:
            s = np.ones(len(tg), bool) if g == "all" else tg == g
            out(f"      {g:<11}{int(s.sum()):>4}" + "".join(f"{100 * v:>{w}.2f}" for v, w in zip(x[s].mean(0), (8, 8, 8, 9, 11))))
        out(f"      rest - none            {S5.fmt(S5.paired(x[:, 2], x[:, 0]))}")
        out(f"      rest - task            {S5.fmt(S5.paired(x[:, 2], x[:, 1]))}")
        out(f"      rest, running - task   {S5.fmt(S5.paired(x[:, 3], x[:, 1]))}")
        out(f"      per trial - task       {S5.fmt(S5.paired(x[:, 4], x[:, 1]))}")
    out()
    S9.write_csv("step10_states.csv", "montage,target,subject,acc_none,acc_task,acc_rest,acc_rest_running,acc_per_trial", rows)


def run():
    t_start = time.time()
    libs = S5._libs()
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 10 headroom tests  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    main = S8.load_cache()
    out("subjects: " + ", ".join(f"{d} {len(main[d])}" for d in S5.DATASETS))
    out()
    mont = [S8.Montage(main, m, libs) for m in S5.MONTAGES]
    S9.guard("P", section_p, mont)
    S9.guard("Q", section_q, mont, libs)
    S9.guard("R", section_r, mont, libs)
    S9.guard("S", section_s, mont)
    try:
        mods = S9.project_modules()
    except Exception as e:
        mods = None
        out(f"[T] skipped: the project modules could not be imported ({e!r})")
    if mods is not None:
        S9.guard("T", section_t, mods, libs, main)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step10_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    try:
        import step7_final_checks as S7
        names = S7.make_bundle()
        print(f"bundle refreshed: {os.path.abspath(S7.BUNDLE)} ({len(names)} files) -- attach this one file in the chat")
    except Exception as e:
        print(f"written: {os.path.abspath(OUTDIR)} (step10_summary.txt and CSV files); bundle not refreshed ({e!r})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
