"""
step13_adaptive_reference.py

Run from the Codes folder, after step 12 (about 15 to 25 minutes, cache only):

    python step13_adaptive_reference.py

Step 12 showed two label-free references with opposite weaknesses:
    task reference  (mean of the unlabeled task trials): best when the classes are balanced,
                    loses accuracy when they are not, exactly as the law predicts;
    rest reference  (mean of the pre-cue baselines): unaffected by class balance,
                    about one point lower when the classes are balanced.

This script tests a rule that chooses between them without labels.

Adaptive reference.  If the task trials are balanced, the task mean and the rest
mean differ only by the usual rest-to-task offset of a subject. If one class
dominates, the task mean is pulled towards that class. The statistic
    delta = w . log( R_rest^(-1/2) R_task R_rest^(-1/2) )
(w: weights of the frozen classifier) measures that pull along the decision
axis. It is standardized with the values of other subjects, whose data are
balanced (z), and the reference is moved from the task mean towards the rest
mean by the fraction  z^2 / (1 + z^2)  along the geodesic. The reference
population for z is either the source dataset or the earlier users of the
target site.

Compared with: task reference, rest reference, a fixed 50/50 blend, and a bias
chosen by information maximization on the unlabeled trials (the idea behind
test-time methods for label shift, implemented here in its simplest form for
the linear classifier; it is NOT a re-implementation of any published method).

Parts
  [X0] all trials of a subject, balanced
  [X1] a reference from 16 unlabeled trials with a fixed class composition
  [X2] pseudo-online streams, recorded order and blocked order

Outputs in corrected_results/: step13_summary.txt, step13_*.csv, refreshed results_bundle.zip.
"""
from __future__ import annotations

import os
import sys
import time
import datetime

import numpy as np

try:
    import step5_benchmark as S5
    import step8_new_analyses as S8
    import step9_reviewer_checks as S9
    import step10_headroom as S10
    import step11_directions as S11
    import step12_rest_anchored as S12
except Exception as _e:                                    # pragma: no cover
    print(f"could not import the earlier step scripts (5, 8 to 12); run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
out = S5.out
COMPOSITIONS = [(8, 8), (12, 4), (4, 12), (14, 2), (2, 14), (16, 0), (0, 16)]
N_DRAW = 6
METHODS = ["task", "rest", "blend", "adaptive_site", "adaptive_source", "infomax"]
LABELS = ["task", "rest", "50/50", "adaptive", "adaptive", "info-max"]
SUB = ["", "", "blend", "(site)", "(source)", "bias"]
BETAS = np.linspace(-3.0, 3.0, 61)


def weights(M, s):
    m = M.task_model[s]
    return m.m.coef_.ravel(), float(m.m.intercept_[0])


def logits(M, s, R, C):
    m = M.task_model[s]
    w, b = weights(M, s)
    return m.ts(S8.apply_ref(R, C), m.ref) @ w + b


def pull(M, s, R_task, R_rest):
    """Position of the task mean along the decision axis when the rest mean is the reference."""
    m = M.task_model[s]
    return float(m.ts(S8.apply_ref(R_rest, R_task[None]), m.ref)[0] @ weights(M, s)[0])


def infomax_bias(f):
    """Bias added to the logits that maximizes  H(mean p) - mean H(p)  on the unlabeled trials."""
    p = 1.0 / (1.0 + np.exp(-np.clip(f[None, :] + BETAS[:, None], -30, 30)))
    h = lambda q: -(q * np.log(q + 1e-12) + (1 - q) * np.log(1 - q + 1e-12))
    return float(BETAS[int(np.argmax(h(p.mean(1)) - h(p).mean(1)))])


class Population:
    """Per model and subject: the pull of balanced data and the spread of the trial logits."""

    def __init__(self, M):
        self.M = M
        self.delta, self.s2 = {}, {}
        for s in M.names:
            for d in M.names:
                self.delta[(s, d)] = np.array([pull(M, s, rt, rb) for rt, rb in zip(M.Rt[d], M.Rb[d])])
                self.s2[(s, d)] = np.array([np.var(logits(M, s, rb, T)) for rb, T in zip(M.Rb[d], M.T[d])])

    def stats(self, s, tgt, j, kind):
        if kind == "source":
            x, v = self.delta[(s, s)], self.s2[(s, s)]
        else:
            keep = np.arange(len(self.delta[(s, tgt)])) != j
            x, v = self.delta[(s, tgt)][keep], self.s2[(s, tgt)][keep]
        return float(x.mean()), float(x.std(ddof=1)), float(v.mean())

    def fraction(self, s, tgt, j, kind, R_task, R_rest, k):
        mu, tau, s2 = self.stats(s, tgt, j, kind)
        z = (pull(self.M, s, R_task, R_rest) - mu) / np.sqrt(tau ** 2 + s2 / max(k, 1))
        return float(z * z / (1.0 + z * z))


def predict_all(M, P, s, tgt, j, R_task, R_rest, k, C_unl, C_test):
    """Predictions (0/1) of every method on C_test, for the frozen model of source s."""
    res = {}
    f_task = logits(M, s, R_task, C_test)
    res["task"] = f_task > 0
    res["rest"] = logits(M, s, R_rest, C_test) > 0
    res["blend"] = logits(M, s, S8.geodesic(R_task, R_rest, 0.5), C_test) > 0
    for kind in ("site", "source"):
        lam = P.fraction(s, tgt, j, kind, R_task, R_rest, k)
        res["adaptive_" + kind] = logits(M, s, S8.geodesic(R_task, R_rest, lam), C_test) > 0
    res["infomax"] = f_task + infomax_bias(logits(M, s, R_task, C_unl)) > 0
    return res


def header(first, width=12):
    out(f"       {first:<{width}}{'n':>5}" + "".join(f"{a:>10}" for a in LABELS)
        + "   adaptive (site) - task  [95% CI]  dz  p")
    out(f"       {'':<{width}}{'':>5}" + "".join(f"{a:>10}" for a in SUB))


# =========================== [X0] all trials, balanced ===========================
def section_x0(st, pops):
    rows = []
    for M, P in zip(st, pops):
        for tgt in M.names:
            srcs = M.sources(tgt)
            for j, T in enumerate(M.T[tgt]):
                y = M.y[tgt][j] == 1
                acc = np.zeros(len(METHODS))
                for s in srcs:
                    r = predict_all(M, P, s, tgt, j, M.Rt[tgt][j], M.Rb[tgt][j], len(y), T, T)
                    acc += np.array([np.mean(r[m] == y) for m in METHODS])
                rows.append((M.name, tgt, M.subj[tgt][j], *(acc / len(srcs)).tolist()))
    out("[X0] All trials of a subject, balanced classes (accuracy %, frozen primary model, no label)")
    for M in st:
        x = S10.by_subject(rows, M.name, range(3, 3 + len(METHODS)))
        out(f"     {M.name}")
        header("")
        out(f"       {'all':<12}{len(x):>5}" + "".join(f"{100 * v:>10.2f}" for v in x.mean(0))
            + "   " + S5.fmt(S5.paired(x[:, 3], x[:, 0])))
        out(f"       adaptive (site) - rest     {S5.fmt(S5.paired(x[:, 3], x[:, 1]))}")
    out()
    S9.write_csv("step13_alltrials.csv", "montage,target,subject," + ",".join("acc_" + m for m in METHODS), rows)


# =========================== [X1] class composition ===========================
def section_x1(st, pops, libs):
    mean_riemann = libs[2]
    rows = []
    for mi, (M, P) in enumerate(zip(st, pops)):
        t0 = time.time()
        for ti, tgt in enumerate(M.names):
            srcs = M.sources(tgt)
            for j, T in enumerate(M.T[tgt]):
                y, B = M.y[tgt][j], M.B[tgt][j]
                i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
                full = float(np.mean([S11.bal_acc((logits(M, s, M.Rt[tgt][j], T) > 0).astype(int), y) for s in srcs]))
                rng = np.random.default_rng([17, mi, ti, int(M.subj[tgt][j])])
                for n1, n0 in COMPOSITIONS:
                    if len(i1) < n1 + 3 or len(i0) < n0 + 3:
                        continue
                    acc = np.zeros(len(METHODS))
                    for _ in range(N_DRAW):
                        idx = np.r_[rng.choice(i1, n1, replace=False), rng.choice(i0, n0, replace=False)].astype(int)
                        rest = np.setdiff1d(np.arange(len(y)), idx)
                        R_task, R_rest = mean_riemann(T[idx]), mean_riemann(B[idx])
                        for s in srcs:
                            r = predict_all(M, P, s, tgt, j, R_task, R_rest, len(idx), T[idx], T[rest])
                            acc += np.array([S11.bal_acc(r[m].astype(int), y[rest]) for m in METHODS]) / len(srcs)
                    rows.append((M.name, tgt, M.subj[tgt][j], n1, n0, *(acc / N_DRAW).tolist(), full))
        print(f"    [X1] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    nm = len(METHODS)
    groups = (("8 + 8", [(8, 8)]), ("12 + 4", [(12, 4), (4, 12)]), ("14 + 2", [(14, 2), (2, 14)]), ("16 + 0", [(16, 0), (0, 16)]))
    out("[X1] Reference from 16 unlabeled trials with a fixed class composition (balanced accuracy % on the other trials)")
    for M in st:
        sel = [r for r in rows if r[0] == M.name]
        fulls = {(r[1], r[2]): r[5 + nm] for r in sel}
        top = np.quantile(list(fulls.values()), 2 / 3)
        for title, keep in (("all subjects", lambda r: True), ("best third of the subjects", lambda r: r[5 + nm] >= top)):
            out(f"     {M.name}, {title}")
            header("composition")
            worst = {}
            for lab, comps in groups:
                by = {}
                for r in sel:
                    if (r[3], r[4]) in comps and keep(r):
                        by.setdefault((r[1], r[2]), []).append(r[5:5 + nm])
                v = np.array([np.mean(x, axis=0) for x in by.values() if len(x) == len(comps)])
                if len(v) < 3:
                    continue
                m = v.mean(0)
                for i, name in enumerate(METHODS):
                    worst[name] = max(worst.get(name, -1.0), float(m.max() - m[i]))
                out(f"       {lab:<12}{len(v):>5}" + "".join(f"{100 * a:>10.2f}" for a in m)
                    + "   " + S5.fmt(S5.paired(v[:, 3], v[:, 0])))
            out("       largest shortfall from the best method over the four compositions:")
            out("         " + " | ".join(f"{(LABELS[i] + ' ' + SUB[i]).strip()} {100 * worst[name]:.2f}"
                                         for i, name in enumerate(METHODS)))
    out()
    S9.write_csv("step13_composition.csv", "montage,target,subject,n_class1,n_class0,"
                 + ",".join("bacc_" + m for m in METHODS) + ",bacc_all_trials", rows)


# =========================== [X2] pseudo-online streams ===========================
def _stream(M, P, tgt, j, order):
    srcs = M.sources(tgt)
    T, B, y = M.T[tgt][j][order], M.B[tgt][j][order], M.y[tgt][j][order] == 1
    n = len(y)
    ok = {m: np.zeros(n) for m in METHODS}
    R_task, R_rest = T[0], B[0]
    for t in range(n):
        if t > 0:
            R_rest = S8.geodesic(R_rest, B[t], 1.0 / (t + 1))
        if t > 1:
            R_task = S8.geodesic(R_task, T[t - 1], 1.0 / t)
        cur = T[t:t + 1]
        for s in srcs:
            if t == 0:                                   # no earlier task trial: only the rest reference exists
                raw = (logits(M, s, np.eye(T.shape[1]), cur) > 0)[0]
                rst = (logits(M, s, R_rest, cur) > 0)[0]
                r = {"task": raw, "infomax": raw, "rest": rst, "blend": rst, "adaptive_site": rst, "adaptive_source": rst}
            else:
                r = {k: v[0] for k, v in predict_all(M, P, s, tgt, j, R_task, R_rest, t, T[:t], cur).items()}
            for m in METHODS:
                ok[m][t] += float(r[m] == y[t]) / len(srcs)
    return [float(ok[m].mean()) for m in METHODS]


def section_x2(st, pops):
    rows = []
    for M, P in zip(st, pops):
        t0 = time.time()
        for tgt in [d for d in M.names if d in S10.ORDERED]:
            for j in range(len(M.T[tgt])):
                y = M.y[tgt][j]
                i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
                rec = _stream(M, P, tgt, j, np.arange(len(y)))
                blk = np.mean([_stream(M, P, tgt, j, np.r_[i1, i0]), _stream(M, P, tgt, j, np.r_[i0, i1])], axis=0)
                rows.append((M.name, tgt, M.subj[tgt][j], *rec, *blk.tolist()))
        print(f"    [X2] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    nm = len(METHODS)
    out("[X2] Pseudo-online streams (accuracy % over the stream; every reference uses earlier trials only)")
    out("     recorded: trials as recorded | blocked: all trials of one class, then all trials of the other")
    for M in st:
        x = S10.by_subject(rows, M.name, range(3, 3 + 2 * nm))
        if len(x) == 0:
            continue
        out(f"     {M.name}")
        header("stream")
        for i, name in enumerate(("recorded", "blocked")):
            v = x[:, nm * i:nm * (i + 1)]
            out(f"       {name:<12}{len(v):>5}" + "".join(f"{100 * a:>10.2f}" for a in v.mean(0))
                + "   " + S5.fmt(S5.paired(v[:, 3], v[:, 0])))
        v = x[:, nm:]
        out(f"       blocked: adaptive (site) - rest   {S5.fmt(S5.paired(v[:, 3], v[:, 1]))}")
    out()
    S9.write_csv("step13_streams.csv", "montage,target,subject," + ",".join(
        f"acc_{o}_{m}" for o in ("recorded", "blocked") for m in METHODS), rows)


def run():
    t_start = time.time()
    libs = S5._libs()
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 13 adaptive reference  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    main = S8.load_cache()
    out("subjects: " + ", ".join(f"{d} {len(main[d])}" for d in S5.DATASETS))
    out()
    try:
        data = S12.load_states(main, libs)
    except Exception as e:
        print(str(e))
        return 1
    st = [S12.States(data, m, libs) for m in S5.MONTAGES]
    pops = [Population(M) for M in st]
    for M, P in zip(st, pops):
        out(f"{M.name}: pull of balanced data along the decision axis, mean and spread over subjects, per frozen model")
        for s in M.names:
            out("    model " + f"{s:<10}" + " | ".join(
                f"{d}: {P.delta[(s, d)].mean():+.2f} +- {P.delta[(s, d)].std(ddof=1):.2f}" for d in M.names))
    out()
    S9.guard("X0", section_x0, st, pops)
    S9.guard("X1", section_x1, st, pops, libs)
    S9.guard("X2", section_x2, st, pops)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step13_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    try:
        import step7_final_checks as S7
        names = S7.make_bundle()
        print(f"bundle refreshed: {os.path.abspath(S7.BUNDLE)} ({len(names)} files) -- attach this one file in the chat")
    except Exception as e:
        print(f"written: {os.path.abspath(OUTDIR)} (step13_summary.txt and CSV files); bundle not refreshed ({e!r})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
