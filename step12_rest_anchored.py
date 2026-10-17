"""
step12_rest_anchored.py

Run from the Codes folder, after step 10 (its part T wrote the rest/task cache)
and with step11_directions.py in the folder (about 5 to 10 minutes, cache only):

    python step12_rest_anchored.py

A new method, and the test that decides whether it is worth putting in the paper.

The problem. Recentering a new user on the mean of the unlabeled TASK trials
assumes that the two classes are equally frequent among those trials. The law
of step 11 (part L2) says how much accuracy is lost when they are not, and real
use of a BCI is not balanced.

The method (rest-anchored recentering). The reference of the user is predicted
from signals that contain no class information:
    R_hat = R_rest^(1/2)  D  R_rest^(1/2)
R_rest : mean covariance of the user's baselines (the 2 s before each cue)
D      : the average rest-to-task offset of OTHER subjects, D_j = R_rest_j^(-1/2) R_task_j R_rest_j^(-1/2)
         (from the source dataset, or from earlier users of the target site; no label is used)
The frozen source classifier of the primary pipeline is then applied unchanged.

Parts
  [W1] all trials, balanced classes: task reference (primary), rest reference alone, rest + offset
  [W2] a reference from 16 trials whose class composition is fixed (8+8, 12+4, 14+2, 16+0):
       task-based reference against rest-anchored reference, with the prediction of the law
  [W3] pseudo-online streams in the recorded order and in a blocked order (all trials of one class,
       then all trials of the other): running task mean against running rest-anchored reference
  [P2] the control for dynamic recentering of step 11, if it has not been run yet

Outputs in corrected_results/: step12_summary.txt, step12_*.csv, refreshed results_bundle.zip.
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
except Exception as _e:                                    # pragma: no cover
    print(f"could not import the earlier step scripts (5, 8, 9, 10, 11); run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
out = S5.out
tn = S8.tn
STATES = os.path.join(OUTDIR, "cache_states_pre2s")
COMPOSITIONS = [(8, 8), (12, 4), (4, 12), (14, 2), (2, 14), (16, 0), (0, 16)]
N_DRAW = 10


def predicted_reference(Rb, D):
    h = S8.sqrtm(Rb)
    return h @ D @ h


class States:
    """Task and rest covariances of one montage, references, offsets and frozen models."""

    def __init__(self, data, montage, libs):
        mean_riemann = libs[2]
        kr, kt = ("R3", "T3") if montage == "3ch" else ("R17", "T17")
        self.name = montage
        self.names = [d for d in S5.MONTAGES[montage] if len(data[d]) >= 3]
        self.T = {d: [tn(r[kt]) for r in data[d]] for d in self.names}
        self.B = {d: [tn(r[kr]) for r in data[d]] for d in self.names}
        self.y = {d: [r["y"] for r in data[d]] for d in self.names}
        self.subj = {d: [r["subj"] for r in data[d]] for d in self.names}
        self.Rt = {d: [mean_riemann(c) for c in self.T[d]] for d in self.names}
        self.Rb = {d: [mean_riemann(c) for c in self.B[d]] for d in self.names}
        self.D = {d: np.array([S8.apply_ref(rb, rt[None])[0] for rb, rt in zip(self.Rb[d], self.Rt[d])])
                  for d in self.names}
        self.D_src = {d: mean_riemann(self.D[d]) for d in self.names}
        self.D_site = {d: [mean_riemann(self.D[d][np.arange(len(self.D[d])) != j]) for j in range(len(self.D[d]))]
                       for d in self.names}
        self.task_model, self.rest_model = {}, {}
        for s in self.names:
            ys = np.concatenate(self.y[s])
            self.task_model[s] = S5.Model(np.concatenate([S8.apply_ref(R, c) for R, c in zip(self.Rt[s], self.T[s])]),
                                          ys, "lr", libs)
            self.rest_model[s] = S5.Model(np.concatenate([S8.apply_ref(R, c) for R, c in zip(self.Rb[s], self.T[s])]),
                                          ys, "lr", libs)

    def sources(self, tgt):
        return [s for s in self.names if s != tgt]


def load_states(main, libs):
    if not os.path.isdir(STATES):
        raise RuntimeError(f"{STATES} not found: run step10_headroom.py (part T) first")
    return {ds: [dict(subj=r["subj"], **S10.get_states(ds, r["subj"], None, libs, None, STATES)) for r in main[ds]]
            for ds in S5.DATASETS}


# =========================== [W1] all trials ===========================
def section_w1(st):
    rows, geo = [], []
    for M in st:
        for tgt in M.names:
            srcs = M.sources(tgt)
            for j, T in enumerate(M.T[tgt]):
                y, Rb, Rt = M.y[tgt][j], M.Rb[tgt][j], M.Rt[tgt][j]
                acc = lambda model, C: float(np.mean([np.mean(model[s].predict(C[s] if isinstance(C, dict) else C) == y)
                                                      for s in srcs]))
                a_task = acc(M.task_model, S8.apply_ref(Rt, T))
                a_rest = acc(M.rest_model, S8.apply_ref(Rb, T))
                a_mis = acc(M.task_model, S8.apply_ref(Rb, T))
                a_src = acc(M.task_model, {s: S8.apply_ref(predicted_reference(Rb, M.D_src[s]), T) for s in srcs})
                a_site = acc(M.task_model, S8.apply_ref(predicted_reference(Rb, M.D_site[tgt][j]), T))
                rows.append((M.name, tgt, M.subj[tgt][j], a_task, a_rest, a_mis, a_src, a_site))
                geo.append((M.name, float(np.sqrt(S8.dist2(Rt, Rb[None])[0])),
                            float(np.sqrt(S8.dist2(Rt, predicted_reference(Rb, M.D_site[tgt][j])[None])[0])),
                            float(np.mean([np.sqrt(S8.dist2(Rt, predicted_reference(Rb, M.D_src[s])[None])[0]) for s in srcs]))))
    out("[W1] All trials, balanced classes (TS-LR, accuracy %; the frozen model is the primary one unless stated)")
    out("     task: own task trials (primary) | rest, rest model: rest reference, model trained on rest-referenced sources |")
    out("     rest only: rest reference fed to the primary model | rest + offset: predicted task reference, with the")
    out("     offset of the source dataset or of the other subjects of the target dataset (site)")
    for M in st:
        x = S10.by_subject(rows, M.name, range(3, 8))
        g = np.array([r[1:] for r in geo if r[0] == M.name])
        out(f"     {M.name}: {len(x)} subjects")
        out(f"       distance from the true task reference: rest reference {np.median(g[:, 0]):.2f} | rest + offset (site) "
            f"{np.median(g[:, 1]):.2f} | rest + offset (source) {np.median(g[:, 2]):.2f}   (medians)")
        names = ["task (primary)", "rest, rest model", "rest only", "rest + offset (source)", "rest + offset (site)"]
        for i, nm in enumerate(names):
            tail = "" if i == 0 else "   against task  " + S5.fmt(S5.paired(x[:, i], x[:, 0]))
            out(f"       {nm:<24}{100 * x[:, i].mean():>7.2f}{tail}")
        out(f"       rest + offset (source) against rest, rest model   {S5.fmt(S5.paired(x[:, 3], x[:, 1]))}")
        out(f"       rest + offset (site) against rest, rest model     {S5.fmt(S5.paired(x[:, 4], x[:, 1]))}")
    out()
    S9.write_csv("step12_alltrials.csv", "montage,target,subject,acc_task,acc_rest_restmodel,acc_rest_only,"
                 "acc_rest_offset_source,acc_rest_offset_site", rows)


# =========================== [W2] class composition of the unlabeled trials ===========================
def section_w2(st, libs):
    mean_riemann = libs[2]
    rows = []
    for mi, M in enumerate(st):
        t0 = time.time()
        for ti, tgt in enumerate(M.names):
            srcs = M.sources(tgt)
            for j, T in enumerate(M.T[tgt]):
                y, B = M.y[tgt][j], M.B[tgt][j]
                i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
                full = float(np.mean([S11.bal_acc(M.task_model[s].predict(S8.apply_ref(M.Rt[tgt][j], T)), y) for s in srcs]))
                rng = np.random.default_rng([16, mi, ti, int(M.subj[tgt][j])])
                for n1, n0 in COMPOSITIONS:
                    if len(i1) < n1 + 3 or len(i0) < n0 + 3:
                        continue
                    acc = np.zeros(4)                     # task mean, rest (rest model), rest + offset source, site
                    for _ in range(N_DRAW):
                        idx = np.r_[rng.choice(i1, n1, replace=False), rng.choice(i0, n0, replace=False)].astype(int)
                        rest = np.setdiff1d(np.arange(len(y)), idx)
                        Rt_k, Rb_k = mean_riemann(T[idx]), mean_riemann(B[idx])
                        Tr, yr = T[rest], y[rest]
                        site = S8.apply_ref(predicted_reference(Rb_k, M.D_site[tgt][j]), Tr)
                        part = np.zeros(4)
                        for s in srcs:
                            part[0] += S11.bal_acc(M.task_model[s].predict(S8.apply_ref(Rt_k, Tr)), yr)
                            part[1] += S11.bal_acc(M.rest_model[s].predict(S8.apply_ref(Rb_k, Tr)), yr)
                            part[2] += S11.bal_acc(M.task_model[s].predict(
                                S8.apply_ref(predicted_reference(Rb_k, M.D_src[s]), Tr)), yr)
                            part[3] += S11.bal_acc(M.task_model[s].predict(site), yr)
                        acc += part / len(srcs)
                    rows.append((M.name, tgt, M.subj[tgt][j], n1, n0, *(acc / N_DRAW).tolist(), full))
        print(f"    [W2] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[W2] Reference from 16 unlabeled trials with a fixed class composition (balanced accuracy % on the other trials)")
    out("     task mean: the usual recentering | law: its prediction | rest + offset: the reference never sees the task trials")
    groups = (("8 + 8", [(8, 8)], 0.5), ("12 + 4", [(12, 4), (4, 12)], 0.75), ("14 + 2", [(14, 2), (2, 14)], 0.875),
              ("16 + 0", [(16, 0), (0, 16)], 1.0))
    for M in st:
        sel = [r for r in rows if r[0] == M.name]
        out(f"     {M.name}")
        out(f"       {'composition':<12}{'n':>5}{'task mean':>11}{'law':>8}{'rest':>8}{'rest+offset':>13}{'rest+offset':>13}"
            "   rest + offset (source) - task mean  [95% CI]  dz  p")
        out(f"       {'':<12}{'':>5}{'':>11}{'':>8}{'':>8}{'(source)':>13}{'(site)':>13}")
        for lab, comps, pi in groups:
            by = {}
            for r in sel:
                if (r[3], r[4]) in comps:
                    by.setdefault((r[1], r[2]), []).append(r[5:])
            v = np.array([np.mean(x, axis=0) for x in by.values() if len(x) == len(comps)])
            if len(v) < 3:
                continue
            m = 100 * v.mean(0)
            out(f"       {lab:<12}{len(v):>5}{m[0]:>11.2f}{100 * S11.law(v[:, 4], 16, pi).mean():>8.2f}{m[1]:>8.2f}{m[2]:>13.2f}"
                f"{m[3]:>13.2f}   {S5.fmt(S5.paired(v[:, 2], v[:, 0]))}")
        # the best third of the subjects, where the law predicts the largest loss
        full = {}
        for r in sel:
            full[(r[1], r[2])] = r[9]
        top = np.quantile(list(full.values()), 2 / 3)
        out("       best third of the subjects:")
        for lab, comps, pi in groups:
            by = {}
            for r in sel:
                if (r[3], r[4]) in comps and r[9] >= top:
                    by.setdefault((r[1], r[2]), []).append(r[5:])
            v = np.array([np.mean(x, axis=0) for x in by.values() if len(x) == len(comps)])
            if len(v) < 3:
                continue
            m = 100 * v.mean(0)
            out(f"       {lab:<12}{len(v):>5}{m[0]:>11.2f}{100 * S11.law(v[:, 4], 16, pi).mean():>8.2f}{m[1]:>8.2f}{m[2]:>13.2f}"
                f"{m[3]:>13.2f}   {S5.fmt(S5.paired(v[:, 2], v[:, 0]))}")
    out()
    S9.write_csv("step12_composition.csv", "montage,target,subject,n_class1,n_class0,bacc_task_mean,bacc_rest_restmodel,"
                 "bacc_rest_offset_source,bacc_rest_offset_site,bacc_all_trials", rows)


# =========================== [W3] pseudo-online streams ===========================
def _stream(M, tgt, j, order):
    """Prequential accuracy over the stream for three causal references and the offline one."""
    srcs = M.sources(tgt)
    T, B, y = M.T[tgt][j][order], M.B[tgt][j][order], M.y[tgt][j][order]
    n = len(y)
    Xt, Xr = np.empty_like(T), np.empty_like(T)
    Xo = {s: np.empty_like(T) for s in srcs}
    Rt = T[0]
    Rb = B[0]
    for t in range(n):
        if t > 0:
            Rb = S8.geodesic(Rb, B[t], 1.0 / (t + 1))          # baselines of trials 0 .. t
        if t > 1:
            Rt = S8.geodesic(Rt, T[t - 1], 1.0 / t)            # task trials 0 .. t-1
        Xt[t] = T[t] if t == 0 else S8.apply_ref(Rt, T[t:t + 1])[0]
        Xr[t] = S8.apply_ref(Rb, T[t:t + 1])[0]
        for s in srcs:
            Xo[s][t] = S8.apply_ref(predicted_reference(Rb, M.D_src[s]), T[t:t + 1])[0]
    off = S8.apply_ref(M.Rt[tgt][j], T)
    return [float(np.mean([np.mean(M.task_model[s].predict(Xt) == y) for s in srcs])),
            float(np.mean([np.mean(M.rest_model[s].predict(Xr) == y) for s in srcs])),
            float(np.mean([np.mean(M.task_model[s].predict(Xo[s]) == y) for s in srcs])),
            float(np.mean([np.mean(M.task_model[s].predict(off) == y) for s in srcs]))]


def section_w3(st):
    rows = []
    for M in st:
        t0 = time.time()
        for tgt in [d for d in M.names if d in S10.ORDERED]:
            for j in range(len(M.T[tgt])):
                y = M.y[tgt][j]
                n = len(y)
                rec = _stream(M, tgt, j, np.arange(n))
                i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
                blk = np.mean([_stream(M, tgt, j, np.r_[i1, i0]), _stream(M, tgt, j, np.r_[i0, i1])], axis=0)
                rows.append((M.name, tgt, M.subj[tgt][j], *rec, *blk.tolist()))
        print(f"    [W3] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[W3] Pseudo-online streams (accuracy % over the stream; targets with recording order)")
    out("     recorded: trials as recorded | blocked: all trials of one class, then all trials of the other (both orders")
    out("     averaged), the extreme case of a user who repeats one command")
    out("     task mean: running mean of the earlier task trials | rest: running rest reference, rest model |")
    out("     rest + offset: running rest reference and the offset of the source dataset | offline: all task trials")
    for M in st:
        x = S10.by_subject(rows, M.name, range(3, 11))
        if len(x) == 0:
            continue
        out(f"     {M.name}: {len(x)} subjects")
        out(f"       {'stream':<10}{'task mean':>11}{'rest':>8}{'rest+offset':>13}{'offline':>9}"
            "   rest + offset - task mean  [95% CI]  dz  p")
        for i, nm in enumerate(("recorded", "blocked")):
            v = x[:, 4 * i:4 * i + 4]
            out(f"       {nm:<10}" + "".join(f"{100 * a:>{w}.2f}" for a, w in zip(v.mean(0), (11, 8, 13, 9)))
                + "   " + S5.fmt(S5.paired(v[:, 2], v[:, 0])))
    out()
    S9.write_csv("step12_streams.csv", "montage,target,subject," + ",".join(
        f"acc_{o}_{m}" for o in ("recorded", "blocked") for m in ("task_mean", "rest", "rest_offset", "offline")), rows)


def run():
    t_start = time.time()
    libs = S5._libs()
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 12 rest-anchored recentering  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    main = S8.load_cache()
    out("subjects: " + ", ".join(f"{d} {len(main[d])}" for d in S5.DATASETS))
    out()
    try:
        data = load_states(main, libs)
    except Exception as e:
        print(str(e))
        return 1
    st = [States(data, m, libs) for m in S5.MONTAGES]
    S9.guard("W1", section_w1, st)
    S9.guard("W2", section_w2, st, libs)
    S9.guard("W3", section_w3, st)
    if hasattr(S11, "section_p2") and not S11.have("step11_dynamic_controls.csv"):
        mont = [S8.Montage(main, m, libs) for m in S5.MONTAGES]
        S9.guard("P2", S11.section_p2, mont, libs, main, None)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step12_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    try:
        import step7_final_checks as S7
        names = S7.make_bundle()
        print(f"bundle refreshed: {os.path.abspath(S7.BUNDLE)} ({len(names)} files) -- attach this one file in the chat")
    except Exception as e:
        print(f"written: {os.path.abspath(OUTDIR)} (step12_summary.txt and CSV files); bundle not refreshed ({e!r})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
