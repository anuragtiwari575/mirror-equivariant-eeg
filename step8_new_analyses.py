"""
step8_new_analyses.py

Run from the Codes folder, after step 5 (about 15 to 25 minutes, cache only):

    python step8_new_analyses.py

Three analyses that go beyond the benchmark. All use the primary pipeline of
step 5 (trace-normalized OAS covariance, every source subject recentered,
tangent-space logistic regression fitted on the source and frozen). No target
label is used except to score.

[G] Which level of recentering is needed? The trials of a target subject are
    recentered with
      none        no reference at all
      source      the centre of the source dataset (nothing from the target site)
      site        the mean of the OTHER subjects of the target dataset
                  (previous users of the same setup; zero trials of the new user)
      own         the mean of the subject's own trials (the main analysis)
[H] Few trials: a shrinkage estimate of the subject's reference. With k
    unlabeled trials the plain estimate is their mean. The shrinkage estimate
    moves from a prior towards that mean along the geodesic by the fraction
    k / (k + k0), where k0 = (dispersion of trials inside a subject) /
    (dispersion of subject means), estimated without labels from the subjects
    that define the prior. Priors: the site (other target subjects) and the
    source centre. Scored on the trials not used for the reference.
[I] Can one tell without labels whether transfer works for a subject? The mean
    confidence of the frozen classifier on a subject's unlabeled trials is
    compared with the subject's true accuracy.

Outputs (new files only), in corrected_results/:
    step8_levels.csv, step8_shrinkage.csv, step8_confidence.csv, step8_summary.txt
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
MIN_REST = 5
out = S5.out


def tn(C):
    return C / np.trace(C, axis1=1, axis2=2)[:, None, None]


# =========================== SPD helpers ===========================
def _eig_fun(M, fun):
    w, V = np.linalg.eigh(0.5 * (M + np.swapaxes(M, -1, -2)))
    return (V * fun(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def sqrtm(M):
    return _eig_fun(M, np.sqrt)


def invsqrtm(M):
    return _eig_fun(M, lambda w: 1.0 / np.sqrt(w))


def dist2(R, C):
    """Squared affine-invariant distance between one matrix R and a stack C."""
    P = invsqrtm(R)
    w = np.linalg.eigvalsh(P @ C @ P)
    return np.sum(np.log(w) ** 2, axis=-1)


def geodesic(P, R, alpha):
    """Point at fraction alpha on the geodesic from P (alpha = 0) to R (alpha = 1)."""
    Ph, Pih = sqrtm(P), invsqrtm(P)
    return Ph @ _eig_fun(Pih @ R @ Pih, lambda w: w ** alpha) @ Ph


def apply_ref(R, C):
    P = invsqrtm(R)
    return P @ C @ P


# =========================== data ===========================
def load_cache():
    data = {}
    for ds in S5.DATASETS:
        recs = []
        for f in sorted(glob.glob(os.path.join(S5.CACHE, f"{ds}_*.npz"))):
            recs.append(S5.get_subject(ds, int(os.path.basename(f)[len(ds) + 1:-4]), None, None))
        recs = sorted(recs, key=lambda r: r["subj"])
        data[ds] = [r for r in recs if not S5.exclusion_reason(ds, r)]
    return data


class Montage:
    """Everything that is needed once per montage: covariances, subject means,
    dispersions, priors and the frozen source models."""

    def __init__(self, data, montage, libs):
        mean_riemann = libs[2]
        key = "C3" if montage == "3ch" else "C17"
        self.name = montage
        self.names = [d for d in S5.MONTAGES[montage] if len(data.get(d, [])) >= 3]
        self.C = {d: [tn(r[key]) for r in data[d]] for d in self.names}
        self.y = {d: [r["y"] for r in data[d]] for d in self.names}
        self.subj = {d: [r["subj"] for r in data[d]] for d in self.names}
        self.R = {d: np.array([mean_riemann(C) for C in self.C[d]]) for d in self.names}
        # dispersion of the trials around their subject mean (per subject)
        self.s2 = {d: np.array([dist2(R, C).mean() for R, C in zip(self.R[d], self.C[d])])
                   for d in self.names}
        self.n = {d: np.array([len(C) for C in self.C[d]]) for d in self.names}
        self.centre, self.k0_src = {}, {}
        for d in self.names:
            self.centre[d] = mean_riemann(self.R[d])
            self.k0_src[d] = self._k0(self.centre[d], self.R[d], self.s2[d], self.n[d])
        self.site, self.k0_site = {}, {}
        for d in self.names:
            priors, k0s = [], []
            for j in range(len(self.R[d])):
                keep = np.arange(len(self.R[d])) != j
                P = mean_riemann(self.R[d][keep])
                priors.append(P)
                k0s.append(self._k0(P, self.R[d][keep], self.s2[d][keep], self.n[d][keep]))
            self.site[d], self.k0_site[d] = priors, np.array(k0s)
        self.models = {}
        for s in self.names:
            Cs = np.concatenate([apply_ref(R, C) for R, C in zip(self.R[s], self.C[s])])
            self.models[s] = S5.Model(Cs, np.concatenate(self.y[s]), "lr", libs)

    @staticmethod
    def _k0(P, R, s2, n):
        """k0 = within-subject dispersion / between-subject dispersion of the means.
        The between-subject term is corrected for the estimation noise of each mean."""
        within = float(np.mean(s2))
        between = float(np.mean(dist2(P, R)) - np.mean(s2 / n))
        return within / max(between, 1e-6 * within)

    def sources(self, tgt):
        return [s for s in self.names if s != tgt]

    def accuracy(self, tgt, j, Ct, idx=None):
        """Accuracy on the given (already recentered) trials, averaged over sources."""
        y = self.y[tgt][j] if idx is None else self.y[tgt][j][idx]
        return float(np.mean([np.mean(self.models[s].predict(Ct) == y) for s in self.sources(tgt)]))


# =========================== [G] levels ===========================
def section_g(mont):
    rows = []
    for M in mont:
        t0 = time.time()
        for tgt in M.names:
            for j, C in enumerate(M.C[tgt]):
                acc = {"none": M.accuracy(tgt, j, C),
                       "site": M.accuracy(tgt, j, apply_ref(M.site[tgt][j], C)),
                       "own": M.accuracy(tgt, j, apply_ref(M.R[tgt][j], C))}
                acc["source"] = float(np.mean([np.mean(M.models[s].predict(apply_ref(M.centre[s], C))
                                                       == M.y[tgt][j]) for s in M.sources(tgt)]))
                rows.append((M.name, tgt, M.subj[tgt][j], acc["none"], acc["source"], acc["site"], acc["own"]))
        print(f"    [G] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[G] Level of the reference used to recenter the target subject (TS-LR, mean accuracy %)")
    out("    none: no recentering | source: centre of the source dataset | site: mean of the other subjects of the")
    out("    target dataset (zero trials of the subject) | own: the subject's own trials (main analysis)")
    for M in mont:
        sel = [r for r in rows if r[0] == M.name]
        out(f"    {M.name}")
        out(f"      {'target':<11}{'n':>4}{'none':>8}{'source':>8}{'site':>8}{'own':>8}{'site share of gain':>21}")
        for g in M.names + ["all"]:
            x = np.array([r[3:] for r in sel if g == "all" or r[1] == g])
            m = 100 * x.mean(0)
            share = (m[2] - m[0]) / (m[3] - m[0]) if m[3] != m[0] else float("nan")
            out(f"      {g:<11}{len(x):>4}{m[0]:>8.2f}{m[1]:>8.2f}{m[2]:>8.2f}{m[3]:>8.2f}{100 * share:>20.0f}%")
        x = np.array([r[3:] for r in sel])
        out(f"      site - none    {S5.fmt(S5.paired(x[:, 2], x[:, 0]))}")
        out(f"      own - site     {S5.fmt(S5.paired(x[:, 3], x[:, 2]))}")
        out(f"      source - none  {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}")
    out()
    with open(os.path.join(OUTDIR, "step8_levels.csv"), "w", encoding="utf-8") as f:
        f.write("montage,target,subject,acc_none,acc_source_centre,acc_site_prior,acc_own\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f}\n")
    return rows


# =========================== [H] shrinkage ===========================
def section_h(mont, libs, levels):
    mean_riemann = libs[2]
    rows = []                         # (montage, target, subject, k, plain, shrunk_site, shrunk_source)
    for mi, M in enumerate(mont):
        t0 = time.time()
        for ti, tgt in enumerate(M.names):
            srcs = M.sources(tgt)
            for j, C in enumerate(M.C[tgt]):
                n, y = len(C), M.y[tgt][j]
                rng = np.random.default_rng([8, mi, ti, int(M.subj[tgt][j])])
                for k in K_LIST:
                    if n - k < MIN_REST:
                        continue
                    a_site = k / (k + M.k0_site[tgt][j])
                    acc = np.zeros(3)
                    for _ in range(N_DRAWS):
                        idx = rng.choice(n, size=k, replace=False)
                        rest = np.setdiff1d(np.arange(n), idx)
                        Rk = C[idx][0] if k == 1 else mean_riemann(C[idx])
                        acc[0] += M.accuracy(tgt, j, apply_ref(Rk, C[rest]), rest)
                        acc[1] += M.accuracy(tgt, j, apply_ref(geodesic(M.site[tgt][j], Rk, a_site), C[rest]), rest)
                        part = []
                        for s in srcs:               # the source-centre prior depends on the source
                            a = k / (k + M.k0_src[s])
                            Ct = apply_ref(geodesic(M.centre[s], Rk, a), C[rest])
                            part.append(np.mean(M.models[s].predict(Ct) == y[rest]))
                        acc[2] += float(np.mean(part))
                    acc /= N_DRAWS
                    rows.append((M.name, tgt, M.subj[tgt][j], k, acc[0], acc[1], acc[2]))
        print(f"    [H] {M.name}: done ({time.time() - t0:.0f} s)", flush=True)
    out("[H] Reference from k unlabeled trials: plain mean versus shrinkage towards a prior (TS-LR, accuracy % on the")
    out("    other trials of the subject). Shrinkage fraction k / (k + k0); k0 estimated without labels.")
    for M in mont:
        sel = [r for r in rows if r[0] == M.name]
        lev = {(r[1], r[2]): r for r in levels if r[0] == M.name}
        subs = sorted({(r[1], r[2]) for r in sel})
        none = np.array([lev[s][3] for s in subs])
        site0 = np.array([lev[s][5] for s in subs])
        own = np.array([lev[s][6] for s in subs])
        k0 = np.concatenate([M.k0_site[d] for d in M.names])
        out(f"    {M.name}: {len(subs)} target subjects")
        out(f"      k0 with the site prior: median {np.median(k0):.2f} (range {k0.min():.2f} to {k0.max():.2f})"
            " | k0 with the source prior: " + ", ".join(f"{d} {M.k0_src[d]:.2f}" for d in M.names))
        out(f"      {'k':>5}{'plain':>8}{'shrunk,':>10}{'shrunk,':>10}   {'shrunk (site) - plain  [95% CI]  dz  p':<48}")
        out(f"      {'':>5}{'':>8}{'site':>10}{'source':>10}")
        out(f"      {0:>5}{100 * none.mean():>8.2f}{100 * site0.mean():>10.2f}{'':>10}   (k = 0: no trials of the subject)")
        for k in K_LIST:
            d = {(r[1], r[2]): r for r in sel if r[3] == k}
            keys = [s for s in subs if s in d]
            if len(keys) < 3:
                continue
            x = np.array([d[s][4:] for s in keys])
            out(f"      {k:>5}{100 * x[:, 0].mean():>8.2f}{100 * x[:, 1].mean():>10.2f}{100 * x[:, 2].mean():>10.2f}"
                f"   {S5.fmt(S5.paired(x[:, 1], x[:, 0]))}")
        out(f"      {'all':>5}{100 * own.mean():>8.2f}   (every trial, main analysis)")
        target90 = none.mean() + 0.9 * (own.mean() - none.mean())
        for label, col in (("plain", 4), ("shrunk, site", 5), ("shrunk, source", 6)):
            reach = [k for k in K_LIST
                     if np.mean([r[col] for r in sel if r[3] == k and (r[1], r[2]) in set(subs)]) >= target90
                     and len([r for r in sel if r[3] == k]) == len(subs)]
            out(f"      smallest k with 90% of the full gain, {label}: " + (str(min(reach)) if reach else "not reached"))
        if site0.mean() >= target90:
            out("      the site prior alone (k = 0) already gives 90% of the full gain")
    out()
    with open(os.path.join(OUTDIR, "step8_shrinkage.csv"), "w", encoding="utf-8") as f:
        f.write("montage,target,subject,k,acc_plain,acc_shrunk_site,acc_shrunk_source\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f}\n")


# =========================== [I] label-free estimate of accuracy ===========================
def _auc(score, positive):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def section_i(mont):
    from scipy.stats import pearsonr, spearmanr
    rows = []                         # (montage, target, subject, acc, confidence, imbalance)
    for M in mont:
        for tgt in M.names:
            for j, C in enumerate(M.C[tgt]):
                Ct = apply_ref(M.R[tgt][j], C)
                acc, conf, imb = [], [], []
                for s in M.sources(tgt):
                    m = M.models[s]
                    p = m.m.predict_proba(m.ts(Ct, m.ref))[:, 1]
                    acc.append(np.mean((p >= 0.5).astype(int) == M.y[tgt][j]))
                    conf.append(np.mean(np.maximum(p, 1 - p)))
                    imb.append(abs(np.mean(p >= 0.5) - 0.5))
                rows.append((M.name, tgt, M.subj[tgt][j], float(np.mean(acc)), float(np.mean(conf)),
                             float(np.mean(imb))))
    out("[I] Label-free indicator of transfer accuracy: mean confidence of the frozen classifier on the unlabeled trials")
    out("    of a subject (primary configuration, averaged over sources) against the true accuracy of the subject")
    for M in mont:
        x = np.array([r[3:] for r in rows if r[0] == M.name])
        acc, conf, imb = x[:, 0], x[:, 1], x[:, 2]
        good = acc >= 0.70
        out(f"    {M.name}: n = {len(acc)} | Pearson r = {pearsonr(conf, acc)[0]:+.2f} | Spearman rho = "
            f"{spearmanr(conf, acc)[0]:+.2f} | mean confidence {100 * conf.mean():.1f}%, mean accuracy {100 * acc.mean():.1f}%")
        out(f"      mean absolute error of confidence as an estimate of accuracy: {100 * np.mean(np.abs(conf - acc)):.1f} points")
        out(f"      AUC for detecting subjects with accuracy >= 70%: {_auc(conf, good):.2f} ({int(good.sum())} of {len(acc)} subjects)")
        out(f"      class imbalance of the predictions against accuracy: Spearman rho = {spearmanr(imb, acc)[0]:+.2f}")
        for tgt in M.names:
            z = np.array([r[3:] for r in rows if r[0] == M.name and r[1] == tgt])
            if len(z) >= 20:
                out(f"      {tgt:<11} n = {len(z):<4} Spearman rho (confidence, accuracy) = {spearmanr(z[:, 1], z[:, 0])[0]:+.2f}")
    out()
    with open(os.path.join(OUTDIR, "step8_confidence.csv"), "w", encoding="utf-8") as f:
        f.write("montage,target,subject,accuracy,mean_confidence,prediction_imbalance\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f}\n")


def run():
    t_start = time.time()
    libs = S5._libs()
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 8 new analyses  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    data = load_cache()
    out("subjects: " + ", ".join(f"{d} {len(data[d])}" for d in S5.DATASETS))
    out()
    mont = []
    for m in S5.MONTAGES:
        t0 = time.time()
        mont.append(Montage(data, m, libs))
        print(f"    prepared {m} ({time.time() - t0:.0f} s)", flush=True)
    # consistency with step 5: the 'own' level must reproduce the primary configuration
    levels = section_g(mont)
    csv = os.path.join(OUTDIR, "benchmark_subjects.csv")
    if os.path.exists(csv):
        ref = {}
        with open(csv, encoding="utf-8") as f:
            f.readline()
            for line in f:
                m, s, t, sub, cov, al, clf, a = line.strip().split(",")
                if (cov, al, clf) == ("tn", "ra_ps", "lr"):
                    ref.setdefault((m, t, int(sub)), []).append(float(a))
        diff = [abs(r[6] - np.mean(ref[(r[0], r[1], r[2])])) for r in levels if (r[0], r[1], r[2]) in ref]
        out(f"check against step 5 (level 'own' versus the primary configuration): largest difference "
            f"{100 * max(diff):.2f} points over {len(diff)} subjects"
            + ("" if max(diff) < 0.005 else "   <-- does NOT match step 5: stop and report this"))
        out()
    section_h(mont, libs, levels)
    section_i(mont)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step8_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    print(f"written: {os.path.abspath(OUTDIR)} (step8_summary.txt and three CSV files)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
