"""
step16_baselines.py

All baselines in one run, one environment and one protocol.

Run from the Codes folder, after step 15 has finished (60 to 90 minutes with the
neural part; 20 to 30 minutes without):

    python step16_baselines.py
    python step16_baselines.py --no-deep
    python step16_baselines.py "D:\\other\\folder\\BCI Database"       (if the fifth dataset is elsewhere)

Protocol (identical for every method). A method is fitted on the labeled trials
of ONE source dataset and applied to every subject of another dataset; it may
use the unlabeled trials of that subject and nothing else. A subject's score is
the mean over source datasets. Targets: the four benchmark datasets (sources:
the other benchmark datasets) and Dreyer2023 (sources: the four benchmark
datasets). Same preprocessing for all: 8-30 Hz, 0.5-2.5 s, OAS covariance,
trace normalization; the neural networks see the same band-passed epochs.

Methods
  no alignment            MDM | tangent-space logistic regression (TS-LR) | CSP + LDA | EEGNet | ShallowConvNet
  EA                      Euclidean alignment (He and Wu 2020): CSP + LDA | EEGNet | ShallowConvNet
  RA                      Riemannian recentering on the task trials (Zanini et al. 2018): MDM | TS-LR (the
                          primary pipeline of this study)
  RPA (unsupervised)      recentering and rescaling (Rodrigues et al. 2019, without the supervised rotation):
                          MDM | TS-LR
  CA + sLDA, CA + CORAL   centroid alignment, tangent features, shrinkage LDA; with CORAL (Sun et al. 2016)
  MEKT-R                  manifold embedded knowledge transfer (Zhang and Wu 2020), ported from the authors'
                          MATLAB code with their default parameters (d=10, T=5, alpha=0.01, beta=0.1, rho=20)
  rest, dual              recentering on the pre-cue baselines; geodesic midpoint of task and rest reference
                          (this study), both with the frozen TS-LR

  The ports of CA + sLDA, CA + CORAL, MEKT-R and RA + MDM were checked on the demo data distributed with
  the MEKT code (BCI-IV-2a, multi-source): 71.91, 72.45, 76.23 and 72.07 % against 71.84, 72.38, 76.31
  and 72.07 % in the published table.

Parts
  [B1] all trials of the target subject (classes balanced): accuracy
  [B2] class-imbalanced unlabeled target data (one class 7 times as frequent as the other):
       balanced accuracy on those trials, next to the balanced accuracy with all trials
  [B3] the neural networks under the same two settings (GPU if PyTorch sees one)

Outputs in corrected_results/: step16_summary.txt, step16_baselines.csv, refreshed results_bundle.zip.
"""
from __future__ import annotations

import os
import re
import sys
import glob
import time
import argparse
import datetime

import numpy as np

try:
    import step5_benchmark as S5
    import step8_new_analyses as S8
    import step9_reviewer_checks as S9
    import step11_directions as S11
    import step12_rest_anchored as S12
    import step13_adaptive_reference as S13
    import step15_external_validation as S15
except Exception as _e:                                    # pragma: no cover
    print(f"could not import the earlier step scripts (5, 8 to 15); run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
out = S5.out
tn = S8.tn
EXT = S15.NAME
EPOCHS_EXT = os.path.join(OUTDIR, "cache_dreyer_epochs")
RATIO = 7
COV_METHODS = [("none_mdm", "No alignment, MDM"), ("none_tslr", "No alignment, TS-LR"), ("none_csp", "No alignment, CSP + LDA"),
               ("ea_csp", "EA, CSP + LDA"), ("ra_mdm", "RA, MDM"), ("ra_tslr", "RA, TS-LR (primary)"),
               ("rpa_mdm", "RPA unsupervised, MDM"), ("rpa_tslr", "RPA unsupervised, TS-LR"),
               ("ca_slda", "CA, shrinkage LDA"), ("ca_coral", "CA + CORAL"), ("mekt_r", "MEKT-R"),
               ("rest_tslr", "Rest reference, TS-LR"), ("dual_tslr", "Dual reference, TS-LR (this study)")]
DEEP_METHODS = [("eegnet_none", "EEGNet, no alignment"), ("eegnet_ea", "EEGNet, EA"),
                ("shallow_none", "ShallowConvNet, no alignment"), ("shallow_ea", "ShallowConvNet, EA")]
RES = {}                                                   # (setting, montage, method, target, subject) -> [(acc, bacc), ...]


def add(setting, montage, method, target, subject, pred, y):
    pred = np.asarray(pred).astype(int)
    RES.setdefault((setting, montage, method, target, subject), []).append(
        (float(np.mean(pred == y)), float(S11.bal_acc(pred, y))))


# =========================== ports of published methods ===========================
def slda_fit(Zs, ys):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Zs, ys)


def knn_heat_laplacian(X, k=10, t=1.0):
    """Graph of the MEKT code: k nearest neighbours and the point itself, weights exp(-d^2/(2 t^2)), symmetrized."""
    sq = np.sum(X ** 2, axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    k = min(k, len(X) - 1)
    idx = np.argsort(D2, axis=1)[:, :k + 1]
    G = np.zeros_like(D2)
    rows = np.repeat(np.arange(len(X)), idx.shape[1])
    G[rows, idx.ravel()] = np.exp(-D2[rows, idx.ravel()] / (2.0 * t ** 2))
    W = np.maximum(G, G.T)
    return np.diag(W.sum(1)) - W


class MektSource:
    def __init__(self, Xs, ys):
        self.Xs, self.ys, self.classes = Xs, ys, np.unique(ys)
        m, mean_tot = Xs.shape[1], Xs.mean(0)
        self.Sw, self.Sb = np.zeros((m, m)), np.zeros((m, m))
        for c in self.classes:
            Xi = Xs[ys == c]
            mc = Xi.mean(0)
            self.Sw += (Xi - mc).T @ (Xi - mc)
            self.Sb += len(Xi) * np.outer(mc - mean_tot, mc - mean_tot)
        self.Us = Xs.T @ ((ys[:, None] == self.classes[None, :]).astype(float) / len(ys))


def mekt(src, Xt, d=10, T=5, alpha=0.01, beta=0.1, rho=20.0):
    from scipy.linalg import eigh
    Xs, ys, m, nt = src.Xs, src.ys, src.Xs.shape[1], len(Xt)
    d = min(d, m)
    Z = np.zeros((m, m))
    P = np.block([[src.Sw, Z], [Z, Z]])
    P0 = np.block([[src.Sb, Z], [Z, Z]])
    L = np.block([[Z, Z], [Z, Xt.T @ knn_heat_laplacian(Xt) @ Xt]])
    Q = np.block([[np.eye(m), -np.eye(m)], [-np.eye(m), 2 * np.eye(m)]])
    Xc = Xt - Xt.mean(0)
    S = np.block([[Z, Z], [Z, Xc.T @ Xc]])
    yt, Zs, Zt = None, None, None
    for it in range(1, T + 1):
        Ut = np.zeros_like(src.Us) if yt is None else Xt.T @ ((yt[:, None] == src.classes[None, :]).astype(float) / nt)
        U = np.vstack([src.Us, -Ut])
        Emin = alpha * P + beta * L + rho * Q + U @ U.T + 1e-3 * np.eye(2 * m)
        _, V = eigh(S + alpha * P0, Emin)                  # largest here = smallest of the pencil of the MATLAB code
        W = V[:, -d:]
        Zs, Zt = Xs @ W[:m], Xt @ W[m:]
        if it > 1:
            yt = slda_fit(Zs, ys).predict(Zt)
    return slda_fit(Zs, ys).predict(Zt)


def rescale(C):
    """RPA stretching: matrices already recentered are raised to the power that gives unit dispersion."""
    w, V = np.linalg.eigh(C)
    lw = np.log(w)
    p = 1.0 / np.sqrt(np.mean(np.sum(lw ** 2, axis=1)))
    return (V * np.exp(p * lw)[:, None, :]) @ np.swapaxes(V, 1, 2)


# =========================== covariance-based methods ===========================
class SourceModels:
    def __init__(self, M, s, libs):
        tangent = libs[3]
        n_pairs = 1 if M.name == "3ch" else 3
        T, ys = M.T[s], np.concatenate(M.y[s])
        none = np.concatenate(T)
        ea = np.concatenate([S8.apply_ref(c.mean(0), c) for c in T])
        ra_list = [S8.apply_ref(R, c) for R, c in zip(M.Rt[s], T)]
        ra = np.concatenate(ra_list)
        rpa = np.concatenate([rescale(c) for c in ra_list])
        self.eye = np.eye(none.shape[1])
        self.mdm_none, self.lr_none = S5.Model(none, ys, "mdm", libs), S5.Model(none, ys, "lr", libs)
        self.csp_none, self.csp_ea = S9.CspLda(none, ys, n_pairs), S9.CspLda(ea, ys, n_pairs)
        self.mdm_ra, self.lr_ra = S5.Model(ra, ys, "mdm", libs), M.task_model[s]
        self.mdm_rpa, self.lr_rpa = S5.Model(rpa, ys, "mdm", libs), S5.Model(rpa, ys, "lr", libs)
        self.Xs, self.ys = tangent(ra, self.eye), ys
        self.lda = slda_fit(self.Xs, ys)
        self.mekt = MektSource(self.Xs, ys)
        self.coral_white = self.Xs @ S8.invsqrtm(np.cov(self.Xs, rowvar=False) + np.eye(self.Xs.shape[1]))


def predict_cov(SM, M, s, T, B, libs):
    """Predictions of every covariance-based method for one unlabeled set (task covariances T, baselines B)."""
    mean_riemann, tangent = libs[2], libs[3]
    Rt, Rb = mean_riemann(T), mean_riemann(B)
    ra = S8.apply_ref(Rt, T)
    rp = rescale(ra)
    Xt = tangent(ra, SM.eye)
    p = {"none_mdm": SM.mdm_none.predict(T), "none_tslr": SM.lr_none.predict(T), "none_csp": SM.csp_none.predict(T),
         "ea_csp": SM.csp_ea.predict(S8.apply_ref(T.mean(0), T)),
         "ra_mdm": SM.mdm_ra.predict(ra), "ra_tslr": SM.lr_ra.predict(ra),
         "rpa_mdm": SM.mdm_rpa.predict(rp), "rpa_tslr": SM.lr_rpa.predict(rp),
         "ca_slda": SM.lda.predict(Xt)}
    Ct = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1])
    p["ca_coral"] = slda_fit(SM.coral_white @ S8.sqrtm(Ct), SM.ys).predict(Xt)
    p["mekt_r"] = mekt(SM.mekt, Xt)
    p["rest_tslr"] = (S13.logits(M, s, Rb, T) > 0).astype(int)
    p["dual_tslr"] = (S13.logits(M, s, S8.geodesic(Rt, Rb, 0.5), T) > 0).astype(int)
    return p


def imbalanced(y, dom, rng):
    major = np.where(y == dom)[0]
    minor = np.where(y != dom)[0]
    k = min(len(minor), max(2, int(round(len(major) / RATIO))))
    return np.sort(np.r_[major, rng.choice(minor, k, replace=False)])


def targets_of(M, G):
    """Every target of a montage: (name, list of subject ids, T, B, y)."""
    res = [(d, M.subj[d], M.T[d], M.B[d], M.y[d]) for d in M.names]
    if G is not None:
        res.append((EXT, G.sid, G.T, G.B, G.y))
    return res


def section_cov(st, tg, libs):
    subsets = {}
    for mi, (M, G) in enumerate(zip(st, tg)):
        t0 = time.time()
        tlist = targets_of(M, G)
        for s in M.names:
            SM = SourceModels(M, s, libs)
            for ti, (tname, ids, Ts, Bs, ys) in enumerate(tlist):
                if tname == s:
                    continue
                for j in range(len(Ts)):
                    T, B, y = Ts[j], Bs[j], ys[j]
                    for m, pred in predict_cov(SM, M, s, T, B, libs).items():
                        add("all", M.name, m, tname, ids[j], pred, y)
                    for dom in (0, 1):
                        key = (M.name, tname, ids[j], dom)
                        if key not in subsets:
                            subsets[key] = imbalanced(y, dom, np.random.default_rng([20, mi, ti, j, dom]))
                        idx = subsets[key]
                        for m, pred in predict_cov(SM, M, s, T[idx], B[idx], libs).items():
                            add("imbalanced", M.name, m, tname, ids[j], pred, y[idx])
            print(f"    [B1/B2] {M.name}: source {s} done ({time.time() - t0:.0f} s)", flush=True)
    return subsets


# =========================== neural networks ===========================
def make_trainer(kind, mock, epochs, device):
    if mock:
        return S11.make_trainer(True, epochs, device)
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

    class Shallow(nn.Module):
        def __init__(self, C, T, F=40, k=25, pool=75, stride=15, p=0.5):
            super().__init__()
            self.temporal = nn.Conv2d(1, F, (1, k))
            self.spatial = nn.Conv2d(F, F, (C, 1), bias=False)
            self.bn = nn.BatchNorm2d(F)
            self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
            self.drop = nn.Dropout(p)
            with torch.no_grad():
                n = self._features(torch.zeros(1, 1, C, T)).numel()
            self.fc = nn.Linear(n, 2)

        def _features(self, x):
            x = self.bn(self.spatial(self.temporal(x)))
            x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
            return self.drop(x).flatten(1)

        def forward(self, x):
            return self.fc(self._features(x.unsqueeze(1)))

    Net = EEGNet if kind == "eegnet" else Shallow

    def train(X, y, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = Net(X.shape[1], X.shape[2]).to(device)
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
                lossf(net(Xt[idx]), yt[idx]).backward()
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


def ext_epochs(sid, folder, mods):
    path = os.path.join(EPOCHS_EXT, f"{sid}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return z["X"], z["y"]
    if folder is None:
        raise RuntimeError("recordings of the fifth dataset not found and no epoch cache")
    mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
    files = sorted(glob.glob(os.path.join(folder, f"{sid}_R*_*.gdf")),
                   key=lambda f: int(re.search(r"_R(\d+)_", os.path.basename(f)).group(1)))
    Xs, ys = [], []
    for f in files:
        try:
            raw = mne.io.read_raw_gdf(f, preload=True, verbose="ERROR")
            X, y = CD._epoch(CD._prep(raw, SM, 8.0, 30.0, 250), {"769": 0, "770": 1}, -2.0, 2.5)
            del raw
        except Exception:
            continue
        if X is None or X.shape[2] < 1126:
            continue
        ok = (np.isfinite(X).all(axis=(1, 2)) & (X[..., :500].var(axis=2).min(axis=1) > 0)
              & (X[..., 625:1126].var(axis=2).min(axis=1) > 0))
        Xs.append(X[ok][..., 625:1126])
        ys.append(np.asarray(y)[ok])
    X, y = np.concatenate(Xs).astype(np.float32), np.concatenate(ys)
    os.makedirs(EPOCHS_EXT, exist_ok=True)
    np.savez(path, X=X, y=y)
    return X, y


def section_deep(st, tg, main, kept, folders, mods, subsets, mock):
    if mock:
        device, epochs, seeds, montages = "mock", 1, [0], ["17ch", "3ch"]
    else:
        import torch
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        epochs, seeds = (60, [0, 1]) if cuda else (20, [0])
        montages = ["17ch", "3ch"] if cuda else ["17ch"]
        out(f"[B3] PyTorch {torch.__version__}, device: {device}"
            + ("" if cuda else "  (no GPU visible to PyTorch: reduced run, 17 channels, one seed, 20 epochs)"))
    loaders = S9.make_loaders((8.0, 30.0), (0.5, 2.5))
    t0 = time.time()
    X, Y, IDS = {}, {}, {}
    for ds in S5.DATASETS:
        X[ds], Y[ds], IDS[ds] = [], [], [r["subj"] for r in main[ds]]
        for r in main[ds]:
            x, y = S11.get_epochs(ds, r["subj"], mods, loaders)
            if not np.array_equal(y, r["y"]):
                raise RuntimeError(f"{ds} subject {r['subj']}: epochs do not match the covariance cache")
            X[ds].append(S11.amp_norm(x))
            Y[ds].append(y)
    if kept:
        X[EXT], Y[EXT], IDS[EXT] = [], [], []
        for r in kept:
            x, y = ext_epochs(r["sid"], folders.get(r["sid"]), mods)
            if not np.array_equal(y, r["y"]):
                raise RuntimeError(f"{EXT} participant {r['sid']}: epochs do not match the covariance cache")
            X[EXT].append(S11.amp_norm(x))
            Y[EXT].append(y)
            IDS[EXT].append(r["sid"])
    print(f"    [B3] epochs ready ({time.time() - t0:.0f} s)", flush=True)
    whiten = lambda P, x: np.einsum("cd,ndt->nct", P, x).astype(np.float32)
    for montage in montages:
        names = [d for d in S5.MONTAGES[montage] if len(X[d]) >= 3]
        tnames = names + ([EXT] if kept else [])
        pick = (lambda d, x: x) if montage == "17ch" else (lambda d, x: x if d == "2b" else x[:, S5.TRIPLET_IN_SM, :])
        Xm = {d: [np.ascontiguousarray(pick(d, x)) for x in X[d]] for d in tnames}
        Xe = {d: [whiten(S11.ea_matrix(x), x) for x in Xm[d]] for d in tnames}
        for kind in ("eegnet", "shallow"):
            train = make_trainer(kind, mock, epochs, device)
            for s in names:
                ys = np.concatenate(Y[s])
                for seed in seeds:
                    f_none = train(np.concatenate(Xm[s]), ys, seed)
                    f_ea = train(np.concatenate(Xe[s]), ys, seed)
                    for t in tnames:
                        if t == s:
                            continue
                        for j, y in enumerate(Y[t]):
                            sid = IDS[t][j]
                            add("all", montage, kind + "_none", t, sid, f_none(Xm[t][j]) > 0, y)
                            add("all", montage, kind + "_ea", t, sid, f_ea(Xe[t][j]) > 0, y)
                            for dom in (0, 1):
                                idx = subsets.get((montage, t, sid, dom))
                                if idx is None:
                                    continue
                                x = Xm[t][j][idx]
                                add("imbalanced", montage, kind + "_none", t, sid, f_none(x) > 0, y[idx])
                                add("imbalanced", montage, kind + "_ea", t, sid, f_ea(whiten(S11.ea_matrix(x), x)) > 0, y[idx])
                    print(f"    [B3] {montage}: {kind}, source {s}, seed {seed} done ({time.time() - t0:.0f} s)", flush=True)


# =========================== report ===========================
def collect(setting, montage, method, targets, col):
    """Per-subject score (mean over sources and repetitions) for the listed targets, with the keys."""
    keys, vals = [], []
    for (st_, mo, me, tg_, sub), v in RES.items():
        if st_ == setting and mo == montage and me == method and tg_ in targets:
            keys.append((tg_, sub))
            vals.append(float(np.mean([x[col] for x in v])))
    order = sorted(range(len(keys)), key=lambda i: (keys[i][0], str(keys[i][1])))
    return [keys[i] for i in order], np.array([vals[i] for i in order])


def report(st, have_ext, methods):
    for M in st:
        bench = list(M.names)
        out(f"[B1] {M.name}: accuracy % with all trials of the target subject (classes balanced), mean over subjects")
        out(f"      {'method':<36}" + "".join(f"{d:>11}" for d in bench) + f"{'benchmark':>11}" + (f"{EXT:>12}" if have_ext else ""))
        kp, prim_b = collect("all", M.name, "ra_tslr", bench, 0)
        _, prim_e = collect("all", M.name, "ra_tslr", [EXT], 0)
        diffs = []
        for key, label in methods:
            kb, vb = collect("all", M.name, key, bench, 0)
            if len(vb) == 0:
                continue
            per = "".join(f"{100 * np.mean([v for (t, _), v in zip(kb, vb) if t == d]):>11.2f}" for d in bench)
            _, ve = collect("all", M.name, key, [EXT], 0)
            out(f"      {label:<36}{per}{100 * vb.mean():>11.2f}" + (f"{100 * ve.mean():>12.2f}" if have_ext and len(ve) else ""))
            if key != "ra_tslr" and kb == kp:
                line = f"      {label:<36}{S5.fmt(S5.paired(vb, prim_b))}"
                if have_ext and len(ve) == len(prim_e) and len(ve):
                    line += f"   |  {S5.fmt(S5.paired(ve, prim_e))}"
                diffs.append(line)
        out(f"      difference from the primary pipeline (RA, TS-LR), in points: benchmark" + (f"   |  {EXT}" if have_ext else ""))
        for line in diffs:
            out(line)
        out()
    for M in st:
        for tset, tname in ((list(M.names), "benchmark"), ([EXT], EXT)):
            if tset == [EXT] and not have_ext:
                continue
            out(f"[B2] {M.name}, {tname}: balanced accuracy % when the unlabeled target trials are balanced (all trials)")
            out(f"      or imbalanced (one class {RATIO} times as frequent as the other); change in points")
            out(f"      {'method':<36}{'balanced':>10}{'imbalanced':>12}   change [95% CI]  dz  p")
            for key, label in methods:
                ka, va = collect("all", M.name, key, tset, 1)
                ki, vi = collect("imbalanced", M.name, key, tset, 1)
                if len(va) == 0 or ka != ki:
                    continue
                out(f"      {label:<36}{100 * va.mean():>10.2f}{100 * vi.mean():>12.2f}   {S5.fmt(S5.paired(vi, va))}")
            ka, va = collect("imbalanced", M.name, "dual_tslr", tset, 1)
            kb, vb = collect("imbalanced", M.name, "ra_tslr", tset, 1)
            if len(va) and ka == kb:
                out(f"      imbalanced data: dual reference - RA, TS-LR     {S5.fmt(S5.paired(va, vb))}")
            out()


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=S15.ROOT)
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--mock-deep", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    t_start = time.time()
    libs = S5._libs()
    if not os.path.isdir(S5.CACHE):
        print(f"{S5.CACHE} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 16 baselines  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    mods = S9.project_modules()
    main = S8.load_cache()
    old = S12.load_states(main, libs)
    st = [S12.States(old, m, libs) for m in S5.MONTAGES]
    kept, folders = [], {}
    if os.path.isdir(S15.CACHE):
        for num, sid, folder in S15.list_subjects(args.root):
            rec = S15.load_subject(sid, folder, mods, libs)
            if int(rec["n_runs"]) >= 4 and min(int((rec["y"] == 0).sum()), int((rec["y"] == 1).sum())) >= 40:
                kept.append(dict(sid=sid, num=num, **rec))
                folders[sid] = folder
    out("subjects: " + ", ".join(f"{d} {len(main[d])}" for d in S5.DATASETS)
        + (f", {EXT} {len(kept)}" if kept else f" | {EXT}: not available (run step 15 first), benchmark only"))
    out()
    tg = [S15.Target(kept, m, libs) if kept else None for m in S5.MONTAGES]
    subsets = section_cov(st, tg, libs)
    methods = list(COV_METHODS)
    if not args.no_deep:
        ok = args.mock_deep
        if not ok:
            try:
                import torch  # noqa: F401
                ok = True
            except Exception as e:
                out(f"[B3] skipped: PyTorch could not be imported ({e!r})")
        if ok:
            n0 = len(RES)
            S9.guard("B3", section_deep, st, tg, main, kept, folders, mods, subsets, args.mock_deep)
            if len(RES) > n0:
                methods += DEEP_METHODS
    report(st, bool(kept), methods)
    rows = [(k[0], k[1], k[2], k[3], k[4], float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
            for k, v in sorted(RES.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    S9.write_csv("step16_baselines.csv", "setting,montage,method,target,subject,accuracy,balanced_accuracy", rows)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step16_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    try:
        import step7_final_checks as S7
        names = S7.make_bundle()
        print(f"bundle refreshed: {os.path.abspath(S7.BUNDLE)} ({len(names)} files) -- attach this one file in the chat")
    except Exception as e:
        print(f"written: {os.path.abspath(OUTDIR)} (step16_summary.txt and step16_baselines.csv); bundle not refreshed ({e!r})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
