"""
Multi-variant within-subject benchmark across all datasets.

Methods:
  classical baselines : CSP+LDA, FBCSP+LDA (no torch, fast)
  proposed variants   : CSPPriorNet with different init / capacity / augmentation
    base   : CSP-init,   K=8,  F=16, kt=32, aug=3
    K16    : CSP-init,   K=16, F=16, kt=32, aug=3   (more spatial filters)
    wideF  : CSP-init,   K=8,  F=32, kt=32, aug=3   (wider temporal)
    aug5   : CSP-init,   K=8,  F=16, kt=32, aug=5   (more augmentation)
    random : random-init K=8,  F=16, kt=32, aug=3   (ablation: is the CSP prior needed?)

Datasets: 2a (4-class, 22 ch), 2b (2-class, 3 ch), PhysioNet and HGD (2-class
LH/RH on a sensorimotor montage; subject subset for tractability).

Every (dataset, method, subject) result is appended to results/master_within.csv
immediately and skipped on restart, so an interruption never loses work.

    python variants.py
"""
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score
from mne.decoding import CSP
import within_loader

try:
    import config
    _RESULTS = config.RESULTS
except Exception:
    config = None
    _RESULTS = "."

def _seed_all(seed=42):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _zscore(X):
    """Per-trial z-score; deep nets (EEGNet-family) need normalized input."""
    return ((X - X.mean(axis=-1, keepdims=True)) /
            (X.std(axis=-1, keepdims=True) + 1e-7)).astype(np.float32)


SM = ["FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4",
      "C6", "CP3", "CP1", "CPZ", "CP2", "CP4"]      # sensorimotor montage for PN/HGD

# Fair shared protocol for every deep method (variants + braindecode baselines).
EPOCHS, PATIENCE, AUG = 500, 75, 3.0
FBCSP_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (8, 30)]

VARIANTS = {
    "base":   dict(init="csp", K=8,  F=16, kt=32, aug=3.0),
    "K16":    dict(init="csp", K=16, F=16, kt=32, aug=3.0),
    "wideF":  dict(init="csp", K=8,  F=32, kt=32, aug=3.0),
    "aug5":   dict(init="csp", K=8,  F=16, kt=32, aug=5.0),
    "random": dict(init="random", K=8, F=16, kt=32, aug=3.0),
}
# braindecode deep baselines: available on request, but NOT run by default within
# subject. A generic harness underfits/collapses several of them (Deep4Net,
# EEGConformer -> chance); for a fair comparison their PUBLISHED 2a/2b numbers are
# cited instead. Pass methods=BD_MODELS explicitly to run them anyway.
BD_MODELS = ["EEGNetv4", "ShallowFBCSPNet", "Deep4Net", "EEGConformer",
             "ATCNet", "EEGITNet", "EEGInception", "EEGInceptionMI", "TCN"]
DEEP_VARIANTS = ["base", "K16", "wideF", "aug5", "random", "FBCSPNet"]
DATASET_NCLS = {"2a": 4, "2b": 2, "PhysioNet": 2, "HGD": 2}


def load_within(dataset, subj):
    if dataset in ("2a", "2b"):
        return within_loader.load_subject(dataset, subj, zscore=False)
    import crossdata
    if dataset == "PhysioNet":
        return crossdata.load_physionet_subject(config.DATA_PHYSIONET, subj, common=SM)
    return crossdata.load_hgd_subject(config.DATA_HGD, subj, common=SM)


def subjects(dataset, pn_n=20, hgd_n=14):
    if dataset in ("2a", "2b"):
        return list(range(1, 10))
    import crossdata
    if dataset == "PhysioNet":
        return [s for s in range(1, 110) if s not in crossdata.PN_BAD][:pn_n]
    return list(range(1, 53))[:hgd_n]


# ---------------- classical baselines ----------------

def csp_lda(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    a = []
    for tr, te in skf.split(X, y):
        c = CSP(n_components=8, reg="ledoit_wolf", log=True)
        lda = LinearDiscriminantAnalysis().fit(c.fit_transform(X[tr], y[tr]), y[tr])
        a.append(accuracy_score(y[te], lda.predict(c.transform(X[te]))))
    return float(np.mean(a))


def fbcsp_lda(dataset, subj):
    bands = []
    for lo, hi in FBCSP_BANDS:
        if dataset in ("2a", "2b"):
            Xb, y = within_loader.load_subject(dataset, subj, l_freq=lo, h_freq=hi, zscore=False)
        else:
            import crossdata
            loader = (crossdata.load_physionet_subject if dataset == "PhysioNet"
                      else crossdata.load_hgd_subject)
            root = config.DATA_PHYSIONET if dataset == "PhysioNet" else config.DATA_HGD
            Xb, y = loader(root, subj, common=SM, l_freq=lo, h_freq=hi)
        bands.append(Xb)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    a = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        ftr, fte = [], []
        for Xb in bands:
            c = CSP(n_components=4, reg="ledoit_wolf", log=True)
            ftr.append(c.fit_transform(Xb[tr], y[tr])); fte.append(c.transform(Xb[te]))
        lda = LinearDiscriminantAnalysis().fit(np.concatenate(ftr, 1), y[tr])
        a.append(accuracy_score(y[te], lda.predict(np.concatenate(fte, 1))))
    return float(np.mean(a))


# ---------------- deep variant (torch) ----------------

def cspnet_cv(X, y, n_classes, n_ch, cfg, epochs=EPOCHS):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from csp_prior_net import build_cspprior
    from augment import augment_train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_all()
    # CSP-based methods use RAW signal (CSP needs raw-covariance structure)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        filt = None
        if cfg["init"] == "csp":
            c = CSP(n_components=min(cfg["K"], n_ch), reg="ledoit_wolf", log=True)
            c.fit(Xtr, ytr); filt = c.filters_
        ti, vi = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                  stratify=ytr, random_state=42)
        Xva, yva = Xtr[vi], ytr[vi]
        Xa, ya = augment_train(Xtr[ti], ytr[ti], factor=cfg["aug"], seed=42)
        Tn = min(Xa.shape[2], Xva.shape[2], Xte.shape[2])
        Xa, Xva2, Xte2 = Xa[:, :, :Tn], Xva[:, :, :Tn], Xte[:, :, :Tn]

        def t(a, d):
            return torch.tensor(np.asarray(a), dtype=d)

        loader = DataLoader(TensorDataset(t(Xa, torch.float32), t(ya, torch.long)),
                            batch_size=16, shuffle=True, drop_last=True)
        model = build_cspprior(n_classes, n_ch, filt, K=cfg["K"], init=cfg["init"],
                               F=cfg["F"], kt=cfg["kt"]).to(device)
        torch.manual_seed(42)
        with torch.no_grad():
            model(t(Xa[:2], torch.float32).to(device))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()
        Xva_t, Xte_t = t(Xva2, torch.float32).to(device), t(Xte2, torch.float32).to(device)
        best, bs, wait = -1.0, None, 0
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); ce(model(xb), yb).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va = accuracy_score(yva, model(Xva_t).argmax(1).cpu().numpy())
            if va > best:
                best, bs, wait = va, {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        if bs is not None:
            model.load_state_dict(bs)
        model.eval()
        with torch.no_grad():
            accs.append(accuracy_score(yte, model(Xte_t).argmax(1).cpu().numpy()))
    return float(np.mean(accs))


def bd_cv(X, y, n_classes, n_ch, name, epochs=EPOCHS):
    """Train a braindecode model under the same within-subject 5-fold protocol."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from benchmark_models import build_model, _flatten, _pick_loss
    from augment import augment_train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_all()
    X = _zscore(X)                                  # deep nets need normalized input
    n_times = X.shape[2]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        ti, vi = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                  stratify=ytr, random_state=42)
        Xva, yva = Xtr[vi], ytr[vi]
        Xa, ya = augment_train(Xtr[ti], ytr[ti], factor=3.0, seed=42)
        Tn = min(Xa.shape[2], Xva.shape[2], Xte.shape[2])
        Xa, Xva2, Xte2 = Xa[:, :, :Tn], Xva[:, :, :Tn], Xte[:, :, :Tn]

        def t(a, d):
            return torch.tensor(np.asarray(a), dtype=d)

        loader = DataLoader(TensorDataset(t(Xa, torch.float32), t(ya, torch.long)),
                            batch_size=16, shuffle=True, drop_last=True)
        torch.manual_seed(42)
        model = build_model(name, n_ch, n_classes, Tn).to(device)
        crit = _pick_loss(model, t(Xa[:2], torch.float32).to(device))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xva_t, Xte_t = t(Xva2, torch.float32).to(device), t(Xte2, torch.float32).to(device)
        best, bs, wait = -1.0, None, 0
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); crit(_flatten(model(xb)), yb).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va = accuracy_score(yva, _flatten(model(Xva_t)).argmax(1).cpu().numpy())
            if va > best:
                best, bs, wait = va, {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        if bs is not None:
            model.load_state_dict(bs)
        model.eval()
        with torch.no_grad():
            accs.append(accuracy_score(yte, _flatten(model(Xte_t)).argmax(1).cpu().numpy()))
    return float(np.mean(accs))


def load_bands(dataset, subj):
    """Return (X_stacked, y): X_stacked is (n, n_bands, C, T) of band-filtered
    signals, sharing labels."""
    Xb, y = [], None
    for lo, hi in FBCSP_BANDS:
        if dataset in ("2a", "2b"):
            X, y = within_loader.load_subject(dataset, subj, l_freq=lo, h_freq=hi, zscore=False)
        else:
            import crossdata
            loader = (crossdata.load_physionet_subject if dataset == "PhysioNet"
                      else crossdata.load_hgd_subject)
            root = config.DATA_PHYSIONET if dataset == "PhysioNet" else config.DATA_HGD
            X, y = loader(root, subj, common=SM, l_freq=lo, h_freq=hi)
        Xb.append(X)
    T = min(x.shape[2] for x in Xb)
    return np.stack([x[:, :, :T] for x in Xb], axis=1).astype(np.float32), y


def fbcspnet_cv(Xs, y, n_classes, n_ch, epochs=EPOCHS):
    """Xs: (n, n_bands, C, T). Per fold: CSP filters per band -> FBCSPNet init -> train."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from fbcsp_net import build_fbcspnet
    from augment import augment_train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_all()
    n_bands = Xs.shape[1]     # FBCSPNet uses RAW band-filtered signal (CSP-based)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        Xtr, ytr, Xte, yte = Xs[tr], y[tr], Xs[te], y[te]
        band_filters = []
        for b in range(n_bands):
            c = CSP(n_components=min(4, n_ch), reg="ledoit_wolf", log=True)
            c.fit(Xtr[:, b], ytr); band_filters.append(c.filters_)
        ti, vi = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                  stratify=ytr, random_state=42)
        Xva, yva = Xtr[vi], ytr[vi]
        # augment on flattened bands, then reshape back
        nb, C, T = Xtr.shape[1], Xtr.shape[2], Xtr.shape[3]
        flat = Xtr[ti].reshape(len(ti), nb * C, T)
        fa, ya = augment_train(flat, ytr[ti], factor=3.0, seed=42)
        Xa = fa.reshape(len(ya), nb, C, fa.shape[2])
        Tn = min(Xa.shape[3], Xva.shape[3], Xte.shape[3])
        Xa, Xva2, Xte2 = Xa[..., :Tn], Xva[..., :Tn], Xte[..., :Tn]

        def t(a, d):
            return torch.tensor(np.asarray(a), dtype=d)

        loader = DataLoader(TensorDataset(t(Xa, torch.float32), t(ya, torch.long)),
                            batch_size=16, shuffle=True, drop_last=True)
        torch.manual_seed(42)
        model = build_fbcspnet(n_classes, n_ch, band_filters, K=4, Fdim=16, kt=32).to(device)
        with torch.no_grad():
            model(t(Xa[:2], torch.float32).to(device))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()
        Xva_t, Xte_t = t(Xva2, torch.float32).to(device), t(Xte2, torch.float32).to(device)
        best, bs, wait = -1.0, None, 0
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); ce(model(xb), yb).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va = accuracy_score(yva, model(Xva_t).argmax(1).cpu().numpy())
            if va > best:
                best, bs, wait = va, {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        if bs is not None:
            model.load_state_dict(bs)
        model.eval()
        with torch.no_grad():
            accs.append(accuracy_score(yte, model(Xte_t).argmax(1).cpu().numpy()))
    return float(np.mean(accs))


# ---------------- driver ----------------

def _path():
    return os.path.join(_RESULTS, "master_within.csv")


def _done():
    p = _path()
    if not os.path.exists(p):
        return set()
    out = set()
    for l in open(p).read().splitlines()[1:]:
        if l.strip():
            d, m, s = l.split(",")[:3]
            out.add((d, m, s))
    return out


def _append(dataset, method, subj, acc):
    os.makedirs(_RESULTS, exist_ok=True)
    p = _path()
    if not os.path.exists(p):
        open(p, "w").write("dataset,method,subject,acc\n")
    open(p, "a").write(f"{dataset},{method},{subj},{acc:.4f}\n")


def run(datasets=("2a", "2b", "PhysioNet", "HGD"), methods=None, epochs=EPOCHS):
    methods = methods or (["CSP", "FBCSP"] + DEEP_VARIANTS)
    done = _done()
    for ds in datasets:
        n_cls = DATASET_NCLS[ds]
        for subj in subjects(ds):
            try:
                X, y = load_within(ds, subj)
            except Exception as e:
                print(f"  {ds} S{subj}: load failed ({e})"); continue
            n_ch = X.shape[1]
            Xs = None
            for m in methods:
                if (ds, m, str(subj)) in done:
                    continue
                try:
                    if m == "CSP":
                        acc = csp_lda(X, y)
                    elif m == "FBCSP":
                        acc = fbcsp_lda(ds, subj)
                    elif m == "FBCSPNet":
                        if Xs is None:
                            Xs, _ = load_bands(ds, subj)
                        acc = fbcspnet_cv(Xs, y, n_cls, n_ch, epochs=epochs)
                    elif m in VARIANTS:
                        acc = cspnet_cv(X, y, n_cls, n_ch, VARIANTS[m], epochs=epochs)
                    else:
                        acc = bd_cv(X, y, n_cls, n_ch, m, epochs=epochs)
                    _append(ds, m, subj, acc)
                    print(f"  {ds:<10} S{str(subj):<3} {m:<16} acc={acc*100:6.2f}", flush=True)
                except Exception as e:
                    print(f"  {ds} S{subj} {m} FAILED: {e}", flush=True)
    summarize()


def summarize():
    p = _path()
    if not os.path.exists(p):
        print("no results yet"); return
    rows = [l.split(",") for l in open(p).read().splitlines()[1:] if l.strip()]
    methods = ["CSP", "FBCSP"] + DEEP_VARIANTS + BD_MODELS
    datasets = sorted({r[0] for r in rows})
    print("\n=== Master within-subject table (mean accuracy %) ===")
    print("  method".ljust(18) + "".join(f"{d:>12}" for d in datasets))
    for m in methods:
        line = f"  {m:<16}"
        for d in datasets:
            vals = [float(r[3]) for r in rows if r[0] == d and r[1] == m]
            line += (f"{np.mean(vals)*100:>12.2f}" if vals else f"{'-':>12}")
        print(line)


def main():
    run()


if __name__ == "__main__":
    main()
