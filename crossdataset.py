"""
R3 study: leave-one-dataset-out, zero-calibration cross-dataset MI decoding.

All four datasets are reduced to a common LH-vs-RH task on the shared montage
(C3, Cz, C4), 250 Hz, band 8-30 Hz, window [0.5, 2.5] s. Each subject is
Euclidean-aligned using its own trials (calibration-free). For each target
dataset, an EEGNet is trained on the pooled subjects of the OTHER three datasets
and evaluated on every target subject WITHOUT using any target labels for
training. This is the transfer setting none of the reference papers report.

Chance for LH vs RH is 50%. Torch lives only in train_pooled / eval_target, so
load_all and the leave-one-dataset-out loop import and unit-test without torch
(test_crossdataset.py). Harmonized data is cached to disk to avoid re-parsing
EDF/BDF on every run.
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from baselines import euclidean_align
import crossdata

try:
    import config
    _RESULTS = config.RESULTS
    _CACHE = config.CACHE
    _ROOTS = {"2a": config.DATA_2A, "2b": config.DATA_2B,
              "PhysioNet": config.DATA_PHYSIONET, "HGD": config.DATA_HGD}
except Exception:
    config = None
    _RESULTS = _CACHE = "."
    _ROOTS = {}


def load_all(pn_n=20, hgd_n=20, use_cache=True):
    """{dataset: {subj: (X, y)}} harmonized LH/RH across all four datasets."""
    cache = os.path.join(_CACHE, f"r3_harmonized_pn{pn_n}_hgd{hgd_n}.npz")
    if use_cache and os.path.exists(cache):
        return np.load(cache, allow_pickle=True)["data"].item()

    def collect(loader, root, subj_iter, cap=None, skip=()):
        d = {}
        for s in subj_iter:
            if s in skip:
                continue
            try:
                X, y = loader(root, s)
                d[s] = (X.astype(np.float32), np.asarray(y))
            except Exception:
                continue
            if cap and len(d) >= cap:
                break
        return d

    out = {
        "2a": collect(crossdata.load_2a_subject, _ROOTS["2a"], range(1, 10)),
        "2b": collect(crossdata.load_2b_subject, _ROOTS["2b"], range(1, 10)),
        "PhysioNet": collect(crossdata.load_physionet_subject, _ROOTS["PhysioNet"],
                             range(1, 110), cap=pn_n, skip=crossdata.PN_BAD),
        "HGD": collect(crossdata.load_hgd_subject, _ROOTS["HGD"], range(1, 53), cap=hgd_n),
    }
    print("  [loaded]", " ".join(f"{k}:{len(v)}" for k, v in out.items()))
    if use_cache:
        os.makedirs(_CACHE, exist_ok=True)
        np.savez_compressed(cache, data=np.array(out, dtype=object))
    return out


def _ea_all(data):
    return {s: (euclidean_align(np.asarray(X, np.float64)).astype(np.float32), y)
            for s, (X, y) in data.items()}


def train_pooled(source, *, epochs=150, lr=1e-3, batch_size=64, patience=30,
                 val_frac=0.15, seed=42, device=None):
    """Train EEGNet on pooled EA-aligned source subjects. Returns (model, device)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from models import EEGNet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xs = np.concatenate([X for (X, y) in source.values()], axis=0)
    ys = np.concatenate([y for (X, y) in source.values()], axis=0)
    n_ch = Xs.shape[1]
    tr, va = train_test_split(np.arange(len(ys)), test_size=val_frac,
                              stratify=ys, random_state=seed)

    def T(a, dt):
        return torch.tensor(np.asarray(a), dtype=dt)

    loader = DataLoader(TensorDataset(T(Xs[tr], torch.float32), T(ys[tr], torch.long)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    Xv, yv = T(Xs[va], torch.float32).to(device), ys[va]

    model = EEGNet(2, n_ch).to(device)
    with torch.no_grad():                       # materialize BOTH lazy heads
        e0 = model.embed(T(Xs[:2], torch.float32).to(device))
        model.head(e0)
        model.proj(e0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    best, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            ce(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_acc = accuracy_score(yv, model(Xv).argmax(1).cpu().numpy())
        if va_acc > best:
            best, best_state, wait = va_acc, {k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def eval_target(bundle, target_data):
    """Per-subject (subject, acc, kappa) on the held-out target dataset."""
    import torch
    model, device = bundle
    model.eval()
    out = []
    for s, (X, y) in target_data.items():
        with torch.no_grad():
            pred = model(torch.tensor(np.asarray(X, np.float32)).to(device)).argmax(1).cpu().numpy()
        out.append((s, float(accuracy_score(y, pred)), float(cohen_kappa_score(y, pred))))
    return out


def _prep_align(all_data, align):
    def _ident(v):
        return {s: (np.asarray(X, np.float32), y) for s, (X, y) in v.items()}
    return {k: (_ea_all(v) if align else _ident(v)) for k, v in all_data.items() if v}


def transfer_matrix(all_data=None, align=True, train_fn=None, eval_fn=None,
                    save=True, **kw):
    """Pairwise source->target transfer: train on ONE source dataset, test on each
    OTHER dataset (zero calibration). Returns matrix[src][tgt] = mean target acc.
    Diagonal (same dataset) is skipped (train/test subject overlap)."""
    all_data = all_data if all_data is not None else load_all()
    all_data = _prep_align(all_data, align)
    train_fn = train_fn or train_pooled
    eval_fn = eval_fn or eval_target
    names = list(all_data)
    print(f"  [transfer matrix, align={'EA' if align else 'none'}]")

    M = {}
    for src in names:
        source = {f"{src}_{s}": xy for s, xy in all_data[src].items()}
        bundle = train_fn(source, **kw)
        M[src] = {}
        for tgt in names:
            if tgt == src:
                M[src][tgt] = None
                continue
            per = eval_fn(bundle, all_data[tgt])
            M[src][tgt] = float(np.mean([a for _, a, _ in per]))
        cells = " ".join(f"{t}:{(M[src][t]*100 if M[src][t] is not None else float('nan')):5.1f}"
                         for t in names)
        print(f"  train {src:<10} -> {cells}")
    if save:
        _save_matrix(M, names, align)
    return M


def _save_matrix(M, names, align=True):
    os.makedirs(_RESULTS, exist_ok=True)
    path = os.path.join(_RESULTS, f"r3_transfer_matrix_EA{int(align)}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("train\\test," + ",".join(names) + "\n")
        for src in names:
            row = [("" if M[src][t] is None else f"{M[src][t]:.4f}") for t in names]
            f.write(f"{src}," + ",".join(row) + "\n")
    print(f"\nSaved {path}")
    return path


def train_pooled_dg(source, *, lam=1.0, epochs=150, lr=1e-3, batch_size=128,
                    patience=30, val_frac=0.15, seed=42, device=None):
    """Like train_pooled, plus a CORAL loss with DATASET as the domain. Each
    dataset contributes many samples per batch, so its feature covariance is far
    better conditioned than the per-subject case that failed. source keys are
    'dataset_subject'."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from models import EEGNet
    from dg import coral_loss

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds_names = sorted(set(k.rsplit("_", 1)[0] for k in source))
    ds_id = {d: i for i, d in enumerate(ds_names)}
    Xs, ys, dm = [], [], []
    for k, (X, y) in source.items():
        Xs.append(X); ys.append(np.asarray(y))
        dm.append(np.full(len(y), ds_id[k.rsplit("_", 1)[0]]))
    Xs = np.concatenate(Xs); ys = np.concatenate(ys); dm = np.concatenate(dm)
    n_ch = Xs.shape[1]

    tr, va = train_test_split(np.arange(len(ys)), test_size=val_frac,
                              stratify=ys, random_state=seed)

    def T(a, dt):
        return torch.tensor(np.asarray(a), dtype=dt)

    loader = DataLoader(
        TensorDataset(T(Xs[tr], torch.float32), T(ys[tr], torch.long), T(dm[tr], torch.long)),
        batch_size=batch_size, shuffle=True, drop_last=True)
    Xv, yv = T(Xs[va], torch.float32).to(device), ys[va]

    model = EEGNet(2, n_ch).to(device)
    with torch.no_grad():
        e0 = model.embed(T(Xs[:2], torch.float32).to(device))
        model.head(e0)
        model.proj(e0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    best, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb, db in loader:
            xb, yb, db = xb.to(device), yb.to(device), db.to(device)
            e = model.embed(xb)
            loss = ce(model.head(e), yb)
            if lam > 0 and db.unique().numel() > 1:
                loss = loss + lam * coral_loss(model.proj(e), db)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_acc = accuracy_score(yv, model(Xv).argmax(1).cpu().numpy())
        if va_acc > best:
            best, best_state, wait = va_acc, {k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def leave_one_dataset_out(all_data=None, train_fn=None, eval_fn=None,
                          align=True, dg_lam=0.0, save=True, **kw):
    all_data = all_data if all_data is not None else load_all()
    all_data = _prep_align(all_data, align)
    if train_fn is None:
        train_fn = (lambda src, **k: train_pooled_dg(src, lam=dg_lam, **k)) \
            if dg_lam > 0 else train_pooled
    eval_fn = eval_fn or eval_target
    names = list(all_data)
    print(f"  [align={'EA' if align else 'none'}, dataset-DG lam={dg_lam}]")

    results = {}
    for target in names:
        source = {}
        for ds in names:
            if ds == target:
                continue
            for s, xy in all_data[ds].items():
                source[f"{ds}_{s}"] = xy
        bundle = train_fn(source, **kw)
        per = eval_fn(bundle, all_data[target])
        accs = np.array([a for _, a, _ in per])
        kaps = np.array([k for _, _, k in per])
        results[target] = {"mean_acc": float(accs.mean()), "std_acc": float(accs.std()),
                           "mean_kappa": float(kaps.mean()), "n": len(per)}
        print(f"  target {target:<10} acc={accs.mean()*100:6.2f} "
              f"(±{accs.std()*100:4.2f})  kappa={kaps.mean():.3f}  over {len(per)} subj")
    if save:
        _save(results, align)
    return results


def _save(results, align=True):
    os.makedirs(_RESULTS, exist_ok=True)
    path = os.path.join(_RESULTS, f"r3_leave_one_dataset_out_EA{int(align)}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("target_dataset,mean_acc,std_acc,mean_kappa,n_subjects\n")
        for t, r in results.items():
            f.write(f"{t},{r['mean_acc']:.4f},{r['std_acc']:.4f},{r['mean_kappa']:.4f},{r['n']}\n")
    print(f"\nSaved {path}")
    return path


def main():
    print("R3: leave-one-dataset-out zero-calibration (LH vs RH, chance 50%)\n")
    leave_one_dataset_out()


if __name__ == "__main__":
    main()
