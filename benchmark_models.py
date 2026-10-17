"""
Multi-method benchmark: run braindecode models through the same R3
leave-one-dataset-out zero-calibration protocol used for our EA+EEGNet, so all
methods are compared in one common environment on the four datasets.

Each model is built with only the constructor arguments it actually accepts
(braindecode 0.8 models differ: some take n_times, others input_window_seconds
+ sfreq), trained on the EA-aligned pooled source subjects, and evaluated on the
held-out target dataset with no target labels. Models that fail to build or train
on the 3-channel input are reported as failed rather than crashing the run.

Only build_model / train_bd / eval_bd touch torch+braindecode; the argument
selection and the results aggregation import and unit-test without them
(test_benchmark.py). Run one quick smoke first, then the full sweep overnight.
"""
from __future__ import annotations
import os
import inspect
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
import crossdataset as R

try:
    import config
    _RESULTS = config.RESULTS
except Exception:
    config = None
    _RESULTS = "."

MODELS = ["EEGNetv4", "ShallowFBCSPNet", "Deep4Net", "EEGConformer", "ATCNet",
          "EEGITNet", "EEGInception", "EEGInceptionMI", "TCN"]


# Per-model overrides for the short (501-sample), 3-channel cross-dataset input.
# Defaults of some models assume longer/ more-channel input and fail otherwise.
_OVERRIDES = {
    "ShallowFBCSPNet": {"filter_time_length": 13, "pool_time_length": 25,
                        "pool_time_stride": 5, "final_conv_length": "auto"},
    "EEGConformer": {"final_fc_length": "auto"},
    "TCN": {"n_blocks": 3, "n_filters": 16, "kernel_size": 7, "drop_prob": 0.3},
}


def _select_kwargs(param_names, n_chans, n_outputs, n_times, sfreq, name=None):
    """Pick, from a common pool, only the constructor args a model accepts,
    then apply any per-model override that the model also accepts."""
    pool = {"n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times,
            "input_window_seconds": float(n_times) / sfreq, "sfreq": sfreq,
            "add_log_softmax": False}
    kw = {k: v for k, v in pool.items() if k in param_names}
    for k, v in _OVERRIDES.get(name, {}).items():
        if k in param_names:
            kw[k] = v
    return kw


def build_model(name, n_chans, n_outputs, n_times, sfreq=250):
    import braindecode.models as M
    cls = getattr(M, name)
    params = inspect.signature(cls.__init__).parameters
    return cls(**_select_kwargs(params, n_chans, n_outputs, n_times, sfreq, name=name))


def _flatten(o):
    """Reduce a model output to (batch, n_classes). braindecode returns
    (B, n_classes) for most models, but some (e.g. TCN) return a per-timestep
    sequence (B, n_classes, T); average over the time axis for those."""
    if o.dim() == 3:
        return o.mean(dim=2)
    if o.dim() > 3:
        return o.reshape(o.shape[0], o.shape[1], -1).mean(dim=2)
    return o


def _pick_loss(model, dummy):
    """Some braindecode models output log-softmax, others raw logits. Detect and
    return the matching loss so training is correct for every model."""
    import torch
    import torch.nn as nn
    with torch.no_grad():
        o = _flatten(model(dummy))
    is_logprob = bool((o <= 1e-3).all().item()) and \
        bool((o.exp().sum(-1) - 1).abs().mean().item() < 0.15)
    return nn.NLLLoss() if is_logprob else nn.CrossEntropyLoss()


def train_bd(name, source, n_times, *, epochs=150, lr=1e-3, batch_size=64,
             patience=30, val_frac=0.15, seed=42, device=None):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xs = np.concatenate([X for X, y in source.values()], axis=0)
    ys = np.concatenate([y for X, y in source.values()], axis=0)
    n_ch = Xs.shape[1]
    tr, va = train_test_split(np.arange(len(ys)), test_size=val_frac,
                              stratify=ys, random_state=seed)

    def T(a, dt):
        return torch.tensor(np.asarray(a), dtype=dt)

    loader = DataLoader(TensorDataset(T(Xs[tr], torch.float32), T(ys[tr], torch.long)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    Xv, yv = T(Xs[va], torch.float32).to(device), ys[va]

    model = build_model(name, n_ch, 2, n_times).to(device)
    crit = _pick_loss(model, T(Xs[:2], torch.float32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = _flatten(model(xb))
            opt.zero_grad()
            crit(out, yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_acc = accuracy_score(yv, _flatten(model(Xv)).argmax(1).cpu().numpy())
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


def eval_bd(model, device, target_data):
    import torch
    model.eval()
    out = []
    for s, (X, y) in target_data.items():
        with torch.no_grad():
            pred = _flatten(model(torch.tensor(np.asarray(X, np.float32)).to(device))).argmax(1).cpu().numpy()
        out.append((s, float(accuracy_score(y, pred)), float(cohen_kappa_score(y, pred))))
    return out


def _append_row(name, row, names):
    """Append one method's result to the CSV immediately (crash-safe)."""
    os.makedirs(_RESULTS, exist_ok=True)
    path = os.path.join(_RESULTS, "benchmark_methods.csv")
    header = "method," + ",".join(names) + ",mean\n"
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
    with open(path, "a", encoding="utf-8") as f:
        if "error" in row:
            f.write(f"{name},FAILED\n")
        else:
            vals = [row[t] for t in names]
            f.write(f"{name}," + ",".join(f"{v:.4f}" for v in vals) + f",{np.mean(vals):.4f}\n")
    return path


def _done_methods():
    """Methods already present in the CSV, so a restart can skip them."""
    path = os.path.join(_RESULTS, "benchmark_methods.csv")
    if not os.path.exists(path):
        return set()
    done = set()
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        if line.strip():
            done.add(line.split(",")[0])
    return done


def run_benchmark(models=None, pn_n=20, hgd_n=20, save=True, skip_done=True, **train_kw):
    models = models or MODELS
    raw = R.load_all(pn_n=pn_n, hgd_n=hgd_n)
    ea = R._prep_align(raw, True)
    names = [k for k in ea if ea[k]]
    n_times = next(iter(ea[names[0]].values()))[0].shape[2]

    done = _done_methods() if (save and skip_done) else set()
    table = {}
    for name in models:
        if name in done:
            print(f"  {name:<16} already done, skipping", flush=True)
            continue
        try:
            per_target = {}
            for tgt in names:
                source = {f"{ds}_{s}": xy for ds in names if ds != tgt
                          for s, xy in ea[ds].items()}
                model, device = train_bd(name, source, n_times, **train_kw)
                per = eval_bd(model, device, ea[tgt])
                per_target[tgt] = float(np.mean([a for _, a, _ in per]))
                print(f"  {name:<16} target {tgt:<10} acc={per_target[tgt]*100:6.2f}", flush=True)
            table[name] = per_target
        except Exception as e:
            print(f"  {name:<16} FAILED: {e}", flush=True)
            table[name] = {"error": str(e)}
        if save:
            _append_row(name, table[name], names)      # checkpoint after each model
    _print(table, names)
    return table


def _save(table, names):
    os.makedirs(_RESULTS, exist_ok=True)
    path = os.path.join(_RESULTS, "benchmark_methods.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("method," + ",".join(names) + ",mean\n")
        for m, row in table.items():
            if "error" in row:
                f.write(f"{m},FAILED\n")
                continue
            vals = [row[t] for t in names]
            cells = ",".join(f"{v:.4f}" for v in vals)
            f.write(f"{m},{cells},{np.mean(vals):.4f}\n")
    print(f"\nSaved {path}")
    return path


def _print(table, names):
    print("\n=== Multi-method cross-dataset zero-cal (LODO, accuracy %) ===")
    print("  method".ljust(18) + "".join(f"{t:>11}" for t in names) + f"{'mean':>9}")
    for m, row in table.items():
        if "error" in row:
            print(f"  {m:<16} FAILED")
            continue
        vals = [row[t] for t in names]
        line = f"  {m:<16}" + "".join(f"{row[t]*100:>11.2f}" for t in names)
        print(line + f"{np.mean(vals)*100:>9.2f}")


def smoke():
    """Quick plumbing check: EEGNetv4 on one target, few epochs."""
    raw = R.load_all()
    ea = R._prep_align(raw, True)
    names = [k for k in ea if ea[k]]
    n_times = next(iter(ea[names[0]].values()))[0].shape[2]
    tgt = names[0]
    source = {f"{ds}_{s}": xy for ds in names if ds != tgt for s, xy in ea[ds].items()}
    model, device = train_bd("EEGNetv4", source, n_times, epochs=5, patience=5)
    per = eval_bd(model, device, ea[tgt])
    acc = np.mean([a for _, a, _ in per])
    print(f"SMOKE ok: EEGNetv4 target {tgt} acc={acc*100:.2f}% (5 epochs)")


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
