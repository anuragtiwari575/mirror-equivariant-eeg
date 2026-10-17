r"""
complexity.py

Measures what each decoder costs in space and in time, and writes the table of the paper. It fits nothing
and reads no recording: the networks are built for the channel count of each dataset and run on random
tensors, so this takes a couple of minutes and can be run while a fit is going (it uses the GPU only briefly;
pass --cpu to keep off it entirely).

    python complexity.py                 all four montages, on the GPU if there is one
    python complexity.py --cpu           on the processor only
    python complexity.py --channels 22   one montage

Reported per decoder, for one window of C channels and 500 samples:

    parameters        the number of learnable scalars
    MACs              multiply-accumulate operations of one forward pass, counted from the layers
    activations       the largest intermediate tensor plus the input, in MB at batch 64 (what dominates
                      the memory a training step needs beyond the parameters)
    train step        wall-clock milliseconds for one optimizer step at batch 64, averaged over repeats
    inference         wall-clock milliseconds for one window, batch 1

Written to corrected_results/paper_assets/tables/table_complexity.tex and .txt.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

try:
    import step24_subject_specific as S24
except Exception as e:                                        # pragma: no cover
    print(f"run this from the Codes folder, next to step24_subject_specific.py ({e!r})")
    sys.exit(1)

OUT = os.path.join("corrected_results", "paper_assets", "tables")
MONTAGE = [("BCI IV 2a", 22), ("BCI IV 2b", 3), ("PhysioNet", 64), ("Dreyer", 27)]
SHOW = ["eegnet", "eegnet_cr", "eegnet_two", "eegnet_rand", "eegnet_mirror",
        "shallow", "shallow_cr", "shallow_two", "shallow_rand", "shallow_mirror",
        "deepnet", "tcnet", "conformer"]
LABEL = {"eegnet": "EEGNet", "eegnet_cr": "EEGNet + CR", "eegnet_two": "EEGNet, two branches",
         "eegnet_rand": "EEGNet + random involution", "eegnet_mirror": "EEGNet + mirror",
         "shallow": "ShallowConvNet", "shallow_cr": "ShallowConvNet + CR",
         "shallow_two": "ShallowConvNet, two branches",
         "shallow_rand": "ShallowConvNet + random involution",
         "shallow_mirror": "ShallowConvNet + mirror",
         "deepnet": "DeepConvNet", "tcnet": "EEG-TCNet", "conformer": "EEG-Conformer"}


def build(model, C, T):
    """The same networks fit_deep builds, without fitting them."""
    base = model[:-3] if model.endswith("_cr") else model
    mk = {"eegnet": lambda: S24.EEGNet(C, T, 2), "shallow": lambda: S24.Shallow(C, T, 2),
          "deepnet": lambda: S24.DeepConvNet(C, T, 2), "tcnet": lambda: S24.EEGTCNet(C, T, 2),
          "conformer": lambda: S24.Conformer(C, T, 2),
          "eegnet_mirror": lambda: S24.Mirror(S24.EEGNet(C, T, 1), list(range(C))[::-1]),
          "shallow_mirror": lambda: S24.Mirror(S24.Shallow(C, T, 1), list(range(C))[::-1]),
          "eegnet_rand": lambda: S24.Mirror(S24.EEGNet(C, T, 1), S24.random_involution(C, 0)),
          "shallow_rand": lambda: S24.Mirror(S24.Shallow(C, T, 1), S24.random_involution(C, 0)),
          "eegnet_two": lambda: S24.TwoHeads(S24.EEGNet(C, T, 1), S24.EEGNet(C, T, 1)),
          "shallow_two": lambda: S24.TwoHeads(S24.Shallow(C, T, 1), S24.Shallow(C, T, 1))}
    return mk[base]()


def count(net, C, T, device):
    """Parameters, MACs of one forward pass, and the largest intermediate tensor at batch 64."""
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    macs, biggest = [0], [0]
    hooks = []

    def hook(mod, inp, outp):
        o = outp[0] if isinstance(outp, tuple) else outp
        n = o.numel()
        biggest[0] = max(biggest[0], n)
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            k = int(np.prod(mod.kernel_size))
            macs[0] += n * k * mod.in_channels // mod.groups
        elif isinstance(mod, nn.Linear):
            macs[0] += n * mod.in_features
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        net(torch.zeros(1, C, T, device=device))
    for h in hooks:
        h.remove()
    return params, macs[0], biggest[0] * 64 * 4 / 2 ** 20          # float32, batch 64, in MB


def timings(net, C, T, device, steps=12, warmup=4):
    """Milliseconds for one optimizer step at batch 64, and for one window at batch 1."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    xb = torch.randn(64, C, T, device=device)
    yb = torch.randint(0, 2, (64,), device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    for _ in range(warmup):
        opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    sync(); t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    sync(); train_ms = 1000 * (time.perf_counter() - t0) / steps
    net.eval()
    x1 = torch.randn(1, C, T, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            net(x1)
        sync(); t0 = time.perf_counter()
        for _ in range(steps * 4):
            net(x1)
        sync(); infer_ms = 1000 * (time.perf_counter() - t0) / (steps * 4)
    return train_ms, infer_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--channels", type=int, help="one montage only")
    ap.add_argument("--samples", type=int, default=S24.CROP)
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    montages = [(f"{args.channels} channels", args.channels)] if args.channels else MONTAGE
    T = args.samples
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)

    rows, txt = [], []
    busy = device == "cpu"
    print(f"device: {device}, window of {T} samples, batch 64")
    if busy:
        print("\n  WARNING: no GPU is being used for these measurements. The parameter, operation and\n"
              "  activation columns are counted and are correct either way, but the two time columns are\n"
              "  wall-clock measurements and are only meaningful on an idle machine. If a fit is running\n"
              "  while this is measured, they are noise: a decoder cannot be faster than the backbone it\n"
              "  evaluates twice. Measure the times again when nothing else is using the machine.\n")
    head = f"{'montage':<12}{'model':<32}{'params':>10}{'MACs/window':>14}{'act MB':>9}{'step ms':>10}{'infer ms':>10}"
    print(head); txt.append(head)
    for name, C in montages:
        for m in SHOW:
            try:
                net = build(m, C, T).to(device)
            except Exception as e:
                print(f"{name:<12}{LABEL.get(m, m):<32}  could not be built ({e!r})")
                continue
            p, mac, act = count(net, C, T, device)
            tr, inf = timings(net, C, T, device)
            line = (f"{name:<12}{LABEL.get(m, m):<32}{p:>10,}{mac:>14,}{act:>9.1f}{tr:>10.1f}{inf:>10.2f}")
            print(line); txt.append(line)
            rows.append([name, LABEL.get(m, m), f"{p:,}", f"{mac / 1e6:.1f}", f"{act:.1f}",
                         f"{tr:.1f}", f"{inf:.2f}"])
            del net
            if device == "cuda":
                torch.cuda.empty_cache()

    with open(os.path.join(OUT, "table_complexity.tex"), "w", encoding="utf-8") as f:
        f.write("% written by complexity.py. Do not edit by hand.\n")
        f.write("\\begingroup\n\\footnotesize\n\\setlength{\\tabcolsep}{4.5pt}\n")
        f.write("\\begin{longtable}{@{}llrrrrr@{}}\n")
        f.write("\\caption{What each decoder costs. Parameters and multiply-accumulate operations (MACs) are "
                "counted from the layers for one window of 500 samples; the activation column is the largest "
                "intermediate tensor at batch 64 in float32; the two time columns are measured on one "
                f"{'GPU' if device == 'cuda' else 'processor'}, for one optimizer step at batch 64 and for one "
                "window at batch 1. The construction evaluates its backbone twice per step and therefore costs "
                "about twice the backbone's time and activation memory, while its head carries one output unit "
                "instead of two and therefore $n+1$ fewer parameters, where $n$ is the input width of "
                "that layer.}\\label{tab:complexity}\\\\\n")
        if busy:
            f.write("% WARNING: the time columns below were measured on the processor, and are only\n"
                    "% meaningful on an idle machine. Re-run complexity.py on an idle GPU before submission.\n")
        head = ("\\textbf{Montage} & \\textbf{Model} & \\textbf{Params} & \\textbf{MACs (M)} & "
                "\\textbf{Act.\\ (MB)} & \\textbf{Step (ms)} & \\textbf{Infer.\\ (ms)}\\\\\n")
        f.write("\\toprule\n" + head + "\\midrule\n\\endfirsthead\n")
        f.write("\\multicolumn{7}{l}{\\textit{Table \\thetable{} continued.}}\\\\\n")
        f.write("\\toprule\n" + head + "\\midrule\n\\endhead\n")
        f.write("\\midrule\n\\multicolumn{7}{r}{\\textit{continued on the next page}}\\\\\n"
                "\\endfoot\n\\bottomrule\n\\endlastfoot\n")
        last = None
        for r in rows:
            r = list(r)
            if r[0] == last:
                r[0] = ""                       # the montage is named once per block, not once per row
            else:
                if last is not None:
                    f.write("\\addlinespace[3pt]\n")
                last = r[0]
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\end{longtable}\n\\endgroup\n")
    with open(os.path.join(OUT, "table_complexity.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
        if busy:
            f.write("\nWARNING: measured on the processor. The two time columns are only meaningful on an\n"
                    "idle machine; the parameter, operation and activation columns are counts and stand.\n")
    print(f"\nwritten: {os.path.join(OUT, 'table_complexity.tex')} and .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
