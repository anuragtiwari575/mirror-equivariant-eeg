"""
show24.py

Reads corrected_results/step24_subject_specific.csv and prints what is in it. Read-only: nothing is trained and
nothing is written, so it can be run while step24_subject_specific.py is still going.

    python show24.py                  the 'session' protocol
    python show24.py cv               the 'cv' protocol
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:
    import step5_benchmark as S5                    # only for the paired statistics
except Exception:
    S5 = None

PATH = os.path.join("corrected_results", "step24_subject_specific.csv")
ORDER = ["csp_lda", "fbcsp", "ts_lr", "eegnet", "shallow", "deepnet", "tcnet", "conformer",
         "eegnet_cr", "shallow_cr", "eegnet_mirror", "shallow_mirror"]
BEST = "shallow_mirror"                         # the model the differences are taken from


COL = {"accuracy": 9, "kappa": 10, "consistency": 11, "lean": 12, "asymmetry": 13}


def read(protocol, what="accuracy"):
    """{(dataset, model): {subject: mean over folds and seeds}}. The columns are read by position, because
    columns were added to the file over time: a line of an earlier format simply has no cell for them."""
    if not os.path.exists(PATH):
        print(f"{PATH} not found; run this from the Codes folder.")
        sys.exit(1)
    i = COL[what]
    latest = {}
    with open(PATH, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 11 or len(p) <= i:           # being written, or that column was not measured yet
                continue
            if p[4] != protocol:
                continue
            try:
                v = float(p[i])
            except ValueError:
                continue
            if not np.isfinite(v):
                continue
            latest[tuple(p[:8])] = v                 # one fit, one value: a later line replaces an earlier one
    acc = {}
    for k, v in latest.items():
        acc.setdefault((k[2], k[6]), {}).setdefault(k[3], []).append(v)
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in acc.items()}


def main():
    protocol = sys.argv[1] if len(sys.argv) > 1 else "session"
    acc = read(protocol)
    if not acc:
        print(f"no rows for protocol '{protocol}' yet")
        return 0
    datasets = sorted({k[0] for k in acc}, key=lambda d: ["2a", "2b", "PhysioNet", "Cho2017", "Dreyer2023"].index(d)
                      if d in ["2a", "2b", "PhysioNet", "Cho2017", "Dreyer2023"] else 9)
    models = [m for m in ORDER if any((d, m) in acc for d in datasets)]
    models += sorted({k[1] for k in acc} - set(models))
    print(f"\nprotocol: {protocol}   accuracy %, mean over subjects (number of subjects in brackets)")
    print(f"{'model':<16}" + "".join(f"{d:>18}" for d in datasets))
    for m in models:
        row = ""
        for d in datasets:
            v = acc.get((d, m))
            row += f"{100 * np.mean(list(v.values())):>12.2f} ({len(v):>3})" if v else f"{'-':>18}"
        print(f"{m:<16}{row}")
    if S5 is None:
        print("\n(step5_benchmark.py not importable here, so no paired tests)")
        return 0
    best = BEST if any((d, BEST) in acc for d in datasets) else "eegnet_mirror"
    print(f"\ndifferences from {best}, in points [95% CI over subjects] dz p")
    for d in datasets:
        mine = acc.get((d, best))
        if not mine:
            continue
        for m in models:
            other = acc.get((d, m))
            if m == best or not other:
                continue
            keys = sorted(set(mine) & set(other))
            if len(keys) >= 3:
                res = S5.paired([mine[s] for s in keys], [other[s] for s in keys])
                print(f"{d:>12}  {best + ' - ' + m:<32}{S5.fmt(res)}  n={len(keys)}")
    cons = read(protocol, "consistency")
    if cons:
        print("\nmirror consistency %: share of trials on which exchanging the hemispheres exchanges the prediction.")
        print("Below 100 even for the mirror models: a trial whose score is zero to within float32 rounding is")
        print("counted as not exchanged. The table below is the measure that does not depend on that.")
        print(f"{'model':<16}" + "".join(f"{d:>18}" for d in datasets))
        for m in models:
            row = ""
            for d in datasets:
                v = cons.get((d, m))
                row += f"{100 * np.mean(list(v.values())):>12.1f} ({len(v):>3})" if v else f"{'-':>18}"
            print(f"{m:<16}{row}")
    asym = read(protocol, "asymmetry")
    if asym:
        print("\nasymmetry of the decision function, mean |s + s'| / mean (|s| + |s'|): zero for the mirror")
        print("construction down to the precision of float32 arithmetic, non-zero for anything that reaches the")
        print("same symmetry only through training")
        print(f"{'model':<16}" + "".join(f"{d:>18}" for d in datasets))
        for m in models:
            row = ""
            for d in datasets:
                v = asym.get((d, m))
                row += f"{np.mean(list(v.values())):>11.2e} ({len(v):>3})" if v else f"{'-':>18}"
            print(f"{m:<16}{row}")
    print("\nSubjects still missing from a column mean that the run has not reached them yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
