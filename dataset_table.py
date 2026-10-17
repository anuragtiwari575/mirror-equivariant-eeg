r"""
dataset_table.py

Writes the data table of the paper from the cached epochs themselves, so that no number in it is typed by
hand. Read-only: it opens the cache files of step24_subject_specific.py and writes two files next to them.
Safe to run while a fit is going, and it needs no GPU.

    python dataset_table.py                  the 4-38 Hz, 0-3 s cache
    python dataset_table.py --band 4,38 --window 0,3

Output
    corrected_results/table_datasets.tex     a booktabs table, \input-ed by the paper
    corrected_results/table_datasets.txt     the same numbers as plain text, to read in the terminal

Per corpus it reports the subjects cached, the channels kept, the recording blocks per subject (the files a
subject was recorded in: sessions or runs), the trials per subject, and the share of the trials that belong to
the right-hand class. Trials are given as the median over subjects with the range in brackets, because the
corpora differ in whether every subject contributes the same number.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

OUTDIR = "corrected_results"
DATASETS = ["2a", "2b", "PhysioNet", "Cho2017", "Dreyer2023"]
SHOWN = {"2a": r"BCI IV 2a \citep{Brunner2008}",
         "2b": r"BCI IV 2b \citep{Leeb2008}",
         "PhysioNet": r"Motor Movement/Imagery \citep{Schalk2004}",
         "Cho2017": r"Cho \citep{Cho2017}",
         "Dreyer2023": r"Dreyer \citep{Dreyer2023}"}


def span(v, fmt="{:g}"):
    """median, and the range in brackets when the subjects differ"""
    v = np.asarray(v, dtype=np.float64)
    med = fmt.format(float(np.median(v)))
    if v.min() == v.max():
        return med
    return f"{med} [{fmt.format(v.min())}--{fmt.format(v.max())}]"


def collect(cache):
    """{dataset: dict of columns}; only the corpora that have something cached"""
    rows = {}
    for ds in DATASETS:
        files = sorted(glob.glob(os.path.join(cache, f"{ds}_*.npz")))
        if not files:
            continue
        chans, blocks, trials, share, bad = set(), [], [], [], 0
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as z:
                    y, block, names = z["y"], z["block"], z["names"]
                    n_t = int(z["X"].shape[0])
            except Exception as e:
                bad += 1
                print(f"  {os.path.basename(path)}: could not be read ({e!r})", file=sys.stderr)
                continue
            chans.add(len(names))
            blocks.append(int(np.unique(block).size))
            trials.append(n_t)
            share.append(float(np.mean(np.asarray(y) == 1)))
        if not trials:
            continue
        rows[ds] = {"subjects": len(trials), "channels": sorted(chans), "blocks": blocks,
                    "trials": trials, "share": share, "unreadable": bad}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="4,38")
    ap.add_argument("--window", default="0,3")
    args = ap.parse_args()
    b = tuple(float(v) for v in args.band.split(","))
    w = tuple(float(v) for v in args.window.split(","))
    cache = os.path.join(OUTDIR, f"cache_step24_{b[0]:g}-{b[1]:g}Hz_{w[0]:g}-{w[1]:g}s")
    if not os.path.isdir(cache):
        print(f"{cache} not found; run this from the Codes folder.")
        return 1
    rows = collect(cache)
    if not rows:
        print(f"nothing cached in {cache}")
        return 1

    txt = [f"from {cache}", "",
           f"{'corpus':<12}{'subjects':>9}{'channels':>10}{'blocks':>14}{'trials':>18}{'right hand %':>14}"]
    tex = [r"% table_datasets.tex -- written by dataset_table.py from the cached epochs. Do not edit by hand.",
           r"\begin{table}[t]", r"\centering",
           r"\caption{The five corpora as they are decoded here: left hand against right hand, all EEG"
           r" channels of the recording, \SI{4}{\hertz}--\SI{38}{\hertz}, \SI{0}{\second} to \SI{3}{\second}"
           r" after the cue, \SI{250}{\hertz}. Trials and blocks are given as the median over subjects with"
           r" the range in brackets where the subjects differ. Every entry is read from the epochs themselves"
           r" by \texttt{dataset\_table.py}.}",
           r"\label{tab:datasets}",
           r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Corpus & Subjects & Channels & Blocks & Trials & Right hand (\%) \\", r"\midrule"]
    for ds, r in rows.items():
        ch = "/".join(str(c) for c in r["channels"])
        txt.append(f"{ds:<12}{r['subjects']:>9}{ch:>10}{span(r['blocks']):>14}{span(r['trials']):>18}"
                   f"{100 * float(np.median(r['share'])):>13.1f}")
        tex.append(f"{SHOWN.get(ds, ds)} & {r['subjects']} & {ch} & {span(r['blocks'])} & "
                   f"{span(r['trials'])} & {100 * float(np.median(r['share'])):.1f} \\\\")
        if r["unreadable"]:
            txt.append(f"{'':<12}({r['unreadable']} cached files of this corpus could not be read)")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    txt += ["", "Blocks are the files a subject was recorded in (sessions or runs); a subject with one block is",
            "split by the per-class last-third rule of the session protocol. The right-hand share says whether",
            "the two classes are balanced within a subject."]

    os.makedirs(OUTDIR, exist_ok=True)
    for name, lines in (("table_datasets.tex", tex), ("table_datasets.txt", txt)):
        with open(os.path.join(OUTDIR, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(txt))
    print(f"\nwritten: {os.path.join(OUTDIR, 'table_datasets.tex')} and table_datasets.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
