"""Drop from corrected_results/step24_scores.csv the trial rows that a re-fit has made stale.

Two kinds of row go:

  1. Every row of a corpus whose epoch window changed. step24_scores.csv records no band and no
     window, so a row fitted at 0-3 s and a row fitted at 1.25-4.25 s are indistinguishable once
     both are in the file, and read_scores() in make_paper_assets.py would pool them. BCI IV 2b is
     being refitted at 1.25-4.25 s, so its old rows cannot stay.

  2. The rows of a corpus/model pair that the queue is about to fit again. step24_subject_specific.py
     appends; it never replaces. So a second fit of the same (dataset, model, subject) leaves two sets
     of trials in the file, and read_scores() averages them while read_rows() keeps only the last
     line of the main result file. The accuracy table and the ROC figure would then be reporting
     different fits of the same cell. Only the pairs the queue refits are dropped; everything else
     stays, which is what keeps ts_lr in the ROC figure.

Nothing is written without --apply. The original is copied to step24_scores_before_<stamp>.csv first.

    python prune_scores.py                 # count what would go, write nothing
    python prune_scores.py --apply         # do it, after backing the file up

Run it with no fitting process going: step24_subject_specific.py holds the file open while it writes.
"""
import argparse
import datetime
import os
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "corrected_results")
CSV = os.path.join(OUTDIR, "step24_scores.csv")

# dataset,subject,protocol,fold,model,seed,trial,y_true,score[,score_mirror]
I_DATASET, I_MODEL = 0, 4

# The header the writer uses now. The file keeps the header it was first created with, because
# step24_subject_specific.py writes one only when the file does not exist, so a file created before
# the mirrored score was recorded carries nine names while its newer rows carry ten fields. The
# readers here skip the first line and count fields, so they are unaffected; pandas is not, and reads
# the columns one place out. Writing the ten-column header fixes that in both directions: the older
# nine-field rows then simply have no score_mirror, which is the truth about them.
HEADER = "dataset,subject,protocol,fold,model,seed,trial,y_true,score,score_mirror\n"

# Every row of these corpora would go: the epoch window they were fitted at is no longer the one the
# paper uses. It is empty now, and must stay empty for 2b: the 2b rows in the scores file were written
# by the 1.25-4.25 s run, the earlier 0-3 s ones having already been removed. The file records no window,
# so nothing in it can tell the two apart and nothing can put them back -- do not add 2b here again.
DROP_ALL = set()

# These corpus/model pairs are about to be fitted again, so their present rows would become duplicates.
# Part 2 of run_all2.py runs  --datasets 2a,Cho2017 --models MAIN6 --scores --redo  over both corpora,
# and it cannot be resumed without --redo: every fit it covers is already in the result file from an
# earlier queue, so without it the store reports them present and nothing is refitted at all. With it,
# any row already in this file becomes a second set of trials for the same cell, which the ROC and
# antisymmetry figures would pool while the accuracy table reads only the later fit. Both corpora are
# dropped for that reason -- 2a because an interrupted restart left part of one subject written on top
# of a complete set, which cannot be separated from it, and Cho2017 because nine of its fifty-two
# subjects were scored before the machine went down.
MAIN6 = {"eegnet", "shallow", "eegnet_cr", "shallow_cr", "eegnet_mirror", "shallow_mirror"}
DROP_PAIRS = {"2a": set(MAIN6), "Cho2017": set(MAIN6)}


def drops(dataset, model):
    return dataset in DROP_ALL or model in DROP_PAIRS.get(dataset, ())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the pruned file; without it nothing changes")
    ap.add_argument("--csv", default=CSV)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"{args.csv} not found; run this from the Codes folder.")
        return 1

    with open(args.csv, encoding="utf-8") as f:
        head = f.readline()
        lines = f.readlines()

    if not head.startswith("dataset,"):
        print(f"the first line is not the header this file should have:\n  {head.rstrip()[:90]}")
        return 1

    keep, gone, short = [], Counter(), 0
    kept, width = Counter(), Counter()
    for line in lines:
        p = line.rstrip("\n").split(",")
        if len(p) <= I_MODEL:
            short += 1
            keep.append(line)                       # a torn line: left alone, the readers skip it
            continue
        ds, m = p[I_DATASET], p[I_MODEL]
        if drops(ds, m):
            gone[(ds, m)] += 1
        else:
            kept[(ds, m)] += 1
            width[len(p)] += 1
            keep.append(line)

    fix_head = head != HEADER
    print(f"  {args.csv}")
    print(f"  {len(lines)} rows, {len(head.split(','))} columns in the header"
          + (f", {short} torn lines left as they are" if short else ""))
    if fix_head:
        print(f"  the header will be rewritten as {len(HEADER.split(','))} columns: {HEADER.strip()}")
    print("  rows kept, by how many fields they carry: "
          + ", ".join(f"{n} fields: {c}" for n, c in sorted(width.items()))
          + "   (10 is a fit that recorded the mirrored score)")
    print()
    print("  dropping:")
    if not gone:
        print("      nothing matches the rules in this script")
    for (ds, m), n in sorted(gone.items()):
        why = "epoch window changed" if ds in DROP_ALL else "will be fitted again"
        print(f"      {ds:<12}{m:<18}{n:>8}   ({why})")
    print(f"      {'':<12}{'total':<18}{sum(gone.values()):>8}")
    print()
    print(f"  keeping {sum(kept.values())} rows over {len(kept)} corpus/model pairs:")
    for ds in sorted({d for d, _ in kept}):
        ms = sorted(m for d, m in kept if d == ds)
        print(f"      {ds:<12}{', '.join(ms)}")

    if not gone and not fix_head:
        print()
        print("  nothing to change.")
        return 0
    if not args.apply:
        print()
        print("  nothing written. add --apply to do it.")
        return 0

    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    backup = os.path.join(os.path.dirname(os.path.abspath(args.csv)), f"step24_scores_before_{stamp}.csv")
    shutil.copy2(args.csv, backup)
    tmp = args.csv + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER)
        f.writelines(keep)
    os.replace(tmp, args.csv)
    print()
    print(f"  backup: {backup}")
    print(f"  written: {args.csv}  ({len(keep)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
