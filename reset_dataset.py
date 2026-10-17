r"""
reset_dataset.py

Removes everything a corpus has already contributed, so that it can be decoded again from the recordings
after the way it is read has changed. Needed for dataset 2a once the evaluation session is added: the keys of
the stored fits stay the same while the data behind them changes, and a stale row is the kind of mistake that
never announces itself.

It does three things, in this order, and asks before each of the last two:
  1. copies corrected_results/step24_subject_specific.csv to ...csv.bak-<timestamp>
  2. rewrites the CSV without the rows of that corpus
  3. deletes the cached epochs of that corpus

    python reset_dataset.py --dataset 2a            what would happen, nothing is changed
    python reset_dataset.py --dataset 2a --apply    do it

Do not run it while a fit of that corpus is going.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
import shutil
import sys

OUTDIR = "corrected_results"
CSV = os.path.join(OUTDIR, "step24_subject_specific.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="2a, 2b, PhysioNet, Cho2017 or Dreyer2023")
    ap.add_argument("--apply", action="store_true", help="without this nothing is changed")
    ap.add_argument("--keep-cache", action="store_true", help="drop the rows but keep the cached epochs")
    args = ap.parse_args()
    ds = args.dataset

    if not os.path.exists(CSV):
        print(f"{CSV} not found; run this from the Codes folder.")
        return 1
    with open(CSV, encoding="utf-8") as f:
        head, lines = f.readline(), f.readlines()
    keep = [ln for ln in lines if ln.split(",")[2:3] != [ds]]
    drop = len(lines) - len(keep)
    caches = sorted(glob.glob(os.path.join(OUTDIR, "cache_step24_*", f"{ds}_*.npz")))

    print(f"corpus: {ds}")
    print(f"  rows in the file:      {len(lines)}")
    print(f"  rows of this corpus:   {drop}   (these would be removed)")
    print(f"  rows that would stay:  {len(keep)}")
    print(f"  cached subject files:  {len(caches)}" + ("   (kept)" if args.keep_cache else "   (would be deleted)"))
    if drop == 0 and not caches:
        print("nothing to do")
        return 0
    if not args.apply:
        print("\nnothing was changed. Add --apply to do it.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{CSV}.bak-{stamp}"
    shutil.copy2(CSV, backup)
    print(f"\nbackup: {backup}")
    tmp = CSV + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(head)
        f.writelines(keep)
    os.replace(tmp, CSV)
    print(f"{drop} rows removed from {CSV}")
    if not args.keep_cache:
        gone = 0
        for p in caches:
            try:
                os.remove(p)
                gone += 1
            except OSError as e:
                print(f"  could not delete {p} ({e})")
        print(f"{gone} cached subject files deleted; the corpus will be read from the recordings again")
    print("\nNow decode the corpus again, for example:")
    print(f"  python step24_subject_specific.py --datasets {ds} --protocol session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
