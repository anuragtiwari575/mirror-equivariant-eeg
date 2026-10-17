"""What is actually in the two result files, so that a restart does not have to be guessed.

Reads corrected_results/step24_subject_specific.csv and corrected_results/step24_scores.csv and
reports, per corpus:

  * the fits present, by window and model, counting distinct keys rather than lines, since --redo
    appends a second line for a key it refits and read_rows() keeps only the last;
  * which of those fits recorded the per-trial scores that the ROC and antisymmetry figures need,
    and which did not. This is the question a restart turns on: a fit already in the store is
    skipped without --redo, and if it holds no scores, no later run will write them.

Nothing is written. Run it from the Codes folder.

    python status.py                 # every corpus
    python status.py 2b Cho2017      # only these
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "corrected_results")
FITS = os.path.join(OUTDIR, "step24_subject_specific.csv")
SCORES = os.path.join(OUTDIR, "step24_scores.csv")


def read_fits():
    """{(dataset, window, model): {(subject, protocol, fold, seed)}}, plus the duplicate-line count."""
    keys, lines, dup, short = defaultdict(set), 0, 0, 0
    seen = set()
    with open(FITS, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            lines += 1
            if len(p) < 11:
                short += 1
                continue
            k = tuple(p[:8])
            if k in seen:
                dup += 1
            seen.add(k)
            keys[(p[2], p[1], p[6])].add((p[3], p[4], p[5], p[7]))
    return keys, lines, dup, short


def read_scores():
    """{(dataset, model): {(subject, protocol, fold, seed): [rows, rows_with_mirror]}}"""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    if not os.path.exists(SCORES):
        return out
    with open(SCORES, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 9:
                continue
            cell = out[(p[0], p[4])][(p[1], p[2], p[3], p[5])]
            cell[0] += 1
            if len(p) > 9 and p[9].strip():
                cell[1] += 1
    return out


def main():
    if not os.path.exists(FITS):
        print(f"{FITS} not found; run this from the Codes folder.")
        return 1
    want = set(sys.argv[1:])
    keys, lines, dup, short = read_fits()
    scores = read_scores()

    print(f"  {FITS}")
    print(f"  {lines} lines, {dup} of them a repeat of a key already seen (harmless: the last one is read)"
          + (f", {short} torn" if short else ""))
    print()

    datasets = sorted({d for d, _w, _m in keys})
    for ds in datasets:
        if want and ds not in want:
            continue
        windows = sorted({w for d, w, _m in keys if d == ds})
        print(f"  == {ds}   window(s) present: {', '.join(windows)}")
        for w in windows:
            models = sorted(m for d, ww, m in keys if d == ds and ww == w)
            subs = {s for d, ww, m in keys if d == ds and ww == w for s, _p, _f, _sd in keys[(d, ww, m)]}
            print(f"     window {w}:  {len(models)} models, {len(subs)} subjects")
            print(f"     {'model':<18}{'fits':>6}{'subj':>6}{'seeds':>7}   {'scored':>7}{'mirrored':>10}")
            for m in models:
                cells = keys[(ds, w, m)]
                nsub = len({c[0] for c in cells})
                nseed = len({c[3] for c in cells})
                sc = scores.get((ds, m), {})
                # a fit is "scored" when its own key appears in the scores file
                scored = sum(1 for c in cells if (c[0], c[1], c[2], c[3]) in sc)
                mirrored = sum(1 for c in cells
                               if sc.get((c[0], c[1], c[2], c[3]), [0, 0])[1] > 0)
                flag = ""
                if scored < len(cells):
                    flag = f"   <-- {len(cells) - scored} fit(s) hold no per-trial scores"
                print(f"     {m:<18}{len(cells):>6}{nsub:>6}{nseed:>7}   {scored:>7}{mirrored:>10}{flag}")
            print()

    extra = sorted({(d, m) for d, m in scores} - {(d, m) for d, _w, m in keys})
    if extra:
        print("  score rows whose fit is not in the result file (left over from a pruned or earlier run):")
        for d, m in extra:
            print(f"     {d:<12}{m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
