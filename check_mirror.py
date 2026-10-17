"""
check_mirror.py

Read-only: checks the mirror permutation of EVERY cached subject of every dataset, not just the first one.
Nothing is trained and nothing is written, so it can be run while a fit is going.

For each subject it asks three questions:
  * does the montage have a complete mirror image at all (otherwise the mirror models are skipped for it);
  * is the permutation an involution, M*M = identity (if it is not, the construction is not equivariant for
    that subject and its consistency cannot be 100 %);
  * are any channel names repeated (a repeated name makes `names.index(twin)` point at the wrong column, which
    is the one way a montage can pass the eye and still fail the involution).

Subjects are grouped by their montage, so 87 subjects with the same cap print as one line.

    python check_mirror.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

CACHE = os.path.join("corrected_results", "cache_step24_4-38Hz_0-3s")
DATASETS = ["2a", "2b", "PhysioNet", "Cho2017", "Dreyer2023"]


def mirror_index(names):
    """The same function as in step24_subject_specific.py."""
    up = [n.upper() for n in names]
    perm = []
    for n in up:
        m = re.match(r"^([A-Z]+?)(\d+|Z)$", n)
        if m is None:
            return None
        if m.group(2) == "Z":
            perm.append(up.index(n))
        else:
            k = int(m.group(2))
            twin = f"{m.group(1)}{k + 1 if k % 2 else k - 1}"
            if twin not in up:
                return None
            perm.append(up.index(twin))
    return perm


def verdict(names):
    """(label, detail) for one montage."""
    perm = mirror_index(names)
    dup = sorted({n.upper() for n in names if [x.upper() for x in names].count(n.upper()) > 1})
    if perm is None:
        return "no mirror image", f"{len(names)} channels, the mirror models are skipped"
    bad = [i for i in range(len(perm)) if perm[perm[i]] != i]
    mid = [names[i] for i in range(len(perm)) if perm[i] == i]
    if bad:
        first = ", ".join(f"{names[i]}->{names[perm[i]]}->{names[perm[perm[i]]]}" for i in bad[:4])
        return "NOT an involution", f"{len(bad)} of {len(perm)} channels do not point back: {first}"
    return "ok", (f"{len(perm)} channels, {len(mid)} on the midline ({', '.join(mid)})"
                  + (f", REPEATED NAMES: {', '.join(dup)}" if dup else ""))


def main():
    if not os.path.isdir(CACHE):
        print(f"{CACHE} not found; run this from the Codes folder.")
        return 1
    worst = 0
    for ds in DATASETS:
        files = sorted(glob.glob(os.path.join(CACHE, f"{ds}_*.npz")))
        print(f"\n{ds}: {len(files)} subjects cached")
        if not files:
            continue
        groups = {}                                     # montage -> (label, detail, [subjects])
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as z:
                    names = [str(s) for s in z["names"]]
            except Exception as e:
                print(f"  {os.path.basename(path)}: could not be read ({e!r})")
                worst = max(worst, 1)
                continue
            sub = os.path.basename(path)[len(ds) + 1:-4]
            g = groups.setdefault(tuple(names), [None, None, []])
            if g[0] is None:
                g[0], g[1] = verdict(names)
            g[2].append(sub)
        for names, (label, detail, subs) in groups.items():
            short = ", ".join(subs[:6]) + (f", ... (+{len(subs) - 6})" if len(subs) > 6 else "")
            mark = {"ok": "ok ", "no mirror image": "-- ", "NOT an involution": "!! "}[label]
            print(f"  {mark}{label:<18} {len(subs):>3} subjects  {detail}")
            print(f"      subjects: {short}")
            print(f"      channels: {', '.join(names)}")
            if label == "NOT an involution":
                worst = 2
            elif label == "no mirror image":
                worst = max(worst, 1)
    print("\n" + ("every cached subject has a valid mirror permutation" if worst == 0 else
                  "at least one subject has no mirror image (its mirror models were skipped)" if worst == 1 else
                  "AT LEAST ONE SUBJECT IS NOT AN INVOLUTION: for those subjects the construction is not "
                  "equivariant and their numbers must not be used"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
