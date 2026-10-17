"""
step7_final_checks.py

Run from the Codes folder, after steps 5 and 6:

    python step7_final_checks.py

[D] Does the distance between a target subject and the source predict transfer?
    For every direction: the Riemannian distance between the source centre and
    the centre of each target subject (trace-normalized covariances), against
    accuracy without alignment, accuracy with recentering, and the gain.
    Spearman correlations inside each direction, combined over directions, with
    a permutation test that shuffles the subjects of each target dataset once
    and applies the same shuffle to all of its sources.
[E] Sensitivity to the re-reference of Cho2017: the benchmark again with Cho2017
    loaded without the common average reference (reads the 52 Cho2017 files once,
    a few minutes; the other datasets come from the cache).
[F] One file with every result so far, for upload:
    corrected_results/results_bundle.zip
    (tables, CSV files and the four loader files; no EEG data, no cache)

Nothing existing is changed.
"""
from __future__ import annotations

import os
import sys
import glob
import time
import zipfile
import datetime

import numpy as np

try:
    import step5_benchmark as S5
except Exception as _e:                                    # pragma: no cover
    print(f"could not import step5_benchmark.py; run this from the Codes folder. ({_e!r})")
    sys.exit(1)

OUTDIR = S5.OUTDIR
CACHE_NOREF = S5.CACHE + "_choNoRef"
N_PERM = 10000
BUNDLE = os.path.join(OUTDIR, "results_bundle.zip")
LOADER_FILES = ["crossdata.py", "within_loader.py", "variants.py", "config.py"]

out = S5.out


def tn(C):
    return C / np.trace(C, axis1=1, axis2=2)[:, None, None]


def load_cache():
    everything = {}
    for ds in S5.DATASETS:
        recs = []
        for f in sorted(glob.glob(os.path.join(S5.CACHE, f"{ds}_*.npz"))):
            recs.append(S5.get_subject(ds, int(os.path.basename(f)[len(ds) + 1:-4]), None, None))
        everything[ds] = sorted(recs, key=lambda r: r["subj"])
    return {ds: [r for r in recs if not S5.exclusion_reason(ds, r)] for ds, recs in everything.items()}


def read_csv(path):
    acc = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            if line.strip():
                m, s, t, sub, cov, al, clf, a = line.strip().split(",")
                acc[(m, s, t, int(sub), cov, al, clf)] = float(a)
    return acc


# =========================== [D] distance ===========================
def airm(A, B):
    P = S5.invsqrtm(A)
    w = np.linalg.eigvalsh(0.5 * ((P @ B @ P) + (P @ B @ P).T))
    return float(np.sqrt(np.sum(np.log(w) ** 2)))


def _ranks(x):
    from scipy.stats import rankdata
    return rankdata(x)


def _rho_from_ranks(rx, ry):
    rx = rx - rx.mean(axis=-1, keepdims=True)
    ry = ry - ry.mean(axis=-1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=-1) * (ry ** 2).sum(axis=-1))
    return (rx * ry).sum(axis=-1) / np.where(den > 0, den, np.nan)


def combined_rho(dist, y, n_perm=N_PERM, seed=0):
    """dist, y: {(source, target): vector over the target subjects, same subject order for
    every source of a target}. Returns the n-weighted mean Spearman rho over directions
    and a two-sided permutation p-value (one shuffle per target dataset, shared by its sources)."""
    dirs = sorted(dist)
    rd = {k: _ranks(dist[k]) for k in dirs}
    ry = {k: _ranks(y[k]) for k in dirs}
    w = np.array([len(rd[k]) for k in dirs], float)
    obs_each = np.array([_rho_from_ranks(rd[k], ry[k]) for k in dirs])
    ok = np.isfinite(obs_each)
    obs = float(np.sum(w[ok] * obs_each[ok]) / np.sum(w[ok]))
    rng = np.random.default_rng(seed)
    targets = sorted({k[1] for k in dirs})
    total = np.zeros(n_perm)
    for t in targets:
        n = len(next(rd[k] for k in dirs if k[1] == t))
        perms = np.argsort(rng.random((n_perm, n)), axis=1)
        for i, k in enumerate(dirs):
            if k[1] == t and ok[i]:
                total += w[i] * _rho_from_ranks(rd[k][perms], ry[k][None, :])
    perm = total / np.sum(w[ok])
    p = (int((np.abs(perm) >= abs(obs) - 1e-12).sum()) + 1) / (n_perm + 1)
    return obs, float(p), dict(zip(dirs, obs_each))


def section_d(main, acc, libs):
    mean_riemann = libs[2]
    out("[D] Distance between each target subject and the source centre, against transfer (TS-LR)")
    out("    distance: affine-invariant Riemannian distance between Riemannian means of trace-normalized covariances")
    out("    rho: Spearman correlation inside each direction, n-weighted mean over directions;")
    out("    p: permutation, one shuffle per target dataset shared by all of its sources")
    for montage, members in S5.MONTAGES.items():
        key = "C3" if montage == "3ch" else "C17"
        names = [d for d in members if len(main.get(d, [])) >= 3]
        centre = {d: mean_riemann(np.concatenate([tn(r[key]) for r in main[d]])) for d in names}
        sub_centre = {d: [mean_riemann(tn(r[key])) for r in main[d]] for d in names}
        out(f"    {montage}: distance between dataset centres")
        out("      " + f"{'':<11}" + "".join(f"{t:>11}" for t in names))
        for s in names:
            out(f"      {s:<11}" + "".join(f"{'-':>11}" if s == t else f"{airm(centre[s], centre[t]):>11.2f}"
                                           for t in names))
        dist, none, prim, gain = {}, {}, {}, {}
        for s in names:
            for t in names:
                if s == t:
                    continue
                subs = [r["subj"] for r in main[t]]
                dist[(s, t)] = np.array([airm(centre[s], c) for c in sub_centre[t]])
                none[(s, t)] = np.array([acc[(montage, s, t, j, "tn", "none", "lr")] for j in subs])
                prim[(s, t)] = np.array([acc[(montage, s, t, j, "tn", "ra_ps", "lr")] for j in subs])
                gain[(s, t)] = prim[(s, t)] - none[(s, t)]
        n_total = sum(len(v) for v in dist.values())
        out(f"    {montage}: {len(dist)} directions, {n_total} source-target-subject pairs; "
            f"median distance {np.median(np.concatenate(list(dist.values()))):.2f}")
        each = {}
        for label, y in (("accuracy without alignment", none), ("accuracy with recentering", prim),
                         ("gain from recentering", gain)):
            rho, p, each[label] = combined_rho(dist, y)
            out(f"      distance vs {label:<28} rho = {rho:+.3f}   {S5.ptxt(p)}")
        out(f"      {'direction':<24}{'n':>5}{'median dist':>13}{'rho none':>10}{'rho recentered':>16}{'rho gain':>10}")
        for k in sorted(dist):
            if len(dist[k]) >= 20:
                out(f"      {k[0] + ' -> ' + k[1]:<24}{len(dist[k]):>5}{np.median(dist[k]):>13.2f}"
                    f"{each['accuracy without alignment'][k]:>+10.2f}"
                    f"{each['accuracy with recentering'][k]:>+16.2f}{each['gain from recentering'][k]:>+10.2f}")
        # thirds of the distance distribution, pooled over directions after ranking inside each direction
        q = np.concatenate([(_ranks(dist[k]) - 0.5) / len(dist[k]) for k in sorted(dist)])
        a0 = np.concatenate([none[k] for k in sorted(dist)])
        a1 = np.concatenate([prim[k] for k in sorted(dist)])
        thirds = ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))
        out("      by third of distance inside each direction (near / middle / far):")
        out("        without alignment " + " / ".join(f"{100 * a0[(q >= lo) & (q < hi)].mean():.1f}" for lo, hi in thirds)
            + "   |   recentered " + " / ".join(f"{100 * a1[(q >= lo) & (q < hi)].mean():.1f}" for lo, hi in thirds))
        with open(os.path.join(OUTDIR, f"distance_pairs_{montage}.csv"), "w", encoding="utf-8") as f:
            f.write("montage,source,target,subject,distance,acc_no_alignment,acc_recentered\n")
            for (s, t), d in sorted(dist.items()):
                for j, r in enumerate(main[t]):
                    f.write(f"{montage},{s},{t},{r['subj']},{d[j]:.6f},{none[(s, t)][j]:.6f},{prim[(s, t)][j]:.6f}\n")
    out()


# =========================== [E] Cho2017 reference ===========================
def load_cho_noref(subj, mods):
    """Cho2017 as the submitted experiments loaded it: channel selection without any re-reference."""
    mne, CD, SM = mods["mne"], mods["CD"], mods["SM"]
    files = sorted(glob.glob(os.path.join(mods["config"].DATA_HGD, f"sub-{subj}", "**",
                                          f"sub-{subj}_*task-imagery*_eeg.bdf"), recursive=True))
    Xs, ys = [], []
    for f in files:
        raw = mne.io.read_raw_bdf(f, preload=True, verbose="ERROR")
        X, y = CD._epoch(CD._prep(raw, SM, 8.0, 30.0, 250), S5.CHO_EVENTS, 0.5, 2.5)
        del raw
        if X is not None:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        raise RuntimeError("no left_hand / right_hand trials found")
    return np.concatenate(Xs), np.concatenate(ys), {"reference": "none (CMS/DRL as recorded)"}


def section_e(main, libs):
    out("[E] Sensitivity to the re-reference of Cho2017 (common average in the main analysis, none here)")
    try:
        import mne
        import config
        import crossdata as CD
        import variants as V
        mods = dict(mne=mne, config=config, CD=CD, SM=list(V.SM))
    except Exception as e:
        out(f"    project modules not importable here ({e!r}); section skipped")
        out()
        return
    saved_cache, saved_loaders = S5.CACHE, S5.LOADERS
    S5.CACHE, S5.LOADERS = CACHE_NOREF, dict(saved_loaders, Cho2017=load_cho_noref)
    noref, t0 = [], time.time()
    try:
        for i, r in enumerate(main["Cho2017"]):
            noref.append(S5.get_subject("Cho2017", r["subj"], mods, libs))
            if (i + 1) % 10 == 0 or i + 1 == len(main["Cho2017"]):
                print(f"    Cho2017 without re-reference: {i + 1}/{len(main['Cho2017'])} ({time.time() - t0:.0f} s)",
                      flush=True)
    except Exception as e:
        out(f"    Cho2017 could not be reloaded ({e!r}); section skipped")
        out()
        return
    finally:
        S5.CACHE, S5.LOADERS = saved_cache, saved_loaders
    if any(not np.array_equal(a["y"], b["y"]) for a, b in zip(noref, main["Cho2017"])):
        out("    <-- trial labels differ between the two loadings; section skipped")
        out()
        return
    variant = dict(main, Cho2017=noref)
    saved = (S5.COVS, S5.ALIGNS, S5.CLFS)
    S5.COVS, S5.ALIGNS, S5.CLFS = ["tn"], ["none", "ra_ps"], ["lr"]
    try:
        for montage in S5.MONTAGES:
            views = {}
            for label, data in (("CAR", main), ("none", variant)):
                acc, names = S5.run_montage(data, montage, libs)
                views[label] = S5.View(acc, data, names)
            out(f"    {montage}")
            for label, v in views.items():
                a, b = v.all_targets("tn", "ra_ps", "lr"), v.all_targets("tn", "none", "lr")
                d, lo, hi, dz, p = S5.paired(a, b)
                out(f"      reference {label:<5} all directions: no alignment {100 * b.mean():.2f}, recentered "
                    f"{100 * a.mean():.2f}, gain {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            vc, vn = views["CAR"], views["none"]
            others = [d for d in vc.names if d != "Cho2017"]
            for cond in ("ra_ps", "none"):
                t_car, t_non = (v.over_sources("Cho2017", "tn", cond, "lr") for v in (vc, vn))
                s_car, s_non = (np.concatenate([v.vec("Cho2017", t, "tn", cond, "lr") for t in others])
                                for v in (vc, vn))
                out(f"      tn {cond:<6} Cho2017 as target: CAR {100 * t_car.mean():.2f}, none {100 * t_non.mean():.2f}, "
                    f"CAR - none {S5.fmt(S5.paired(t_car, t_non))}")
                out(f"      tn {cond:<6} Cho2017 as source: CAR {100 * s_car.mean():.2f}, none {100 * s_non.mean():.2f}, "
                    f"CAR - none {S5.fmt(S5.paired(s_car, s_non))}")
    finally:
        S5.COVS, S5.ALIGNS, S5.CLFS = saved
    try:
        import step6_references as S6
        for montage, key in (("3ch", "C3"), ("17ch", "C17")):
            w_car = np.array([S6.within_subject(tn(r[key]), r["y"], "lr", libs) for r in main["Cho2017"]])
            w_non = np.array([S6.within_subject(tn(r[key]), r["y"], "lr", libs) for r in noref])
            out(f"    {montage} within-subject accuracy of Cho2017 (TS-LR): CAR {100 * w_car.mean():.2f}, "
                f"none {100 * w_non.mean():.2f}, CAR - none {S5.fmt(S5.paired(w_car, w_non))}")
    except Exception as e:
        out(f"    (within-subject comparison skipped: {e!r})")
    out()


# =========================== [F] bundle ===========================
def make_bundle():
    names = []
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUTDIR)):
            path = os.path.join(OUTDIR, f)
            if os.path.isfile(path) and os.path.abspath(path) != os.path.abspath(BUNDLE):
                z.write(path, arcname=f"results/{f}")
                names.append(f)
        for f in LOADER_FILES:
            if os.path.isfile(f):
                z.write(f, arcname=f"loaders/{f}")
                names.append(f)
    return names


def run():
    t_start = time.time()
    libs = S5._libs()
    csv = os.path.join(OUTDIR, "benchmark_subjects.csv")
    if not os.path.isdir(S5.CACHE) or not os.path.exists(csv):
        print(f"{S5.CACHE} or {csv} not found. Run step5_benchmark.py first, from the Codes folder.")
        return 1
    S5._LINES.clear()
    out("Step 7 final checks  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    main = load_cache()
    out("subjects in the main analysis: " + ", ".join(f"{d} {len(main[d])}" for d in S5.DATASETS))
    out()
    section_d(main, read_csv(csv), libs)
    section_e(main, libs)
    out(f"total time: {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(OUTDIR, "step7_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(S5._LINES) + "\n")
    names = make_bundle()
    print(f"[F] bundle written: {os.path.abspath(BUNDLE)} "
          f"({os.path.getsize(BUNDLE) / 1e6:.1f} MB, {len(names)} files)")
    print("    upload this one file in the chat")
    return 0


if __name__ == "__main__":
    sys.exit(run())
