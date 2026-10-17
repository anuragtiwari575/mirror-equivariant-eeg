r"""
make_paper_assets.py

Builds every table and every figure of the paper from corrected_results/step24_subject_specific.csv and the
cached epochs. Read-only: it trains nothing, needs no GPU, writes only into corrected_results/paper_assets/,
and is safe to run while a fit is going.

    python make_paper_assets.py                       the four corpora, the session protocol
    python make_paper_assets.py --datasets all        include every corpus present in the file
    python make_paper_assets.py --protocol cv         the cross-validation protocol instead

What it writes, into corrected_results/paper_assets/

    tables/table_datasets.tex      the corpora: subjects, channels, blocks, trials, class balance
    tables/table_accuracy.tex      accuracy and kappa of every model on every corpus
    tables/table_symmetry.tex      mirror consistency, asymmetry and lean of every model
    tables/table_contrasts.tex     the paired comparisons, with Holm-corrected p-values
    tables/table_learning.tex      the learning curve, where those fits exist

    figures/fig1_asymmetry.pdf     asymmetry of the decision function, by model  (the headline figure)
    figures/fig2_paired.pdf        per-subject accuracy, backbone against backbone + construction
    figures/fig3_learning.pdf      accuracy against the share of the training trials
    figures/fig4_confusion.pdf     confusion matrices of the main models
    figures/fig5_forest.pdf        every paired difference with its interval, in one forest plot
    figures/fig6_spread.pdf        the spread over subjects of every model

    per_subject.csv                one line per corpus, model and subject: what the tables are built from
    summary.txt                    everything above as plain text, to read in the terminal
    MANIFEST.txt                   what was built, from how many fits, and what was missing

Every figure is written as .pdf (for the manuscript) and .png (to look at). No number is typed by hand
anywhere in this file.

A note on the confusion matrices. The per-fit file stores accuracy and the lean (recall of the right hand
minus recall of the left), not the four cells. With the number of test trials of each class, which this script
recovers from the cached labels by repeating the split rule, the two recalls follow exactly from those two
numbers, and the four cells follow from the recalls. Where the cache is missing, the script falls back to
assuming the test set is balanced and says so in MANIFEST.txt.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages        # noqa: F401  (kept for one-file output if wanted)

# ---------------------------------------------------------------- one style for every figure
# Two settings here matter more than the choice of typeface.
#
# pdf.fonttype 42 embeds the text as TrueType outlines. Matplotlib's default is Type 3, which
# carries no hinting, is rendered softly by most viewers at the sizes used in a paper, and is
# refused by several publishers. This is the usual reason figure text looks blurred next to the
# body text of the same page while the lines around it stay sharp.
#
# FIGW is the text width of main.tex (A4 less its 2 cm margins). Every figure is drawn at that
# width, so \includegraphics[width=\textwidth] scales none of them. A figure drawn 16 in wide and
# set at 6.7 in has its 8 pt labels printed at 3.3 pt; one drawn 3 in wide has them printed at
# 18 pt. Keeping the width fixed is what makes the figures of a paper look like one set.
FIGW = 6.7

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "patch.linewidth": 0.8,
    "figure.dpi": 150,
})
if "Georgia" not in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
    print("  note: Georgia was not found; the figures fall back to the next serif in the list")


def panel_grid(n, ncol=3, cell=2.15, extra=0.70, sharey=False, width=FIGW):
    """n panels wrapped into rows of at most ncol, at the paper's text width.

    One row of five panels at this width leaves each of them 1.3 in across, which is where the
    unreadable axis labels came from. Wrapping keeps a panel wide enough to carry its own ticks.
    Unused cells of the last row are hidden rather than left as empty axes.
    """
    ncol = max(1, min(ncol, n))
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, cell * nrow + extra),
                             squeeze=False, sharey=sharey)
    flat = [a for row in axes for a in row]
    for a in flat[n:]:
        a.set_visible(False)
    return fig, flat[:n]

OUTDIR = "corrected_results"
CSV = os.path.join(OUTDIR, "step24_subject_specific.csv")
ASSETS = os.path.join(OUTDIR, "paper_assets")
FIGDIR = os.path.join(ASSETS, "figures")
TABDIR = os.path.join(ASSETS, "tables")

FOUR = ["2a", "2b", "PhysioNet", "Dreyer2023"]
ALL = ["2a", "2b", "PhysioNet", "Cho2017", "Dreyer2023"]
SHOWN = {"2a": "BCI IV 2a", "2b": "BCI IV 2b", "PhysioNet": "Motor Movement/Imagery",
         "Cho2017": "Cho", "Dreyer2023": "Dreyer"}
CITE = {"2a": r"\citep{brunner2008}", "2b": r"\citep{leeb2008}", "PhysioNet": r"\citep{schalk2004}",
        "Cho2017": r"\citep{cho2017}", "Dreyer2023": r"\citep{dreyer2023}"}

ORDER = ["csp_lda", "fbcsp", "ts_lr", "eegnet", "shallow", "deepnet", "tcnet", "conformer",
         "atcnet", "ctnet", "eegnet_cr", "shallow_cr", "eegnet_two", "shallow_two",
         "eegnet_rand", "shallow_rand", "eegnet_mirror", "shallow_mirror",
         "conformer_cr", "conformer_mirror"]
# The five families of decoder, in the order Section~\ref{sec:baselines} introduces them. A long table is
# broken into these blocks with a heading, so that a reader can see at a glance which rows are baselines,
# which carry the same prior as an augmentation, which are controls, and which are the construction.
GROUP = {}
GROUP.update({m: "Classical pipelines" for m in ("csp_lda", "fbcsp", "ts_lr")})
GROUP.update({m: "Convolutional networks" for m in ("eegnet", "shallow", "deepnet", "tcnet", "conformer",
                                                    "atcnet", "ctnet")})
GROUP.update({m: "The same prior as a training-time augmentation"
              for m in ("eegnet_cr", "shallow_cr", "conformer_cr")})
GROUP.update({m: "Controls that isolate what the symmetry buys"
              for m in ("eegnet_two", "shallow_two", "eegnet_rand", "shallow_rand")})
GROUP.update({m: "The construction" for m in ("eegnet_mirror", "shallow_mirror", "conformer_mirror")})

LABEL = {"csp_lda": "CSP + LDA", "fbcsp": "FBCSP", "ts_lr": "Tangent space + LR",
         "eegnet": "EEGNet", "shallow": "ShallowConvNet", "deepnet": "DeepConvNet",
         "tcnet": "EEG-TCNet", "conformer": "EEG-Conformer", "atcnet": "ATCNet", "ctnet": "CTNet",
         "eegnet_cr": "EEGNet + CR", "shallow_cr": "ShallowConvNet + CR",
         "eegnet_mirror": "EEGNet + mirror", "shallow_mirror": "ShallowConvNet + mirror",
         "conformer_cr": "EEG-Conformer + CR", "conformer_mirror": "EEG-Conformer + mirror",
         "eegnet_rand": "EEGNet + random involution", "shallow_rand": "ShallowConvNet + random involution",
         "eegnet_two": "EEGNet, two branches", "shallow_two": "ShallowConvNet, two branches"}
PAIRS = [("eegnet_mirror", "eegnet", "the construction against its backbone"),
         ("shallow_mirror", "shallow", "the construction against its backbone"),
         ("eegnet_mirror", "eegnet_cr", "the construction against the same symmetry as augmentation"),
         ("shallow_mirror", "shallow_cr", "the construction against the same symmetry as augmentation"),
         ("eegnet_cr", "eegnet", "the augmentation against its backbone"),
         ("shallow_cr", "shallow", "the augmentation against its backbone"),
         ("shallow_mirror", "eegnet_mirror", "which backbone carries the construction"),
         ("eegnet_mirror", "eegnet_rand", "against a random involution"),
         ("shallow_mirror", "shallow_rand", "against a random involution"),
         ("eegnet_mirror", "eegnet_two", "against two branches of the same cost"),
         ("shallow_mirror", "shallow_two", "against two branches of the same cost"),
         ("conformer_mirror", "conformer", "the construction against its backbone"),
         ("conformer_mirror", "conformer_cr", "the construction against the same symmetry as augmentation"),
         ("shallow_mirror", "ts_lr", "the best new model against the best classical pipeline")]

ABLATION = [("eegnet_mirror", "eegnet_rand", "the anatomical exchange against a random involution"),
            ("shallow_mirror", "shallow_rand", "the same control on the other backbone"),
            ("eegnet_mirror", "eegnet_two", "the construction against two branches of the same cost"),
            ("shallow_mirror", "shallow_two", "the same control on the other backbone")]

# Each contrast belongs to one family of questions. Holm's correction is applied WITHIN a family, which is
# the pre-declared primary analysis, and also across the whole table, which is the most conservative reading;
# both are reported so that a reader can take either. Families are a property of the question, not of the
# result: they were fixed by writing them here, and they do not change when the numbers do.
FAMILY = {("eegnet_mirror", "eegnet"): "construction vs backbone",
          ("shallow_mirror", "shallow"): "construction vs backbone",
          ("eegnet_mirror", "eegnet_cr"): "construction vs augmentation",
          ("shallow_mirror", "shallow_cr"): "construction vs augmentation",
          ("eegnet_cr", "eegnet"): "augmentation vs backbone",
          ("shallow_cr", "shallow"): "augmentation vs backbone",
          ("shallow_mirror", "eegnet_mirror"): "which backbone",
          ("eegnet_mirror", "eegnet_rand"): "ablation",
          ("shallow_mirror", "shallow_rand"): "ablation",
          ("eegnet_mirror", "eegnet_two"): "ablation",
          ("shallow_mirror", "shallow_two"): "ablation",
          ("conformer_mirror", "conformer"): "construction vs backbone",
          ("conformer_mirror", "conformer_cr"): "construction vs augmentation",
          ("shallow_mirror", "ts_lr"): "vs the best classical pipeline"}

# A random involution is only a control where there are enough channels for a random pairing to differ from
# the anatomical one. BCI IV 2b has three electrodes (C3, Cz, C4), so of the three non-identity involutions
# one IS the anatomical exchange; the control cannot be constructed there and the comparison is dropped.
NO_RANDOM_CONTROL = {"2b"}

# The epoch window each corpus is read at, as it appears in the first two columns of the result file.
# It is per corpus because the paradigms differ: in BCI IV 2b the arrow is shown for 1.25 s before the
# imagery begins, so a window starting at the cue would spend its first 1.25 s on no imagery at all.
# A corpus with no row at its preferred window falls back to whichever window it does have, and the
# choice is printed, so that two windows can never be averaged together without anyone noticing.
WINDOW = {"2b": "1.25-4.25"}
DEFAULT_WINDOW = "0-3"

# Short names, for the tables where the full one would not fit on the page.
SHORT = {"csp_lda": "CSP+LDA", "fbcsp": "FBCSP", "ts_lr": "TS+LR", "eegnet": "EEGNet", "shallow": "Shallow",
         "deepnet": "DeepConv", "tcnet": "TCNet", "conformer": "Conformer", "eegnet_cr": "EEGNet+CR",
         "shallow_cr": "Shallow+CR", "eegnet_mirror": "EEGNet+mir", "shallow_mirror": "Shallow+mir",
         "eegnet_rand": "EEGNet+rnd", "shallow_rand": "Shallow+rnd", "eegnet_two": "EEGNet+2br",
         "shallow_two": "Shallow+2br",
         # Without these two the contrast table falls back to the raw model name, and the underscore
         # in it is a subscript to LaTeX outside maths: the table compiled to "Missing $ inserted".
         "conformer_cr": "Conformer+CR", "conformer_mirror": "Conformer+mir"}
COL = {"accuracy": 9, "kappa": 10, "consistency": 11, "lean": 12, "asymmetry": 13}

# one colour-blind-safe set, used everywhere
C_BACKBONE, C_CR, C_MIRROR, C_CLASSIC = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C"


def colour(model):
    if model.endswith("_mirror"):
        return C_MIRROR
    if model.endswith("_cr"):
        return C_CR
    if model in ("csp_lda", "fbcsp", "ts_lr"):
        return C_CLASSIC
    return C_BACKBONE


# =========================== statistics ===========================
def _local_paired(a, b, n_boot=5000, seed=0):
    """mean difference in points, 95 % bootstrap interval, Cohen's dz, sign-flip permutation p."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    mean = float(np.mean(d))
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    dz = mean / sd if sd > 0 else 0.0
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, n))
    null = (flips * d).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(mean)) + 1) / (n_boot + 1))
    return mean, lo, hi, dz, p


def _study_paired(a, b):
    """The study's own function, when step5_benchmark can be imported, so that the tables and the console
    report of step24 cannot disagree. Falls back to the identical definitions implemented here."""
    try:
        import step5_benchmark as S5
        r = S5.paired(list(a), list(b))
    except Exception:
        return None
    try:
        if isinstance(r, dict):
            g = (lambda *k: next(r[x] for x in k if x in r))
            return (float(g("mean", "diff", "delta")), float(g("lo", "low", "ci_lo")),
                    float(g("hi", "high", "ci_hi")), float(g("dz", "d")), float(g("p", "pval", "p_value")))
        if isinstance(r, (tuple, list)) and len(r) >= 5:
            return tuple(float(v) for v in r[:5])
    except Exception:
        return None
    return None


def paired(a, b):
    """`a` and `b` are accuracies as fractions. Returns (mean, lo, hi, dz, p) with the mean and the interval
    in accuracy POINTS.

    The study's own function is used when it can be imported, so that these tables cannot disagree with the
    console report of step24. That function returns points while this file computes fractions, so the scale is
    established rather than assumed: the mean difference is deterministic, so comparing it against the
    locally computed one says which of the two scales came back. If it matches neither, the definitions differ
    and the local computation is used, which is the one documented in the paper."""
    lm, llo, lhi, ldz, lp = _local_paired(a, b)
    r = _study_paired(a, b)
    if r is not None:
        m = r[0]
        if abs(m - 100 * lm) <= max(1e-6, 1e-3 * abs(100 * lm)):
            return r                                                  # already in points
        if abs(m - lm) <= max(1e-9, 1e-3 * abs(lm)):
            return 100 * r[0], 100 * r[1], 100 * r[2], r[3], r[4]     # in fractions
    return 100 * lm, 100 * llo, 100 * lhi, ldz, lp


def holm(ps):
    """Holm-corrected p-values, in the order given."""
    ps = np.asarray(ps, dtype=np.float64)
    order = np.argsort(ps)
    out = np.empty_like(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


# =========================== reading ===========================
def read_rows(protocol):
    """{(dataset, model, subject): {column: value}} with one value per fit, folds and seeds averaged.
    A later line of the file replaces an earlier one with the same key, so a refitted cell counts once."""
    if not os.path.exists(CSV):
        print(f"{CSV} not found; run this from the Codes folder.")
        sys.exit(1)
    latest, n_lines, n_short, seen_win = {}, 0, 0, {}
    with open(CSV, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            n_lines += 1
            if len(p) < 11:                                  # a line still being written
                n_short += 1
                continue
            if p[4] != protocol:
                continue
            latest[tuple(p[:8])] = p
            seen_win[(p[2], p[1])] = seen_win.get((p[2], p[1]), 0) + 1
    # one window per corpus, chosen before anything is averaged
    # One window per corpus, and no substitute for it. A corpus whose fits at the wanted window are
    # missing for this protocol is dropped from it rather than read at another window: falling back
    # put the 25 % and 50 % points of BCI IV 2b's learning curve at 0-3 s and its 100 % point at
    # 1.25-4.25 s, which is two preprocessings joined into one line and not a curve at all.
    use = {}
    for ds in sorted({d for d, _w in seen_win}):
        want = WINDOW.get(ds, DEFAULT_WINDOW)
        have = {w: n for (d, w), n in seen_win.items() if d == ds}
        if not have.get(want):
            print(f"  note: {ds} has no fit at {want}s for protocol '{protocol}', so it is left out of it"
                  f" (it has {', '.join(f'{w}s ({n})' for w, n in sorted(have.items()))})")
            continue                                   # no entry in use[]: every row of it is dropped below
        use[ds] = want
        if len(have) > 1:
            other = ", ".join(f"{w}s ({n})" for w, n in have.items() if w != want)
            print(f"  note: {ds} read at {want}s; rows at {other} are in the file and are ignored")
    latest = {k: p for k, p in latest.items() if p[1] == use.get(p[2])}

    per_fit = defaultdict(lambda: defaultdict(list))
    for k, p in latest.items():
        for name, i in COL.items():
            if len(p) > i:
                try:
                    v = float(p[i])
                except ValueError:
                    continue
                if np.isfinite(v):
                    per_fit[(k[2], k[6], k[3])][name].append(v)
        per_fit[(k[2], k[6], k[3])]["n_test"].append(float(p[8]))
    out = {k: {c: float(np.mean(v)) for c, v in d.items()} for k, d in per_fit.items()}
    return out, len(latest), n_lines, n_short


def by_model(rows, dataset, model, column="accuracy"):
    """{subject: value} for one corpus and model."""
    return {s: d[column] for (ds, m, s), d in rows.items() if ds == dataset and m == model and column in d}


# =========================== the cache, for the confusion matrices ===========================
def _split_session(y, block, min_test=10, min_class=4):
    """The rule of step24_subject_specific.py, repeated here so that this script needs no GPU library.
    Any disagreement with the stored number of test trials makes the script fall back to assuming balance."""
    y, b = np.asarray(y), np.asarray(block)
    test = None
    if b.max() > 0:
        cand = np.nonzero(b == b.max())[0]
        if min(int((y[cand] == c).sum()) for c in (0, 1)) >= min_class:
            test = cand
    if test is None:
        test = np.sort(np.concatenate(
            [np.nonzero(y == c)[0][int(int((y == c).sum()) * 2 / 3):] for c in (0, 1)]))
    train = np.setdiff1d(np.arange(len(y)), test)
    if len(test) < min_test or len(train) < 2 * min_test:
        return None
    for idx in (train, test):
        if min(int((y[idx] == c).sum()) for c in (0, 1)) < min_class:
            return None
    return test


def test_class_counts(cache, dataset, subject, n_test):
    """(n_left, n_right) of the test set, from the cached labels; None if they cannot be recovered."""
    path = os.path.join(cache, f"{dataset}_{subject}.npz")
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            y, block = np.asarray(z["y"]), np.asarray(z["block"])
        if y.ndim != 1 or y.shape != block.shape or y.size == 0:
            return None
        te = _split_session(y, block)
    except Exception:                                         # a cache file this script cannot interpret
        return None
    if te is None or abs(len(te) - n_test) > 0.5:             # not the split these numbers came from
        return None
    return int((y[te] == 0).sum()), int((y[te] == 1).sum())


def confusion(acc, lean, n0, n1):
    """The four cells, in trials, from accuracy and lean. With r1 - r0 = lean and n0*r0 + n1*r1 = acc*(n0+n1),
    both recalls follow exactly. Rows are the true class, columns the predicted one."""
    n = n0 + n1
    r0 = (acc * n - n1 * lean) / n                            # recall of the left hand
    r1 = r0 + lean
    r0, r1 = float(np.clip(r0, 0, 1)), float(np.clip(r1, 0, 1))
    return np.array([[r0 * n0, (1 - r0) * n0], [(1 - r1) * n1, r1 * n1]])


# =========================== tables ===========================
def tex_table(path, caption, label, header, rows, align, size="small", fit=True, long=False, group=None,
              row_groups=None):
    """Write one table so that it fits the page, whatever the numbers turn out to be.

    size    font size command used inside the table
    fit     scale a float table down to the text width if it would otherwise run off the right edge
    long    use longtable, which breaks across pages with a repeated header; a table of more than about
            thirty rows must use it, because a float cannot break and silently loses its last rows
    group   index of a column whose repeated values are printed once per run and blanked afterwards,
            with a rule between runs; this is what keeps a corpus name from being repeated sixteen times
    row_groups  one family name per row; the table is broken into blocks with that name as a heading, which
            is what makes a sixteen-row table readable instead of one undifferentiated column of names
    """
    body = []
    last, lastfam = None, None
    for i, r in enumerate(rows):
        r = list(r)
        if group is not None:
            if r[group] == last:
                r[group] = ""
            else:
                if last is not None:
                    body.append("\\addlinespace[3pt]")
                last = r[group]
                lastfam = None                       # each block of the first column repeats the headings
        if row_groups is not None and row_groups[i] != lastfam:
            if lastfam is not None:
                body.append("\\addlinespace[2pt]")
            body.append(f"\\multicolumn{{{len(header)}}}{{@{{}}l}}{{\\itshape {row_groups[i]}}} \\\\[1pt]")
            lastfam = row_groups[i]
        body.append(" & ".join(r) + " \\\\")
    head = " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\"
    with open(path, "w", encoding="utf-8") as f:
        f.write("% written by make_paper_assets.py. Do not edit by hand.\n")
        if long:
            f.write(f"\\begingroup\n\\{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n")
            f.write(f"\\begin{{longtable}}{{{align}}}\n")
            f.write(f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n")
            f.write(f"\\toprule\n{head}\n\\midrule\n\\endfirsthead\n")
            f.write(f"\\multicolumn{{{len(header)}}}{{l}}{{\\textit{{Table \\thetable{{}} continued.}}}}\\\\\n")
            f.write(f"\\toprule\n{head}\n\\midrule\n\\endhead\n")
            f.write(f"\\midrule\n\\multicolumn{{{len(header)}}}{{r}}{{\\textit{{continued on the next page}}}}\\\\\n"
                    "\\endfoot\n\\bottomrule\n\\endlastfoot\n")
            f.write("\n".join(body) + "\n")
            f.write("\\end{longtable}\n\\endgroup\n")
        else:
            f.write("\\begin{table}[!tbp]\n\\centering\n")
            f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
            f.write(f"\\{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n")
            if fit:
                f.write("\\resizebox{\\textwidth}{!}{%\n")
            f.write(f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n")
            f.write("\n".join(body) + "\n")
            f.write("\\bottomrule\n\\end{tabular}%\n")
            if fit:
                f.write("}\n")
            f.write("\\end{table}\n")


def table_datasets(cache, datasets, out):
    rows, txt = [], []
    for ds in datasets:
        files = sorted(f for f in os.listdir(cache) if f.startswith(f"{ds}_") and f.endswith(".npz")) \
            if os.path.isdir(cache) else []
        if not files:
            continue
        ch, blocks, trials, share = set(), [], [], []
        for fn in files:
            try:
                with np.load(os.path.join(cache, fn), allow_pickle=False) as z:
                    ch.add(len(z["names"]))
                    blocks.append(int(np.unique(z["block"]).size))
                    trials.append(int(z["X"].shape[0]))
                    share.append(float(np.mean(np.asarray(z["y"]) == 1)))
            except Exception:
                continue
        if not trials:
            continue

        def span(v):
            v = np.asarray(v, float)
            return f"{np.median(v):g}" if v.min() == v.max() else f"{np.median(v):g} [{v.min():g}--{v.max():g}]"
        rows.append([f"{SHOWN.get(ds, ds)} {CITE.get(ds, '')}", str(len(trials)),
                     "/".join(str(c) for c in sorted(ch)), span(blocks), span(trials),
                     f"{100 * float(np.median(share)):.1f}"])
        txt.append(f"{ds:<12}{len(trials):>9}{'/'.join(str(c) for c in sorted(ch)):>10}"
                   f"{span(blocks):>14}{span(trials):>18}{100 * float(np.median(share)):>13.1f}")
    tex_table(os.path.join(TABDIR, "table_datasets.tex"),
              "The corpora as they are decoded here: left hand against right hand, all EEG channels of the "
              "recording, 4--38 Hz, 0 to 3 s after the cue, 250 Hz. Blocks are the files a subject was "
              "recorded in. Medians over subjects, with the range where subjects differ.",
              "tab:datasets",
              ["Corpus", "Subjects", "Channels", "Blocks", "Trials", "Right hand (\\%)"], rows,
              "lrrrrr")
    out.append(f"{'corpus':<12}{'subjects':>9}{'channels':>10}{'blocks':>14}{'trials':>18}{'right hand %':>14}")
    out += txt
    out.append("")


def table_accuracy(rows, datasets, models, out):
    tex, txt, fams = [], [], []
    head = [f"{'model':<24}"] + [f"{d:>22}" for d in datasets]
    txt.append("".join(head))
    for m in models:
        cells, cells_txt = [], ""
        for d in datasets:
            a, k = by_model(rows, d, m), by_model(rows, d, m, "kappa")
            if a:
                mean = 100 * float(np.mean(list(a.values())))
                sd = 100 * float(np.std(list(a.values()), ddof=1)) if len(a) > 1 else 0.0
                kp = float(np.mean(list(k.values()))) if k else float("nan")
                cells.append(f"{mean:.2f} $\\pm$ {sd:.2f}")
                cells_txt += f"{mean:>12.2f} ({kp:.2f}) {len(a):>3}"
            else:
                cells.append("--")
                cells_txt += f"{'-':>22}"
        if any(c != "--" for c in cells):
            tex.append([LABEL.get(m, m)] + cells)
            fams.append(GROUP.get(m, "Other"))
            txt.append(f"{LABEL.get(m, m):<24}{cells_txt}")
    tex_table(os.path.join(TABDIR, "table_accuracy.tex"),
              "Accuracy (\\%), mean $\\pm$ standard deviation over subjects, for the held-out-block protocol. "
              "Every subject contributes the mean over its folds and seeds. Chance is 50\\%.",
              "tab:accuracy", ["Model"] + [SHOWN.get(d, d) for d in datasets], tex,
              "l" + "r" * len(datasets), size="small", row_groups=fams)
    out.append("Accuracy %, mean over subjects (kappa in brackets, then the number of subjects)")
    out += txt
    out.append("")


def table_symmetry(rows, datasets, models, out):
    """The asymmetry of the decision function, one number per cell. Consistency and lean say different
    things and live in the metrics table; packing three numbers into one cell made this one unreadable."""
    tex, txt, fams = [], [], []
    txt.append(f"{'model':<26}" + "".join(f"{d:>14}" for d in datasets))
    for m in models:
        cells, cells_txt = [], ""
        for d in datasets:
            a = by_model(rows, d, m, "asymmetry")
            if a:
                v = float(np.mean(list(a.values())))
                cells.append("\\textbf{0}" if v == 0 else f"{v:.2f}")
                cells_txt += f"{v:>14.3f}"
            else:
                cells.append("--")
                cells_txt += f"{'-':>14}"
        if any(c != "--" for c in cells):
            tex.append([LABEL.get(m, m)] + cells)
            fams.append(GROUP.get(m, "Other"))
            txt.append(f"{LABEL.get(m, m):<26}{cells_txt}")
    tex_table(os.path.join(TABDIR, "table_symmetry.tex"),
              "Asymmetry (\\ref{eq:asym}) of each decoder's decision function: 0 when the decoder exchanges "
              "its two classes exactly as the hemispheres of its input are exchanged, 1 when the exchange "
              "never changes which class is favoured. The construction is exactly 0 everywhere, by "
              "Proposition~\\ref{prop:equivariance}. The classical pipelines have no real-valued score of "
              "this kind. Mirror consistency and lean are in Table~\\ref{tab:metrics}.",
              "tab:symmetry", ["Model"] + [SHOWN.get(d, d) for d in datasets], tex,
              "l" + "c" * len(datasets), size="small", row_groups=fams)
    out.append("asymmetry of the decision function")
    out += txt
    out.append("")


def pstr(v):
    """A p-value for prose: a floor rather than a rounded zero, then four
    decimals below 0.001 and three above it."""
    if not np.isfinite(v):
        return "---"
    return "<0.0001" if v < 0.00005 else (f"{v:.4f}" if v < 0.001 else f"{v:.3f}")


def pcell(v):
    """A p-value for a table column: uniform width, but never 0.0000, which
    would assert that a permutation test resolved a difference exactly."""
    if not np.isfinite(v):
        return "---"
    return "<0.0001" if v < 0.00005 else f"{v:.4f}"


def table_contrasts(rows, datasets, out):
    recs = []
    for d in datasets:
        for a, b, note in PAIRS:
            if b.endswith("_rand") and d in NO_RANDOM_CONTROL:
                continue
            va, vb = by_model(rows, d, a), by_model(rows, d, b)
            keys = sorted(set(va) & set(vb))
            if len(keys) < 3:
                continue
            mean, lo, hi, dz, p = paired([va[s] for s in keys], [vb[s] for s in keys])
            recs.append([d, a, b, note, mean, lo, hi, dz, p, len(keys)])
    if not recs:
        out.append("(no contrast had enough subjects yet)\n")
        return []
    adj = holm([r[8] for r in recs])                               # across the whole table
    fam = [FAMILY.get((r[1], r[2]), "other") for r in recs]         # and within each family of questions
    fadj = [float("nan")] * len(recs)
    for name in sorted(set(fam)):
        idx = [i for i, f in enumerate(fam) if f == name]
        for i, pa in zip(idx, holm([recs[i][8] for i in idx])):
            fadj[i] = float(pa)
    tex, txt = [], []
    for r, pa, pf in zip(recs, adj, fadj):
        d, a, b, note, mean, lo, hi, dz, p, n = r
        tex.append([SHOWN.get(d, d), f"{SHORT.get(a, a)} $-$ {SHORT.get(b, b)}", f"{mean:+.2f}",
                    f"[{lo:+.2f}, {hi:+.2f}]", f"{dz:+.2f}", pcell(p), pcell(pf), pcell(pa), str(n)])
        txt.append(f"{d:>12}  {a + ' - ' + b:<36}{mean:>+7.2f} [{lo:>+6.2f}, {hi:>+6.2f}] "
                   f"dz={dz:>+5.2f} p={pstr(p)} p_fam={pstr(pf)} p_all={pstr(pa)} n={n}   {note}")
    tex_table(os.path.join(TABDIR, "table_contrasts.tex"),
              "Paired differences in accuracy points over subjects, with a 95\\% bootstrap interval (5000 "
              "resamples), Cohen's $d_z$, and the two-sided $p$-value of a sign-flip permutation test. The "
              "last two $p$-columns are that value after Holm's correction within the family of questions the "
              "contrast belongs to, and after Holm's correction across every contrast in the table. The "
              "families were fixed in advance and are listed in Section~\\ref{sec:stats}; both corrections "
              "are given so that the reader may take either. A $p$ below $10^{-4}$ is written $<0.0001$ "
              "rather than rounded, since a permutation test over 5000 resamples cannot resolve a value "
              "to zero. $n$ is the number of subjects for which both decoders of the row produced a fit, "
              "and it is the per-row value rather than the corpus size that the test was run on.",
              "tab:contrasts",
              ["Corpus", "Contrast", "Diff.", "95\\% CI", "$d_z$", "$p$",
               "$p_{\\mathrm{fam}}$", "$p_{\\mathrm{all}}$", "$n$"],
              tex, "llrrrrrrr", size="footnotesize", long=True, group=0)
    out.append("Paired differences, points [95% CI] dz p (Holm within family, then across the whole table)")
    out += txt
    out.append("")
    return [r + [float(pa), float(pf)] for r, pa, pf in zip(recs, adj, fadj)]


def table_learning(protocol, datasets, models, out):
    """The learning curve, read from the session@<fraction> rows, if any were computed."""
    fracs = []
    with open(CSV, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) > 8 and p[4].startswith(protocol + "@"):
                fracs.append(p[4].split("@", 1)[1])
    fracs = sorted(set(fracs), key=float)
    if not fracs:
        out.append("(no learning-curve fits in the file)\n")
        return None
    curves = {}
    for fr in fracs + ["1"]:
        pr = protocol if fr == "1" else f"{protocol}@{fr}"
        r, _, _, _ = read_rows(pr)
        for d in datasets:
            for m in models:
                v = by_model(r, d, m)
                if v:
                    curves[(d, m, float(fr))] = (100 * float(np.mean(list(v.values()))), len(v))
    tex, txt = [], []
    shown_fr = sorted({k[2] for k in curves})
    txt.append(f"{'corpus':<12}{'model':<24}" + "".join(f"{100 * fr:>12.0f}%" for fr in shown_fr))
    partial = [fr for fr in shown_fr if fr < 1.0]
    for d in datasets:
        for m in models:
            cells = [f"{curves[(d, m, fr)][0]:.2f}" if (d, m, fr) in curves else "--" for fr in shown_fr]
            # a model that was only ever fitted on the full training set has no curve and would fill the
            # table with rows of dashes; it is already in Table~\ref{tab:accuracy}
            if any((d, m, fr) in curves for fr in partial):
                tex.append([SHOWN.get(d, d), SHORT.get(m, m)] + cells)
                txt.append(f"{d:<12}{LABEL.get(m, m):<24}" + "".join(f"{c:>13}" for c in cells))
    tex_table(os.path.join(TABDIR, "table_learning.tex"),
              "Accuracy (\\%) against the share of the training trials used; the test set is the same in "
              "every column. The right-most column is the full training set.",
              "tab:learning", ["Corpus", "Model"] + [f"{100 * fr:.0f}\\%" for fr in shown_fr], tex,
              "ll" + "r" * len(shown_fr), size="small", group=0)
    out.append("Learning curve: accuracy % against the share of the training trials")
    out += txt
    out.append("")
    return curves


def write_numbers(rows, datasets, models, contrasts):
    r"""numbers.tex: one LaTeX macro per quantity, so that the prose of the paper can quote a number without
    anyone typing it. \accPhysioShallowMirror, \asymDreyerShallowCr, \nsubDreyer, and for every contrast
    \diffDreyerShallowMirrorVShallowCr together with \ci..., \dz..., \pu... (uncorrected) and \ph... (Holm)."""
    ds_tag = {"2a": "Twoa", "2b": "Twob", "PhysioNet": "Physio", "Cho2017": "Cho", "Dreyer2023": "Dreyer"}
    md_tag = {"csp_lda": "CspLda", "fbcsp": "Fbcsp", "ts_lr": "TsLr", "eegnet": "Eegnet", "shallow": "Shallow",
              "deepnet": "Deepnet", "tcnet": "Tcnet", "conformer": "Conformer", "atcnet": "Atcnet",
              "ctnet": "Ctnet", "eegnet_cr": "EegnetCr", "shallow_cr": "ShallowCr",
              "eegnet_mirror": "EegnetMirror", "shallow_mirror": "ShallowMirror",
              "eegnet_rand": "EegnetRand", "shallow_rand": "ShallowRand",
              "eegnet_two": "EegnetTwo", "shallow_two": "ShallowTwo",
              "conformer_cr": "ConformerCr", "conformer_mirror": "ConformerMirror"}

    def tag(d, m=None):
        return ds_tag.get(d, re.sub(r"[^A-Za-z]", "", d)) + (md_tag.get(m, re.sub(r"[^A-Za-z]", "", m or "")))
    lines = [r"% numbers.tex -- written by make_paper_assets.py. Every number the prose quotes lives here,",
             r"% so that no result is typed by hand. Rebuild it whenever the results change."]
    for d in datasets:
        n = max((len(by_model(rows, d, m)) for m in models), default=0)
        lines.append(f"\\newcommand{{\\nsub{tag(d)}}}{{{n}}}")
        for m in models:
            a = by_model(rows, d, m)
            if a:
                lines.append(f"\\newcommand{{\\acc{tag(d, m)}}}{{{100 * float(np.mean(list(a.values()))):.2f}}}")
            s = by_model(rows, d, m, "asymmetry")
            if s:
                v = float(np.mean(list(s.values())))
                lines.append(f"\\newcommand{{\\asym{tag(d, m)}}}{{{'0' if v == 0 else f'{v:.2f}'}}}")
            c = by_model(rows, d, m, "consistency")
            if c:
                lines.append(f"\\newcommand{{\\cons{tag(d, m)}}}{{{100 * float(np.mean(list(c.values()))):.1f}}}")
            ln = by_model(rows, d, m, "lean")
            if ln:
                lines.append(f"\\newcommand{{\\lean{tag(d, m)}}}{{{100 * float(np.mean(list(ln.values()))):+.1f}}}")
    for d, a, b, _note, mean, lo, hi, dz, p, n, ph, pf in contrasts:
        key = f"{tag(d, a)}V{md_tag.get(b, re.sub(r'[^A-Za-z]', '', b))}"
        lines += [f"\\newcommand{{\\diff{key}}}{{{mean:+.2f}}}",
                  f"\\newcommand{{\\ci{key}}}{{[{lo:+.2f}, {hi:+.2f}]}}",
                  f"\\newcommand{{\\dz{key}}}{{{dz:+.2f}}}",
                  f"\\newcommand{{\\pu{key}}}{{{pstr(p)}}}",
                  f"\\newcommand{{\\pf{key}}}{{{pstr(pf)}}}",
                  f"\\newcommand{{\\ph{key}}}{{{pstr(ph)}}}",
                  f"\\newcommand{{\\nn{key}}}{{{n}}}"]

    # How the construction does subject by subject against its own backbone, over every corpus and
    # both backbones at once. A mean difference can be positive while a large share of the subjects
    # are behind, and on this data they are: the paper quotes these three counts rather than saying
    # that accuracy is never harmed, which is true only of the means.
    won = lost = tied = 0
    for d in datasets:
        for base, mir in (("eegnet", "eegnet_mirror"), ("shallow", "shallow_mirror")):
            A, B = by_model(rows, d, base), by_model(rows, d, mir)
            for s in sorted(set(A) & set(B)):
                won += B[s] > A[s]
                lost += B[s] < A[s]
                tied += B[s] == A[s]
    lines += [f"\\newcommand{{\\subjAhead}}{{{won}}}",
              f"\\newcommand{{\\subjBehind}}{{{lost}}}",
              f"\\newcommand{{\\subjEqual}}{{{tied}}}",
              f"\\newcommand{{\\subjTotal}}{{{won + lost + tied}}}"]

    with open(os.path.join(ASSETS, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines) - 2



def table_metrics(rows, datasets, models, cache, out):
    """Sensitivity, specificity, precision, F1 and balanced accuracy, from the confusion matrix of each
    subject. With the number of test trials of each class these follow exactly from the stored accuracy and
    lean, so no new fit is needed; the rows where the lean was not recorded are left out and counted."""
    tex, txt, fams, skipped = [], [], [], 0
    txt.append(f"{'corpus':<12}{'model':<26}{'sens':>8}{'spec':>8}{'prec':>8}{'F1':>8}{'bal acc':>9}"
               f"{'MCC':>8}{'cons':>8}{'lean':>8}{'n':>5}")
    for d in datasets:
        for m in models:
            acc, lean, nt = by_model(rows, d, m), by_model(rows, d, m, "lean"), by_model(rows, d, m, "n_test")
            keys = sorted(set(acc) & set(lean) & set(nt))
            if not keys:
                skipped += 1
                continue
            M = np.zeros((2, 2))
            for sub in keys:
                counts = test_class_counts(cache, d, sub, nt[sub]) or (nt[sub] / 2.0, nt[sub] / 2.0)
                M += confusion(acc[sub], lean[sub], *counts)
            tn, fp, fn, tp = M[0, 0], M[0, 1], M[1, 0], M[1, 1]      # class 1 (right hand) is the positive one
            sens = tp / (tp + fn) if tp + fn else float("nan")
            spec = tn / (tn + fp) if tn + fp else float("nan")
            prec = tp / (tp + fp) if tp + fp else float("nan")
            f1 = 2 * prec * sens / (prec + sens) if prec + sens else float("nan")
            bal = 0.5 * (sens + spec)
            den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))   # Matthews correlation coefficient
            mcc = (tp * tn - fp * fn) / den if den else float("nan")
            cv = by_model(rows, d, m, "consistency")
            cons = f"{100 * float(np.mean(list(cv.values()))):.1f}" if cv else "--"
            lv = float(np.mean(list(lean[s_] for s_ in keys)))
            tex.append([SHOWN.get(d, d), SHORT.get(m, m)] + [f"{100 * v:.1f}" for v in (sens, spec, prec, f1, bal)]
                       + [f"{mcc:.3f}", cons, f"{100 * lv:+.1f}", str(len(keys))])
            fams.append(GROUP.get(m, "Other"))
            txt.append(f"{d:<12}{LABEL.get(m, m):<26}" + "".join(f"{100 * v:>8.1f}" for v in (sens, spec, prec, f1))
                       + f"{100 * bal:>9.1f}{mcc:>8.3f}{cons:>8}{100 * lv:>+8.1f}{len(keys):>5}")
    if not tex:
        out.append("(no model had the lean recorded, so no metrics table)\n")
        return
    tex_table(os.path.join(TABDIR, "table_metrics.tex"),
              "Per-class behaviour, with the right hand as the positive class, pooled over the subjects of "
              "each corpus. Sensitivity, specificity, precision, $F_1$, balanced accuracy and the Matthews "
              "correlation coefficient follow exactly from the accuracy and the lean of each fit together "
              "with the number of test trials of each class. The last two columns are the mirror "
              "consistency (\\ref{eq:cons}), the share of test trials whose predicted class changes when "
              "the hemispheres are exchanged, and the lean, the recall of the right hand minus the recall "
              "of the left. Every column of this table is computed from the stored per-trial scores of a "
              "fit, which the accuracy of Table~\\ref{tab:accuracy} does not require, so $n$ here is the "
              "number of subjects whose per-trial scores were retained rather than the number who were "
              "decoded. It equals the corpus everywhere except for four decoders on Cho, where the scores "
              "of one subject were not retained and $n$ is 51 against the 52 subjects that corpus "
              "contributes to every accuracy and every contrast; no subject is excluded from any other "
              "table on this ground.",
              "tab:metrics",
              ["Corpus", "Model", "Sens.", "Spec.", "Prec.", "$F_1$", "Bal. acc.", "MCC",
               "Cons.", "Lean", "$n$"],
              tex, "llrrrrrrrrr", size="footnotesize", long=True, group=0, row_groups=fams)
    out.append("Sensitivity / specificity / precision / F1 / balanced accuracy / MCC (right hand positive)")
    out += txt
    out.append("")


def read_scores(protocol):
    """{(dataset, model, subject): (y, score)} from step24_scores.csv, if that file exists."""
    path = os.path.join(OUTDIR, "step24_scores.csv")
    if not os.path.exists(path):
        return {}
    acc = defaultdict(lambda: ([], []))
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 9 or p[2] != protocol:
                continue
            try:
                y, sc = int(p[7]), float(p[8])
            except ValueError:
                continue
            if np.isfinite(sc):
                acc[(p[0], p[4], p[1])][0].append(y)
                acc[(p[0], p[4], p[1])][1].append(sc)
    return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in acc.items()}


def read_mirror_scores(protocol, dataset=None):
    """{(dataset, model): (score, mirrored score)} for the fits that recorded both, which the study does
    whenever --scores is given. This is what lets the central identity be drawn rather than asserted."""
    path = os.path.join(OUTDIR, "step24_scores.csv")
    if not os.path.exists(path):
        return {}
    acc = defaultdict(lambda: ([], []))
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 10 or p[2] != protocol or not p[9].strip():
                continue
            if dataset and p[0] != dataset:
                continue
            try:
                sc, scm = float(p[8]), float(p[9])
            except ValueError:
                continue
            if np.isfinite(sc) and np.isfinite(scm):
                acc[(p[0], p[4])][0].append(sc)
                acc[(p[0], p[4])][1].append(scm)
    return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in acc.items()}


def fig_antisymmetry(protocol, built, out):
    """s(x) against s(Mx), one panel per decoder. The construction lies on the line s(Mx) = -s(x) exactly;
    the augmentation scatters around it; the plain backbone has no relation to it. This is the paper's
    central claim drawn rather than stated."""
    sc = read_mirror_scores(protocol)
    if not sc:
        out.append("(no mirrored scores in step24_scores.csv, so no antisymmetry figure; "
                   "re-run the study with --scores)\n")
        return
    want = [("eegnet", "shallow"), ("eegnet_cr", "shallow_cr"), ("eegnet_mirror", "shallow_mirror")]
    ds = sorted({k[0] for k in sc}, key=lambda d: -sum(len(sc[k][0]) for k in sc if k[0] == d))[0]
    show = [m for pair in want for m in pair if (ds, m) in sc]
    if not show:
        return
    fig, axes = plt.subplots(1, len(show), figsize=(6.9, 6.9 / len(show) + 0.85), squeeze=False)
    for j, m in enumerate(show):
        ax = axes[0][j]
        x, ym = sc[(ds, m)]
        lim = float(np.percentile(np.abs(np.concatenate([x, ym])), 99.5)) or 1.0
        ax.plot([-lim, lim], [lim, -lim], color="0.55", lw=1.0, ls="--", zorder=1)
        ax.scatter(x, ym, s=4, alpha=0.35, color=colour(m), edgecolor="none", zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(SHORT.get(m, m), fontsize=8, pad=3)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0); ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_xticks([-lim, 0, lim]); ax.set_yticks([-lim, 0, lim])
        ax.set_xticklabels(["", "0", ""], fontsize=7.5); ax.set_yticklabels(["", "0", ""], fontsize=7.5)
        if j == 0:
            ax.set_ylabel(r"$s(Mx)$", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel(r"$s(x)$, the score of the trial as recorded", fontsize=9, y=0.03)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig10_antisymmetry")
    built.append("fig10_antisymmetry")
    out.append(f"Antisymmetry figure drawn on {SHOWN.get(ds, ds)}; "
               f"the dashed line is s(Mx) = -s(x)\n")


def roc_one(y, sc, grid):
    """The ROC of one subject, interpolated on a common grid of false-positive rates, and its area."""
    order = np.argsort(-sc)
    y = np.asarray(y)[order]
    pos, neg = float((y == 1).sum()), float((y == 0).sum())
    if pos == 0 or neg == 0:
        return None, float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(y == 1) / pos])
    fpr = np.concatenate([[0.0], np.cumsum(y == 0) / neg])
    return np.interp(grid, fpr, tpr), float(np.trapezoid(tpr, fpr))


def fig_roc(datasets, protocol, built, out):
    sc = read_scores(protocol)
    if not sc:
        out.append("(no step24_scores.csv, so no ROC curves; run the study with --scores)\n")
        return
    show = [m for m in ("ts_lr", "eegnet", "eegnet_cr", "eegnet_mirror", "shallow", "shallow_cr",
                        "shallow_mirror") if any(k[1] == m for k in sc)]
    ds = [d for d in datasets if any(k[0] == d for k in sc)]
    if not show or not ds:
        return
    grid = np.linspace(0, 1, 201)
    fig, axes = panel_grid(len(ds), ncol=2, cell=2.75, extra=0.95)
    lines = [f"{'corpus':<12}{'model':<26}{'AUC':>8}{'n':>5}"]
    for j, d in enumerate(ds):
        ax = axes[j]
        for m in show:
            curves, aucs = [], []
            for (dd, mm, sub), (y, v) in sc.items():
                if dd == d and mm == m:
                    c, a = roc_one(y, v, grid)
                    if c is not None:
                        curves.append(c)
                        aucs.append(a)
            if curves:
                # colour and dash say plain, augmented or constructed; the weight says which
                # backbone, so that the two curves of a family are not drawn identically.
                ax.plot(grid, np.mean(curves, axis=0),
                        lw=2.0 if m.startswith("shallow") else 1.1, color=colour(m),
                        ls="-" if m.endswith("_mirror") else ("--" if m.endswith("_cr") else ":"),
                        label=f"{LABEL.get(m, m)} ({np.mean(aucs):.3f})")
                lines.append(f"{d:<12}{LABEL.get(m, m):<26}{np.mean(aucs):>8.3f}{len(aucs):>5}")
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
        ax.set_title(SHOWN.get(d, d), fontsize=10)
        ax.set_xlabel("false positive rate", fontsize=9)
        if j == 0:
            ax.set_ylabel("true positive rate", fontsize=9)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(frameon=False, fontsize=6.6, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle("ROC, averaged over subjects at each false-positive rate; mean AUC in brackets", fontsize=9)
    fig.tight_layout()
    save(fig, "fig7_roc")
    built.append("fig7_roc")
    out.append("Mean area under the ROC curve, over subjects")
    out += lines
    out.append("")


def fig_curves(datasets, protocol, built, out):
    path = os.path.join(OUTDIR, "step24_curves.csv")
    if not os.path.exists(path):
        out.append("(no step24_curves.csv, so no training curves; run the study with --curves)\n")
        return
    acc = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 8 or p[2] != protocol:
                continue
            try:
                acc[(p[0], p[4])][int(p[6])].append(float(p[7]))
            except ValueError:
                continue
    if not acc:
        return
    ds = [d for d in datasets if any(k[0] == d for k in acc)]
    if not ds:                      # curves exist, but for none of the datasets asked for
        have = ", ".join(sorted({k[0] for k in acc})) or "none"
        out.append(f"(no training curves for the datasets selected; there are curves for: {have})\n")
        return
    fig, axes = panel_grid(len(ds), ncol=2, cell=2.45, extra=0.95)
    for j, d in enumerate(ds):
        ax = axes[j]
        for m in sorted({k[1] for k in acc if k[0] == d}, key=lambda z: ORDER.index(z) if z in ORDER else 99):
            ep = sorted(acc[(d, m)])
            ax.plot(ep, [float(np.mean(acc[(d, m)][e])) for e in ep],
                    lw=1.9 if m.startswith("shallow") else 1.0, color=colour(m),
                    ls="-" if m.endswith("_mirror") else ("--" if m.endswith("_cr") else ":"),
                    label=LABEL.get(m, m))
        ax.set_title(SHOWN.get(d, d), fontsize=10)
        ax.set_xlabel("epoch", fontsize=9)
        if j == 0:
            ax.set_ylabel("training loss", fontsize=9)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle("Training loss, mean over subjects", fontsize=9)
    fig.tight_layout()
    save(fig, "fig8_curves")
    built.append("fig8_curves")


def fig_ablation(rows, datasets, built, out):
    recs = []
    for d in datasets:
        for a, b, note in ABLATION:
            if b.endswith("_rand") and d in NO_RANDOM_CONTROL:     # see NO_RANDOM_CONTROL
                continue
            va, vb = by_model(rows, d, a), by_model(rows, d, b)
            keys = sorted(set(va) & set(vb))
            if len(keys) >= 3:
                mean, lo, hi, dz, p = paired([va[s] for s in keys], [vb[s] for s in keys])
                recs.append((f"{SHOWN.get(d, d)}:  {SHORT.get(a, a)} $-$ {SHORT.get(b, b)}".replace("$-$", "\u2212"),
                             mean, lo, hi, p, len(keys), note))
    if not recs:
        out.append("(the control models have not been fitted yet, so no ablation figure)\n")
        return
    fig, ax = plt.subplots(figsize=(6.6, 0.30 * len(recs) + 1.0))
    ys = np.arange(len(recs))[::-1]
    for y, r in zip(ys, recs):
        ax.plot([r[2], r[3]], [y, y], color="0.4", lw=1.4)
        ax.plot([r[1]], [y], "o", ms=5, color=C_MIRROR if r[1] > 0 else C_CR)
    ax.axvline(0, color="0.5", lw=0.9, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in recs], fontsize=8)
    ax.grid(axis="x", color="0.92", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("difference in accuracy points (95 % bootstrap interval)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig9_ablation")
    built.append("fig9_ablation")
    out.append("Ablation: the construction against its controls")
    for r in recs:
        label = r[0].replace("\u2212", "-")               # the figure keeps the typeset minus, the text does not
        out.append(f"      {label:<58}{r[1]:>+7.2f} [{r[2]:>+6.2f}, {r[3]:>+6.2f}] p={r[4]:.4f} n={r[5]}   {r[6]}")
    out.append("")


# =========================== figures ===========================
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_asymmetry(rows, datasets, models, built):
    """Horizontal bars: the model names are long, and on a vertical axis they are read without rotation."""
    ms = [m for m in models if any(by_model(rows, d, m, "asymmetry") for d in datasets)]
    if not ms:
        return
    ms = ms[::-1]                                              # first model at the top
    ds = [d for d in datasets if any(by_model(rows, d, m, "asymmetry") for m in ms)]
    h = 0.30 * len(ms) + 1.35
    fig, ax = plt.subplots(figsize=(6.6, h))
    step = 0.80 / max(1, len(ds))
    zeros = []
    for j, d in enumerate(ds):
        ys, vs = [], []
        for i, m in enumerate(ms):
            v = by_model(rows, d, m, "asymmetry")
            if v:
                y = i + (j - (len(ds) - 1) / 2) * step
                val = float(np.mean(list(v.values())))
                ys.append(y)
                vs.append(val)
                if val <= 0:
                    zeros.append(i)                            # the model row, not the individual bar
        if ys:
            ax.barh(ys, vs, height=step * 0.88, label=SHOWN.get(d, d), alpha=0.92, zorder=2)
    # a bar of length zero is invisible, so the value is written where the bar would have started
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([LABEL.get(m, m) for m in ms], fontsize=8.5)
    ax.set_xlim(0, max(0.1, 1.04 * max((float(np.mean(list(by_model(rows, d, m, "asymmetry").values())))
                                        for d in ds for m in ms if by_model(rows, d, m, "asymmetry")),
                                       default=1.0)))
    ax.set_xlabel(r"asymmetry $A$   (0 = exchanges its classes exactly, 1 = never)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="0.9", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for i in sorted(set(zeros)):                               # one label per model, not one per dataset
        ax.text(0.01 * ax.get_xlim()[1], i, "exactly 0", fontsize=8, va="center", ha="left",
                fontweight="bold", color="0.15", zorder=5)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right", handlelength=1.4, handleheight=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig1_asymmetry")
    built.append("fig1_asymmetry")


def fig_paired(rows, datasets, built):
    """One panel per (backbone, dataset). Tick labels and axis titles are drawn once per row and column
    instead of once per panel, which is what makes ten panels legible across a text width."""
    pairs = [("eegnet", "eegnet_mirror"), ("shallow", "shallow_mirror")]
    cols = [d for d in datasets if any(by_model(rows, d, b) and by_model(rows, d, a) for a, b in pairs)]
    if not cols:
        return
    w = 6.9
    side = w / len(cols)
    fig, axes = plt.subplots(len(pairs), len(cols), figsize=(w, side * len(pairs) + 0.75),
                             squeeze=False)
    for i, (base, mir) in enumerate(pairs):
        for j, d in enumerate(cols):
            ax = axes[i][j]
            a, b = by_model(rows, d, base), by_model(rows, d, mir)
            keys = sorted(set(a) & set(b))
            if keys:
                x = [100 * a[s_] for s_ in keys]
                y = [100 * b[s_] for s_ in keys]
                lo = max(0, min(min(x), min(y)) - 4)
                hi = min(100, max(max(x), max(y)) + 4)
                ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.9, ls="--", zorder=1)
                ax.scatter(x, y, s=13, color=C_MIRROR, edgecolor="white", linewidth=0.4, zorder=3)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_xticks([int(lo) + 5 - int(lo) % 5, int(hi) - int(hi) % 5])
                ax.set_yticks(ax.get_xticks())
                won = sum(1 for p_, q_ in zip(x, y) if q_ > p_)
                ax.text(0.5, 0.035, f"{won}/{len(keys)}", transform=ax.transAxes, fontsize=7.5,
                        ha="center", color="0.2")
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
            if i == 0:
                ax.set_title(SHOWN.get(d, d), fontsize=9, pad=4)
            if j == 0:
                ax.set_ylabel(f"{LABEL.get(base, base)}\n+ mirror  (\\%)".replace("\\%", "%"),
                              fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("accuracy of the backbone alone (%)", fontsize=9, y=0.015)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig2_paired")
    built.append("fig2_paired")


def fig_learning(curves, datasets, built, notes=None):
    """Accuracy against the share of the training trials, one panel per corpus that has a curve.

    A corpus enters only when some model has at least two points on it. The earlier test asked
    whether the corpus appeared in the curve file at all, which a corpus with only its full-data
    point does, so three of the five panels were drawn as empty axes with default 0-to-1 ticks.
    Those corpora have no curve and belong out of the figure, not in it as blank panels.
    """
    if not curves:
        return
    MODELS_HERE = ("eegnet", "eegnet_mirror", "shallow", "shallow_mirror")

    def points(d, m):
        return sorted((fr, v[0]) for (dd, mm, fr), v in curves.items() if dd == d and mm == m)

    ds = [d for d in datasets if any(len(points(d, m)) >= 2 for m in MODELS_HERE)]
    if not ds:
        return
    left_out = [d for d in datasets if d not in ds and any(k[0] == d for k in curves)]
    if left_out and notes is not None:
        notes.append("  fig3_learning: no curve on " + ", ".join(left_out)
                     + " (one point only), so they are not drawn\n")

    fig, axes = panel_grid(len(ds), ncol=3, cell=2.35, extra=0.85)
    for ax, d in zip(axes, ds):
        # Colour says constrained or not, and the line and marker say which backbone. The earlier
        # version spent both colour and line style on the same distinction, so the two backbones
        # were drawn identically and the legend had two pairs of entries that looked the same.
        for m in MODELS_HERE:
            pts = points(d, m)
            if len(pts) >= 2:
                ax.plot([100 * p[0] for p in pts], [p[1] for p in pts],
                        marker="o" if m.startswith("eegnet") else "s", ms=3.6,
                        color=colour(m), ls="-" if m.startswith("eegnet") else "--",
                        label=LABEL.get(m, m))
        ax.set_title(SHOWN.get(d, d))
        ax.set_xlabel("share of the training trials (%)")
        ax.set_ylabel("accuracy (%)")
        ax.spines[["top", "right"]].set_visible(False)
    # one legend for the figure: the four names are long and repeating them in every panel
    # costs more room than the curves themselves.
    h, l = axes[0].get_legend_handles_labels()
    if l:
        fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig3_learning")
    built.append("fig3_learning")


def fig_confusion(rows, datasets, cache, built, notes):
    def has(d, m):                                            # a cell needs the lean, not only the accuracy
        return bool(set(by_model(rows, d, m)) & set(by_model(rows, d, m, "lean")))
    want = ("eegnet", "eegnet_mirror", "shallow", "shallow_mirror")    # four, so each matrix is readable
    show = [m for m in want if all(has(d, m) for d in datasets)]
    ds = [d for d in datasets if any(has(d, m) for m in show)]
    if not show:                                              # nothing is complete: show what there is
        show = [m for m in want if any(has(d, m) for d in datasets)]
        ds = [d for d in datasets if any(has(d, m) for m in show)]
    if not show or not ds:
        return
    # Lay the cells out wider than tall. One matrix is 2x2 and needs very little room, so a column
    # of them down the page spends most of a page on white space, which is what the earlier layout
    # did. The larger of the two counts goes on the columns; with a single model the corpora are
    # wrapped instead of stacked, so five of them make two rows rather than five.
    if len(show) == 1:
        ncol = min(3, len(ds))
        pairs = [(d, show[0]) for d in ds]
        col_title, row_label = (lambda k: SHOWN.get(ds[k], ds[k])), (lambda k: None)
    elif len(show) <= len(ds):
        ncol = len(ds)
        pairs = [(d, m) for m in show for d in ds]
        col_title, row_label = (lambda k: SHOWN.get(ds[k], ds[k])), (lambda k: LABEL.get(show[k], show[k]))
    else:
        ncol = len(show)
        pairs = [(d, m) for d in ds for m in show]
        col_title, row_label = (lambda k: LABEL.get(show[k], show[k])), (lambda k: SHOWN.get(ds[k], ds[k]))
    nrow = -(-len(pairs) // ncol)
    cell = min(1.75, FIGW / ncol)
    fig, axgrid = plt.subplots(nrow, ncol, squeeze=False,
                               figsize=(cell * ncol, cell * nrow + 0.75))
    axes_flat = [a for row in axgrid for a in row]
    for a in axes_flat[len(pairs):]:
        a.set_visible(False)
    approx = 0
    for idx, (d, m) in enumerate(pairs):
        i, j = divmod(idx, ncol)
        ax = axes_flat[idx]
        acc, lean = by_model(rows, d, m), by_model(rows, d, m, "lean")
        nt = by_model(rows, d, m, "n_test")
        M = np.zeros((2, 2))
        for s in sorted(set(acc) & set(lean) & set(nt)):
            counts = test_class_counts(cache, d, s, nt[s])
            if counts is None:
                counts = (nt[s] / 2.0, nt[s] / 2.0)
                approx += 1
            M += confusion(acc[s], lean[s], *counts)
        if M.sum() > 0:
            P = M / M.sum(axis=1, keepdims=True)
            ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{100 * P[r, c]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if P[r, c] > 0.55 else "0.15")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["L", "R"])
            ax.set_yticklabels(["L", "R"])
        else:
            ax.text(0.5, 0.5, "not\nrecorded", ha="center", va="center", fontsize=7, color="0.55",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        # With one model the corpora are wrapped, so a column carries no fixed meaning and every
        # panel is named. Otherwise the grid is regular and the names go on the first row and column.
        if len(show) == 1:
            ax.set_title(SHOWN.get(d, d))
        else:
            if i == 0:
                ax.set_title(col_title(j))
            if j == 0 and row_label(i):
                ax.set_ylabel(row_label(i))
    fig.suptitle("Confusion matrices, % of each true class (rows true, columns predicted)")
    fig.tight_layout()
    save(fig, "fig4_confusion")
    built.append("fig4_confusion")
    if approx:
        notes.append(f"{approx} subject-model cells of the confusion figure assumed a balanced test set "
                     f"because the cached labels did not match the stored number of test trials")


def fig_forest(rows, datasets, built):
    recs = []
    for d in datasets:
        for a, b, _ in PAIRS[:4]:
            va, vb = by_model(rows, d, a), by_model(rows, d, b)
            keys = sorted(set(va) & set(vb))
            if len(keys) >= 3:
                mean, lo, hi, _, _ = paired([va[s] for s in keys], [vb[s] for s in keys])
                recs.append((f"{SHOWN.get(d, d)}:  {SHORT.get(a, a)} \u2212 {SHORT.get(b, b)}", mean, lo, hi))
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(6.6, 0.30 * len(recs) + 1.0))
    ys = np.arange(len(recs))[::-1]
    for y, (_, mean, lo, hi) in zip(ys, recs):
        ax.plot([lo, hi], [y, y], color="0.4", lw=1.4)
        ax.plot([mean], [y], "o", ms=5, color=C_MIRROR if mean > 0 else C_CR)
    ax.axvline(0, color="0.5", lw=0.9, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in recs], fontsize=8)
    ax.grid(axis="x", color="0.92", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("difference in accuracy points (95 % bootstrap interval)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig5_forest")
    built.append("fig5_forest")


SPREAD_MODELS = ("csp_lda", "ts_lr", "eegnet", "shallow",
                 "eegnet_cr", "shallow_cr", "eegnet_mirror", "shallow_mirror")


def fig_spread(rows, datasets, models, built, notes=None):
    """Spread of accuracy over subjects, boxes laid along the y axis.

    Two changes from the earlier version, both forced by measurement rather than taste. The model
    names are long, and rotating sixteen of them under a panel 1.3 in wide made them overlap each
    other to the point of being unreadable; along the y axis they are read horizontally and cannot
    collide. And the figure shows the eight decoders the claims of this paper rest on rather than
    all sixteen: sixteen rows in each of five panels is a full page for a figure whose numbers are
    already in Table~1, while eight fit in half of one.
    """
    ms_all = [m for m in SPREAD_MODELS if m in models]
    ds = [d for d in datasets if any(by_model(rows, d, m) for m in ms_all)]
    if not ds or not ms_all:
        return
    dropped = [m for m in models if m not in ms_all and any(by_model(rows, d, m) for d in ds)]
    if dropped and notes is not None:
        notes.append("  fig6_spread: drawn for " + ", ".join(LABEL.get(m, m) for m in ms_all)
                     + "; the other " + str(len(dropped)) + " decoders are in the accuracy table\n")

    fig, axes = panel_grid(len(ds), ncol=3, cell=0.22 * len(ms_all) + 0.85, extra=0.55, sharey=True)
    rng = np.random.default_rng(0)
    for k, (ax, d) in enumerate(zip(axes, ds)):
        ms = [m for m in ms_all if by_model(rows, d, m)]
        data = [[100 * v for v in by_model(rows, d, m).values()] for m in ms]
        pos = np.arange(len(ms), 0, -1.0)                      # first model at the top
        bp = ax.boxplot(data, vert=False, positions=pos, widths=0.62,
                        patch_artist=True, showfliers=False)
        for patch, m in zip(bp["boxes"], ms):
            patch.set_facecolor(colour(m))
            patch.set_alpha(0.55)
            patch.set_edgecolor("0.3")
        for med in bp["medians"]:
            med.set_color("0.15")
        for p, vals in zip(pos, data):
            ax.scatter(vals, np.full(len(vals), p) + rng.normal(0, 0.06, len(vals)),
                       s=4, color="0.25", alpha=0.55, zorder=3)
        ax.axvline(50, color="0.6", lw=0.8, ls=":")
        ax.set_yticks(pos)
        ax.set_yticklabels([LABEL.get(m, m) for m in ms] if ax.get_subplotspec().is_first_col()
                           else [""] * len(ms))
        ax.set_title(SHOWN.get(d, d))
        ax.set_xlabel("accuracy over subjects (%)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig6_spread")
    built.append("fig6_spread")


# =========================== main ===========================
def main():
    # On Windows a redirected stdout defaults to cp1252, which cannot encode the characters used in the
    # figures. Ask for UTF-8 and fall back to replacing what will not encode, so that writing the summary
    # to a log file can never stop the run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="four", help="four (the default), all, or a comma-separated list")
    ap.add_argument("--protocol", default="session")
    ap.add_argument("--band", default="4,38")
    ap.add_argument("--window", default="0,3")
    args = ap.parse_args()
    b = tuple(float(v) for v in args.band.split(","))
    w = tuple(float(v) for v in args.window.split(","))
    cache = os.path.join(OUTDIR, f"cache_step24_{b[0]:g}-{b[1]:g}Hz_{w[0]:g}-{w[1]:g}s")
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(TABDIR, exist_ok=True)

    rows, n_fits, n_lines, n_short = read_rows(args.protocol)
    present = {k[0] for k in rows}
    if args.datasets == "four":
        datasets = [d for d in FOUR if d in present]
    elif args.datasets == "all":
        datasets = [d for d in ALL if d in present] + sorted(present - set(ALL))
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip() in present]
    models = [m for m in ORDER if any((d, m) in {(k[0], k[1]) for k in rows} for d in datasets)]
    models += sorted({k[1] for k in rows if k[0] in datasets} - set(models))
    if not datasets or not models:
        print(f"nothing to build: the file has no rows for protocol '{args.protocol}'")
        return 1

    out, built, notes = [], [], []
    out.append(f"Paper assets, protocol '{args.protocol}', built from {CSV}")
    out.append(f"{n_fits} fits, {len(datasets)} corpora, {len(models)} models")
    out.append("")
    table_datasets(cache, datasets, out)
    table_accuracy(rows, datasets, models, out)
    table_symmetry(rows, datasets, models, out)
    contrasts = table_contrasts(rows, datasets, out)
    curves = table_learning(args.protocol, datasets, models, out)
    table_metrics(rows, datasets, models, cache, out)
    n_macros = write_numbers(rows, datasets, models, contrasts)

    fig_asymmetry(rows, datasets, models, built)
    fig_paired(rows, datasets, built)
    fig_learning(curves, datasets, built, out)
    fig_confusion(rows, datasets, cache, built, notes)
    fig_forest(rows, datasets, built)
    fig_spread(rows, datasets, models, built, out)
    fig_roc(datasets, args.protocol, built, out)
    fig_antisymmetry(args.protocol, built, out)
    fig_curves(datasets, args.protocol, built, out)
    fig_ablation(rows, datasets, built, out)

    with open(os.path.join(ASSETS, "per_subject.csv"), "w", encoding="utf-8") as f:
        f.write("dataset,model,subject,accuracy,kappa,consistency,lean,asymmetry,n_test\n")
        for (d, m, s), v in sorted(rows.items()):
            if d in datasets:
                f.write(f"{d},{m},{s}," + ",".join(
                    f"{v.get(c, float('nan')):.6f}" for c in
                    ("accuracy", "kappa", "consistency", "lean", "asymmetry", "n_test")) + "\n")
    with open(os.path.join(ASSETS, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(ASSETS, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write(f"protocol: {args.protocol}\ncorpora: {', '.join(datasets)}\nmodels: {', '.join(models)}\n")
        f.write(f"fits used: {n_fits} (of {n_lines} lines; {n_short} lines were incomplete and skipped)\n")
        f.write("figures: " + (", ".join(built) if built else "none") + "\n")
        f.write("tables: table_datasets, table_accuracy, table_symmetry, table_contrasts, table_learning, table_metrics\n")
        f.write(f"numbers.tex: {n_macros} macros for the prose\n")
        for n in notes:
            f.write(f"note: {n}\n")
        missing = [(d, m) for d in datasets for m in models if not by_model(rows, d, m)]
        if missing:
            f.write("empty cells (corpus, model): " + ", ".join(f"{d}/{m}" for d, m in missing) + "\n")
    print("\n".join(out))
    print(f"\nwritten: {ASSETS}")
    print(f"  tables  -> {TABDIR}")
    print(f"  figures -> {FIGDIR}  ({', '.join(built) if built else 'none'})")
    print(f"  also numbers.tex ({n_macros} macros), per_subject.csv, summary.txt, MANIFEST.txt")
    if notes:
        print("notes: " + "; ".join(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
