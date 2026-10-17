r"""
make_method_figure.py

Draws the two figures of the paper that show the method rather than a result, and therefore need no data:

    figures/fig1_construction.pdf   (a) the montage and the exchange of hemispheres, (b) the two branches of
                                    the head, (c) what the exchange does to the output, (d) the hypothesis
                                    space, and where a constraint, an augmentation and a penalty act in it
    figures/fig2_pipeline.pdf       the study design: epoching, windows, the split, what is fitted on what

Both are vector drawings; nothing here is fitted or measured.

    python make_method_figure.py [--outdir figures]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse

BLUE, ORANGE, GREEN, GREY = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C"
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})

# a 10-10-like subset, drawn by hand on a head outline: (name, x, y)
MONTAGE = [("F3", -0.46, 0.55), ("Fz", 0.0, 0.62), ("F4", 0.46, 0.55),
           ("FC3", -0.48, 0.27), ("FCz", 0.0, 0.30), ("FC4", 0.48, 0.27),
           ("C3", -0.50, -0.02), ("Cz", 0.0, -0.02), ("C4", 0.50, -0.02),
           ("CP3", -0.48, -0.31), ("CPz", 0.0, -0.34), ("CP4", 0.48, -0.31),
           ("P3", -0.40, -0.59), ("Pz", 0.0, -0.63), ("P4", 0.40, -0.59)]


def head(ax, cx=0.0, cy=0.0, r=1.0):
    ax.add_patch(Circle((cx, cy), r, fill=False, lw=1.1, ec="0.35"))
    ax.plot([cx - 0.11 * r, cx, cx + 0.11 * r], [cy + r * 0.99, cy + r * 1.16, cy + r * 0.99],
            color="0.35", lw=1.1)                                   # the nose, so left and right are fixed
    ax.plot([cx - 0.02 * r, cx + 0.02 * r], [cy - r, cy - r], color="0.35", lw=1.1)


def panel_montage(ax):
    ax.set_title("(a)  the exchange of hemispheres $M$", loc="left", fontsize=9)
    head(ax)
    for name, x, y in MONTAGE:
        mid = abs(x) < 1e-9
        ax.add_patch(Circle((x, y), 0.085, fc="white", ec=GREY if mid else "0.45", lw=0.9, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=6.4, zorder=4)
    for name, x, y in MONTAGE:                                      # one arrow per pair, drawn once
        if x <= 0:
            continue
        ax.add_patch(FancyArrowPatch((x - 0.02, y + 0.085), (-x + 0.02, y + 0.085), arrowstyle="<|-|>",
                                     mutation_scale=7, lw=0.8, color=GREEN, alpha=0.8,
                                     connectionstyle="arc3,rad=0.42", zorder=2))
    ax.text(0.0, -1.34, "midline electrodes map to themselves", ha="center", va="top", fontsize=7.6)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.62, 1.3)


def _box(ax, x, y, w, h, text, fc="white", ec="0.35", fs=7.8, lw=0.9, zorder=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=zorder))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=zorder + 1)


def _arrow(ax, p, q, color="0.35", lw=0.9, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def panel_head(ax):
    ax.set_title("(b)  the decoder", loc="left", fontsize=9)
    _box(ax, 0.12, 0.78, 0.30, 0.16, "window $x$")
    _arrow(ax, (0.12, 0.70), (0.12, 0.60))
    _box(ax, 0.12, 0.50, 0.30, 0.16, "mirror $Mx$", fc="#EAF3EA", ec=GREEN)
    _arrow(ax, (0.27, 0.78), (0.52, 0.60), rad=-0.15)
    _arrow(ax, (0.27, 0.50), (0.52, 0.34), rad=-0.15)
    _box(ax, 0.70, 0.60, 0.30, 0.17, "$g$", fc="#EDF1F8", ec=BLUE, fs=10)
    _box(ax, 0.70, 0.30, 0.30, 0.17, "$g$", fc="#EDF1F8", ec=BLUE, fs=10)
    ax.text(0.70, 0.45, "same weights", ha="center", va="center", fontsize=6.8, color=BLUE)
    _arrow(ax, (0.85, 0.60), (0.97, 0.60))
    _arrow(ax, (0.85, 0.30), (0.97, 0.30))
    ax.text(1.00, 0.60, r"$f_2 = g(x)$        right hand", ha="left", va="center", fontsize=8)
    ax.text(1.00, 0.30, r"$f_1 = g(Mx)$     left hand", ha="left", va="center", fontsize=8)
    ax.text(0.12, 0.10, r"$s(x) = f_2 - f_1 = g(x) - g(Mx)$,   $\hat y = \mathbb{1}[s>0]$",
            ha="left", va="center", fontsize=8.6)
    ax.text(0.12, -0.02, "one output unit, no other change to the backbone", ha="left", va="center",
            fontsize=7.4, color="0.35")
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.12, 0.95)


def panel_property(ax):
    ax.set_title("(c)  what the exchange does", loc="left", fontsize=9)
    rows = [(0.74, "$x$", r"$f(x) = [\,g(Mx),\ g(x)\,]$", "white", "0.35"),
            (0.40, "$Mx$", r"$f(Mx) = [\,g(x),\ g(Mx)\,]$", "#EAF3EA", GREEN)]
    for y, lab, expr, fc, ec in rows:
        _box(ax, 0.16, y, 0.24, 0.16, lab, fc=fc, ec=ec, fs=9)
        _arrow(ax, (0.29, y), (0.44, y))
        ax.text(0.48, y, expr, ha="left", va="center", fontsize=8.6)
    ax.add_patch(FancyArrowPatch((0.16, 0.66), (0.16, 0.48), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color=GREEN))
    ax.text(0.21, 0.57, "$M$", ha="left", va="center", fontsize=8.5, color=GREEN)
    ax.text(0.02, 0.16, r"$M^{2}=I \;\Longrightarrow\; f(Mx) = P\,f(x)"
                        r" \;\Longrightarrow\; s(Mx) = -\,s(x)$", ha="left", va="center", fontsize=8.6)
    ax.text(0.02, 0.05, "for every input and every value of the weights", ha="left", va="center",
            fontsize=7.6, color="0.3")
    ax.set_xlim(0.0, 1.72)
    ax.set_ylim(0.0, 0.92)


def panel_space(ax):
    ax.set_title("(d)  where each approach acts", loc="left", fontsize=9)
    ax.add_patch(Ellipse((0.5, 0.5), 0.92, 0.66, fc="#F4F4F6", ec="0.5", lw=0.9))
    ax.text(0.5, 0.86, "all functions the backbone can represent", ha="center", fontsize=7.8, color="0.3")
    ax.plot([0.5, 0.5], [0.19, 0.81], color=GREEN, lw=1.6)
    ax.text(0.52, 0.215, "equivariant\nfunctions", ha="left", va="bottom", fontsize=7.6, color=GREEN)
    ax.plot([0.5], [0.50], marker="o", ms=5.5, color=GREEN, zorder=4)
    ax.text(0.475, 0.50, "construction", ha="right", va="center", fontsize=7.4, color=GREEN)
    ax.plot([0.80], [0.60], marker="o", ms=5, color=ORANGE, zorder=4)
    _arrow(ax, (0.785, 0.605), (0.625, 0.575), color=ORANGE, lw=1.0, rad=0.25)
    ax.text(0.83, 0.64, "augmentation,\npenalty: pulled\ntowards it", ha="left", va="center",
            fontsize=7.4, color=ORANGE)
    ax.plot([0.265], [0.46], marker="o", ms=5, color=BLUE, zorder=4)
    ax.text(0.265, 0.40, "plain network:\nanywhere", ha="center", va="top", fontsize=7.4, color=BLUE)
    ax.set_xlim(0.02, 1.28)
    ax.set_ylim(0.10, 0.98)


def figure_construction(outdir):
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.22], height_ratios=[1.15, 0.95],
                          wspace=0.10, hspace=0.26)
    for ax, fn in ((fig.add_subplot(gs[0, 0]), panel_montage), (fig.add_subplot(gs[0, 1]), panel_head),
                   (fig.add_subplot(gs[1, 0]), panel_property), (fig.add_subplot(gs[1, 1]), panel_space)):
        ax.set_axis_off()
        fn(ax)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig1_construction.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_pipeline(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.25, 1.0, 1.15]})
    for ax in axes:
        ax.set_axis_off()

    ax = axes[0]                                                    # (a) one trial, its windows
    ax.set_title("(a)  one trial", loc="left", fontsize=9)
    ax.add_patch(FancyBboxPatch((0.04, 0.60), 0.92, 0.17, boxstyle="round,pad=0.008",
                                fc="#EDF1F8", ec=BLUE, lw=0.9))
    ax.text(0.50, 0.685, "epoch, 0 to 3 s after the cue", ha="center", va="center", fontsize=7.6)
    for k in range(6):
        x0 = 0.04 + k * 0.092
        ax.add_patch(FancyBboxPatch((x0, 0.30 - k * 0.035), 0.46, 0.028, boxstyle="round,pad=0.004",
                                    fc=GREEN, ec="none", alpha=0.55))
    ax.text(0.04, 0.03, "six windows of 2 s, every 0.2 s;\nno window crosses a trial boundary",
            ha="left", va="bottom", fontsize=7.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.92)

    ax = axes[1]                                                    # (b) the split
    ax.set_title("(b)  the split", loc="left", fontsize=9)
    ax.add_patch(FancyBboxPatch((0.06, 0.52), 0.56, 0.20, boxstyle="round,pad=0.01",
                                fc="#EDF1F8", ec=BLUE, lw=0.9))
    ax.text(0.34, 0.62, "earlier blocks\ntraining", ha="center", va="center", fontsize=7.6)
    ax.add_patch(FancyBboxPatch((0.66, 0.52), 0.28, 0.20, boxstyle="round,pad=0.01",
                                fc="#FBEFE6", ec=ORANGE, lw=0.9))
    ax.text(0.80, 0.62, "last block\ntest", ha="center", va="center", fontsize=7.6)
    _arrow(ax, (0.06, 0.45), (0.94, 0.45), color="0.45")
    ax.text(0.50, 0.38, "recording order", ha="center", va="top", fontsize=7.2, color="0.35")
    ax.text(0.06, 0.20, "the split comes first; the per-channel\nmean and standard deviation are then\n"
                        "computed on the training trials alone",
            ha="left", va="center", fontsize=7.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.92)

    ax = axes[2]                                                    # (c) training and scoring
    ax.set_title("(c)  fitting and scoring", loc="left", fontsize=9)
    _box(ax, 0.30, 0.74, 0.44, 0.15, "windows of the\ntraining trials", fs=7.4)
    _arrow(ax, (0.30, 0.655), (0.30, 0.575))
    _box(ax, 0.30, 0.50, 0.44, 0.15, "fit the decoder\n(one budget for all)", fs=7.4, fc="#EDF1F8", ec=BLUE)
    _arrow(ax, (0.52, 0.50), (0.74, 0.50))
    _box(ax, 0.95, 0.50, 0.40, 0.15, "frozen", fs=7.6)
    _box(ax, 0.30, 0.22, 0.44, 0.15, "windows of a\ntest trial", fs=7.4, fc="#FBEFE6", ec=ORANGE)
    _arrow(ax, (0.52, 0.22), (0.74, 0.22))
    ax.text(0.76, 0.22, r"$S=\sum_w s(x_w)$", ha="left", va="center", fontsize=8)
    ax.text(0.04, 0.04, "one score per trial; the class is $S>0$", ha="left", va="bottom", fontsize=7.4)
    ax.set_xlim(0, 1.5); ax.set_ylim(0, 0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig2_pipeline.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    figure_construction(args.outdir)
    figure_pipeline(args.outdir)
    print(f"written: {os.path.join(args.outdir, 'fig1_construction.pdf')} and fig2_pipeline.pdf (and .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
