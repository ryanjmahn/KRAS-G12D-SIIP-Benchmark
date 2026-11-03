#!/usr/bin/env python3
"""
make_fig3.py — regenerate the mean-performance figure (Fig. 3) from the
committed phase-3 CSVs, with the tool labels the paper uses.

Fixes the label on the existing figure: the specialist arm is AE-PocketMiner,
not the original PocketMiner, and the paper cites both separately.

Conformers where a pocket-based tool proposed nothing within 14 A of the site
are scored as zero, not dropped — dropping them would report each tool's mean
over a different subset of conformers.

Usage:
    python make_fig3.py                    # writes fig3_mean_performance.png
    python make_fig3.py --grayscale
    python make_fig3.py --format pdf
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# file, display label, recall column, mcc column
ARMS = [
    ("phase3_fpocket.csv",     "fpocket",        "union_recall", "union_mcc"),
    ("phase3_p2rank.csv",      "P2Rank",         "union_recall", "union_mcc"),
    ("phase3_pocketminer.csv", "AE-PocketMiner", "recall",       "mcc"),
]

COLOR = {"recall": "#1B6CA8", "mcc": "#C1571C"}
GREY = {"recall": "#3A3A3A", "mcc": "#A8A8A8"}


def read_arm(path, rec_col, mcc_col):
    """Return (recalls, mccs) with blanks treated as zero."""
    rec, mcc = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            def val(c):
                v = (row.get(c) or "").strip()
                return float(v) if v else 0.0
            rec.append(val(rec_col))
            mcc.append(val(mcc_col))
    return np.array(rec), np.array(mcc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg", "tiff"])
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--width", type=float, default=3.2)
    ap.add_argument("--height", type=float, default=2.0)
    args = ap.parse_args()

    pal = GREY if args.grayscale else COLOR

    labels, rec_mean, rec_sd, mcc_mean, mcc_sd, n_zero = [], [], [], [], [], []
    for fname, label, rc, mc in ARMS:
        path = os.path.join(args.repo, fname)
        if not os.path.exists(path):
            sys.exit(f"missing {path} — run this from the repository root")
        rec, mcc = read_arm(path, rc, mc)
        labels.append(label)
        rec_mean.append(rec.mean()); rec_sd.append(rec.std(ddof=1))
        mcc_mean.append(mcc.mean()); mcc_sd.append(mcc.std(ddof=1))
        n_zero.append(int((rec == 0).sum()))
        print(f"{label:15s} n={len(rec):2d}  recall {rec.mean():.3f} +/- {rec.std(ddof=1):.3f}"
              f"   MCC {mcc.mean():.3f} +/- {mcc.std(ddof=1):.3f}   zeros {int((rec==0).sum())}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    ax.bar(x - w / 2, rec_mean, w, yerr=rec_sd, capsize=2.5,
           color=pal["recall"], label="Recall",
           error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="#333333"))
    ax.bar(x + w / 2, mcc_mean, w, yerr=mcc_sd, capsize=2.5,
           color=pal["mcc"], edgecolor="#6f6f6f", linewidth=0.4,
           label="Matthews correlation coefficient",
           error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="#333333"))

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("AE-PocketMiner", "AE-Pocket\nMiner") for l in labels],
                       fontsize=7.5)
    ax.set_ylabel("Mean value across\nthe conformational ladder", fontsize=7.5)
    ax.set_ylim(0, max(max(rec_mean) + max(rec_sd), max(mcc_mean) + max(mcc_sd)) * 1.35)
    ax.legend(fontsize=6.5, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e2e2e2", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)

    out = os.path.join(args.outdir, f"fig3_mean_performance.{args.format}")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()