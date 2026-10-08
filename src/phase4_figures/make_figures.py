#!/usr/bin/env python3
"""
make_figures.py — every figure and derived table for the KRAS G12D
cryptic-pocket benchmark, generated from the committed CSVs.
Figure numbers match the paper.

Reads:
    data/ladder/ladder_dense.csv      conf, siip_volume
    results/phase3_p2rank.csv         conf, union_recall, union_mcc, dca, n_fragments
    results/phase3_fpocket.csv        conf, union_recall, union_mcc, dca, n_fragments
    results/phase3_pocketminer.csv    conf, recall, mcc, npred
    results/anchors_all.csv           structure, tool, recall, mcc, precision, dca

Writes (to --outdir, default figures/):
    fig1_volume.*             pocket volume, anchors and ladder ends
    fig2_anchor_mcc.*         localization accuracy on the two structures
    fig3_pm_probability.*     threshold-independent check on AE-PocketMiner
    fig4_mean_performance.*   ladder means with SD
    fig5_dca_vs_volume.*      P2Rank localization distance against volume
and table2_clusters.csv (ladder grouped by volume cluster) to --tabledir,
default results/.

Any input that is missing is reported on stderr and its figure is skipped —
nothing is invented. Values baked in below are only those measured outside the
CSVs (mean predicted probabilities, anchor volumes under the ladder rule);
edit them here if they change.

Usage:
    python src/phase4_figures/make_figures.py
    python src/phase4_figures/make_figures.py --grayscale --format pdf
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import FIGURES, LADDER_CSV, RESULTS

# Measured outside the phase-3 CSVs — update here if re-measured.
# Single fpocket cavity at the site on each anchor (Phase 1 openness check; paper Methods).
ANCHOR_VOLUME_SINGLE = {"5US4": 245, "7RPZ": 822}
# Anchors measured with the ladder's near-His95 rule (src/phase2_ladder/anchor_volumes.py).
ANCHOR_VOLUME_LADDER_RULE = {"5US4": 281, "7RPZ": 811}
# Printed as "PM mean prob" by src/phase3_benchmark/score_anchors.py.
PM_PROB = {  # (mean at true SII-P residues, protein-wide mean)
    "7RPZ": (0.606, 0.514),
    "5US4": (0.537, 0.514),
}
CLUSTER_EDGES = [0, 350, 550, 10_000]  # volume bins for the cluster table
CLUSTER_NAMES = ["closed (~280)", "intermediate (~400)", "open (~720)"]

COLOR = {"a": "#1B6CA8", "b": "#C1571C", "dark": "#243B53", "light": "#7FA8C9",
         "miss": "#C0392B"}
GREY = {"a": "#3A3A3A", "b": "#A8A8A8", "dark": "#3A3A3A", "light": "#9A9A9A",
        "miss": "#3A3A3A"}


def warn(m):
    print(f"[make_figures] {m}", file=sys.stderr)


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def num(row, key):
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def setup(gray):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })
    return GREY if gray else COLOR


def finish(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e2e2e2", linewidth=0.5)
    ax.set_axisbelow(True)


def fig1(pal, ladder, out, size):
    if not ladder:
        return warn("fig1 skipped: no ladder data")
    vols = [v for v, _, _ in ladder]
    values = [ANCHOR_VOLUME_SINGLE["5US4"], round(min(vols)),
              round(max(vols)), ANCHOR_VOLUME_SINGLE["7RPZ"]]
    labels = ["5US4\nclosed\n(exp.)", "Ladder\nclosed end",
              "Ladder\nopen end", "7RPZ\nopen\n(exp.)"]
    fig, ax = plt.subplots(figsize=size)
    bars = ax.bar(range(4), values, width=0.62,
                  color=[pal["dark"], pal["light"], pal["light"], pal["dark"]])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 18, str(v),
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Pocket volume\n(cubic angstroms)", fontsize=7.5)
    ax.set_ylim(0, max(values) * 1.16)
    finish(ax); fig.tight_layout(pad=0.3); fig.savefig(out); plt.close(fig)


def fig2(pal, anchors, out, size):
    if not anchors:
        return warn("fig2 skipped: anchors_all.csv missing")
    tools = ["fpocket", "P2Rank", "AE-PocketMiner"]
    o = [anchors.get(("7RPZ", t), {}).get("mcc") or 0.0 for t in tools]
    c = [anchors.get(("5US4", t), {}).get("mcc") or 0.0 for t in tools]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=size)
    for vals, off, col, lab, edge in [(o, -w/2, pal["a"], "Open (7RPZ)", None),
                                      (c, w/2, pal["light"], "Closed (5US4)", "#6f6f6f")]:
        bars = ax.bar(x + off, vals, w, color=col, label=lab,
                      edgecolor=edge, linewidth=0.4 if edge else 0)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("AE-PocketMiner", "AE-Pocket\nMiner") for t in tools],
                       fontsize=7.5)
    ax.set_ylabel("Matthews correlation\ncoefficient", fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.17))
    finish(ax); fig.tight_layout(pad=0.3); fig.savefig(out); plt.close(fig)


def fig4(pal, arms, out, size):
    if not arms:
        return warn("Fig. 4 skipped: phase-3 CSVs missing")
    labels = [a[0] for a in arms]
    rm = [np.mean(a[1]) for a in arms]; rs = [np.std(a[1], ddof=1) for a in arms]
    mm = [np.mean(a[2]) for a in arms]; ms = [np.std(a[2], ddof=1) for a in arms]
    x = np.arange(len(arms)); w = 0.36
    fig, ax = plt.subplots(figsize=size)
    ek = dict(elinewidth=0.7, capthick=0.7, ecolor="#333333")
    ax.bar(x - w/2, rm, w, yerr=rs, capsize=2.5, color=pal["a"],
           label="Recall", error_kw=ek)
    ax.bar(x + w/2, mm, w, yerr=ms, capsize=2.5, color=pal["b"],
           label="Matthews correlation coefficient", error_kw=ek)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("AE-PocketMiner", "AE-Pocket\nMiner") for l in labels],
                       fontsize=7.5)
    ax.set_ylabel("Mean value across\nthe conformational ladder", fontsize=7.5)
    ax.set_ylim(0, max(max(rm) + max(rs), max(mm) + max(ms)) * 1.35)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    finish(ax); fig.tight_layout(pad=0.3); fig.savefig(out); plt.close(fig)


def fig5(pal, ladder, out, size):
    pts = [(v, d) for v, _, d in ladder if d is not None]
    if not pts:
        return warn("Fig. 5 skipped: no DCA values")
    v, d = zip(*pts)
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(v, d, s=20, color=pal["a"])
    ax.set_xlabel("Pocket volume (cubic angstroms)", fontsize=7.5)
    ax.set_ylabel("Distance from predicted centre\nto ligand (angstroms)", fontsize=7.5)
    ax.set_ylim(0, max(d) * 1.25)
    finish(ax); fig.tight_layout(pad=0.3); fig.savefig(out); plt.close(fig)


def fig3(pal, out, size):
    labels = ["7RPZ\n(open)", "5US4\n(closed)"]
    site = [PM_PROB["7RPZ"][0], PM_PROB["5US4"][0]]
    whole = [PM_PROB["7RPZ"][1], PM_PROB["5US4"][1]]
    x = np.arange(2); w = 0.34
    fig, ax = plt.subplots(figsize=size)
    for vals, off, col, lab, edge in [(site, -w/2, pal["a"], "True SII-P residues", None),
                                      (whole, w/2, pal["light"], "Protein-wide mean", "#6f6f6f")]:
        bars = ax.bar(x + off, vals, w, color=col, label=lab,
                      edgecolor=edge, linewidth=0.4 if edge else 0)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Mean predicted\nprobability", fontsize=7.5)
    ax.set_ylim(0, 0.78)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.17))
    finish(ax); fig.tight_layout(pad=0.3); fig.savefig(out); plt.close(fig)


def cluster_table(ladder, out):
    if not ladder:
        return warn("table2 skipped: no ladder data")
    rows = []
    for i, name in enumerate(CLUSTER_NAMES):
        lo, hi = CLUSTER_EDGES[i], CLUSTER_EDGES[i + 1]
        grp = [(v, r) for v, r, _ in ladder if lo <= v < hi]
        if not grp:
            continue
        det = [r for _, r in grp if r is not None]
        rows.append({
            "cluster": name,
            "n_conformers": len(grp),
            "volume_min": round(min(v for v, _ in grp), 1),
            "volume_max": round(max(v for v, _ in grp), 1),
            "n_detected": len(det),
            "mean_recall_when_detected": round(float(np.mean(det)), 3) if det else "",
        })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{'cluster':22s} {'n':>3s} {'vol range':>16s} {'det':>5s} {'mean recall':>12s}")
    for r in rows:
        print(f"{r['cluster']:22s} {r['n_conformers']:3d} "
              f"{r['volume_min']:7.1f}-{r['volume_max']:<8.1f} "
              f"{r['n_detected']:5d} {str(r['mean_recall_when_detected']):>12s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(FIGURES))
    ap.add_argument("--tabledir", default=str(RESULTS))
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg", "tiff"])
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--width", type=float, default=3.2)
    ap.add_argument("--height", type=float, default=1.9)
    args = ap.parse_args()

    pal = setup(args.grayscale)
    size = (args.width, args.height)
    os.makedirs(args.outdir, exist_ok=True)
    ext = args.format

    # ladder: [(volume, recall_or_None, dca_or_None)]
    vol_rows = read_csv(str(LADDER_CSV))
    p2_rows = read_csv(str(RESULTS / "phase3_p2rank.csv"))
    ladder = []
    if vol_rows and p2_rows:
        vols = {int(r["conf"]): num(r, "siip_volume") for r in vol_rows}
        for r in p2_rows:
            c = int(r["conf"])
            if vols.get(c) is None:
                continue
            rec = num(r, "union_recall")
            dca = num(r, "dca")
            ladder.append((vols[c], rec if rec else None, dca))
    else:
        warn("ladder_dense.csv or phase3_p2rank.csv missing")

    # anchors
    arows = read_csv(str(RESULTS / "anchors_all.csv"))
    anchors = {}
    if arows:
        for r in arows:
            anchors[(r["structure"], r["tool"])] = {
                "recall": num(r, "recall"), "mcc": num(r, "mcc"),
                "precision": num(r, "precision"), "dca": num(r, "dca")}

    # per-arm ladder values for fig3
    arms = []
    for fname, label, rc, mc in [
        ("phase3_fpocket.csv", "fpocket", "union_recall", "union_mcc"),
        ("phase3_p2rank.csv", "P2Rank", "union_recall", "union_mcc"),
        ("phase3_pocketminer.csv", "AE-PocketMiner", "recall", "mcc"),
    ]:
        rows = read_csv(str(RESULTS / fname))
        if not rows:
            warn(f"{fname} missing — excluded from Fig. 4")
            continue
        rec = [num(r, rc) or 0.0 for r in rows]
        mcc = [num(r, mc) or 0.0 for r in rows]
        arms.append((label, np.array(rec), np.array(mcc)))
        print(f"{label:15s} n={len(rec):2d}  recall {np.mean(rec):.3f} +/- {np.std(rec, ddof=1):.3f}"
              f"   MCC {np.mean(mcc):.3f} +/- {np.std(mcc, ddof=1):.3f}"
              f"   zeros {sum(1 for x in rec if x == 0)}")

    o = lambda n: os.path.join(args.outdir, f"{n}.{ext}")
    fig1(pal, ladder, o("fig1_volume"), size)
    fig2(pal, anchors, o("fig2_anchor_mcc"), size)
    fig3(pal, o("fig3_pm_probability"), size)
    fig4(pal, arms, o("fig4_mean_performance"), size)
    fig5(pal, ladder, o("fig5_dca_vs_volume"), size)
    os.makedirs(args.tabledir, exist_ok=True)
    cluster_table(ladder, os.path.join(args.tabledir, "table2_clusters.csv"))
    print(f"\nwrote figures to {args.outdir}/ as .{ext}")


if __name__ == "__main__":
    main()