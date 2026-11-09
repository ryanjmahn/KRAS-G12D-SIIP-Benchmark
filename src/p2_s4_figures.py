"""
P2 s4: Phase 2 figures, all from committed results/ CSVs (the same split of work as p1_s7_figures.py).

  figures/p2/p2_fig1_target_null   can anything tell you whether to trust a target's prediction? (the answer
                                   is no, including the tool's own confidence) -- results/p2_reliability.csv
  figures/p2/p2_fig2_residue_gain  among a tool's top-10 residues, how much perturbation stability adds over
                                   the tool's own score, per generator -- results/p2_reliability_tests.csv
  figures/p2/p2_fig3_coverage      risk-coverage: precision against the fraction of each target's top-10 kept
                                   -- results/p2_coverage.csv, results/p2_coverage_points.csv

Where the held-out results exist (results/*_test.csv), each figure draws them beside the training folds as a
hollow or dashed second arm: the training folds are where the effect was found, the test split is where it
either reproduces or does not, and a reader should see both on one axis.

Captions (the figure-creation statements) -> figures/p2/CAPTIONS.md.
Usage (structprep env): python src/p2_s4_figures.py [--gen random] [--format png pdf]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGURES, RESULTS
from figstyle import COLOR, GREY, INK, boot_ci, dot, finish, save, style
from p2_s0_pilot import GENS
from p2_s1_reliability import TOPK, auc

OUT = FIGURES / "p2"
TOOLS = ["fpocket", "P2Rank", "AE-PocketMiner"]          # pLM-NN is sequence-only: identical on every conformer
SIGNALS = [("own_max", 1, "its own top score"), ("own_top", 1, f"its own top-{TOPK} mean"),
           ("rho", 1, "ranking stability (ρ)"), ("jaccard", 1, f"top-{TOPK} overlap"),
           ("sd", -1, "per-residue SD (negated)")]
GEN_LABEL = dict(nma1="lowest mode", nmaK="10-mode mix", random="random motion")
N_BOOT = 2000


def have(name):
    """The held-out counterpart of a results file, if the test sweep has been run."""
    f = RESULTS / f"{name}_test.csv"
    return pd.read_csv(f) if f.exists() else None


def target_auc(g, sig, sign, n=N_BOOT, seed=0, strat=False):
    """AUC for 'this target's apo AUPRC is above the median', with a bootstrap CI over targets.

    With `strat`, targets are split into base-rate quintiles and the AUC is computed inside each against
    that stratum's own median, then averaged by stratum size. Per-target AUPRC has a floor at the base
    rate, and the base rate falls with protein size, so the unstratified number credits any size-like
    quantity with predicting quality (src/p1_s9_confound.py).
    """
    s, ap = sign * g[sig].to_numpy(), g.apo_auprc.to_numpy()
    keep = ~np.isnan(s)
    s, ap = s[keep], ap[keep]
    q = (pd.qcut(g.base_rate.to_numpy()[keep], 5, labels=False, duplicates="drop")
         if strat and "base_rate" in g else np.zeros(len(s), int))
    good = np.empty(len(ap), bool)
    for j in np.unique(q):
        good[q == j] = ap[q == j] > np.median(ap[q == j])

    def est(idx):
        parts = [(np.sum(q[idx] == j), auc(good[idx][q[idx] == j], s[idx][q[idx] == j]))
                 for j in np.unique(q)]
        parts = [(w, v) for w, v in parts if w and not np.isnan(v)]
        return sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else np.nan

    rng = np.random.default_rng(seed)
    b = [est(i) for i in rng.integers(0, len(s), (n, len(s)))]
    return est(np.arange(len(s))), *np.nanpercentile(b, [2.5, 97.5])


def fig1(args):
    """The target-level picture, raw and with the base-rate confound removed."""
    r = pd.read_csv(RESULTS / "p2_reliability.csv")
    tg = RESULTS / "p2_robustness_targets.csv"
    if tg.exists():
        r = r.merge(pd.read_csv(tg)[["parent", "tool", "base_rate"]], on=["parent", "tool"], how="left")
    sets = [(r, False, "as measured")] + ([(r, True, "within base-rate quintiles")]
                                          if "base_rate" in r else [])
    fig, axes = plt.subplots(1, len(TOOLS), figsize=(7.2, 2.5), sharey=True, sharex=True)
    ys = np.arange(len(SIGNALS))[::-1]
    ns = []
    for ax, tool in zip(axes, TOOLS):
        for rr, strat, _ in sets:
            g = rr[(rr.tool == tool) & (rr.gen == args.gen)]
            ns.append(g.parent.nunique())
            off = -0.16 if strat else (0.16 if len(sets) > 1 else 0)
            for y, (sig, sign, _) in zip(ys, SIGNALS):
                dot(ax, y + off, *target_auc(g, sig, sign, strat=strat), COLOR[tool], strat)
        ax.axvline(0.5, color=GREY, lw=1, ls="--", zorder=1)
        ax.set_title(tool, fontsize=8, pad=4)
        finish(ax)
    axes[0].set_yticks(ys, [s[2] for s in SIGNALS])
    axes[0].set_ylim(-0.75, len(SIGNALS) - 0.4)
    axes[0].set_xlim(0.3, 0.7)
    axes[0].set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    for ax in axes:
        ax.text(0.5, -0.72, " chance", color="#555555", fontsize=6.5, va="center")
    if len(sets) > 1:
        axes[1].plot([], [], "o", color="#555555", label="as measured")
        axes[1].plot([], [], "o", color="#555555", mfc="white", mew=1.5, label="base-rate stratified")
        axes[1].legend(frameon=False, fontsize=6.5, loc="lower left", borderpad=0)
    fig.supxlabel("AUC for separating this tool's well-predicted targets from the rest", fontsize=8, y=0.02)
    fig.tight_layout(pad=0.4, rect=(0, 0.05, 1, 1))
    save(fig, "p2_fig1_target_null", args.format, OUT)
    tr = sets[0][0]
    raw = {(t_, s): target_auc(tr[(tr.tool == t_) & (tr.gen == args.gen)], s, g)[0]
           for t_ in TOOLS for s, g, _ in SIGNALS}
    st = {(t_, s): target_auc(tr[(tr.tool == t_) & (tr.gen == args.gen)], s, g, strat=True)[0]
          for t_ in TOOLS for s, g, _ in SIGNALS} if len(sets) > 1 else {}
    own = f"{st.get(('P2Rank', 'own_max'), float('nan')):.2f}" if st else "n/a"
    return (f"**Figure P2-1. What predicts whether a target's prediction is any good — and why the obvious "
            f"measurement says the wrong thing (n = {max(ns)} training targets).** For each tool, the AUC for "
            f"separating its well-predicted targets from the rest, using each candidate confidence signal; "
            f"bars are bootstrap 95% CIs over targets ({N_BOOT:,} resamples), uncorrected for the 15 "
            f"comparisons shown. The top two signals are free, needing no ensemble: the tool's own highest apo "
            f"score and the mean of its top-{TOPK}. The lower three are measured over 10 perturbed conformers "
            f"({GEN_LABEL[args.gen]}). **Filled:** targets split against one global median of apo AUPRC. "
            f"**Hollow:** split within base-rate quintiles. The two disagree because per-target AUPRC has a "
            f"floor at the pocket-residue base rate, which falls with protein size, so the filled markers "
            f"credit or penalise signals for tracking size rather than prediction quality. Corrected, the "
            f"probability tools' own confidence separates well from badly predicted targets at AUC "
            f"{st.get(('AE-PocketMiner', 'own_max'), float('nan')):.2f} and {own}, where the raw measurement "
            f"put it near chance, while fpocket's own score and every perturbation signal stay there.")


def fig2(args):
    """Residue level, where something does survive: stability against the tool's own score."""
    t = pd.read_csv(RESULTS / "p2_reliability_tests.csv")
    tt = have("p2_reliability_tests")
    per = {"": pd.read_csv(RESULTS / "p2_reliability.csv")}
    if have("p2_reliability") is not None:
        per["_test"] = have("p2_reliability")
    rows = [(tool, gen) for tool in TOOLS for gen in GENS]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, sharex=True)
    ys = np.arange(len(rows))[::-1]
    for ax, sig in zip(axes, ("freq", "combo")):
        for y, (tool, gen) in zip(ys, rows):
            arms = [(t, per[""], False, 0.16 if tt is not None else 0)]
            if tt is not None:
                arms.append((tt, per["_test"], True, -0.16))
            for tests, raw, hollow, off in arms:
                x = tests[(tests.tool == tool) & (tests.gen == gen) & (tests.signal == sig)]
                if not len(x):
                    continue
                x = x.iloc[0]
                g = raw[(raw.tool == tool) & (raw.gen == gen)]
                d = (g[f"{sig}_auc_topk"] - g.own_auc_topk).dropna().to_numpy()
                lo, hi = boot_ci(d)                          # paired over targets, matching the Wilcoxon
                dot(ax, y + off, x["diff"], lo, hi, COLOR[tool], hollow)
                if not hollow:
                    p = f"{x.p_holm:.0e}".replace("e-0", "×10⁻") if x.p_holm < 1e-3 else f"{x.p_holm:.3f}"
                    ax.text(1.02, y, f"p = {p}", transform=ax.get_yaxis_transform(), fontsize=6, va="center",
                            color=INK if x.p_holm < 0.05 else "#9a9a9a")
        ax.axvline(0, color="#555555", lw=0.8, zorder=1)
        finish(ax)
    axes[0].set_yticks(ys, [f"{t_} · {GEN_LABEL[g]}" for t_, g in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0].set_title("a  stability alone", loc="left", fontsize=8)
    axes[1].set_title("b  stability + the tool's own score", loc="left", fontsize=8)
    fig.supxlabel("gain in AUC over the tool's own score (mean over targets)", fontsize=8, y=0.02)
    fig.tight_layout(pad=0.4, w_pad=5, rect=(0, 0.05, 1, 1))
    save(fig, "p2_fig2_residue_gain", args.format, OUT)
    return (f"**Figure P2-2. Among a tool's top-{TOPK} residues, stability under perturbation adds a little to "
            f"its ranking.** Mean per-target gain in AUC (true pocket residues ranked above false ones) over "
            f"the tool's own apo score, for each tool and each of the three motion generators. (a) Stability "
            f"alone: how often the residue stays in a conformer's top-{TOPK}. (b) Stability rank-summed with "
            f"the tool's own score, so nothing is fitted. Paired Wilcoxon over targets, Holm-corrected across "
            f"all 18 rows, printed at the right of each row and greyed where p_holm ≥ 0.05; bars are bootstrap 95% CIs of the mean paired difference."
            + (" Hollow: the held-out test split." if tt is not None else "") +
            f" Random motion matches or beats the normal modes everywhere, so the signal is not about dynamics. "
            f"fpocket gains most because it has the least to lose: it scores a residue by the best pocket "
            f"containing it, so its top-{TOPK} is one tied block on 84% of targets and it cannot rank inside "
            f"its own shortlist at all.")


def fig3(args):
    """What the gain is worth in practice: precision against how much you keep."""
    cv, pt = pd.read_csv(RESULTS / "p2_coverage.csv"), pd.read_csv(RESULTS / "p2_coverage_points.csv")
    cvt = have("p2_coverage")
    # one y-axis for all three panels: the point is that the gains are small and differ between tools, and
    # per-panel scaling would magnify fpocket's 0.03 into the largest-looking effect in the figure
    fig, axes = plt.subplots(1, len(TOOLS), figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, tool in zip(axes, TOOLS):
        for r, dashed in [(cv, False)] + ([(cvt, True)] if cvt is not None else []):
            g = r[(r.gen == args.gen) & (r.tool == tool)]
            base = g[(g.selector == "own") & (g.coverage == 1.0)].precision.iloc[0]
            ax.axhline(base, color=GREY, lw=1, ls=(0, (4, 3)), zorder=1)
            for sel, lw, alpha in (("own", 1.6, 0.45), ("combo", 2.2, 1.0)):
                s = g[g.selector == sel].sort_values("coverage")
                ax.plot(s.coverage * 100, s.precision, ls=(0, (2, 1.5)) if dashed else "-", lw=lw,
                        color=COLOR[tool], alpha=alpha, zorder=3 if sel == "combo" else 2)
            if not dashed:
                ax.text(0, base, f" {base:.2f}", color="#666666", fontsize=6.5, va="bottom", ha="left")
        x = pt[(pt.tool == tool) & (pt.selector == "combo") & (pt.coverage == 0.2)].iloc[0]
        sig = x.gain_lo > 0
        ax.annotate(f"{x.gain_vs_own:+.3f} at 20%\n[{x.gain_lo:+.3f}, {x.gain_hi:+.3f}]",
                    xy=(20, x.precision), xytext=(26, x.precision + 0.055), fontsize=6.5,
                    color=INK if sig else "#9a9a9a", va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#b4b4b4",
                                    shrinkA=0, shrinkB=3, connectionstyle="arc3,rad=0"))
        ax.set_title(tool, fontsize=8, pad=4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        finish(ax, grid="y")
    axes[0].set_ylabel("precision among the kept residues")
    axes[0].set_ylim(0.10, 0.47)
    h = [plt.Line2D([], [], color="#444", lw=2.2), plt.Line2D([], [], color="#444", lw=1.6, alpha=0.5),
         plt.Line2D([], [], color=GREY, lw=1, ls=(0, (4, 3)))]
    axes[1].legend(h, ["stability + the tool's own score", "the tool's own score", "keep everything (base rate)"],
                   frameon=False, fontsize=6.5, loc="lower left", handlelength=1.8, borderpad=0)
    fig.supxlabel(f"share of each target's top-{TOPK} residues kept (%)", fontsize=8, y=0.02)
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0.05, 1, 1))
    save(fig, "p2_fig3_coverage", args.format, OUT)
    n = int(cv.n_targets.max())
    return (f"**Figure P2-3. Withholding the least stable predictions buys precision only where the tool has no "
            f"ranking of its own (n = {n} training targets).** Each target contributes its top-{TOPK} apo "
            f"residues; at a given coverage every target keeps that same share of them, ranked within the "
            f"target, so no target can crowd out another. Annotated: the gain of stability + own score over the "
            f"own score alone at 20% coverage, with a paired cluster-bootstrap 95% CI over targets. fpocket's "
            f"own curve is flat on its base rate because it cannot order its own shortlist; stability supplies "
            f"the ordering. P2Rank and AE-PocketMiner already rank their shortlists, and gain nothing."
            + (" Dashed: the held-out test split." if cvt is not None else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="random", choices=GENS)
    ap.add_argument("--format", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()
    style()
    caps = [f(args) for f in (fig1, fig2, fig3)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CAPTIONS.md").write_text("# Phase 2 figures\n\n" + "\n\n".join(caps) + "\n")
    print("\n\n".join(caps))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
