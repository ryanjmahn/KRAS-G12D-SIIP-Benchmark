"""
P1 s7: Phase 1 figures, all from committed results/ CSVs (plus fpocket outputs in runs/ for the KRAS volume panel).

  figures/p1/p1_fig1_performance   detection (site-hit rate, Wilson CI), MCC and AUPRC (bootstrap CI of the mean),
                                   any-rank vs top-3 scoring for the pocket tools; 218 scored test targets
  figures/p1/p1_fig2_failure_modes share of targets per locked failure tag, per tool (small multiples)
  figures/p1/p1_fig3_robustness    curated subset: share of targets whose detection flips along the ladder, and mean
                                   best-minus-apo MCC (headroom), lowest-mode NMA vs random-motion control
  figures/p1/p1_fig4_kras_axes     KRAS ladder: fpocket site volume vs pocket Cα RMSD to holo (two panels, one axis each)
Captions (the figure-creation statements) -> figures/p1/CAPTIONS.md.

Usage (structprep env): python src/p1_s7_figures.py [--format png pdf]
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
from paths import FIGURES, RESULTS, RUNS
from figstyle import COLOR, GREY, TOOLS, dot, finish, style
from figstyle import save as _save      # shared with Phase 2, see src/figstyle.py
from p1_s5_robustness import LADDERS, site_volumes
from p1_s6_aggregate import boot_ci

OUT = FIGURES / "p1"
TAGS = ["success", "partial", "offset", "fragmentation", "over_spread", "miss"]
KRAS = "kras_g12d_siip"


def save(fig, name, fmts):
    _save(fig, name, fmts, OUT)


def fig1(fmts):
    summ = pd.read_csv(RESULTS / "p1_stats_summary.csv").set_index(["setting", "tool", "metric"])
    rates = pd.read_csv(RESULTS / "p1_stats_rates.csv").set_index(["setting", "tool", "rate"])
    master = pd.read_csv(RESULTS / "p1_master.csv")
    base = master[master.primary & (master.split == "test")].drop_duplicates("target_id").base_rate.mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    ys = np.arange(len(TOOLS))[::-1]
    for y, t in zip(ys, TOOLS):
        for setting, off, hollow in (("primary", 0.12, False), ("deployable", -0.12, True)):
            if hollow and t not in ("fpocket", "P2Rank"):
                continue
            yy = y + (off if t in ("fpocket", "P2Rank") else 0)
            r = rates.loc[(setting, t, "site_hit")]
            dot(axes[0], yy, r.share, r.lo, r.hi, COLOR[t], hollow)
            m = summ.loc[(setting, t, "mcc")]
            dot(axes[1], yy, m["mean"], m.mean_lo, m.mean_hi, COLOR[t], hollow)
            if setting == "primary":
                a = summ.loc[(setting, t, "auprc")]
                dot(axes[2], y, a["mean"], a.mean_lo, a.mean_hi, COLOR[t])
    pw = pd.read_csv(RESULTS / "p1_stats_pairwise.csv")
    def rev(setting):
        x = pw[(pw.setting == setting) & (pw.metric == "mcc") & (pw.tool_a == "P2Rank") & (pw.tool_b == "fpocket")].iloc[0]
        win = "P2Rank" if x.mean_diff > 0 else "fpocket"
        return f"{win} > {'fpocket' if win == 'P2Rank' else 'P2Rank'}", f"p = {x.p_holm:.0e}"
    (w1, p1), (w2, p2) = rev("primary"), rev("deployable")
    axes[1].text(0.98, 0.38, f"any rank: {w1}\n{p1}\ntop-3: {w2}\n{p2} (paired, Holm)",
                 transform=axes[1].transAxes, ha="right", va="center", fontsize=6.5, color="#333333")
    axes[2].axvline(base, color=GREY, lw=1, ls="--")
    axes[2].text(base, len(TOOLS) - 0.55, f" chance {base:.2f}", color="#555555", fontsize=7, va="center")
    axes[0].set_yticks(ys, TOOLS)
    axes[0].set_ylim(-0.6, len(TOOLS) - 0.4)
    for ax, lab in zip(axes, ["a  site detected (share of targets)", "b  residue MCC (mean)", "c  AUPRC (mean)"]):
        ax.set_title(lab, loc="left", fontsize=8)
        finish(ax)
    axes[0].set_xlim(0, 1)
    axes[0].plot([], [], "o", color="#555555", label="any rank")
    axes[0].plot([], [], "o", color="#555555", mfc="white", mew=1.5, label="top-3 pockets only")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(pad=0.4)
    save(fig, "p1_fig1_performance", fmts)
    return (f"**Figure P1-1. Detection and localization on the CryptoBench test split (n = 218 scored targets).** "
            f"(a) Share of targets with a prediction at the site (pocket centre within 4 Å of a ligand atom; "
            f"probability tools: any residue above threshold), Wilson 95% CI. (b) Mean residue-level MCC, bootstrap "
            f"95% CI (10,000 resamples). Filled: any-rank scoring; hollow: only the top-3 pockets may count "
            f"(fpocket and P2Rank). The ranking of the two pocket tools reverses between the two rules (paired Wilcoxon, Holm-corrected, in the panel). (c) Mean AUPRC, "
            f"bootstrap 95% CI; dashed line is the mean pocket-residue base rate ({base:.3f}). Thresholds: "
            f"AE-PocketMiner 0.7, pLM-NN 0.75.")


def fig2(fmts):
    rates = pd.read_csv(RESULTS / "p1_stats_rates.csv")
    rates = rates[rates.setting == "primary"].set_index(["tool", "rate"])
    fig, axes = plt.subplots(1, len(TAGS), figsize=(7.2, 1.9), sharey=True, sharex=True)
    ys = np.arange(len(TOOLS))[::-1]
    for ax, tag in zip(axes, TAGS):
        for y, t in zip(ys, TOOLS):
            r = rates.loc[(t, f"label_{tag}")]
            ax.barh(y, r.share, height=0.62, color=COLOR[t])
            if r.k:
                ax.plot([r.lo, r.hi], [y, y], color="#333333", lw=0.8)
            ax.text(r.hi + 0.03, y, f"{r.share:.0%}", va="center", fontsize=6.5, color="#333333")
        ax.set_title(tag.replace("_", "-"), fontsize=8)
        finish(ax)
    axes[0].set_yticks(ys, TOOLS)
    axes[0].set_xlim(0, 1)
    axes[0].set_xticks([0, 0.5, 1], ["0", ".5", "1"])
    fig.tight_layout(pad=0.4, w_pad=0.6)
    save(fig, "p1_fig2_failure_modes", fmts)
    return ("**Figure P1-2. Failure modes on the test split (n = 218).** Share of targets assigned each tag (whiskers: "
            "Wilson 95% CI). Tags are assigned in order miss → offset → fragmentation → over-spread → partial/success, "
            "with rules locked on the training folds before the test split was tagged (results/p1_tag_rules.json). "
            "Each tool has a characteristic failure: P2Rank misses, fpocket is offset, the probability tools over-spread.")


def fig3(fmts):
    r = pd.read_csv(RESULTS / "p1_robustness.csv", keep_default_na=False, na_values=[""])   # ladder "null" != NaN
    tests = pd.read_csv(RESULTS / "p1_robustness_null_test.csv")
    rows = [("fpocket", "gate4", "fpocket"), ("fpocket", "gate4_top3", "fpocket top-3"), ("P2Rank", "gate4", "P2Rank"),
            ("P2Rank", "gate4_top3", "P2Rank top-3"), ("AE-PocketMiner", "t0.7", "AE-PocketMiner")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    ys = np.arange(len(rows))[::-1]
    jit = np.random.default_rng(0)
    for y, (t, v, lab) in zip(ys, rows):
        g = r[(r.tool == t) & (r.variant == v)]
        for ax, met in zip(axes, ("flips", "headroom")):
            w = g.pivot(index="parent", columns="ladder", values=met)
            d = (w["nma"] - w["null"]).to_numpy()
            ax.scatter(d, y + jit.uniform(-0.22, 0.22, len(d)), s=7, color=COLOR[t], alpha=0.35, lw=0, zorder=2)
            lo, hi = boot_ci(d, np.mean)
            ax.plot([lo, hi], [y, y], color="#222222", lw=1.6, zorder=3)
            ax.plot(d.mean(), y, "o", ms=6, color=COLOR[t], mec="#222222", mew=1, zorder=4)
            x = tests[(tests.tool == t) & (tests.variant == v) & (tests.metric == met)].iloc[0]
            ax.text(1.0, y, f"{x.null_higher}:{x.nma_higher}  p = {x.p:.1g}", transform=ax.get_yaxis_transform(),
                    fontsize=6.5, va="center", color="#333333")
    for ax in axes:
        ax.axvline(0, color="#555555", lw=0.8, zorder=1)
        finish(ax)
    axes[0].set_yticks(ys, [x[2] for x in rows])
    axes[0].set_title("a  detection flips along the ladder: NMA minus control", loc="left", fontsize=8)
    axes[1].set_title("b  headroom (best - apo MCC): NMA minus control", loc="left", fontsize=8)
    axes[0].set_xlabel("< more flips on the control          more on NMA >", fontsize=7)
    axes[1].set_xlabel("< more headroom on the control          more on NMA >", fontsize=7)
    n = r.parent.nunique()
    fig.tight_layout(pad=0.4, w_pad=4)
    save(fig, "p1_fig3_robustness", fmts)
    return (f"**Figure P1-3. The lowest normal mode does no better than random motion (curated subset, n = {n} targets, "
            f"25 conformers per ladder).** Each small dot is one target: its value on the ladder along the lowest ANM mode "
            f"(both directions, ±1.42 Å global Cα RMSD, no residue > 12 Å) minus its value on a control ladder along a "
            f"random smooth motion at the same amplitudes. Large dot: mean difference, bar: bootstrap 95% CI. Right: "
            f"targets higher on control : higher on NMA, and the paired Wilcoxon p. (a) Number of times site detection "
            f"changes along the ordered ladder. (b) Headroom, the best conformer's MCC minus the apo's. The control "
            f"gives more headroom for every structure tool, so best-of-25 gains are not evidence that the mode opens "
            f"the pocket. pLM-NN is sequence-only, identical on every conformer, and omitted.")


def fig4(fmts):
    panels = []
    for ld, (folder, sfx) in LADDERS.items():
        e = pd.read_csv(RESULTS / f"p1_ensembles{sfx}.csv")
        e = e[e.parent == KRAS].sort_values("conf")
        e["site_volume"] = e.target_id.map(site_volumes(RUNS / folder, e.target_id))
        panels.append((ld, e))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), sharex=True)
    for ld, e in panels:
        kw = dict(color=COLOR["fpocket"]) if ld == "nma" else dict(color=GREY, ls="--", lw=1.5)
        lab = "lowest normal mode" if ld == "nma" else "random smooth motion (control)"
        axes[0].plot(e.ens_rmsd, e.site_volume, marker="o", ms=4, label=lab, **kw)
        axes[1].plot(e.ens_rmsd, e.pocket_rmsd_holo, marker="o", ms=4, **kw)
    apo = panels[0][1].set_index("conf").loc[12]
    big = panels[0][1].loc[panels[0][1].site_volume.idxmax()]
    for ax, col, txt in ((axes[0], "site_volume", "largest volume"), (axes[1], "pocket_rmsd_holo", "same conformer")):
        ax.plot(big.ens_rmsd, big[col], "o", ms=11, mfc="none", mec="#222222", mew=1.2, zorder=5)
        ax.annotate(txt, (big.ens_rmsd, big[col]), xytext=(6, -12), textcoords="offset points", ha="left",
                    va="top", fontsize=7, color="#222222")
    for ax, col in zip(axes, ("site_volume", "pocket_rmsd_holo")):
        ax.axvline(0, color="#c9c9c4", lw=1)
        finish(ax, "y")
        ax.set_xlabel("ladder position: signed global Cα RMSD from apo (Å)")
    axes[0].set_title("a  fpocket site volume (Å³)", loc="left", fontsize=8)
    axes[1].set_title("b  SII-P pocket Cα RMSD to holo (Å; lower = more holo-like)", loc="left", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(pad=0.4)
    save(fig, "p1_fig4_kras_axes", fmts)
    e = panels[0][1]
    hi = e.loc[e.site_volume.idxmax()]
    best = {ld: x.pocket_rmsd_holo.min() for ld, x in panels}
    return (f"**Figure P1-4. The two openness axes disagree on KRAS G12D.** 25-conformer ladders from the prepared apo "
            f"(vertical line). "
            f"(a) Volume of the largest fpocket pocket within 8 Å of the ligand. (b) Cα RMSD of the true pocket residues "
            f"to the holo after superposing on them (tool-independent). Along the normal mode the site volume rises "
            f"from {apo.site_volume:.0f} Å³ (apo) to {hi.site_volume:.0f} Å³ (ladder position {hi.ens_rmsd:+.2f} Å, circled), "
            f"while the pocket RMSD to holo goes from {apo.pocket_rmsd_holo:.2f} to {hi.pocket_rmsd_holo:.2f} Å: "
            f"a larger fpocket volume is not a more holo-like pocket. The closest approach to holo "
            f"is {best['nma']:.2f} Å along the normal mode and {best['null']:.2f} Å along the random control.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", nargs="*", default=["png", "pdf"])
    fmts = ap.parse_args().format
    style()
    caps = [f() for f in (lambda: fig1(fmts), lambda: fig2(fmts), lambda: fig3(fmts), lambda: fig4(fmts))]
    (OUT / "CAPTIONS.md").write_text("# Phase 1 figures\n\nGenerated by `python src/p1_s7_figures.py` from results/.\n\n"
                                     + "\n\n".join(caps) + "\n")
    print("\n\n".join(caps))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
