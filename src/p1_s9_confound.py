"""
P1 s9: per-target AUPRC is a confounded way to ask which targets a predictor handles well.

A residue-level predictor is normally scored per target with AUPRC, and targets are then compared, pooled, or
sorted into "well predicted" and "badly predicted". AUPRC has a floor at the positive rate, and in this task
the positive rate is set by geometry: a pocket is roughly a fixed number of residues, so the share of pocket
residues falls as the protein grows. Per-target AUPRC therefore tracks protein size whether or not the
predictor did anything useful, and any quantity correlated with size inherits that and looks informative.

This is not specific to cryptic pockets or to these tools. It applies wherever a per-instance score with a
prevalence-dependent floor is compared across instances of varying prevalence, which covers most
residue-level, token-level and pixel-level prediction.

What this script shows, from the committed results/p1_master.csv (no re-running of anything):
  1. how strongly each metric tracks the base rate and the protein size, for all four tools
  2. that AUROC and the base-rate-normalised lift do not share the problem, so there is a cheap fix
  3. what it costs: the share of targets whose "well predicted" label changes once the comparison is made
     within base-rate strata rather than against one global median

Outputs: results/p1_confound.csv, printed tables.
Usage (structprep env): python src/p1_s9_confound.py [--split test]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS
from p1_s6_aggregate import holm

METRICS = [("auprc", "AUPRC"), ("auroc", "AUROC"), ("mcc", "MCC"), ("f1", "F1"),
           ("auprc_lift", "AUPRC / base rate")]
NQ = 5


def main():
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
    f = RESULTS / ("p1_master.csv" if split == "test" else "p1_master_train.csv")
    m = pd.read_csv(f)
    m = m[(m.split == split) & m.primary].drop_duplicates(["target_id", "tool"])
    print(f"{m.target_id.nunique()} {split} targets x {m.tool.nunique()} tools, primary variants only")
    print(f"base rate: median {m.base_rate.median():.3f}, range {m.base_rate.min():.3f}-{m.base_rate.max():.3f}")
    print(f"base rate vs protein size: Spearman {spearmanr(m.base_rate, m.n_res).statistic:+.3f}\n")

    rows = []
    for tool, g in m.groupby("tool"):
        for col, lab in METRICS:
            if col not in g or g[col].isna().all():
                continue
            rows.append(dict(tool=tool, metric=lab, n=int(g[col].notna().sum()),
                             rho_base=spearmanr(g[col], g.base_rate, nan_policy="omit").statistic,
                             rho_size=spearmanr(g[col], g.n_res, nan_policy="omit").statistic))
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / "p1_confound.csv", index=False)
    print("1-2. how much each per-target metric tracks the base rate and the protein size (Spearman)")
    for key, lab in (("rho_base", "vs base rate"), ("rho_size", "vs protein size")):
        print(f"\n  {lab}")
        print(r.pivot(index="metric", columns="tool", values=key)
               .loc[[x[1] for x in METRICS]].round(3).to_string())

    print(f"\n3. relabelling cost: targets called 'well predicted' by a global AUPRC median, versus by the")
    print(f"   median within their own base-rate quintile ({NQ} quintiles)")
    out = []
    for tool, g in m.groupby("tool"):
        a = g.auprc.to_numpy()
        if np.isnan(a).all():
            continue
        glob = a > np.median(a)
        q = pd.qcut(g.base_rate, NQ, labels=False, duplicates="drop").to_numpy()
        strat = np.empty(len(a), bool)
        for j in np.unique(q):
            strat[q == j] = a[q == j] > np.median(a[q == j])
        out.append(dict(tool=tool, n=len(a), relabelled=float((glob != strat).mean())))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    print("\nEvery such relabelled target is one the raw comparison places on the wrong side.")

    # 4. the question that decides how much this matters: does the confound change which TOOL wins?
    # If only per-target labels move, the fix is hygiene. If a published pairwise ranking moves, results
    # already in the literature may be artefacts of the metric.
    print("\n4. pairwise tool comparisons under each metric: paired Wilcoxon over the targets both scored,")
    print("   Holm-corrected within each metric. 'winner' is the higher mean; n.s. means p_holm >= 0.05.")
    tools = sorted(m.tool.unique())
    cmp_rows = []
    for col, lab in METRICS:
        tests = []
        for i, a in enumerate(tools):
            for b in tools[i + 1:]:
                w = m[m.tool.isin([a, b])].pivot_table(index="target_id", columns="tool", values=col).dropna()
                if len(w) < 10:
                    continue
                d = (w[a] - w[b]).to_numpy()
                tests.append(dict(metric=lab, pair=f"{a} vs {b}", n=len(d), diff=d.mean(),
                                  winner=a if d.mean() > 0 else b,
                                  p=wilcoxon(d).pvalue if (d != 0).any() else np.nan))
        t = pd.DataFrame(tests)
        if len(t):
            t["p_holm"] = holm(t.p.fillna(1).to_numpy())
            cmp_rows.append(t)
    c = pd.concat(cmp_rows, ignore_index=True)
    c["call"] = np.where(c.p_holm < 0.05, c.winner, "n.s.")
    c.to_csv(RESULTS / "p1_confound_pairs.csv", index=False)
    piv = c.pivot(index="pair", columns="metric", values="call")[[x[1] for x in METRICS if x[1] in set(c.metric)]]
    print(piv.to_string())
    # Two different things move these calls and they must not be conflated. The base-rate confound is
    # isolated by AUPRC vs AUPRC / base rate: same ranking, same threshold-free scoring, only the floor
    # removed. Every other disagreement also changes what is being measured (AUROC weights the negatives
    # differently; MCC and F1 score one thresholded set instead of a ranking), so those flips are about the
    # choice of metric family, which is a separate point.
    ref, fix = "AUPRC", "AUPRC / base rate"
    direct = [(p, piv.loc[p, ref], piv.loc[p, fix]) for p in piv.index if piv.loc[p, fix] != piv.loc[p, ref]]
    other = [(p, c, piv.loc[p, ref], piv.loc[p, c]) for p in piv.index for c in piv.columns
             if c not in (ref, fix) and piv.loc[p, c] != piv.loc[p, ref]]
    print(f"\nremoving the base-rate floor alone changes {len(direct)} of {len(piv)} pairwise conclusions:")
    for p, a, b in direct or [("  (none)", "", "")]:
        print(f"  {p}: AUPRC says {a}, base-rate-normalised says {b}")
    print(f"\nseparately, {len(other)} of {len(piv) * (len(piv.columns) - 2)} calls differ between metric")
    print("families (threshold-free vs thresholded); that is a different argument, not the confound:")
    for p, c, a, b in other:
        print(f"  {p}: AUPRC says {a}, {c} says {b}")

    # 5. is that second group really about the metric, or just about the operating point? MCC and F1 score
    # one thresholded set, and the probability tools are known to over-spread at their defaults, so a flip
    # could be calibration rather than anything the metric measures. The test: hand EVERY tool its best
    # setting on each target - the probability tools their best threshold from the sweep, the pocket tools
    # their best gate and rank cutoff - which upper-bounds what choosing an operating point can buy.
    sweep, sens = RESULTS / "p1_prob_sweep.csv", RESULTS / "p1_master_sens.csv"
    if not (sweep.exists() and sens.exists()):
        print("\n5. skipped: needs results/p1_prob_sweep.csv and results/p1_master_sens.csv "
              "(python src/p1_s3_score.py --topk 1 3 5 --suffix _sens)")
        return
    sw = pd.read_csv(sweep)
    sw = sw[sw.split == split]
    cols = [c for c in sw.columns if c.startswith("mcc_t")]
    best = sw.assign(mcc=sw[cols].max(axis=1))[["target_id", "tool", "mcc"]]
    ms = pd.read_csv(sens)
    ms = ms[(ms.split == split) & ms.variant.str.startswith("gate")]
    best = pd.concat([best, ms.groupby(["target_id", "tool"], as_index=False).mcc.max()], ignore_index=True)
    print(f"\n5. the same pairs on MCC at each tool's BEST per-target setting ({len(cols)} thresholds for the")
    print("   probability tools, every gate and rank cutoff for the pocket tools); an oracle upper bound.")
    rows5 = []
    for i, a in enumerate(tools):
        for b in tools[i + 1:]:
            w = best[best.tool.isin([a, b])].pivot_table(index="target_id", columns="tool", values="mcc").dropna()
            if len(w) < 10:
                continue
            d = (w[a] - w[b]).to_numpy()
            rows5.append(dict(pair=f"{a} vs {b}", n=len(d), diff=d.mean(),
                              winner=a if d.mean() > 0 else b,
                              p=wilcoxon(d).pvalue if (d != 0).any() else np.nan))
    o = pd.DataFrame(rows5)
    o["p_holm"] = holm(o.p.fillna(1).to_numpy())
    o["call"] = np.where(o.p_holm < 0.05, o.winner, "n.s.")
    o.to_csv(RESULTS / "p1_confound_oracle.csv", index=False)
    o["MCC at default"] = [piv.loc[p, "MCC"] for p in o.pair]
    print(o[["pair", "n", "diff", "p_holm", "call", "MCC at default"]].round(4).to_string(index=False))
    same = [p for p, x, y in zip(o.pair, o.call, o["MCC at default"]) if x == y]
    print(f"\n{len(same)} of {len(o)} calls are unchanged when every tool gets its best operating point.")
    print("Unchanged means the thresholded/threshold-free disagreement is not merely a badly chosen threshold.")


if __name__ == "__main__":
    main()
