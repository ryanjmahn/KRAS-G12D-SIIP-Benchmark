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

Outputs: results/p1_confound.csv, p1_confound_pairs.csv, p1_leaderboard.csv, printed tables.
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
from figstyle import boot_ci

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

    # 4. the question that decides how much this matters: does the choice change which TOOL wins?
    # Every metric is evaluated on the same targets, and the thresholded metrics appear twice: once at the
    # default operating point and once at each tool's BEST per-target setting. That second form is what
    # separates "the metrics disagree" from "one tool was handed a bad threshold" - an oracle no user could
    # reach, so if a disagreement survives it, the operating point is not the explanation.
    sweep, sens = RESULTS / "p1_prob_sweep.csv", RESULTS / "p1_master_sens.csv"
    if not (sweep.exists() and sens.exists()):
        sys.exit("needs results/p1_prob_sweep.csv and results/p1_master_sens.csv "
                 "(python src/p1_s3_score.py --topk 1 3 5 --suffix _sens)")
    sw = pd.read_csv(sweep); sw = sw[sw.split == split]
    ms = pd.read_csv(sens); ms = ms[(ms.split == split) & ms.variant.str.startswith("gate")]

    frames = {lab: m[["target_id", "tool", col]].rename(columns={col: "v"})
              for col, lab in METRICS if col in m}
    for met in ("mcc", "f1"):                       # oracle operating point, per target, per tool
        cols = [c for c in sw.columns if c.startswith(f"{met}_t")]
        best = sw.assign(v=sw[cols].max(axis=1))[["target_id", "tool", "v"]]
        best = pd.concat([best, ms.groupby(["target_id", "tool"], as_index=False)[met].max()
                                 .rename(columns={met: "v"})], ignore_index=True)
        frames[f"{met.upper()} @ best"] = best

    tools = sorted(m.tool.unique())
    rows = []
    for lab, fr in frames.items():
        tests = []
        for i, a in enumerate(tools):
            for b in tools[i + 1:]:
                w = fr[fr.tool.isin([a, b])].pivot_table(index="target_id", columns="tool", values="v").dropna()
                if len(w) < 10 or not {a, b} <= set(w.columns):
                    continue
                d = (w[a] - w[b]).to_numpy()
                lo, hi = boot_ci(d)                 # bootstrap over targets, so every call carries its CI
                tests.append(dict(metric=lab, pair=f"{a} vs {b}", n=len(d), diff=d.mean(), lo=lo, hi=hi,
                                  p=wilcoxon(d).pvalue if (d != 0).any() else np.nan,
                                  winner=a if d.mean() > 0 else b))
        t = pd.DataFrame(tests)
        if len(t):
            t["p_holm"] = holm(t.p.fillna(1).to_numpy())    # corrected within each metric
            rows.append(t)
    c = pd.concat(rows, ignore_index=True)
    c["call"] = np.where(c.p_holm < 0.05, c.winner, "n.s.")
    c["decided"] = (c.lo > 0) | (c.hi < 0)          # does the bootstrap CI exclude a tie?
    c.to_csv(RESULTS / "p1_confound_pairs.csv", index=False)

    order = [l for l in frames if l in set(c.metric)]
    print("\n4. which tool wins, under each metric. '@ best' gives every tool its best per-target operating")
    print("   point (19 thresholds for the probability tools, every gate and rank cutoff for the pocket")
    print("   tools). Calls are Holm-corrected within each metric; n.s. means the pair is not separated.")
    print(c.pivot(index="pair", columns="metric", values="call")[order].to_string())
    print("\n   mean paired difference with a bootstrap 95% CI over targets (first tool minus second):")
    for lab in order:
        g = c[c.metric == lab]
        print(f"\n   {lab}")
        print(g[["pair", "n", "diff", "lo", "hi", "p_holm", "decided"]].round(4).to_string(index=False))

    ref, fix = "AUPRC", "AUPRC / base rate"
    piv = c.pivot(index="pair", columns="metric", values="call")
    direct = [(p, piv.loc[p, ref], piv.loc[p, fix]) for p in piv.index if piv.loc[p, fix] != piv.loc[p, ref]]
    print(f"\n5. removing the base-rate floor alone changes {len(direct)} of {len(piv)} pairwise conclusions:")
    for p, a, b in direct or [("  (none)", "", "")]:
        print(f"   {p}: AUPRC says {a}, base-rate-normalised says {b}")

    fam = [(p, mc, piv.loc[p, ref], piv.loc[p, mc]) for p in piv.index
           for mc in ("MCC @ best", "F1 @ best") if mc in piv and piv.loc[p, mc] != piv.loc[p, ref]]
    nb = len([mc for mc in ("MCC @ best", "F1 @ best") if mc in piv])
    print(f"\n6. and {len(fam)} of {len(piv) * nb} calls still differ from AUPRC when the thresholded metrics")
    print("   are given their best operating point, so the disagreement is not a badly chosen threshold:")
    for p, mc, a, b in fam:
        print(f"   {p}: AUPRC says {a}, {mc} says {b}")

    print("\n7. the leaderboard, before and after the free fix (mean over targets, best first):")
    lb = pd.DataFrame({lab: fr.groupby("tool").v.mean() for lab, fr in frames.items()})[order]
    lb.to_csv(RESULTS / "p1_leaderboard.csv")
    print(lb.round(4).to_string())
    print("\n   rank order:")
    for lab in order:
        print(f"   {lab:>18s}: " + "  >  ".join(lb[lab].sort_values(ascending=False).index))


if __name__ == "__main__":
    main()
