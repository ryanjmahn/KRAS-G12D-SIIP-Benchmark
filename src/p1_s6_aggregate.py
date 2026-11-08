"""
P1 s6: failure-mode tags (docs/evaluation_protocol.md §6); cross-target statistics follow in this module.

Tags, per tool per target, from the primary variant (pocket tools gate4, probability tools their primary t):
  miss           nothing at the site: no pocket within OFFSET_GATE of the ligand / no true residue >= t
  offset         pocket tools: no pocket within 4 Å of the ligand, but one within OFFSET_GATE
  fragmentation  pocket tools: >= FRAG_MIN_POCKETS at-site pockets and no single one recovers
                 FRAG_MAX_SHARE of what their union recovers
  over_spread    predicts > SPREAD_MIN x the true pocket size with precision < SPREAD_MAX_PRECISION
Tags may overlap. Primary label = first tag in ORDER, else success (MCC >= SUCCESS_MCC) or partial.

Rules are locked on the training folds before the test split is tagged:
  python src/p1_s6_aggregate.py lock    training folds: prevalence + sensitivity grid -> results/p1_tag_rules.json
  python src/p1_s6_aggregate.py tags    test split with the locked rules -> results/p1_tags.csv

Statistics over the CryptoBench test targets (the KRAS anchor is not part of CryptoBench and is left out):
  python src/p1_s6_aggregate.py stats   -> results/p1_stats_{summary,pairwise,rates,mcnemar}.csv
  per tool: mean/median with 95% bootstrap CIs (10,000 resamples of targets); between tools: paired Wilcoxon
  signed-rank, Holm-corrected within each metric, matched-pairs rank-biserial effect size; rates: Wilson 95%
  intervals, exact McNemar between tools.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS

POCKET_TOOLS = {"fpocket", "P2Rank"}
ORDER = ["miss", "offset", "fragmentation", "over_spread"]
RULES = dict(OFFSET_GATE=8.0, FRAG_MIN_POCKETS=2, FRAG_MAX_SHARE=0.67, SPREAD_MIN=2.0,
             SPREAD_MAX_PRECISION=0.3, SUCCESS_MCC=0.5)


def tag_table(master, rules):
    prim = master[master.primary]
    near = master[master.variant == f"gate{rules['OFFSET_GATE']:g}"].set_index(["target_id", "tool"]).site_hit
    rows = []
    for r in prim.itertuples():
        pocket = r.tool in POCKET_TOOLS
        close = bool(near.get((r.target_id, r.tool), False)) if pocket else bool(r.site_hit)
        t = dict(miss=not r.site_hit and not close,
                 offset=pocket and not r.site_hit and close,
                 fragmentation=pocket and r.fragments >= rules["FRAG_MIN_POCKETS"]
                 and r.best_single_recall < rules["FRAG_MAX_SHARE"] * r.recall,
                 over_spread=bool(r.spread > rules["SPREAD_MIN"] and r.precision < rules["SPREAD_MAX_PRECISION"]))
        label = next((k for k in ORDER if t[k]), None) or ("success" if r.mcc >= rules["SUCCESS_MCC"] else "partial")
        rows.append(dict(target_id=r.target_id, tool=r.tool, mcc=r.mcc) | t | dict(label=label))
    return pd.DataFrame(rows)


def prevalence(tags):
    rates = tags.groupby("tool")[ORDER].mean()
    labels = tags.groupby("tool").label.value_counts(normalize=True).unstack(fill_value=0)
    return rates.join(labels[[c for c in ORDER + ["partial", "success"] if c in labels]], rsuffix="_label")


def lock():
    train = pd.read_csv(RESULTS / "p1_master_train.csv")
    tags = tag_table(train, RULES)
    print(f"training folds, {tags.target_id.nunique()} targets: tag rates (overlapping) | primary-label shares (_label)")
    print(prevalence(tags).round(3).to_string())
    print("\nsensitivity on the training folds (share of targets tagged):")
    for mp, sh in itertools.product((2, 3), (0.5, 0.67, 0.8)):
        r = tag_table(train, RULES | dict(FRAG_MIN_POCKETS=mp, FRAG_MAX_SHARE=sh))
        print(f"  fragmentation >= {mp} pockets, share < {sh}:",
              r[r.tool.isin(POCKET_TOOLS)].groupby("tool").fragmentation.mean().round(3).to_dict())
    for s in (1.5, 2.0, 3.0):
        r = tag_table(train, RULES | dict(SPREAD_MIN=s))
        print(f"  over-spread, spread > {s}:", r.groupby("tool").over_spread.mean().round(3).to_dict())
    for g in (6.0, 8.0, 10.0):
        r = tag_table(train, RULES | dict(OFFSET_GATE=g))
        print(f"  offset, near = {g:g} Å:", r[r.tool.isin(POCKET_TOOLS)].groupby("tool").offset.mean().round(3).to_dict())
    (RESULTS / "p1_tag_rules.json").write_text(json.dumps(RULES | {"locked_on": "CryptoBench training folds 0-3"}, indent=2))
    print(f"\nlocked -> {RESULTS / 'p1_tag_rules.json'}")


def tags():
    rules = json.loads((RESULTS / "p1_tag_rules.json").read_text())
    t = tag_table(pd.read_csv(RESULTS / "p1_master.csv"), rules)
    t.to_csv(RESULTS / "p1_tags.csv", index=False)
    print(f"test split, {t.target_id.nunique()} targets, locked rules: tag rates | primary-label shares (_label)")
    print(prevalence(t).round(3).to_string())
    print(f"-> {RESULTS / 'p1_tags.csv'}")


# ---------- statistics ----------

N_BOOT, SEED = 10_000, 0
# Which variant each tool is compared under. "primary": any-rank gate / documented threshold.
# "deployable": pocket tools limited to their top 3 pockets (protocol §4).
SETTINGS = {"primary": {"fpocket": "gate4", "P2Rank": "gate4", "AE-PocketMiner": "t0.7", "pLM-NN": "t0.75"},
            "deployable": {"fpocket": "gate4_top3", "P2Rank": "gate4_top3", "AE-PocketMiner": "t0.7", "pLM-NN": "t0.75"}}
SUMMARY_METRICS = ["mcc", "recall", "precision", "spread", "auprc", "auroc"]


def boot_ci(x, stat, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return tuple(np.percentile(stat(x[idx], axis=1), [2.5, 97.5]))


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation: (sum of ranks of positive - negative diffs) / total; zeros dropped."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def table(master, setting):
    """target x tool frame of the variant each tool is compared under."""
    pick = SETTINGS[setting]
    m = master[master.split != "anchor"]
    return pd.concat([m[(m.tool == t) & (m.variant == v)] for t, v in pick.items()])


def stats():
    master = pd.read_csv(RESULTS / "p1_master.csv")
    summary, pairwise, rates, mcn = [], [], [], []
    for setting in SETTINGS:
        t = table(master, setting)
        for tool, g in t.groupby("tool"):
            for met in SUMMARY_METRICS:
                x = g[met].to_numpy(float)
                summary.append(dict(setting=setting, tool=tool, metric=met, n=int((~np.isnan(x)).sum()),
                                    mean=np.nanmean(x), mean_lo=boot_ci(x, np.mean)[0], mean_hi=boot_ci(x, np.mean)[1],
                                    median=np.nanmedian(x), median_lo=boot_ci(x, np.median)[0],
                                    median_hi=boot_ci(x, np.median)[1]))
        for met in ("mcc", "auprc"):
            if met == "auprc" and setting != "primary":
                continue                                   # AUPRC does not depend on the setting
            wide = t.pivot(index="target_id", columns="tool", values=met)
            fam = []
            for a, b in itertools.combinations(sorted(wide.columns), 2):
                d = (wide[a] - wide[b]).dropna()
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                fam.append(dict(setting=setting, metric=met, tool_a=a, tool_b=b, n=len(d),
                                median_diff=d.median(), mean_diff=d.mean(), a_wins=int((d > 0).sum()),
                                b_wins=int((d < 0).sum()), ties=int((d == 0).sum()), p=p,
                                rank_biserial=rank_biserial(d)))
            for row, padj in zip(fam, holm([r["p"] for r in fam])):
                pairwise.append(row | dict(p_holm=padj))
        # rates: detection (pocket tools), any-signal-at-site (all), and tag labels
        for col, tools in (("top1", POCKET_TOOLS), ("top3", POCKET_TOOLS), ("site_hit", None)):
            sub = t if tools is None else t[t.tool.isin(tools)]
            for tool, g in sub.groupby("tool"):
                k, n = int(g[col].astype(bool).sum()), len(g)
                rates.append(dict(setting=setting, tool=tool, rate=col, k=k, n=n, share=k / n,
                                  lo=wilson(k, n)[0], hi=wilson(k, n)[1]))
            wide = sub.pivot(index="target_id", columns="tool", values=col).astype(bool)
            for a, b in itertools.combinations(sorted(wide.columns), 2):
                only_a, only_b = int((wide[a] & ~wide[b]).sum()), int((~wide[a] & wide[b]).sum())
                p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
                mcn.append(dict(setting=setting, rate=col, tool_a=a, tool_b=b, only_a=only_a, only_b=only_b, p=p))
    tags = pd.read_csv(RESULTS / "p1_tags.csv")
    tags = tags[tags.target_id != "kras_g12d_siip"]
    for tool, g in tags.groupby("tool"):
        for lab in ORDER + ["partial", "success"]:
            k, n = int((g.label == lab).sum()), len(g)
            rates.append(dict(setting="primary", tool=tool, rate=f"label_{lab}", k=k, n=n, share=k / n,
                              lo=wilson(k, n)[0], hi=wilson(k, n)[1]))
    out = {"summary": pd.DataFrame(summary), "pairwise": pd.DataFrame(pairwise),
           "rates": pd.DataFrame(rates), "mcnemar": pd.DataFrame(mcn)}
    mc = out["mcnemar"]
    mc["p_holm"] = mc.groupby(["setting", "rate"]).p.transform(lambda s: holm(s.to_numpy()))
    for k, df in out.items():
        df.to_csv(RESULTS / f"p1_stats_{k}.csv", index=False)

    s = out["summary"]
    n = int(s.n.max())
    print(f"{n} CryptoBench test targets; 95% bootstrap CIs\n")
    for setting in SETTINGS:
        v = s[(s.setting == setting) & (s.metric.isin(["mcc", "auprc"]))]
        print(f"[{setting}] mean (95% CI):")
        for tool, g in v.groupby("tool"):
            print("  " + f"{tool:15}" + "  ".join(f"{r.metric} {r['mean']:.3f} ({r.mean_lo:.3f}-{r.mean_hi:.3f})"
                                                 for _, r in g.iterrows()))
    print("\npaired Wilcoxon, Holm-corrected within metric; r = rank-biserial (positive favours tool_a):")
    for _, r in out["pairwise"].iterrows():
        star = "*" if r.p_holm < 0.05 else " "
        print(f"  {star} [{r.setting:10} {r.metric:5}] {r.tool_a:14} vs {r.tool_b:14} "
              f"wins {r.a_wins:3}-{r.b_wins:<3} median diff {r.median_diff:+.3f}  r {r.rank_biserial:+.2f}  p_holm {r.p_holm:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["lock", "tags", "stats"])
    {"lock": lock, "tags": tags, "stats": stats}[ap.parse_args().step]()
