"""
T0: the per-(target, tool) feature table and the three target definitions.

Features come from runs/trust_cache (src/t0_extract.py) and must be computable from the apo structure and the
tools' own outputs alone. Targets come from the truth labels, which is correct for a label and forbidden for
a feature; tests/test_t0_leakage.py enforces the separation by rebuilding the table with the labels permuted
and requiring every feature column to come back bit-identical.

  Tier 1  the tool's own signal        own_max, own_top10, own_mean, n_pred, spread, and for pocket tools
                                       the top two pocket scores and the gap between them
  Tier 2  cross-tool consensus         pairwise Jaccard of predicted sets, Spearman of score vectors, centre
                                       distance between pocket tools, and consensus core sizes
  Tier 3  structural context           n_res, B-factor statistics, radius of gyration, compactness
  Tier 4  conformational stability     rho / jaccard / sd, joined from results/p2_reliability*.csv

`base_rate` is carried for stratification only and is never a feature: it is the confound the parent paper
identifies, and a model given it would relearn protein size.

Outputs: results/trust/features.csv
Usage (structprep env): python src/t0_features.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, RUNS

warnings.filterwarnings("ignore")
CACHE = RUNS / "trust_cache"
OUT = RESULTS / "trust"
TOOLS = ["fpocket", "P2Rank", "AE-PocketMiner", "pLM-NN"]
POCKET = {"fpocket", "P2Rank"}
TOPK = 10


def jaccard(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def per_target(f, coords):
    from p1_s3_score import threshold_free
    z = np.load(f, allow_pickle=False)
    keys, hit = [str(k) for k in z["keys"]], z["hit"].astype(bool)
    truth = {k for k, h in zip(keys, hit) if h}
    have = [t for t in TOOLS if f"score|{t}" in z.files]
    sc = {t: z[f"score|{t}"].astype(float) for t in have}
    pr = {t: z[f"pred|{t}"].astype(bool) for t in have}
    bf = z["bfac"].astype(float)
    c = coords.get(f.stem)
    rg = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())) if c is not None else np.nan

    rows = []
    for t in have:
        a, p = sc[t], pr[t]
        top = np.argsort(-a, kind="stable")[:TOPK]
        r = dict(parent=f.stem, tool=t, n_res=len(keys), base_rate=float(hit.mean()),
                 # --- targets (ground truth; never features) ---
                 y_top10hit=bool(hit[top].any()),
                 y_auprc_lift=threshold_free(dict(zip(keys, a)), truth, keys)["auprc"] / max(hit.mean(), 1e-9),
                 # --- tier 1 ---
                 own_max=float(a.max()), own_top10=float(np.sort(a)[-TOPK:].mean()), own_mean=float(a.mean()),
                 n_pred=int(p.sum()), spread=float(p.sum() / len(keys)),
                 # --- tier 3 ---
                 bfac_mean=float(np.nanmean(bf)), bfac_sd=float(np.nanstd(bf)),
                 bfac_top=float(np.nanmean(bf[top])), rg=rg,
                 compact=float(len(keys) ** (1 / 3) / rg) if rg and np.isfinite(rg) else np.nan)
        if t in POCKET and f"pocket|{t}" in z.files:
            s1, s2, gap, npk, vol = z[f"pocket|{t}"].astype(float)
            r |= dict(pk_score1=s1, pk_score2=s2, pk_gap=gap, pk_n=npk, pk_vol=vol)

        # --- tier 2: agreement with every other tool that ran on this target ---
        for o in TOOLS:
            if o == t or o not in have:
                continue
            r[f"jac_{o}"] = jaccard(p, pr[o])
            r[f"rho_{o}"] = float(spearmanr(a, sc[o]).statistic)
            if t in POCKET and o in POCKET and f"centre|{t}" in z.files and f"centre|{o}" in z.files:
                r[f"dca_{o}"] = float(np.linalg.norm(z[f"centre|{t}"] - z[f"centre|{o}"]))
        stack = np.vstack([pr[o] for o in have])
        r["cons_k2"] = int((stack.sum(0) >= 2).sum())
        r["cons_k3"] = int((stack.sum(0) >= 3).sum())
        r["cons_frac"] = float(np.mean([jaccard(p, pr[o]) for o in have if o != t])) if len(have) > 1 else np.nan
        rows.append(r)
    return rows


def main():
    files = sorted(CACHE.glob("*.npz"))
    if not files:
        sys.exit("no cache; run: python src/t0_extract.py")
    from p1_adapters import ca_table
    coords = {}
    for f in files:                                 # radius of gyration needs the CA coordinates
        p = RUNS / "p1" / f.stem / f"{f.stem}.pdb"
        if p.exists():
            coords[f.stem] = ca_table(p)[1]
    d = pd.DataFrame([r for f in files for r in per_target(f, coords)])

    split = {}
    for m in sorted((RUNS / "p1").glob("*/meta.json")):
        j = json.loads(m.read_text())
        split[j["target_id"]] = "test" if j["split"] == "test" else "train"
    d["split"] = d.parent.map(split)

    # target (a) needs the per-target MCC, which lives in the scored master tables
    mcc = pd.concat([pd.read_csv(RESULTS / n)[["target_id", "tool", "mcc", "primary"]]
                     for n in ("p1_master.csv", "p1_master_train.csv") if (RESULTS / n).exists()])
    mcc = mcc[mcc.primary].drop_duplicates(["target_id", "tool"]).rename(columns={"target_id": "parent"})
    d = d.merge(mcc[["parent", "tool", "mcc"]], on=["parent", "tool"], how="left")
    d["y_mcc30"] = d.mcc > 0.3
    d["y_mcc_med"] = d.mcc > d.groupby("tool").mcc.transform("median")

    # tier 4: the perturbation-stability signals already measured, where they exist
    rel = [pd.read_csv(RESULTS / n) for n in ("p2_reliability.csv", "p2_reliability_test.csv")
           if (RESULTS / n).exists()]
    if rel:
        r = pd.concat(rel)
        r = r[r.gen == "random"][["parent", "tool", "rho", "jaccard", "sd"]]
        d = d.merge(r.rename(columns={"rho": "pert_rho", "jaccard": "pert_jac", "sd": "pert_sd"}),
                    on=["parent", "tool"], how="left")

    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "features.csv", index=False)
    print(f"{len(d)} rows: {d.parent.nunique()} targets x {d.tool.nunique()} tools")
    print(d.groupby(["split", "tool"]).size().unstack(fill_value=0).to_string())
    miss = d.isna().mean().sort_values(ascending=False)
    print("\ncolumns with missing values (no silent imputation anywhere):")
    print(miss[miss > 0].round(3).to_string() if (miss > 0).any() else "  none")
    print(f"\n-> {OUT / 'features.csv'}")


if __name__ == "__main__":
    main()
