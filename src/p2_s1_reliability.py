"""
P2 s1: is a prediction's stability under small perturbation a usable confidence signal?

Phase 1 found every structure tool's output moves under <= 1.4 Å of backbone motion (detection flips on 44% of
targets for fpocket, 28% for P2Rank), and the pilot (p2_s0_pilot.py) found ensembles add no accuracy. This asks
the other question: the instability is there, can it be READ? Nothing here is trained; these are the raw signals
a confidence estimate would be built from.

Per target, per tool, per generator (nma1 / nmaK / random, 10 conformers each, from p2_s0_pilot.py):
  rho       mean Spearman of the per-residue scores, apo vs conformer (threshold-free ranking stability)
  jaccard   mean Jaccard of the top-TOPK residues, apo vs conformer
  sd        mean per-residue SD across the conformers
Each is compared against the confidence a practitioner already has without any ensemble:
  own_max   the tool's highest apo score
  own_top   the mean of its TOPK highest apo scores

TARGET level (is this target's prediction trustworthy?): Spearman of each signal with the apo AUPRC across
targets, and the AUC for separating well-predicted targets (apo AUPRC above the tool's median) from the rest.
RESIDUE level (is this residue trustworthy?): among the apo's TOPK residues, does how often a residue stays in
the conformers' top-TOPK (its frequency) rank the true pocket residues above the false ones better than the
tool's own apo score does? Paired over targets.

Outputs: results/p2_reliability.csv (per target), results/p2_reliability_summary.csv, printed summary.
Usage (structprep env): python src/p2_s1_reliability.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS
from p2_s0_pilot import GENS, N_CONF, OUT, P1, sample, tool_scores

TOPK = 10


def auc(y, s):
    """P(score of a positive > score of a negative), ties counted as half."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().to_numpy()
    n1 = int(y.sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def topk(s, k=TOPK):
    return set(np.argsort(-np.asarray(s), kind="stable")[:k].tolist())


def per_target(tid):
    from p1_adapters import ca_table
    from p1_s3_score import threshold_free
    meta = json.loads((P1 / tid / "meta.json").read_text())
    keys, _ = ca_table(P1 / tid / f"{tid}.pdb")
    truth = set(meta["truth"]) & set(keys)
    hit = np.array([k in truth for k in keys])
    apo = tool_scores(P1, tid, keys)
    rows = []
    for tool, a in apo.items():
        base = dict(parent=tid, tool=tool, n_res=len(keys),
                    apo_auprc=threshold_free(dict(zip(keys, a)), truth, keys)["auprc"],
                    own_max=float(a.max()), own_top=float(np.sort(a)[-TOPK:].mean()),
                    own_auc_topk=auc(hit[sorted(topk(a))], a[sorted(topk(a))]))
        ta = topk(a)
        for gen in GENS:
            confs = [tool_scores(OUT / gen, f"{tid}__c{k:02d}", keys).get(tool) for k in range(N_CONF)]
            if any(c is None for c in confs):
                continue
            C = np.vstack(confs)
            rho = np.nanmean([spearmanr(a, c).statistic for c in confs])
            jac = np.mean([len(ta & topk(c)) / len(ta | topk(c)) for c in confs])
            # how often each of the apo's top-TOPK residues stays in a conformer's top-TOPK
            freq = np.array([np.mean([i in topk(c) for c in confs]) for i in sorted(ta)])
            idx = sorted(ta)
            # does stability ADD to the tool's own score? rank-sum of the two, so nothing is fitted
            combo = pd.Series(a[idx]).rank().to_numpy() + pd.Series(freq).rank().to_numpy()
            rows.append(base | dict(gen=gen, rho=rho, jaccard=jac, sd=float(C.std(0).mean()),
                                    freq_auc_topk=auc(hit[idx], freq), combo_auc_topk=auc(hit[idx], combo)))
    return rows


def main():
    ids = sample()
    r = pd.DataFrame([x for tid in ids for x in per_target(tid)])
    r.to_csv(RESULTS / "p2_reliability.csv", index=False)

    out = []
    for (tool, gen), g in r.groupby(["tool", "gen"]):
        good = g.apo_auprc > g.apo_auprc.median()
        row = dict(tool=tool, gen=gen, n=len(g))
        for sig, sign in (("rho", 1), ("jaccard", 1), ("sd", -1), ("own_max", 1), ("own_top", 1)):
            row[f"rho_{sig}"] = spearmanr(g[sig], g.apo_auprc).statistic
            row[f"auc_{sig}"] = auc(good, sign * g[sig])
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(RESULTS / "p2_reliability_summary.csv", index=False)

    print(f"{r.parent.nunique()} training targets. TARGET level: does the signal say which predictions are good?")
    print("  rho_* = Spearman with apo AUPRC; auc_* = AUC for 'apo AUPRC above this tool's median'.")
    print("  rho/jaccard/sd need an ensemble; own_max/own_top are free (no ensemble).")
    print(s.round(3).to_string(index=False))

    print("\nRESIDUE level, among each apo's top-10 residues: AUC of the true pocket residues over the false ones.")
    print("  freq = how often the residue stays in a conformer's top-10; own = the tool's apo score.")
    for (tool, gen), g in r.groupby(["tool", "gen"]):
        line = f"  {tool:15s} {gen:7s} own {g.own_auc_topk.mean():.3f}"
        for col in ("freq_auc_topk", "combo_auc_topk"):
            d = (g[col] - g.own_auc_topk).dropna()
            p = wilcoxon(d).pvalue if len(d) and (d != 0).any() else np.nan
            line += f" | {col.split('_')[0]} {g[col].mean():.3f} ({d.mean():+.3f}, p={p:.3g})"
        print(line + f"  n={int(g.own_auc_topk.notna().sum())}")


if __name__ == "__main__":
    main()
