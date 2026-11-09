"""
P1 s8: is the fpocket / P2Rank reversal a property of the tools, or of where we drew two arbitrary lines?

Phase 1's one inferential claim is that the ranking of the two pocket tools flips between scoring rules:
fpocket wins when a hit at any rank counts, P2Rank wins when only the top-ranked pockets may count. Two
constants decide that: the site gate (a pocket counts if its centre is within 4 Å of a ligand atom) and the
rank cutoff (top 3). Both are conventions, and published "found" criteria range over 4-12 Å. If the reversal
survives both sweeps it is a property of the tools; if it flips, it is a property of the convention and the
paper has to say so.

Reads results/p1_master_sens.csv, written by:
    python src/p1_s3_score.py --topk 1 3 5 --suffix _sens
Paired over the targets both tools scored, Wilcoxon signed-rank, Holm-corrected over every row.

Outputs: results/p1_sensitivity.csv, printed table.
Usage (structprep env): python src/p1_s8_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS
from p1_s6_aggregate import holm
from figstyle import boot_ci

PAIR = ("fpocket", "P2Rank")
METRIC = "mcc"


def main():
    f = RESULTS / "p1_master_sens.csv"
    if not f.exists():
        sys.exit("run first: python src/p1_s3_score.py --topk 1 3 5 --suffix _sens")
    m = pd.read_csv(f)
    m = m[m.split == "test"]
    rows = []
    for variant, g in m[m.tool.isin(PAIR)].groupby("variant"):
        w = g.pivot_table(index="target_id", columns="tool", values=METRIC).dropna()
        if len(w) < 10 or not set(PAIR) <= set(w.columns):
            continue
        d = (w[PAIR[0]] - w[PAIR[1]]).to_numpy()
        lo, hi = boot_ci(d)
        rows.append(dict(variant=variant, n=len(d), fpocket=w[PAIR[0]].mean(), p2rank=w[PAIR[1]].mean(),
                         diff=d.mean(), lo=lo, hi=hi, wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                         p=wilcoxon(d).pvalue if (d != 0).any() else np.nan,
                         winner=PAIR[0] if d.mean() > 0 else PAIR[1]))
    r = pd.DataFrame(rows)
    order = [v for v in ["gate4", "gate6", "gate8", "gate10", "gate12",
                         "gate4_top1", "gate4_top3", "gate4_top5"] if v in set(r.variant)]
    r = r.set_index("variant").loc[order].reset_index()
    r["p_holm"] = holm(r.p.fillna(1).to_numpy())
    r.to_csv(RESULTS / "p1_sensitivity.csv", index=False)

    print(f"fpocket minus P2Rank, mean residue {METRIC.upper()} on the test split, paired over targets.")
    print("Top block sweeps the site gate (any rank counts); bottom block sweeps the rank cutoff at a 4 A gate.")
    print(r.round(4).to_string(index=False))
    gates = r[~r.variant.str.contains("top")]
    tops = r[r.variant.str.contains("top")]
    print(f"\ngate sweep 4-12 A: winner is {set(gates.winner)}, "
          f"significant in {int((gates.p_holm < 0.05).sum())}/{len(gates)} settings")
    print(f"rank sweep top-1/3/5: winner is {set(tops.winner)}, "
          f"significant in {int((tops.p_holm < 0.05).sum())}/{len(tops)} settings")
    flips = set(r.winner)
    print(f"\n{'REVERSAL HOLDS' if len(flips) > 1 else 'NO REVERSAL'}: "
          f"{'the winner changes with the scoring rule' if len(flips) > 1 else 'one tool wins everywhere'}")


if __name__ == "__main__":
    main()
