"""
T2 transfer: is there one trust rule, or four?

Four bespoke rules are a weaker claim than one shared rule: a shared rule would say that what makes a
cryptic-pocket prediction believable is a property of the prediction, not of the tool that made it. This
applies each tool's frozen rule to every other tool's held-out features and reads the diagonal against the
off-diagonal.

A rule also only transfers if the features mean the same thing in both places, which they do not entirely:
`pk_gap` exists only for the pocket tools, and agreement features are named per partner. Columns a borrowed
rule asks for and the borrowing tool does not have are filled with the lending tool's training median, which
is what the frozen artifact does for any missing value.

Also fitted here: a single pooled rule trained on all four tools' training rows at once, which is the
honest "one rule" candidate rather than a borrowed one.

Outputs: results/trust/t2_transfer.csv, printed matrix.
Usage (structprep env): python src/t2_transfer.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS
from t1_consensus import NQ, _strat
from t2_model import COVER, T1, fit_logit, prep, tier2
from t3_calibrate import TARGET, TOOLS
from t3_rule import RULE, trust

warnings.filterwarnings("ignore")


def main():
    d = pd.read_csv(RESULTS / "trust" / "features.csv")
    rule = json.loads(RULE.read_text())
    te = d[d.split == "test"]

    rows = []
    for lender in TOOLS:
        for borrower in TOOLS:
            g = te[te.tool == borrower].reset_index(drop=True)
            y = g[TARGET].to_numpy(bool)
            q = pd.qcut(g.base_rate, NQ, labels=False, duplicates="drop").to_numpy()
            s = np.array([trust(x, lender, rule)[0] for x in g.to_dict("records")])
            rows.append(dict(lender=lender, borrower=borrower, n=len(g),
                             auc=_strat(y, s, q, np.arange(len(y)))))

    # a genuinely shared rule: one model fitted on every tool's training rows together
    tr_all = d[d.split == "train"]
    shared = {}
    ok = [c for c in te.columns if c in tr_all.columns
          and tr_all[c].notna().mean() >= 0.5 and te[c].notna().mean() >= 0.5]
    cols = [c for c in T1 if c in ok] + [c for c in tier2(tr_all.columns)
                                         if c in ok and "pLM-NN" not in c]
    Xtr, _ = prep(tr_all, tr_all, cols)
    f = fit_logit(Xtr, tr_all[TARGET].to_numpy(float))
    for borrower in TOOLS:
        g = te[te.tool == borrower].reset_index(drop=True)
        _, Xte = prep(tr_all, g, cols)
        y = g[TARGET].to_numpy(bool)
        q = pd.qcut(g.base_rate, NQ, labels=False, duplicates="drop").to_numpy()
        shared[borrower] = _strat(y, f(Xte), q, np.arange(len(y)))
        rows.append(dict(lender="pooled (all tools)", borrower=borrower, n=len(g), auc=shared[borrower]))

    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / "trust" / "t2_transfer.csv", index=False)
    m = r.pivot(index="lender", columns="borrower", values="auc").reindex(
        index=TOOLS + ["pooled (all tools)"], columns=TOOLS)
    print("held-out stratified AUC: rows = whose rule, columns = whose predictions it judges")
    print(m.round(3).to_string())

    own = pd.Series({t: m.loc[t, t] for t in TOOLS})
    bor = pd.Series({t: m.loc[[x for x in TOOLS if x != t], t].max() for t in TOOLS})
    out = pd.DataFrame(dict(own_rule=own, best_borrowed=bor,
                            pooled=pd.Series(shared), loss_if_shared=own - pd.Series(shared)))
    print("\neach tool judged by its own rule, by the best borrowed one, and by the pooled rule")
    print(out.round(3).to_string())
    print(f"\nmean cost of using one pooled rule instead of four: "
          f"{out.loss_if_shared.mean():+.3f} AUC")


if __name__ == "__main__":
    main()
