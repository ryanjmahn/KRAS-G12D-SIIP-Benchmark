"""
T3: the frozen rule applied to the KRAS G12D Switch-II pocket, the target this project started on.

A worked example, not evidence (K7: n = 1). The point is to show what the rule would have told someone
looking at this structure before they knew the answer, and to check it against what the parent study
measured on the same target.

KRAS is handled separately rather than added to results/trust/features.csv: that table defines the training
set, and the anchor would be silently folded into it. Its features are computed here and discarded.

Outputs: results/trust/t3_kras.csv, printed table.
Usage (structprep env): python src/t3_kras.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import srcpath  # noqa: F401  (puts the sibling folders on the path)
from paths import RESULTS, RUNS
from t0_extract import CACHE, one
from t0_features import per_target
from t3_calibrate import TARGET, TOOLS
from t3_rule import trust

warnings.filterwarnings("ignore")
KRAS = "kras_g12d_siip"


def main():
    tdir = RUNS / "p1" / KRAS
    if not tdir.exists():
        sys.exit(f"no prepared anchor at {tdir}")
    f = CACHE / f"{KRAS}.npz"
    if not f.exists():                              # extract on demand; the anchor is not in the sweep
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **one(KRAS, tdir))
        print(f"extracted the anchor -> {f}")

    from p1_adapters import ca_table
    rows = per_target(f, {KRAS: ca_table(tdir / f"{KRAS}.pdb")[1]})
    feat = {r["tool"]: r for r in rows}
    meta = json.loads((tdir / "meta.json").read_text())
    print(f"{KRAS}: {meta['apo_chains']} chain(s), {len(meta['truth'])} true SII-P residues, "
          f"base rate {rows[0]['base_rate']:.3f}\n")

    out = []
    for tool in TOOLS:
        if tool not in feat:
            continue
        r = feat[tool]
        s, p, act = trust(r, tool)
        out.append(dict(tool=tool, probability=p, action=act, truth=bool(r[TARGET]),
                        jac_pLM_NN=r.get("jac_pLM-NN", np.nan), own_max=r["own_max"]))
    t = pd.DataFrame(out)
    t["rule_right"] = (t.action == "investigate") == t.truth
    t.to_csv(RESULTS / "trust" / "t3_kras.csv", index=False)

    print("what the rule would have said, before anyone looked at the holo structure")
    print(t.round(3).to_string(index=False))
    print("\n  truth       = the tool's top-10 really did contain a true SII-P residue")
    print("  rule_right  = the recommendation matched that outcome")
    print(f"\n  {int(t.rule_right.sum())}/{len(t)} recommendations matched the outcome. "
          "One target is an illustration, not evidence.")


if __name__ == "__main__":
    main()
