"""
P2 s6: does doubling the perturbation amplitude rescue the normal modes?

Every other Phase 2 result is measured at one amplitude (MAX_RMSD = 1.42 A global C-alpha RMSD), and at that
amplitude a random smooth deformation matches or beats the lowest normal mode everywhere. The obvious
objection is that 1.42 A is simply too small for collective motion to matter, and that the modes would pull
ahead if the backbone were moved further. This compares the two generators at 1x and 2x amplitude.

The comparison is restricted to the targets present in BOTH caches and uses the same code path for each, so
the amplitude is the only thing that differs. Without that restriction the 2x run (a 200-target subsample)
would be compared against the 1x run (all 846) and the target mix would be confounded with the amplitude.

Reads runs/p2_cache (1x) and runs/p2_cache_amp2 (2x, built by
    python src/p2_s2_scale.py run --scale 2 --gens nma1,random --tag amp2 --limit 200
).
Outputs: results/p2_amplitude.csv, printed table.
Usage (structprep env): python src/p2_s6_amplitude.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS
from figstyle import boot_ci
from p2_s1_reliability import per_target
from p2_s2_scale import cache_dir

warnings.filterwarnings("ignore")

GENS = ("nma1", "random")
SIGNAL = "freq_auc_topk"        # stability alone: the quantity the generators are being compared on


def table(cache, ids):
    return pd.DataFrame([r for f in sorted(cache.glob("*.npz")) if f.stem in ids for r in per_target(f)])


def main():
    one, two = cache_dir("train"), cache_dir("train", "amp2")
    if not two.exists():
        sys.exit("no 2x cache; run: python src/p2_s2_scale.py run --scale 2 --gens nma1,random "
                 "--tag amp2 --limit 200")
    ids = {f.stem for f in two.glob("*.npz")} & {f.stem for f in one.glob("*.npz")}
    print(f"{len(ids)} targets cached at both amplitudes")
    r = {"1x": table(one, ids), "2x": table(two, ids)}

    rows = []
    for amp, t in r.items():
        for (tool, gen), g in t[t.gen.isin(GENS)].groupby(["tool", "gen"]):
            d = (g[SIGNAL] - g.own_auc_topk).dropna().to_numpy()
            lo, hi = boot_ci(d)
            rows.append(dict(tool=tool, gen=gen, amp=amp, n=len(d), own=g.own_auc_topk.mean(),
                             gain=d.mean(), lo=lo, hi=hi))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "p2_amplitude.csv", index=False)
    print("\ngain in top-10 AUC from stability alone, over the tool's own score, on the SAME targets")
    print(t.pivot_table(index=["tool", "gen"], columns="amp", values="gain").round(4).to_string())

    print("\nrandom minus the lowest normal mode, paired over targets (positive = random wins)")
    out = []
    for amp, tt in r.items():
        for tool, g in tt.groupby("tool"):
            w = g[g.gen.isin(GENS)].pivot_table(index="parent", columns="gen", values=SIGNAL).dropna()
            if len(w) < 10:
                continue
            d = (w["random"] - w["nma1"]).to_numpy()
            lo, hi = boot_ci(d)
            out.append(dict(tool=tool, amp=amp, n=len(d), diff=d.mean(), lo=lo, hi=hi,
                            p=wilcoxon(d).pvalue if (d != 0).any() else np.nan))
    o = pd.DataFrame(out).sort_values(["tool", "amp"])
    print(o.round(4).to_string(index=False))
    print("\nIf the modes needed a larger amplitude, this difference would shrink from 1x to 2x.")


if __name__ == "__main__":
    main()
