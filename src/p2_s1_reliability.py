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
Usage (structprep env): python src/p2_s1_reliability.py [--test]   # --test reads the held-out split
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS
from p2_s0_pilot import GENS
from p2_s2_scale import cache_dir
from p1_s6_aggregate import holm

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


def per_target(f, z=None, name=None):
    """One cached target (runs/p2_cache/<id>.npz, written by p2_s2_scale.py).

    `z` and `name` override the file, so a control can feed in the same arrays with the labels permuted
    (tests/test_p2_null.py) without touching the cache.
    """
    from p1_s3_score import threshold_free
    z = np.load(f, allow_pickle=False) if z is None else z
    name = name or f.stem
    keys, hit = [str(k) for k in z["keys"]], np.asarray(z["hit"]).astype(bool)
    truth = {k for k, h in zip(keys, hit) if h}
    names = list(z.files) if hasattr(z, "files") else list(z)
    tools = sorted(n.split("|", 1)[1] for n in names if n.startswith("apo|"))
    rows = []
    for tool in tools:
        a = z[f"apo|{tool}"].astype(float)
        base = dict(parent=name, tool=tool, n_res=len(keys),
                    apo_auprc=threshold_free(dict(zip(keys, a)), truth, keys)["auprc"],
                    own_max=float(a.max()), own_top=float(np.sort(a)[-TOPK:].mean()),
                    own_auc_topk=auc(hit[sorted(topk(a))], a[sorted(topk(a))]))
        ta, idx = topk(a), sorted(topk(a))
        for gen in GENS:
            if f"{gen}|{tool}" not in names:
                continue
            C = z[f"{gen}|{tool}"].astype(float)
            rho = np.nanmean([spearmanr(a, c).statistic for c in C])
            jac = np.mean([len(ta & topk(c)) / len(ta | topk(c)) for c in C])
            freq = np.array([np.mean([i in topk(c) for c in C]) for i in idx])
            # does stability ADD to the tool's own score? rank-sum of the two, so nothing is fitted
            combo = pd.Series(a[idx]).rank().to_numpy() + pd.Series(freq).rank().to_numpy()
            rows.append(base | dict(gen=gen, rho=rho, jaccard=jac, sd=float(C.std(0).mean()),
                                    freq_auc_topk=auc(hit[idx], freq), combo_auc_topk=auc(hit[idx], combo)))
    return rows


def main():
    split = "test" if "--test" in sys.argv else "train"
    sfx = "" if split == "train" else "_test"        # the held-out split writes beside the training results
    files = sorted(cache_dir(split).glob("*.npz"))
    if not files:
        sys.exit(f"no {split} cache; run: python src/p2_s2_scale.py run" + (" --test" if sfx else ""))
    r = pd.DataFrame([x for f in files for x in per_target(f)])
    r.to_csv(RESULTS / f"p2_reliability{sfx}.csv", index=False)

    out = []
    for (tool, gen), g in r.groupby(["tool", "gen"]):
        good = g.apo_auprc > g.apo_auprc.median()
        row = dict(tool=tool, gen=gen, n=len(g))
        for sig, sign in (("rho", 1), ("jaccard", 1), ("sd", -1), ("own_max", 1), ("own_top", 1)):
            row[f"rho_{sig}"] = spearmanr(g[sig], g.apo_auprc).statistic
            row[f"auc_{sig}"] = auc(good, sign * g[sig])
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(RESULTS / f"p2_reliability_summary{sfx}.csv", index=False)

    print(f"{r.parent.nunique()} {split} targets. TARGET level: does the signal say which predictions are good?")
    print("  rho_* = Spearman with apo AUPRC; auc_* = AUC for 'apo AUPRC above this tool's median'.")
    print("  rho/jaccard/sd need an ensemble; own_max/own_top are free (no ensemble).")
    print(s.round(3).to_string(index=False))

    tests = []
    for (tool, gen), g in r.groupby(["tool", "gen"]):
        for col in ("freq_auc_topk", "combo_auc_topk"):
            d = (g[col] - g.own_auc_topk).dropna()
            tests.append(dict(tool=tool, gen=gen, signal=col.split("_")[0], n=len(d),
                              own=g.own_auc_topk.mean(), mean=g[col].mean(), diff=d.mean(),
                              wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                              p=wilcoxon(d).pvalue if len(d) and (d != 0).any() else np.nan))
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t.p.fillna(1).to_numpy())        # one family: every tool x generator x signal
    t.to_csv(RESULTS / f"p2_reliability_tests{sfx}.csv", index=False)
    print("\nRESIDUE level, among each apo's top-10 residues: AUC of true pocket residues over false ones.")
    print("  freq = how often the residue survives perturbation; combo = freq rank-summed with the tool's own")
    print("  score; own = the tool's apo score alone. p_holm corrects over all rows below.")
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
