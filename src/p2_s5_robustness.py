"""
P2 s5: the three questions a reviewer asks of the Phase 2 result, all answered from the existing cache.

a) HOW MANY CONFORMERS does the residue-level signal need? Everything so far used 10. If the gain is already
   there at 2 or 3, the method costs almost nothing; if it is still climbing at 10, the reported number
   understates it and the honest claim is a lower bound.

b) IS TOP-10 THE REASON? Every residue-level number is computed among each tool's top-10 apo residues. A
   result that exists only at that cutoff is a result about the cutoff, so the same test runs at 5 and 20.

c) DID WE ACTUALLY TRY TO FIND A TARGET-LEVEL SIGNAL? p2_s1 tested five candidates, three of them our own
   ensembles, and found nothing (AUC 0.42-0.59). The obvious objection is that we did not look hard enough.
   So the same test runs over signals a practitioner has for free and would reach for first:
     n_res          protein size
     bfac_mean      mean C-alpha B-factor: crystallographic disorder, the standard cheap flexibility proxy
     bfac_top       mean B-factor of the tool's own top-10 residues
     n_pred         how many residues the tool marks (its own spread)
   and, as a reference, one signal that is NOT available at prediction time:
     base_rate      the true share of pocket residues
   The reference matters. If nothing predicts apo AUPRC, that could mean the outcome is simply noise. If the
   base rate predicts it well while nothing observable does, the outcome is predictable in principle and the
   null is a statement about what a practitioner can see, which is the claim we want to make.

Outputs: results/p2_robustness_conformers.csv, _topk.csv, _baselines.csv, printed tables.
Usage (structprep env, ~6 min): python src/p2_s5_robustness.py [--test]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, RUNS
from p2_s1_reliability import auc, topk
from p2_s2_scale import cache_dir
from p2_s4_figures import TOOLS

warnings.filterwarnings("ignore")

GEN = "random"                      # the cheapest generator, and never worse than the normal modes
N_CONF, TOPKS = 10, (5, 10, 20)
BASELINES = [("n_res", -1, "protein size (smaller is better?)"),
             ("bfac_mean", -1, "mean B-factor (less disorder)"),
             ("bfac_top", -1, "B-factor of the top-10"),
             ("n_pred", -1, "residues the tool marks"),
             ("base_rate", 1, "true pocket share  [ORACLE]")]


def betas(tid, keys):
    """Per-residue C-alpha B-factor for a prepared target, aligned to the cache's key order."""
    from prody import parsePDB
    a = parsePDB(str(RUNS / "p1" / tid / f"{tid}.pdb"), subset="ca")
    k = [f"{c}_{n}{i.strip()}" for c, n, i in zip(a.getChids(), a.getResnums(), a.getIcodes())]
    b = dict(zip(k, a.getBetas()))
    return np.array([b.get(x, np.nan) for x in keys], float)


def rows_for(f, want_b):
    """One pass over a cached target: the conformer sweep, the top-k sweep and the baseline table."""
    from p1_s3_score import threshold_free
    z = np.load(f, allow_pickle=False)
    keys, hit = [str(k) for k in z["keys"]], z["hit"].astype(bool)
    truth = {k for k, h in zip(keys, hit) if h}
    bf = betas(f.stem, keys) if want_b else None
    conf, tops, base = [], [], []
    for tool in TOOLS:
        if f"apo|{tool}" not in z.files or f"{GEN}|{tool}" not in z.files:
            continue
        a = z[f"apo|{tool}"].astype(float)
        C = z[f"{GEN}|{tool}"].astype(float)
        idx10 = sorted(topk(a, 10))
        own10 = auc(hit[idx10], a[idx10])
        for m in range(1, N_CONF + 1):                      # (a) conformers
            fr = np.array([np.mean([i in topk(c) for c in C[:m]]) for i in idx10])
            conf.append(dict(parent=f.stem, tool=tool, m=m, own=own10, freq=auc(hit[idx10], fr)))
        for k in TOPKS:                                     # (b) shortlist size
            idx = sorted(topk(a, k))
            fr = np.array([np.mean([i in topk(c, k) for c in C]) for i in idx])
            tops.append(dict(parent=f.stem, tool=tool, k=k, own=auc(hit[idx], a[idx]),
                             freq=auc(hit[idx], fr)))
        if want_b:                                          # (c) free target-level baselines
            base.append(dict(parent=f.stem, tool=tool, n_res=len(keys),
                             apo_auprc=threshold_free(dict(zip(keys, a)), truth, keys)["auprc"],
                             bfac_mean=np.nanmean(bf), bfac_top=np.nanmean(bf[idx10]),
                             n_pred=float((a > 0).sum()), base_rate=hit.mean()))
    return conf, tops, base


def show(r, by, label):
    """Mean paired gain of stability over the tool's own score, with a bootstrap CI over targets."""
    from figstyle import boot_ci
    out = []
    for (tool, v), g in r.groupby(["tool", by]):
        d = (g.freq - g.own).dropna().to_numpy()
        lo, hi = boot_ci(d)
        out.append(dict(tool=tool, **{by: v}, n=len(d), own=g.own.mean(), gain=d.mean(), lo=lo, hi=hi))
    t = pd.DataFrame(out)
    print(f"\n{label}")
    print(t.round(4).to_string(index=False))
    return t


def main():
    split = "test" if "--test" in sys.argv else "train"
    sfx = "_test" if split == "test" else ""
    files = sorted(cache_dir(split).glob("*.npz"))
    if not files:
        sys.exit(f"no {split} cache; run: python src/p2_s2_scale.py run" + (" --test" if sfx else ""))
    print(f"{len(files)} {split} targets, generator {GEN}")

    conf, tops, base = [], [], []
    for f in files:
        c, t, b = rows_for(f, want_b=True)
        conf += c; tops += t; base += b
    conf, tops, base = pd.DataFrame(conf), pd.DataFrame(tops), pd.DataFrame(base)

    a = show(conf, "m", "(a) conformers used: gain in top-10 AUC over the tool's own score")
    b = show(tops, "k", "(b) shortlist size: gain in top-k AUC over the tool's own score")
    a.to_csv(RESULTS / f"p2_robustness_conformers{sfx}.csv", index=False)
    b.to_csv(RESULTS / f"p2_robustness_topk{sfx}.csv", index=False)

    base.to_csv(RESULTS / f"p2_robustness_targets{sfx}.csv", index=False)
    out = []
    for tool, g in base.groupby("tool"):
        good = (g.apo_auprc > g.apo_auprc.median()).to_numpy()
        # AUPRC rises with the base rate by construction, and the base rate falls with protein size, so any
        # size-like signal "predicts quality" without saying anything about whether the site was found.
        # The stratified column removes that: targets are split into base-rate quintiles and the AUC is
        # computed inside each, then averaged by stratum size.
        q = pd.qcut(g.base_rate, 5, labels=False, duplicates="drop").to_numpy()
        ap = g.apo_auprc.to_numpy()
        # "well predicted" is redefined inside each stratum: against the global median the lowest base-rate
        # quintile would be almost all bad and the highest almost all good, and the AUC would be meaningless
        gq = np.concatenate([(ap[q == j] > np.median(ap[q == j])) for j in np.unique(q)])
        ordq = np.concatenate([np.where(q == j)[0] for j in np.unique(q)])
        good_s = np.empty(len(ap), bool); good_s[ordq] = gq
        for sig, sign, _ in BASELINES:
            s = sign * g[sig].to_numpy()
            parts = [(np.sum(q == j), auc(good_s[q == j], s[q == j])) for j in np.unique(q)]
            parts = [(w, v) for w, v in parts if not np.isnan(v)]
            out.append(dict(tool=tool, signal=sig, n=len(g), auc=auc(good, s),
                            auc_strat=sum(w * v for w, v in parts) / sum(w for w, _ in parts)))
    t = pd.DataFrame(out)
    t.to_csv(RESULTS / f"p2_robustness_baselines{sfx}.csv", index=False)
    print("\n(c) target level, free signals a practitioner already has: AUC for 'this target is well predicted'")
    print("    raw = as measured; strat = within base-rate quintiles, which removes the trivial route by")
    print("    which anything correlated with protein size predicts AUPRC. base_rate is an ORACLE row:")
    print("    it needs the answer, and it is here to show the outcome IS predictable in principle.")
    for col, lab in (("auc", "raw"), ("auc_strat", "stratified")):
        print(f"\n  {lab}")
        print(t.pivot(index="signal", columns="tool", values=col)
               .loc[[s[0] for s in BASELINES]].round(3).to_string())

    # (d) the same correction applied to p2_s1's own five signals, which is where the null claim lives
    rel = RESULTS / f"p2_reliability{sfx}.csv"
    if rel.exists():
        r = pd.read_csv(rel)
        r = r[r.gen == GEN].merge(base[["parent", "tool", "base_rate"]], on=["parent", "tool"], how="inner")
        out = []
        for tool, g in r.groupby("tool"):
            good = (g.apo_auprc > g.apo_auprc.median()).to_numpy()
            q = pd.qcut(g.base_rate, 5, labels=False, duplicates="drop").to_numpy()
            ap = g.apo_auprc.to_numpy()
            gq = np.concatenate([(ap[q == j] > np.median(ap[q == j])) for j in np.unique(q)])
            ordq = np.concatenate([np.where(q == j)[0] for j in np.unique(q)])
            good_s = np.empty(len(ap), bool); good_s[ordq] = gq
            for sig, sign in (("own_max", 1), ("own_top", 1), ("rho", 1), ("jaccard", 1), ("sd", -1)):
                s = sign * g[sig].to_numpy()
                parts = [(np.sum(q == j), auc(good_s[q == j], s[q == j])) for j in np.unique(q)]
                parts = [(w, v) for w, v in parts if not np.isnan(v)]
                out.append(dict(tool=tool, signal=sig, auc=auc(good, s),
                                auc_strat=sum(w * v for w, v in parts) / sum(w for w, _ in parts)))
        d = pd.DataFrame(out)
        d.to_csv(RESULTS / f"p2_robustness_strat{sfx}.csv", index=False)
        print("\n(d) p2_s1's five signals, stratified by base-rate quintile")
        print(d.pivot(index="signal", columns="tool", values="auc_strat").round(3).to_string())


if __name__ == "__main__":
    main()
