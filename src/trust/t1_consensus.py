"""
T1: does cross-tool agreement say whether a prediction is any good?

The parent paper found nothing at target level: no perturbation signal, and the tool's own confidence only
weakly (own_max, stratified AUC 0.636 / 0.609 / 0.549). Perturbing one tool asks the same model the same
question twice. Four tools disagreeing is different evidence, and fpocket, P2Rank, AE-PocketMiner and pLM-NN
are now scored on every target without ever having been compared.

Evaluated on the 846 TRAINING targets only; the 218 test targets are not touched until Phase 2.

Everything is measured within base-rate quintiles. Per-target AUPRC has a floor at the base rate, which falls
with protein size, so an unstratified number rewards any feature that tracks size -- the confound the parent
paper is about, reappearing one level up. Raw is printed beside it; stratified is the honest one.

Pitfall K5: if every tool fires on easy targets, consensus may only be re-measuring difficulty. So each
feature is reported with its rank correlation to own_max and with its *partial* correlation to the target
controlling for own_max. A feature at AUC 0.65 that correlates 0.9 with own_max has added nothing.

Outputs: results/trust/t1_consensus.csv, printed tables.
Usage (structprep env): python src/t1_consensus.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import srcpath  # noqa: F401  (puts the sibling folders on the path)
from paths import RESULTS
from p1_s6_aggregate import holm
from p2_s1_reliability import auc

warnings.filterwarnings("ignore")
TOOLS = ["fpocket", "P2Rank", "AE-PocketMiner", "pLM-NN"]
STRUCT = ["fpocket", "P2Rank", "AE-PocketMiner"]        # pLM-NN is the only sequence-only tool
TARGETS = [("y_top10hit", "top-10 holds a true residue"), ("y_mcc30", "MCC > 0.3"),
           ("y_mcc_med", "MCC above the tool's median")]
NQ, N_BOOT, SEED = 5, 2000, 0


def _strat(y, s, q, idx):
    parts = [(np.sum(q[idx] == j), auc(y[idx][q[idx] == j], s[idx][q[idx] == j])) for j in np.unique(q)]
    parts = [(w, v) for w, v in parts if w and not np.isnan(v)]
    return sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else np.nan


def strat_auc(y, s, q, n=N_BOOT, seed=SEED):
    """AUC inside each base-rate quintile, averaged by stratum size, with a bootstrap CI over targets."""
    keep = ~(np.isnan(s) | pd.isna(y))
    y, s, q = np.asarray(y, bool)[keep], np.asarray(s, float)[keep], np.asarray(q)[keep]
    if len(y) < 30 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, len(y)
    rng = np.random.default_rng(seed)
    b = [_strat(y, s, q, i) for i in rng.integers(0, len(y), (n, len(y)))]
    return _strat(y, s, q, np.arange(len(y))), *np.nanpercentile(b, [2.5, 97.5]), len(y)


def auc_vs_ref(y, scores, ref, q, n=N_BOOT, seed=SEED):
    """Each feature against the reference on the rows where BOTH are defined, resampled together.

    Pairing per feature rather than over one global intersection: dca_* is undefined for probability tools
    and pk_* where a tool found fewer than two pockets, and demanding every column at once silently drops
    whole tools from the comparison.

    Two p-values per feature, both two-sided from the bootstrap: against chance, and against the reference
    (the tool's own score). The second is the one that matters -- a feature can beat chance and still add
    nothing to what the practitioner already has.
    """
    y0, q0, r0 = np.asarray(y), np.asarray(q), np.asarray(scores[ref], float)
    p2 = lambda b, h0: float(min(1.0, 2 * min(np.nanmean(b <= h0), np.nanmean(b >= h0))))
    out = {}
    for k, v in scores.items():
        v = np.asarray(v, float)
        ok = ~(pd.isna(y0) | np.isnan(v) | np.isnan(r0))
        yy, qq, vv, rr = np.asarray(y0, bool)[ok], q0[ok], v[ok], r0[ok]
        if len(yy) < 30 or len(np.unique(yy)) < 2:
            continue
        rng = np.random.default_rng(seed)           # same seed -> same resamples, so the pairing holds
        I = rng.integers(0, len(yy), (n, len(yy)))
        bf = np.array([_strat(yy, vv, qq, i) for i in I])
        br = bf if k == ref else np.array([_strat(yy, rr, qq, i) for i in I])
        d = bf - br
        out[k] = dict(auc=_strat(yy, vv, qq, np.arange(len(yy))), n=int(len(yy)),
                      lo=float(np.nanpercentile(bf, 2.5)), hi=float(np.nanpercentile(bf, 97.5)),
                      p_chance=p2(bf, 0.5), gain=float(np.nanmean(d)),
                      gain_lo=float(np.nanpercentile(d, 2.5)), gain_hi=float(np.nanpercentile(d, 97.5)),
                      p_vs_ref=np.nan if k == ref else p2(d, 0.0))
    return out


def partial(y, s, base):
    """Spearman of feature with target, with own_max regressed out of both (rank residuals)."""
    keep = ~(np.isnan(s) | np.isnan(base) | pd.isna(y))
    if keep.sum() < 30:
        return np.nan
    r = lambda v: rankdata(np.asarray(v, float)[keep])
    rb = r(base)
    res = lambda v: r(v) - np.polyval(np.polyfit(rb, r(v), 1), rb)
    return float(spearmanr(res(s), res(np.asarray(y, float))).statistic)


def features_for(tool, g):
    """Tier-2 features available for this tool, with the sign that makes 'higher = more trustworthy'."""
    cols = [c for c in g.columns if g[c].notna().any()]
    out = []
    for o in TOOLS:
        if o == tool:
            continue
        for p, sign, lab in (("jac", 1, "Jaccard"), ("rho", 1, "Spearman"), ("dca", -1, "centre distance")):
            c = f"{p}_{o}"
            if c in cols:
                out.append((c, sign, f"{lab} vs {o}"))
    for c, lab in (("cons_k2", "consensus core, >=2 tools"), ("cons_k3", "consensus core, >=3 tools"),
                   ("cons_frac", "mean Jaccard with the other tools")):
        if c in cols:
            out.append((c, 1, lab))
    return out


def main():
    f = RESULTS / "trust" / "features.csv"
    if not f.exists():
        sys.exit("no features; run: python src/t0_features.py")
    d = pd.read_csv(f)
    d = d[d.split == "train"]
    print(f"{d.parent.nunique()} training targets x {d.tool.nunique()} tools "
          f"(the 218 test targets are untouched)\n")

    rows = []
    for tool, g in d.groupby("tool"):
        g = g.reset_index(drop=True)
        q = pd.qcut(g.base_rate, NQ, labels=False, duplicates="drop").to_numpy()
        for ycol, ylab in TARGETS:
            if ycol not in g or g[ycol].notna().sum() < 30:
                continue
            y = g[ycol]
            if ycol == "y_mcc_med":                 # the median must be taken inside the stratum
                m = g.mcc.to_numpy()
                y = pd.Series([m[i] > np.nanmedian(m[q == q[i]]) for i in range(len(m))])
            feats = [("own_max", 1, "the tool's own top score"),
                     ("own_top10", 1, "mean of its top 10")] + features_for(tool, g)
            base = g["own_max"].to_numpy(float)
            lab = {c: l for c, _, l in feats}
            sign = {c: sg for c, sg, _ in feats}
            res = auc_vs_ref(y, {c: sign[c] * g[c].to_numpy(float) for c, _, _ in feats}, "own_max", q)
            flat = np.zeros(len(g), int)            # the same AUC with no stratification, for comparison
            for c, st in res.items():
                raw, *_ = strat_auc(y, sign[c] * g[c].to_numpy(float), flat)
                rows.append(dict(tool=tool, target=ylab, feature=lab[c], col=c, **st, auc_raw=raw,
                                 rho_own=float(spearmanr(g[c], base, nan_policy="omit").statistic),
                                 partial=partial(y, g[c].to_numpy(float), base)))
    r = pd.DataFrame(rows)
    # Holm across every feature-vs-own_max comparison in the study: one family, as the spec asks.
    fam = r.col != "own_max"
    r.loc[fam, "p_holm"] = holm(r.loc[fam, "p_vs_ref"].fillna(1).to_numpy())
    (RESULTS / "trust").mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / "trust" / "t1_consensus.csv", index=False)
    print(f"Holm applied over {int(fam.sum())} feature-vs-own_max comparisons\n")

    for ycol, ylab in TARGETS:
        g = r[r.target == ylab]
        if g.empty:
            continue
        print(f"=== target: {ylab} ===")
        print("  stratified AUC [95% CI] | raw (unstratified) | gain over own_max, Holm p | r with own_max | "
              "partial corr.")
        for tool, h in g.groupby("tool"):
            print(f"\n  {tool}")
            for _, x in h.sort_values("auc", ascending=False).iterrows():
                star = "  <-- independent evidence" if x.col.endswith("pLM-NN") and tool in STRUCT else ""
                gain = "      reference      " if x.col == "own_max" else \
                    f"{x.gain:+.3f} p{'<0.001' if x.p_holm < 1e-3 else f'={x.p_holm:.3f}'}"
                print(f"    {x.feature:<34s} {x.auc:.3f} [{x.lo:.3f},{x.hi:.3f}]  raw {x.auc_raw:.3f}  "
                      f"{gain}  r{x.rho_own:+.2f}  pc{x.partial:+.2f}{star}")
        print()

    print("=== verdict: best Tier-2 feature against own_max, per tool and target ===")
    v = []
    for (tool, tgt), g in r.groupby(["tool", "target"]):
        own = g[g.col == "own_max"].iloc[0]
        t2 = g[~g.col.isin(["own_max", "own_top10"])]
        if t2.empty:
            continue
        b = t2.loc[t2.auc.idxmax()]
        v.append(dict(tool=tool, target=tgt, own_max=own.auc, best=b.feature, best_auc=b.auc,
                      gain=b.gain, gain_ci=f"[{b.gain_lo:+.3f},{b.gain_hi:+.3f}]", p_holm=b.p_holm,
                      r_own=b.rho_own, partial=b.partial))
    v = pd.DataFrame(v)
    print(v.round(4).to_string(index=False))
    print(f"\n  survives Holm at 0.05: {int((v.p_holm < 0.05).sum())}/{len(v)} tool x target combinations")

    # The obvious alternative explanation: "agreeing with pLM-NN" may just mean "pLM-NN is confident here,
    # so this target is easy for everyone". Control by using pLM-NN's own score on the same target as the
    # trust signal for the other tool. If difficulty were doing the work, the two columns would match.
    plm = d[d.tool == "pLM-NN"].set_index("parent")[["own_max"]].rename(columns={"own_max": "plm_conf"})
    ctl = []
    for tool in STRUCT:
        g = d[d.tool == tool].join(plm, on="parent").reset_index(drop=True)
        q = pd.qcut(g.base_rate, NQ, labels=False, duplicates="drop").to_numpy()
        for ycol, ylab in TARGETS:
            if ycol not in g or g[ycol].notna().sum() < 30:
                continue
            ctl.append(dict(tool=tool, target=ylab,
                            own_max=strat_auc(g[ycol], g.own_max.to_numpy(float), q)[0],
                            plm_confidence=strat_auc(g[ycol], g.plm_conf.to_numpy(float), q)[0],
                            jaccard_vs_plm=strat_auc(g[ycol], g["jac_pLM-NN"].to_numpy(float), q)[0]))
    c = pd.DataFrame(ctl)
    c["agreement_over_difficulty"] = c.jaccard_vs_plm - c.plm_confidence
    c.to_csv(RESULTS / "trust" / "t1_difficulty_control.csv", index=False)
    print("\n=== control: is consensus only measuring that pLM-NN finds the target easy? ===")
    print("  if so, pLM-NN's own confidence would score as well as agreeing with it.")
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
