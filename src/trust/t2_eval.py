"""
T2 eval: the held-out 218. Fit on all 846 training targets, predict once, never tune.

The model class was fixed on the family-disjoint training folds by t2_model.py and is not revisited here.
Candidates for the decision-relevant target are the fitted `logit T1+T2` and the unfitted `jac_pLM-NN`,
against the tool's own score and a random ranking.

Two readouts, because they answer different questions:

  stratified AUC      is the signal real, or is it protein size? Measured inside base-rate quintiles, as the
                      parent paper's Leg 3 requires. This is the honest measure of ranking quality.
  risk-coverage       what does a practitioner get? They hold 218 targets and a budget for k of them; they
                      cannot stratify, they just rank and take the top. So this one is pooled, which is the
                      situation it describes. AURC is its area; lower is better.

Selective accuracy at coverage 0.1 / 0.25 / 0.5 is the practical readout: the share of kept targets on which
the tool was right. The coverage grid is fixed up front so every method is compared at matched coverage (K6).

Outputs: results/trust/t2_heldout.csv, results/trust/t2_riskcoverage.csv, printed tables.
Usage (structprep env): python src/t2_eval.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import srcpath  # noqa: F401  (puts the sibling folders on the path)
from paths import RESULTS
from figstyle import boot_ci
from t1_consensus import NQ, TARGETS, _strat
from t2_model import COVER, T1, fit_logit, prep, tier2

warnings.filterwarnings("ignore")
GRID = np.round(np.arange(0.05, 1.001, 0.05), 2)
POINTS = (0.1, 0.25, 0.5)
N_BOOT, SEED = 2000, 0


def risk_coverage(y, s, cov=GRID):
    """Error rate among the top `cov` fraction by trust score. Ties broken by a fixed hash-free order."""
    o = np.argsort(-s, kind="stable")
    e = (~y.astype(bool))[o].astype(float)
    n = np.maximum(np.round(np.asarray(cov) * len(e)).astype(int), 1)
    return np.cumsum(e)[n - 1] / n


def aurc(y, s):
    return float(np.mean(risk_coverage(y, s, GRID)))


def main():
    f = RESULTS / "trust" / "features.csv"
    d = pd.read_csv(f)
    tr_all, te_all = d[d.split == "train"], d[d.split == "test"]
    print(f"fit on {tr_all.parent.nunique()} training targets, evaluate once on "
          f"{te_all.parent.nunique()} held-out targets\n")

    rows, curves = [], []
    for tool in sorted(d.tool.unique()):
        tr, te = tr_all[tr_all.tool == tool].reset_index(drop=True), te_all[te_all.tool == tool].reset_index(drop=True)
        ok = [c for c in tr.columns if tr[c].notna().mean() >= COVER and te[c].notna().mean() >= COVER]
        cols = [c for c in T1 if c in ok] + [c for c in tier2(tr.columns) if c in ok]
        q = pd.qcut(te.base_rate, NQ, labels=False, duplicates="drop").to_numpy()
        for ycol, ylab in TARGETS:
            if ycol not in te or te[ycol].notna().sum() < 30:
                continue
            ytr, yte = tr[ycol], te[ycol]
            if ycol == "y_mcc_med":                 # median taken inside the stratum, as in training
                for df, yy, qq in ((tr, "ytr", None), (te, "yte", q)):
                    m = df.mcc.to_numpy()
                    qx = qq if qq is not None else pd.qcut(df.base_rate, NQ, labels=False,
                                                           duplicates="drop").to_numpy()
                    v = pd.Series([m[i] > np.nanmedian(m[qx == qx[i]]) for i in range(len(m))])
                    ytr, yte = (v, yte) if yy == "ytr" else (ytr, v)
            keep = yte.notna().to_numpy()
            cand = {"own_max": te.own_max.to_numpy(float)}
            if "jac_pLM-NN" in te and te["jac_pLM-NN"].notna().mean() >= COVER:
                cand["jac_pLM-NN"] = te["jac_pLM-NN"].to_numpy(float)
            if cols:
                mtr = ytr.notna().to_numpy()
                Xtr, Xte = prep(tr[mtr], te, cols)
                cand["logit T1+T2"] = fit_logit(Xtr, ytr[mtr].to_numpy(float))(Xte)
            # a single random ranking is one draw: with n~218 its AUC scatters by ~0.04, so one unlucky
            # permutation can post 0.63 and look like a signal. Average the baseline over many draws.
            def rand_rows(yb, qm, n=200):
                rng = np.random.default_rng(SEED)
                P = [rng.permutation(len(yb)).astype(float) for _ in range(n)]
                return dict(auc_strat=float(np.nanmean([_strat(yb, p_, qm, np.arange(len(yb))) for p_ in P])),
                            aurc=float(np.mean([aurc(yb, p_) for p_ in P])),
                            **{f"acc@{c:g}": float(np.mean([1 - risk_coverage(yb, p_, [c])[0] for p_ in P]))
                               for c in POINTS})

            for name, s in cand.items():
                m = keep & ~np.isnan(s)
                if m.sum() < 30 or len(np.unique(yte[m])) < 2:
                    continue
                yb, sb = yte[m].to_numpy().astype(bool), s[m]
                a = _strat(yb, sb, q[m], np.arange(int(m.sum())))
                rc = risk_coverage(yb, sb)
                rng = np.random.default_rng(SEED)
                b = [aurc(yb[i], sb[i]) for i in rng.integers(0, len(yb), (N_BOOT, len(yb)))]
                rows.append(dict(tool=tool, target=ylab, method=name, n=int(m.sum()),
                                 auc_strat=a, aurc=aurc(yb, sb),
                                 aurc_lo=float(np.percentile(b, 2.5)), aurc_hi=float(np.percentile(b, 97.5)),
                                 base_err=float(1 - yb.mean()),
                                 **{f"acc@{c:g}": 1 - risk_coverage(yb, sb, [c])[0] for c in POINTS}))
                curves += [dict(tool=tool, target=ylab, method=name, coverage=c, risk=v)
                           for c, v in zip(GRID, rc)]
            yb0 = yte[keep].to_numpy().astype(bool)
            if keep.sum() >= 30 and len(np.unique(yb0)) > 1:
                rows.append(dict(tool=tool, target=ylab, method="random", n=int(keep.sum()),
                                 aurc_lo=np.nan, aurc_hi=np.nan, base_err=float(1 - yb0.mean()),
                                 **rand_rows(yb0, q[keep])))
    r, cv = pd.DataFrame(rows), pd.DataFrame(curves)
    (RESULTS / "trust").mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / "trust" / "t2_heldout.csv", index=False)
    cv.to_csv(RESULTS / "trust" / "t2_riskcoverage.csv", index=False)

    order = ["random", "own_max", "jac_pLM-NN", "logit T1+T2"]
    for tgt, g in r.groupby("target"):
        print(f"=== {tgt} ===")
        print("  stratified AUC | AURC [95% CI] (lower better) | selective accuracy at 10/25/50% coverage")
        for tool, h in g.groupby("tool"):
            print(f"\n  {tool}   (keeping everything, the tool is right on "
                  f"{1 - h.base_err.iloc[0]:.0%} of targets)")
            for m in order:
                x = h[h.method == m]
                if x.empty:
                    continue
                x = x.iloc[0]
                print(f"    {m:<14s} AUC {x.auc_strat:.3f}  AURC {x.aurc:.3f} "
                      f"[{x.aurc_lo:.3f},{x.aurc_hi:.3f}]  "
                      f"acc {x['acc@0.1']:.2f} / {x['acc@0.25']:.2f} / {x['acc@0.5']:.2f}")
        print()


if __name__ == "__main__":
    main()
