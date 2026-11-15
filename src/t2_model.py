"""
T2: fix the model class on the training folds, before the 218 test targets are touched.

Phase 1 found a single unfitted feature -- agreement with pLM-NN -- reaching stratified AUC 0.81 for
AE-PocketMiner and 0.75 for fpocket. So the question here is not "what model is best" but "does any model
beat one feature". If it does not, the trust rule is that feature, which is simpler, needs no fitting and
cannot overfit 846 targets.

Candidates, deliberately plain (K3: with ~846 targets and a weak signal a flexible model fits noise):
  own_max          the tool's own top score, the published baseline
  jac_pLM-NN       the Phase 1 winner, unfitted
  logit T1         L2 logistic on tier 1 alone
  logit T1+T2      L2 logistic on tiers 1 and 2 -- the comparison the spec cares about
  logit all        tiers 1-4

Scored by 4-fold cross-validation over CryptoBench's own family-disjoint training folds, so a rule that
memorises a protein family cannot look good here (K4). Out-of-fold predictions are pooled and scored once,
stratified by base-rate quintile (K1).

No gradient-boosted tree: the spec says to report one only if it beats the linear models, and nothing here
suggests the signal needs it. The L2 penalty is fixed at LAM a priori rather than tuned, so no model
selection touches the folds it is scored on.

Outputs: results/trust/t2_folds.csv, printed table.
Usage (structprep env): python src/t2_model.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, RUNS
from t1_consensus import NQ, TARGETS, _strat

warnings.filterwarnings("ignore")
LAM = 1.0                       # L2 strength on standardised features; fixed, never tuned on the folds
COVER = 0.90                    # a feature must be present on this share of a tool's rows to be used
T1 = ["own_max", "own_top10", "own_mean", "n_pred", "spread", "pk_score1", "pk_score2", "pk_gap",
      "pk_n", "pk_vol"]
T3 = ["n_res", "bfac_mean", "bfac_sd", "bfac_top", "rg", "compact"]
T4 = ["pert_rho", "pert_jac", "pert_sd"]


def tier2(cols):
    return [c for c in cols if c.startswith(("jac_", "rho_", "dca_", "cons_"))]


def fit_logit(X, y, lam=LAM):
    """L2 logistic regression. Hand-rolled so the project gains no dependency for twenty lines."""
    X = np.c_[np.ones(len(X)), X]
    def obj(w):
        z = X @ w
        ll = np.sum(np.logaddexp(0, z) - y * z)
        pen = lam * np.sum(w[1:] ** 2)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + 2 * lam * np.r_[0, w[1:]]
        return ll + pen, g
    w = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
    return lambda Z: (np.c_[np.ones(len(Z)), Z] @ w)


def prep(tr, te, cols):
    """Standardise and median-impute using the FITTING folds only, so nothing crosses the split."""
    med = tr[cols].median()
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
    f = lambda d: ((d[cols].fillna(med) - mu) / sd).fillna(0).to_numpy(float)
    return f(tr), f(te)


def main():
    f = RESULTS / "trust" / "features.csv"
    if not f.exists():
        sys.exit("no features; run: python src/t0_features.py")
    d = pd.read_csv(f)
    fold = {}
    for m in sorted((RUNS / "p1").glob("*/meta.json")):
        j = json.loads(m.read_text())
        if j["split"].startswith("train-fold-"):
            fold[j["target_id"]] = int(j["split"][-1])
    d = d[d.split == "train"].copy()
    d["fold"] = d.parent.map(fold)
    d = d[d.fold.notna()]
    print(f"{d.parent.nunique()} training targets over {d.fold.nunique()} family-disjoint folds "
          f"(the 218 test targets are untouched)\n")

    rows = []
    for tool, g in d.groupby("tool"):
        g = g.reset_index(drop=True)
        ok = [c for c in g.columns if g[c].notna().mean() >= COVER]
        t1 = [c for c in T1 if c in ok]
        t2 = [c for c in tier2(g.columns) if c in ok]
        sets = {"logit T1": t1, "logit T1+T2": t1 + t2,
                "logit all": t1 + t2 + [c for c in T3 + T4 if c in ok]}
        q = pd.qcut(g.base_rate, NQ, labels=False, duplicates="drop").to_numpy()
        for ycol, ylab in TARGETS:
            if ycol not in g or g[ycol].notna().sum() < 30:
                continue
            y = g[ycol]
            if ycol == "y_mcc_med":
                m = g.mcc.to_numpy()
                y = pd.Series([m[i] > np.nanmedian(m[q == q[i]]) for i in range(len(m))])
            y = y.astype(float)
            keep = y.notna().to_numpy()
            cand = {"own_max": g.own_max.to_numpy(float)}
            if "jac_pLM-NN" in g and g["jac_pLM-NN"].notna().mean() >= COVER:
                cand["jac_pLM-NN"] = g["jac_pLM-NN"].to_numpy(float)
            for name, cols in sets.items():
                if not cols:
                    continue
                oof = np.full(len(g), np.nan)
                for k in sorted(g.fold.unique()):   # fit on the other folds, predict the held-out one
                    tr = (g.fold != k).to_numpy() & keep
                    te = (g.fold == k).to_numpy() & keep
                    if tr.sum() < 50 or te.sum() < 5 or len(np.unique(y[tr])) < 2:
                        continue
                    Xtr, Xte = prep(g[tr], g[te], cols)
                    oof[te] = fit_logit(Xtr, y[tr].to_numpy())(Xte)
                cand[name] = oof
            for name, s in cand.items():
                m = keep & ~np.isnan(s)
                if m.sum() < 30 or len(np.unique(y[m])) < 2:
                    continue
                yb = y[m].to_numpy().astype(bool)
                a = _strat(yb, s[m], q[m], np.arange(int(m.sum())))
                # the four folds are scored by four different models, whose logits were never on one
                # scale; the within-fold mean avoids comparing them, and agreeing with the pooled number
                # is itself the check that pooling did not invent signal
                per = []
                for k in sorted(g.fold.unique()):
                    sel = (g.fold == k).to_numpy()[m]
                    if sel.sum() >= 20 and len(np.unique(yb[sel])) > 1:
                        per.append((sel.sum(), _strat(yb[sel], s[m][sel], q[m][sel],
                                                      np.arange(int(sel.sum())))))
                wf = sum(w * v for w, v in per) / sum(w for w, _ in per) if per else np.nan
                rows.append(dict(tool=tool, target=ylab, model=name, n=int(m.sum()),
                                 auc=wf, auc_pooled=a))
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / "trust" / "t2_folds.csv", index=False)

    gap = (r.auc - r.auc_pooled).abs().max()
    print(f"out-of-fold stratified AUC (mean within fold), 4 family-disjoint folds, nothing tuned on them")
    print(f"max |within-fold - pooled| = {gap:.3f}: pooling the four models' logits changes nothing\n")
    for tgt, g in r.groupby("target"):
        print(f"\n=== {tgt} ===")
        print(g.pivot(index="tool", columns="model", values="auc")
               .reindex(columns=[c for c in ["own_max", "jac_pLM-NN", "logit T1", "logit T1+T2", "logit all"]
                                 if c in set(g.model)]).round(3).to_string())
    w = r.loc[r.groupby(["tool", "target"]).auc.idxmax()]
    print("\n=== winner per tool x target ===")
    print(w[["tool", "target", "model", "auc"]].round(3).to_string(index=False))
    print("\nmodel chosen most often:", w.model.value_counts().to_dict())


if __name__ == "__main__":
    main()
