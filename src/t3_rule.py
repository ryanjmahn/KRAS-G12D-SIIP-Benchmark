"""
T3: the rule itself -- fit it, freeze it, and expose it as something a user can call.

    from t3_rule import trust
    trust(row, "fpocket")  ->  (score, probability, action)

`row` is one line of results/trust/features.csv: whatever src/t0_extract.py and src/t0_features.py compute
from an apo structure and the four tools' outputs. No holo structure, no ligand, no truth labels.

The question answered is target (b): will this tool's top-10 residues contain at least one true pocket
residue? That is what a user faces before committing effort, and it is the only definition independent of how
many residues a tool chooses to flag.

Coefficients and calibrator are fitted once on the 846 training targets and written to
results/trust/rule.json, so the released artifact is a file, not a re-run. The operating point is fixed at
P_ACT and its precision is measured on the 218 held-out targets.

Usage (structprep env):
  python src/t3_rule.py fit        refit from features.csv -> results/trust/rule.json
  python src/t3_rule.py check      apply the frozen rule to the held-out split and report its operating point
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, RUNS
from t2_model import COVER, T1, fit_logit, tier2
from t3_calibrate import TARGET, TOOLS, ece, pava

warnings.filterwarnings("ignore")
RULE = RESULTS / "trust" / "rule.json"            # shipped: structure tools only
RULE_FULL = RESULTS / "trust" / "rule_full.json"  # same, plus agreement with pLM-NN (the ablation)
P_ACT = 0.70                    # "investigate": the probability above which a user should spend effort
P_SKIP = 0.40                   # below this, the prediction is more likely wrong than right


def action(p):
    return "investigate" if p >= P_ACT else ("skip" if p < P_SKIP else "inspect manually")


def trust(row, tool, rule=None):
    """(score, calibrated probability, recommended action) for one target under one tool."""
    r = rule or json.loads(RULE.read_text())
    if tool not in r["tools"]:
        raise KeyError(f"no rule for {tool}; fitted tools are {sorted(r['tools'])}")
    m = r["tools"][tool]
    x = np.array([float(row.get(c, np.nan)) for c in m["cols"]])
    x = np.where(np.isnan(x), m["median"], x)
    z = (x - np.array(m["mu"])) / np.array(m["sd"])
    s = float(m["w"][0] + np.dot(m["w"][1:], z))
    p = float(np.interp(s, m["knots"], m["fit"])) if m["cal"] == "isotonic" else \
        float(1 / (1 + np.exp(-(m["platt"][0] + m["platt"][1] * s))))
    p = min(max(p, 0.0), 1.0)
    return s, p, action(p)


def fit(use_plm=False, path=None):
    """Fit one rule per tool. `use_plm` keeps the agreement-with-pLM-NN features.

    The shipped rule drops them. Agreement among the three structure tools carries the same signal (for
    fpocket, slightly more), and a rule that needs no sequence model is one a user can actually run -- and
    it cannot be answered with "if you have pLM-NN, just use its prediction".
    """
    d = pd.read_csv(RESULTS / "trust" / "features.csv")
    fold = {}
    for m in sorted((RUNS / "p1").glob("*/meta.json")):
        j = json.loads(m.read_text())
        if j["split"].startswith("train-fold-"):
            fold[j["target_id"]] = int(j["split"][-1])
    out = {"target": TARGET, "p_act": P_ACT, "p_skip": P_SKIP, "uses_plmnn": use_plm, "tools": {}}
    for tool in TOOLS:
        tr = d[(d.split == "train") & (d.tool == tool)].reset_index(drop=True)
        tr = tr.assign(fold=tr.parent.map(fold))
        tr = tr[tr.fold.notna()].reset_index(drop=True)
        te = d[(d.split == "test") & (d.tool == tool)]
        ok = [c for c in te.columns if c in tr.columns
              and tr[c].notna().mean() >= COVER and te[c].notna().mean() >= COVER]
        t2 = [c for c in tier2(tr.columns) if c in ok and (use_plm or "pLM-NN" not in c)]
        cols = [c for c in T1 if c in ok] + t2
        y = tr[TARGET].to_numpy(float)
        med, mu = tr[cols].median(), tr[cols].mean()
        sd = tr[cols].std().replace(0, 1)
        X = ((tr[cols].fillna(med) - mu) / sd).fillna(0).to_numpy(float)
        Xa = np.c_[np.ones(len(X)), X]
        f = fit_logit(X, y)
        w = np.linalg.lstsq(Xa, f(X), rcond=None)[0]            # recover the weights the closure applied
        # the calibrator is fitted on OUT-OF-FOLD scores: fitting it on the model's own training
        # predictions makes it look sharper than it is, and the probability is the thing a user acts on
        s = np.full(len(tr), np.nan)
        for k in sorted(tr.fold.unique()):
            a, b = (tr.fold != k).to_numpy(), (tr.fold == k).to_numpy()
            s[b] = fit_logit(X[a], y[a])(X[b])
        iso = pava(s, y)
        knots = np.linspace(s.min(), s.max(), 200)
        pl = fit_logit(s[:, None], y)
        p_pl = 1 / (1 + np.exp(-pl(s[:, None])))
        p_is = np.clip(iso(s), 0, 1)
        cal = "isotonic" if np.mean((p_is - y) ** 2) <= np.mean((p_pl - y) ** 2) else "platt"
        b = np.linalg.lstsq(np.c_[np.ones(len(s)), s], pl(s[:, None]), rcond=None)[0]
        out["tools"][tool] = dict(cols=cols, median=med[cols].tolist(), mu=mu[cols].tolist(),
                                  sd=sd[cols].tolist(), w=w.tolist(), cal=cal,
                                  knots=knots.tolist(), fit=np.clip(iso(knots), 0, 1).tolist(),
                                  platt=b.tolist(), n_train=int(len(tr)))
    path = path or RULE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"fitted on {d[d.split == 'train'].parent.nunique()} training targets -> {path.name}"
          f"  (pLM-NN agreement: {'included' if use_plm else 'excluded'})")
    for t, m in out["tools"].items():
        print(f"  {t:<15s} {len(m['cols'])} features, {m['cal']} calibration")


def check(path=None):
    r = json.loads((path or RULE).read_text())
    d = pd.read_csv(RESULTS / "trust" / "features.csv")
    te = d[d.split == "test"]
    rows = []
    for tool in TOOLS:
        g = te[te.tool == tool]
        p = np.array([trust(x, tool, r)[1] for x in g.to_dict("records")])
        y = g[TARGET].to_numpy(bool)
        act = p >= P_ACT
        rows.append(dict(tool=tool, n=len(g), base_rate=y.mean(), ece=ece(p, y),
                         n_investigate=int(act.sum()),
                         precision=float(y[act].mean()) if act.any() else np.nan,
                         lift=float(y[act].mean() - y.mean()) if act.any() else np.nan,
                         n_skip=int((p < P_SKIP).sum()),
                         skip_err=float(y[p < P_SKIP].mean()) if (p < P_SKIP).any() else np.nan))
    t = pd.DataFrame(rows)
    sfx = "_full" if r.get("uses_plmnn") else ""   # the two variants must not overwrite each other
    t.to_csv(RESULTS / "trust" / f"t3_operating_point{sfx}.csv", index=False)
    print(f'{(path or RULE).name} applied to {te.parent.nunique()} held-out targets, '
          f'"investigate" at p >= {P_ACT}')
    print(t.round(3).to_string(index=False))
    print("\n  precision = share of investigated targets where the tool really was right")
    print("  lift      = that, minus what you would get investigating everything")
    print("  skip_err  = share of SKIPPED targets the tool was actually right about (the cost of abstaining)")
    # pLM-NN is excluded from the gate, not excused: its training split holds 811 positives against 35
    # negatives, so there is nothing for a rule to learn and it flags every target. That is reported as a
    # limitation wherever its row appears. A structure tool showing no lift would be a real regression.
    gated = t[t.tool != "pLM-NN"]
    bad = [x.tool for _, x in gated.iterrows() if np.isfinite(x.lift) and x.lift <= 0]
    flat = [x.tool for _, x in t.iterrows() if x.tool == "pLM-NN" and x.lift <= 0]
    if flat:
        print("\n  note: pLM-NN shows no lift (35 training negatives); excluded from the gate by design")
    print(("FAIL: no lift for " + ", ".join(bad)) if bad else "ok: every structure tool gains from the rule")
    return 1 if bad else 0


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if step == "fit":
        fit(use_plm=False, path=RULE)
        fit(use_plm=True, path=RULE_FULL)
        sys.exit(0)
    bad = check(RULE)
    print()
    check(RULE_FULL)
    sys.exit(bad)
