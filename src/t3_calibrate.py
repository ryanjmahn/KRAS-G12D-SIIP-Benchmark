"""
T3: turn the trust score into a probability a user can threshold, and draw the two figures.

The held-out result ranks targets well (stratified AUC 0.77-0.80 for fpocket and AE-PocketMiner on target
(b), "the tool's top-10 holds a true pocket residue"). A ranking is not yet usable: "only chase targets where
the tool is at least 80% likely to be right" needs a calibrated probability, and the expected calibration
error says whether that number means what it claims.

The calibrator is fitted on OUT-OF-FOLD training predictions -- the model never sees the targets its own
calibrator is fitted on, and the 218 test targets are used once, for evaluation only. Platt (a 1-D logistic
on the score) and isotonic (pool-adjacent-violators) are both fitted; the one with the lower out-of-fold ECE
is chosen before the test set is touched.

Outputs: results/trust/t3_calibration.csv, figures/trust/t3_fig1_riskcoverage.{png,pdf},
         figures/trust/t3_fig2_reliability.{png,pdf}, figures/trust/CAPTIONS.md
Usage (structprep env): python src/t3_calibrate.py [--format png pdf]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGURES, RESULTS, RUNS
from figstyle import COLOR, GREY, INK, finish, save, style
from t2_eval import GRID, POINTS, risk_coverage
from t2_model import COVER, T1, fit_logit, prep, tier2

TARGET = "y_top10hit"           # settled: the only definition independent of how many residues a tool flags
TOOLS = ["fpocket", "P2Rank", "AE-PocketMiner", "pLM-NN"]
OUT = FIGURES / "trust"
NBIN = 10


def pava(x, y):
    """Isotonic regression by pool-adjacent-violators; returns a step function of the score."""
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o].astype(float)
    v, w = list(ys), [1.0] * len(ys)
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1]:
            i += 1
            continue
        v[i] = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
        w[i] += w[i + 1]
        del v[i + 1], w[i + 1]
        i = max(i - 1, 0)
    knots, fit = [], []
    k = 0
    for val, wt in zip(v, w):
        knots.append(xs[k]); fit.append(val); k += int(wt)
    return lambda z: np.interp(z, np.array(knots), np.array(fit))


def ece(p, y, n=NBIN):
    """Expected calibration error: |confidence - accuracy| per equal-width bin, weighted by bin size."""
    b = np.clip((p * n).astype(int), 0, n - 1)
    return float(sum(np.mean(b == k) * abs(p[b == k].mean() - y[b == k].mean())
                     for k in range(n) if (b == k).any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", nargs="+", default=["png", "pdf"])
    ap.add_argument("--with-plmnn", action="store_true",
                    help="keep the agreement-with-pLM-NN features (the ablation, rule_full.json)")
    args = ap.parse_args()
    sfx = "" if not args.with_plmnn else "_full"
    d = pd.read_csv(RESULTS / "trust" / "features.csv")
    fold = {}
    for m in sorted((RUNS / "p1").glob("*/meta.json")):
        j = json.loads(m.read_text())
        if j["split"].startswith("train-fold-"):
            fold[j["target_id"]] = int(j["split"][-1])

    # Figures describe the FROZEN artifact (results/trust/rule.json), not a parallel refit: two code
    # paths computing "the calibration of the rule" would publish two different ECEs for one thing.
    from t3_rule import RULE, RULE_FULL, trust
    rule = json.loads((RULE_FULL if args.with_plmnn else RULE).read_text())
    rows, curves, rel = [], [], []
    for tool in TOOLS:
        te = d[(d.split == "test") & (d.tool == tool)].reset_index(drop=True)
        yte = te[TARGET].to_numpy(bool)
        sp = [trust(x, tool, rule) for x in te.to_dict("records")]
        s_te = np.array([x[0] for x in sp])
        p_te = np.clip(np.array([x[1] for x in sp]), 1e-6, 1 - 1e-6)
        rows.append(dict(tool=tool, n_test=len(te), chosen=rule["tools"][tool]["cal"],
                         n_features=len(rule["tools"][tool]["cols"]),
                         ece_test=ece(p_te, yte), brier_test=float(np.mean((p_te - yte) ** 2)),
                         brier_base=float(np.mean((yte.mean() - yte) ** 2)),
                         base_rate=float(yte.mean())))
        for c in GRID:
            curves.append(dict(tool=tool, coverage=c, risk=risk_coverage(yte, s_te, [c])[0],
                               risk_own=risk_coverage(yte, te.own_max.to_numpy(float), [c])[0],
                               risk_base=float(1 - yte.mean())))
        b = np.clip((p_te * NBIN).astype(int), 0, NBIN - 1)
        for k in range(NBIN):
            if (b == k).any():
                rel.append(dict(tool=tool, bin=(k + 0.5) / NBIN, conf=float(p_te[b == k].mean()),
                                acc=float(yte[b == k].mean()), n=int((b == k).sum())))
    r, cv, rl = pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(rel)
    r.to_csv(RESULTS / "trust" / f"t3_calibration{sfx}.csv", index=False)
    print(r.round(4).to_string(index=False))

    style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharex=True, sharey=True)
    for ax, tool in zip(axes, TOOLS):
        g = cv[cv.tool == tool]
        ax.axhline(g.risk_base.iloc[0], color=GREY, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.plot(g.coverage * 100, g.risk_own, lw=1.6, color=COLOR[tool], alpha=0.45, zorder=2)
        ax.plot(g.coverage * 100, g.risk, lw=2.2, color=COLOR[tool], zorder=3)
        ax.set_title(tool, fontsize=8, pad=4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        finish(ax, grid="y")
    axes[0].set_ylabel("risk among kept targets")
    axes[0].set_ylim(0, 0.75)
    h = [plt.Line2D([], [], color="#444", lw=2.2), plt.Line2D([], [], color="#444", lw=1.6, alpha=0.5),
         plt.Line2D([], [], color=GREY, lw=1, ls=(0, (4, 3)))]
    axes[0].legend(h, ["trust score", "the tool's own score", "keep everything"],
                   frameon=False, fontsize=6.5, loc="lower right", borderpad=0)
    fig.supxlabel("targets kept, ranked by trust score (%)", fontsize=8, y=0.02)
    fig.tight_layout(pad=0.4, rect=(0, 0.05, 1, 1))
    save(fig, f"t3_fig1_riskcoverage{sfx}", args.format, OUT)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharex=True, sharey=True)
    for ax, tool in zip(axes, TOOLS):
        g = rl[rl.tool == tool]
        x = r[r.tool == tool].iloc[0]
        ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.scatter(g.conf, g.acc, s=np.sqrt(g.n) * 6, color=COLOR[tool], alpha=0.85, lw=0, zorder=3)
        ax.plot(g.conf, g.acc, color=COLOR[tool], lw=1.2, alpha=0.6, zorder=2)
        ax.set_title(f"{tool}\nECE {x.ece_test:.3f}", fontsize=7.5, pad=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        finish(ax, grid="y")
    axes[0].set_ylabel("observed accuracy")
    fig.supxlabel("predicted probability that the tool is right", fontsize=8, y=0.02)
    fig.tight_layout(pad=0.4, rect=(0, 0.05, 1, 1))
    save(fig, f"t3_fig2_reliability{sfx}", args.format, OUT)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CAPTIONS.md").write_text(
        "# Trust-rule figures\n\n"
        f"**Figure T1. Withholding the targets a tool is least likely to get right (n = {r.n_test.iloc[0]} "
        "held-out targets).** Risk is the share of kept targets where the tool's top-10 contains no true "
        "pocket residue. Targets are ranked by the trust score (L2 logistic on the tool's own signal plus "
        "cross-tool agreement, fitted on 846 training targets) or by the tool's own top score, and the top "
        "fraction is kept. Dashed: keeping everything.\n\n"
        "**Figure T2. The trust score is calibrated (held-out).** Predicted probability against observed "
        "accuracy in ten equal-width bins, marker area proportional to bin size; the diagonal is perfect "
        "calibration. The calibrator (Platt or isotonic, whichever had the lower out-of-fold error) was "
        "fitted on out-of-fold training predictions and applied once to the test split.\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
