"""
P1 s3: score every tool on every usable target under docs/evaluation_protocol.md (§3 site gate, §5 metrics).

Library: score_pockets / score_probs / threshold_free take the adapters' common forms (p1_adapters.py).
Driver (main): reads runs/p1/<target>/meta.json (prep) and the raw tool outputs (p1_s2_run_baselines.py), and writes
  results/p1_master.csv        one row per target x tool x variant (primary variants: gate4, t0.7)
  results/p1_pm_sweep.csv      AE-PocketMiner MCC per target per threshold (t* is picked on training folds, not here)
  results/p1_excluded.csv      targets prep marked unusable, with the reason
Ground truth is restricted to residues present in the structure.

Usage (structprep env): python src/p1_s3_score.py [--split test]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A pocket is "at the site" if its centre is within DCA_SUCCESS of any ligand heavy atom: the standard
# detection rule doubles as the gate. The KRAS gate (14 Å from His95 Cα) does not generalize: on 7RPZ a
# 14 Å gate around the true-residue centroid admits nucleotide-side pockets (fpocket 5, 7).
DCA_SUCCESS = 4.0
GATE_SWEEP = (4.0, 6.0, 8.0, 10.0, 12.0)   # sensitivity: published "found" criteria range over 4-12 Å
TOP_K = 3                 # rank-limited variant of pocket tools
SITE_RADIUS = 14.0        # only for the gated variant of probability tools (Cα within this of the site centre)
FIXED_T = 0.7             # KRAS threshold for probability tools
SWEEP = np.round(np.arange(0.05, 0.951, 0.05), 2)


def confusion(pred, truth, universe):
    pred, truth = pred & universe, truth & universe
    tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
    tn = len(universe) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return dict(precision=prec, recall=rec, f1=f1, mcc=mcc, n_pred=len(pred), n_true=len(truth),
                spread=len(pred) / len(truth) if truth else np.nan)


def site_center(truth, keys, ca):
    idx = [i for i, k in enumerate(keys) if k in truth]
    if not idx:
        raise ValueError("no ground-truth residue is present in the structure")
    return ca[idx].mean(axis=0)


def score_pockets(pockets, truth, keys, ca, lig_atoms, gate=DCA_SUCCESS):
    """Pocket tools: union and best-single localization at the site, plus site-agnostic detection.
    `gate` sets which pockets count as at the site; detection (top1/top3) always uses DCA_SUCCESS."""
    universe = set(keys)
    truth = truth & universe
    lig_center = lig_atoms.mean(axis=0)
    dist = lambda p: np.linalg.norm(lig_atoms - p["center"], axis=1).min()
    at = [p for p in pockets if dist(p) <= gate]
    union = set().union(*(p["residues"] for p in at))
    m = confusion(union, truth, universe)
    m["best_single_recall"] = max((confusion(p["residues"], truth, universe)["recall"] for p in at), default=0.0)
    m["fragments"] = len(at)
    m["dca"] = min((np.linalg.norm(p["center"] - lig_center) for p in at), default=np.nan)
    ranked = sorted(pockets, key=lambda p: p["rank"])
    hit = [dist(p) <= DCA_SUCCESS for p in ranked]
    m["top1"], m["top3"] = any(hit[:1]), any(hit[:3])
    m["best_rank"] = next((i + 1 for i, h in enumerate(hit) if h), np.nan)   # position in the ranked list
    m["site_hit"] = bool(at)
    return m


def score_probs(scores, truth, keys, ca, t=FIXED_T, gated=False):
    """Probability tools: residues >= t, whole protein (primary) or within SITE_RADIUS of the site (gated)."""
    universe = set(keys)
    truth = truth & universe
    allowed = universe
    if gated:
        sc = site_center(truth, keys, ca)
        allowed = {k for k, c in zip(keys, ca) if np.linalg.norm(c - sc) <= SITE_RADIUS}
    pred = {k for k in allowed if scores[k] >= t}
    m = confusion(pred, truth, universe)
    m["site_hit"] = bool(pred & truth)
    return m


def threshold_free(scores, truth, keys):
    """AUPRC (average precision) and AUROC over all residues; ties share one rank."""
    s = np.array([scores[k] for k in keys])
    y = np.array([k in truth for k in keys])
    if y.all() or not y.any():
        return dict(auprc=np.nan, auroc=np.nan)
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    hits = np.cumsum(ys)
    auprc = float((hits[ys] / (np.flatnonzero(ys) + 1)).mean())
    r = rankdata(s)
    auroc = float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
    return dict(auprc=auprc, auroc=auroc)


def sweep(scores, truth, keys, ca):
    return {float(t): score_probs(scores, truth, keys, ca, t)["mcc"] for t in SWEEP}


def score_target(tdir, p2_out, pm_out):
    """All tools, all variants, for one prepared target."""
    from p1_adapters import (ca_table, fpocket, p2rank, p2rank_residue_scores, per_residue_probs,
                             pocket_residue_scores)
    meta = json.loads((tdir / "meta.json").read_text())
    tid = meta["target_id"]
    keys, ca = ca_table(tdir / f"{tid}.pdb")
    truth = set(meta["truth"]) & set(keys)
    lig = np.array(meta["ligand_xyz"])
    base = dict(target_id=tid, split=meta["split"], n_res=len(keys), n_true=len(truth),
                base_rate=len(truth) / len(keys))   # chance AUPRC; chance MCC is 0
    rows, sweep_row = [], None

    pocket_tools = []
    if (tdir / f"{tid}_out" / f"{tid}_info.txt").exists():
        fp = fpocket(tdir / f"{tid}_out")
        pocket_tools.append(("fpocket", fp, pocket_residue_scores(fp, keys)))
    if (p2_out / f"{tid}.pdb_predictions.csv").exists():
        pocket_tools.append(("P2Rank", p2rank(p2_out / f"{tid}.pdb_predictions.csv"),
                             p2rank_residue_scores(p2_out / f"{tid}.pdb_residues.csv", keys)))
    for tool, pockets, rscores in pocket_tools:
        tf = threshold_free(rscores, truth, keys)
        for g in GATE_SWEEP:
            rows.append(base | dict(tool=tool, variant=f"gate{g:g}", primary=g == DCA_SUCCESS)
                        | score_pockets(pockets, truth, keys, ca, lig, gate=g) | tf)
        # Deployable variant: only the top TOP_K ranked pockets may count. Without it a tool that outputs
        # many pockets (fpocket median 19 vs P2Rank 4) gets oracle credit for a low-ranked hit.
        top = sorted(pockets, key=lambda p: p["rank"])[:TOP_K]
        rows.append(base | dict(tool=tool, variant=f"gate{DCA_SUCCESS:g}_top{TOP_K}", primary=False)
                    | score_pockets(top, truth, keys, ca, lig) | tf | {"n_pockets": len(pockets)})

    pm_file = pm_out / f"{tid}-preds.npy"
    if pm_file.exists():
        pm = per_residue_probs(pm_file, keys)
        tf = threshold_free(pm, truth, keys)
        rows.append(base | dict(tool="AE-PocketMiner", variant=f"t{FIXED_T}", primary=True)
                    | score_probs(pm, truth, keys, ca) | tf)
        rows.append(base | dict(tool="AE-PocketMiner", variant=f"t{FIXED_T}_gated", primary=False)
                    | score_probs(pm, truth, keys, ca, gated=True) | tf)
        sweep_row = dict(target_id=tid, split=meta["split"]) | {f"mcc_t{t:.2f}": v for t, v in sweep(pm, truth, keys, ca).items()}
    return rows, sweep_row


def main():
    from paths import RESULTS, RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="*", default=["test", "anchor"], help="meta.json splits to score")
    args = ap.parse_args()
    p1 = RUNS / "p1"
    rows, sweeps, excluded = [], [], []
    for m in sorted(p1.glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta["split"] not in args.split:
            continue
        if not meta.get("usable", True):
            excluded.append(dict(target_id=meta["target_id"], truth_near_ligand=meta["truth_near_ligand"],
                                 label_map=meta.get("label_map")))
            continue
        r, s = score_target(m.parent, p1 / "p2rank_out", p1 / "pocketminer_out")
        rows += r
        if s:
            sweeps.append(s)
    master = pd.DataFrame(rows)
    master["auprc_lift"] = master.auprc / master.base_rate
    master.to_csv(RESULTS / "p1_master.csv", index=False)
    pd.DataFrame(sweeps).to_csv(RESULTS / "p1_pm_sweep.csv", index=False)
    pd.DataFrame(excluded).to_csv(RESULTS / "p1_excluded.csv", index=False)

    prim = master[master.primary]
    n = prim.target_id.nunique()
    print(f"scored {n} targets ({len(excluded)} excluded) -> {RESULTS / 'p1_master.csv'}\n")
    summary = prim.groupby("tool").agg(targets=("target_id", "nunique"), mcc_median=("mcc", "median"),
                                       mcc_mean=("mcc", "mean"), recall=("recall", "median"),
                                       precision=("precision", "median"), spread=("spread", "median"),
                                       top1=("top1", "mean"), top3=("top3", "mean"),
                                       auprc=("auprc", "median"), auprc_lift=("auprc_lift", "median"))
    print("primary variants (medians over targets; top1/top3 = share of targets):")
    print(summary.round(3).to_string())
    gates = master[master.variant.str.startswith("gate")].pivot_table(index="tool", columns="variant",
                                                                       values="mcc", aggfunc="mean")
    print(f"\ngate sensitivity and top-{TOP_K} variant, pocket tools (mean MCC):")
    print(gates[[f"gate{g:g}" for g in GATE_SWEEP] + [f"gate{DCA_SUCCESS:g}_top{TOP_K}"]].round(3).to_string())


if __name__ == "__main__":
    main()
