"""
P1 s3: score one tool on one structure under docs/evaluation_protocol.md (§3 site gate, §5 metrics).

Inputs are the adapters' common forms (p1_adapters.py): pockets, or per-residue scores.
Ground truth is a set of residue keys; it is restricted to residues present in the structure.
"""
import numpy as np
from scipy.stats import rankdata

# A pocket is "at the site" if its centre is within DCA_SUCCESS of any ligand heavy atom: the standard
# detection rule doubles as the gate. The KRAS gate (14 Å from His95 Cα) does not generalize: on 7RPZ a
# 14 Å gate around the true-residue centroid admits nucleotide-side pockets (fpocket 5, 7).
DCA_SUCCESS = 4.0
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


def score_pockets(pockets, truth, keys, ca, lig_atoms):
    """Pocket tools: union and best-single localization at the site, plus site-agnostic detection."""
    universe = set(keys)
    truth = truth & universe
    lig_center = lig_atoms.mean(axis=0)
    near_lig = lambda p: np.linalg.norm(lig_atoms - p["center"], axis=1).min() <= DCA_SUCCESS
    at = [p for p in pockets if near_lig(p)]
    union = set().union(*(p["residues"] for p in at))
    m = confusion(union, truth, universe)
    m["best_single_recall"] = max((confusion(p["residues"], truth, universe)["recall"] for p in at), default=0.0)
    m["fragments"] = len(at)
    m["dca"] = min((np.linalg.norm(p["center"] - lig_center) for p in at), default=np.nan)
    ranked = sorted(pockets, key=lambda p: p["rank"])
    hit = [near_lig(p) for p in ranked]
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
