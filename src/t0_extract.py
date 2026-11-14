"""
T0: persist what every tool actually predicted, per target, with no ground truth involved.

Tier 2 of TRUST_RULE_EXPERIMENT.md needs the predicted residue sets and per-residue score vectors of all four
tools on the same target, and those are computed at runtime today and thrown away. This caches them once.

  runs/trust_cache/<target>.npz
    keys                      residue ids, the common order for every vector below
    hit                       true pocket residue (the TARGET definitions only; never a feature)
    score|<tool>              per-residue score, all four tools
    pred|<tool>               predicted residue mask at the tool's own operating point
    pocket|<tool>             [score1, score2, score1-score2, n_pockets, volume1] for the pocket tools
    centre|<tool>             centre of the top-ranked pocket (pocket tools only)
    bfac                      per-residue C-alpha B-factor, for Tier 3

LEAKAGE: the predicted sets here are deliberately NOT the gated ones used for scoring. `score_pockets` in
p1_s3_score keeps only pockets within 4 A of a ligand atom, which needs the holo structure; anything derived
from it — including p1_master's n_pred and spread — is contaminated for this purpose. A pocket tool's
prediction here is its own top-ranked pockets, a probability tool's is its residues above its own threshold.

Usage (structprep env): python src/t0_extract.py [--split test|train|all]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RUNS

CACHE = RUNS / "trust_cache"
P1 = RUNS / "p1"
TOP_N = 3                       # a pocket tool's "prediction": its top-ranked pockets, gate-free
PROB_T = {"AE-PocketMiner": 0.7, "pLM-NN": 0.75}
SPLITS = {"test": ["test"], "train": [f"train-fold-{i}" for i in range(4)]}
SPLITS["all"] = SPLITS["test"] + SPLITS["train"]


def pockets_for(tool, tdir, tid):
    from p1_adapters import fpocket, p2rank
    if tool == "fpocket":
        d = tdir / f"{tid}_out"
        return fpocket(d) if (d / f"{tid}_info.txt").exists() else None
    f = P1 / "p2rank_out" / f"{tid}.pdb_predictions.csv"
    return p2rank(f) if f.exists() else None


def one(tid, tdir):
    from p1_adapters import ca_table, per_residue_probs, pocket_residue_scores, p2rank_residue_scores
    from prody import parsePDB
    keys, _ = ca_table(tdir / f"{tid}.pdb")
    meta = json.loads((tdir / "meta.json").read_text())
    truth = set(meta["truth"]) & set(keys)
    idx = {k: i for i, k in enumerate(keys)}
    out = {"keys": np.array(keys), "hit": np.array([k in truth for k in keys])}

    for tool in ("fpocket", "P2Rank"):
        ps = pockets_for(tool, tdir, tid)
        if ps is None:
            continue
        # P2Rank emits its own per-residue scores; fpocket does not, so a residue takes the best score of any
        # pocket holding it. This is what p1_s3_score and the published reliability numbers use.
        rc = P1 / "p2rank_out" / f"{tid}.pdb_residues.csv"
        s = p2rank_residue_scores(rc, keys) if tool == "P2Rank" and rc.exists() else pocket_residue_scores(ps, keys)
        out[f"score|{tool}"] = np.array([s[k] for k in keys], np.float32)
        rank = sorted(ps, key=lambda p: p["rank"])
        top = {k for p in rank[:TOP_N] for k in p["residues"] if k in idx}
        out[f"pred|{tool}"] = np.array([k in top for k in keys])
        sc = [p["score"] for p in rank]
        vol = rank[0].get("volume", np.nan) if rank else np.nan
        out[f"pocket|{tool}"] = np.array(
            [sc[0] if sc else np.nan, sc[1] if len(sc) > 1 else np.nan,
             (sc[0] - sc[1]) if len(sc) > 1 else np.nan, len(ps), vol], np.float32)
        out[f"centre|{tool}"] = np.asarray(rank[0]["center"], np.float32) if rank else np.full(3, np.nan, np.float32)

    for tool, folder in (("AE-PocketMiner", "pocketminer_out"), ("pLM-NN", "plmnn_out")):
        f = P1 / folder / f"{tid}-preds.npy"
        if not f.exists():
            continue
        s = per_residue_probs(f, keys)
        v = np.array([s[k] for k in keys], np.float32)
        out[f"score|{tool}"] = v
        out[f"pred|{tool}"] = v >= PROB_T[tool]

    b = parsePDB(str(tdir / f"{tid}.pdb"), subset="ca")
    bk = dict(zip([f"{c}_{n}{i.strip()}" for c, n, i in zip(b.getChids(), b.getResnums(), b.getIcodes())],
                  b.getBetas()))
    out["bfac"] = np.array([bk.get(k, np.nan) for k in keys], np.float32)
    return out


def main():
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "all"
    want = set(SPLITS[split])
    CACHE.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in CACHE.glob("*.npz")}
    todo = []
    for m in sorted(P1.glob("*/meta.json")):
        d = json.loads(m.read_text())
        if d["split"] in want and d.get("usable", True) and d["target_id"] not in done:
            todo.append((d["target_id"], m.parent))
    print(f"{len(done)} cached, {len(todo)} to extract ({split})", flush=True)
    bad = 0
    for i, (tid, tdir) in enumerate(todo, 1):
        try:
            np.savez_compressed(CACHE / f"{tid}.npz", **one(tid, tdir))
        except Exception as e:                      # one bad structure must not stop the sweep
            bad += 1
            print(f"  failed {tid}: {type(e).__name__}: {e}", flush=True)
        if i % 100 == 0:
            print(f"  [{i}/{len(todo)}]", flush=True)
    n = len(list(CACHE.glob("*.npz")))
    print(f"done: {n} targets cached, {bad} failed")


if __name__ == "__main__":
    main()
