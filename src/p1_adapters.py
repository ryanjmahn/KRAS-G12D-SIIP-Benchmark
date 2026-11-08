"""
P1 adapters: each tool's raw output -> one common form (docs/evaluation_protocol.md §4).

Residue key: "{chain}_{resnum}{icode}", the same encoding CryptoBench uses ("A_95").
Pocket: dict(rank=int, center=np.array(3), residues=set of keys, score=float).
Residue scores: dict key -> float, for every residue in the structure.
"""
import csv
from pathlib import Path

import numpy as np
from prody import confProDy, parsePDB

confProDy(verbosity="none")


def ca_table(pdb):
    """Residue keys and Cα coordinates of the protein, in file order."""
    ca = parsePDB(str(pdb)).select("protein and name CA")
    keys = [f"{c}_{n}{i.strip()}" for c, n, i in zip(ca.getChids(), ca.getResnums(), ca.getIcodes())]
    return keys, ca.getCoords()


def fpocket(out_dir):
    """Pockets from an fpocket <name>_out/ folder. Rank = fpocket's pocket number (ordered by its score).

    Residues = fpocket's own pocket-lining atoms (pockets/pocketN_atm.pdb), the counterpart of P2Rank's
    residue_ids. A "Cα within 4.5 Å of an alpha sphere" rule was tried and loses about half the true
    residues on KRAS 7RPZ (recall 0.92 -> 0.40), because spheres touch side chains, not Cα.
    """
    out_dir = Path(out_dir)
    name = out_dir.name[:-len("_out")]
    spheres = {}
    for line in open(out_dir / f"{name}_out.pdb"):
        if line.startswith("HETATM") and " STP " in line:
            spheres.setdefault(int(line[22:26]), []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    scores, cur = {}, None
    for line in open(out_dir / f"{name}_info.txt"):
        s = line.strip()
        if s.startswith("Pocket"):
            cur = int(s.split()[1])
        elif s.startswith("Score :") and cur is not None:
            scores[cur] = float(s.split()[-1])
    pockets = []
    for pn, sph in sorted(spheres.items()):
        lining = {f"{l[21]}_{int(l[22:26])}{l[26].strip()}"
                  for l in open(out_dir / "pockets" / f"pocket{pn}_atm.pdb") if l.startswith("ATOM")}
        pockets.append(dict(rank=pn, center=np.mean(sph, axis=0), score=scores.get(pn, 0.0), residues=lining))
    return pockets


def _rows(path):
    """P2Rank CSVs pad both headers and values with spaces."""
    with open(path) as f:
        for r in csv.DictReader(f):
            yield {k.strip(): v.strip() for k, v in r.items()}


def p2rank(predictions_csv):
    return [dict(rank=int(r["rank"]), score=float(r["probability"]),
                 center=np.array([float(r["center_x"]), float(r["center_y"]), float(r["center_z"])]),
                 residues=set(r["residue_ids"].split()))
            for r in _rows(predictions_csv)]


def p2rank_residue_scores(residues_csv, keys):
    got = {f'{r["chain"]}_{r["residue_label"]}': float(r["probability"]) for r in _rows(residues_csv)}
    return {k: got.get(k, 0.0) for k in keys}


def pocket_residue_scores(pockets, keys):
    """Per-residue score for pocket tools: the best score of any pocket containing the residue, else 0."""
    best = dict.fromkeys(keys, 0.0)
    for p in pockets:
        for k in p["residues"] & best.keys():
            best[k] = max(best[k], p["score"])
    return best


def per_residue_probs(npy, keys):
    """AE-PocketMiner / pLM-NN style output: one probability per residue, in file order."""
    probs = np.load(npy).reshape(-1)
    if len(probs) != len(keys):
        raise ValueError(f"{npy}: {len(probs)} probabilities for {len(keys)} residues")
    return dict(zip(keys, probs.astype(float)))
