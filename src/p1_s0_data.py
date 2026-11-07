"""
P1 s0: download CryptoBench, reproduce its official splits, write one pair manifest.

Source: Skrhak et al., Bioinformatics 41(1) btae745 (2025); data on OSF (osf.io/pz4a9, MIT).
dataset.json maps apo PDB ID -> list of holo entries. Pocket residues are "{auth_chain}_{auth_resnum}".
Splits are separate files in the same schema: folds/test.json and folds/train-fold-{0..3}.json.

Outputs:
  data/cryptobench/raw/...        downloaded files (gitignored)
  data/cryptobench/pairs.csv      one row per apo-holo pair, with its split

Usage (structprep env):
  python src/p1_s0_data.py          # JSONs only (~16 MB)
  python src/p1_s0_data.py --cif    # also the 1.1 GB CIF archive (needed from Phase 1 on)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CRYPTOBENCH

RAW = CRYPTOBENCH / "raw"
# OSF file IDs, read from the OSF API listing of node pz4a9 on 2026-11-07
FILES = {
    "dataset.json": "ta2ju",
    "folds.json": "5s93p",
    "README.md": "x9dce",
    "folds/test.json": "q2n43",
    "folds/train-fold-0.json": "7bvs4",
    "folds/train-fold-1.json": "akfgn",
    "folds/train-fold-2.json": "nmj39",
    "folds/train-fold-3.json": "btdqy",
}
CIF_ZIP = ("auxiliary-data/cif-files.zip", "c5vp8")
SPLITS = {"test": "folds/test.json", **{f"train-fold-{k}": f"folds/train-fold-{k}.json" for k in range(4)}}
REQUIRED = ["holo_pdb_id", "apo_chain", "holo_chain", "apo_pocket_selection"]


def download(rel, osf_id):
    out = RAW / rel
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {rel}")
    with requests.get(f"https://osf.io/download/{osf_id}/", stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(out)  # no half-written file left behind if the download dies


def pairs_from(dataset, split=None):
    """Flatten {apo_id: [holo entries]} into one row per pair."""
    rows = []
    for apo, holos in dataset.items():
        for h in holos:
            missing = [k for k in REQUIRED if k not in h]
            if missing:
                raise KeyError(f"{apo}: missing {missing}; entry keys are {sorted(h)}")
            pocket = h["apo_pocket_selection"]
            rows.append({
                "apo_pdb_id": apo.lower(),
                "apo_chain": h["apo_chain"],
                "holo_pdb_id": h["holo_pdb_id"].lower(),
                "holo_chain": h["holo_chain"],
                "ligand": h.get("ligand"),
                "ligand_index": h.get("ligand_index"),
                "ligand_chain": h.get("ligand_chain"),
                "is_main_holo": bool(h.get("is_main_holo_structure", False)),
                "multichain": "-" in str(h["apo_chain"]) or "-" in str(h["holo_chain"]),
                "n_pocket_residues": len(pocket),
                "apo_pocket": ";".join(pocket),
                "split": split,
            })
    return pd.DataFrame(rows)


def assign_splits(dataset, split_sets):
    """split_sets: {split_name: {apo_id: [holos]}}. Each apo must sit in exactly one split, all covered."""
    owner = {}
    for name, d in split_sets.items():
        for apo in d:
            apo = apo.lower()
            assert apo not in owner, f"{apo} is in both {owner[apo]} and {name}"
            owner[apo] = name
    ids = {a.lower() for a in dataset}
    assert set(owner) == ids, (
        f"split files do not cover dataset.json: {len(ids - set(owner))} apo IDs unassigned, "
        f"{len(set(owner) - ids)} in splits but not in dataset")
    pairs = pairs_from(dataset)
    pairs["split"] = pairs["apo_pdb_id"].map(owner)
    n_split_pairs = sum(len(h) for d in split_sets.values() for h in d.values())
    assert n_split_pairs == len(pairs), f"pair count differs: splits {n_split_pairs} vs dataset {len(pairs)}"
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", action="store_true", help="also download the 1.1 GB CIF archive")
    args = ap.parse_args()

    for rel, osf_id in FILES.items():
        download(rel, osf_id)
    if args.cif:
        download(*CIF_ZIP)
        print(f"unzip it with:  unzip -q {RAW / CIF_ZIP[0]} -d {RAW / 'auxiliary-data'}")

    dataset = json.loads((RAW / "dataset.json").read_text())
    split_sets = {name: json.loads((RAW / rel).read_text()) for name, rel in SPLITS.items()}
    pairs = assign_splits(dataset, split_sets)
    pairs.to_csv(CRYPTOBENCH / "pairs.csv", index=False)

    print(f"\nwrote {CRYPTOBENCH / 'pairs.csv'}")
    print(f"apo structures: {pairs.apo_pdb_id.nunique()}   apo-holo pairs: {len(pairs)}")
    print(pairs.groupby("split").agg(apo=("apo_pdb_id", "nunique"), pairs=("apo_pdb_id", "size"),
                                     multichain=("multichain", "sum")).to_string())
    print("most common ligands:", Counter(pairs.ligand).most_common(8))
    kras = pairs[pairs.apo_pdb_id.isin(["5us4"]) | pairs.holo_pdb_id.isin(["7rpz"])]
    print("KRAS anchor (5US4/7RPZ) in CryptoBench:", "yes\n" + kras.to_string() if len(kras) else "no")
    folds_meta = json.loads((RAW / "folds.json").read_text())
    print("folds.json (not used; for reference):", type(folds_meta).__name__,
          list(folds_meta)[:5] if isinstance(folds_meta, dict) else folds_meta[:2])


if __name__ == "__main__":
    main()
