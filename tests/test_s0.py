"""Self-checks for the Phase 0 loaders on tiny synthetic data. Run: python tests/test_s0.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p1_s0_data import assign_splits
import pandas as pd

from p1_s0_curate import chain_info, length_mismatch, reasons, shortlist


def holo(hid, chain="A", main=True):
    return {"holo_pdb_id": hid, "apo_chain": chain, "holo_chain": chain, "ligand": "ADP",
            "ligand_index": "1", "ligand_chain": chain, "is_main_holo_structure": main,
            "apo_pocket_selection": ["A_10", "A_11"]}


def test_splits():
    ds = {"1abc": [holo("2abc"), holo("3abc", main=False)], "4xyz": [holo("5xyz", chain="A-B")]}
    pairs = assign_splits(ds, {"test": {"1abc": ds["1abc"]}, "train-fold-0": {"4xyz": ds["4xyz"]}})
    assert len(pairs) == 3 and set(pairs.split) == {"test", "train-fold-0"}
    assert pairs.multichain.tolist() == [False, False, True]
    for bad_splits in ({"test": ds, "train-fold-0": {"4xyz": []}},   # apo in two splits
                       {"test": {"1abc": ds["1abc"]}}):              # apo not covered
        try:
            assign_splits(ds, bad_splits)
        except AssertionError:
            continue
        raise AssertionError("bad split went undetected")


def entry(res, olig, uniprot, length, chain="A"):
    return {"rcsb_entry_info": {"resolution_combined": [res], "experimental_method": "X-ray"},
            "struct": {"title": "t"},
            "assemblies": [{"rcsb_struct_symmetry": [{"oligomeric_state": olig, "kind": "Global Symmetry"}]}],
            "polymer_entities": [{"rcsb_polymer_entity": {"pdbx_description": "p"},
                                  "entity_poly": {"rcsb_sample_sequence_length": length},
                                  "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": [chain], "uniprot_ids": [uniprot]}}]}


def row(apo, holo_, n_pocket=16, ligand="6IC"):
    r = {"n_pocket_residues": n_pocket, "ligand": ligand}
    for side, e in (("apo", apo), ("holo", holo_)):
        r.update({f"{side}_{k}": v for k, v in chain_info(e, "A").items()})
    return r


def test_criteria():
    good = entry(1.8, "Monomer", "P01116", 169)
    assert reasons(row(good, entry(1.3, "Monomer", "P01116", 170))) == []
    assert reasons(row(good, entry(2.9, "Monomer", "P01116", 169))) == ["holo resolution > 2.5"]
    assert reasons(row(good, entry(1.5, "Homo 2-mer", "P01116", 169))) == ["oligomeric state differs"]
    kras = row(entry(1.83, "Monomer", "P01116", 189), entry(1.3, "Monomer", "P01116", 170))
    assert reasons(kras) == [] and length_mismatch(kras)   # 5US4/7RPZ: valid pair, only flagged
    assert "UniProt differs/missing" in reasons(row(good, entry(1.5, "Monomer", "Q99999", 169)))
    assert chain_info(good, "Z")["uniprot"] == ""        # pocket chain not in any entity
    assert reasons(row(good, good, n_pocket=5)) == ["pocket < 8 residues"]
    assert reasons(row(good, good, ligand="EPE")) == ["ligand is additive/ion"]


def test_shortlist():
    df = pd.DataFrame([
        # id, ligand, oligomer, pfam, resolution
        ("a", "ATP", "Monomer", "PF1", 1.0),       # cofactor: ranked after drug-like
        ("b", "XYZ", "Monomer", "PF1", 2.0),       # drug-like wins PF1
        ("c", "XYZ", "Homo 2-mer", "PF2", 1.0),
        ("d", "XYZ", "Homo 2-mer", "PF3", 1.1),    # second multimer: over cap of 1
        ("e", "GLC", "Monomer", "PF4", 1.0),
        ("f", "XYZ", "Monomer", "", 1.5),          # no Pfam: still eligible
    ], columns=["apo_pdb_id", "ligand", "apo_oligomer", "apo_pfam", "apo_resolution"]).assign(apo_uniprot=lambda d: d.apo_pdb_id)
    got = list(shortlist(df, n=10, max_multimers=1).apo_pdb_id)
    assert got == ["f", "b", "c", "e"], got


if __name__ == "__main__":
    test_splits()
    test_criteria()
    test_shortlist()
    print("ok")
