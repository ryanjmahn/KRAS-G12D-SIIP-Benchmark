"""
P1 s0: shortlist candidates for the curated deep-analysis subset (~30-40 targets).

Reads data/cryptobench/pairs.csv (from p1_s0_data.py), pulls apo and holo metadata from the
RCSB GraphQL API, applies the roadmap's panel criteria and writes every candidate with the
reason it passed or failed. Picking the final 30-40 (family diversity, mechanism) is a human
step: copy the chosen rows into data/curated_subset/manifest.csv.

Criteria (edit here):
  - main holo per apo only, single-chain pocket
  - X-ray, resolution <= MAX_RES for both apo and holo
  - same oligomeric state in biological assembly 1 (the MAT2A dimer lesson)
  - same UniProt accession on the pocket chain (construct match); a length difference > LEN_TOL
    is only flagged (length_mismatch), since tags and disordered tails make raw length unreliable
  - pocket >= MIN_POCKET residues; ligand not a crystallization additive or ion
  - test split only by default, so P2 never trains on a target it is evaluated on

Outputs: data/curated_subset/candidates.csv (every pair, with fail reasons)
         data/curated_subset/shortlist_draft.csv (~35 picks: one per Pfam family, drug-like and
         monomers first, at most MAX_MULTIMERS multimers; a starting point, not the final panel)
Usage (structprep env): python src/p1_s0_curate.py [--any-split]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CRYPTOBENCH, CURATED

MAX_RES = 2.5
LEN_TOL = 0.10
MIN_POCKET = 8                 # fewer residues makes precision/MCC noise
# crystallization additives and ions: not a real binding event
ADDITIVES = {"EPE", "MLI", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "SO4", "PO4", "ACT", "FMT", "MPD",
             "TRS", "CIT", "DMS", "IMD", "MES", "BME", "TLA", "SIN", "BU3", "CL", "NA", "K", "MG", "ZN", "CA",
             "MOO", "WO4", "VO4", "NO3", "IOD", "BR"}
# shortcut: hand-made code lists miss some ligands; switch to RCSB chem_comp type/weight if the panel grows
# kept, but ranked below drug-like ligands in the draft shortlist
COFACTORS = {"ATP", "ADP", "AMP", "ANP", "ACP", "GTP", "GDP", "GNP", "GSP", "GMP", "CTP", "UTP", "C5P",
             "NAD", "NAI", "NAP", "NDP", "FAD", "FMN", "SAM", "SAH", "COA", "HEM", "PLP", "TPP",
             "A", "C", "G", "U", "AR6", "APR", "CMP", "IMP"}
SUGARS = {"GLC", "BGC", "XYP", "BDP", "MAN", "GAL", "NAG", "FUC", "BMA", "SUC", "TRE", "MAL", "XYS"}
SHORTLIST_N = 35
MAX_MULTIMERS = 10             # keep some interface-adjacent pockets without making prep the bottleneck
QUERY = """query($ids:[String!]!){ entries(entry_ids:$ids){ rcsb_id
  rcsb_entry_info{ resolution_combined experimental_method }
  struct{ title }
  assemblies{ rcsb_struct_symmetry{ oligomeric_state kind } }
  polymer_entities{ rcsb_polymer_entity{ pdbx_description }
    rcsb_polymer_entity_annotation{ type annotation_id }
    entity_poly{ rcsb_sample_sequence_length }
    rcsb_polymer_entity_container_identifiers{ auth_asym_ids uniprot_ids } } } }"""


def fetch_meta(pdb_ids, batch=200):
    out = {}
    ids = sorted({i.upper() for i in pdb_ids})
    for k in range(0, len(ids), batch):
        r = requests.post("https://data.rcsb.org/graphql", timeout=120,
                          json={"query": QUERY, "variables": {"ids": ids[k:k + batch]}})
        r.raise_for_status()
        for e in r.json()["data"]["entries"] or []:
            out[e["rcsb_id"].lower()] = e
        print(f"RCSB metadata {min(k + batch, len(ids))}/{len(ids)}")
    return out


def chain_info(entry, chain):
    """Resolution, method, oligomeric state (assembly 1) and the pocket chain's UniProt/length."""
    if entry is None:
        return {}
    info = entry["rcsb_entry_info"] or {}
    res = info.get("resolution_combined") or [None]
    asm = (entry.get("assemblies") or [{}])[0].get("rcsb_struct_symmetry") or []
    glob = next((s for s in asm if s.get("kind") == "Global Symmetry"), asm[0] if asm else {})
    ent = next((p for p in entry.get("polymer_entities") or []
                if chain in (p["rcsb_polymer_entity_container_identifiers"].get("auth_asym_ids") or [])), None)
    return {
        "method": info.get("experimental_method"),
        "resolution": res[0],
        "oligomer": glob.get("oligomeric_state"),
        "uniprot": ";".join((ent or {}).get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids") or []),
        "length": ((ent or {}).get("entity_poly") or {}).get("rcsb_sample_sequence_length"),
        "protein": ((ent or {}).get("rcsb_polymer_entity") or {}).get("pdbx_description"),
        "pfam": ";".join(sorted({a["annotation_id"] for a in (ent or {}).get("rcsb_polymer_entity_annotation") or []
                                 if a.get("type") == "Pfam"})),
        "title": (entry.get("struct") or {}).get("title"),
    }


def reasons(row):
    """Failed criteria for one apo-holo pair; empty list = passes."""
    bad = []
    for side in ("apo", "holo"):
        if row[f"{side}_method"] != "X-ray":
            bad.append(f"{side} not X-ray")
        elif row[f"{side}_resolution"] is None or pd.isna(row[f"{side}_resolution"]) or row[f"{side}_resolution"] > MAX_RES:
            bad.append(f"{side} resolution > {MAX_RES}")
    if row["apo_oligomer"] != row["holo_oligomer"]:
        bad.append("oligomeric state differs")
    if not row["apo_uniprot"] or row["apo_uniprot"] != row["holo_uniprot"]:
        bad.append("UniProt differs/missing")
    if row["n_pocket_residues"] < MIN_POCKET:
        bad.append(f"pocket < {MIN_POCKET} residues")
    if str(row["ligand"]).upper() in ADDITIVES:
        bad.append("ligand is additive/ion")
    return bad


def ligand_class(lig):
    lig = str(lig).upper()
    return "cofactor" if lig in COFACTORS else "sugar" if lig in SUGARS else "drug-like"


def shortlist(passing, n=SHORTLIST_N, max_multimers=MAX_MULTIMERS):
    """Greedy draft: drug-like ligands, then monomers, then resolution; no Pfam family used twice."""
    ranked = passing.assign(
        _lig=passing.ligand.map(lambda l: ["drug-like", "cofactor", "sugar"].index(ligand_class(l))),
        _multi=passing.apo_oligomer != "Monomer",
    ).sort_values(["_lig", "_multi", "apo_resolution"])
    picked, families, multimers = [], set(), 0
    for idx, r in ranked.iterrows():
        fams = set(filter(None, str(r.apo_pfam or "").split(";"))) or {f"no-pfam:{r.apo_uniprot}"}
        if fams & families or (r._multi and multimers >= max_multimers):
            continue
        picked.append(idx)
        families |= fams
        multimers += r._multi
        if len(picked) == n:
            break
    return passing.loc[picked]


def length_mismatch(row):
    """Soft flag, not a failure: KRAS 5US4/7RPZ (189 vs 170 aa) is a valid pair and differs by >10%."""
    la, lh = row["apo_length"], row["holo_length"]
    return not la or not lh or pd.isna(la) or pd.isna(lh) or abs(la - lh) / max(la, lh) > LEN_TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--any-split", action="store_true", help="include training-fold targets too")
    args = ap.parse_args()

    pairs = pd.read_csv(CRYPTOBENCH / "pairs.csv")
    cand = pairs[pairs.is_main_holo & ~pairs.multichain]
    if not args.any_split:
        cand = cand[cand.split == "test"]
    meta = fetch_meta(set(cand.apo_pdb_id) | set(cand.holo_pdb_id))

    rows = []
    for _, p in cand.iterrows():
        row = p.to_dict()
        for side in ("apo", "holo"):
            for k, v in chain_info(meta.get(p[f"{side}_pdb_id"]), p[f"{side}_chain"]).items():
                row[f"{side}_{k}"] = v
        row["fail_reasons"] = "; ".join(reasons(row))
        row["length_mismatch"] = length_mismatch(row)  # check these by eye: tags vs truncated pocket region
        rows.append(row)
    out = pd.DataFrame(rows)
    out["passes"] = out.fail_reasons == ""
    out["ligand_class"] = out.ligand.map(ligand_class)
    out["apo_pfam"] = out.apo_pfam.fillna("")
    # one candidate per protein: the best-resolved passing pair per UniProt accession
    out = out.sort_values(["passes", "apo_resolution"], ascending=[False, True])
    out["first_for_uniprot"] = ~out.duplicated("apo_uniprot") & out.passes

    CURATED.mkdir(parents=True, exist_ok=True)
    out.to_csv(CURATED / "candidates.csv", index=False)
    print(f"\nwrote {CURATED / 'candidates.csv'}: {len(out)} pairs, {out.passes.sum()} pass, "
          f"{out.first_for_uniprot.sum()} distinct UniProt among passing")
    print(out.fail_reasons.str.split("; ").explode().value_counts().drop("", errors="ignore").to_string())

    draft = shortlist(out[out.first_for_uniprot])
    cols = ["apo_pdb_id", "apo_chain", "holo_pdb_id", "holo_chain", "ligand", "ligand_class",
            "n_pocket_residues", "apo_resolution", "holo_resolution", "apo_oligomer", "apo_pfam",
            "length_mismatch", "apo_protein"]
    draft[cols].to_csv(CURATED / "shortlist_draft.csv", index=False)
    print(f"\nwrote {CURATED / 'shortlist_draft.csv'}: {len(draft)} targets, "
          f"{(draft.apo_oligomer != 'Monomer').sum()} multimers, ligands {draft.ligand_class.value_counts().to_dict()}")
    print("draft only: review each row, then copy your picks (with a justification) into manifest.csv")


if __name__ == "__main__":
    main()
