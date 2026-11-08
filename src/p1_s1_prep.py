"""
P1 s1: uniform structure prep for every Phase 1 target (docs/evaluation_protocol.md §1-2).

Targets: the CryptoBench test split (main holo per apo) + the KRAS anchor from the curated manifest.
Per target, from the RCSB mmCIFs fetched by `p1_s0_data.py --structures`:
  apo  -> first model, the pocket chain(s) only, protein only (waters, ligands, ions removed),
          one conformer per atom, MSE -> MET, other non-standard residues dropped,
          then PDB2PQR 3.6.1 (AMBER, PROPKA, pH 7.4) as for KRAS        -> runs/p1/<target>/<target>.pdb
  holo -> the ligand's heavy atoms, superposed onto the apo by Kabsch on matched Cα
          near the ligand                                              -> runs/p1/<target>/meta.json
Summary: results/p1_prep_log.csv (status per target; failures are logged, not fatal).

Usage (structprep env): python src/p1_s1_prep.py [--only TARGET ...] [--workers 8]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ANSWER_KEY, CRYPTOBENCH, CURATED, RESULTS, RUNS

CIF = CRYPTOBENCH / "raw" / "cif"
OUT = RUNS / "p1"
STANDARD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
LOCAL = 12.0           # superpose on matched Cα within this of the ligand (hinge motion elsewhere is ignored)
MIN_LOCAL, MIN_ALL = 6, 10
CORE_DEV = 3.0         # Å; iterative core superposition threshold
QC_DIST, QC_MIN_CONTACT, QC_MAX_RMSD = 6.0, 0.5, 4.0   # flags only; flagged targets are still prepared


def targets():
    pairs = pd.read_csv(CRYPTOBENCH / "pairs.csv")
    t = pairs[(pairs.split == "test") & pairs.is_main_holo].copy()
    t["target_id"] = t.apo_pdb_id + "_" + t.holo_pdb_id
    t["truth"] = t.apo_pocket.str.split(";")
    man = pd.read_csv(CURATED / "manifest.csv")
    extra = man[man.source == "this_repo"]
    key = open(ANSWER_KEY / "answer_key_siip_residues.txt").read().split("\n")
    kras_truth = [f"A_{r}" for r in next(l for l in key if l and not l.startswith("#")).split(",")]
    for _, r in extra.iterrows():
        t.loc[len(t) + 10_000] = dict(target_id=r.target_id, apo_pdb_id=r.apo_pdb_id, apo_chain=r.apo_chain,
                                      holo_pdb_id=r.holo_pdb_id, holo_chain=r.holo_chain, ligand=r.ligand,
                                      ligand_index=None, ligand_chain=None, truth=kras_truth, split="anchor")
    return t


def load(pdb_id):
    st = gemmi.read_structure(str(CIF / f"{pdb_id}.cif.gz"))
    st.setup_entities()
    while len(st) > 1:
        del st[1]
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    return st


def clean_apo(st, chains):
    st.remove_ligands_and_waters()
    model = st[0]
    for i in reversed(range(len(model))):
        if model[i].name not in chains:
            del model[i]
    missing = set(chains) - {c.name for c in model}
    if missing:
        raise ValueError(f"apo chain(s) {sorted(missing)} not found")
    dropped = 0
    for chain in model:
        for i in reversed(range(len(chain))):
            res = chain[i]
            if res.name == "MSE":                       # selenomethionine: same side chain as MET
                res.name, res.het_flag = "MET", "A"
                for a in res:
                    if a.name == "SE":
                        a.name, a.element = "SD", gemmi.Element("S")
            if res.name not in STANDARD or not all(res.find_atom(n, "*") for n in ("N", "CA", "C")):
                del chain[i]                            # PDB2PQR cannot rebuild a residue without its backbone
                dropped += 1
    if any(len(c.name) > 1 for c in model):
        raise ValueError("multi-character chain ID does not fit PDB format")
    return st, dropped


def ca_list(chain):
    return [(r.name, np.array(r["CA"][0].pos.tolist())) for r in chain
            if r.name in STANDARD | {"MSE"} and r.find_atom("CA", "*")]


def aligned_pairs(query, target):
    """Cα pairs at identical residues of a global sequence alignment (numbering-independent)."""
    q = ["MET" if n == "MSE" else n for n, _ in query]
    t = ["MET" if n == "MSE" else n for n, _ in target]
    cigar = gemmi.align_string_sequences(q, t, [], gemmi.AlignmentScoring("b")).cigar_str()
    pairs, i, j, num = [], 0, 0, ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
            continue
        for _ in range(int(num)):
            if ch == "M":
                if q[i] == t[j]:
                    pairs.append((query[i][1], target[j][1]))
                i, j = i + 1, j + 1
            elif ch == "I":
                i += 1
            else:
                j += 1
        num = ""
    return pairs


def kabsch(P, Q):
    """R, t with R @ p + t ~ q, minimizing RMSD."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def ligand_atoms(holo, name, index, chain):
    """Heavy atoms of the holo ligand. Falls back to index+chain when the PDB has since renamed the
    ligand code (e.g. CryptoBench 'C' is 'DCM' in current 1KDP). Returns (xyz, name found)."""
    def find(match):
        hits = [(r.name, a.pos.tolist()) for ch in holo[0] if chain is None or ch.name == chain
                for r in ch if match(r) and (index is None or r.seqid.num == index)
                for a in r if a.element.name != "H"]
        return hits
    hits = find(lambda r: r.name == name)
    if not hits and index is not None:
        hits = find(lambda r: r.het_flag == "H" and r.name != "HOH")
    if not hits:
        raise ValueError(f"ligand {name} {index} {chain} not found in holo")
    return np.array([p for _, p in hits]), hits[0][0]


def transfer_ligand(apo, holo, apo_chains, holo_chains, lig):
    """Superpose holo onto apo on Cα paired by sequence alignment, chain by chain."""
    P, Q = [], []
    for ac, hc in zip(apo_chains, holo_chains):
        for h, a in aligned_pairs(ca_list(holo[0][hc]), ca_list(apo[0][ac])):
            P.append(h)
            Q.append(a)
    P, Q = np.array(P), np.array(Q)
    if len(P) < MIN_ALL:
        raise ValueError(f"only {len(P)} identical aligned residues between apo and holo")
    near = np.linalg.norm(P[:, None] - lig[None], axis=2).min(1) <= LOCAL
    sel = near if near.sum() >= MIN_LOCAL else np.ones(len(P), bool)
    for _ in range(5):              # refit on the stable core: residues that move with the pocket opening are dropped
        R, t = kabsch(P[sel], Q[sel])
        dev = np.linalg.norm(P @ R.T + t - Q, axis=1)
        core = sel & (dev <= CORE_DEV)
        if core.sum() < MIN_LOCAL or (core == sel).all():
            break
        sel = core
    rmsd = float(np.sqrt((dev[sel] ** 2).mean()))
    return lig @ R.T + t, int(sel.sum()), rmsd


def prep(row):
    tid = row["target_id"]
    d = OUT / tid
    d.mkdir(parents=True, exist_ok=True)
    try:
        apo_chains = str(row["apo_chain"]).split("-")
        holo_chains = str(row["holo_chain"]).split("-")
        apo, dropped = clean_apo(load(row["apo_pdb_id"]), apo_chains)
        clean = d / "clean.pdb"
        apo.write_pdb(str(clean))
        r = subprocess.run(["pdb2pqr", "--ff=AMBER", "--titration-state-method=propka", "--with-ph=7.4",
                            "--keep-chain", f"--pdb-output={tid}.pdb", "clean.pdb", f"{tid}.pqr"],
                           cwd=d, capture_output=True, text=True)
        if r.returncode or not (d / f"{tid}.pdb").exists():
            raise RuntimeError("pdb2pqr failed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
        (d / "pdb2pqr.log").write_text(r.stdout + r.stderr)

        index = None if pd.isna(row["ligand_index"]) else int(float(row["ligand_index"]))
        lchain = None if pd.isna(row["ligand_chain"]) else str(row["ligand_chain"])
        holo = load(row["holo_pdb_id"])
        lig_raw, lig_name = ligand_atoms(holo, row["ligand"], index, lchain)
        lig, n_fit, rmsd = transfer_ligand(apo, holo, apo_chains, holo_chains, lig_raw)

        prepared = gemmi.read_structure(str(d / f"{tid}.pdb"))
        residues = {f"{c.name}_{r.seqid.num}{r.seqid.icode.strip()}": r for c in prepared[0] for r in c}
        truth = list(row["truth"])
        present = [k for k in truth if k in residues]
        if not present:
            raise ValueError("no ground-truth residue is present in the prepared apo")
        # QC: the transferred ligand should sit on the true pocket residues (the apo pocket may be
        # partly closed, hence 6 Å rather than a contact cutoff)
        near = [min(np.linalg.norm(lig - np.array(a.pos.tolist()), axis=1).min()
                    for a in residues[k] if a.element.name != "H") <= QC_DIST for k in present]
        contact = float(np.mean(near))
        meta = dict(target_id=tid, split=row["split"], apo=row["apo_pdb_id"], holo=row["holo_pdb_id"],
                    apo_chains=apo_chains, ligand=lig_name, truth=truth, n_truth_present=len(present),
                    ligand_xyz=np.round(lig, 3).tolist(), fit_ca=n_fit, fit_rmsd=round(rmsd, 3),
                    truth_near_ligand=round(contact, 3), dropped_residues=dropped)
        (d / "meta.json").write_text(json.dumps(meta))
        flag = "; ".join(f for f, bad in (("ligand off-pocket", contact < QC_MIN_CONTACT),
                                          (f"fit RMSD > {QC_MAX_RMSD}", rmsd > QC_MAX_RMSD),
                                          (f"ligand renamed {row['ligand']}->{lig_name}", lig_name != row["ligand"]))
                         if bad)
        return dict(target_id=tid, status="ok", n_truth=len(truth), n_truth_present=len(present),
                    fit_ca=n_fit, fit_rmsd=round(rmsd, 3), truth_near_ligand=round(contact, 3),
                    dropped_residues=dropped, qc_flag=flag, error="")
    except Exception as e:
        return dict(target_id=tid, status="failed", error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="target IDs to (re)prepare")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t = targets()
    if args.only:
        t = t[t.target_id.isin(args.only)]
    rows = t.to_dict("records")
    with ProcessPoolExecutor(args.workers) as pool:
        log = pd.DataFrame(list(pool.map(prep, rows)))
    if not args.only:
        log.to_csv(RESULTS / "p1_prep_log.csv", index=False)
    ok = log[log.status == "ok"]
    print(f"prepared {len(ok)}/{len(log)} targets into {OUT}")
    if len(ok):
        print(f"fit RMSD median {ok.fit_rmsd.median():.2f} Å, max {ok.fit_rmsd.max():.2f} Å; "
              f"truth residues present: {int(ok.n_truth_present.sum())}/{int(ok.n_truth.sum())}; "
              f"truth residues within {QC_DIST} Å of ligand: median {ok.truth_near_ligand.median():.2f}")
        for _, r in ok[ok.qc_flag != ""].iterrows():
            print("  QC", r.target_id, r.qc_flag, f"(fit {r.fit_rmsd} Å, near {r.truth_near_ligand})")
    for _, r in log[log.status != "ok"].iterrows():
        print("  FAILED", r.target_id, r.error)


if __name__ == "__main__":
    main()
