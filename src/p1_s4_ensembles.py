"""
P1 s4: MD-free conformational ensembles for the curated subset (docs/evaluation_protocol.md §8).

Same method as the KRAS ladder (src/phase2_ladder/build_ladder_dense.py): ProDy ANM on the Cα of the prepared
apo, lowest non-trivial mode, every residue displaced rigidly with its Cα, up to 1.42 Å global Cα RMSD
(capped so no residue moves more than 12 Å, the KRAS "torn" rule). KRAS used
one direction (+1, chosen by validation); a general target has no such knowledge without looking at the holo,
so the ladder runs both ways: 12 steps each side + the apo itself = 25 conformers. The holo is never an input.

Per conformer: runs/p1_ens/<target>__cNN/<target>__cNN.pdb + meta.json (the parent's, split="ensemble"), so the
P1 runner and scorer work unchanged. pLM-NN is sequence-only, so each conformer reuses the parent's prediction.
Openness, independent of every tool: Cα RMSD of the true pocket residues to the holo (after superposing on them).
Summary: results/p1_ensembles.csv (incl. seconds per target to build).

Control (--null): same amplitudes, both directions and cap, but along a random smooth motion (SMOOTH Å correlation
length) instead of the lowest mode -> runs/p1_ens_null, results/p1_ensembles_null.csv. It is the noise baseline
for "some conformer beats the apo": best-of-25 is partly selection luck, which the control measures.

Usage (structprep env): python src/p1_s4_ensembles.py [--null]
"""
import json
import zlib
import sys
import time
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from prody import ANM, confProDy, parsePDB, writePDB

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CURATED, RESULTS, RUNS
from p1_s1_prep import kabsch, load, residue_pairs, targets

confProDy(verbosity="none")
P1, OUT, OUT_NULL = RUNS / "p1", RUNS / "p1_ens", RUNS / "p1_ens_null"
N_SIDE, MAX_RMSD, MAX_DISP = 12, 1.42, 12.0      # KRAS ladder range; > 12 Å displacement = torn (KRAS rule)
SMOOTH = 8.0             # Å; correlation length of the null ladder's random motion


def holo_pocket_ca(tid, row, apo_pdb):
    """Holo Cα for each true pocket residue, paired by sequence alignment."""
    apo = gemmi.read_structure(str(apo_pdb))
    holo = load(row["holo_pdb_id"])
    pairs = residue_pairs(apo, holo, str(row["apo_chain"]).split("-"), str(row["holo_chain"]).split("-"))
    truth = set(row["truth"])
    return {a: hca for a, _, _, hca in pairs if a in truth}


def pocket_rmsd(ca_by_key, holo_ca):
    keys = [k for k in holo_ca if k in ca_by_key]
    P, Q = np.array([ca_by_key[k] for k in keys]), np.array([holo_ca[k] for k in keys])
    R, t = kabsch(P, Q)
    return float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean())), len(keys)


def random_smooth_mode(ca_xyz, seed):
    """Null direction: random per-residue vectors smoothed over SMOOTH Å (neighbours move together, so local
    geometry stays intact like a normal mode's), net translation removed. Same shape as an ANM eigenvector."""
    g = np.random.default_rng(seed).normal(size=ca_xyz.shape)
    d2 = ((ca_xyz[:, None] - ca_xyz[None]) ** 2).sum(-1)
    v = np.exp(-d2 / (2 * SMOOTH ** 2)) @ g
    v -= v.mean(0)
    return v / np.linalg.norm(v)


def build(tid, row, null=False, r_max=None):
    t0 = time.time()
    src = P1 / tid
    meta = json.loads((src / "meta.json").read_text())
    full = parsePDB(str(src / f"{tid}.pdb"))
    ca = full.select("protein and name CA")
    if null:
        evec = random_smooth_mode(ca.getCoords(), seed=zlib.crc32(tid.encode()))   # hash() is salted per process
    else:
        anm = ANM(tid)
        anm.buildHessian(ca)
        anm.calcModes(n_modes=1)
        evec = anm[0].getEigvec().reshape(-1, 3)
    unit = np.sqrt((evec ** 2).sum(1).mean())            # Cα RMSD per unit amplitude
    ca_keys = [f"{c}_{n}{i.strip()}" for c, n, i in zip(ca.getChids(), ca.getResnums(), ca.getIcodes())]
    index = {k: i for i, k in enumerate(ca_keys)}
    atom_idx = np.array([index.get(f"{c}_{n}{i.strip()}", -1)
                         for c, n, i in zip(full.getChids(), full.getResnums(), full.getIcodes())])
    moving = atom_idx >= 0
    holo_ca = holo_pocket_ca(tid, row, src / f"{tid}.pdb")
    t_modes = time.time() - t0

    # Cap the amplitude so no residue moves more than MAX_DISP: the lowest mode is often a floppy terminus,
    # and 1.42 Å global RMSD then tears it (13 of 36 targets, up to 27.6 Å, before this cap).
    cap = MAX_DISP * unit / np.linalg.norm(evec, axis=1).max()
    r_max = min(MAX_RMSD if r_max is None else r_max, cap)   # null: same amplitude as the target's NMA ladder
    out = OUT_NULL if null else OUT
    rows = []
    for k, r in enumerate(np.linspace(-r_max, r_max, 2 * N_SIDE + 1)):
        disp = evec * (r / unit)
        coords = full.getCoords().copy()
        coords[moving] += disp[atom_idx[moving]]
        conf = full.copy()
        conf.setCoords(coords)
        cid = f"{tid}__c{k:02d}"
        d = out / cid
        d.mkdir(parents=True, exist_ok=True)
        writePDB(str(d / f"{cid}.pdb"), conf)
        prmsd, n_fit = pocket_rmsd(dict(zip(ca_keys, ca.getCoords() + disp)), holo_ca)
        in_pocket = [i for i, kk in enumerate(ca_keys) if kk in holo_ca]
        pocket_disp = float(np.sqrt((disp[in_pocket] ** 2).sum(1).mean()))   # how far the pocket itself moved
        max_disp = float(np.linalg.norm(disp, axis=1).max())
        (d / "meta.json").write_text(json.dumps(meta | dict(target_id=cid, parent=tid, conf=k, split="ensemble",
                                                            ens_rmsd=round(float(r), 3), max_disp=round(max_disp, 3))))
        pm = P1 / "plmnn_out" / f"{tid}-preds.npy"           # sequence-only: same prediction for every conformer
        link = out / "plmnn_out" / f"{cid}-preds.npy"
        if pm.exists() and not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(pm)
        rows.append(dict(parent=tid, conf=k, target_id=cid, ens_rmsd=float(r), max_disp=max_disp,
                         torn=max_disp > MAX_DISP + 1e-6, pocket_rmsd_holo=prmsd, pocket_fit_ca=n_fit, pocket_disp=pocket_disp))
    elapsed = time.time() - t0
    for x in rows:
        x |= dict(seconds_modes=t_modes, seconds_total=elapsed, n_ca=len(ca_keys))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true", help="control ladder along a random smooth motion instead")
    null = ap.parse_args().null
    man = pd.read_csv(CURATED / "manifest.csv")
    t = targets(("test",)).set_index("target_id")
    amp = pd.read_csv(RESULTS / "p1_ensembles.csv").groupby("parent").ens_rmsd.max() if null else {}
    rows = []
    for n, tid in enumerate(man.target_id, 1):
        r = build(tid, t.loc[tid], null=null, r_max=amp[tid] if null else None)
        rows += r
        print(f"[{n}/{len(man)}] {tid}: {len(r[0:1]) and r[0]['n_ca']} Cα, {r[0]['seconds_total']:.1f} s, "
              f"apo pocket RMSD to holo {r[N_SIDE]['pocket_rmsd_holo']:.2f} Å, "
              f"range over ladder {min(x['pocket_rmsd_holo'] for x in r):.2f}-{max(x['pocket_rmsd_holo'] for x in r):.2f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / ("p1_ensembles_null.csv" if null else "p1_ensembles.csv"), index=False)
    per = df.groupby("parent").first()
    print(f"\n{len(df)} conformers for {df.parent.nunique()} targets -> {OUT_NULL if null else OUT}; torn: {int(df.torn.sum())}")
    print(f"build time per target: median {per.seconds_total.median():.1f} s, max {per.seconds_total.max():.1f} s")


if __name__ == "__main__":
    main()
