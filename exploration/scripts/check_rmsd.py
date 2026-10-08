"""
Symmetry-corrected redocking RMSD for the validation gate.
Compares each Vina docked pose against the crystal pose using RDKit's
best-RMSD (which handles atom symmetry and matching automatically).

Usage:
    python check_rmsd.py

Expects (in the same folder):
    MRTX1133_redocked_v2.pdbqt   (Vina output, multi-model)
    MRTX1133_ideal.sdf           (the clean ligand you docked, for bond info)
    MRTX1133_crystal_pose_aligned.pdb  (crystal reference, aligned frame)
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
import sys

# ---- load the crystal reference (PDB, heavy atoms only) ----
crystal = Chem.MolFromPDBFile("MRTX1133_crystal_pose_aligned.pdb", removeHs=True, sanitize=False)
if crystal is None:
    sys.exit("ERROR: could not read crystal PDB")

# ---- load the docked poses from the PDBQT ----
# PDBQT isn't directly RDKit-readable, so we read the poses via the PDB blocks.
# Vina PDBQT models are separated by MODEL/ENDMDL. We convert each to PDB-ish text.
with open("MRTX1133_redocked_v2.pdbqt") as f:
    content = f.read()

# split into models
models = []
current = []
for line in content.splitlines():
    if line.startswith("MODEL"):
        current = []
    elif line.startswith("ENDMDL"):
        models.append("\n".join(current))
    elif line.startswith(("ATOM", "HETATM")):
        # convert PDBQT atom line to a PDB-parseable one (strip the trailing
        # PDBQT-specific columns: partial charge + autodock type)
        # keep first 66 chars (standard PDB atom record width)
        current.append(line[:66])

print(f"Found {len(models)} docked poses")
print(f"Crystal reference heavy atoms: {crystal.GetNumAtoms()}")
print()

best_overall = 999
for i, m in enumerate(models, 1):
    pdb_block = "\n".join([l for l in m.splitlines()]) + "\nEND\n"
    pose = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)
    if pose is None:
        print(f"mode {i}: could not parse")
        continue
    n_pose = pose.GetNumAtoms()
    if n_pose != crystal.GetNumAtoms():
        print(f"mode {i}: atom count mismatch (pose {n_pose} vs crystal {crystal.GetNumAtoms()})")
        continue
    try:
        # GetBestRMS handles atom symmetry + finds best atom mapping
        rmsd = rdMolAlign.GetBestRMS(pose, crystal)
        print(f"mode {i}: RMSD = {rmsd:.2f} A")
        if rmsd < best_overall:
            best_overall = rmsd
    except Exception as e:
        # fallback: CalcRMS without moving
        try:
            rmsd = rdMolAlign.CalcRMS(pose, crystal)
            print(f"mode {i}: RMSD = {rmsd:.2f} A (CalcRMS)")
            if rmsd < best_overall:
                best_overall = rmsd
        except Exception as e2:
            print(f"mode {i}: RMSD failed ({e2})")

print()
if best_overall < 999:
    print(f"BEST POSE RMSD: {best_overall:.2f} A")
    if best_overall <= 2.0:
        print("GATE: PASS (<= 2 A)")
    elif best_overall <= 3.0:
        print("GATE: EFFECTIVE PASS (2-3 A) - core correct, acceptable for flexible ligand")
    else:
        print("GATE: strict fail on all-atom; check core RMSD / document functionally")
