"""
Phase 2 answer keys + ladder validation (items 1, 2, 3).
All derived from the HOLO crystal (7RPZ) + aligned MRTX1133 pose.
Independent of every benchmarked tool.

1. True SII-P residues = protein residues within 4.5 A of MRTX1133
   (gold standard for precision/recall/MCC in Phase 3). His95 checked.
2. True ligand position = MRTX1133 crystal center (DCA reference).
3. Ladder validation = switch-II RMSD of most-open conformer (conf 11) vs
   7RPZ's real switch-II ("closest approach to true open").
"""
from prody import *
import numpy as np

# ---- load holo receptor (aligned frame) + aligned ligand ----
holo = parsePDB('7RPZ_H.pdb')                          # aligned receptor
lig  = parsePDB('MRTX1133_crystal_pose_aligned.pdb')   # aligned ligand (44 heavy atoms)

lig_coords = lig.getCoords()
lig_center = lig_coords.mean(axis=0)

# ===== 1. TRUE SII-P RESIDUES (within 4.5 A of any ligand atom) =====
prot = holo.select('protein')
true_residues = set()
prot_coords = prot.getCoords()
prot_resnums = prot.getResnums()
for i, pc in enumerate(prot_coords):
    dmin = np.min(np.linalg.norm(lig_coords - pc, axis=1))
    if dmin <= 4.5:
        true_residues.add(int(prot_resnums[i]))

true_residues = sorted(true_residues)
print("="*60)
print("1. TRUE SII-P RESIDUES (contact MRTX1133 within 4.5 A):")
print("   ", true_residues)
print("   His95 present?", 95 in true_residues)
print("   count:", len(true_residues))

# ===== 2. TRUE LIGAND POSITION (DCA reference) =====
print("\n" + "="*60)
print("2. TRUE LIGAND POSITION (MRTX1133 center, DCA reference):")
print(f"   center = ({lig_center[0]:.2f}, {lig_center[1]:.2f}, {lig_center[2]:.2f})")

# ===== 3. LADDER VALIDATION: conf 11 switch-II RMSD vs 7RPZ =====
conf11 = parsePDB('ladder/lad_11.pdb')
s2_conf = conf11.select('resnum 65 to 76 and name CA')
s2_holo = holo.select('resnum 65 to 76 and name CA')

print("\n" + "="*60)
print("3. LADDER VALIDATION (most-open conf 11 vs true open 7RPZ):")
if s2_conf is not None and s2_holo is not None and s2_conf.numAtoms()==s2_holo.numAtoms():
    rmsd = np.sqrt(np.mean(np.sum((s2_conf.getCoords()-s2_holo.getCoords())**2, axis=1)))
    print(f"   switch-II RMSD (conf11 vs 7RPZ) = {rmsd:.2f} A")
    print("   -> closest approach of MD-free ladder to true open state")
    if rmsd < 3.0:
        print("   VERDICT: lands close -> MD-free motions reproduce the opening. Clean, non-circular.")
    else:
        print("   VERDICT: does not fully reach open -> report honestly.")
else:
    print("   (atom count mismatch — check switch-II selection)")
    if s2_conf: print("   conf11 switch-II CAs:", s2_conf.numAtoms())
    if s2_holo: print("   7RPZ switch-II CAs:", s2_holo.numAtoms())

# ---- save the answer keys ----
with open('answer_key_siip_residues.txt','w') as f:
    f.write("# True SII-P residues (within 4.5 A of MRTX1133 in 7RPZ)\n")
    f.write("# Gold standard for precision/recall/MCC in Phase 3\n")
    f.write(",".join(str(r) for r in true_residues) + "\n")
    f.write(f"# His95 present: {95 in true_residues}\n")
    f.write(f"# True ligand center (DCA ref): {lig_center[0]:.3f},{lig_center[1]:.3f},{lig_center[2]:.3f}\n")

print("\nSaved answer_key_siip_residues.txt")