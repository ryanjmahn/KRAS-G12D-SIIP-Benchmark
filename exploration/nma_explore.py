from prody import *
import numpy as np

# Load the closed structure (build the ladder from CLOSED only)
structure = parsePDB('5US4_H.pdb')
print("Loaded:", structure.numAtoms(), "atoms")

# Select CA atoms of the protein for NMA (standard for ANM)
calphas = structure.select('protein and name CA')
print("CA atoms for NMA:", calphas.numAtoms())

# Build the Anisotropic Network Model (ANM) — this is the NMA
anm = ANM('5US4 NMA')
anm.buildHessian(calphas)
anm.calcModes(n_modes=10)  # compute the 10 lowest-frequency modes

print("\nComputed", anm.numModes(), "modes")

# Switch-II region is roughly residues 60-76; alpha-3 helix ~87-104
# Let's see which low-freq mode most involves switch-II movement
switch2 = calphas.select('resnum 65 to 76')  # 60-64 missing, so 65-76
print("Switch-II CAs available:", switch2.numAtoms() if switch2 else 0)

# For each of the first few modes, measure how much switch-II moves
print("\nMode | switch-II involvement (higher = more switch-II motion)")
for i in range(anm.numModes()):
    mode = anm[i]
    # get the movement vector for switch-II residues
    all_resnums = calphas.getResnums()
    s2_mask = (all_resnums >= 65) & (all_resnums <= 76)
    evec = mode.getEigvec().reshape(-1, 3)
    s2_motion = np.linalg.norm(evec[s2_mask])
    total_motion = np.linalg.norm(evec)
    frac = s2_motion / total_motion
    print(f"  {i+1}  |  {frac:.3f}")