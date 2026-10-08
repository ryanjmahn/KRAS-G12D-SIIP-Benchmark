from prody import *
import numpy as np

structure = parsePDB('5US4_H.pdb')
calphas = structure.select('protein and name CA')

anm = ANM('5US4 NMA')
anm.buildHessian(calphas)
anm.calcModes(n_modes=10)

# Test the top switch-II modes: 1, 8, 10, 2
# Push along each mode in both directions and measure if the pocket opens.
# Openness proxy: distance between a switch-II CA and an alpha-3 CA.
# Use residue 71 (switch-II) and 95 (His95, alpha-3) — both present in 5US4.

def openness(atoms):
    s2 = atoms.select('resnum 71 and name CA')
    a3 = atoms.select('resnum 95 and name CA')
    if s2 is None or a3 is None:
        return None
    return calcDistance(s2, a3)[0]

base = openness(calphas)
print(f"Closed 5US4 openness (res71-His95 CA dist): {base:.2f} A")
print(f"(For reference, open 7RPZ was ~different — we'll grade later)")
print()

for mode_idx in [0, 7, 9, 1]:  # modes 1, 8, 10, 2 (0-indexed)
    mode = anm[mode_idx]
    for direction, sign in [("+", 1), ("-", -1)]:
        # push along the mode by a scaling factor
        for scale in [50, 100, 150]:
            coords = calphas.getCoords().copy()
            evec = mode.getEigvec().reshape(-1, 3)
            new_coords = coords + sign * scale * evec
            calphas.setCoords(new_coords)
            o = openness(calphas)
            calphas.setCoords(coords)  # reset
            print(f"Mode {mode_idx+1} {direction} scale {scale}: openness = {o:.2f} A (delta {o-base:+.2f})")
    print()