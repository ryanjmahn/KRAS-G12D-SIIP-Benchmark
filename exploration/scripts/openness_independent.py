"""
Independent openness metric (NOT tool-based) for the ladder.
Computes switch-II <-> alpha-3 helix CA-CA distances across conformers.
Pure geometry -> not circular with any benchmarked tool (unlike fpocket volume).

Tests several residue pairs and reports which best tracks closed->open,
plus switch-II RMSD from the closed reference (secondary metric).
"""
from prody import *
import numpy as np, glob, os

# ladder conformers (conf 0-11, the clean set)
files = sorted(glob.glob('ladder/lad_*.pdb'))[:12]

# closed reference for switch-II RMSD
closed = parsePDB('ladder/lad_00.pdb')
closed_s2 = closed.select('resnum 65 to 76 and name CA')  # 60-64 missing
closed_s2_coords = closed_s2.getCoords()

# candidate switch-II residues (present: 65-76) x alpha-3 residues (~90-100)
s2_candidates = [66, 68, 71, 72, 76]
a3_candidates = [90, 95, 99]   # 95 = His95

def caget(atoms, resnum):
    sel = atoms.select(f'resnum {resnum} and name CA')
    return sel.getCoords()[0] if sel is not None else None

# For each conformer, compute all candidate distances + switch-II RMSD
print("Computing independent openness metrics across 12 conformers...\n")

dist_data = {(s2,a3): [] for s2 in s2_candidates for a3 in a3_candidates}
s2_rmsds = []

for f in files:
    conf = parsePDB(f)
    # switch-II RMSD from closed (secondary metric)
    s2 = conf.select('resnum 65 to 76 and name CA')
    if s2 is not None and s2.numAtoms() == closed_s2_coords.shape[0]:
        rmsd = np.sqrt(np.mean(np.sum((s2.getCoords()-closed_s2_coords)**2, axis=1)))
    else:
        rmsd = None
    s2_rmsds.append(rmsd)
    # candidate distances
    for s2r in s2_candidates:
        for a3r in a3_candidates:
            c1 = caget(conf, s2r); c2 = caget(conf, a3r)
            d = np.linalg.norm(c1-c2) if (c1 is not None and c2 is not None) else None
            dist_data[(s2r,a3r)].append(d)

# find the pair with the largest closed->open change (most responsive)
print("Candidate switch-II<->alpha3 pairs, by how much they change (open-closed):")
print("pair (s2-a3) | closed dist | open dist | range")
print("-"*55)
best_pair=None; best_range=0
for (s2r,a3r), vals in dist_data.items():
    v = [x for x in vals if x is not None]
    if len(v) < 12: continue
    rng = max(v)-min(v)
    print(f"  {s2r}-{a3r}    |   {v[0]:5.1f}   |  {v[-1]:5.1f}  |  {rng:4.1f}")
    if rng > best_range:
        best_range = rng; best_pair=(s2r,a3r)

print(f"\nMost responsive pair: {best_pair[0]}-{best_pair[1]} (range {best_range:.1f} A)")
print("\nFINAL independent openness axis:")
print("conf | switchII-a3 dist (PRIMARY) | switchII RMSD (SECONDARY)")
print("-"*60)
best_vals = dist_data[best_pair]
rows=[]
for i,f in enumerate(files):
    d = best_vals[i]; r = s2_rmsds[i]
    print(f"  {i:2d} |        {d:5.2f}          |     {r:.2f}" if d and r else f"  {i:2d} | n/a")
    rows.append((i, d, r))

with open('ladder_openness_independent.csv','w') as fh:
    fh.write(f"conf,switch2_a3_dist_res{best_pair[0]}_{best_pair[1]},switch2_rmsd_from_closed\n")
    for i,d,r in rows:
        fh.write(f"{i},{d if d else ''},{r if r else ''}\n")
print("\nSaved ladder_openness_independent.csv")
print("This is your PRIMARY openness axis (tool-independent, not circular).")