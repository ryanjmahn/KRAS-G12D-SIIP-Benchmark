"""Diagnostic: which residues does fpocket predict at conf 23 (most open)
vs the true SII-P residues?"""
from prody import *
import numpy as np, collections

conf = parsePDB('ladder_dense/dense_23.pdb')
pk = 'ladder_dense/dense_23_out/dense_23_out.pdb'
his = conf.select('resnum 95 and name CA').getCoords()[0]

# collect alpha spheres per pocket
sph = collections.defaultdict(list)
for line in open(pk):
    if line.startswith('HETATM') and ' STP ' in line:
        x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54]); pn=int(line[22:26])
        sph[pn].append((x,y,z))

# pockets near His95
near = [pn for pn in sph if any(np.linalg.norm(np.array(s)-his)<=14 for s in sph[pn])]
print("pockets near His95:", near)

ca = conf.select('protein and name CA')
ca_coords = ca.getCoords(); ca_nums = ca.getResnums()
for pn in near:
    s = np.array(sph[pn])
    pred = set()
    for i, cc in enumerate(ca_coords):
        if np.min(np.linalg.norm(s-cc, axis=1)) <= 4.5:
            pred.add(int(ca_nums[i]))
    print(f"pocket {pn}: {len(sph[pn])} spheres -> predicts residues {sorted(pred)}")

print("\nTRUE present residues:", [9,10,11,12,16,58,59,65,68,69,72,78,88,92,95,96,99,100,102,103])