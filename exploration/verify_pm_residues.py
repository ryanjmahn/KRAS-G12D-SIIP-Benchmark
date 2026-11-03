"""Print PocketMiner's probability for each TRUE SII-P residue (conf 23),
so we see exactly what it assigned to the pocket of interest."""
from prody import *
import numpy as np, os

TRUE_PRESENT=[9,10,11,12,16,58,59,65,68,69,72,78,88,92,95,96,99,100,102,103]
cid=23
pdb=f'ladder_dense/dense_{cid:02d}.pdb'
npy=f'ae-pocketminer/results/pocketminer/dense_{cid:02d}-preds.npy'
if not os.path.exists(npy):
    npy=f'results/pocketminer/dense_{cid:02d}-preds.npy'

preds=np.load(npy).flatten()
conf=parsePDB(pdb)
ca=conf.select('protein and name CA')
resnums=list(ca.getResnums())

# build residue->prob map by ORDER (index i of preds -> resnums[i])
n=min(len(preds),len(resnums))
res2prob={int(resnums[i]): float(preds[i]) for i in range(n)}

print("PocketMiner probability for each TRUE SII-P residue (conf 23):")
print("residue | probability")
probs=[]
for r in TRUE_PRESENT:
    p=res2prob.get(r)
    if p is not None:
        print(f"  {r:3d}   |  {p:.2f}")
        probs.append(p)
    else:
        print(f"  {r:3d}   |  (not in structure)")
print(f"\nMean prob over true SII-P residues: {np.mean(probs):.2f}")
print(f"Mean prob over ALL residues: {np.mean(preds):.2f}")
print("\nIf SII-P mean is much higher than overall mean -> PocketMiner")
print("DOES favor the SII-P. If similar -> it does NOT distinguish it.")