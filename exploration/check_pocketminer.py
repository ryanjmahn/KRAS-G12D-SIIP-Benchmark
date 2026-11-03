"""
Verify PocketMiner probability -> residue mapping.
Shows: array length vs residue count, and WHICH residues get the top
probabilities (should be near the true SII-P if working correctly).
"""
from prody import *
import numpy as np, os

TRUE_PRESENT=[9,10,11,12,16,58,59,65,68,69,72,78,88,92,95,96,99,100,102,103]

# check conf 23 (most open) — SII-P should be clearest here
cid=23
pdb=f'ladder_dense/dense_{cid:02d}.pdb'
npy=None
for c in [f'ae-pocketminer/results/pocketminer/dense_{cid:02d}-preds.npy',
          f'results/pocketminer/dense_{cid:02d}-preds.npy']:
    if os.path.exists(c): npy=c; break

print("npy path:", npy)
preds=np.load(npy)
preds=np.array(preds).flatten()
print("prediction array length:", len(preds))

conf=parsePDB(pdb)
ca=conf.select('protein and name CA')
resnums=ca.getResnums()
print("number of CA residues:", len(resnums))
print("residue number range:", resnums.min(), "to", resnums.max())

if len(preds)!=len(resnums):
    print("\n*** LENGTH MISMATCH — mapping is unreliable ***")
    n=min(len(preds),len(resnums))
    preds=preds[:n]; resnums=resnums[:n]

# top 15 highest-probability residues
order=np.argsort(preds)[::-1][:15]
print("\nTop 15 residues by PocketMiner probability:")
print("(if working: these should cluster near true SII-P residues)")
for i in order:
    rn=int(resnums[i]); p=preds[i]
    star="*TRUE*" if rn in TRUE_PRESENT else ""
    print(f"  residue {rn}: prob {p:.2f} {star}")

print("\nTrue SII-P residues:", TRUE_PRESENT)
# how many of top-20 predicted are true?
top20=set(int(resnums[i]) for i in np.argsort(preds)[::-1][:20])
overlap=top20 & set(TRUE_PRESENT)
print(f"\nOf PocketMiner's top-20 residues, {len(overlap)} are true SII-P: {sorted(overlap)}")