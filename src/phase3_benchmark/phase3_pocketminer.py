"""
PHASE 3 — PocketMiner sweep across the 24-conformer dense ladder.
PocketMiner outputs PER-RESIDUE cryptic-pocket probabilities (a .npy array,
one value per residue). We threshold to get predicted residues, then score
against the true SII-P residues (same logic as fpocket/P2Rank).

Runs PocketMiner via its xtal_predict.py on all 24 conformers, then scores.
NOTE: PocketMiner lives in the aepocketminer conda env. This script assumes
you've ALREADY run PocketMiner on the conformers (produces -preds.npy files).
If not, it will tell you to run PocketMiner first.
"""
from prody import *
import numpy as np, os, csv, glob

TRUE_ALL=[9,10,11,12,16,58,59,60,61,62,63,64,65,68,69,72,78,88,92,95,96,99,100,102,103]
MISSING={60,61,62,63,64}
TRUE_PRESENT=set(r for r in TRUE_ALL if r not in MISSING)

# PocketMiner probability threshold for calling a residue "predicted cryptic"
# PocketMiner's paper uses ~0.7 as a high-confidence cutoff; we test a
# moderate threshold. Adjustable.
THRESHOLD = 0.7

openness={}
with open('ladder_dense.csv') as f:
    for row in csv.DictReader(f):
        c=int(row['conf']); openness[c]=float(row['switch2_rmsd']) if row['switch2_rmsd'] else 0.0

def metrics(pred_res, prot_res):
    truth=TRUE_PRESENT & prot_res
    TP=len(pred_res&truth);FP=len(pred_res-truth);FN=len(truth-pred_res)
    TN=len(prot_res-pred_res-truth)
    prec=TP/(TP+FP) if (TP+FP)>0 else 0
    rec =TP/(TP+FN) if (TP+FN)>0 else 0
    f1  =2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc=(TP*TN-FP*FN)/den if den>0 else 0
    return prec,rec,f1,mcc

def score_conformer(cid):
    pdb=f'ladder_dense/dense_{cid:02d}.pdb'
    # PocketMiner preds .npy — search common locations
    candidates = [
        f'ae-pocketminer/results/pocketminer/dense_{cid:02d}-preds.npy',
        f'results/pocketminer/dense_{cid:02d}-preds.npy',
        f'ae-pocketminer/results/dense_{cid:02d}-preds.npy',
    ]
    npy=None
    for c in candidates:
        if os.path.exists(c): npy=c; break
    if npy is None:
        return None  # not run yet
    preds=np.load(npy)
    # preds is per-residue probability, in the order PocketMiner processed
    # residues (protein residues). Map to residue numbers via the structure.
    conf=parsePDB(pdb)
    ca=conf.select('protein and name CA')
    resnums=ca.getResnums()
    prot_res=set(int(x) for x in resnums)
    # preds length should match number of CA residues
    preds=np.array(preds).flatten()
    if len(preds)!=len(resnums):
        # try to align — sometimes preds has extra/fewer; truncate to min
        n=min(len(preds),len(resnums))
        preds=preds[:n]; resnums=resnums[:n]
    pred_res=set(int(resnums[i]) for i in range(len(preds)) if preds[i]>=THRESHOLD)
    detected = len(pred_res & TRUE_PRESENT)>0
    maxprob = float(np.max(preds)) if len(preds) else 0
    prec,rec,f1,mcc=metrics(pred_res, prot_res)
    return dict(detected=detected,maxprob=maxprob,precision=prec,recall=rec,f1=f1,mcc=mcc,npred=len(pred_res))

# check if PocketMiner has been run
any_found = any(score_conformer(c) is not None for c in range(24))
if not any_found:
    print("PocketMiner predictions (.npy) not found for the conformers.")
    print("You need to RUN PocketMiner on the ladder conformers first:")
    print("  1. conda activate aepocketminer")
    print("  2. cd ae-pocketminer")
    print("  3. copy conformers into inputs/:  cp ../ladder_dense/dense_*.pdb inputs/")
    print("  4. python src/xtal_predict.py my_config.yaml")
    print("  5. then re-run THIS script (in structprep env)")
    raise SystemExit

print(f"PHASE 3 — PocketMiner (threshold={THRESHOLD})")
print("conf | open | det | maxprob | recall | MCC | #pred")
print("-"*54)
rows=[]
for c in range(24):
    o=openness.get(c,0.0)
    r=score_conformer(c)
    if r:
        print(f" {c:2d}  | {o:.2f} |  {'Y' if r['detected'] else 'N'}  |  {r['maxprob']:.2f}   | {r['recall']:.2f}   | {r['mcc']:.2f}| {r['npred']}")
        rows.append((c,o,r))
    else:
        print(f" {c:2d}  | {o:.2f} |  -  |   -     |  -     |  -  | -")
        rows.append((c,o,None))

with open('phase3_pocketminer.csv','w') as f:
    f.write("conf,openness,detected,maxprob,recall,mcc,f1,precision,npred\n")
    for c,o,r in rows:
        if r:
            f.write(f"{c},{o},{int(r['detected'])},{r['maxprob']:.3f},{r['recall']:.3f},{r['mcc']:.3f},{r['f1']:.3f},{r['precision']:.3f},{r['npred']}\n")
        else:
            f.write(f"{c},{o},,,,,,,\n")
det=sum(1 for _,_,r in rows if r and r['detected'])
print(f"\nDETECTION RATE: PocketMiner {det}/24")
print("Saved phase3_pocketminer.csv")