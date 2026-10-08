import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import P2RANK_ANCHOR_OUT, PM_PREDS, RESULTS, STRUCTURES
import os, csv, numpy as np
from prody import parsePDB
from phase3_p2rank import parse_predictions

KEY = {9,10,11,12,16,58,59,60,61,62,63,64,65,68,69,72,78,88,92,95,96,99,100,102,103}
NEAR = 14.0
PM_THRESH = 0.7
LIG_CENTER = np.array([-36.787, 37.918, 9.395])

def metrics(pred, truth, universe):
    TP=len(pred&truth); FP=len(pred-truth); FN=len(truth-pred)
    TN=len(universe-pred-truth)
    rec=TP/(TP+FN) if TP+FN else 0.0
    prec=TP/(TP+FP) if TP+FP else 0.0
    d=((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    mcc=(TP*TN-FP*FN)/d if d else 0.0
    return prec, rec, mcc, len(pred)

def universe_and_his(pdb):
    ca = parsePDB(pdb).select('protein and name CA')
    his = parsePDB(pdb).select('resnum 95 and name CA')
    return set(int(r) for r in ca.getResnums()), his.getCoords()[0]

rows=[]
for tag, pdb, p2dir in [("5US4",str(STRUCTURES/"5US4_H.pdb"),str(P2RANK_ANCHOR_OUT/"p2rank_5US4")),
                        ("7RPZ",str(STRUCTURES/"7RPZ_H.pdb"),str(P2RANK_ANCHOR_OUT/"p2rank_7RPZ"))]:
    uni, his = universe_and_his(pdb)

    # --- P2Rank: union of pockets within NEAR of His95 ---
    pk = parse_predictions(f"{p2dir}/{os.path.basename(pdb)}_predictions.csv")
    near=[p for p in pk if np.linalg.norm(p['center']-his)<=NEAR]
    union=set().union(*[p['residues'] for p in near]) if near else set()
    dca = min(np.linalg.norm(p['center']-LIG_CENTER) for p in near) if near else None
    pr,rc,mc,n = metrics(union, KEY, uni)
    rows.append([tag,"P2Rank",len(near),f"{dca:.2f}" if dca else "NA",f"{pr:.3f}",f"{rc:.3f}",f"{mc:.3f}",n])

    # --- fpocket: union of pockets within NEAR of His95 ---
    pdir=f"{pdb.replace('.pdb','')}_out/pockets"
    near_f=[]; res_f=set(); dmin=None
    for f in sorted(os.listdir(pdir)):
        if not f.endswith("_atm.pdb"): continue
        s=parsePDB(os.path.join(pdir,f))
        c=s.getCoords().mean(axis=0); d=np.linalg.norm(c-his); d_lig=np.linalg.norm(c-LIG_CENTER)
        if d<=NEAR:
            near_f.append(f); res_f |= set(int(r) for r in s.getResnums())
            dmin=d_lig if dmin is None else min(dmin,d_lig)
    pr,rc,mc,n = metrics(res_f, KEY, uni)
    rows.append([tag,"fpocket",len(near_f),f"{dmin:.2f}" if dmin else "NA",f"{pr:.3f}",f"{rc:.3f}",f"{mc:.3f}",n])

    # --- AE-PocketMiner: residues above threshold ---
    npy=str(PM_PREDS / f"{tag}_H-preds.npy")
    preds=np.load(npy).flatten()
    resl=sorted(uni)
    pred_pm={r for r,p in zip(resl,preds) if p>=PM_THRESH}
    pr,rc,mc,n = metrics(pred_pm, KEY, uni)
    mean_key=float(np.mean([p for r,p in zip(resl,preds) if r in KEY]))
    mean_all=float(np.mean(preds))
    rows.append([tag,"AE-PocketMiner","NA","NA",f"{pr:.3f}",f"{rc:.3f}",f"{mc:.3f}",n])
    print(f"{tag} PM mean prob: key {mean_key:.3f} vs protein-wide {mean_all:.3f}")

with open(RESULTS / "anchors_all.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["structure","tool","n_pockets_near","dca","precision","recall","mcc","n_predicted"])
    w.writerows(rows)

for r in rows: print(*r, sep="\t")