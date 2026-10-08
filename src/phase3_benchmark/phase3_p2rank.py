"""
PHASE 3 — P2Rank sweep across the 24-conformer dense ladder.
Same union-scoring + fragmentation logic as fpocket, for fair comparison.

P2Rank outputs a predictions CSV per structure with ranked pockets, each
listing its residue_ids. We identify pockets near His95, take the union of
their residues, and score vs the answer key.
"""
from prody import *
import numpy as np, os, subprocess, csv, glob
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LADDER_CSV, LADDER_PDBS, P2RANK_LADDER_OUT, PRANK, RESULTS

TRUE_ALL=[9,10,11,12,16,58,59,60,61,62,63,64,65,68,69,72,78,88,92,95,96,99,100,102,103]
MISSING={60,61,62,63,64}
TRUE_PRESENT=[r for r in TRUE_ALL if r not in MISSING]
LIG_CENTER=np.array([-36.787,37.918,9.395])

# P2Rank location
PRANK = str(PRANK)

openness={}
with open(LADDER_CSV) as f:
    for row in csv.DictReader(f):
       c = int(row['conf']); openness[c] = float(row['siip_volume']) if row['siip_volume'] else 0.0

def run_p2rank(pdbpath, outdir):
    # run P2Rank predict; -o sets output dir
    subprocess.run([PRANK,'predict','-f',pdbpath,'-o',outdir],
                   capture_output=True, text=True)

def parse_predictions(csvpath):
    """Return list of pockets: each dict with center (x,y,z), rank, and
    set of residue numbers (parsed from residue_ids like 'A_95 A_96')."""
    pockets=[]
    if not os.path.exists(csvpath): return pockets
    with open(csvpath) as f:
        reader=csv.reader(f)
        header=[h.strip() for h in next(reader)]
        # find columns
        def col(name):
            for i,h in enumerate(header):
                if h.strip()==name: return i
            return None
        ci_rank=col('rank'); ci_x=col('center_x'); ci_y=col('center_y'); ci_z=col('center_z')
        ci_res=col('residue_ids')
        for parts in reader:
            if len(parts)<len(header): continue
            try:
                rank=int(parts[ci_rank])
                cx=float(parts[ci_x]); cy=float(parts[ci_y]); cz=float(parts[ci_z])
                resids=parts[ci_res].strip()
                resnums=set()
                for tok in resids.split():
                    # format like A_95
                    if '_' in tok:
                        try: resnums.add(int(tok.split('_')[1]))
                        except: pass
                pockets.append(dict(rank=rank,center=np.array([cx,cy,cz]),residues=resnums))
            except: pass
    return pockets

def analyze(pdbpath, near_radius=14.0):
    base=os.path.basename(pdbpath).replace('.pdb','')
    outdir=str(P2RANK_LADDER_OUT / base)
    os.makedirs(P2RANK_LADDER_OUT, exist_ok=True)
    run_p2rank(pdbpath, outdir)
    # find the predictions csv
    pred_csv=None
    for pat in glob.glob(f'{outdir}/*predictions*.csv'):
        pred_csv=pat; break
    if pred_csv is None: return None
    pockets=parse_predictions(pred_csv)
    if not pockets: return None
    conf=parsePDB(pdbpath)
    prot_res=set(int(x) for x in conf.select('protein and name CA').getResnums())
    his=conf.select('resnum 95 and name CA')
    if his is None: return None
    h=his.getCoords()[0]
    # pockets near His95 (by center within near_radius)
    near=[p for p in pockets if np.linalg.norm(p['center']-h)<=near_radius]
    if not near: return None
    n_frag=len(near)
    truth=set(TRUE_PRESENT)&prot_res
    # union of residues from near pockets
    union_res=set()
    for p in near: union_res |= (p['residues'] & prot_res)
    # best single pocket recall
    best_single=0
    for p in near:
        pr=p['residues']&prot_res
        rec_pn=len(pr&truth)/len(truth) if truth else 0
        best_single=max(best_single,rec_pn)
    TP=len(union_res&truth);FP=len(union_res-truth);FN=len(truth-union_res)
    TN=len(prot_res-union_res-truth)
    prec=TP/(TP+FP) if (TP+FP)>0 else 0
    rec =TP/(TP+FN) if (TP+FN)>0 else 0
    f1  =2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc=(TP*TN-FP*FN)/den if den>0 else 0
    # DCA + rank of best (nearest) near pocket
    nearest=min(near,key=lambda p:np.linalg.norm(p['center']-LIG_CENTER))
    dca=np.linalg.norm(nearest['center']-LIG_CENTER)
    best_rank=min(p['rank'] for p in near)
    return dict(n_frag=n_frag,dca=dca,best_rank=best_rank,best_single=best_single,
                precision=prec,recall=rec,f1=f1,mcc=mcc)

if __name__ == "__main__":
    print("PHASE 3 — P2Rank (union scoring + fragmentation)")
    print("conf | open | #frag | top_rank | DCA | union_rec | best1_rec | MCC")
    print("-"*68)
    rows=[]
    for c in range(24):
        p=str(LADDER_PDBS / f'dense_{c:02d}.pdb')
        r=analyze(p); o=openness.get(c,0.0)
        if r:
            print(f" {c:2d}  | {o:.2f} |   {r['n_frag']}   |   {r['best_rank']:>2}     | {r['dca']:4.1f}| {r['recall']:.2f}      | {r['best_single']:.2f}      | {r['mcc']:.2f}")
            rows.append((c,o,r))
        else:
            print(f" {c:2d}  | {o:.2f} |   -   |    -     |  -  |  -        |  -        |  -")
            rows.append((c,o,None))

    with open(RESULTS / 'phase3_p2rank.csv','w') as f:
        f.write("conf,openness,n_fragments,top_rank,dca,union_precision,union_recall,union_f1,union_mcc,best_single_recall\n")
        for c,o,r in rows:
            if r:
                f.write(f"{c},{o},{r['n_frag']},{r['best_rank']},{r['dca']:.2f},{r['precision']:.3f},{r['recall']:.3f},{r['f1']:.3f},{r['mcc']:.3f},{r['best_single']:.3f}\n")
            else:
                f.write(f"{c},{o},,,,,,,,\n")
    print("\nSaved phase3_p2rank.csv")