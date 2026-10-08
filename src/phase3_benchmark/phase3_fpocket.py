"""
PHASE 3 — fpocket sweep (v3, UNION scoring + fragmentation).
Scores the UNION of all fpocket pockets near His95 (fair coverage), and
records how many pockets the SII-P fragments into (the fragmentation finding).

Per conformer, vs openness (switch-II RMSD):
 - n_fragments: how many fpocket pockets sit near the true site (fragmentation)
 - union recall/precision/F1/MCC: coverage of true SII-P residues by ALL near pockets
 - best-single-pocket recall (for contrast with union)
 - DCA of the largest near-His95 pocket
"""
from prody import *
import numpy as np, os, subprocess, csv

TRUE_ALL=[9,10,11,12,16,58,59,60,61,62,63,64,65,68,69,72,78,88,92,95,96,99,100,102,103]
MISSING={60,61,62,63,64}
TRUE_PRESENT=[r for r in TRUE_ALL if r not in MISSING]
LIG_CENTER=np.array([-36.787,37.918,9.395])

openness={}
with open('ladder_dense.csv') as f:
    for row in csv.DictReader(f):
        c=int(row['conf']); openness[c]=float(row['switch2_rmsd']) if row['switch2_rmsd'] else 0.0

def parse_volume_line(line):
    s=line.strip()
    if s.startswith('Volume') and 'score' not in s.lower():
        p=s.split(':')
        if len(p)==2:
            try: return float(p[1].strip())
            except: return None
    return None

def analyze(pdbpath, near_radius=14.0, res_radius=4.5):
    subprocess.run(['fpocket','-f',pdbpath], capture_output=True, text=True)
    base=os.path.basename(pdbpath).replace('.pdb',''); outdir=pdbpath.replace('.pdb','_out')
    info=os.path.join(outdir,f"{base}_info.txt"); pk=os.path.join(outdir,f"{base}_out.pdb")
    conf=parsePDB(pdbpath)
    prot_ca=conf.select('protein and name CA')
    prot_res=set(int(x) for x in prot_ca.getResnums())
    if not (os.path.exists(info) and os.path.exists(pk)): return None
    his=conf.select('resnum 95 and name CA')
    if his is None: return None
    h=his.getCoords()[0]
    sph={}
    with open(pk) as fh:
        for line in fh:
            if line.startswith('HETATM') and ' STP ' in line:
                try:
                    x=float(line[30:38]);y=float(line[38:46]);z=float(line[46:54]);pn=int(line[22:26])
                    sph.setdefault(pn,[]).append((x,y,z))
                except: pass
    if not sph: return None
    vols={}; cur=None
    with open(info) as fh:
        for line in fh:
            if line.strip().startswith('Pocket'):
                try: cur=int(line.split()[1])
                except: cur=None
            else:
                v=parse_volume_line(line)
                if v is not None and cur is not None: vols[cur]=v
    # pockets near His95
    near=[pn for pn in sph if any(np.linalg.norm(np.array(s)-h)<=near_radius for s in sph[pn])]
    if not near: return None
    n_frag=len(near)
    ca_coords=prot_ca.getCoords(); ca_nums=prot_ca.getResnums()
    def pocket_residues(pn):
        s=np.array(sph[pn]); r=set()
        for i,cc in enumerate(ca_coords):
            if np.min(np.linalg.norm(s-cc,axis=1))<=res_radius: r.add(int(ca_nums[i]))
        return r
    # UNION of all near pockets
    union_res=set()
    for pn in near: union_res |= pocket_residues(pn)
    # best single pocket recall (for contrast)
    truth=set(TRUE_PRESENT)&prot_res
    best_single=0
    for pn in near:
        pr=pocket_residues(pn)
        rec_pn=len(pr&truth)/len(truth) if truth else 0
        best_single=max(best_single,rec_pn)
    # union metrics
    TP=len(union_res&truth);FP=len(union_res-truth);FN=len(truth-union_res)
    TN=len(prot_res-union_res-truth)
    prec=TP/(TP+FP) if (TP+FP)>0 else 0
    rec =TP/(TP+FN) if (TP+FN)>0 else 0
    f1  =2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc=(TP*TN-FP*FN)/den if den>0 else 0
    # DCA of largest near pocket
    biggest=max(near,key=lambda pn: vols.get(pn,0))
    center=np.mean(sph[biggest],axis=0)
    dca=np.linalg.norm(center-LIG_CENTER)
    return dict(n_frag=n_frag,dca=dca,best_single=best_single,
                precision=prec,recall=rec,f1=f1,mcc=mcc)

print("PHASE 3 — fpocket v3 (union scoring + fragmentation)")
print("conf | open | #frag | DCA | union_rec | best1_rec | union_MCC")
print("-"*64)
rows=[]
for c in range(24):
    p=f'ladder_dense/dense_{c:02d}.pdb'
    r=analyze(p); o=openness.get(c,0.0)
    if r:
        print(f" {c:2d}  | {o:.2f} |   {r['n_frag']}   | {r['dca']:4.1f}| {r['recall']:.2f}      | {r['best_single']:.2f}      | {r['mcc']:.2f}")
        rows.append((c,o,r))
    else:
        print(f" {c:2d}  | {o:.2f} |   -   |  -  |  -        |  -        |  -")
        rows.append((c,o,None))

with open('phase3_fpocket.csv','w') as f:
    f.write("conf,openness,n_fragments,dca,union_precision,union_recall,union_f1,union_mcc,best_single_recall\n")
    for c,o,r in rows:
        if r:
            f.write(f"{c},{o},{r['n_frag']},{r['dca']:.2f},{r['precision']:.3f},{r['recall']:.3f},{r['f1']:.3f},{r['mcc']:.3f},{r['best_single']:.3f}\n")
        else:
            f.write(f"{c},{o},,,,,,,\n")
print("\nSaved phase3_fpocket.csv")
