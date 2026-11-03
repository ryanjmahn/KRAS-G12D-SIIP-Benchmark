"""
PHASE 3 — CORRECTED SCORING for fpocket AND P2Rank.
Fix: anchor detection on the TRUE SII-P RESIDUES (the answer key), not on
His95 proximity. The SII-P is an extended site; a His95-distance cutoff
wrongly excluded pockets at the P-loop/switch-II end. Now: a tool "detects"
the SII-P if any of its pockets overlaps the true residues; we score the
union of all pockets that touch the true site.

Same logic applied to BOTH tools -> fair, consistent, artifact-free.
Re-scores existing tool outputs (no re-running the tools).
"""
from prody import *
import numpy as np, os, csv, glob

TRUE_ALL=[9,10,11,12,16,58,59,60,61,62,63,64,65,68,69,72,78,88,92,95,96,99,100,102,103]
MISSING={60,61,62,63,64}
TRUE_PRESENT=set(r for r in TRUE_ALL if r not in MISSING)
LIG_CENTER=np.array([-36.787,37.918,9.395])

openness={}
with open('ladder_dense.csv') as f:
    for row in csv.DictReader(f):
        c=int(row['conf']); openness[c]=float(row['switch2_rmsd']) if row['switch2_rmsd'] else 0.0

def metrics(union_res, prot_res):
    truth=TRUE_PRESENT & prot_res
    TP=len(union_res&truth);FP=len(union_res-truth);FN=len(truth-union_res)
    TN=len(prot_res-union_res-truth)
    prec=TP/(TP+FP) if (TP+FP)>0 else 0
    rec =TP/(TP+FN) if (TP+FN)>0 else 0
    f1  =2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc=(TP*TN-FP*FN)/den if den>0 else 0
    return prec,rec,f1,mcc

# ---------- FPOCKET rescoring ----------
def parse_volume_line(line):
    s=line.strip()
    if s.startswith('Volume') and 'score' not in s.lower():
        p=s.split(':')
        if len(p)==2:
            try: return float(p[1].strip())
            except: return None
    return None

def fpocket_rescore(cid):
    pdb=f'ladder_dense/dense_{cid:02d}.pdb'
    outdir=f'ladder_dense/dense_{cid:02d}_out'
    pk=os.path.join(outdir,f'dense_{cid:02d}_out.pdb')
    if not os.path.exists(pk): return None
    conf=parsePDB(pdb); prot_ca=conf.select('protein and name CA')
    prot_res=set(int(x) for x in prot_ca.getResnums())
    ca_coords=prot_ca.getCoords(); ca_nums=prot_ca.getResnums()
    sph={}
    for line in open(pk):
        if line.startswith('HETATM') and ' STP ' in line:
            try:
                x=float(line[30:38]);y=float(line[38:46]);z=float(line[46:54]);pn=int(line[22:26])
                sph.setdefault(pn,[]).append((x,y,z))
            except: pass
    if not sph: return None
    def pocket_res(pn):
        s=np.array(sph[pn]); r=set()
        for i,cc in enumerate(ca_coords):
            if np.min(np.linalg.norm(s-cc,axis=1))<=4.5: r.add(int(ca_nums[i]))
        return r
    # pockets that OVERLAP the true site
    overlapping=[pn for pn in sph if pocket_res(pn)&TRUE_PRESENT]
    detected=len(overlapping)>0
    union=set()
    for pn in overlapping: union|=pocket_res(pn)
    prec,rec,f1,mcc=metrics(union,prot_res)
    n_frag=len(overlapping)
    return dict(detected=detected,n_frag=n_frag,precision=prec,recall=rec,f1=f1,mcc=mcc)

# ---------- P2RANK rescoring ----------
def p2rank_rescore(cid):
    pdb=f'ladder_dense/dense_{cid:02d}.pdb'
    csvp=f'p2rank_out/dense_{cid:02d}/dense_{cid:02d}.pdb_predictions.csv'
    if not os.path.exists(csvp): return None
    conf=parsePDB(pdb)
    prot_res=set(int(x) for x in conf.select('protein and name CA').getResnums())
    pockets=[]
    with open(csvp) as f:
        reader=csv.reader(f); header=[x.strip() for x in next(reader)]
        def col(n):
            for i,x in enumerate(header):
                if x.strip()==n: return i
            return None
        ci_res=col('residue_ids')
        for parts in reader:
            if len(parts)<len(header): continue
            resnums=set()
            for tok in parts[ci_res].split():
                if '_' in tok:
                    try: resnums.add(int(tok.split('_')[1]))
                    except: pass
            pockets.append(resnums)
    overlapping=[p for p in pockets if p&TRUE_PRESENT]
    detected=len(overlapping)>0
    union=set()
    for p in overlapping: union|=(p&prot_res)
    prec,rec,f1,mcc=metrics(union,prot_res)
    n_frag=len(overlapping)
    return dict(detected=detected,n_frag=n_frag,precision=prec,recall=rec,f1=f1,mcc=mcc)

# ---------- run both, write combined csv ----------
print("CORRECTED SCORING (anchored on true SII-P residues)")
print("conf | open | fpocket det/rec/MCC | P2Rank det/rec/MCC")
print("-"*60)
rows=[]
for c in range(24):
    o=openness.get(c,0.0)
    fp=fpocket_rescore(c); pr=p2rank_rescore(c)
    fp_s = f"{'Y' if fp['detected'] else 'N'}/{fp['recall']:.2f}/{fp['mcc']:.2f}" if fp else "n/a"
    pr_s = f"{'Y' if pr['detected'] else 'N'}/{pr['recall']:.2f}/{pr['mcc']:.2f}" if pr else "n/a"
    print(f" {c:2d}  | {o:.2f} |   {fp_s:>16}   |   {pr_s:>16}")
    rows.append((c,o,fp,pr))

with open('phase3_corrected.csv','w') as f:
    f.write("conf,openness,fp_detected,fp_nfrag,fp_recall,fp_mcc,p2_detected,p2_nfrag,p2_recall,p2_mcc\n")
    for c,o,fp,pr in rows:
        def g(d,k): return d[k] if d else ''
        fpd = int(fp['detected']) if fp else ''
        prd = int(pr['detected']) if pr else ''
        f.write(f"{c},{o},{fpd},{g(fp,'n_frag')},{g(fp,'recall') if fp else ''},{g(fp,'mcc') if fp else ''},"
                f"{prd},{g(pr,'n_frag')},{g(pr,'recall') if pr else ''},{g(pr,'mcc') if pr else ''}\n")

# detection-rate summary
fp_det=sum(1 for _,_,fp,_ in rows if fp and fp['detected'])
pr_det=sum(1 for _,_,_,pr in rows if pr and pr['detected'])
print(f"\nDETECTION RATE (SII-P found): fpocket {fp_det}/24, P2Rank {pr_det}/24")
print("Saved phase3_corrected.csv")