"""
ANCHOR CHECK — all 3 tools on the 2 REAL experimental structures (no NMA).
5US4 (closed) and 7RPZ (open). Same scoring as Phase 3.

Key question: does PocketMiner still over-spread on the REAL 7RPZ holo
structure? If yes -> finding is real (not NMA artifact). If it localizes
well on real 7RPZ -> the over-spread was an NMA artifact.
"""
from prody import *
import numpy as np, os, subprocess, csv

TRUE_ALL=[9,10,11,12,16,58,59,60,61,62,63,64,65,68,69,72,78,88,92,95,96,99,100,102,103]
TRUE_PRESENT=set(TRUE_ALL)  # for the anchors, 60-64 present in 7RPZ; 5US4 missing them

def metrics(pred_res, prot_res):
    truth=TRUE_PRESENT & prot_res
    TP=len(pred_res&truth);FP=len(pred_res-truth);FN=len(truth-pred_res)
    TN=len(prot_res-pred_res-truth)
    prec=TP/(TP+FP) if (TP+FP)>0 else 0
    rec =TP/(TP+FN) if (TP+FN)>0 else 0
    den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc=(TP*TN-FP*FN)/den if den>0 else 0
    return prec,rec,mcc

def parse_volume_line(line):
    s=line.strip()
    if s.startswith('Volume') and 'score' not in s.lower():
        p=s.split(':')
        if len(p)==2:
            try: return float(p[1].strip())
            except: return None
    return None

# ---- FPOCKET ----
def fpocket_score(name):
    pdb=f'{name}.pdb'
    subprocess.run(['fpocket','-f',pdb], capture_output=True, text=True)
    outdir=f'{name}_out'; pk=f'{outdir}/{name}_out.pdb'
    if not os.path.exists(pk): return None
    conf=parsePDB(pdb); prot_ca=conf.select('protein and name CA')
    prot_res=set(int(x) for x in prot_ca.getResnums())
    ca_c=prot_ca.getCoords(); ca_n=prot_ca.getResnums()
    sph={}
    for line in open(pk):
        if line.startswith('HETATM') and ' STP ' in line:
            try:
                x=float(line[30:38]);y=float(line[38:46]);z=float(line[46:54]);pn=int(line[22:26])
                sph.setdefault(pn,[]).append((x,y,z))
            except: pass
    def pres(pn):
        s=np.array(sph[pn]); r=set()
        for i,cc in enumerate(ca_c):
            if np.min(np.linalg.norm(s-cc,axis=1))<=4.5: r.add(int(ca_n[i]))
        return r
    overl=[pn for pn in sph if pres(pn)&TRUE_PRESENT]
    union=set()
    for pn in overl: union|=pres(pn)
    p,r,m=metrics(union,prot_res)
    return dict(nfrag=len(overl),recall=r,mcc=m,npred=len(union))

# ---- P2RANK ----
def p2rank_score(name):
    outdir=f'p2rank_out/{name}'
    os.makedirs('p2rank_out',exist_ok=True)
    subprocess.run(['./p2rank_2.4.2/prank','predict','-f',f'{name}.pdb','-o',outdir],
                   capture_output=True,text=True)
    import glob
    cs=glob.glob(f'{outdir}/*predictions*.csv')
    if not cs: return None
    conf=parsePDB(f'{name}.pdb')
    prot_res=set(int(x) for x in conf.select('protein and name CA').getResnums())
    pockets=[]
    with open(cs[0]) as f:
        rd=csv.reader(f); hdr=[x.strip() for x in next(rd)]
        ci=[i for i,x in enumerate(hdr) if x.strip()=='residue_ids'][0]
        for parts in rd:
            if len(parts)<len(hdr): continue
            rn=set()
            for tok in parts[ci].split():
                if '_' in tok:
                    try: rn.add(int(tok.split('_')[1]))
                    except: pass
            pockets.append(rn)
    overl=[p for p in pockets if p&TRUE_PRESENT]
    union=set()
    for p in overl: union|=(p&prot_res)
    pr,r,m=metrics(union,prot_res)
    return dict(nfrag=len(overl),recall=r,mcc=m,npred=len(union))

# ---- POCKETMINER ----
def pm_score(name, threshold=0.7):
    npy=f'ae-pocketminer/results/pocketminer/{name}-preds.npy'
    if not os.path.exists(npy): return None
    preds=np.load(npy).flatten()
    conf=parsePDB(f'{name}.pdb')
    ca=conf.select('protein and name CA'); resn=list(ca.getResnums())
    n=min(len(preds),len(resn))
    res2p={int(resn[i]):float(preds[i]) for i in range(n)}
    prot_res=set(res2p.keys())
    pred_res=set(r for r,p in res2p.items() if p>=threshold)
    pr,r,m=metrics(pred_res,prot_res)
    # over-spread check
    siip_probs=[res2p[r] for r in TRUE_PRESENT if r in res2p]
    siip_mean=np.mean(siip_probs) if siip_probs else 0
    overall_mean=np.mean(list(res2p.values()))
    return dict(nfrag='-',recall=r,mcc=m,npred=len(pred_res),
                siip_mean=siip_mean,overall_mean=overall_mean)

print("="*64)
print("ANCHOR CHECK — all 3 tools on REAL experimental structures")
print("="*64)
for name,label in [('5US4_H','CLOSED (5US4)'),('7RPZ_H','OPEN (7RPZ)')]:
    print(f"\n--- {label} ---")
    fp=fpocket_score(name); pr=p2rank_score(name); pm=pm_score(name)
    if fp: print(f"  fpocket    : recall {fp['recall']:.2f}, MCC {fp['mcc']:.2f}, {fp['npred']} residues, {fp['nfrag']} pockets")
    if pr: print(f"  P2Rank     : recall {pr['recall']:.2f}, MCC {pr['mcc']:.2f}, {pr['npred']} residues, {pr['nfrag']} pockets")
    if pm:
        print(f"  PocketMiner: recall {pm['recall']:.2f}, MCC {pm['mcc']:.2f}, {pm['npred']} residues (thresh 0.7)")
        print(f"               SII-P mean prob {pm['siip_mean']:.2f} vs overall {pm['overall_mean']:.2f}  (gap {pm['siip_mean']-pm['overall_mean']:+.2f})")

print("\n" + "="*64)
print("KEY: On OPEN 7RPZ, is PocketMiner's SII-P mean >> overall mean?")
print(" - big gap -> PocketMiner DOES localize on real structure (NMA was the issue)")
print(" - small gap -> PocketMiner over-spreads even on real holo (finding is REAL)")