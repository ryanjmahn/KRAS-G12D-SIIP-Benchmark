"""
DENSE CONFORMATIONAL LADDER (Phase 2 - Option 1)
24 conformers, finely sampled in the PROVEN-INTACT range (CA-RMSD 0 -> 1.42),
where conf 0-11 stayed intact before. No tearing, more points, hits the
20-50 target.

- Built from 5US4 (closed) ONLY. 7RPZ never referenced.
- SIGN=+1 (the validated opening direction).
- Records stable His95 SII-P volume + switch-II RMSD (independent axis).
"""
from prody import *
import numpy as np, os, subprocess

os.makedirs('ladder_dense', exist_ok=True)

full = parsePDB('5US4_H.pdb')
calphas = full.select('protein and name CA')
anm = ANM('5US4'); anm.buildHessian(calphas); anm.calcModes(n_modes=10)
evec = anm[0].getEigvec().reshape(-1, 3)
ca_start = calphas.getCoords()
ca_resnums = calphas.getResnums()
per_ca_norm2 = np.sum(evec**2, axis=1)
rms_per_unit_alpha = np.sqrt(np.mean(per_ca_norm2))

SIGN = +1
# 24 conformers, 0 -> 1.42 CA-RMSD (the intact range from before)
target_rmsds = np.linspace(0.0, 1.42, 24)

# closed reference for switch-II RMSD (independent metric)
closed_s2 = full.select('resnum 65 to 76 and name CA').getCoords()

def build_full(alpha, outpath):
    disp = {rn: SIGN*alpha*evec[i] for i,rn in enumerate(ca_resnums)}
    conf = full.copy()
    coords = conf.getCoords().copy(); resnums = conf.getResnums()
    maxd=0.0
    for i in range(conf.numAtoms()):
        d=disp.get(resnums[i])
        if d is not None:
            coords[i]=coords[i]+d
            nd=np.linalg.norm(d)
            if nd>maxd: maxd=nd
    conf.setCoords(coords); writePDB(outpath, conf)
    return maxd

def parse_volume_line(line):
    s=line.strip()
    if s.startswith('Volume') and 'score' not in s.lower():
        p=s.split(':')
        if len(p)==2:
            try: return float(p[1].strip())
            except: return None
    return None

def stable_vol_and_rmsd(pdbpath, radius=14.0):
    subprocess.run(['fpocket','-f',pdbpath], capture_output=True, text=True)
    base=os.path.basename(pdbpath).replace('.pdb',''); outdir=pdbpath.replace('.pdb','_out')
    info=os.path.join(outdir,f"{base}_info.txt"); pk=os.path.join(outdir,f"{base}_out.pdb")
    conf=parsePDB(pdbpath)
    # switch-II RMSD (independent axis)
    s2=conf.select('resnum 65 to 76 and name CA')
    s2rmsd = np.sqrt(np.mean(np.sum((s2.getCoords()-closed_s2)**2,axis=1))) if s2 is not None else None
    # volume (supporting)
    vol=None
    if os.path.exists(info) and os.path.exists(pk):
        his=conf.select('resnum 95 and name CA')
        if his is not None:
            h=his.getCoords()[0]; near=set()
            with open(pk) as fh:
                for line in fh:
                    if line.startswith('HETATM') and ' STP ' in line:
                        try:
                            x=float(line[30:38]);y=float(line[38:46]);z=float(line[46:54]);pn=int(line[22:26])
                            if np.linalg.norm(np.array([x,y,z])-h)<=radius: near.add(pn)
                        except: pass
            vols={}; cur=None
            with open(info) as fh:
                for line in fh:
                    if line.strip().startswith('Pocket'):
                        try: cur=int(line.split()[1])
                        except: cur=None
                    else:
                        v=parse_volume_line(line)
                        if v is not None and cur is not None: vols[cur]=v
            nv=[vols[p] for p in near if p in vols]
            vol=max(nv) if nv else None
    return s2rmsd, vol

print("conf | CA-RMSD | switchII-RMSD (axis) | SII-P vol | intact")
print("-"*62)
rows=[]
for k,tr in enumerate(target_rmsds):
    alpha=tr/rms_per_unit_alpha if rms_per_unit_alpha>0 else 0
    p=f'ladder_dense/dense_{k:02d}.pdb'
    maxd=build_full(alpha,p)
    s2r,vol=stable_vol_and_rmsd(p)
    intact="yes" if maxd<12 else "TORN"
    vs=f"{vol:.1f}" if vol else "n/a"
    s2s=f"{s2r:.2f}" if s2r is not None else "n/a"
    print(f"  {k:2d} |  {tr:.2f}   |      {s2s:>5}        |  {vs:>6}  | {intact}")
    rows.append((k,tr,s2r,vol,maxd))

with open('ladder_dense.csv','w') as f:
    f.write("conf,target_ca_rmsd,switch2_rmsd,siip_volume,max_disp\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]:.3f},{r[2] if r[2] else ''},{r[3] if r[3] else ''},{r[4]:.2f}\n")
print("\nSaved ladder_dense.csv + ladder_dense/dense_*.pdb (24 conformers)")