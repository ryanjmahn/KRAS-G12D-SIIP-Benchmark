"""
FINAL CONFORMATIONAL LADDER (Phase 2)
- Traverse ONLY the opening direction (-Mode 1) of 5US4.
- Fine steps, capped in the intact range (no tearing).
- Robust SII-P volume via His95, with a stability tweak: search alpha spheres
  within a fixed radius of His95 and take that pocket's volume; if the nearest
  pocket flips, we still anchor on His95 proximity.
- Built from 5US4 (closed) ONLY. 7RPZ never referenced.
"""
from prody import *
import numpy as np
import os, subprocess

os.makedirs('ladder', exist_ok=True)

full = parsePDB('5US4_H.pdb')
calphas = full.select('protein and name CA')
anm = ANM('5US4'); anm.buildHessian(calphas); anm.calcModes(n_modes=10)
mode1 = anm[0]
evec = mode1.getEigvec().reshape(-1, 3)      # per-CA displacement (unit-ish)
ca_start = calphas.getCoords()
ca_resnums = calphas.getResnums()

# Normalize the mode so we can step by target CA-RMSD.
# RMSD contributed by moving all CAs by (alpha*evec):
#   rmsd = alpha * sqrt(mean(||evec_i||^2))
per_ca_norm2 = np.sum(evec**2, axis=1)
rms_per_unit_alpha = np.sqrt(np.mean(per_ca_norm2))

SIGN = +1                    # opening direction
target_rmsds = np.linspace(0.0, 2.2, 18)   # 18 conformers, 0 -> 2.2 A CA-RMSD
                                            # (2.2 was intact earlier; stops before tearing)

def build_full(alpha, outpath):
    """Move all atoms by their residue CA displacement * alpha (SIGN applied)."""
    disp = {rn: SIGN * alpha * evec[i] for i, rn in enumerate(ca_resnums)}
    conf = full.copy()
    coords = conf.getCoords().copy()
    resnums = conf.getResnums()
    maxd = 0.0
    for i in range(conf.numAtoms()):
        d = disp.get(resnums[i])
        if d is not None:
            coords[i] = coords[i] + d
            nd = np.linalg.norm(d)
            if nd > maxd: maxd = nd
    conf.setCoords(coords)
    writePDB(outpath, conf)
    return maxd

def parse_volume_line(line):
    s = line.strip()
    if s.startswith('Volume') and 'score' not in s.lower():
        p = s.split(':')
        if len(p) == 2:
            try: return float(p[1].strip())
            except: return None
    return None

def siip_volume_his95(pdbpath, radius=12.0):
    subprocess.run(['fpocket', '-f', pdbpath], capture_output=True, text=True)
    base = os.path.basename(pdbpath).replace('.pdb', '')
    outdir = pdbpath.replace('.pdb', '_out')
    info = os.path.join(outdir, f"{base}_info.txt")
    pockets_pdb = os.path.join(outdir, f"{base}_out.pdb")
    if not (os.path.exists(info) and os.path.exists(pockets_pdb)):
        return None, None
    conf = parsePDB(pdbpath)
    his95 = conf.select('resnum 95 and name CA')
    if his95 is None: return None, None
    h = his95.getCoords()[0]
    # count spheres of each pocket within radius of His95; pick pocket with
    # the MOST spheres near His95 (more stable than single nearest sphere)
    pcount = {}
    with open(pockets_pdb) as fh:
        for line in fh:
            if line.startswith('HETATM') and ' STP ' in line:
                try:
                    x=float(line[30:38]);y=float(line[38:46]);z=float(line[46:54])
                    pn=int(line[22:26])
                    if np.linalg.norm(np.array([x,y,z])-h) <= radius:
                        pcount[pn] = pcount.get(pn,0)+1
                except: pass
    if not pcount: return None, None
    sp = max(pcount, key=pcount.get)   # pocket best-represented near His95
    vols={}; cur=None
    with open(info) as fh:
        for line in fh:
            if line.strip().startswith('Pocket'):
                try: cur=int(line.split()[1])
                except: cur=None
            else:
                v=parse_volume_line(line)
                if v is not None and cur is not None: vols[cur]=v
    return sp, vols.get(sp)

print("conf | target CA-RMSD | max atom disp | SII-P vol (A^3) | intact")
print("(closed ~243 ; open target ~822)")
print("-"*66)
rows=[]
for k, tr in enumerate(target_rmsds):
    alpha = tr / rms_per_unit_alpha if rms_per_unit_alpha>0 else 0
    p = f'ladder/lad_{k:02d}.pdb'
    maxd = build_full(alpha, p)
    pn, vol = siip_volume_his95(p)
    intact = "yes" if maxd < 12 else "TORN"
    volstr = f"{vol:.1f}" if vol else "n/a"
    print(f"  {k:2d} |    {tr:4.2f}     |   {maxd:5.2f}      |  {volstr:>7}  | {intact}")
    rows.append((k, tr, maxd, pn, vol))

with open('ladder_openness.csv','w') as f:
    f.write("conf,target_ca_rmsd,max_atom_disp,siip_pocket,siip_volume\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]},{r[4] if r[4] else ''}\n")
print("\nSaved ladder_openness.csv + ladder/lad_*.pdb")