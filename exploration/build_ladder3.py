"""
BUILD THE CONFORMATIONAL LADDER (Phase 2) - PROPER version.
Uses ProDy traverseMode with physically-sensible RMSD amplitudes so the
structure stays intact (no loops flying off).

- Built from 5US4 (CLOSED) only. 7RPZ never referenced.
- traverseMode moves along Mode 1 with controlled RMSD steps.
- Measures SII-P volume (His95-identified) per conformer.
"""
from prody import *
import numpy as np
import os, subprocess

os.makedirs('ladder', exist_ok=True)

# ---- closed structure + ANM ----
full = parsePDB('5US4_H.pdb')
calphas = full.select('protein and name CA')
anm = ANM('5US4'); anm.buildHessian(calphas); anm.calcModes(n_modes=10)

# ---- traverse Mode 1 with sensible amplitude ----
# traverseMode returns an Ensemble of conformers along the mode.
# n_steps controls how many frames each direction; rmsd sets the max
# CA-RMSD of the extreme frame from the start (keep modest so structure
# stays physical). We found Mode1 MINUS opens; traverse gives both dirs.
mode1 = anm[0]

# generate conformers: 10 steps each direction, max RMSD ~4 A at the extreme
ens = traverseMode(mode1, calphas, n_steps=10, rmsd=4.0)
print("traverseMode produced", ens.numConfs(), "conformers (CA level)")

# ens frames are CA coordinates along the mode. We want the MINUS direction
# (opening). traverseMode order: goes from -rmsd ... 0 ... +rmsd.
# We'll map each CA frame back onto the full structure via CA displacement.
coordsets = ens.getCoordsets()   # shape (nconf, nCA, 3)
ca_start = calphas.getCoords()
ca_resnums = calphas.getResnums()

def build_full(ca_new, outpath):
    """Apply CA displacement (ca_new - ca_start) to all atoms by residue."""
    disp = {rn: ca_new[i] - ca_start[i] for i, rn in enumerate(ca_resnums)}
    conf = full.copy()
    coords = conf.getCoords().copy()
    resnums = conf.getResnums()
    for i in range(conf.numAtoms()):
        d = disp.get(resnums[i])
        if d is not None:
            coords[i] = coords[i] + d
    conf.setCoords(coords)
    writePDB(outpath, conf)

def parse_volume_line(line):
    s = line.strip()
    if s.startswith('Volume') and 'score' not in s.lower():
        parts = s.split(':')
        if len(parts) == 2:
            try: return float(parts[1].strip())
            except: return None
    return None

def siip_volume_his95(pdbpath):
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
    pmin = {}
    with open(pockets_pdb) as fh:
        for line in fh:
            if line.startswith('HETATM') and ' STP ' in line:
                try:
                    x=float(line[30:38]);y=float(line[38:46]);z=float(line[46:54])
                    pn=int(line[22:26]); d=np.linalg.norm(np.array([x,y,z])-h)
                    if pn not in pmin or d<pmin[pn]: pmin[pn]=d
                except: pass
    if not pmin: return None, None
    sp = min(pmin, key=pmin.get)
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

# measure CA-RMSD of each frame from start, so we have a clean openness-independent x
print("\nframe | CA-RMSD from closed | SII-P vol (A^3) | intact?")
print("(closed SII-P ~250-338 ; open target ~822)")
print("-"*60)
rows=[]
for i in range(coordsets.shape[0]):
    ca_new = coordsets[i]
    rmsd = np.sqrt(np.mean(np.sum((ca_new - ca_start)**2, axis=1)))
    p = f'ladder/frame_{i:02d}.pdb'
    build_full(ca_new, p)
    pn, vol = siip_volume_his95(p)
    # intact check: max atom displacement shouldn't be crazy (>15 A = torn)
    maxdisp = np.max(np.linalg.norm(ca_new - ca_start, axis=1))
    intact = "yes" if maxdisp < 15 else "TORN"
    volstr = f"{vol:.1f}" if vol else "n/a"
    print(f"  {i:2d}  |  {rmsd:5.2f}  |  {volstr:>7}  |  {intact}")
    rows.append((i, rmsd, pn, vol, maxdisp))

with open('ladder_openness.csv','w') as f:
    f.write("frame,ca_rmsd,siip_pocket,siip_volume,max_disp\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]:.3f},{r[2]},{r[3] if r[3] else ''},{r[4]:.2f}\n")
print("\nSaved ladder_openness.csv + ladder/frame_*.pdb")