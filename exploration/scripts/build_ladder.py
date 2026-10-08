"""
BUILD THE CONFORMATIONAL LADDER (Phase 2) - fixed volume parsing.
Fix: match only the exact 'Volume :' line, not 'Volume score:'.
"""
from prody import *
import numpy as np
import os, subprocess

os.makedirs('ladder', exist_ok=True)

full = parsePDB('5US4_H.pdb')
calphas = full.select('protein and name CA')
anm = ANM('5US4'); anm.buildHessian(calphas); anm.calcModes(n_modes=10)
evec = anm[0].getEigvec().reshape(-1, 3)
ca_resnums = calphas.getResnums()
ca_map = {rn: evec[i] for i, rn in enumerate(ca_resnums)}
SIGN = -1
scales = list(range(0, 115, 5))

def make_conformer(scale, outpath):
    conf = full.copy()
    coords = conf.getCoords().copy()
    resnums = conf.getResnums()
    for i in range(conf.numAtoms()):
        d = ca_map.get(resnums[i])
        if d is not None:
            coords[i] = coords[i] + SIGN * scale * d
    conf.setCoords(coords)
    writePDB(outpath, conf)

def parse_volume_line(line):
    """Return the volume ONLY from a line like 'Volume : 822.335'
    (NOT 'Volume score: 4.174')."""
    s = line.strip()
    # must start with 'Volume' AND the next token be ':' (not 'score')
    if s.startswith('Volume') and 'score' not in s.lower():
        # format: 'Volume :   822.335'
        parts = s.split(':')
        if len(parts) == 2:
            try:
                return float(parts[1].strip())
            except:
                return None
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
    if his95 is None:
        return None, None
    h_coord = his95.getCoords()[0]

    pocket_min_dist = {}
    with open(pockets_pdb) as fh:
        for line in fh:
            if line.startswith('HETATM') and ' STP ' in line:
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    pnum = int(line[22:26])
                    d = np.linalg.norm(np.array([x, y, z]) - h_coord)
                    if pnum not in pocket_min_dist or d < pocket_min_dist[pnum]:
                        pocket_min_dist[pnum] = d
                except:
                    pass
    if not pocket_min_dist:
        return None, None
    siip_pocket = min(pocket_min_dist, key=pocket_min_dist.get)

    # read that pocket's volume (fixed parser)
    vols = {}
    cur = None
    with open(info) as fh:
        for line in fh:
            if line.strip().startswith('Pocket'):
                try:
                    cur = int(line.split()[1])
                except:
                    cur = None
            else:
                v = parse_volume_line(line)
                if v is not None and cur is not None:
                    vols[cur] = v
    return siip_pocket, vols.get(siip_pocket)

print("scale | SII-P pocket# | SII-P volume (A^3)")
print("(closed ~245-338 ; open target ~822)")
print("-" * 45)
rows = []
for s in scales:
    p = f'ladder/conf_{s:03d}.pdb'
    make_conformer(s, p)
    pnum, vol = siip_volume_his95(p)
    volstr = f"{vol:.1f}" if vol else "n/a"
    print(f"  {s:3d}  |  {str(pnum):>4}  |  {volstr}")
    rows.append((s, pnum, vol))

with open('ladder_openness.csv', 'w') as f:
    f.write("scale,siip_pocket,siip_volume\n")
    for s, pnum, vol in rows:
        f.write(f"{s},{pnum},{vol if vol else ''}\n")
print("\nSaved ladder_openness.csv and ladder/ conformers.")