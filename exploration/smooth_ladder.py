"""
Re-measure SII-P openness MORE STABLY from the already-generated ladder
conformers (ladder/lad_*.pdb). No regeneration needed.

Fix for the noise: instead of "pocket with most spheres near His95" (which
flips between a small adjacent cavity and the main opening pocket), we take
the LARGEST pocket that has ANY alpha-sphere within `radius` of His95.
The main opening pocket dominates once open, so this tracks the real opening
and ignores tiny adjacent cavities.
"""
import numpy as np
import os, subprocess, glob

def parse_volume_line(line):
    s = line.strip()
    if s.startswith('Volume') and 'score' not in s.lower():
        p = s.split(':')
        if len(p) == 2:
            try: return float(p[1].strip())
            except: return None
    return None

def stable_siip_volume(pdbpath, radius=14.0):
    subprocess.run(['fpocket', '-f', pdbpath], capture_output=True, text=True)
    base = os.path.basename(pdbpath).replace('.pdb', '')
    outdir = pdbpath.replace('.pdb', '_out')
    info = os.path.join(outdir, f"{base}_info.txt")
    pockets_pdb = os.path.join(outdir, f"{base}_out.pdb")
    if not (os.path.exists(info) and os.path.exists(pockets_pdb)):
        return None
    # His95 CA
    from prody import parsePDB
    conf = parsePDB(pdbpath)
    his = conf.select('resnum 95 and name CA')
    if his is None: return None
    h = his.getCoords()[0]
    # which pockets have >=1 sphere within radius of His95?
    near = set()
    with open(pockets_pdb) as fh:
        for line in fh:
            if line.startswith('HETATM') and ' STP ' in line:
                try:
                    x=float(line[30:38]);y=float(line[38:46]);z=float(line[46:54])
                    pn=int(line[22:26])
                    if np.linalg.norm(np.array([x,y,z])-h) <= radius:
                        near.add(pn)
                except: pass
    if not near: return None
    # read volumes; return the LARGEST among pockets near His95
    vols={}; cur=None
    with open(info) as fh:
        for line in fh:
            if line.strip().startswith('Pocket'):
                try: cur=int(line.split()[1])
                except: cur=None
            else:
                v=parse_volume_line(line)
                if v is not None and cur is not None: vols[cur]=v
    near_vols = [vols[p] for p in near if p in vols]
    return max(near_vols) if near_vols else None

# process the intact conformers (lad_00 .. lad_14; 15-17 were torn)
files = sorted(glob.glob('ladder/lad_*.pdb'))
print("conf | stable SII-P volume (A^3)")
print("-"*38)
rows=[]
# rough target CA-RMSD per conf (from generation: linspace 0..2.2 over 18)
trs = np.linspace(0.0, 2.2, 18)
for i, f in enumerate(files):
    v = stable_siip_volume(f)
    tr = trs[i] if i < len(trs) else None
    vs = f"{v:.1f}" if v else "n/a"
    print(f"  {i:2d} | {vs}")
    rows.append((i, tr, v))

with open('ladder_openness_smoothed.csv','w') as fh:
    fh.write("conf,target_ca_rmsd,siip_volume\n")
    for i,tr,v in rows:
        fh.write(f"{i},{(tr if tr is not None else 0):.3f},{v if v else ''}\n")
print("\nSaved ladder_openness_smoothed.csv")