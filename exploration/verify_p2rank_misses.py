"""
Verify P2Rank 'misses' are real detection failures, not cutoff artifacts.
For each missed conformer, show every predicted pocket, its distance to
His95, and whether its residues overlap the true SII-P.
"""
from prody import *
import numpy as np, csv, os

TRUE_PRESENT=[9,10,11,12,16,58,59,65,68,69,72,78,88,92,95,96,99,100,102,103]

# the conformers that came back as misses
MISSED = [0, 8, 9, 16, 17, 18, 19]

def check(conf_id):
    pdb=f'ladder_dense/dense_{conf_id:02d}.pdb'
    csvp=f'p2rank_out/dense_{conf_id:02d}/dense_{conf_id:02d}.pdb_predictions.csv'
    conf=parsePDB(pdb)
    his=conf.select('resnum 95 and name CA')
    if his is None:
        print(f"conf {conf_id}: no His95?"); return
    h=his.getCoords()[0]
    if not os.path.exists(csvp):
        print(f"conf {conf_id}: NO predictions csv (P2Rank didn't run)"); return
    print(f"\n=== conf {conf_id} ===")
    with open(csvp) as f:
        reader=csv.reader(f)
        header=[x.strip() for x in next(reader)]
        def col(n):
            for i,x in enumerate(header):
                if x.strip()==n: return i
            return None
        ci_x=col('center_x');ci_y=col('center_y');ci_z=col('center_z')
        ci_res=col('residue_ids');ci_rank=col('rank')
        for parts in reader:
            if len(parts)<len(header): continue
            try:
                rank=int(parts[ci_rank])
                c=np.array([float(parts[ci_x]),float(parts[ci_y]),float(parts[ci_z])])
                dist=np.linalg.norm(c-h)
                resnums=set()
                for tok in parts[ci_res].split():
                    if '_' in tok:
                        try: resnums.add(int(tok.split('_')[1]))
                        except: pass
                overlap = resnums & set(TRUE_PRESENT)
                print(f"  pocket rank {rank}: {dist:.1f} A from His95, "
                      f"true-residue overlap: {sorted(overlap) if overlap else 'NONE'}")
            except: pass

# check a closed miss and open misses
for cid in MISSED:
    check(cid)

print("\n\nINTERPRETATION:")
print("- If NO pocket is close to His95 AND overlaps true residues -> real miss.")
print("- If a pocket IS near His95 / overlaps but was >14 A -> cutoff too tight.")