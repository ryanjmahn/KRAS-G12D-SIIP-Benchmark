import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import srcpath  # noqa: F401
from paths import STRUCTURES
from build_ladder_dense import stable_vol_and_rmsd

for tag, pdb in [("5US4", str(STRUCTURES / "5US4_H.pdb")), ("7RPZ", str(STRUCTURES / "7RPZ_H.pdb"))]:
    out = stable_vol_and_rmsd(pdb)
    print(tag, out)