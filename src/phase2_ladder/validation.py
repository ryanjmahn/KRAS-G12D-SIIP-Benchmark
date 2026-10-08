import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LADDER_CSV
d = pd.read_csv(LADDER_CSV)
print(d[["switch2_rmsd","siip_volume"]].corr(method="spearman"))
