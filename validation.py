import pandas as pd
d = pd.read_csv("ladder_dense.csv")
print(d[["switch2_rmsd","siip_volume"]].corr(method="spearman"))
