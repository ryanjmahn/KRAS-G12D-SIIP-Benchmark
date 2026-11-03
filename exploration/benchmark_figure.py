"""
Compile Phase-3 results into publication-style figures + tables.
Reads: phase3_corrected.csv (fpocket + P2Rank), phase3_pocketminer.csv,
       ladder_dense.csv (openness axis).
Produces: figures (PNG) + summary tables (printed + CSV).

Uses matplotlib. Openness axis = switch-II RMSD from closed.
"""
import csv, numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt

def read_csv(path):
    rows=[]
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def fnum(x):
    try: return float(x)
    except: return np.nan

# ---- load data ----
corr = read_csv('phase3_corrected.csv')      # fpocket + P2Rank (corrected scoring)
pm   = read_csv('phase3_pocketminer.csv')    # PocketMiner

# build aligned arrays by conf
def col(rows, key):
    return {int(r['conf']): fnum(r.get(key,'')) for r in rows if r.get('conf','')!=''}

openness = col(corr,'openness')
fp_rec = col(corr,'fp_recall'); fp_mcc = col(corr,'fp_mcc')
p2_rec = col(corr,'p2_recall'); p2_mcc = col(corr,'p2_mcc')
pm_rec = col(pm,'recall');      pm_mcc = col(pm,'mcc')

confs = sorted(openness.keys())
x   = [openness[c] for c in confs]
fpr = [fp_rec.get(c,np.nan) for c in confs]
p2r = [p2_rec.get(c,np.nan) for c in confs]
pmr = [pm_rec.get(c,np.nan) for c in confs]
fpm = [fp_mcc.get(c,np.nan) for c in confs]
p2m = [p2_mcc.get(c,np.nan) for c in confs]
pmm = [pm_mcc.get(c,np.nan) for c in confs]

# ============ FIGURE 1: Recall vs openness ============
plt.figure(figsize=(7,5))
plt.plot(x, fpr, 'o-', label='fpocket (geometry)', color='#888888')
plt.plot(x, p2r, 's-', label='P2Rank (general ML)', color='#2E75B6')
plt.plot(x, pmr, '^-', label='PocketMiner (cryptic specialist)', color='#C00000')
plt.xlabel('Pocket openness (switch-II RMSD from closed, Å)')
plt.ylabel('SII-P recall (fraction of true residues recovered)')
plt.title('Cryptic-pocket detection vs. openness')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('fig1_recall_vs_openness.png', dpi=200)
plt.close()

# ============ FIGURE 2: MCC vs openness ============
plt.figure(figsize=(7,5))
plt.plot(x, fpm, 'o-', label='fpocket', color='#888888')
plt.plot(x, p2m, 's-', label='P2Rank', color='#2E75B6')
plt.plot(x, pmm, '^-', label='PocketMiner', color='#C00000')
plt.xlabel('Pocket openness (switch-II RMSD from closed, Å)')
plt.ylabel('MCC (prediction quality)')
plt.title('Prediction quality vs. openness')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('fig2_mcc_vs_openness.png', dpi=200)
plt.close()

# ============ FIGURE 3: mean performance bar chart ============
def meanig(a): 
    arr=np.array([v for v in a if not np.isnan(v)])
    return arr.mean() if len(arr) else 0, (arr.std() if len(arr) else 0)
tools=['fpocket','P2Rank','PocketMiner']
rec_means=[meanig(fpr)[0],meanig(p2r)[0],meanig(pmr)[0]]
rec_sd=[meanig(fpr)[1],meanig(p2r)[1],meanig(pmr)[1]]
mcc_means=[meanig(fpm)[0],meanig(p2m)[0],meanig(pmm)[0]]
mcc_sd=[meanig(fpm)[1],meanig(p2m)[1],meanig(pmm)[1]]
xpos=np.arange(3); w=0.35
plt.figure(figsize=(7,5))
plt.bar(xpos-w/2, rec_means, w, yerr=rec_sd, label='Recall', color='#2E75B6', capsize=4)
plt.bar(xpos+w/2, mcc_means, w, yerr=mcc_sd, label='MCC', color='#C00000', capsize=4)
plt.xticks(xpos, tools); plt.ylabel('Mean value across ladder')
plt.title('Mean detection performance by tool (±SD)')
plt.legend(); plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig('fig3_mean_performance.png', dpi=200)
plt.close()

# ============ SUMMARY TABLE ============
print("="*66)
print("SUMMARY TABLE — mean performance across the 24-conformer ladder")
print("="*66)
print(f"{'Tool':<14}{'Mean recall':>14}{'Mean MCC':>12}{'Recall(open end)':>18}")
def openend(a): 
    vals=[v for v in a[-6:] if not np.isnan(v)]  # last 6 = most open
    return np.mean(vals) if vals else 0
print(f"{'fpocket':<14}{meanig(fpr)[0]:>14.2f}{meanig(fpm)[0]:>12.2f}{openend(fpr):>18.2f}")
print(f"{'P2Rank':<14}{meanig(p2r)[0]:>14.2f}{meanig(p2m)[0]:>12.2f}{openend(p2r):>18.2f}")
print(f"{'PocketMiner':<14}{meanig(pmr)[0]:>14.2f}{meanig(pmm)[0]:>12.2f}{openend(pmr):>18.2f}")

with open('results_summary.csv','w') as f:
    f.write("tool,mean_recall,mean_mcc,recall_open_end\n")
    f.write(f"fpocket,{meanig(fpr)[0]:.3f},{meanig(fpm)[0]:.3f},{openend(fpr):.3f}\n")
    f.write(f"P2Rank,{meanig(p2r)[0]:.3f},{meanig(p2m)[0]:.3f},{openend(p2r):.3f}\n")
    f.write(f"PocketMiner,{meanig(pmr)[0]:.3f},{meanig(pmm)[0]:.3f},{openend(pmr):.3f}\n")

print("\nSaved figures: fig1_recall_vs_openness.png, fig2_mcc_vs_openness.png, fig3_mean_performance.png")
print("Saved table: results_summary.csv")