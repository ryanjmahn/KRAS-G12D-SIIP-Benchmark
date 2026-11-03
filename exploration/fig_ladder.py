"""
FIGURE — Ladder validation / cryptic-pocket collapse.
Shows SII-P pocket volume AND switch-II RMSD across the 24-conformer ladder,
demonstrating the smooth closed->open gradient (the openness axis is real).
Reads ladder_dense.csv.
"""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

conf=[]; rmsd=[]; vol=[]
with open('ladder_dense.csv') as f:
    for row in csv.DictReader(f):
        conf.append(int(row['conf']))
        rmsd.append(float(row['switch2_rmsd']) if row['switch2_rmsd'] else 0.0)
        vol.append(float(row['siip_volume']) if row['siip_volume'] else np.nan)

# sort by conf
order=np.argsort(conf)
rmsd=np.array(rmsd)[order]; vol=np.array(vol)[order]

fig, ax1 = plt.subplots(figsize=(7.5,5))

# left axis: pocket volume vs switch-II RMSD (openness)
color1='#2E75B6'
ax1.plot(rmsd, vol, 'o-', color=color1, markersize=6)
ax1.set_xlabel('Switch-II RMSD from closed state (Å)  →  increasing openness')
ax1.set_ylabel('SII-P pocket volume (Å³)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(alpha=0.3)

# shade the three regimes
ax1.axhspan(0, 330, alpha=0.06, color='gray')      # closed band
ax1.axhspan(330, 550, alpha=0.06, color='orange')  # transition band
ax1.axhspan(550, 900, alpha=0.06, color='green')   # open band

# reference lines for real anchor volumes
ax1.axhline(822, ls='--', color='green', alpha=0.6, lw=1)
ax1.text(0.05, 835, 'true open (7RPZ) = 822 Å³', color='green', fontsize=8)
ax1.axhline(245, ls='--', color='gray', alpha=0.6, lw=1)
ax1.text(0.05, 200, 'true closed (5US4) ≈ 245 Å³', color='gray', fontsize=8)

# regime labels
ax1.text(0.15, 300, 'CLOSED', fontsize=9, color='gray', weight='bold')
ax1.text(1.3, 470, 'TRANSITION', fontsize=9, color='#B45F06', weight='bold')
ax1.text(2.6, 700, 'OPEN', fontsize=9, color='green', weight='bold')

plt.title('Conformational ladder: cryptic SII-P opens under MD-free motion')
plt.tight_layout()
plt.savefig('fig_ladder_validation.png', dpi=200)
plt.close()
print("Saved fig_ladder_validation.png")
print(f"Ladder spans switch-II RMSD {rmsd.min():.2f}–{rmsd.max():.2f} Å, "
      f"volume {np.nanmin(vol):.0f}–{np.nanmax(vol):.0f} Å³")