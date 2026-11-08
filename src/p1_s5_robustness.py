"""
P1 s5: conformational robustness of every tool across the curated-subset ensembles (docs/evaluation_protocol.md §8).

Two ladders per target (p1_s4_ensembles.py): "nma" (lowest normal mode) and "null" (random smooth motion, same
amplitudes and cap). Each scored by p1_s3_score.py into results/p1_master_ensemble{,_null}.csv.
Per tool per target per ladder, over 25 conformers (primary variant; pocket tools also top-3):
  det_rate      share of conformers with something at the site
  flips         changes of that yes/no along the ordered ladder (instability of detection)
  mcc_apo / mcc_mean / mcc_sd / mcc_best    MCC on the apo, across the ladder, its spread and best conformer
  headroom      mcc_best - mcc_apo. On the null ladder this is selection luck plus generic sensitivity,
                so nma - null is the part attributable to the normal mode itself
  rho_open      Spearman of MCC with pocket closeness to holo (-RMSD); only where the ladder moves the
                pocket >= MOVE_MIN Å toward holo
  rho_volume    Spearman of MCC with the fpocket site volume (largest pocket within 8 Å of the ligand);
                fpocket-dependent axis, reported next to the tool-independent one
Subtle openings (G4): targets in the lowest third of apo->holo pocket RMSD.
Outputs: results/p1_robustness.csv (ladder x tool x target), results/p1_robustness_null_test.csv, printed summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, RUNS
from p1_adapters import fpocket

APO_CONF = 12            # middle of the 25-conformer ladder = the unperturbed apo
MOVE_MIN = 0.25          # Å the ladder must bring the pocket toward holo before a trend is measured
VOLUME_GATE = 8.0        # Å; pockets this close to a ligand atom count toward the site volume
VARIANTS = {"fpocket": ["gate4", "gate4_top3"], "P2Rank": ["gate4", "gate4_top3"],
            "AE-PocketMiner": ["t0.7"], "pLM-NN": ["t0.75"]}
LADDERS = {"nma": ("p1_ens", ""), "null": ("p1_ens_null", "_null")}


def site_volumes(root, ids):
    """Largest fpocket pocket volume near the ligand, per conformer (0 if none)."""
    out = {}
    for cid in ids:
        lig = np.array(json.loads((root / cid / "meta.json").read_text())["ligand_xyz"])
        vols = [p["volume"] for p in fpocket(root / cid / f"{cid}_out")
                if np.linalg.norm(lig - p["center"], axis=1).min() <= VOLUME_GATE]
        out[cid] = max(vols, default=0.0)
    return out


def spear(x, y):
    return spearmanr(x, y).statistic if np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan


def per_target(g):
    g = g.sort_values("conf")
    hit = g.site_hit.astype(bool).to_numpy()
    mcc = g.mcc.to_numpy()
    apo = float(g[g.conf == APO_CONF].mcc.iloc[0])
    approach = g.pocket_rmsd_holo.iloc[APO_CONF] - g.pocket_rmsd_holo.min()
    return dict(det_rate=hit.mean(), flips=int((hit[1:] != hit[:-1]).sum()), mcc_apo=apo, mcc_mean=mcc.mean(),
                mcc_sd=mcc.std(), mcc_best=mcc.max(), headroom=mcc.max() - apo, approach_to_holo=approach,
                apo_pocket_rmsd=g.pocket_rmsd_holo.iloc[APO_CONF], pocket_disp_max=g.pocket_disp.max(),
                rho_open=spear(mcc, -g.pocket_rmsd_holo.to_numpy()) if approach >= MOVE_MIN else np.nan,
                rho_volume=spear(mcc, g.site_volume.to_numpy()),
                rho_axes=spear(g.site_volume.to_numpy(), -g.pocket_rmsd_holo.to_numpy()))


def ladder_rows(name, folder, sfx):
    m = pd.read_csv(RESULTS / f"p1_master_ensemble{sfx}.csv")
    ens = pd.read_csv(RESULTS / f"p1_ensembles{sfx}.csv")
    ens["site_volume"] = ens.target_id.map(site_volumes(RUNS / folder, ens.target_id))
    m = m.merge(ens[["target_id", "parent", "conf", "pocket_rmsd_holo", "pocket_disp", "site_volume"]], on="target_id")
    rows = []
    for tool, variants in VARIANTS.items():
        for v in variants:
            for parent, g in m[(m.tool == tool) & (m.variant == v)].groupby("parent"):
                rows.append(dict(ladder=name, tool=tool, variant=v, parent=parent) | per_target(g))
    return rows


def main():
    r = pd.DataFrame([x for name, (folder, sfx) in LADDERS.items() for x in ladder_rows(name, folder, sfx)])
    cut = r[r.ladder == "nma"].groupby("parent").apo_pocket_rmsd.first().quantile(1 / 3)
    r["subtle"] = r.apo_pocket_rmsd <= cut
    r.to_csv(RESULTS / "p1_robustness.csv", index=False)

    nma = r[r.ladder == "nma"]
    moving = nma[nma.approach_to_holo >= MOVE_MIN].parent.nunique()
    print(f"{nma.parent.nunique()} targets x 25 conformers per ladder; NMA ladder brings the pocket >= {MOVE_MIN} Å "
          f"toward holo on {moving} targets; subtle = apo pocket RMSD to holo <= {cut:.2f} Å")
    disp = r.groupby("ladder").pocket_disp_max.median()
    print(f"pocket displacement at the ladder ends (median): nma {disp['nma']:.2f} Å, null {disp['null']:.2f} Å\n")

    s = r.groupby(["tool", "variant", "ladder"]).agg(
        det_rate=("det_rate", "median"), unstable=("flips", lambda x: (x > 0).mean()),
        mcc_sd=("mcc_sd", "median"), headroom=("headroom", "mean"), gains=("headroom", lambda x: (x >= 0.1).mean()),
        rho_open=("rho_open", "median"), rho_volume=("rho_volume", "median")).unstack("ladder")
    print("per tool, NMA vs null ladder: unstable = share of targets whose detection flips; headroom = mean (best - apo MCC);\n"
          "gains = share with a conformer >= 0.1 MCC above the apo; rho_volume = median Spearman MCC vs fpocket site volume")
    print(s[["unstable", "headroom", "gains", "mcc_sd", "rho_volume"]].round(3).to_string())

    tests = []
    for (tool, v), g in r.groupby(["tool", "variant"]):
        w = g.pivot(index="parent", columns="ladder", values=["headroom", "flips"])
        for met in ("headroom", "flips"):
            d = (w[met]["nma"] - w[met]["null"]).dropna()
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            tests.append(dict(tool=tool, variant=v, metric=met, nma_mean=w[met]["nma"].mean(),
                              null_mean=w[met]["null"].mean(), nma_higher=int((d > 0).sum()),
                              null_higher=int((d < 0).sum()), p=p))
    t = pd.DataFrame(tests)
    t.to_csv(RESULTS / "p1_robustness_null_test.csv", index=False)
    print("\nNMA vs null, paired over targets (Wilcoxon):")
    print(t.round(3).to_string(index=False))

    axes = nma.drop_duplicates("parent").rho_axes
    print(f"\nagreement of the two openness axes (Spearman fpocket site volume vs closeness to holo), NMA ladder: "
          f"median {axes.median():.2f}, {int((axes > 0.5).sum())}/{axes.notna().sum()} targets > 0.5")
    sub = nma.groupby(["tool", "variant", "subtle"]).agg(mcc_apo=("mcc_apo", "mean"), headroom=("headroom", "mean")).unstack()
    print("\nNMA ladder, subtle vs larger openings (mean MCC on the apo, mean headroom):")
    print(sub.round(3).to_string())


if __name__ == "__main__":
    main()
