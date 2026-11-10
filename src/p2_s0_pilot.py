"""
P2 s0 (feasibility checkpoint): does any cheap MD-free ensemble carry per-residue signal a model could use?

Phase 1 found the lowest normal mode no better than random motion (docs/evaluation_protocol.md §8). Before building
features and models, this pilot asks the go/no-go question on TRAINING-FOLD targets only (test split untouched):
does aggregating a tool's per-residue scores over an ensemble beat the same tool on the apo alone, and beat the
same aggregation over a random-motion control?

Three generators, N_CONF conformers each, identical amplitude schedule (k-th conformer at global Cα RMSD
MAX_RMSD*(k+1)/N_CONF, no residue moved > MAX_DISP):
  nma1   lowest ANM mode, alternating sign (the Phase 1 ladder, thinned)
  nmaK   random thermal mix of the K lowest ANM modes (coefficients ~ N(0, 1/eigenvalue)), new mix per conformer
  random random smooth motion (p1_s4_ensembles.random_smooth_mode), new seed per conformer
Tools: fpocket, P2Rank, AE-PocketMiner (pLM-NN is sequence-only). Per residue, per tool: apo score, ensemble
max and mean (apo included). Metric: per-target AUPRC (threshold-free, p1_s3_score.threshold_free).
Go if nmaK (or nma1) aggregation beats apo AND beats random aggregation (paired Wilcoxon over targets).

Usage (structprep env):
  python src/p2_s0_pilot.py build
  for g in nma1 nmaK random; do python src/p1_s2_run_baselines.py --root runs/p2_pilot/$g --log-suffix _p2pilot_$g; done
  python src/p2_s0_pilot.py score
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from prody import ANM, confProDy, parsePDB, writePDB
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, RUNS
from p1_s4_ensembles import MAX_DISP, MAX_RMSD, random_smooth_mode

confProDy(verbosity="none")
P1, OUT = RUNS / "p1", RUNS / "p2_pilot"
GENS = ("nma1", "nmaK", "random")
N_TARGETS, N_CONF, K_MODES, SEED = 100, 10, 10, 0
SCALE = 1.0                 # multiplies the amplitude schedule; p2_s2_scale --scale sets it
TRAIN = [f"train-fold-{i}" for i in range(4)]


def sample():
    ids = []
    for m in sorted(P1.glob("*/meta.json")):
        d = json.loads(m.read_text())
        if d["split"] in TRAIN and d.get("usable", True):
            ids.append(d["target_id"])
    return sorted(np.random.default_rng(SEED).choice(ids, N_TARGETS, replace=False))


def directions(gen, ca, tid):
    """N_CONF unit-free displacement fields (n_ca x 3), one per conformer."""
    rng = np.random.default_rng(zlib.crc32(f"{tid}/{gen}".encode()))
    if gen == "random":
        return [random_smooth_mode(ca, zlib.crc32(f"{tid}/{k}".encode())) for k in range(N_CONF)]
    anm = ANM(tid)
    anm.buildHessian(ca)
    anm.calcModes(n_modes=K_MODES)
    vecs, vals = anm.getEigvecs(), anm.getEigvals()
    if gen == "nma1":
        return [vecs[:, 0].reshape(-1, 3) * (1 if k % 2 == 0 else -1) for k in range(N_CONF)]
    return [(vecs @ rng.normal(0, 1 / np.sqrt(vals))).reshape(-1, 3) for k in range(N_CONF)]


def build_target(tid):
    src = P1 / tid
    meta = json.loads((src / "meta.json").read_text())
    full = parsePDB(str(src / f"{tid}.pdb"))
    ca = full.select("protein and name CA")
    index = {(c, n, i): j for j, (c, n, i) in enumerate(zip(ca.getChids(), ca.getResnums(), ca.getIcodes()))}
    atom_idx = np.array([index.get(k, -1) for k in zip(full.getChids(), full.getResnums(), full.getIcodes())])
    moving = atom_idx >= 0
    rows = []
    for gen in GENS:
        for k, v in enumerate(directions(gen, ca.getCoords(), tid)):
            unit = np.sqrt((v ** 2).sum(1).mean())
            amp = min(MAX_RMSD * SCALE * (k + 1) / N_CONF,
                      MAX_DISP * SCALE * unit / np.linalg.norm(v, axis=1).max())
            disp = v * (amp / unit)
            coords = full.getCoords().copy()
            coords[moving] += disp[atom_idx[moving]]
            conf = full.copy()
            conf.setCoords(coords)
            cid = f"{tid}__c{k:02d}"
            d = OUT / gen / cid
            d.mkdir(parents=True, exist_ok=True)
            writePDB(str(d / f"{cid}.pdb"), conf)
            (d / "meta.json").write_text(json.dumps(meta | dict(target_id=cid, parent=tid, conf=k, split="p2_pilot")))
            rows.append(dict(gen=gen, parent=tid, conf=k, rmsd=amp, max_disp=float(np.linalg.norm(disp, axis=1).max())))
    return rows


def tool_scores(root, cid, keys):
    """{tool: per-residue score array} for one prepared structure under root (runs/p1 layout)."""
    from p1_adapters import fpocket, p2rank_residue_scores, per_residue_probs, pocket_residue_scores
    out = {}
    if (root / cid / f"{cid}_out" / f"{cid}_info.txt").exists():
        out["fpocket"] = pocket_residue_scores(fpocket(root / cid / f"{cid}_out"), keys)
    if (root / "p2rank_out" / f"{cid}.pdb_residues.csv").exists():
        out["P2Rank"] = p2rank_residue_scores(root / "p2rank_out" / f"{cid}.pdb_residues.csv", keys)
    if (root / "pocketminer_out" / f"{cid}-preds.npy").exists():
        out["AE-PocketMiner"] = per_residue_probs(root / "pocketminer_out" / f"{cid}-preds.npy", keys)
    return {t: np.array([s[k] for k in keys]) for t, s in out.items()}


def score_target(tid):
    from p1_adapters import ca_table
    from p1_s3_score import threshold_free
    meta = json.loads((P1 / tid / "meta.json").read_text())
    keys, _ = ca_table(P1 / tid / f"{tid}.pdb")
    truth = set(meta["truth"]) & set(keys)
    apo = tool_scores(P1, tid, keys)
    rows = []
    for tool, a in apo.items():
        auprc = lambda s: threshold_free(dict(zip(keys, s)), truth, keys)["auprc"]
        rows.append(dict(parent=tid, tool=tool, agg="apo", gen="-", auprc=auprc(a)))
        for gen in GENS:
            confs = [tool_scores(OUT / gen, f"{tid}__c{k:02d}", keys).get(tool) for k in range(N_CONF)]
            if any(c is None for c in confs):
                continue
            stack = np.vstack([a] + confs)
            for agg, s in (("max", stack.max(0)), ("mean", stack.mean(0))):
                rows.append(dict(parent=tid, tool=tool, agg=agg, gen=gen, auprc=auprc(s)))
    return rows


def paired(w, a, b):
    d = (w[a] - w[b]).dropna()
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    return dict(comparison=f"{a} - {b}", n=len(d), mean_diff=d.mean(), median_diff=d.median(),
                a_wins=int((d > 0).sum()), b_wins=int((d < 0).sum()), p=p)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    ids = sample()
    if step == "build":
        rows = [r for n, tid in enumerate(ids, 1) for r in build_target(tid)]
        pd.DataFrame(rows).to_csv(RESULTS / "p2_pilot_ensembles.csv", index=False)
        print(f"{len(rows)} conformers for {len(ids)} training targets -> {OUT}")
    elif step == "score":
        r = pd.DataFrame([x for tid in ids for x in score_target(tid)])
        r.to_csv(RESULTS / "p2_pilot.csv", index=False)
        r["arm"] = np.where(r["agg"] == "apo", "apo", r.gen + "_" + r["agg"])
        tests = []
        for tool, g in r.groupby("tool"):
            w = g.pivot(index="parent", columns="arm", values="auprc")
            for agg in ("max", "mean"):
                for gen in ("nma1", "nmaK"):
                    tests += [dict(tool=tool) | paired(w, f"{gen}_{agg}", "apo"),
                              dict(tool=tool) | paired(w, f"{gen}_{agg}", f"random_{agg}")]
                tests.append(dict(tool=tool) | paired(w, f"random_{agg}", "apo"))
        t = pd.DataFrame(tests)
        t.to_csv(RESULTS / "p2_pilot_tests.csv", index=False)
        print("mean AUPRC per tool and arm:")
        print(r.pivot_table(index="tool", columns="arm", values="auprc").round(3).to_string())
        print("\npaired Wilcoxon over targets:")
        print(t.round(4).to_string(index=False))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
