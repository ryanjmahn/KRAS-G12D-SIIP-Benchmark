"""
Sanity gate 1a: the original KRAS scripts reproduce Table I (results/anchors_all.csv) from scratch.

Re-runs fpocket and P2Rank on the two crystal structures (old outputs deleted first), then
score_anchors.py, and checks every number against Table I within 0.01.
AE-PocketMiner predictions are reused from ae-pocketminer/results/pocketminer (TF env; see SETUP.md).

Run (structprep env, Java 17): python tests/test_kras_reproduces.py
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import P2RANK_ANCHOR_OUT, PRANK, RESULTS, STRUCTURES

# Table I as published: (structure, tool) -> (precision, recall, mcc, n_predicted)
TABLE_I = {
    ("5US4", "P2Rank"): (0.000, 0.000, 0.000, 0),
    ("5US4", "fpocket"): (0.538, 0.280, 0.318, 13),
    ("5US4", "AE-PocketMiner"): (0.163, 0.280, 0.024, 43),
    ("7RPZ", "P2Rank"): (1.000, 0.720, 0.828, 18),
    ("7RPZ", "fpocket"): (1.000, 0.920, 0.953, 23),
    ("7RPZ", "AE-PocketMiner"): (0.222, 0.400, 0.125, 45),
}
TOL = 0.01


def regenerate():
    for tag in ("5US4", "7RPZ"):
        pdb = STRUCTURES / f"{tag}_H.pdb"
        shutil.rmtree(STRUCTURES / f"{tag}_H_out", ignore_errors=True)
        # fpocket segfaults on an absolute path containing a space (this repo's parent folder has one)
        subprocess.run(["fpocket", "-f", pdb.name], cwd=pdb.parent, check=True, capture_output=True)
        out = P2RANK_ANCHOR_OUT / f"p2rank_{tag}"
        shutil.rmtree(out, ignore_errors=True)
        subprocess.run([str(PRANK), "predict", "-f", str(pdb), "-o", str(out)], check=True, capture_output=True)
        assert (out / f"{tag}_H.pdb_predictions.csv").exists(), f"P2Rank wrote nothing for {tag} (Java 17 on PATH?)"


def check():
    subprocess.run([sys.executable, str(ROOT / "src/phase3_benchmark/score_anchors.py")], check=True, capture_output=True)
    got = {(r["structure"], r["tool"]): r for r in csv.DictReader(open(RESULTS / "anchors_all.csv"))}
    bad = []
    for key, (prec, rec, mcc, n) in TABLE_I.items():
        r = got[key]
        for name, want, have in (("precision", prec, float(r["precision"])), ("recall", rec, float(r["recall"])),
                                 ("mcc", mcc, float(r["mcc"]))):
            if abs(want - have) > TOL:
                bad.append(f"{key} {name}: Table I {want}, now {have}")
        if int(r["n_predicted"]) != n:
            bad.append(f"{key} n_predicted: Table I {n}, now {r['n_predicted']}")
    assert not bad, "KRAS does not reproduce:\n  " + "\n  ".join(bad)


def unified():
    """Gate 1b: KRAS re-scored under docs/evaluation_protocol.md, side by side with Table I."""
    from prody import parsePDB
    from p1_adapters import (ca_table, fpocket, p2rank, p2rank_residue_scores, per_residue_probs,
                             pocket_residue_scores)
    from p1_s3_score import score_pockets, score_probs, threshold_free
    from paths import LIGAND, PM_PREDS

    key = {f"A_{r}" for r in (9, 10, 11, 12, 16, 58, 59, 60, 61, 62, 63, 64, 65, 68, 69, 72, 78, 88, 92,
                              95, 96, 99, 100, 102, 103)}
    lig = parsePDB(str(LIGAND / "MRTX1133_crystal_pose_aligned.pdb")).getCoords()  # same frame as both anchors
    rows = []
    for tag in ("5US4", "7RPZ"):
        keys, ca = ca_table(STRUCTURES / f"{tag}_H.pdb")
        p2dir = P2RANK_ANCHOR_OUT / f"p2rank_{tag}"
        fp = fpocket(STRUCTURES / f"{tag}_H_out")
        p2 = p2rank(p2dir / f"{tag}_H.pdb_predictions.csv")
        pm = per_residue_probs(PM_PREDS / f"{tag}_H-preds.npy", keys)
        rows += [
            (tag, "fpocket", score_pockets(fp, key, keys, ca, lig) | threshold_free(pocket_residue_scores(fp, keys), key, keys)),
            (tag, "P2Rank", score_pockets(p2, key, keys, ca, lig)
             | threshold_free(p2rank_residue_scores(p2dir / f"{tag}_H.pdb_residues.csv", keys), key, keys)),
            (tag, "AE-PocketMiner", score_probs(pm, key, keys, ca) | threshold_free(pm, key, keys)),
            (tag, "AE-PocketMiner (gated)", score_probs(pm, key, keys, ca, gated=True)),
        ]
    cols = ["n_true", "precision", "recall", "mcc", "n_pred", "best_single_recall", "fragments", "dca",
            "top1", "top3", "best_rank", "auprc", "auroc"]
    with open(RESULTS / "kras_unified.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["structure", "tool"] + cols)
        for tag, tool, m in rows:
            w.writerow([tag, tool] + [m.get(c, "") if not isinstance(m.get(c), float) else f"{m[c]:.3f}" for c in cols])
    print(f"\nwrote {RESULTS / 'kras_unified.csv'}\n{'':6}{'tool':24}{'MCC Table I':>12}{'MCC unified':>12}"
          f"{'recall':>8}{'n_true':>7}{'AUPRC':>7}{'top3':>6}")
    for tag, tool, m in rows:
        old = TABLE_I.get((tag, tool), (None, None, None))[2]
        print(f"{tag:6}{tool:24}{'' if old is None else f'{old:.3f}':>12}{m['mcc']:12.3f}{m['recall']:8.3f}"
              f"{m['n_true']:7}{m.get('auprc', float('nan')):7.3f}{str(m.get('top3', '')):>6}")
    mcc = {tool: m["mcc"] for tag, tool, m in rows if tag == "7RPZ"}
    if not mcc["fpocket"] > mcc["P2Rank"] > mcc["AE-PocketMiner"]:
        print("NOTE: the 7RPZ ranking fpocket > P2Rank > AE-PocketMiner changes under the unified protocol")


if __name__ == "__main__":
    regenerate()
    check()
    print("sanity gate 1a passed: Table I reproduced from scratch")
    unified()
