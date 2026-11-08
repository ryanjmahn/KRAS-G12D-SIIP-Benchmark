"""Self-checks for the P1 scoring math on hand-computable cases. Run: python tests/test_s3.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p1_adapters import pocket_residue_scores
from p1_s3_score import confusion, score_pockets, score_probs, threshold_free

KEYS = [f"A_{i}" for i in range(1, 11)]
CA = np.array([[i * 3.0, 0, 0] for i in range(10)])          # residues 3 Å apart on a line
TRUTH = {"A_1", "A_2", "A_3"}                                  # site centre at x = 3


def test_confusion():
    m = confusion({"A_1", "A_2", "A_9"}, TRUTH, set(KEYS))    # tp 2, fp 1, fn 1, tn 6
    assert abs(m["precision"] - 2 / 3) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["mcc"] - (2 * 6 - 1 * 1) / (3 * 3 * 7 * 7) ** 0.5) < 1e-9
    assert confusion(set(), TRUTH, set(KEYS))["mcc"] == 0.0


def test_threshold_free():
    m = threshold_free({"a": 0.9, "b": 0.8, "c": 0.7}, {"a", "c"}, ["a", "b", "c"])
    assert abs(m["auprc"] - (1 + 2 / 3) / 2) < 1e-9 and abs(m["auroc"] - 0.5) < 1e-9
    assert np.isnan(threshold_free({"a": 1.0}, {"a"}, ["a"])["auprc"])


def test_pockets():
    lig = np.array([[3.0, 1.0, 0.0]])
    pockets = [dict(rank=1, center=np.array([27.0, 0, 0]), residues={"A_10"}, score=0.9),   # far: not at site
               dict(rank=2, center=np.array([3.0, 2.0, 0]), residues={"A_1", "A_2"}, score=0.5),
               dict(rank=3, center=np.array([4.5, 0, 0]), residues={"A_3", "A_4"}, score=0.4),
               dict(rank=4, center=np.array([9.0, 0, 0]), residues={"A_3"}, score=0.3)]       # 6.1 Å from ligand: out
    m = score_pockets(pockets, TRUTH, KEYS, CA, lig)
    assert m["fragments"] == 2 and m["recall"] == 1.0 and m["precision"] == 0.75
    assert abs(m["best_single_recall"] - 2 / 3) < 1e-9 and abs(m["dca"] - 1.0) < 1e-9
    assert not m["top1"] and m["top3"] and m["best_rank"] == 2
    assert pocket_residue_scores(pockets, KEYS)["A_3"] == 0.4
    wide = score_pockets(pockets, TRUTH, KEYS, CA, lig, gate=7.0)    # wider gate admits the 6.1 Å pocket
    assert wide["fragments"] == 3 and wide["top3"] == m["top3"]       # detection does not depend on the gate


def test_probs_gate():
    scores = dict.fromkeys(KEYS, 0.0) | {"A_1": 0.9, "A_10": 0.9}
    assert score_probs(scores, TRUTH, KEYS, CA)["n_pred"] == 2
    assert score_probs(scores, TRUTH, KEYS, CA, gated=True)["n_pred"] == 1     # A_10 is 24 Å from the site


if __name__ == "__main__":
    test_confusion()
    test_threshold_free()
    test_pockets()
    test_probs_gate()
    print("ok")
