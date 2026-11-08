"""Self-checks for P1 prep geometry. Run: python tests/test_s1.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p1_s1_prep import align_names, kabsch


def test_kabsch():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3))
    a = 0.7
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Q = P @ Rz.T + [5, -2, 1]
    R, t = kabsch(P, Q)
    assert np.allclose(P @ R.T + t, Q, atol=1e-8)


def test_alignment_ignores_numbering():
    holo = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU"]
    apo = ["TRP"] + holo[:4] + ["MSE"] + holo[5:]    # extra N-terminal TRP; PHE->MSE is a mismatch, skipped
    pairs = align_names(holo, apo)
    assert len(pairs) == 9
    assert all(j == i + 1 for i, j in pairs)          # paired correctly despite the one-residue offset
    assert align_names(["MSE"], ["MET"]) == [(0, 0)]  # selenomethionine counts as MET
    assert len(align_names(holo, apo, identical=False)) == 10   # mismatch kept when asked


if __name__ == "__main__":
    test_kabsch()
    test_alignment_ignores_numbering()
    print("ok")
