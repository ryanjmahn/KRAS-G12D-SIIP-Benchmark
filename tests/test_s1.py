"""Self-checks for P1 prep geometry. Run: python tests/test_s1.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p1_s1_prep import aligned_pairs, kabsch


def test_kabsch():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3))
    a = 0.7
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Q = P @ Rz.T + [5, -2, 1]
    R, t = kabsch(P, Q)
    assert np.allclose(P @ R.T + t, Q, atol=1e-8)


def test_alignment_ignores_numbering():
    names = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU"]
    holo = [(n, np.array([i, 0.0, 0.0])) for i, n in enumerate(names)]
    apo = [("TRP", np.zeros(3))] + [(n, np.array([i, 1.0, 0.0])) for i, n in enumerate(names[:4] + ["MSE"] + names[5:])]
    pairs = aligned_pairs(holo, apo)          # extra N-terminal TRP in apo; PHE->MET is a mismatch, skipped
    assert len(pairs) == 9
    assert all(h[0] == a[0] for h, a in pairs)  # same position index = correctly paired despite the offset


if __name__ == "__main__":
    test_kabsch()
    test_alignment_ignores_numbering()
    print("ok")
