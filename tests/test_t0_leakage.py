"""
Leakage gate: no trust feature may know anything about the answer.

The targets (does the top-10 contain a true residue, is MCC above 0.3, AUPRC lift) are computed from the
truth labels, which is what a label is for. No *feature* may be. The risk is an indirect path rather than an
obvious one -- p1_master's n_pred and spread, for instance, come from a pocket set already filtered to
within 4 A of a ligand atom, so they carry the holo structure with them.

A comment cannot enforce that, so this rebuilds the whole feature table from caches whose truth labels have
been permuted inside each target, and requires every feature column to come back bit-identical. Anything
that touched the labels, however indirectly, moves. The target columns are expected to move, and a run in
which they did not would mean the permutation never reached the pipeline, so that is checked too.

Run (structprep env, ~2 min): python tests/test_t0_leakage.py [n_targets]
"""
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import t0_features as T
from p1_adapters import ca_table
from paths import RUNS

warnings.filterwarnings("ignore")
N_DEFAULT, SEED = 150, 0
TARGET_COLS = {"y_top10hit", "y_auprc_lift", "y_mcc30", "y_mcc_med", "mcc"}
IDENT = {"parent", "tool", "split"}


def build(cache, files, coords):
    T.CACHE = cache
    return pd.DataFrame([r for f in files for r in T.per_target(f, coords)]).sort_values(
        ["parent", "tool"]).reset_index(drop=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    real = sorted((RUNS / "trust_cache").glob("*.npz"))
    if not real:
        sys.exit("no cache; run: python src/t0_extract.py")
    rng = np.random.default_rng(SEED)
    real = [real[i] for i in sorted(rng.choice(len(real), min(n, len(real)), replace=False))]
    coords = {f.stem: ca_table(RUNS / "p1" / f.stem / f"{f.stem}.pdb")[1] for f in real
              if (RUNS / "p1" / f.stem / f"{f.stem}.pdb").exists()}

    tmp = Path(tempfile.mkdtemp())
    try:
        for f in real:                              # same scores, labels shuffled inside each target
            z = np.load(f, allow_pickle=False)
            d = {k: z[k] for k in z.files}
            d["hit"] = rng.permutation(d["hit"])
            np.savez_compressed(tmp / f.name, **d)
        a = build(RUNS / "trust_cache", real, coords)
        b = build(tmp, sorted(tmp.glob("*.npz")), coords)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"{len(a)} rows over {a.parent.nunique()} targets, labels permuted within each target")
    feats = [c for c in a.columns if c not in TARGET_COLS | IDENT]
    moved, same = [], []
    for c in feats:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        eq = np.array_equal(x, y) or np.allclose(np.nan_to_num(x.astype(float), nan=-9e9),
                                                 np.nan_to_num(y.astype(float), nan=-9e9), rtol=0, atol=0)
        (same if eq else moved).append(c)
    print(f"  {len(same)}/{len(feats)} feature columns unchanged under label permutation")

    bad = []
    if moved:
        bad.append("features that move when the labels are permuted (they are reading the answer): "
                   + ", ".join(moved))
    hit_t = [c for c in TARGET_COLS if c in a and not a[c].equals(b[c])]
    if "y_top10hit" not in hit_t:
        bad.append("y_top10hit did not change: the permutation never reached the pipeline, so this run "
                   "proves nothing")
    if "base_rate" in same:
        print("  base_rate is invariant, as it must be: permuting within a target preserves the count")
    print(f"  target columns that moved, as expected: {sorted(hit_t)}")

    print("\n".join("FAIL: " + m for m in bad) if bad else "\nok")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
