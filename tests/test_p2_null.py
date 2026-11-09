"""
Sanity gate: the Phase 2 analysis must find nothing when there is nothing to find.

The reported effect is small (fpocket +0.07 AUC, +0.03 precision at 20% coverage), and one earlier version of
the coverage analysis produced a much larger apparent effect that turned out to be an artefact of how
residues were pooled across targets. So the pipeline is run end to end on the real tool scores with the TRUTH
LABELS PERMUTED inside each target. Nothing else changes: the same cache, the same perturbation scores, the
same code path, the same number of positives per target.

A single permutation is not a test. The per-target AUC gain is undefined unless a target's top-10 holds both
a true and a false residue, so only a minority of targets contribute and one draw scatters widely (sd ~0.05
on 150 targets). The test therefore builds a permutation distribution over N_PERM draws and asks:

  1. unbiased     the permuted gains average to zero, within the standard error of the N_PERM draws
  2. separated    the real fpocket gain is at least Z sd above the permuted mean
  3. flat         with permuted labels, keeping the most stable 20% is no better than keeping everything
  4. detectable   the real labels on the same sample do show the fpocket effect, so the test can fail

Measured on all 846 training targets: permuted mean +0.005, sd 0.015, real +0.075, i.e. 4.5 sd clear.
Run (structprep env, ~25 min): python tests/test_p2_null.py [n_targets] [n_perm]   # n_targets 0 = all
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_s1_reliability import per_target
from p2_s2_scale import CACHE
from p2_s3_coverage import GRID, curve, pooled

warnings.filterwarnings("ignore")           # constant-input Spearman on flat fpocket arrays is expected

N_TARGETS, N_PERM, SEED = 0, 10, 0      # 0 = every cached target: at 150 the null sd is 0.046 and the
                                        # real effect clears it by only ~2 sd, so a smaller run flakes
KEY = ("fpocket", "random", "freq")         # the effect the paper rests on
Z = 3.0                                     # how many permuted sd the real effect must clear
FLAT = 0.03                                 # allowed |precision lift| at 20% coverage under permuted labels


def shuffled(f, rng):
    """The cached target with `hit` permuted; every score array is left exactly as it is."""
    z = np.load(f, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["hit"] = rng.permutation(d["hit"])
    return d


def table(files, permute, rng):
    return pd.DataFrame([row for f in files
                         for row in per_target(f, z=shuffled(f, rng) if permute else None, name=f.stem)])


def gains(r):
    """Mean per-target AUC gain over the tool's own score, per tool x generator x signal."""
    return {(tool, gen, col.split("_")[0]): (g[col] - g.own_auc_topk).dropna().mean()
            for (tool, gen), g in r.groupby(["tool", "gen"])
            for col in ("freq_auc_topk", "combo_auc_topk")}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_TARGETS
    k = int(sys.argv[2]) if len(sys.argv) > 2 else N_PERM
    files = sorted(CACHE.glob("*.npz"))
    if not files:
        sys.exit("no cache; run: python src/p2_s2_scale.py run")
    if n:
        pick = np.random.default_rng(SEED).choice(len(files), min(n, len(files)), replace=False)
        files = [files[i] for i in sorted(pick)]
    print(f"{len(files)} cached targets, {k} label permutations within each target")

    draws = [gains(table(files, True, np.random.default_rng(100 + s))) for s in range(k)]
    perm = np.array([d[KEY] for d in draws])
    mu, sd, se = perm.mean(), perm.std(ddof=1), perm.std(ddof=1) / np.sqrt(k)
    real = gains(table(files, False, None))
    got = real[KEY]
    print(f"  permuted {KEY[0]}/{KEY[1]}/{KEY[2]}: mean {mu:+.4f}  sd {sd:.4f}  se {se:.4f}  "
          f"range [{perm.min():+.4f}, {perm.max():+.4f}]")
    print(f"  real     {KEY[0]}/{KEY[1]}/{KEY[2]}: {got:+.4f}  ({(got - mu) / sd:.1f} permuted sd above the null)")

    bad = []
    if abs(mu) > 2 * se:
        bad.append(f"permuted gains are biased: mean {mu:+.4f}, se {se:.4f}")
    if got < mu + Z * sd:
        bad.append(f"real gain {got:+.4f} is under {Z} sd above the permuted mean {mu:+.4f} (sd {sd:.4f})")
    if got <= 0:
        bad.append(f"real labels give {got:+.4f}: the test cannot detect the effect it guards")

    rng = np.random.default_rng(SEED)
    d = pooled("random", CACHE)
    d = d[d.parent.isin({f.stem for f in files})].copy()
    d["hit"] = np.concatenate([rng.permutation(g.hit.to_numpy())
                               for _, g in d.groupby(["parent", "tool"], sort=True)])
    for tool, g in d.groupby("tool"):
        c = curve(g, "combo", GRID)
        lift = c[1] - c[-1]                         # GRID is 0.1, 0.2, ... so c[1] is 20% coverage
        print(f"  coverage {tool:>15s}: lift at 20% under permuted labels = {lift:+.4f}")
        if abs(lift) > FLAT:
            bad.append(f"permuted labels still move {tool}'s precision by {lift:+.4f} at 20% coverage")

    print("\n".join("FAIL: " + b for b in bad) if bad else "ok")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
