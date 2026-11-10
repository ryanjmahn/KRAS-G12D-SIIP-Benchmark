"""
P2 s3: risk-coverage curves. If you only chase the predictions that survive perturbation, how often are you right?

p2_s1_reliability.py found no target-level signal: nothing, the tools' own scores included, says whether a
target's prediction is worth anything (AUC 0.42-0.59). So selective prediction cannot work by withholding
targets, and the only honest form of it is per residue. Every target's top-TOPK apo residues are pooled across
all 846 training targets, ranked by one selector, and precision is read off as a function of how large a
fraction is kept. That is the number a practitioner wants, and pooling also uses every target, unlike the
per-target AUC in s1 (undefined when a top-TOPK is all one class).

Selectors, all computed inside one target so they pool on a common scale:
  own     the tool's own apo score, as its rank within that target's top-TOPK
  freq    how often the residue stays in a conformer's top-TOPK (already 0-1, target-independent)
  combo   rank-sum of the two
Raw pooled apo scores are reported alongside `own` as a sensitivity check: fpocket volumes and P2Rank
probabilities are not on a common scale between targets, so pooling them unnormalised can mislead.

Selector values are coarse (10-20 levels), so ties are broken by a fixed hash of the residue, not by target
order; without that, a coverage slice would be filled from whichever targets happen to come first.

Outputs: results/p2_coverage.csv (curves), results/p2_coverage_points.csv (precision + cluster-bootstrap CI at
20/50/100% coverage), printed summary. The figure is drawn separately, by p2_s4_figures.py.
Usage (structprep env): python src/p2_s3_coverage.py [--gen random] [--test] [--format png pdf]
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS
from p2_s0_pilot import GENS
from p2_s1_reliability import TOPK, topk
from p2_s2_scale import cache_dir

TOOLS = ["fpocket", "P2Rank", "AE-PocketMiner"]
SELECTORS = ["own", "freq", "combo"]
GRID = np.round(np.arange(0.1, 1.001, 0.1), 2)      # coverage levels: keep each target's top 1, 2, ... of TOPK
POINTS = (0.2, 0.5, 1.0)
N_BOOT, SEED = 2000, 0


def _slot(v, tie):
    """Within-target position by `v`, descending, as a fraction of the target's residues (best = 1/n).

    Pooling raw selector values would not work: fpocket scores a residue by the best pocket containing it, so
    81% of targets have a single tied value across their whole top-TOPK, and an average rank would park every
    one of those residues mid-table while targets that happen to have distinct values filled the top of the
    pool. A coverage slice would then be a sample of targets, not of confident residues. Ranking inside the
    target, with ties broken by a fixed hash, keeps every target's contribution equal at every coverage.
    """
    o = np.lexsort((tie, -np.asarray(v, float)))
    s = np.empty(len(o))
    s[o] = np.arange(1, len(o) + 1) / len(o)
    return s


def pooled(gen, cache):
    """One row per (target, tool, residue): the hit label and each selector's within-target coverage slot."""
    rows = []
    for f in sorted(cache.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        keys, hit = [str(k) for k in z["keys"]], z["hit"].astype(bool)
        for tool in TOOLS:
            if f"apo|{tool}" not in z.files or f"{gen}|{tool}" not in z.files:
                continue
            a = z[f"apo|{tool}"].astype(float)
            idx = sorted(topk(a))
            C = z[f"{gen}|{tool}"].astype(float)
            freq = np.array([np.mean([i in topk(c) for c in C]) for i in idx])
            own = pd.Series(a[idx]).rank().to_numpy()
            tie = np.array([zlib.crc32(f"{f.stem}/{keys[i]}".encode()) / 2**32 for i in idx])
            sl = {s: _slot(v, tie) for s, v in (("own", a[idx]), ("freq", freq),
                                                ("combo", own + pd.Series(freq).rank().to_numpy()))}
            rows += [dict(parent=f.stem, tool=tool, res=keys[i], hit=bool(hit[i]), own_raw=float(a[i]),
                          freq_raw=float(freq[j]), **{s: v[j] for s, v in sl.items()})
                     for j, i in enumerate(idx)]
    return pd.DataFrame(rows)


def curve(d, sel, cov=GRID):
    """Precision when each target keeps only its top `cov` fraction by `sel`."""
    q, hit = d[sel].to_numpy(), d.hit.to_numpy()
    return np.array([hit[q <= c].mean() for c in np.atleast_1d(cov)])


def _boot(d, sels, cov, n=N_BOOT, seed=SEED):
    """Cluster bootstrap over targets (residues within a target are not independent). One set of resampled
    targets serves every selector, so differences between them stay paired."""
    q = {s: d[s].to_numpy() for s in sels}
    hit = d.hit.to_numpy()
    groups = [v.to_numpy() for v in d.reset_index(drop=True).groupby("parent", sort=True).groups.values()]
    rng = np.random.default_rng(seed)
    out = {s: np.empty((n, len(cov))) for s in sels}
    for b in range(n):
        i = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        hi = hit[i]
        for s in sels:
            qi = q[s][i]
            out[s][b] = [hi[qi <= c].mean() for c in cov]
    return out


def boot_points(d, sel, cov=POINTS, n=N_BOOT, seed=SEED):
    return np.percentile(_boot(d, [sel], cov, n, seed)[sel], [2.5, 97.5], axis=0)


def boot_gain(d, sel, ref="own", cov=POINTS, n=N_BOOT, seed=SEED):
    """Paired CI and two-sided bootstrap p for (sel - ref) precision at each coverage."""
    o = _boot(d, [sel, ref], cov, n, seed)
    g = o[sel] - o[ref]
    lo, hi = np.percentile(g, [2.5, 97.5], axis=0)
    p = 2 * np.minimum((g <= 0).mean(0), (g >= 0).mean(0))
    return g.mean(0), lo, hi, np.minimum(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="random", choices=GENS)
    ap.add_argument("--format", nargs="+", default=["png", "pdf"])
    ap.add_argument("--test", action="store_true", help="read the held-out test split instead of training")
    args = ap.parse_args()
    split = "test" if args.test else "train"
    sfx = "_test" if args.test else ""
    cache = cache_dir(split)
    if not cache.exists():
        sys.exit(f"no {split} cache; run: python src/p2_s2_scale.py run" + (" --test" if sfx else ""))

    curves, points = [], []
    for gen in GENS:                                  # every generator in the CSV, the chosen one in the figure
        d = pooled(gen, cache)
        for tool, g in d.groupby("tool"):
            for sel in SELECTORS:
                p = curve(g, sel)
                curves += [dict(gen=gen, tool=tool, selector=sel, coverage=c, precision=v,
                                n_res=len(g), n_targets=g.parent.nunique()) for c, v in zip(GRID, p)]
                if gen == args.gen:
                    lo, hi = boot_points(g, sel)
                    gn, glo, ghi, gp = boot_gain(g, sel)
                    points += [dict(tool=tool, selector=sel, coverage=c, precision=v, lo=l, hi=h,
                                    gain_vs_own=m, gain_lo=a, gain_hi=b, gain_p=q)
                               for c, v, l, h, m, a, b, q in
                               zip(POINTS, curve(g, sel, np.asarray(POINTS)), lo, hi, gn, glo, ghi, gp)]
        if gen == args.gen:
            dsel = d
    cv = pd.DataFrame(curves)
    cv.to_csv(RESULTS / f"p2_coverage{sfx}.csv", index=False)
    pt = pd.DataFrame(points)
    pt.to_csv(RESULTS / f"p2_coverage_points{sfx}.csv", index=False)

    print(f"Pooled top-{TOPK} apo residues over {dsel.parent.nunique()} {split} targets, generator {args.gen}.")
    print("Precision when only the top fraction by each selector is kept (95% cluster bootstrap over targets).")
    print(pt.round(3).to_string(index=False))

    # sensitivity: the other pooling convention, one global ranking of raw values, which lets a target
    # contribute more or fewer residues at a given coverage
    w = pt[pt.coverage == 0.2].pivot(index="tool", columns="selector", values="precision")
    for sel, col in (("own", "own_raw"), ("freq", "freq_raw")):
        w[f"{sel}_global"] = [
            float(np.cumsum(g.sort_values(col, ascending=False).hit.to_numpy())[max(round(.2 * len(g)), 1) - 1]
                  / max(round(.2 * len(g)), 1)) for _, g in dsel.groupby("tool")]
    print("\nsensitivity at 20% coverage: ranking inside each target (used above) vs one global ranking of raw")
    print("values. They differ because a global ranking lets some targets contribute all 10 and others none.")
    print(w[["own", "own_global", "freq", "freq_global"]].round(3).to_string())


if __name__ == "__main__":
    main()
