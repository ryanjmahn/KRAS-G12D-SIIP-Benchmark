"""
P2 s2: run the reliability pilot over every usable training-fold target (~846), in chunks.

The conformers themselves are bulk: 846 targets x 3 generators x 10 conformers is ~38 GB of PDBs and tool
output, far more than this machine has free. So a chunk of targets is built, run through the tools, reduced to
the only thing the analysis needs — one score per residue per conformer — and then deleted. The cache is ~100 MB
for the whole training set, and every later analysis (p2_s1_reliability.py, risk-coverage curves) reads it
instead of re-running anything.

  runs/p2_cache/<target>.npz    keys, hit (true pocket residue), "apo|<tool>", "<gen>|<tool>" (N_CONF x n_res)

Resumable: a target with a cache file is skipped, so an interrupted run continues where it stopped.

Usage (structprep env, Java 17):
  python src/p2_s2_scale.py cache     # cache the pilot's 100 targets from the conformers already on disk
  python src/p2_s2_scale.py run       # build, run, cache and clean every remaining training target
  python src/p2_s2_scale.py status
  ... --test        the same for the held-out test split, into runs/p2_cache_test
  ... --scale 2 --gens nma1,random --tag amp2 --limit 200    a reduced check at another amplitude
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RUNS
from p2_s0_pilot import GENS, N_CONF, OUT, P1, TRAIN, build_target, tool_scores

CACHE = RUNS / "p2_cache"               # the training sweep; the held-out test split caches alongside it
SPLITS = dict(train=TRAIN, test=["test"])
CHUNK = 40          # targets per chunk: ~1,200 conformers, about 2 GB before cleaning. Each chunk starts a
                    # JVM and loads TensorFlow once per generator, so small chunks are mostly startup cost;
                    # a chunk that loses a tool is retried on the next pass (see complete()).


def complete(f):
    """A cached target counts as done only if every generator ran for every tool the apo has: a chunk whose
    P2Rank or AE-PocketMiner invocation failed otherwise looks finished and is never retried."""
    try:
        z = np.load(f, allow_pickle=False)
    except Exception:
        return False
    tools = [n.split("|", 1)[1] for n in z.files if n.startswith("apo|")]
    return bool(tools) and all(f"{g}|{t}" in z.files for g in GENS for t in tools)


def cache_dir(split="train", tag=None):
    """Caches never collide: a tagged run (a different amplitude, say) gets its own folder."""
    if tag:
        return RUNS / f"p2_cache_{tag}"
    return CACHE if split == "train" else RUNS / f"p2_cache_{split}"


def training_ids(split="train"):
    ids = []
    for m in sorted(P1.glob("*/meta.json")):
        d = json.loads(m.read_text())
        if d["split"] in SPLITS[split] and d.get("usable", True):
            ids.append(d["target_id"])
    return sorted(ids)


def extract(tid, cache=None):
    """Reduce one target's conformers to per-residue score arrays. Returns False if nothing was cached."""
    cache = cache or CACHE
    from p1_adapters import ca_table
    keys, _ = ca_table(P1 / tid / f"{tid}.pdb")
    meta = json.loads((P1 / tid / "meta.json").read_text())
    truth = set(meta["truth"]) & set(keys)
    apo = tool_scores(P1, tid, keys)
    if not apo:
        return False
    data = {"keys": np.array(keys), "hit": np.array([k in truth for k in keys])}
    data |= {f"apo|{t}": a.astype(np.float32) for t, a in apo.items()}
    for gen in GENS:
        for t in apo:
            cs = [tool_scores(OUT / gen, f"{tid}__c{k:02d}", keys).get(t) for k in range(N_CONF)]
            if all(c is not None for c in cs):
                data[f"{gen}|{t}"] = np.vstack(cs).astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / f"{tid}.npz", **data)
    return True


def run_tools(gen):
    subprocess.run([sys.executable, str(Path(__file__).parent / "p1_s2_run_baselines.py"),
                    "--root", str(OUT / gen), "--log-suffix", f"_p2scale_{gen}"], check=False)


def main():
    global GENS
    import argparse
    import p2_s0_pilot
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="", choices=["", "cache", "run", "status"])
    ap.add_argument("--test", action="store_true", help="the held-out split instead of the training folds")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the perturbation amplitude")
    ap.add_argument("--gens", default=None, help="comma-separated subset of the generators")
    ap.add_argument("--tag", default=None, help="write to runs/p2_cache_<tag> instead of the default cache")
    ap.add_argument("--limit", type=int, default=0, help="only the first N targets (a reduced check)")
    args = ap.parse_args()
    split = "test" if args.test else "train"
    step = args.step
    p2_s0_pilot.SCALE = args.scale
    if args.gens:                               # both modules read their own GENS, so both are set
        GENS = tuple(args.gens.split(","))
        p2_s0_pilot.GENS = GENS
    ids = training_ids(split)
    if args.limit:
        ids = ids[:args.limit]
    CA = cache_dir(split, args.tag)
    done = {p.stem for p in CA.glob("*.npz") if complete(p)} if CA.exists() else set()
    if step == "status":
        print(f"{len(done)}/{len(ids)} {split} targets cached in {CA}")
        return
    if step == "cache":                       # conformers already on disk from the 100-target pilot
        n = sum(extract(t, CA) for t in ids if t not in done and (OUT / GENS[0] / f"{t}__c00").exists())
        print(f"cached {n} targets already built; {len(set(done) | set())} total")
        return
    if step != "run":
        sys.exit(__doc__)

    todo = [t for t in ids if t not in done]
    print(f"{len(done)}/{len(ids)} already cached; {len(todo)} to go, {CHUNK} per chunk", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        for gen in GENS:                      # a clean slate, so the tools only see this chunk
            shutil.rmtree(OUT / gen, ignore_errors=True)
        built = []
        for tid in chunk:
            try:
                build_target(tid)
                built.append(tid)
            except Exception as e:            # one bad structure must not stop the sweep
                print(f"  build failed {tid}: {type(e).__name__}: {e}", flush=True)
        for gen in GENS:
            run_tools(gen)
        n = sum(extract(t, CA) for t in built)
        for gen in GENS:
            shutil.rmtree(OUT / gen, ignore_errors=True)
        el = time.time() - t0
        print(f"[{i + len(chunk)}/{len(todo)}] cached {n}/{len(chunk)}; {el / 60:.0f} min elapsed, "
              f"~{el / (i + len(chunk)) * (len(todo) - i - len(chunk)) / 60:.0f} min left", flush=True)
    print(f"done: {len(list(CA.glob('*.npz')))} {split} targets cached")


if __name__ == "__main__":
    main()
