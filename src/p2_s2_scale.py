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

CACHE = RUNS / "p2_cache"
CHUNK = 40          # targets per chunk: ~1,200 conformers, about 2 GB before cleaning


def training_ids():
    ids = []
    for m in sorted(P1.glob("*/meta.json")):
        d = json.loads(m.read_text())
        if d["split"] in TRAIN and d.get("usable", True):
            ids.append(d["target_id"])
    return sorted(ids)


def extract(tid):
    """Reduce one target's conformers to per-residue score arrays. Returns False if nothing was cached."""
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
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{tid}.npz", **data)
    return True


def run_tools(gen):
    subprocess.run([sys.executable, str(Path(__file__).parent / "p1_s2_run_baselines.py"),
                    "--root", str(OUT / gen), "--log-suffix", f"_p2scale_{gen}"], check=False)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    ids = training_ids()
    done = {p.stem for p in CACHE.glob("*.npz")} if CACHE.exists() else set()
    if step == "status":
        print(f"{len(done)}/{len(ids)} training targets cached in {CACHE}")
        return
    if step == "cache":                       # conformers already on disk from the 100-target pilot
        n = sum(extract(t) for t in ids if t not in done and (OUT / GENS[0] / f"{t}__c00").exists())
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
        n = sum(extract(t) for t in built)
        for gen in GENS:
            shutil.rmtree(OUT / gen, ignore_errors=True)
        el = time.time() - t0
        print(f"[{i + len(chunk)}/{len(todo)}] cached {n}/{len(chunk)}; {el / 60:.0f} min elapsed, "
              f"~{el / (i + len(chunk)) * (len(todo) - i - len(chunk)) / 60:.0f} min left", flush=True)
    print(f"done: {len(list(CACHE.glob('*.npz')))} targets cached")


if __name__ == "__main__":
    main()
