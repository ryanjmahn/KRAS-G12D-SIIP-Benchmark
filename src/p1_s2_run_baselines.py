"""
P1 s2: run fpocket, P2Rank and AE-PocketMiner on every prepared target (runs/p1/<target>/<target>.pdb).

Raw native outputs are kept untouched:
  fpocket        runs/p1/<target>/<target>_out/
  P2Rank         runs/p1/p2rank_out/<target>.pdb_{predictions,residues}.csv   (one batch run)
  AE-PocketMiner runs/p1/pocketminer_out/<target>-preds.npy                  (aepocketminer env, one batch run)
Status per target -> results/p1_run_log.csv; tool versions -> results/p1_tools.json.
Existing outputs are skipped unless --force. A tool failing on a target is logged, not fatal.

Usage (structprep env, Java 17): python src/p1_s2_run_baselines.py [--tools fpocket p2rank pocketminer] [--force]
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PRANK, RESULTS, ROOT, RUNS

P1 = RUNS / "p1"
P2_OUT = P1 / "p2rank_out"
PM_IN, PM_OUT = P1 / "pocketminer_in", P1 / "pocketminer_out"
PM_REPO = (ROOT / "ae-pocketminer").resolve()


def prepared():
    return sorted(p.parent.name for p in P1.glob("*/meta.json"))


def run_fpocket(tid, force):
    d = P1 / tid
    if (d / f"{tid}_out" / f"{tid}_info.txt").exists() and not force:
        return
    shutil.rmtree(d / f"{tid}_out", ignore_errors=True)
    # relative name + cwd: fpocket segfaults on absolute paths containing a space
    subprocess.run(["fpocket", "-f", f"{tid}.pdb"], cwd=d, capture_output=True)


def run_p2rank(tids, force):
    todo = [t for t in tids if force or not (P2_OUT / f"{t}.pdb_predictions.csv").exists()]
    if not todo:
        return
    ds = P1 / "p2rank.ds"
    ds.write_text("\n".join(f"{t}/{t}.pdb" for t in todo) + "\n")
    r = subprocess.run([str(PRANK), "predict", ds.name, "-o", P2_OUT.name, "-threads", "8"],
                       cwd=P1, capture_output=True, text=True)
    (P1 / "p2rank.log").write_text(r.stdout + r.stderr)


def run_pocketminer(tids, force):
    todo = [t for t in tids if force or not (PM_OUT / f"{t}-preds.npy").exists()]
    if not todo:
        return
    shutil.rmtree(PM_IN, ignore_errors=True)
    PM_IN.mkdir(parents=True)
    PM_OUT.mkdir(exist_ok=True)
    for t in todo:
        (PM_IN / f"{t}.pdb").symlink_to(P1 / t / f"{t}.pdb")
    cfg = P1 / "pocketminer.yaml"
    cfg.write_text(json.dumps({          # JSON is valid YAML
        "nn_path": str(PM_REPO / "models" / "pocketminer"), "input_pdb_directory": str(PM_IN),
        "output_directory": str(PM_OUT), "use_attention": False, "debug": False}))
    r = subprocess.run(["conda", "run", "-n", "aepocketminer", "python", "src/xtal_predict.py", str(cfg)],
                       cwd=PM_REPO, capture_output=True, text=True)
    (P1 / "pocketminer.log").write_text(r.stdout + r.stderr)


def versions():
    v = {"fpocket": "4.2.3 (envs/structprep.yml)", "p2rank": Path(PRANK).resolve().parent.name}
    try:
        v["ae-pocketminer_commit"] = subprocess.run(["git", "-C", str(PM_REPO), "rev-parse", "HEAD"],
                                                    capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="*", default=["fpocket", "p2rank", "pocketminer"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    tids = prepared()
    print(f"{len(tids)} prepared targets")

    if "fpocket" in args.tools:
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda t: run_fpocket(t, args.force), tids))
        print("fpocket done")
    if "p2rank" in args.tools:
        run_p2rank(tids, args.force)
        print("P2Rank done")
    if "pocketminer" in args.tools:
        run_pocketminer(tids, args.force)
        print("AE-PocketMiner done")

    log = pd.DataFrame({"target_id": tids})
    log["fpocket"] = [(P1 / t / f"{t}_out" / f"{t}_info.txt").exists() for t in tids]
    log["p2rank"] = [(P2_OUT / f"{t}.pdb_predictions.csv").exists() for t in tids]
    log["pocketminer"] = [(PM_OUT / f"{t}-preds.npy").exists() for t in tids]
    log.to_csv(RESULTS / "p1_run_log.csv", index=False)
    (RESULTS / "p1_tools.json").write_text(json.dumps(versions(), indent=2))
    for tool in ("fpocket", "p2rank", "pocketminer"):
        missing = log.loc[~log[tool], "target_id"].tolist()
        print(f"{tool}: {log[tool].sum()}/{len(log)} ok" + (f"; missing {missing[:10]}" if missing else ""))


if __name__ == "__main__":
    main()
