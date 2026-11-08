"""
P1 s6: failure-mode tags (docs/evaluation_protocol.md §6); cross-target statistics follow in this module.

Tags, per tool per target, from the primary variant (pocket tools gate4, probability tools their primary t):
  miss           nothing at the site: no pocket within OFFSET_GATE of the ligand / no true residue >= t
  offset         pocket tools: no pocket within 4 Å of the ligand, but one within OFFSET_GATE
  fragmentation  pocket tools: >= FRAG_MIN_POCKETS at-site pockets and no single one recovers
                 FRAG_MAX_SHARE of what their union recovers
  over_spread    predicts > SPREAD_MIN x the true pocket size with precision < SPREAD_MAX_PRECISION
Tags may overlap. Primary label = first tag in ORDER, else success (MCC >= SUCCESS_MCC) or partial.

Rules are locked on the training folds before the test split is tagged:
  python src/p1_s6_aggregate.py lock    training folds: prevalence + sensitivity grid -> results/p1_tag_rules.json
  python src/p1_s6_aggregate.py tags    test split with the locked rules -> results/p1_tags.csv
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS

POCKET_TOOLS = {"fpocket", "P2Rank"}
ORDER = ["miss", "offset", "fragmentation", "over_spread"]
RULES = dict(OFFSET_GATE=8.0, FRAG_MIN_POCKETS=2, FRAG_MAX_SHARE=0.67, SPREAD_MIN=2.0,
             SPREAD_MAX_PRECISION=0.3, SUCCESS_MCC=0.5)


def tag_table(master, rules):
    prim = master[master.primary]
    near = master[master.variant == f"gate{rules['OFFSET_GATE']:g}"].set_index(["target_id", "tool"]).site_hit
    rows = []
    for r in prim.itertuples():
        pocket = r.tool in POCKET_TOOLS
        close = bool(near.get((r.target_id, r.tool), False)) if pocket else bool(r.site_hit)
        t = dict(miss=not r.site_hit and not close,
                 offset=pocket and not r.site_hit and close,
                 fragmentation=pocket and r.fragments >= rules["FRAG_MIN_POCKETS"]
                 and r.best_single_recall < rules["FRAG_MAX_SHARE"] * r.recall,
                 over_spread=bool(r.spread > rules["SPREAD_MIN"] and r.precision < rules["SPREAD_MAX_PRECISION"]))
        label = next((k for k in ORDER if t[k]), None) or ("success" if r.mcc >= rules["SUCCESS_MCC"] else "partial")
        rows.append(dict(target_id=r.target_id, tool=r.tool, mcc=r.mcc) | t | dict(label=label))
    return pd.DataFrame(rows)


def prevalence(tags):
    rates = tags.groupby("tool")[ORDER].mean()
    labels = tags.groupby("tool").label.value_counts(normalize=True).unstack(fill_value=0)
    return rates.join(labels[[c for c in ORDER + ["partial", "success"] if c in labels]], rsuffix="_label")


def lock():
    train = pd.read_csv(RESULTS / "p1_master_train.csv")
    tags = tag_table(train, RULES)
    print(f"training folds, {tags.target_id.nunique()} targets: tag rates (overlapping) | primary-label shares (_label)")
    print(prevalence(tags).round(3).to_string())
    print("\nsensitivity on the training folds (share of targets tagged):")
    for mp, sh in itertools.product((2, 3), (0.5, 0.67, 0.8)):
        r = tag_table(train, RULES | dict(FRAG_MIN_POCKETS=mp, FRAG_MAX_SHARE=sh))
        print(f"  fragmentation >= {mp} pockets, share < {sh}:",
              r[r.tool.isin(POCKET_TOOLS)].groupby("tool").fragmentation.mean().round(3).to_dict())
    for s in (1.5, 2.0, 3.0):
        r = tag_table(train, RULES | dict(SPREAD_MIN=s))
        print(f"  over-spread, spread > {s}:", r.groupby("tool").over_spread.mean().round(3).to_dict())
    for g in (6.0, 8.0, 10.0):
        r = tag_table(train, RULES | dict(OFFSET_GATE=g))
        print(f"  offset, near = {g:g} Å:", r[r.tool.isin(POCKET_TOOLS)].groupby("tool").offset.mean().round(3).to_dict())
    (RESULTS / "p1_tag_rules.json").write_text(json.dumps(RULES | {"locked_on": "CryptoBench training folds 0-3"}, indent=2))
    print(f"\nlocked -> {RESULTS / 'p1_tag_rules.json'}")


def tags():
    rules = json.loads((RESULTS / "p1_tag_rules.json").read_text())
    t = tag_table(pd.read_csv(RESULTS / "p1_master.csv"), rules)
    t.to_csv(RESULTS / "p1_tags.csv", index=False)
    print(f"test split, {t.target_id.nunique()} targets, locked rules: tag rates | primary-label shares (_label)")
    print(prevalence(t).round(3).to_string())
    print(f"-> {RESULTS / 'p1_tags.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["lock", "tags"])
    {"lock": lock, "tags": tags}[ap.parse_args().step]()
