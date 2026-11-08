"""Self-checks for failure-mode tags on hand-built rows. Run: python tests/test_s6.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p1_s6_aggregate import RULES, tag_table


def row(tid, tool, variant, primary, **m):
    base = dict(target_id=tid, tool=tool, variant=variant, primary=primary, site_hit=False, mcc=0.0,
                recall=0.0, precision=0.0, spread=0.0, fragments=0, best_single_recall=0.0)
    return base | m


def test_tags():
    rows = [
        row("a", "fpocket", "gate4", True, site_hit=True, mcc=0.7, recall=0.8, precision=0.8, spread=1.0,
            fragments=1, best_single_recall=0.8),                                     # success
        row("b", "fpocket", "gate4", True),                                           # nothing anywhere: miss
        row("c", "fpocket", "gate4", True), row("c", "fpocket", "gate8", False, site_hit=True),   # offset
        row("d", "P2Rank", "gate4", True, site_hit=True, mcc=0.4, recall=0.6, precision=0.6, spread=1.0,
            fragments=2, best_single_recall=0.3),                                     # fragmentation
        row("e", "AE-PocketMiner", "t0.7", True, site_hit=True, mcc=0.1, recall=0.5, precision=0.1,
            spread=5.0),                                                              # over-spread
        row("f", "AE-PocketMiner", "t0.7", True, site_hit=True, mcc=0.3, recall=0.4, precision=0.4, spread=1.0),
    ]
    t = tag_table(pd.DataFrame(rows), RULES).set_index("target_id").label.to_dict()
    assert t == dict(a="success", b="miss", c="offset", d="fragmentation", e="over_spread", f="partial"), t


if __name__ == "__main__":
    test_tags()
    print("ok")
