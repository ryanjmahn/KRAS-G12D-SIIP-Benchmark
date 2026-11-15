"""Put every src/ subpackage on sys.path, so a module can be imported by its bare name from anywhere.

The scripts are grouped into folders for reading, but they are run directly (`python src/trust/t3_rule.py`)
and import each other flatly (`from p1_s6_aggregate import holm`). Rather than convert thirty files to a
package and change every documented command to `python -m`, each one adds src/ to the path and imports this,
which adds the siblings. One place to change if the layout moves again.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
for _p in [_SRC, *(p for p in sorted(_SRC.iterdir()) if p.is_dir() and not p.name.startswith(("_", ".")))]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
