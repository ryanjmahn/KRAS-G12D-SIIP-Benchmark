"""Repo locations shared by every pipeline script, so they run from any directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
STRUCTURES = DATA / "structures"
LIGAND = DATA / "ligand"
DOCKING = DATA / "docking"
LADDER_CSV = DATA / "ladder" / "ladder_dense.csv"
ANSWER_KEY = DATA / "ANSWERKEY_DO_NOT_USE_AS_LADDER_INPUT"

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Generated locally, gitignored
RUNS = ROOT / "runs"
LADDER_PDBS = RUNS / "ladder_dense"        # dense_00.pdb .. dense_23.pdb (+ fpocket *_out/)
OLD_LADDER_PDBS = RUNS / "ladder"          # superseded 12-conformer ladder
P2RANK_LADDER_OUT = RUNS / "p2rank_out"
P2RANK_ANCHOR_OUT = RUNS / "anchors_out"

# Tools, installed locally (see SETUP.md), gitignored
PRANK = ROOT / "p2rank_2.4.2" / "prank"
PM_PREDS = ROOT / "ae-pocketminer" / "results" / "pocketminer"
