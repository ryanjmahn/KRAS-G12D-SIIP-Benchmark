<div align="center">

# KRAS G12D Cryptic-Pocket Benchmark

Geometry outperforms machine learning on an open cryptic pocket — and the ranking of cryptic-pocket
predictors depends as much on how you score them as on the tools.

Experiment complete · CJSJ quarterfinalist, under review · [Read the paper](docs/paper.pdf)

</div>

Three MD-free pocket predictors on the KRAS G12D Switch-II pocket, then four tools across all of CryptoBench.
The SII-P is cryptic: it forms only when a drug binds, and it is the site of the G12D inhibitor MRTX1133.

## The KRAS study

Two GDP-bound crystal structures (5US4 closed, 7RPZ open), plus a 24-conformer ladder built from the closed
structure along its lowest normal mode. Answer key: the 25 residues within 4.5 Å of MRTX1133 in 7RPZ. 7RPZ is
quarantined — never an input to the ladder. MRTX1133 redocked at **0.96 Å RMSD**.

| Tool | 7RPZ (open) MCC | 5US4 (closed) MCC | Mean MCC across the ladder |
|---|---|---|---|
| fpocket | **0.95** | 0.32 | 0.33 |
| P2Rank | 0.83 | 0.00 | **0.41** |
| AE-PocketMiner | 0.12 | 0.02 | 0.10 |

- Once a cryptic pocket has formed it is geometrically ordinary, and fpocket finds it.
- The cryptic-pocket specialist ranked last on both structures; it predicts *which residues open*, not where
  a ligand binds, so this is a task mismatch rather than a defect.
- Detection rate alone inverts the conclusion. Localization metrics correct it.
- Volume is necessary but not sufficient: P2Rank missed 4 conformers at open-state volume.

*Limits: normal modes only, so side chains are approximate; fpocket supplies both a benchmarked tool and the
openness axis; one pocket on one target, so the ranking should not be generalized.*

## Extension: how cryptic-pocket predictors are compared

Scales the KRAS finding to CryptoBench and asks what the KRAS study could not — how much of a tool's standing
comes from the tool, and how much from the choices made when scoring it. 218 test targets, four tools
(fpocket, P2Rank, AE-PocketMiner, pLM-NN), 846 training targets for the perturbation work.

**Three evaluation choices each reorder the leaderboard, and each survives its own sensitivity check.**

- **The scoring rule.** fpocket beats P2Rank on MCC when a pocket at any rank counts (+0.080); P2Rank wins
  when only the top 3 may (−0.145). fpocket wins at every site gate from 4–12 Å, P2Rank at every cutoff of
  1/3/5, so it is not a knife-edge.
- **The metric family — the whole leaderboard inverts.** By AUPRC, AUROC or base-rate-normalised AUPRC:
  pLM-NN > P2Rank > AE-PocketMiner > **fpocket last**. By MCC or F1: **fpocket first** > P2Rank > pLM-NN >
  AE-PocketMiner. Give every tool its best per-target operating point — all 19 thresholds, every gate and
  cutoff, an oracle no user reaches — and the order does not move.
- **A base-rate confound inside the threshold-free metrics.** Per-target AUPRC has a floor at the
  pocket-residue base rate, which falls with protein size (ρ = −0.604), so it tracks size for all four tools,
  pLM-NN included. AUROC and AUPRC ÷ base rate do not. Judging targets within base-rate quintiles relabels
  ~20% of them, though it leaves the rank order unchanged.

**One tool cannot rank inside its own shortlist.** fpocket scores a residue by the best pocket containing it,
so 81% of targets have a single tied score across its whole top-10. Its residue AUC of 0.492 is the absence
of a ranking, not a weak one.

**When should you believe a prediction?** These tools are unstable (fpocket's detection flips on 44% of
targets under ≤1.4 Å of backbone motion, P2Rank's on 28%) and over-spread on 48–61% of targets. Perturbing
the backbone and re-running shows which predictions survive:

- No signal — the tools' own confidence included — says whether a *target's* prediction is good. Corrected
  for the base rate, the probability tools' own scores reach AUC 0.64 and 0.61; fpocket's stays at chance.
- Per residue, stability supplies the ranking fpocket lacks: +0.070 AUC, 4.5σ against permuted labels, and
  0.146 → 0.178 precision keeping each target's two most stable top-10 residues. Tools that already rank
  gain nothing. The held-out split reproduces the effect size but is underpowered for significance.
- **Random motion matches or beats normal modes** across six measurements at two amplitudes (1.42 and
  2.84 Å), and significantly so on held-out for AE-PocketMiner. The signal is not about dynamics.

Nothing is fitted — the combined signal is a parameter-free rank-sum — so no split can leak into another.
**Target venue: ICML 2027.**

## Repo layout

```
src/  phase1_validation/ phase2_ladder/ phase3_benchmark/ phase4_figures/   the KRAS study
      p1_s0..s9_*.py    extension: data, prep, baselines, scoring, ensembles, stats, figures,
                        gate/cutoff sensitivity, the metric and base-rate analysis
      p2_s0..s6_*.py    perturbation ensembles, reliability, chunked sweep, risk-coverage,
                        figures, ablations, amplitude
      p1_adapters.py    tool output -> residue sets    figstyle.py  shared figure style
data/ structures, ligand, docking, cryptobench/, curated_subset/
results/ per-tool CSVs   figures/ p1/ (Figs 1-5) p2/ (Figs 6-8)   docs/ paper.pdf, evaluation_protocol.md
tests/ self-checks incl. test_p2_null.py (permuted-label control)   envs/ conda environments
```

## Running it

```bash
conda env create -f envs/structprep.yml && conda env create -f envs/aepocketminer.yml
```

Install P2Rank 2.4.2 and AE-PocketMiner in the repo root ([SETUP.md](SETUP.md)). The KRAS study runs through
`src/phase1_validation/` → `phase4_figures/` in order. The extension runs the `p1_s*` then `p2_s*` scripts in
numeric order; each script's docstring gives its exact command and environment.

`p2_s2_scale.py` keeps one chunk of conformers on disk at a time (~2 GB) and reduces each target to
per-residue arrays in `runs/p2_cache/` (~60 MB), so the sweep never needs the ~38 GB the full set would take.
If downloads fail to resolve `*.objectstorage.*.oraclecloud.com`, your DNS is blocking OSF's storage host.

## Stack

Python · ProDy · fpocket · P2Rank · AE-PocketMiner · AutoDock Vina · Meeko · RDKit · NumPy · pandas · matplotlib · SciPy

## Citation

```bibtex
@unpublished{ahn2026kras,
  author = {Ryan Jaemin Ahn},
  title  = {Geometry Outperforms Machine Learning on an Open Cryptic Pocket:
            Benchmarking Three MD-Free Computational Pocket Predictors on
            KRAS G12D Switch-II Pocket (SII-P)},
  note   = {Manuscript under review, Columbia Junior Science Journal},
  year   = {2026}
}
```

## License

MIT for the code. Structures (5US4, 7RPZ) and MRTX1133 (ligand 6IC) are from the RCSB Protein Data Bank.
