<div align="center">

# KRAS G12D Cryptic-Pocket Benchmark

Geometry outperforms machine learning on an open cryptic pocket.

**fpocket on the open structure (7RPZ): recall 0.92, MCC 0.95**

Experiment complete · CJSJ quarterfinalist, still under review · [Read the paper](docs/paper.pdf)

</div>

This project benchmarks three pocket-prediction tools that run without molecular dynamics (MD) on the KRAS G12D Switch-II pocket (SII-P). The SII-P is a cryptic pocket: it only forms when a drug binds, and it is the site of the G12D inhibitor MRTX1133. The tools are fpocket (pure geometry), P2Rank (general-purpose machine learning) and AE-PocketMiner (a graph neural network trained specifically for cryptic pockets). Each tool was run on the closed and open crystal structures and on a 24-conformer closed-to-open ladder, and scored against the residues MRTX1133 actually touches.

## Why

KRAS was considered undruggable for about three decades until the Switch-II pocket was found to open under a bound ligand (Ostrem et al., 2013). MRTX1133 later used the same pocket in the G12D mutant, which drives the majority of pancreatic ductal adenocarcinoma cases (Waters & Der, 2018). Because a cryptic pocket is absent from the ground-state structure, researchers rely on software to find it. Most of those tools were developed on static, visible pockets, and the cryptic-pocket specialist is usually assumed to be the right default. This project tests that assumption on a clinically important target.

## Structures, answer key and docking check

Two high-resolution KRAS G12D crystal structures, both GDP-bound, so the only difference between them is the pocket state:

| PDB | State | Resolution | SII-P volume | fpocket druggability |
|---|---|---|---|---|
| 5US4 | apo, pocket collapsed | 1.83 Å | ~245 Å³ | 0.001 |
| 7RPZ | MRTX1133-bound, pocket open | 1.30 Å | ~822 Å³ | 0.684 |

- **Answer key:** the 25 residues with an atom within 4.5 Å of MRTX1133 in 7RPZ (Switch-II, helix 3 and the P-loop end of the ligand footprint). It is defined by what the drug touches, not by any tool's output, so it is independent of the methods being tested.
- **Quarantine:** 7RPZ is used only for the answer key, the docking check and scoring. It is never an input to the ladder ([`data/ANSWERKEY_DO_NOT_USE_AS_LADDER_INPUT/`](data/ANSWERKEY_DO_NOT_USE_AS_LADDER_INPUT/)).
- **Openness metric:** SII-P volume measured by fpocket, located by position near the His95 Cα so the same site can be tracked as the structure changes.
- **Docking check:** MRTX1133 redocked into 7RPZ with AutoDock Vina recovered the crystal pose at **0.96 Å RMSD** (pass threshold ≤ 2 Å), with a predicted affinity of −12.7 kcal/mol.

## The MD-free conformational ladder

24 conformers were generated from closed 5US4 only, by displacing it in graded steps along its lowest-frequency normal mode (ProDy). This avoids MD, matching the sampling most labs can actually afford.

- SII-P volume rose from ~269 Å³ to ~750 Å³, **91% of the open crystal structure**.
- The opening is not gradual. Conformers fall into three volume clusters, and the jump from ~400 Å³ to ~750 Å³ at conformer 15 is where the Switch-II wall forms.

<p align="center"><img src="figures/fig1_volume.png" alt="SII-P volume across the two crystal structures and the ends of the ladder" width="560"></p>

## Benchmarking the three tools

All three tools were scored on the same residue set (chain A) with the same rules: a predicted pocket counts only if part of it is within 14 Å of His95, and a residue counts if it is within 4.5 Å of such a pocket. The 14 Å limit stops a tool from earning credit for the nucleotide site, which is already open. AE-PocketMiner gives per-residue probabilities, so residues at or above 0.7 count as predicted. MCC is the main metric because it also credits what a tool correctly leaves alone.

**On the crystal structures:**

| Tool | Structure | Recall | MCC | Precision | Residues flagged |
|---|---|---|---|---|---|
| fpocket | 7RPZ (open) | 0.92 | 0.95 | 1.00 | 23 |
| P2Rank | 7RPZ (open) | 0.72 | 0.83 | 1.00 | 18 |
| AE-PocketMiner | 7RPZ (open) | 0.40 | 0.12 | 0.22 | 45 |
| fpocket | 5US4 (closed) | 0.28 | 0.32 | 0.54 | 13 |
| P2Rank | 5US4 (closed) | 0.00 | 0.00 | n/a | 0 |
| AE-PocketMiner | 5US4 (closed) | 0.28 | 0.02 | 0.16 | 43 |

- On open 7RPZ, fpocket's predicted pocket centre was 2.3 Å from the ligand and P2Rank's was 3.6 Å. Both had perfect precision, so the gap between them is coverage, not false positives.
- On closed 5US4, P2Rank correctly proposed no pocket at the site. fpocket produced false positives.
- AE-PocketMiner flagged broadly on both structures. Its mean probability on true SII-P residues was 0.606 against 0.514 protein-wide on 7RPZ, and 0.537 against 0.514 on 5US4. Because this check uses no cutoff and both structures are experimental, the broad flagging is not caused by the 0.7 threshold or by the generated conformers.

<p align="center"><img src="figures/fig2_anchor_mcc.png" alt="MCC of each tool on the open and closed crystal structures" width="560"></p>

**Across the 24-conformer ladder**, the ordering changes. P2Rank has the highest mean MCC (0.41 ± 0.27), followed by fpocket (0.27 ± 0.07) and AE-PocketMiner (0.08 ± 0.03), even though AE-PocketMiner has the highest mean recall.

<p align="center"><img src="figures/fig4_mean_performance.png" alt="Mean recall and MCC across the ladder" width="560"></p>

P2Rank found no pocket within 14 Å of the site in 7 of 24 conformers:

| Volume cluster | Conformers | Volume (Å³) | P2Rank detected | Mean recall when detected |
|---|---|---|---|---|
| Closed (~270) | 10 | 269–291 | 7 | 0.33 |
| Intermediate (~400) | 5 | 396–421 | 5 | 0.33 |
| Open (~720) | 9 | 682–750 | 5 | 0.41 |

## What the KRAS results show

- **Geometry wins on an open pocket.** Once the cryptic pocket has formed, it is geometrically an ordinary cavity, which fpocket's alpha spheres find well.
- **The specialist ranked last on both crystal structures.** AE-PocketMiner predicts which residues take part in pocket opening, not where a ligand binds. KRAS is flexible in several regions, so broad flagging has a structural explanation. This reads as a mismatch between the tool's target and the task, not as a defect in the tool.
- **Detection rate alone gives the wrong answer.** By detection rate, AE-PocketMiner would look successful and P2Rank would look like it failed on the closed structure. Both readings are backwards. Localization metrics (MCC, precision, distance to the ligand), computed from the same data, correct them.
- **Volume is necessary but not sufficient.** Four of P2Rank's misses were at open-state volume while their same-volume neighbours were detected. Once the Switch-II wall is present, detection depends more on local side-chain arrangement than on pocket size. Where P2Rank did detect the site, its predicted centre moved from ~4.4 Å from the ligand in the closed cluster to 3.4–3.9 Å in the open one.

<p align="center"><img src="figures/fig5_dca_vs_volume.png" alt="P2Rank distance from predicted pocket centre to the ligand, against pocket volume" width="560"></p>

Every arm, including the two crystal structures, was re-scored through the same scoring code before the paper was written. An earlier comparison that had not been scored that way was retired as a result, and none of its numbers appear here.

## Limitations

- Conformers come from normal mode analysis alone, so side chains are approximate. fpocket's fragmentation of the pocket across the ladder may reflect that rather than the tool.
- fpocket is both a benchmarked tool and the source of the openness axis. Scoring is unaffected because the answer key comes from the drug's contacts, but the axis is not independent of that tool.
- AE-PocketMiner's broad flagging may be specific to KRAS, which is flexible in several regions.
- This is one pocket on one target, so the ranking should not be generalized.

## Extension: how cryptic-pocket predictors are compared

Cryptic pockets are defined by motion, yet fast predictors, AE-PocketMiner included, score one static
structure at a time. The extension scales the KRAS finding to the whole of CryptoBench and asks a question
the KRAS study could not: how much of a tool's apparent standing comes from the tool, and how much from the
choices made when scoring it. It is one piece of work and one paper.

**Gaps targeted.** Primary:

- static methods miss dynamics and handle apo conformations poorly
- subtle, small-magnitude openings are missed
- evaluation is inconsistent and prone to leakage

Secondary, diagnosed by the benchmark:

- over-prediction (low specificity)
- qualitative rather than quantitative output
- cryptic-pocket specialists don't reliably beat general tools

**Data at two scales.**

| Set | Size | Used for |
|---|---|---|
| Full CryptoBench | 1,107 apo structures, 5,493 apo–holo pairs; official sequence-clustered splits (885 training in 4 folds, 222 test) | Large-scale benchmark and perturbation sweeps |
| Curated subset | 35 CryptoBench test targets + KRAS G12D as anchor | Expensive per-target work: ensembles and conformational robustness |

The curated subset is drawn from the test split only. Nothing here is fitted, so no split can leak into another.

**Baselines:** fpocket, P2Rank, AE-PocketMiner, pLM-NN (the CryptoBench baseline), and a newer tool such as Seq2Pocket or UniPocket if it can be run.

Work moves phase by phase: each phase starts only when the previous one's exit criteria are met, and the sanity gates are never skipped.

### Setup

CryptoBench downloaded and its official splits reproduced exactly (`src/p1_s0_data.py`), the curated
subset chosen by RCSB metadata filters (`src/p1_s0_curate.py`), and the evaluation protocol written
down before any tool was run ([`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)).

### What the benchmark shows

Four tools (fpocket, P2Rank, AE-PocketMiner, pLM-NN) on 218 CryptoBench test targets, plus 25-conformer ensembles for the 36-target curated subset. Figures and captions: [`figures/p1/`](figures/p1/CAPTIONS.md).

- **Sanity gate 1 passed:** Table I is reproduced from scratch, and KRAS re-scored under the unified protocol is in `results/kras_unified.csv`.
- **The ranking depends on the scoring rule, and the rule is a choice.** fpocket beats P2Rank on MCC when any pocket can count; P2Rank wins when only the top-3 pockets can. Both effects are significant after Holm correction, and neither is an artefact of where the lines were drawn: fpocket wins at every site gate from 4 to 12 Å and P2Rank at every rank cutoff of 1, 3 and 5 (`src/p1_s8_sensitivity.py`).
- **The ranking also depends on the metric, and the whole leaderboard inverts** (`src/p1_s9_confound.py`). Ranked by AUPRC, AUROC or base-rate-normalised AUPRC: pLM-NN > P2Rank > AE-PocketMiner > **fpocket last**. Ranked by MCC or F1: **fpocket first** > P2Rank > pLM-NN > AE-PocketMiner. The best and worst tools swap places. This is not a badly chosen operating point: give every tool its best per-target setting — all 19 thresholds for the probability tools, every gate and rank cutoff for the pocket tools, an oracle no user reaches — and the order is unchanged, with 8 of 12 pairwise calls still disagreeing with AUPRC. Each call carries a bootstrap 95% CI over targets and is Holm-corrected within its metric.
- **A base-rate confound sits inside the threshold-free metrics.** Per-target AUPRC has a floor at the pocket-residue base rate, which falls with protein size (Spearman −0.604), so it tracks size for every tool — pLM-NN included, which shares no machinery with the structure tools (AUPRC vs base rate 0.49–0.62 across all four). AUROC (0.01–0.25) and AUPRC divided by the base rate (−0.12 to −0.01) do not. Judging targets within base-rate quintiles instead of against one global median **relabels about 20% of them** (18.8–21.6% per tool) and overturns 1 of 6 pairwise conclusions. The rank order itself is unchanged by the correction, so this is a confound in per-target claims rather than in the leaderboard.
- **pLM-NN has the best AUPRC.** The MD-trained AE-PocketMiner is significantly worse than both P2Rank and pLM-NN, so the original KRAS claim holds.
- **Each tool has a characteristic failure.** P2Rank misses, fpocket is offset, and the probability tools over-spread. The tag rules were locked on the training folds.
- **Robustness:** detection flips along the ladder for 44% of targets with fpocket and 28% with P2Rank. The lowest normal mode gives *less* best-conformer gain than a random-motion control, and fpocket volume does not track how holo-like the pocket is.


1. Prepare every structure the same way: chain selection, protonation at pH 7.4, cleanup.
2. Run every baseline on every structure, keeping raw outputs and logging failures (for example, PocketMiner on multi-chain inputs).
3. Write one adapter per tool to the common residue set. Probability tools are scored with a threshold sweep.
4. Score detection, localization and AUPRC per target.
5. **Sanity gate 1:** reproduce this repo's KRAS numbers (Table I, `results/anchors_all.csv`), then re-score KRAS under the unified protocol and explain every change.
6. Scale to the full CryptoBench test set and the curated subset.
7. Generate normal-mode ensembles (optionally with subsampled AlphaFold conformers) for the curated subset, reusing the KRAS ladder code.
8. Score every baseline across each ensemble: detection rate, how much MCC varies across conformers, and how MCC tracks openness.
9. Aggregate across targets (bootstrap confidence intervals, paired Wilcoxon tests with Holm correction, effect sizes) and tag failure modes: miss, fragmentation, over-spread, offset.
10. Figures.

### When should you believe a prediction?

The benchmark measured two things that matter to a user more than average accuracy. These tools are **unstable**: fpocket's site detection changes on 44% of targets under no more than 1.4 Å of backbone motion, P2Rank's on 28%. And they are **poorly calibrated**: at their default thresholds the probability tools mark far more residues than any pocket holds, on 48–61% of targets. The next question is what a practitioner can do about that. Given one apo structure and no holo, can you tell which predictions to trust?

The probe is perturbation. Move the backbone slightly, re-run the tool, and see what survives: a prediction that holds up under motion the protein could plausibly make is a different thing from one that appears in a single snapshot. Three generators give the motion — the lowest normal mode, a thermal mix of the ten lowest modes, and a random smooth deformation as the control — at matched amplitudes, so any signal has to beat random perturbation rather than merely beat the apo. The ensembles are a measuring instrument here, not an accuracy device: averaging a tool's scores over them moves AUPRC by only +0.01–0.02, and the random ensemble moves it just as much.

*Findings (`src/p2_s1_reliability.py`, all 846 usable training targets, 3 generators × 10 conformers each, built and reduced in chunks by `src/p2_s2_scale.py`; Holm correction over all 18 tool × generator × signal tests):*

- **Target level: the raw null is itself a confounded measurement.** Taken at face value, no signal separates well-predicted from badly-predicted targets (AUC 0.42–0.59) and neither does the tool's own confidence (0.52–0.55). But AUPRC rises with the pocket-residue base rate, which falls with protein size, so size-like quantities predict "quality" without saying anything about whether the site was found — raw protein size scores AUC 0.643 on fpocket, beating every signal we built. Within base-rate quintiles that collapses to 0.45, and the picture inverts: **the probability tools' own confidence does carry weak target-level signal** (own_max AUC 0.636 AE-PocketMiner, 0.609 P2Rank), while fpocket's stays near chance (0.549) and every perturbation signal stays at or below it (0.49–0.59). Crystallographic B-factors, the standard cheap flexibility proxy, are useless either way (0.49–0.54).
- **Residue level, a real but small gain.** Among a tool's top-10 residues, perturbation stability rank-summed with the tool's own score ranks true pocket residues above false ones better than the score alone: fpocket +0.070 AUC (own 0.492, p_holm 2×10⁻⁶), P2Rank +0.024 (own 0.571, p_holm 0.0011), AE-PocketMiner +0.010 (own 0.546, p_holm 0.15 — not significant). Stability alone, without the tool's score, only helps fpocket (+0.075, p_holm 5×10⁻⁵), whose own score is a coin flip because it ties (see the next bullet).
- **Withholding the unstable predictions helps only fpocket.** Keeping each target's two most stable top-10 residues: fpocket 0.146 → **0.178** precision (paired cluster bootstrap over targets, +0.035, 95% CI 0.021–0.050, p < 0.001), P2Rank 0.377 → 0.383 and AE-PocketMiner 0.265 → 0.267, both with CIs spanning zero. The reason is visible in fpocket's own curve, which is flat on the base rate: it scores a residue by the best pocket containing it, so **81% of targets have a single tied score across the whole top-10** and the tool cannot rank inside its own shortlist at all. Perturbation stability supplies the ranking fpocket lacks; where a tool already has one, it adds nothing.
- **The residue-level effect needs the conformers and is not an artefact of the shortlist size.** fpocket's gain climbs monotonically with the number of conformers (+0.023 at 1, +0.050 at 5, +0.075 at 10) and is still rising, so 10 conformers is a lower bound; for P2Rank and AE-PocketMiner fewer than ~5 conformers actively *hurts*. Widening the shortlist strengthens the effect rather than explaining it: fpocket +0.055 at top-5, +0.075 at top-10, +0.103 at top-20, as its own AUC falls from 0.500 to 0.477.
- **Random motion beats the normal modes on the held-out split too.** Paired over targets, random minus lowest-mode stability is positive for all three tools on both splits, and significant on held-out for AE-PocketMiner (+0.044, p 0.003; training +0.025, p 0.0005). P2Rank (+0.034, p 0.07) and fpocket (+0.030, p 0.14) are positive but not significant on 149 and 81 targets. This claim, unlike the fpocket stability gain below, does carry held-out support.
- **Held out: the effect size replicates, the significance does not.** Re-run on the official CryptoBench test split (218 targets, never used to develop any of this), fpocket's residue-level gain is +0.055 AUC against +0.075 on the training folds, and its precision at 20% coverage is +0.028 against +0.033 — the same direction and nearly the same size. Neither clears significance: the per-target AUC test keeps only the 81 test targets whose top-10 holds both classes (against 331 in training), and the coverage CI is [-0.007, +0.060] against [+0.018, +0.047]. The honest reading is that the test split reproduces the estimate but is roughly four times too small to confirm it, not that the effect failed to replicate.
- **Random perturbation works as well as the normal modes or better** in every one of the nine combinations, which makes the signal cheap to compute and rules out a dynamics explanation for it.

Perturbation-stability signals are measured against the tools' own confidence over all 846 usable
training targets and confirmed on the held-out test split, with risk–coverage curves, a permuted-label
control, and ablations over conformer count, shortlist size and perturbation amplitude. Nothing is
fitted — the combined signal is a parameter-free rank-sum — so there is no training set to leak from.

### Writing and submission

One paper covering both phases, since the findings are one argument: the ranking of cryptic-pocket
predictors moves under the scoring rule, under the metric family and under base-rate normalisation,
each robust to its own sensitivity check. A public reproducibility bundle (code, configs, data
manifest) ships with it, and a mentor reviews before submission.

**Target venue: ICML 2027.** A fuller benchmark resource — the KRAS case study, the locked failure
taxonomy and the full tool × target table — is a separate later paper for a bioinformatics venue,
a different contribution rather than a slice of this one.

## Repo layout

```
src/
  paths.py              shared file locations
  phase1_validation/    rmsd_check.py (docking check), answer_key.py
  phase2_ladder/        build_ladder_dense.py, anchor_volumes.py, validation.py
  phase3_benchmark/     phase3_fpocket.py, phase3_p2rank.py, phase3_pocketminer.py, score_anchors.py
  phase4_figures/       make_figures.py
  p1_s0_data.py         extension: download CryptoBench, reproduce splits -> data/cryptobench/pairs.csv
  p1_s0_curate.py       extension: filter candidates, draft the curated subset
  p1_s1_prep.py         uniform prep, ligand transfer, labels -> runs/p1/<target>/
  p1_s2_run_baselines.py, p1_s2b_plmnn.py   run fpocket / P2Rank / AE-PocketMiner, and pLM-NN
  p1_adapters.py        tool output -> residue sets and per-residue scores
  p1_s3_score.py        per-target metrics -> results/p1_master*.csv
  p1_s4_ensembles.py    NMA ladders and the random-motion control
  p1_s5_robustness.py   robustness across conformers
  p1_s6_aggregate.py    failure tags and cross-target statistics
  p1_s7_figures.py      figures/p1/
  p1_s8_sensitivity.py  is the fpocket/P2Rank reversal robust to the site gate and rank cutoff?
  figstyle.py           figure style shared by both phases
  p2_s0_pilot.py        perturbation ensembles (normal modes, mode mix, random control) + tool scores
  p2_s1_reliability.py  is perturbation stability a confidence signal?
  p2_s2_scale.py        the same over every training target, chunked (conformers cached, then deleted)
  p2_s3_coverage.py     risk-coverage curves: precision against the fraction of predictions kept
  p2_s4_figures.py      reliability figures, from the committed CSVs
  p2_s5_robustness.py   ablations (conformer count, shortlist size) and free confidence baselines
  p2_s6_amplitude.py    does doubling the perturbation amplitude rescue the normal modes?
data/                   structures, ligand, docking files, ladder volumes, 7RPZ answer key
  cryptobench/          pairs.csv (raw/ downloads are gitignored)
  curated_subset/       manifest.csv (final panel), candidates.csv, shortlist_draft.csv
tests/                  test_s0.py (loader self-checks)
results/                per-tool CSVs, anchor table (Table I), cluster table (Table II)
figures/                paper Figs. 1–5
docs/                   paper.pdf (the manuscript), evaluation_protocol.md (extension)
envs/                   conda environments
exploration/            superseded and one-off scripts, kept for transparency
```

## Running it

```bash
git clone https://github.com/ryanjmahn/KRASG12DBenchmarking.git
cd KRASG12DBenchmarking
conda env create -f envs/structprep.yml
conda env create -f envs/aepocketminer.yml
```

Install P2Rank 2.4.2 and AE-PocketMiner in the repo root, as described in [SETUP.md](SETUP.md). Then run the scripts in order with `conda activate structprep`. They can be run from any directory.

```bash
python src/phase1_validation/rmsd_check.py          # docking check: 0.96 Å
python src/phase2_ladder/build_ladder_dense.py      # 24 conformers -> runs/ladder_dense/
python src/phase3_benchmark/phase3_fpocket.py       # a few seconds
python src/phase3_benchmark/phase3_p2rank.py        # about 1–2 minutes
# run AE-PocketMiner on runs/ladder_dense/*.pdb in the aepocketminer env (see SETUP.md), then:
python src/phase3_benchmark/phase3_pocketminer.py
python src/phase3_benchmark/score_anchors.py        # Table I
python src/phase4_figures/make_figures.py           # Figs. 1–5 + Table II
```

`answer_key.py` regenerates the answer key, but it overwrites the notes at the end of the saved file, so it doesn't need to be re-run.

**Extension, data and setup** (same env):

```bash
python src/p1_s0_data.py            # CryptoBench JSONs (~16 MB) + split check; add --structures for test-split mmCIFs from RCSB
python src/p1_s0_curate.py          # RCSB metadata -> candidates.csv, shortlist_draft.csv
python tests/test_s0.py
```

**Extension, the benchmark** (structprep env; pLM-NN uses `envs/plm.yml`):

```bash
python src/p1_s1_prep.py && python src/p1_s1_prep.py --split train-fold-0 train-fold-1 train-fold-2 train-fold-3
python src/p1_s2_run_baselines.py
python src/p1_s2b_plmnn.py embed && python src/p1_s2b_plmnn.py predict
python src/p1_s3_score.py --split train-fold-0 train-fold-1 train-fold-2 train-fold-3 && python src/p1_s3_score.py
python src/p1_s4_ensembles.py && python src/p1_s4_ensembles.py --null
python src/p1_s2_run_baselines.py --root runs/p1_ens --log-suffix _ensemble
python src/p1_s2_run_baselines.py --root runs/p1_ens_null --log-suffix _ensemble_null
python src/p1_s3_score.py --split ensemble --root runs/p1_ens --suffix _ensemble
python src/p1_s3_score.py --split ensemble --root runs/p1_ens_null --suffix _ensemble_null
python src/p1_s5_robustness.py
python src/p1_s6_aggregate.py lock && python src/p1_s6_aggregate.py tags && python src/p1_s6_aggregate.py stats
python src/p1_s7_figures.py
```

**Extension, perturbation and reliability** (structprep env):

```bash
conda activate structprep
python src/p2_s0_pilot.py build
for g in nma1 nmaK random; do python src/p1_s2_run_baselines.py --root runs/p2_pilot/$g --log-suffix _p2pilot_$g; done
python src/p2_s0_pilot.py score
python src/p2_s2_scale.py cache && python src/p2_s2_scale.py run
python src/p2_s1_reliability.py
python src/p2_s3_coverage.py
python src/p2_s4_figures.py
python src/p2_s5_robustness.py
python src/p2_s2_scale.py run --scale 2 --gens nma1,random --tag amp2 --limit 200
python src/p2_s6_amplitude.py
python tests/test_p2_null.py
```

`p2_s2_scale.py` keeps only one chunk of conformers on disk at a time (about 2 GB) and reduces each target to per-residue score arrays in `runs/p2_cache/` (about 60 MB for the whole training set), so the sweep never needs the ~38 GB the full set of conformers would occupy.

If downloads fail with a name-resolution error for `*.objectstorage.*.oraclecloud.com`, your DNS is blocking OSF's storage; switch to a public resolver (e.g. 1.1.1.1) for the download.

## Stack

Python · ProDy · fpocket · P2Rank · AE-PocketMiner · AutoDock Vina · Meeko · RDKit · NumPy · pandas · matplotlib

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

MIT for the code. The protein structures (5US4, 7RPZ) and MRTX1133 (ligand 6IC) are from the RCSB Protein Data Bank.
