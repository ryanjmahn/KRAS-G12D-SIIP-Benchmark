# Evaluation protocol (Project 1)

This protocol fixes how every tool is scored on every target, before any multi-target result is seen. It generalizes the KRAS G12D scoring in this repo to many targets and to tools with different output types. Constants marked **[provisional]** may be tuned on the CryptoBench training folds only, never on the test split or the curated subset.

## 1. Inputs

- **Targets.** The CryptoBench test split (222 apo structures), plus the curated subset in `data/curated_subset/manifest.csv` (35 CryptoBench test targets and the KRAS G12D anchor).
- **Evaluated structure.** The **apo** structure, which is the realistic use case, since the pocket is not yet visible. The holo structure is used only to define the answer key, as 7RPZ was for KRAS.
- **Chains.** The apo chain(s) named in the pair. Every tool sees the same prepared file (Phase 1 `p1_s1_prep.py`).

## 2. Ground truth

- **CryptoBench targets:** the pair's `apo_pocket_selection` as distributed, in author numbering (`A_123`). We do not re-derive it, so results stay comparable with CryptoBench.
- **KRAS anchor:** the existing answer key, the 25 residues within 4.5 Å of MRTX1133 in 7RPZ (`data/ANSWERKEY_DO_NOT_USE_AS_LADDER_INPUT/`).
- **Ligand position:** the holo ligand is superposed onto the apo structure by Cα alignment of the true pocket residues. Its heavy atoms and centre are the reference for distances.
- **Universe (for true negatives):** every standard residue with a Cα atom in the evaluated chain(s).

To check: CryptoBench's contact cutoff for pocket residues, and whether it matches our 4.5 Å. If it differs, report KRAS under both definitions.

## 3. Site gate

A tool earns credit only for predictions at the true site, so a correct but irrelevant pocket elsewhere (for KRAS, the always-open nucleotide site) does not count.

- **Site centre** = centroid of the Cα atoms of the true pocket residues in the evaluated structure. This generalizes the His95 Cα used for KRAS.
- A predicted pocket is **at the site** if its centre is within **14 Å** of the site centre.

The gate uses the answer key, so the localization metrics measure "how well is the true site described, given the tool put something there." The site-agnostic detection metrics in §5 complement this.

**Decision recorded:** the KRAS scripts used two gates (pocket centre within 14 Å in `score_anchors.py`; any alpha sphere within 14 Å in `phase3_fpocket.py`) and two fpocket residue rules. This protocol uses the pocket-centre gate and the residue rule in §4 for every tool. KRAS is re-scored under it, and the difference from Table I is reported (§9).

## 4. Converting tool outputs to residues

Each tool's raw output is kept untouched in `results/raw/`. An adapter (`src/p1_adapters/`) turns it into two things: a **predicted residue set** and a **per-residue score**.

| Output type | Tools | Predicted residue set | Per-residue score |
|---|---|---|---|
| Ranked pockets | fpocket, P2Rank | Union of residues of all at-site pockets. fpocket: residues with a Cα within 4.5 Å of any of the pocket's alpha spheres. P2Rank: its `residue_ids` list | The score of the best-scoring pocket that contains the residue (fpocket pocket score, P2Rank residue probability); 0 if in no pocket |
| Per-residue probability | AE-PocketMiner, pLM-NN | Residues with probability ≥ threshold *t*, whole protein (no gate) | The probability itself |

**Pocket tools** are also scored on the **best single at-site pocket**, so taking the union across fragments can't inflate recall.

**Probability tools:**
- Scored across a threshold sweep (*t* = 0.05 to 0.95, step 0.05).
- Reported at two fixed thresholds: 0.7, the cutoff used for KRAS, and *t\**, the threshold that maximizes mean MCC on the CryptoBench **training folds** **[provisional]**.
- A second variant restricts predictions to residues within 14 Å of the site centre. This tests whether over-prediction away from the site drives the result.

**Decision recorded:** the primary score for probability tools is ungated, as for KRAS, because they make no pocket-level claim to gate on. The gated variant is a sensitivity analysis.

## 5. Metrics

All metrics are computed per target, then aggregated (§7). Residues are never pooled across targets, so large proteins don't dominate.

**Detection** (did the tool find the site?):

| Metric | Definition |
|---|---|
| Top-1 / Top-3 success | The 1st / any of the top 3 ranked pockets has its centre within 4 Å of any ligand heavy atom (the standard DCA criterion) |
| Best rank | Rank of the first pocket meeting that criterion; missing if none |
| Site hit (probability tools) | ≥ 1 true residue at or above *t* |

**Localization** (how well is the site described?):

| Metric | Definition |
|---|---|
| Precision, recall, F1 | Over the predicted residue set vs the true residues |
| **MCC (primary)** | Matthews correlation over the universe. It also credits what the tool correctly leaves out |
| DCA | Distance from the nearest at-site pocket centre to the ligand centre (Å). Pocket tools only |
| Fragments | Number of at-site pockets |
| Spread ratio | Predicted residues ÷ true residues |

**Threshold-independent** (any tool with a per-residue score): **AUPRC** (primary, since true residues are rare) and AUROC, computed per target over the universe.

**Empty predictions:**
- A tool that predicts nothing at the site gets recall 0 and MCC 0, and is counted as a detection miss.
- A tool that crashes or cannot run on a target (for example, PocketMiner on some multi-chain inputs) is logged in `results/raw/failures.csv`. It is scored as a miss in the primary analysis, and the results are also reported on the set of targets every tool completed.

## 6. Failure-mode tags

These tags are applied per tool per target, from the metrics above, and may overlap. Cut-offs are **[provisional]** and fixed on the training folds before test results are read.

| Tag | Rule |
|---|---|
| Miss | Nothing at the site (no at-site pocket, or no true residue ≥ *t*) |
| Fragmentation | ≥ 3 at-site pockets **and** best-single recall < 0.5 × union recall |
| Over-spread | Precision < 0.3 **and** spread ratio > 2 |
| Offset | At-site pocket present but DCA > 4 Å **and** recall < 0.5 |
| Success | MCC ≥ 0.5 and none of the above |

## 7. Statistics across targets

- **Per tool:** mean and median of each metric, with 95% bootstrap confidence intervals (10,000 resamples of targets).
- **Between tools:** paired Wilcoxon signed-rank tests on per-target MCC and AUPRC, Holm-corrected across all tool pairs, with the matched-pairs rank-biserial correlation as effect size.
- **Detection rates:** proportions with Wilson 95% intervals; McNemar's test between tools.
- **Failure modes:** the share of targets carrying each tag per tool, to test whether the KRAS failure modes recur elsewhere.
- **The test split is used once.** All tuning (*t\**, tag cut-offs) happens on the training folds first.

## 8. Conformational robustness (curated subset only)

- **Ensemble:** normal-mode ladder from the apo structure, using the Phase 2 KRAS method (ProDy, lowest-frequency mode, graded displacement). Optionally, subsampled AlphaFold conformers. The holo structure is never an input (the 7RPZ quarantine rule).
- **Openness:** fpocket volume of the pocket nearest the site centre, as for KRAS. Because fpocket is also a benchmarked tool, openness is also reported as the Cα RMSD of the true pocket residues to the holo structure, which is independent of every tool.
- **Per tool per target:**
  - detection rate across conformers
  - mean and SD of MCC across conformers (instability)
  - Spearman correlation of MCC with openness (does the tool only work once the pocket is open?)
- **Subtle openings (G4):** targets whose apo-to-holo pocket Cα RMSD is in the lowest third of the subset are analysed separately.

## 9. Splits and leakage

- **Splits:** CryptoBench's official folds, which group proteins by sequence similarity. They were reproduced exactly by `p1_s0_data.py`. Test is held out. Training folds 0–3 are for tuning here and for training in Project 2.
- **Curated subset:** drawn from the test split only.
- **Leakage check** (`tests/test_no_leakage.py`, sanity gate 2). It fails if any test target shares a UniProt accession with a training target, or if any test–train chain pair exceeds 30% sequence identity (MMseqs2) **[provisional]**.
- **KRAS:** not in CryptoBench. Before KRAS is used to evaluate Project 2, check the training folds for Ras-family proteins (Pfam PF00071) and report any found.

**Sanity gate 1** (`tests/test_kras_reproduces.py`):
- (a) Under the original KRAS scripts, the pipeline reproduces `results/anchors_all.csv` within 0.01. On 7RPZ: fpocket recall 0.920 and MCC 0.953; P2Rank recall 0.720 and MCC 0.828; AE-PocketMiner recall 0.400 and MCC 0.125.
- (b) KRAS is then re-scored under this protocol and every change from (a) is explained.

## 10. Curated subset criteria

Implemented in `src/p1_s0_curate.py`; the final choice and reasons are in `manifest.csv`.

- CryptoBench test split; main holo per apo; single-chain pocket
- X-ray, resolution ≤ 2.5 Å, apo and holo
- Same oligomeric state in biological assembly 1
- Same UniProt accession on the pocket chain; length differences > 10% flagged for manual review
- Pocket ≥ 8 residues; ligand not a crystallization additive or ion
- At most one target per Pfam family; drug-like ligands preferred; at most 10 multimers
