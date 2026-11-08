# Evaluation protocol (Project 1)

This protocol fixes how every tool is scored on every target, before any multi-target result is seen. It generalizes the KRAS G12D scoring in this repo to many targets and to tools with different output types. Constants marked **[provisional]** may be tuned on the CryptoBench training folds only, never on the test split or the curated subset.

## 1. Inputs

- **Targets.** The CryptoBench test split (222 apo structures), plus the curated subset in `data/curated_subset/manifest.csv` (35 CryptoBench test targets and the KRAS G12D anchor).
- **Evaluated structure.** The **apo** structure, which is the realistic use case, since the pocket is not yet visible. The holo structure is used only to define the answer key, as 7RPZ was for KRAS.
- **Chains.** The apo chain(s) named in the pair. Every tool sees the same prepared file (`src/p1_s1_prep.py`).
- **Preparation.** Every target is prepared the same way:
  - Start from the current RCSB mmCIF and keep the first model and the pocket chain(s).
  - Keep protein only: waters, ligands, ions and cofactors are removed. KRAS's GDP is removed too, and its 5US4 scores are unchanged.
  - Keep one conformer per atom.
  - Convert selenomethionine (MSE) to MET. Drop other non-standard residues and any residue missing backbone N, Cα or C.
  - Run PDB2PQR 3.6.1 (AMBER, PROPKA, pH 7.4), as for KRAS.
- **Ligand transfer.**
  - **Superposition:** holo Cα are paired with apo Cα by global sequence alignment, so numbering can differ. The fit is a Kabsch superposition on paired Cα within 12 Å of the ligand, refitted on the stable core (pairs deviating ≤ 3 Å) so residues that move with the opening don't distort it.
  - **QC:** a target is excluded from scoring when fewer than half its present true residues have an atom within 6 Å of the transferred ligand.
  - **Renamed ligands:** when the PDB has since renamed a ligand code, it is found by index and chain instead.
  - **Logs:** status per target in `results/p1_prep_log.csv`.

## 2. Ground truth

- **CryptoBench targets:** the pair's `apo_pocket_selection` as distributed, in author numbering (`A_123`). We do not re-derive it, so results stay comparable with CryptoBench.
- **KRAS anchor:** the existing answer key, the 25 residues within 4.5 Å of MRTX1133 in 7RPZ (`data/ANSWERKEY_DO_NOT_USE_AS_LADDER_INPUT/`).
- **Ligand position:** the holo ligand is superposed onto the apo structure by Cα alignment of the true pocket residues. Its heavy atoms and centre are the reference for distances.
- **Universe (for true negatives):** every standard residue with a Cα atom in the evaluated chain(s).

**Checked against current RCSB entries (220 test targets):**
- **Holo labels:** every CryptoBench holo pocket residue is present in today's holo file. The furthest residue of each pocket from the ligand is 4.34 Å at the median, so CryptoBench's cutoff is about 4–4.5 Å, compatible with the 4.5 Å KRAS key.
- **Apo labels:** 97.7% of apo pocket residues align (by sequence) onto the holo pocket.
- **Exclusion:** targets where fewer than half map are dropped as label/numbering mismatches (`label_map` in `meta.json`; `1tmi_3iwd` and `3bk9_7p46`).

**Chance rate.** Every target records its base rate (true residues ÷ all residues): chance AUPRC equals the base rate, and chance MCC is 0. AUPRC is also reported as lift over the base rate.

## 3. Site gate

A tool earns credit only for predictions at the true site, so a correct but irrelevant pocket elsewhere (for KRAS, the always-open nucleotide site) does not count.

- A predicted pocket is **at the site** if its centre is within **4 Å of any ligand heavy atom** (the ligand superposed onto the evaluated structure). This is the standard DCA detection criterion, so one rule serves both detection and the gate, and it needs no target-specific radius.
- **Site centre** = centroid of the Cα atoms of the true pocket residues. It is used only for the gated variant of probability tools (§4).

The gate uses the ligand position, so the localization metrics measure "how well is the true site described, given the tool put something there." The site-agnostic detection metrics in §5 complement this.

**Decision recorded (from sanity gate 1b on KRAS):**
- **Gates:** the KRAS scripts used two gates: pocket centre within 14 Å of His95 Cα in `score_anchors.py`, and any alpha sphere within 14 Å in `phase3_fpocket.py`.
- **Why 14 Å doesn't carry over:** His95 sits 7.7 Å from the ligand centre. Moved to the true-residue centroid, a 14 Å gate admits nucleotide-side pockets on 7RPZ (fpocket pockets 5 and 7), and fpocket's MCC drops from 0.953 to 0.719 for that reason alone.
- **What the 4 Å rule keeps:** exactly the SII-P pocket for both tools on 7RPZ (fpocket pocket 1, P2Rank pocket 2).

## 4. Converting tool outputs to residues

Each tool's raw output is kept untouched in `runs/p1/` (`src/p1_s2_run_baselines.py`; versions in `results/p1_tools.json`). An adapter (`src/p1_adapters.py`) turns it into two things: a **predicted residue set** and a **per-residue score**.

| Output type | Tools | Predicted residue set | Per-residue score |
|---|---|---|---|
| Ranked pockets | fpocket, P2Rank | Union of residues of all at-site pockets. fpocket: residues of its pocket-lining atoms (`pocketN_atm.pdb`). P2Rank: its `residue_ids` list | The score of the best-scoring pocket that contains the residue (fpocket pocket score, P2Rank residue probability); 0 if in no pocket |
| Per-residue probability | AE-PocketMiner, pLM-NN | Residues with probability ≥ threshold *t*, whole protein (no gate) | The probability itself |

**fpocket residue rule.** The lining-atom rule is fpocket's own residue output, the counterpart of P2Rank's list. The alternative used in `phase3_fpocket.py` (Cα within 4.5 Å of an alpha sphere) misses residues whose side chains, not Cα, line the pocket. On 7RPZ it drops recall from 0.92 to 0.40.

**Rank-limited (top-3) variant.** The primary gate credits an at-site pocket at any rank. fpocket outputs a median of 19 pockets per structure against P2Rank's 4, so it often has *some* pocket at the site (71% vs 58% of targets) but ranks it lower (median 4th vs 1st).

- **Required co-report:** every pocket tool is also scored with only its top 3 ranked pockets eligible (`gate4_top3`), which is how the output is used in practice.
- **It changes the ranking:** on the 219-target test split, mean MCC is fpocket 0.424 / P2Rank 0.342 under the primary gate, but fpocket 0.183 / P2Rank 0.327 under top-3.

**Gate sensitivity.** Published "found" criteria range over 4–12 Å (Nandakumar, NeurIPS 2026 AI4DD workshop). Every pocket-tool result is therefore also reported at gates of 4, 6, 8, 10 and 12 Å.

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
- A tool that crashes or cannot run on a target (for example, PocketMiner on some multi-chain inputs) is logged in `results/p1_run_log.csv` (prep failures in `results/p1_prep_log.csv`). It is scored as a miss in the primary analysis, and the results are also reported on the set of targets every tool completed.

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

**Sanity gate 1** (`tests/test_kras_reproduces.py`; both parts **passed** on 2026-11-08):
- **(a)** fpocket and P2Rank are re-run from scratch, and the original `score_anchors.py` reproduces `results/anchors_all.csv` (Table I) exactly.
- **(b)** KRAS is re-scored under this protocol and written to `results/kras_unified.csv`.
  - **7RPZ:** identical to Table I for all three tools (fpocket MCC 0.953, P2Rank 0.828, AE-PocketMiner 0.125).
  - **5US4, the one change:** fpocket's MCC goes from 0.318 to 0.523, for two reasons. The answer key is restricted to the 20 residues present (residues 60–64 are unresolved in 5US4), and the 4 Å gate drops two off-site pockets that the 14 Å His95 gate admitted.

## 10. Curated subset criteria

Implemented in `src/p1_s0_curate.py`; the final choice and reasons are in `manifest.csv`.

- CryptoBench test split; main holo per apo; single-chain pocket
- X-ray, resolution ≤ 2.5 Å, apo and holo
- Same oligomeric state in biological assembly 1
- Same UniProt accession on the pocket chain; length differences > 10% flagged for manual review
- Pocket ≥ 8 residues; ligand not a crystallization additive or ion
- At most one target per Pfam family; drug-like ligands preferred; at most 10 multimers
