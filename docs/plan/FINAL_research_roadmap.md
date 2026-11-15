# FINAL RESEARCH ROADMAP — Conformation-Aware Cryptic-Pocket Prediction
### Diagnose → Treat: a multi-target benchmark + a conformation-aware method

> **How to use this doc:** phase-ordered execution plan (NOT date-ordered).
> Move to the next phase when the current phase's EXIT CRITERIA are met — work
> by completion, not by calendar. Each phase names concrete scripts a coding
> agent (Claude in VS Code) can generate and run. Do not skip the sanity gates.
> (The only time-sensitive item is confirming the exact workshop deadline in
> Phase 3 when you're ready to submit — look it up then, don't plan around it now.)
>
> **INTEGRITY (non-negotiable):** The coding agent generates SCRIPTS and
> infrastructure. YOU write all paper prose, interpretation, conclusions, and
> titles; YOU verify every result against the data before any claim. Check each
> venue's AI-use policy before submitting. Keep a prompt log (AI_DISCLOSURE.md);
> write per-figure creation statements.
>
> **DATA SCALE:** Use FULL CryptoBench (~885 train / ~222 test, 4 folds) for
> method training + large-scale benchmark. Use a CURATED SUBSET (~30-40) only for
> the expensive per-target work (ensembles, conformational robustness, docking-
> usability). ~25 was never the whole panel — it is the deep-analysis subset.
>
> **SCOPE HONESTY:** Two papers. P1 (benchmark) is the reliable publishable
> FLOOR; P2 (method) is the high-upside, higher-risk layer. If P2 underperforms,
> P1 still stands. Protect OpenWM time (deliberate split, not all-out on everything).

---

## GAPS THIS WORK TARGETS (reference while writing motivation)

PRIMARY:
- G1 — static methods miss dynamics / can't handle apo conformations
- G4 — subtle / small-magnitude openings are missed
- G5 — inconsistent, leakage-prone, non-standardized evaluation

SECONDARY (diagnosed by benchmark, motivate method):
- G2 — over-prediction / low specificity
- G3 — qualitative not quantitative (no usability/druggability prediction)
- G8 — specialist tools don't reliably beat general ones

OPTIONAL:
- G6 — physical realism (NMA vs generative) · G7 — multi-chain/interface pockets

Baselines to beat / compare (field moved fast — don't benchmark PocketMiner
alone): fpocket, P2Rank, PocketMiner, pLM-NN (CryptoBench baseline), and a newer
tool (Seq2Pocket / UniPocket) if runnable.

---

## REPO SCAFFOLD (agent: create this first)

```
cryptic-conf/
  data/
    cryptobench/            # downloaded dataset + official splits
    curated_subset/         # ~30-40 targets for deep analysis (manifest.csv)
    ensembles/              # generated conformers (gitignored; re-derivable)
    ground_truth/           # true cryptic residues per target (cached)
  configs/   tools.yaml  panel.yaml  model.yaml
  src/
    p1_s0_data.py           # load CryptoBench, parse splits, build manifests
    p1_s1_prep.py           # uniform structure prep
    p1_s2_run_baselines.py  # fpocket/P2Rank/PocketMiner/pLM-NN; RAW outputs
    p1_adapters/            # per-tool output -> common residue representation
    p1_s3_score.py          # detection + localization metrics (+ threshold-indep)
    p1_s4_ensembles.py      # NMA ladders (+opt subsampled AF) for curated subset
    p1_s5_robustness.py     # score baselines across ensembles (conf. robustness)
    p1_s6_aggregate.py      # cross-target stats, failure-mode tags
    p1_s7_figures.py
    p2_s0_features.py       # per-conformer per-residue feature extraction
    p2_s1_dataset.py        # assemble training tensors; family-disjoint splits
    p2_s2_model.py          # conformation-aware aggregation model
    p2_s3_train.py          # training loop, checkpoints, logging
    p2_s4_eval.py           # eval vs baselines on held-out (uses p1_s3 harness)
    p2_s5_ablations.py      # static-vs-ensemble, ensemble size, aggregation type
    p2_s6_figures.py
  tests/
    test_kras_reproduces.py # SANITY GATE 1
    test_no_leakage.py      # SANITY GATE 2 (splits are family-disjoint)
  env/  environment_prep.yml  environment_pm.yml  environment_ml.yml
  AI_DISCLOSURE.md   README.md
```

================================================================
## PHASE 0 — SETUP & SCOPING (low compute, high value)
================================================================
Goal: data in hand, splits understood, infra scaffolded. No modeling yet.

Agent tasks / scripts:
- [ ] Scaffold repo (above). Create 3 conda envs (prep/geometry+Java17;
      pocketminer+tf; ml/pytorch).
- [ ] `p1_s0_data.py`: download CryptoBench, parse official train/test + 4 folds,
      emit data/cryptobench manifests.
- [ ] Define curated subset (~30-40): diverse families, clean apo+holo, known
      ligand, <2.5 A, matched construct + OLIGOMERIC STATE (MAT2A dimer lesson),
      includes KRAS G12D SII-P as anchor. Write curated_subset/manifest.csv.
- [ ] Write the evaluation-protocol spec (YOUR words): formal detection vs
      localization definitions; representation-agnostic scoring; threshold
      handling; the family-disjoint split rule.

Author (not agent):
- [ ] Draft 3-4 title candidates; pick a working title
- [ ] Read the core + newer baselines (pLM-NN, Seq2Pocket, UniPocket) so
      positioning is current

EXIT CRITERIA (move on only when all true):
- [ ] CryptoBench loaded, splits reproduced, manifests written
- [ ] Curated subset finalized and justified
- [ ] Protocol spec written

================================================================
## PHASE 1 — BENCHMARK PIPELINE (DIAGNOSIS; the reliable floor)
================================================================
Goal: run all baselines across the data, score consistently, prove the
static-structure + specificity + evaluation gaps systematically.

Agent tasks / scripts:
- [ ] `p1_s1_prep.py`: uniform prep (chain selection, protonation pH 7.4,
      cleanup). Same prep for all tools.
- [ ] `p1_s2_run_baselines.py`: run fpocket, P2Rank, PocketMiner, pLM-NN on
      every structure; write RAW native outputs to results/raw/. Record tool
      versions/params. (Note: PocketMiner fails on multi-chain — log, don't crash.)
- [ ] `p1_adapters/`: one adapter per tool → common predicted-residue set.
      PocketMiner adapter uses a THRESHOLD SWEEP, not one value.
- [ ] `p1_s3_score.py`: ground truth = residues within ~4.5 A of holo ligand;
      compute DETECTION (found + rank) and LOCALIZATION (precision/recall/F1/
      MCC/DCA); plus a THRESHOLD-INDEPENDENT measure for probability tools.

>>> SANITY GATE 1 (`tests/test_kras_reproduces.py`): run the pipeline on KRAS
    alone; must reproduce prior results (P2Rank ~0.80 recall / ~0.61 MCC on
    7RPZ; PocketMiner over-spreads; fpocket fragments). If not -> fix before
    scaling. <<<

- [ ] Scale to full CryptoBench test set (~222) + curated subset.
- [ ] `p1_s4_ensembles.py`: generate NMA ladders (+opt subsampled AF) for the
      CURATED SUBSET only (expensive). Reuse validated KRAS NMA code.
- [ ] `p1_s5_robustness.py`: score baselines across each ensemble; quantify
      prediction instability vs openness (the conformational-robustness axis).
- [ ] `p1_s6_aggregate.py`: cross-target statistics (paired tests, effect sizes,
      CIs — NOT eyeballing); tag failure modes (fragmentation/over-spread/offset).
- [ ] `p1_s7_figures.py`: detection+localization comparisons; robustness figure;
      failure-mode summary. Record figure-creation statements.

Author (not agent):
- [ ] Verify the diagnosis against the data: do static tools over-spread /
      miss dynamic openings / get beaten inconsistently? State honestly.

EXIT CRITERIA:
- [ ] Sanity gate 1 passed
- [ ] Master results table (tools × targets × metrics)
- [ ] Conformational-robustness result on curated subset
- [ ] Statistical aggregation done
- [ ] P1 is a standalone publishable result (floor secured)

================================================================
## PHASE 2 — CONFORMATION-AWARE METHOD (TREATMENT; high upside)
================================================================
Goal: a predictor that integrates signals across a cheap MD-free ensemble to
beat static baselines — ESPECIALLY on subtle openings.

>>> FEASIBILITY CHECKPOINT (decide before building): ensemble generation does
    NOT scale to ~1,100 proteins cheaply. Choose: (a) generate lighter/fewer
    conformers per protein so it scales to the full train set, OR (b) train on a
    large static-feature set + reserve full ensembles for the curated subset.
    Pick based on measured per-protein ensemble cost from Phase 1 s4. <<<

Agent tasks / scripts:
- [ ] `p2_s0_features.py`: per conformer, per residue — base-tool scores,
      local geometry, burial, conservation (if available).
- [ ] `p2_s1_dataset.py`: assemble tensors; enforce FAMILY-DISJOINT train/val/
      test splits (use CryptoBench folds; verify no >X% sequence overlap).
- [ ] `p2_s2_model.py`: conformation-aware aggregation (simplest first):
      (a) learned pooling over conformers (attention / mean+variance)
      (b) sequence model over the ordered openness ladder
      (c) light per-conformer GNN + cross-conformer aggregation
- [ ] `p2_s3_train.py`: train; checkpoint; log; early stopping on val.
- [ ] `p2_s4_eval.py`: evaluate on held-out via the P1 harness; compare to
      ALL baselines; report on the SUBTLE-OPENING subset specifically.
- [ ] `p2_s5_ablations.py`: static-only vs ensemble (does dynamics help?);
      ensemble size; aggregation type; feature importance.
- [ ] `p2_s6_figures.py`: method-vs-baseline; ablations; subtle-opening gains.

>>> SANITY GATE 2 (`tests/test_no_leakage.py`): confirm no train/test family
    leakage. A reviewer WILL check; leakage invalidates the method. <<<

Author (not agent):
- [ ] Verify improvements are real on held-out data; run ablations yourself.
- [ ] HONEST reporting: if it only helps on some targets / opening magnitudes,
      say so. A partial/negative result is reported truthfully (P1 still stands).

EXIT CRITERIA:
- [ ] Feasibility checkpoint resolved (ensemble scaling decided)
- [ ] Leakage-free splits confirmed
- [ ] Method evaluated vs all baselines on held-out, incl. subtle openings
- [ ] Ablations isolate the dynamics contribution
- [ ] Honest success/failure characterization written

================================================================
## PHASE 3 — WRITING & SUBMISSION
================================================================
Author-led (agent only assists with understanding + grammar on YOUR drafts):
- [ ] P1 paper (benchmark) — write to workshop standard, your words
- [ ] P2 paper (method) — write, your words; motivated explicitly by P1
- [ ] Figures finalized + creation statements
- [ ] AI_DISCLOSURE.md complete; prompt log organized
- [ ] Reproducibility bundle (code + configs + data manifest) public
- [ ] Verify each venue's AI-use policy
- [ ] Identify target workshop(s) and CONFIRM their exact deadlines when you are
      ready to submit (ICML / NeurIPS molecular-ML or evaluation workshops; a
      bioinformatics IEEE venue such as BIBM for the benchmark). This is the one
      time-sensitive lookup — do it at submission time.
- [ ] Mentor review (professor / NeurIPS-network contact) with buffer time
- [ ] Submit; optional arXiv preprint if polished + endorsement arranged

================================================================
## REVIEWER-ANTICIPATION (keep visible)
================================================================
P1: large enough N; representation-agnostic scoring; conformational-robustness
axis; proper stats; reproducibility; positioning vs CryptoBench/CryptoSite/
PocketMiner/pLM-NN.
P2: FAMILY-DISJOINT leakage-free splits; beats baselines on held-out (esp.
subtle openings); ablations isolating dynamics; honest failure analysis;
MD-free compute cost reported as a selling point vs full-MD; compared against
CURRENT baselines (not just PocketMiner).

================================================================
## STANDING DECISIONS / RISKS
================================================================
- [ ] Ensemble scaling (Phase 2 feasibility checkpoint) — the make-or-break
      engineering decision
- [ ] Method may not beat baselines — real risk; P1 is the floor that de-risks it
- [ ] Field moves fast — keep baselines current (pLM-NN, Seq2Pocket, UniPocket)
- [ ] OpenWM competes for deep-work hours — deliberate split
- [ ] Integrity: scripts via agent OK; science, verification, writing = you

================================================================
## SLIP VALVE (no fixed dates)
================================================================
If Phase 2 runs long or OpenWM needs you, pause P2 and submit P1 to the next
suitable workshop cycle on its own — the benchmark is a complete result. P1's
harness + ensembles carry over to P2 whenever you resume. Nothing is wasted by
working at your own pace rather than to a deadline.
