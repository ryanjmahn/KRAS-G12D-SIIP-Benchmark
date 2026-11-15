# When to Trust a Cryptic-Pocket Prediction — Experiment Spec

## What this project is

**The parent study.** *The Ranking Is Not the Tool* establishes that three defensible
evaluation choices reorder the cryptic-pocket leaderboard on the 218-target CryptoBench
test split: the scoring rule (any-rank vs top-*n*), the metric family (MCC/F1 vs
AUPRC/AUROC), and whether per-target scores are normalized by the pocket-residue base rate.
The third is not a choice but a confound — per-target AUPRC has a floor that falls with
protein size, so it tracks size for all four tools including the sequence-only pLM-NN
baseline, which shares no machinery with the structure tools.

**What the paper currently lacks.** It is three negative results and a recommendation. It
diagnoses and stops. Its own stated next step — and the line already on the project page —
is *"a rule for when to trust a prediction."* This spec is that rule.

**Why it is the right addition.** It converts the paper from *"here are three problems"*
into *"here are three problems, and here is the one constructive thing that follows from
fixing them."* It is also precisely what the ICLR 2027 Foundation Models for Science
workshop's **"reliable evaluation"** track asks for (contamination-aware benchmarks,
uncertainty calibration, reproducible baselines) — the venue currently targeted at the
Feb 1, 2027 deadline.

**The question, stated precisely.** Given an apo structure and a tool's prediction, and
*without* access to the holo structure, can we predict whether that prediction is any good?
Equivalently: a structural biologist with budget to follow up on *k* of 218 targets wants
the *k* where the tool is right. Can we rank targets to deliver that?

**Where it stands today.** The best existing signal is the tool's own highest residue score
(`own_max`), at **AUC 0.636** for AE-PocketMiner, **0.609** for P2Rank, **0.549** for
fpocket — all evaluated *within base-rate strata*, which is the honest number. Every
perturbation-derived signal sits between 0.50 and 0.59. So there is a weak positive to beat,
not a blank slate.

---

## 1. Framing: selective prediction, not classification

Frame this as **selective prediction** (a.k.a. prediction with a reject option), which is a
standard ML setting with a standard evaluation protocol. That framing matters: it makes the
contribution legible to an ML audience rather than reading as a bioinformatics heuristic.

- A **trust score** `s(x) ∈ ℝ` is assigned per target, using no ground truth.
- At **coverage** `c`, you keep the top `c` fraction of targets by `s` and abstain on the rest.
- **Risk** is the error rate on the kept set. The **risk–coverage curve** and its area
  (**AURC**) are the headline metrics; selective accuracy at `c ∈ {0.1, 0.25, 0.5}` is the
  practical readout.

This also makes the deliverable concrete: not "a classifier," but a curve that tells a user
*how much accuracy they buy by investigating fewer targets*.

---

## 2. Feature tiers

All features must be computable from the **apo structure and tool outputs alone**. If a
feature needs the holo structure, it is leakage.

**Tier 1 — the tool's own signal** *(exists; weak baseline to beat)*
`own_max`, `own_top` (mean of top-10), `n_pred`, prediction `spread`. For pocket tools also:
top pocket score, number of pockets, pocket volume, and the **score gap between pocket 1 and
pocket 2** (a margin signal, currently uncomputed and the kind of thing that often carries
more information than the top score alone).

**Tier 2 — cross-tool consensus** *(THE untested idea; highest expected value)*
You have fpocket, P2Rank, AE-PocketMiner and pLM-NN on every target and have never asked
whether they agree.
- pairwise **Jaccard** between tools' predicted residue sets
- **DCA between tools' predicted centres** (do they point at the same place?)
- **Spearman correlation** between per-residue score vectors
- **consensus core size**: residues predicted by ≥ k tools
- **structure-vs-sequence agreement**: does a structural tool agree with pLM-NN? This pair
  is the most informative because the two share *no machinery* — agreement is genuinely
  independent evidence, which is exactly the property an ensemble of five MLPs lacks.

**Tier 3 — structural context** *(partly tested; weak)*
`n_res`, B-factor statistics, secondary-structure fractions, SASA, radius of gyration,
compactness. **The base rate itself must NOT be a feature** — it is the confound the paper
identifies. It enters only as the stratification variable.

**Tier 4 — conformational stability** *(tested; weak, include for completeness)*
`rho`, `jaccard`, `sd` from the perturbation ladders. Expect ~0.50–0.59; include so the
comparison is complete and the earlier negative result is carried forward honestly.

---

## 3. Target definitions

Report all three; (b) is the practically meaningful one.

- **(a) Quality-threshold:** is the target's MCC above a fixed absolute threshold (0.3) —
  and separately, above the tool's own median?
- **(b) Decision-relevant:** does the tool's **top-10 residue set contain ≥1 true pocket
  residue**? This is the question a user actually faces before committing effort.
- **(c) Continuous:** AUPRC lift over base rate, as a regression target.

---

## 4. Evaluation protocol — dictated by the paper's own Leg 3

This is the part most likely to be got wrong, and the paper's own finding specifies it.

- **Splits:** train on the **846 CryptoBench training targets**, test on the **218 held-out
  test targets**. Use CryptoBench's family-disjoint folds for internal validation. No tuning
  on test, ever.
- **Stratify by base rate.** Evaluate within base-rate quintiles and report the stratified
  number as primary. An unstratified trust model will learn protein size and post a flattering
  AUC that means nothing — this is Leg 3 reappearing one level up, and a reviewer who read
  your Section 6 will check for it immediately.
- **Report both** raw and stratified AUC side by side, exactly as the paper does elsewhere,
  and say in the text that stratified is the honest one.
- **CIs by target bootstrap**, Holm correction across the feature-set comparisons.
- **Per tool.** A trust rule is tool-specific; fit and evaluate one per tool, and report
  whether a single shared rule transfers.

---

## PHASE 0 — Feature extraction

- [ ] `t0_residue_sets.py` — re-run `src/p1_adapters.py` over the cached raw tool outputs in
      `runs/p1/` to emit, per (target, tool), the **predicted residue set** and the
      **per-residue score vector**. These are computed at runtime today and not persisted;
      Tier 2 is impossible without them. Persist to `results/trust/residue_sets.parquet`.
- [ ] `t0_features.py` — build the per-(target, tool) feature table across all four tiers.
      Join to the existing `p2_reliability.csv` columns rather than recomputing them.
- [ ] `t0_targets.py` — compute the three target definitions from `p1_master.csv`.
- [ ] **Leakage assertion:** no feature may be derived from the holo structure, the ligand,
      the true residue set, or any metric computed against them. Write this as a test over
      the feature table's provenance, not as a comment.

**EXIT CRITERIA**
- [ ] Feature table covers 846 train + 218 test targets × 4 tools, no missing-value columns
      silently imputed
- [ ] Leakage test passes
- [ ] Tier-1 features reproduce the published `own_max` AUCs (0.636 / 0.609 / 0.549
      stratified) — if they do not, the pipeline is wrong before any modelling

---

## PHASE 1 — Does cross-tool consensus carry signal?

The single highest-value question, answerable before any model is fit.

- [ ] `t1_consensus.py` — compute every Tier-2 feature; report each one's **stratified AUC**
      against each target definition, with bootstrap CIs.
- [ ] Specifically isolate **structure-vs-pLM-NN agreement** as its own row. Independent
      evidence is the mechanism that would make consensus work where within-tool
      perturbation failed; if any consensus feature carries signal, this is the one.
- [ ] `t1_compare.py` — consensus features vs `own_max` vs base rate, same strata, same CIs.

> **Novelty check before you write.** Moore 2026 (v2) already reports *detector consensus*
> for **coverage** (union of four detectors reaching 92.2%), and its title names "the limits
> of detector consensus." Your question is different — consensus as a **target-level trust
> signal for selective prediction**, not as a coverage-union strategy. State that distinction
> explicitly in the paper; do not claim consensus itself as novel.

**EXIT CRITERIA**
- [ ] Every Tier-2 feature has a stratified AUC with CI against all three targets
- [ ] A one-paragraph verdict: does cross-tool agreement beat `own_max`, and by how much

---

## PHASE 2 — The combined trust rule

- [ ] `t2_model.py` — fit **deliberately simple** models on the 846 training targets:
      logistic regression (L2 and L1) and a small gradient-boosted tree. Nothing deeper —
      with ~846 targets and a weak signal, a large model will overfit and the paper's whole
      posture is anti-overclaiming.
- [ ] `t2_eval.py` — evaluate on the 218 held-out targets: stratified AUC, **risk–coverage
      curve, AURC**, selective accuracy at c ∈ {0.1, 0.25, 0.5}.
- [ ] `t2_ablate.py` — leave-one-tier-out ablation. The decision-relevant question is whether
      **Tier 2 adds anything over Tier 1**; that comparison is the result.
- [ ] `t2_transfer.py` — does a rule fit on one tool transfer to another? A shared rule would
      be a much stronger claim than four bespoke ones.

**EXIT CRITERIA**
- [ ] Held-out risk–coverage curves for every tool, with a random and an `own_max` baseline
- [ ] Tier-2 contribution isolated with CIs
- [ ] Honest statement of the best achievable stratified AUC and what it buys at c = 0.25

---

## PHASE 3 — Calibration and the deliverable

- [ ] `t3_calibrate.py` — isotonic or Platt calibration on a training-fold split; report
      **ECE and reliability diagrams**, plus Brier score. A trust score that is not
      calibrated is not usable as a threshold, and calibration metrics are the vocabulary
      the reliable-evaluation audience expects.
- [ ] `t3_rule.py` — emit the actual artifact: a function `trust(apo_structure, tool_output)
      → (score, calibrated_probability, recommended_action)` with a documented operating
      point and its measured precision.
- [ ] `t3_kras.py` — apply it to the KRAS G12D anchor as a worked example and report what it
      would have said about each tool's prediction on 5US4 and 7RPZ. This closes the loop
      with the paper's existing close-up.

**EXIT CRITERIA**
- [ ] Calibration reported with ECE and reliability diagram
- [ ] A runnable rule released with the code
- [ ] KRAS worked example written

---

## 5. How to read the outcome — decide before you look

- **Consensus works (stratified AUC ≳ 0.72):** the headline is *"cryptic-pocket predictions
  are not self-certifying, but they are cross-certifying."* Strong constructive close, and
  it explains *why* within-tool perturbation failed while cross-tool agreement succeeds —
  independent machinery produces independent errors; five MLPs with one architecture do not.
  This is the same lesson as the paper's Leg-2 metric-family split, one level up.
- **Consensus adds little (≈ 0.65, i.e. no better than `own_max`):** the honest headline is
  *"no cheap signal certifies a cryptic-pocket prediction, including agreement between
  independent tools."* This is a **complete** negative — it closes the space of cheap
  remedies rather than leaving it open — and it sits naturally beside Moore 2026's "limits
  of detector consensus." Report it as a result, not a failure.
- **Mixed by tool:** likely, given `own_max` already ranges 0.55–0.64. Report per-tool and
  resist averaging into a single flattering number.

Pre-commit to publishing whichever occurs. The pull toward tuning until something works is
exactly the failure mode the parent paper is about.

---

## 6. Pitfalls

**K1 — Stratification is not optional.** An unstratified trust model learns protein size.
Your own Leg 3 proves per-target AUPRC tracks the base rate at ρ = 0.49–0.62 across tools
including sequence-only pLM-NN. If you report a raw AUC as the headline you have reproduced
the confound you published a paper about.

**K2 — Leakage through the target definition.** Targets (a)–(c) are computed from ground
truth, which is correct — but no *feature* may touch it. Watch for indirect paths: anything
derived from `p1_master`'s `precision/recall/mcc/auprc` columns, or from the transferred
ligand position, is contaminated.

**K3 — 846 training targets is small for a learned rule.** Use regularized linear models
first; report the tree only if it beats them on held-out. With a weak signal and a modest
*n*, a flexible model will fit noise and the held-out number will look fine by chance if you
peek. Fix the model class on training folds before touching the 218.

**K4 — Family leakage.** CryptoBench's folds are family-disjoint; use them. A trust rule that
memorizes families will look excellent and transfer to nothing.

**K5 — Consensus features can be near-duplicates of `own_max`.** If all four tools fire
strongly on easy targets, consensus may just be re-measuring difficulty. Check the
correlation between Tier-2 features and `own_max` before claiming independent contribution,
and report the partial contribution, not the marginal AUC.

**K6 — Selective-prediction metrics need matched coverage.** Comparing AUC at different
coverages is meaningless. Fix the coverage grid up front and compare like with like.

**K7 — The KRAS anchor is n = 1.** The worked example in Phase 3 is an illustration, not
evidence. Do not let it carry a claim.

---

## 7. Where it goes in the paper

Recommended: a **fourth section** of the existing paper, after the three legs and before
Recommendations — framed as *"the constructive consequence."* The title still holds, since
the trust rule is downstream of getting the evaluation right, and §4's protocol is only
valid because Leg 3 established the stratification requirement.

If it grows past ~1.5 pages, or if Phase 1 produces a strong positive worth developing
properly, it becomes a second paper and the parent stays as-is. Decide at the end of
Phase 2, from the size of the result.

**Integrity:** the agent generates scripts; the author verifies every number against the
data and writes all prose. Maintain the prompt log and figure-creation statements already
established for this project.
