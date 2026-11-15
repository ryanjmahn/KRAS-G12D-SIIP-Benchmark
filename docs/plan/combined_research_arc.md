# Combined Research Arc — Cryptic-Pocket Prediction: Diagnose → Treat

> **Two-project arc, one coherent story.** Project 1 (benchmark extension)
> DIAGNOSES a systematic flaw in cryptic-pocket predictors. Project 2 (method)
> TREATS it. Shared infrastructure makes "both" tractable. Sequential, not
> parallel: the diagnosis defines the method's objective AND provides its
> evaluation harness.
>
> **INTEGRITY (read first):** A coding agent may generate scripts/infrastructure.
> But research framing, written prose (intro/methods/interpretation/conclusions),
> titles, and every claim's verification against data are the author's own work.
> Each venue's AI-use policy must be checked before submission. Maintain a prompt
> log; disclose AI-assisted code; write figure-creation statements.
>
> **BANDWIDTH REALITY:** This is two papers' worth of work on top of OpenWM
> (high momentum), a16z Speedrun, HAA, possible June 2027 SF move. Sequence it
> and protect OpenWM time. Trying to run both cryptic projects AND OpenWM at full
> intensity simultaneously is the failure mode.

---

## 1. The arc in one paragraph

Existing cryptic-pocket predictors (geometry, general-ML, cryptic-specialist)
behave inconsistently, and the specialist does not reliably beat the generalist
(shown on KRAS; the Jan-2026 literature independently documents the same
"fast but qualitative / low-specificity" weakness). **Project 1** establishes
this systematically across many targets and formalizes a representation-agnostic
evaluation protocol (diagnosis). **Project 2** builds a lightweight augmentation
layer that corrects the dominant diagnosed failure — over-prediction / poor
specificity — by re-ranking/calibrating existing tools' outputs toward the true
cryptic site (treatment). The benchmark is the method's motivation, training-data
source, and evaluation harness.

---

## 2. Why diagnose-then-treat (not from-scratch model)

- A method needs a REASON to exist. The benchmark proves the flaw is real and
  systematic → the method is "the justified fix," not "yet another model."
- The diagnosis defines the method's exact objective (improve specificity/MCC),
  its success metric (beat baselines on the benchmark's own metrics), and its
  evaluation harness (the benchmark pipeline) — all for free.
- A from-scratch GNN to "beat PocketMiner" is lab-scale, compute-heavy, competes
  head-on with the tool authors, and IGNORES the author's own finding. Not
  tractable solo and less coherent.
- Strategic identity: prior accepted work (TAE "Contaminated Ancestors", AIDaR
  "MedMNIST") is evaluation-audit. This arc evolves that from "find the flaw" to
  "find the flaw AND build the fix" — critic → builder, credibly.

---

## 3. Shared infrastructure (the efficiency that makes "both" feasible)

Build ONCE in Project 1, reuse in Project 2:

| Asset (built in P1) | Reused in P2 as |
|---|---|
| Multi-target panel (apo/holo + ground-truth residues) | Train/val/test data for the method |
| Unified scoring layer (detection + localization metrics) | How the method is evaluated vs baselines |
| Baseline tool outputs (fpocket, P2Rank, PocketMiner) | The method's INPUT (it augments these) |
| Per-tool failure-mode tags (fragmentation/over-spread/offset) | The method's design targets |
| Structure-prep pipeline | Feature extraction for the method |

Design P1 WITH P2 in mind: choose the panel so it can double as method
train/eval data (sufficient size, clean splits, no leakage); design the scoring
layer to be callable on the method's outputs too.

---

## 4. PROJECT 1 — Benchmark extension (DIAGNOSIS)

**Goal:** establish, across N≈15–25 cryptic pockets, that predictors
over-spread / conflate detection with localization, via a representation-agnostic
evaluation protocol.

(Full build spec lives in the separate build-spec doc. Summary of stages:)
- s0 panel assembly (criteria: clean apo+holo, known ligand, <2.5 Å, matched
  construct + oligomeric state — the MAT2A dimer lesson)
- s1 uniform structure prep
- s2 run tools, preserve RAW outputs
- s3 unified scoring: detection metrics + localization metrics (precision,
  recall, MCC, DCA) + threshold-independent measure for probability tools
- s4 (optional) conformational-robustness across NMA ladders
- s5 cross-target aggregation + proper statistics + failure-mode tagging
- s6 figures + reproducibility bundle
- SANITY GATE: reproduce prior KRAS numbers before scaling the panel

**Output that feeds P2:** the panel, the harness, the baseline numbers, and —
critically — the characterization of the DOMINANT failure mode across targets
(hypothesis: over-spread / low specificity). THAT defines what P2 builds.

**Target venue:** ICML/NeurIPS 2027 evaluation or molecular-ML workshop.

---

## 5. PROJECT 2 — Specificity/calibration method (TREATMENT)

> Lock the exact design AFTER P1 reveals the dominant failure mode. The plan
> below assumes "over-spread / poor specificity" (the current hypothesis). Stay
> open: let the cross-target diagnosis tell you what to build.

**Goal:** a lightweight augmentation layer that sharpens existing predictions
toward the true cryptic site, improving specificity/MCC without a from-scratch model.

**Design sketch (augmentation, not replacement):**
- INPUT: an existing tool's per-residue signal (e.g., PocketMiner probabilities)
  + structural/geometric context features (local geometry, conservation if
  available, residue burial, neighbor scores).
- MODEL: a small, trainable re-ranker/calibrator (candidates, simplest-first):
  - (a) a calibration/post-hoc layer (isotonic / Platt / temperature scaling)
    that recalibrates probabilities against structural context
  - (b) a lightweight per-residue classifier (shallow GNN or MLP on local
    structural features + the base tool's score) trained to suppress background
    false positives
  - (c) a context-aware re-ranking that scores residues relative to the
    protein-wide distribution (directly attacks "everything looks 0.5")
- TRAINING SIGNAL: ground-truth residues from the P1 panel (true pocket =
  residues within ~4.5 Å of the holo ligand).
- SPLITS: strict target-level train/val/test (no protein-family leakage across
  splits — a reviewer WILL check this).

**Success criterion (unambiguous, from P1's harness):**
- Improve MCC / precision / specificity vs. the raw base tool on HELD-OUT targets,
  without destroying recall. Report honestly if it only partially works.

**Honest risks:**
- The layer may not meaningfully beat the baseline → that's a negative result,
  less publishable as a "method." Budget iteration; be ready to report honestly.
- Data leakage is the classic method-paper killer — enforce clean splits.
- Keep scope at "augment existing tools." Drifting to "train a big new model"
  exits the tractable zone.

**Target venue:** ICML/NeurIPS 2027 molecular-ML workshop (method track); a
strong version could aim at a bioinformatics venue later.

---

## 6. Sequencing + timeline (sequential with overlap)

```
Nov 2026   P1 scoping: panel (with P2 in mind) + evaluation-protocol spec
Dec 2026   P1 build: pipeline + adapters; SANITY GATE on KRAS
Jan 2027   P1 run: full panel; master results
Feb 2027   P1 analysis + stats + FAILURE-MODE DIAGNOSIS  <-- defines P2
           P2 kickoff: lock method design from the diagnosis; assemble
                       train/val/test splits from the panel
Mar 2027   P1 writing (author) + identify ICML workshops/deadlines
           P2 build: the augmentation layer; train/eval on the harness
Apr 2027   P1 submit (workshop). P2 iterate: beat baselines on held-out targets
May 2027   P2 writing (author) + submit (workshop); optional preprints
```

Reality check: if OpenWM/Speedrun/HAA dominate, slip P2 to the NeurIPS 2027
workshop cycle (~Aug/Sep deadline). P1's harness carries over; nothing wasted.

---

## 7. What makes each paper publishable (reviewer-anticipation)

**P1 (benchmark):**
- [ ] N large enough for generalization; diverse families/mechanisms
- [ ] Representation-agnostic scoring (the methodological contribution)
- [ ] Proper cross-target statistics
- [ ] Reproducibility bundle; positioning vs CryptoBench/CryptoSite/PocketMiner

**P2 (method):**
- [ ] Clean, leakage-free target-level splits
- [ ] Beats the base tool on held-out targets on the diagnosed metric
- [ ] Ablations (what part of the layer does the work)
- [ ] Honest failure analysis (where/why it doesn't help)
- [ ] Motivated explicitly by P1's diagnosis (cite the arc)

---

## 8. Literature anchors (read; they ground both projects)

- PocketMiner (Meller 2023, Nat Commun) — the specialist baseline
- "AI-based cryptic pocket methods are fast and qualitative vs quantitative
  simulations" (biorxiv Jan 2026) — documents the exact weakness P2 treats
- CryptoBench (Bioinformatics 2025) — panel data source + prior benchmark
- "Decrypting cryptic pockets with physics-based simulations and AI"
  (Curr Opin Struct Biol 2025) — field review
- Accelerating Cryptic Pocket Discovery w/ AlphaFold (Meller 2023) — MD-free
  ensemble context for the conformational axis

---

## 9. The one honest decision still open

The arc is sound and the shared infrastructure is real — but it is still two
papers on top of OpenWM. Do P1's November scoping first (cheap, reversible). If
it energizes you AND your bandwidth holds, continue into the build. If OpenWM
needs you, the scoping carries over to a later cycle. Protect the highest-momentum
work; don't let a two-paper arc crowd out OpenWM by default.
