# Combined Research Arc (v2) — Conformation-Aware Cryptic-Pocket Prediction

> **Updated after HAA/Speedrun apps submitted → more bandwidth.** The ambitious
> version is now a realistic commitment. Two-project arc: P1 benchmark
> (DIAGNOSIS, reliable floor) → P2 conformation-aware method (TREATMENT, high
> upside). Shared infrastructure. Sequential with overlap.
>
> **INTEGRITY:** A coding agent may generate scripts/infra. Research framing,
> written prose, titles, and verification of every claim against data are the
> author's own. Check each venue's AI-use policy; keep a prompt log; disclose
> AI-assisted code; write figure-creation statements.
>
> **BANDWIDTH:** Apps done = real runway. But OpenWM (J accelerator, airfoil
> paper) still competes for deep-work hours. Deliberate split, not all-out on
> everything.

---

## 1. The thesis (why this arc, in one paragraph)

Cryptic pockets are defined by MOTION, yet every fast predictor (PocketMiner
included) scores a SINGLE STATIC structure — and the Jan-2026 literature shows
these methods are "fast but qualitative," missing subtle/dynamic openings.
**P1 (benchmark)** establishes this failure systematically across many targets
and formalizes a representation-agnostic evaluation protocol. **P2 (method)**
builds a CONFORMATION-AWARE predictor that integrates pocket signals across a
cheap MD-free conformational ensemble (the NMA ladder, already validated on
KRAS) to detect cryptic pockets static tools miss. The benchmark diagnoses the
static-structure limitation; the method treats it with dynamics.

Novelty hooks: (1) attacks a frontier gap the field just named (dynamics
sensitivity), (2) uses an asset almost no one deploys this way (cheap MD-free
ensembles as predictor input), (3) a real method, not post-processing.

---

## 2. Why conformation-aware (vs the calibration-layer version)

- Calibration layer = incremental post-processing of an existing tool's output.
- Conformation-aware = a new predictor that consumes ENSEMBLES, not snapshots —
  a substantive methodological contribution that targets the named gap.
- Uses your unique validated asset (NMA ladder); differentiates from every
  static-structure baseline.
- Coherent culmination of your original openness-focused study.

HONEST RISK (unchanged by bandwidth): it must actually beat static baselines,
especially on subtle/hard openings. If it only partially works, that's a weaker
method result. Mitigation: the benchmark (P1) is the reliable floor — it stands
alone as a publishable result even if the method underperforms.

---

## 3. Shared infrastructure (build once in P1, reuse in P2)

| Asset (P1) | Reused in P2 |
|---|---|
| Multi-target panel (apo/holo + ground truth) | Train/val/test proteins |
| Structure-prep pipeline | Per-conformer prep for ensembles |
| NMA ladder generator (from KRAS) | **Ensemble generation per target** (core P2 input) |
| Unified scoring layer (detection + localization) | Evaluation of the method vs baselines |
| Baseline tool outputs (fpocket/P2Rank/PocketMiner) | Static baselines to beat |
| Failure-mode tags | Confirms static tools miss dynamic openings (the P2 mandate) |

KEY DESIGN COUPLING: P1 must generate not just static scores but, for each
target, a cheap conformational ENSEMBLE (NMA ladder, optionally +subsampled
AlphaFold). That ensemble is both (a) P1's conformational-robustness evidence and
(b) P2's model input. Design them together from day one.

---

## 4. PROJECT 1 — Multi-target benchmark (DIAGNOSIS / reliable floor)

Goal: across N≈15–25 cryptic pockets, show static predictors (a) over-spread /
conflate detection with localization, and (b) are unstable / miss openings
across conformational states. Deliver a representation-agnostic protocol +
a conformational-robustness axis.

Stages (see build-spec doc for detail):
- s0 panel assembly (clean apo+holo, known ligand, <2.5 Å, matched construct +
  oligomeric state — MAT2A dimer lesson)
- s1 uniform prep
- s2 run static tools; preserve RAW outputs
- s2b **generate MD-free ensemble per target (NMA ladder; +subsampled AF opt.)**
- s3 unified scoring: detection + localization (precision/recall/MCC/DCA) +
  threshold-independent measure for probability tools
- s4 **conformational-robustness: score static tools across each ensemble;
  quantify prediction instability across openness** (this is both a P1 result
  AND the motivation + data substrate for P2)
- s5 cross-target stats + failure-mode tagging
- s6 figures + repro bundle
- SANITY GATE: reproduce prior KRAS numbers before scaling

Output feeding P2: the ensembles, the harness, the baselines, and the evidence
that static tools miss dynamic openings (the mandate for a conformation-aware method).

Venue: ICML/NeurIPS 2027 evaluation or molecular-ML workshop.

---

## 5. PROJECT 2 — Conformation-aware cryptic predictor (TREATMENT / high upside)

Goal: a predictor that integrates pocket signals across a cheap MD-free ensemble
to detect cryptic pockets better than static tools — especially subtle openings.

**Design:**
- INPUT: per target, a cheap conformational ensemble (NMA ladder + optional
  subsampled-AlphaFold conformers), NOT a single structure.
- PER-CONFORMER SIGNAL: for each conformer, extract features — base-tool scores
  (PocketMiner probs, P2Rank, fpocket), local geometry, residue burial,
  conservation if available.
- AGGREGATION MODEL (the novel core): learn a function that integrates the
  per-residue signals ACROSS conformers into one prediction of "this region
  forms a cryptic pocket across the accessible conformational space."
  Candidate architectures (simplest-first):
  - (a) learned pooling over conformers (attention / max / mean+variance) on top
    of per-conformer per-residue features
  - (b) a small temporal/sequence model over the ordered ladder (openness as the
    axis) — captures HOW the signal evolves as the pocket opens
  - (c) a light GNN per conformer + cross-conformer aggregation
- TRAINING SIGNAL: ground-truth cryptic residues (within ~4.5 Å of holo ligand)
  from the P1 panel. Train on apo→ensemble, predict the known cryptic site.
- SPLITS: strict target-level, family-disjoint train/val/test (no leakage — the
  classic method-paper killer; a reviewer WILL check).

**Success criterion (from P1's harness):**
- Beat static baselines (esp. PocketMiner) on MCC/precision/recall on held-out
  targets — and CRUCIALLY on the subtle/hard-opening subset (the frontier gap).
- Report honestly if it only helps on some targets / some opening magnitudes.

**Headline claim (if it works):** "Static cryptic-pocket predictors miss dynamic
openings; a conformation-aware model using cheap MD-free ensembles improves
detection of subtle cryptic pockets, without full MD."

**Ablations to plan:** static-only vs ensemble (does dynamics help?); ensemble
size; which aggregation; which base features matter.

Venue: ICML/NeurIPS 2027 molecular-ML workshop; strong version → bioinformatics
venue (e.g., IEEE BIBM) later.

---

## 6. Timeline (apps done → realistic full commitment)

```
Nov 2026  P1 scoping: panel (sized for P2 training, family-disjoint splits) +
          evaluation-protocol spec + ensemble-generation plan
Dec 2026  P1 build: pipeline + ensemble generation + adapters; KRAS SANITY GATE
Jan 2027  P1 run: full panel (static scores + ensembles + robustness)
Feb 2027  P1 analysis/stats/diagnosis -> confirms static tools miss dynamics
          P2 kickoff: build aggregation model on the ensembles + harness
Mar 2027  P1 writing (author); ID ICML workshops/deadlines
          P2 train/eval: beat static baselines on held-out, esp. hard openings
Apr 2027  P1 submit (workshop); P2 ablations + iteration
May 2027  P2 writing (author) + submit; optional preprints
```
If it runs long or OpenWM needs you: slip P2 to NeurIPS 2027 (~Aug/Sep). P1's
harness + ensembles carry over; nothing wasted.

---

## 7. Reviewer-anticipation

**P1:** N large enough; representation-agnostic scoring; conformational-robustness
axis; proper cross-target stats; reproducibility; positioning vs
CryptoBench/CryptoSite/PocketMiner.

**P2:** leakage-free family-disjoint splits; beats static baselines on held-out
(esp. subtle openings); ablations isolating the dynamics contribution; honest
failure analysis; motivated by P1's diagnosis; compute cost reported (MD-free =
cheap is a selling point vs full-MD methods like FAST).

---

## 8. Literature anchors

- PocketMiner (Meller 2023) — static specialist baseline
- "AI cryptic methods are fast + qualitative vs quantitative simulations"
  (biorxiv Jan 2026) — documents the dynamics-sensitivity gap P2 targets
- Accelerating Cryptic Pocket Discovery w/ AlphaFold (Meller 2023) — MD-free
  ensemble precedent
- CryptoBench (2025) — panel source + prior benchmark
- "Decrypting cryptic pockets with simulations + AI" (Curr Opin 2025) — review
- AlphaFlow / BioEmu — generative-ensemble context (and their realism failures)

---

## 9. The honest standing decision

Bandwidth is freed, so the ambitious arc is realistic — but OpenWM still
competes. Commit to P1 scoping in November (serves both projects, reversible).
The arc's design keeps P1 as a reliable publishable floor so the P2 method risk
never sinks the whole effort. Decide the OpenWM-vs-this split deliberately; "more
free" is runway, not infinite.
