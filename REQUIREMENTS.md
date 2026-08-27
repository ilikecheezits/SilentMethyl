# SilentMethyl — requirements tracker

Living record of (A) the seven strengthening requirements from mentor feedback and
(B) the stated criteria of each candidate journal. Update the **Status** and
**Evidence** columns as work lands. Evidence should point at a path under
`results/` or `data/external/`, never at a claim.

Status vocabulary: `not-started` · `in-progress` · `done` · `n/a`

Plan: Stage A (data, no GPU) → Stage B (inference on frozen checkpoints) →
Stage C (one training pass) → Stage D (re-score + write).

---

## A. Mentor's seven requirements

| # | Requirement | Needs retraining? | Stage | Status | Evidence |
|---|---|---|---|---|---|
| 1 | Multi-cohort testing | No — inference | B.2 | **done** | `results/journal/tcga_tumor_domain_shift/` |
| 2 | Repeated chromosome-blocked splits | **Yes — the only one** | C.1 | not-started | |
| 3 | Stronger baselines and ablations | No / cheap re-heads | A.2, B.5, C.2–C.3 | not-started | |
| 4 | Uncertainty calibration | No — post-hoc | B.3 | **done (analysis)** | `results/journal/rc_uncertainty{,_conditional}/` |
| 5 | Ancestry analyses | No — analysis | B.1 | in-progress | `results/journal/genoa_variant_evaluation/` (AFR arm) |
| 6 | Independent variant evaluation | No — inference | B.1 | **done** | `results/journal/genoa_variant_{scoring,evaluation}/` |
| 7 | Regulatory enrichment | No — inference | B.4 | **done** | `results/journal/motif_disruption/` |

### Detail

**1. Multi-cohort testing.** eGTEx 9 tissues; public GEO breast EPIC/450K series;
TCGA-BRCA tumours as a shifted domain; ENCODE/Roadmap WGBS. Zero-shot scoring of
existing checkpoints. Include the head-to-head against DeepMethylation's published
numbers (avg AUROC 0.909, EPIC R² 0.58, genome-wide AUROC 0.618, WGBS R² 0.17–0.20).

*Result, script 21 (2026-08-26, `results/journal/tcga_tumor_domain_shift/`).*
699 unpaired TCGA-BRCA tumours, zero GPU.

**Critical framing.** Model inputs are reference sequence and MCF-10A context;
neither depends on disease state, so **predictions are identical for normal and
tumour**. This holds predictions fixed and swaps the target. It measures the
sequence-and-context-determined component of the tumour methylome. Do **not**
describe it as the model generalising to tumours — a reviewer who notices the
inputs are unchanged will read that as overclaiming.

| model | β MAE normal → tumour | ROC-AUC normal → tumour |
|---|---|---|
| fusion | 0.0975 → 0.1180 | 0.9696 → 0.9448 |
| sequence | 0.1079 → 0.1296 | 0.9590 → 0.9308 |
| epi | 0.1395 → **0.1396** | 0.9188 → 0.9105 |

65% of held-out probes move less than 0.05 between normal and tumour; error rises
monotonically with how far the target moved (0.0769 → 0.4082 across shift bins).

**Open check — the epi model does not degrade at all**, and its M MAE improves
(1.4565 → 1.3992). Two explanations: genuine robustness of coarse chromatin
signal, or the bounded-range compression of §4 reappearing because tumour β is
less bimodal. Resolve before claiming either; if it is compression, that is a
third instance of our own methodological finding inside our own results and
belongs in the discussion. Also compare tumour-vs-normal error *within* shift bins
(`metrics_by_target_shift.csv`) — low-shift probes may simply be easier probes.

**2. Repeated chromosome-blocked splits.** Design: 4 additional folds at seed 42
only, combined with the existing 3-seed chr8–9 fold. Report seed SD from fold 1
(n=3) and fold SD across 5 folds (n=5). 12 runs, not 45.
*Known limitation:* no fold × seed interaction estimate. State this explicitly in
the methods rather than letting a reviewer find it.

**3. Stronger baselines and ablations.** Three layers:
- L1 competitors on our data — CpGenie (`gifford-lab/CpGenie`), DeepCpG (`PMBio/deepcpg`)
- L2 our model on their data — DeepMethylation's 9-tissue EPIC/WGBS setup
- L3 neutral ground — BEND (`frederikkemarin/BEND`), 7 tasks, 13 models pre-scored
Plus CPU baselines: CpG density, GC, k-mers, manifest annotation, and
**neighbouring-probe methylation** (currently absent; strongest known predictor).
Ablations: window at 400 / 1000 / 2000 bp only; all fusion-mechanism and
feature-group variants are free re-heads on frozen towers.

**4. Uncertainty calibration.** Temperature scaling on validation; conformal
prediction for distribution-free intervals on Δβ̂; ECE, Brier, PICP, CRPS.
Deep ensembles and a heteroscedastic head were dropped — both need retraining and
conformal gives a coverage guarantee without either.

*Result, script 16 (2026-08-24, `results/journal/rc_uncertainty/`).*
Spearman(FWD–RC disagreement, |error|) = **0.50** for sequence and fusion
(0.35–0.39 for epi), consistent across all 3 seeds, block-bootstrap CIs tight.
So the disagreement genuinely tracks local error.

But on selective prediction the ranking was:
`boundary_distance` < `combined` < `cross_seed_sd` < `rc_disagreement` << `random`.
The zero-parameter control **−|β̂ − 0.5| beat everything in 8/9 runs, including
the 3-seed ensemble SD.** RC disagreement beat random 9/9 but lost to cross-seed
SD 9/9 and to the heuristic 9/9.

*Resolved, script 17 (2026-08-24, `results/journal/rc_uncertainty_conditional/`).*
**The heuristic's advantage was the bounded-range artifact.** Evidence:

| | β error (bounded) | M error (unbounded) |
|---|---|---|
| boundary within-stratum ρ | +0.1569 | **+0.0461** |
| cross_seed_sd within-stratum ρ | +0.1381 | +0.1179 |
| rc_disagreement within-stratum ρ | +0.1069 | +0.0997 |
| cross_seed_sd beats boundary (stratified AURC) | 5/9 | **9/9** |
| rc_disagreement beats boundary | 0/9 | **9/9** |

boundary_distance's within-stratum correlation collapses 3.4× when the target is
moved to the unbounded logit (M) scale, while both real estimators hold. On M,
both beat the heuristic 9/9 in stratified selection and add incremental value
(cross_seed_sd 6/9, rc_disagreement 7/9). Absolute β error is mechanically
compressed near 0 and 1 — the heuristic was measuring headroom, not difficulty.

**Decisions that follow:**
1. Report uncertainty on **M-value error**, not β. State why explicitly.
2. Expected ordering holds on M: cross_seed_sd > rc_disagreement > boundary.
   Uncertainty estimation works; it was the β metric that was confounded.
3. **RC disagreement retains ~75% of the ensemble's incremental information**
   (partial ρ +0.0936 vs +0.1238) from a SINGLE model at zero extra cost.
   Report as a cheap single-model alternative, not as an ensemble replacement.
4. Conformal intervals still carry the section — coverage holds regardless.

**Spin-off contribution (free, worth a paragraph + supplementary figure):**
*Uncertainty evaluation on bounded bimodal targets is confounded by range
compression.* The field routinely reports β MAE; any model that is merely
confident at extreme β̂ will look well-calibrated on β and be exposed on M. We
recommend logit-scale evaluation. This generalises beyond SilentMethyl and costs
nothing to state.

*Robustness confirmed, script 17 strata sweep + script 18 figure
(`results/journal/rc_uncertainty_conditional_s{20,50}/`, `rc_uncertainty_figure/`).*
The 10-decile β ambiguity WAS under-stratification. Within-stratum ρ, pooled:

| estimator | β s=10 | β s=20 | β s=50 | M s=10 | M s=20 | M s=50 |
|---|---|---|---|---|---|---|
| cross_seed_sd | +0.1381 | +0.1049 | +0.0915 | +0.1179 | +0.1020 | +0.0952 |
| rc_disagreement | +0.1069 | +0.0789 | +0.0678 | +0.0997 | +0.0859 | +0.0793 |
| boundary_distance | +0.1569 | +0.0852 | **+0.0369** | +0.0461 | +0.0217 | **+0.0099** |

At 50 strata both real estimators beat the heuristic on **both** scales
(β: cross_seed_sd 9/9, rc_disagreement 8/9; M: 9/9 and 9/9). Requirement 4 is
closed — no outstanding checks.

**The crisp diagnostic to state in the paper:** a genuine uncertainty signal ranks
error about equally well on either scale. cross_seed_sd (+0.0915 β / +0.0952 M) and
rc_disagreement (+0.0678 / +0.0793) are scale-stable; boundary_distance decays to
near zero (+0.0369 / +0.0099). Scale-instability is the signature of a metric
artifact. Figure: `rc_uncertainty_figure/uncertainty_scale_stability.{png,pdf}`.

RC disagreement retains **~75–83%** of the ensemble's signal depending on measure
(partial ρ +0.0936 vs +0.1238 = 76%; within-stratum at s=50, 0.0793/0.0952 = 83%)
from a single model at zero extra cost.

**5. Ancestry analyses.** Cross-ancestry replication from published summary
statistics. GoDMC, the 2024 East Asian/European study and the gnomAD audit were all
dropped — GoDMC and gnomAD bought less than they cost in GB, and the mQTL summary
statistics already carry ancestry-matched allele frequencies (`af_genoa`) while the
HM450 manifest carries `MASK_snp5_common` / `MASK_snp5_GMAF1p`.

**The AFR arm is done** (see §6): GENOA is African American, and the requirement-6
result above *is* the African American replication — 4,037 genome-wide significant
pairs, ρ = +0.152, distance-matched AUROC 0.570. The comparison arm is eGTEx
(European-dominant), already in `results/journal/egtex_mqtl_positive_control/`.

**TCGA is not powered and must not carry this requirement.** GDC open ancestry calls
joined to the 97 training normals give EUR 84, AFR 4, SAS 1, unlabelled 8
(`data/external/tcga_ancestry/ancestry_summary.json`). Any per-group error estimate
on n=4 is noise. Report the group sizes, state plainly which strata are powered, and
rest the claim on GENOA vs eGTEx.

*Remaining:* the formal GENOA-vs-eGTEx contrast on a common metric, with the caveat
that the two differ in tissue as well as ancestry, so the comparison is not a clean
ancestry contrast and should not be presented as one.

**6. Independent variant evaluation.** From n=81 to millions. Report signed ρ,
direction agreement, AUROC, magnitude correlation — stratified by CpG-alteration
status, distance bin, effect decile, cohort ancestry.

*Cohort built, scripts 19 + `data/build_genoa_scoring_input.py` (2026-08-25).*
GENOA (Shang et al., *Nat Commun* 2023; Zenodo 10.5281/zenodo.7697509), African
American blood cohort, harmonised to hg38. 5.3 GB raw → 941,455 pairs → **66,495
held-out pairs, 39,657 unique variants, 19,081 of 26,570 test probes**. Allele
mismatch 0.1% (731/942,186), which certifies the liftover. CpGs were never lifted:
probe IDs are platform-stable and join the hg38 manifest directly.

**Effect-allele hazard, fixed.** GENOA is GEMMA output, so `beta` is keyed to the
minor allele; the model's Δ is keyed to hg38 REF→ALT. **11.3% of pairs (106,628)
have REF as the minor allele**, and for those the two run opposite. Always compare
against `beta_genoa_ref_to_alt`. Raw `beta_genoa` mixes conventions, does not
raise, and silently deflates both headline metrics.

*Result, script 20 (2026-08-25, `results/journal/genoa_variant_evaluation/`).*
Seed ensemble, non-CpG-altering, 500 block-bootstrap resamples over 1 Mb blocks.

| metric | fusion | sequence |
|---|---|---|
| signed ρ, p<5e-8 | **+0.1523** [+0.117, +0.188] | +0.1497 [+0.109, +0.184] |
| direction agreement | **0.5534** [0.536, 0.570] | 0.5541 [0.538, 0.573] |
| AUROC, distance-matched | **0.5700** [0.554, 0.585] | 0.5623 [0.546, 0.578] |
| AUROC, marginal | 0.6002 [0.586, 0.614] | 0.6026 [0.589, 0.617] |
| AUROC, distance alone | 0.5953 [0.583, 0.609] | — |

**Report the distance-matched AUROC, not the marginal one.** Significant meQTLs sit
closer to their CpG (median 191 vs 258 bp), |Δ| is larger for nearer variants
(ρ = −0.29), and distance *alone* classifies at 0.5953 — statistically
indistinguishable from the model's marginal 0.6002. State that explicitly and then
show the matched result: the signal is orthogonal to distance, not absent.
Volunteering the baseline is far stronger than having it extracted in review.

**Dilution gradient — the evidence the signal is real.** GENOA ships every cis pair
tested, not the meQTLs discovered; 71% of held-out non-CpG-altering pairs have
p > 0.05, which is why the pooled ρ is only 0.056.

| stratum | n | signed ρ | direction |
|---|---|---|---|
| p < 5e-8 | 4,037 | +0.1523 [+0.117, +0.184] | 0.5534 |
| 0.05 – 0.5 | 15,891 | +0.0232 [+0.006, +0.040] | 0.5084 |
| **p > 0.5** | 13,330 | **−0.0037 [−0.020, +0.014]** | **0.4966** |

The null stratum is indistinguishable from zero on both metrics — a negative
control inside the same data. Middle strata (0.085 / 0.099 / 0.097) are not
strictly monotone and their CIs overlap; the meaningful contrast is significant
versus null. Figure: `plots/significance_gradient.{png,pdf}`.

**Honest framing.** ρ ≈ 0.15 and 55% direction agreement is a *weak* predictor.
What makes it publishable is scale, independence and rigour — 4,037 genome-wide
significant pairs from a different tissue, ancestry and platform generation, on
probes the model never saw, against the previous n=81 and AUROC 0.512.

**Second spin-off contribution (see §4 for the first).** *Gated multimodal fusion
improves absolute methylation prediction but contributes nothing to variant-effect
prediction, because the context features are allele-invariant.* Paired equivalence
intervals, fusion − sequence: signed ρ +0.0026 [−0.0020, +0.0070]; direction
−0.0007 [−0.0064, +0.0051]; AUROC within distance bin −0.0021 [−0.0053, +0.0011].
Tight intervals, so this is equivalence, not failure to detect. Mechanism: the
context vector is identical for REF and ALT, so in
Δ = [dna_mut·g_mut − dna_wt·g_wt] + epi·[g_mut − g_wt] it survives only through a
gate shift driven by the sequence change. `gate_modulation.csv` closes that route
too — flat across all four gate-share quartiles, eight of eight intervals covering
zero, largest |effect| 0.007 (a non-significant trend in the predicted direction,
worth one sentence and no more). Applies to every sequence-plus-context model doing
variant scoring.

**Consequence: do not acquire blood ENCODE tracks.** The blood/breast mismatch is
not what limits variant-effect prediction here; allele-invariance is. Tissue-matched
context would not change these numbers.

**7. Regulatory enrichment.** JASPAR CORE vertebrates (877 matrices, 831 tested)
scanned on both strands across the 42,866 non-CpG-altering held-out GENOA pairs.
For each pair the best wild-type hit covering the variant is found, and the mutant
is scored *at that same site* so "disruption" cannot be the motif relocating.

*Result, script 22 (2026-08-26, `results/journal/motif_disruption/`).*

**The positive finding: ETS-family motif disruption predicts hypermethylation.**
Per-factor coupling = Spearman(Δ relative motif score, Δ M̂) among covered pairs:

| factor | n | ρ | +GC/dist | +substitution | motif GC |
|---|---|---|---|---|---|
| ELF4 | 1,005 | −0.2659 | −0.2683 | **−0.2814** | 0.55 |
| FEV | 2,068 | −0.2567 | −0.2592 | −0.2568 | 0.56 |
| EHF | 1,711 | −0.2512 | −0.2533 | −0.2515 | 0.51 |
| ELF1 | 1,352 | −0.2416 | −0.2457 | −0.2326 | 0.54 |
| ETV1 | 1,606 | −0.2372 | −0.2416 | −0.2401 | 0.49 |
| GABPA | 968 | −0.2330 | −0.2352 | −0.2296 | 0.50 |

Negative coupling = motif weakening accompanied by predicted **hyper**methylation,
the direction expected for factors whose binding protects CpGs from methylation.

**It survives all three artifact controls, which is why it is reportable:**

1. *Not a global offset.* Median coupling across all 831 motifs is **−0.0037**,
   IQR [−0.0453, +0.0376]; 51.4% negative; 185 significantly negative vs 155
   significantly positive at q<0.05. The null is centred and symmetric, so the ETS
   group sits ~5 IQRs outside it.
2. *Not GC or distance.* Partials are unchanged or stronger. There is a library-wide
   GC gradient (ρ = −0.20), but ETS motifs are mid-GC (0.44–0.65); if GC drove it,
   SP1-like motifs at GC 0.85–0.90 would top the table. They do not.
3. *Not substitution composition.* ETS cores are purine-rich, so weakening one
   usually means a G/A→C/T change, and pyrimidine-rich sequence is generally more
   methylated. Residualising on canonical substitution class (6 categories) changes
   nothing.

**Three framing rules — violating any of them invites correction:**

- **One family, not fifteen factors.** Top-15 mean pairwise Jaccard of covered sets
  is **0.278**, max **0.792**. Write "the ETS core motif, represented by N JASPAR
  matrices with mean pairwise overlap 0.28."
- **Recovery of known biology by an unsupervised route, not new biology.** ETS and
  GABPA sites are an established hallmark of unmethylated CpG-island promoters. The
  claim is that a model never shown a motif recovered this de novo, and that it
  holds on held-out probes in a different tissue, ancestry and platform generation.
- **The canonical panel is enriched, not missed.** The 18 known
  methylation-sensitive factors have median rank 136 of 831 — chance is 416. Say
  "recovered above chance but decisively outranked by ETS."

**The two null results belong in the same figure.** Disruption *magnitude* predicts
nothing (continuous ρ = −0.0109 [−0.0228, +0.0017]; strong vs weak median |ΔM̂|
0.0177 vs 0.0184), and meQTL discrimination is not concentrated in strong
disruptors (AUROC 0.5918 vs 0.5962, difference −0.0043, intervals overlapping).
The model did not learn "breaking motifs matters" as a general rule; it learned
something signed and family-specific. Reporting only the positive would look like
fishing.

*Method note.* A binary inside/outside-a-motif split does not work: with the full
library scanned, **100%** of variants fall inside some occurrence at the
conventional 0.80 cutoff, leaving no background. Q1 and Q3 use a top-versus-bottom
quartile contrast on disruption magnitude, matched on exact distance and GC
quintile.

---

## A2. Retraining plan — improved training data

Retraining is back on the table (2026-08-26). This section exists so the feature
decision is made **before** the requirement-2 folds are run, not after.

### The scheduling point that dominates everything else

Requirement 2 needs four additional chromosome-blocked folds — a training campaign
that has to happen regardless. Any change to the training data should ride along
with it. Deciding features afterwards means training twice.

**And the expensive part is reusable.** The sequence tower sees only DNA; changing
*context* features leaves it untouched. Per seed that is ~10 h context + ~20 h
fusion instead of the ~60 h a full retrain implies — the 30 h DNABERT-2 fine-tune
is not repeated. Re-scoring Stage B against new checkpoints is one command per
script, because 19–22 were written against a frozen manifest.

### Tier 1 — do these. Cheap, and each is justified by one of our own results.

**1. Select checkpoints on M-value MAE, not β MAE.** `run_config.json` currently
records `checkpoint_metric: regression_head_beta_mae_after_RC_averaging`. We
demonstrated in §4 that absolute β error is compressed near 0 and 1 — so the
checkpoint is currently chosen on the metric we published a paper section arguing
is confounded. One-line change; costs nothing; and leaving it as-is is the kind of
inconsistency a reviewer enjoys finding.

**2. Multi-scale context features.** The seven tracks are averaged over a 100-bp
window while the sequence model sees 1,000 bp. Extracting each track at 100 bp /
1 kb / 10 kb gives 21 features instead of 7, and window-mean phyloP at the same
scales replaces two single-base values. No new downloads — this is what
`data/reference/*.bw` was retained for. Deterministic CPU work, array-able by
chromosome.

**3. Sequence-derived context features — the direct fix for §6.** Add CpG count,
GC fraction, and CpG observed/expected computed from the **input** window rather
than from a reference track. These change between wild-type and mutant, which
makes the context tower allele-dependent for the first time and is the only
principled route past the equivalence result. It also predicts its own test: if
the fusion-minus-sequence equivalence interval moves off zero after this change,
the mechanism we proposed is confirmed; if it does not, our explanation was
incomplete and that is worth knowing.

*Note:* script 19 currently builds one fixed context vector per probe. Making
context allele-dependent requires recomputing these features from the mutant
sequence at scoring time. Small change, but it must land with the retrain.

### Tier 2 — consider, decide on evidence

- `MASK_snp5_common` / `MASK_snp5_GMAF1p` as a **feature flag**, not a filter.
  A common SNP under the probe body makes β unreliable; flagging keeps the probe
  universe intact, and changing the universe would break comparability with every
  result already in hand.
- `n_samples_observed` as a feature, or coverage-weighted loss. Script 03 found
  predictions stable at lower coverage, so expected value is low — but it is free.

### Tier 3 — do NOT do

- **Multi-task training on mQTL effects.** Circularity risk against our own
  evaluation, and drastic by any measure.
- **Changing the probe universe.** Every existing result would need re-deriving.
- **Adding tissues.** Scoped out deliberately; DeepMethylation already holds that
  ground.

### On adding a truly-healthy baseline cohort

Raised 2026-08-26. The motivation is real and it is already a stated limitation in
the manuscript: targets are medians of 97 **tumour-adjacent** normals, which may
carry field effects and cell-composition heterogeneity.

**hg19 is not an obstacle.** Array data joins on probe ID, which is
platform-stable — the same fact that let the GENOA harmonizer lift SNP positions
without ever lifting a CpG. A GEO series gives probe ID → β; join to the hg38
manifest and the coordinates come along. No liftover, no chain file. This concern
can be dropped. It only returns if the cohort is WGBS or RRBS, which is a
different measurement with coverage-dependent noise and should be treated as a
separate question.

**Small n is disqualifying for training and irrelevant for validation.** A
20–40-sample median is noisier than the existing 97-sample median. Swapping it in
trades a known bias for added variance, which is very likely a net downgrade. But
a validation cohort does not need many samples — it needs to be independent.

**So use it as a validation cohort, and the limitation becomes a result.** Score
the frozen checkpoints against a truly-healthy cohort:

- comparable performance → field effects are not materially contaminating the
  targets, stated with evidence instead of listed as a caveat;
- better → interesting, and worth explaining;
- worse → the contamination is real and now quantified.

All three outcomes are publishable, and it serves requirement 1 at the same time.
`GSE213478` (eGTEx methylation) is already on the candidate list in §D and is the
obvious first look, since eGTEx donors are not breast-cancer patients.

**If it is to touch training at all, the non-drastic use is target weighting, not
target replacement.** Probes where the healthy cohort and TCGA normals disagree
strongly are the field-effect-suspect probes. Down-weighting them, or supplying
the disagreement as a feature, uses a small cohort for what small cohorts are good
at — identifying unreliable targets — without asking it to define targets.

**Two confounds to name before interpreting any disagreement:**

- *Cell composition.* Reduction-mammoplasty and post-mortem tissue differ from
  tumour-adjacent tissue in epithelial/stromal/adipose fractions. A β difference
  is not automatically evidence of field effects.
- *Age.* TCGA normals come from cancer patients and skew older; methylation is
  strongly age-dependent. Check whether the cohort ships age metadata, and if it
  does, condition on it.

### HAZARD: the reverse-complement swap is positional, and it will break silently

Read this before touching `TABULAR_FEATURES`.

`make_rc_context()` — in `scripts/05_matched_background.py` and mirrored in
`scripts/19_genoa_variant_scoring.py` — builds the reverse-complement context
vector like this:

```python
rc_tab[:, -2], rc_tab[:, -1] = tab[:, -1].clone(), tab[:, -2].clone()
```

It swaps the **last two columns by position**, relying on
`TABULAR_FEATURES` ending with `Target_Base_PhyloP_100way_1` and `_2` — the C and
G of the target CpG, which genuinely do exchange under reverse complementation.

Append any new feature to the end of that list and the swap silently exchanges the
wrong two features on every RC pass. Nothing raises. Every RC-averaged prediction
in the project becomes subtly wrong, and it would be almost undetectable after the
fact.

**Fix before adding features, not after:** derive the swap indices from the feature
names rather than from position, e.g. resolve `PHYLOP_1` / `PHYLOP_2` through
`TABULAR_FEATURES.index(...)` once and swap those. Both copies of
`make_rc_context` must change together. Ten lines, and it makes the feature list
safe to extend.

*Related, but loud rather than silent:* `EpigeneticEncoder` and
`EpigeneticOnlyModel` default to `tabular_dim=9`, and existing checkpoints were
trained at 9 features. Loading a 9-feature checkpoint into a 21-feature model
fails on `strict=True`, which is the behaviour we want — it cannot pass unnoticed.

Note also that most proposed new features are strand-symmetric (window averages,
GC fraction, CpG counts do not change under reverse complementation), so only the
two positional phyloP features need swapping at all. Keeping the RC transformation
explicit and name-based makes that assumption visible instead of implicit.

### Order of operations

0. **Decide the healthy-cohort question first** (previous subsection). It is a
   search task: which HM450/EPIC healthy-breast series has usable sample counts
   *and* age metadata. The answer changes whether step 2 also produces a
   target-quality weight column.
1. Fix the positional RC swap (hazard above). Do this before anything else touches
   the feature list.
2. Confirm the feature list — Tier 1, plus any Tier 2 that survives.
3. Re-extract context features → new `train/val/test.csv`. CPU, array by
   chromosome. Entry point is `data/build_training_data.py`; the feature list
   itself lives in `scripts/training_common.py`.
4. Change the checkpoint metric to M-value MAE in the three
   `scripts/01_train_*_journal.py` files.
5. **One** training campaign: 3 seeds × (context + fusion) on the new features,
   *plus* the four requirement-2 folds, submitted together. The sequence tower is
   not retrained unless its input changes.
6. Update `make_rc_context` consumers and script 19's per-probe context vector,
   which currently assumes context is fixed per probe rather than per allele.
7. Re-run scripts 16–18 and 19–22 against the new checkpoints.
8. Keep the current checkpoints and results. The old model becomes the ablation
   showing what the new features bought — a better paper than quietly replacing it.
9. Rebuild `supplementary_package/` and the `reproducibility/` audits, which are
   stale as of 2026-08-26 and predate scripts 19–22.

### One thing to verify first

The manuscript states none of the 418,486 probes are flagged by `MASK_general`.
485,577 − 418,486 = 67,091, which matches the number SeSAMe masks on HM450, so
this almost certainly means masking was applied upstream and the survivors are
clean. Worth a one-line confirmation against the manifest before building new
training data on that assumption.

---

## B. Journal criteria

| Journal | IF | Stated criterion | Met by | Status |
|---|---|---|---|---|
| Nature Machine Intelligence | 29.8 | Broadly generalizable AI framework validated across several genomic tasks | B.5 (BEND, 7 tasks) | at-risk — see note |
| Nature Genetics | 25.5 | Major genetic or disease discovery from large independent public cohorts | — | not pursued (deliberate) |
| Nature Communications | 18.1 | Cross-tissue + cross-cohort + cross-ancestry validation, plus substantial new biological findings derived computationally | B.1, B.2, B.4 | **triad met; biology clause now has a candidate (ETS, §7)** — arguable, not safe |
| Genome Medicine | 10.8 | Clearer clinical relevance, multiple external cohorts, comparisons with leading methylation and variant-effect predictors | D.2, B.2, L1–L3 | best-served target |

### Notes on risk

**NMI — weakened by the compute compression.** BEND uses a frozen-embedding
protocol, so we enter as an *embedder*, and our embedder is DNABERT-2, which is
already on the BEND leaderboard. The gated fusion cannot help on BEND tasks
because BEND supplies no epigenomic context features. "Several genomic tasks" is
satisfied in letter, not in spirit. Treat NMI as a low-probability first
submission, not a plan target. Recovering it would require multi-task *training*,
which was cut on cost grounds.

**Nature Communications — now arguable, still not safe.** The validation triad is
met. The biology clause has a candidate: the ETS-family result in §7, which
survives three artifact controls and sits five IQRs outside a centred null.

Honest ledger, both sides:

*For.* A model given no motif information recovered a specific, signed,
family-level regulatory relationship, on held-out probes in a different tissue,
ancestry and platform. Three independent controls fail to explain it. Two
additional methodological contributions (§4 range compression, §6 allele-invariance)
generalise past this model.

*Against.* Predictive performance is modest (ρ ≈ 0.15, distance-matched AUROC
0.570), and the marginal AUROC does not beat a distance-only baseline. The ETS–CpG
island relationship is **established biology**, so this is unsupervised recovery
rather than discovery. And it is one motif family, not a broad regulatory map.

*Realistic read.* Perhaps a 1-in-4 shot at Nature Communications; Genome Medicine
remains the high-probability outcome. Submitting to Nature Communications first
costs roughly 6–8 weeks on rejection and nothing else, so the sequencing is:
Nature Communications → Genome Medicine or Nucleic Acids Research.

*What would move the odds materially:* a second, independent biological finding
that is not recovery of known biology — a tissue-differential pattern from the
cross-tissue design, or a characterised variant class nobody has described.

**Genome Medicine — now the strongest fit.** Two of three criteria are met
emphatically ("multiple external cohorts" and "comparisons with leading
predictors" are the best-supported parts of the whole plan). Only clinical
relevance is thin, and it is the cheapest gap to close (one ClinVar figure,
~1 week). Mentor called this "the best-matching journal above 10" — the
compressed plan happens to optimise for exactly that.

**Also in range:** Nucleic Acids Research (~15) is above the >10 bar, is the
natural venue for a benchmark-plus-resource paper, and has materially better odds
than anything Nature-branded.

---

## C. Prior art to check before claiming novelty

| Territory | Status | Holder |
|---|---|---|
| Multi-modality variant effect, 1 Mb | taken | AlphaGenome, Borzoi, Enformer — **none predict methylation** |
| Multi-tissue sequence+context methylation with variant deltas | taken | DeepMethylation (9 tissues) |
| Methylation foundation models on profiles | adjacent | MethylGPT, CpGPT — no sequence input, cannot score variants |
| RC-equivariant architectures | taken | Shrikumar 2017; Zhou 2022; Caduceus 2024 |
| RC-consistency as fine-tuning objective | taken | RCCR (arXiv 2509.18529) — excludes variant effects and methylation |
| Sequence→methylation at foundation-model scale | **open** | the gap AlphaGenome left |
| Calibrated genomic variant effects | **open** | no major predictor ships uncertainty |

**Correction on record:** SilentMethyl averages FWD and RC predictions, and
½[f(x) + f(RC(x))] is exactly RC-invariant. The reported outputs are *not*
strand-inconsistent. Do not claim otherwise. An equivariant trunk is justified
only on efficiency (1× vs 2× inference), not correctness — default is skip.

---

## D. Data acquisition status

Run `python -u data/acquire_external_cohorts.py --check` for live status.
The frozen manifest is written to `data/external/external_manifest.json`.

| Source | Build | Liftover needed | Acquisition | Status |
|---|---|---|---|---|
| GENOA meQTL | hg19 | **yes** | automatic (Zenodo) | **done** — raw deleted, 30 MB harmonized retained |
| TCGA-BRCA tumours (from the matrix on disk) | hg38 | no | none | **done** — 699 unpaired of 791 |
| TCGA ancestry calls (GDC open) | n/a | no | manual | **done** — not powered, see §5 |
| ClinVar GRCh38 VCF | hg38 | no | automatic | **done** |
| JASPAR CORE vertebrates | n/a | no | manual | **done** |
| HOCOMOCO core | n/a | no | manual | **done** |
| CpGenie, DeepCpG | n/a | no | git clone | **done** — environments not yet stood up |
| GoDMC mQTL | hg19 | yes | manual | **dropped** — redundant with GENOA + eGTEx |
| gnomAD population AF | hg38 | no | manual | **dropped** — `af_genoa` + HM450 SNP masks cover it |
| BEND task data | hg38 | no | git clone | **dropped** — see NMI note in §B |
| ENCODE blood chromatin tracks | hg38 | no | manual | **not needed** — see §6, allele-invariance |

Stage A is closed. 31 GB → 26 GB, 728 → 629 files after cleanup. Raw GENOA is gone;
provenance survives via SHA-256 in `external_manifest.json` plus the Zenodo DOI.
`data/reference/*.bw` (14 GB) is deliberately retained: the C.2 window ablation may
need to re-extract context features at 400 and 2,000 bp.

**Build hazard.** GENOA and GoDMC are hg19; this project is hg38. The chain file
`data/reference/hg19ToHg38.over.chain.gz` is already in the repo. Liftover happens
in the harmonization step and writes new files — raw downloads stay byte-identical
to the published release so they remain checksum-verifiable against the source.
