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
| 3 | Stronger baselines and ablations | No — CPU only | A.2, B.5, C.2–C.3 | **scripted, not yet run** | `scripts/23_sequence_baselines.py` |
| 4 | Uncertainty calibration | No — post-hoc | B.3 | **done (analysis)** | `results/journal/rc_uncertainty{,_conditional}/` |
| 5 | Ancestry analyses | No — analysis | B.1 | in-progress | `results/journal/genoa_variant_evaluation/` (AFR arm) |
| 6 | Independent variant evaluation | No — inference | B.1 | **done (GENOA)**; tissue-matched eGTEx arm in progress | `results/journal/genoa_variant_{scoring,evaluation}/` |
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

*Update (2026-08-27).* Once the full eGTEx Breast Mammary association file runs
through the same scorer (see §6), the contrast is between two arms measured by one
pipeline rather than between an arm and a legacy 81-variant result. It is still not
a clean ancestry contrast — tissue moves with ancestry — and the honest framing is
that the tissue-matched arm bounds model performance while the GENOA arm bounds
transfer. Presenting the difference between them as an ancestry effect would be
wrong and a reviewer will say so.

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

---

**The tissue-matched arm — eGTEx Breast Mammary Tissue (in progress, 2026-08-27).**

*Why, stated plainly.* GENOA is blood. A weak GENOA correlation is confounded with a
tissue change, so it cannot falsify the model — and a reader is entitled to ask why
the primary external validation of a breast-tissue model was run in blood. It reads
as an availability decision, because it was one. The fix is to make the
tissue-matched arm primary and demote GENOA to what it actually is: a cross-tissue,
cross-ancestry **transfer** arm, which is a real and separately interesting result.

*Source.* eGTEx methylation mQTLs, EPIC arrays, `BreastMammaryTissue.mQTLs.regular.txt.gz`
(`https://storage.googleapis.com/egtex/methylation/epic-arrays/mQTLs/`), 45,279,365,738
bytes compressed. This is the **same study** as the 81 lead variants already in
`data/egtex_breast_mqtl_heldout.csv` — the complete association set rather than the
lead subset, so it supersedes those 81 rather than adding a new cohort. Nine columns,
no header: `probeID variant_id dist ma_samples ma_count maf pval_nominal slope slope_se`.

*Three things that make this cleaner than GENOA.* Already hg38 (`b38` suffix in the
variant ID) — no liftover, no chain file. REF and ALT are encoded in the variant ID —
no minor/major ambiguity, so the effect-allele bug that silently flipped 11.3% of
GENOA pairs cannot recur in the same form. Distance is precomputed.

*One thing that is worse.* eGTEx assays **EPIC**; the model is trained on **HM450**.
Probe IDs are platform-stable, so no liftover is needed, but EPIC-only probes have no
training-data counterpart and are dropped and counted.

*Pipeline.* `data/harmonize_egtex_mqtl.py` (new). Two stages, because 45 GB:
stage 1 streams the gzip through `awk` keeping rows within ±600 bp of a probe
(~0.06% of the file — awk rather than Python because this is a multi-billion-line
scan); stage 2 does the real work in pandas. Then `scripts/19` and `scripts/20`
unchanged apart from the column generalisation below.

*What the harmonizer refuses to trust, and why each is a GENOA lesson:*

1. **Their distance column.** Distance is recomputed from HM450 `CpG_beg` and the
   `reported − recomputed` distribution is reported. A constant ±1 offset gets
   detected instead of inherited. The window bounds are **imported** from
   `data/build_genoa_scoring_input.py` (`MIN_SCOREABLE_OFFSET`/`MAX_SCOREABLE_OFFSET`),
   not restated, so the two cohorts cannot drift apart.
2. **The sign convention.** GTEx/tensorQTL documents `slope` as keyed to ALT, which
   matches the model's REF→ALT Δ. That is exactly the kind of assumption that cost us
   106,628 flipped GENOA pairs, so it is verified against an internal positive
   control: **variants that destroy the target CpG must lower methylation**. If the
   mean slope over those comes back positive at z > 3, the script prints the evidence
   and exits without writing. The check is free and non-circular — those variants are
   excluded from scoring anyway.
3. **The reference base.** Every REF is checked against `hg38.fa`; the mismatch rate
   is logged loudly, as with GENOA's 0.1%.

*Column generalisation (scripts 19 and 20, 2026-08-27).* The cohort effect column is
no longer hard-coded. Script 19 gained `--effect-column` / `--pvalue-column`
(default `auto`), resolving `beta_ref_to_alt` (eGTEx) or `beta_genoa_ref_to_alt`
(GENOA) and copying it to a canonical name. Script 20 reads the canonical names and
**back-fills from the legacy ones**, so the existing GENOA score files still run
untouched — no rescoring. Script 20 also gained `--cohort {GENOA,eGTEx}`, which
swaps only the wording and the tissue caveat so the eGTEx arm never inherits
"GENOA is blood". It changes no computation.

*Tests.* `data/_test_egtex_harmonizer.py` builds a synthetic genome, manifest, split
CSVs and all-pairs file, then asserts: window boundaries at exactly −499/+500,
masked-probe drop, EPIC-only drop, indel drop, REF-mismatch drop, split assignment,
and a **deliberately sign-flipped run that must fail** — it does. Script 20 was
separately re-run on synthetic legacy-named score files to confirm the rename is
backwards-compatible.

*Expected scale.* ~2–3M rows survive stage 1; roughly half sit on HM450 probes;
after the exact window and target-CpG exclusion the held-out (chr8–9) stratum should
land in the same order of magnitude as GENOA's 66,495. **Not yet run — do not quote
a number until `egtex_scoring_summary.json` exists.**

**PRE-REGISTERED SIGNIFICANCE THRESHOLD (declared 2026-08-28, BEFORE any eGTEx
model score existed — the GPU array was still queued).**

`data/egtex_significance_threshold.py` on the study's own permutation output:

| FDR | mCpG probes | % of 754,054 tested | calibrated nominal cutoff |
|---|---|---|---|
| 0.01 | 7,527 | 1.00% | 2.312e-06 |
| **0.05** | **13,256** | **1.76%** | **1.483e-05** |
| 0.10 | 18,923 | 2.51% | 4.348e-05 |

The cohort's own calibration is **297x less strict than the 5e-8 we had been
using**. Held-out counts:

| definition | pairs |
|---|---|
| p < 5e-8 (the GWAS constant, wrong test) | 295 |
| calibrated nominal p <= 1.483e-05 | 772 |
| **two-stage: mCpG probe AND calibrated nominal** | **596** |
| probe's own lead variant, in window | 80 |

**Declared primary definition: the two-stage set, n = 596.** Nominal-only (772)
is the sensitivity analysis. Both are reported; the threshold-free significance
gradient remains the primary evidence the signal is real.

*Power, computed before the fact.* At n = 596 a direction agreement of 0.553
carries a naive 95% interval of [0.513, 0.593] — excluding chance, where n = 295
gave [0.496, 0.610] and did not. Signed rho of 0.15 gives [0.070, 0.229]. Block
bootstrapping will widen both. So the tissue-matched arm moves from *cannot
support a claim* to *marginal but viable*. That is an honest description and the
one to use — it is not a rescue.

*Why only 1.76% of probes are mCpGs.* eGTEx Breast has roughly 50-100 donors.
This is a power ceiling in the cohort, not a defect in our pipeline: the
held-out and model-visible mCpG fractions agree at 1.8% and 1.7%.

*model_visible gives 8,508 two-stage pairs* — 14x the held-out set and
well-powered, which is what makes the memorisation contrast worth running.

*Manuscript consequence, once the numbers land.* eGTEx Breast Mammary becomes the
primary requirement-6 result; GENOA moves to a transfer subsection; the 81 lead
variants are superseded and their table row is dropped. Do not restructure the
manuscript before the numbers exist.

**3. Stronger baselines and ablations.** *Reprioritised to first place, 2026-08-27.*

*Why this moved to the front.* We report signed rho = 0.152 and 55.3% direction
agreement on held-out meQTLs and we do not know whether that is good, because no
competing predictor has been scored on the same pairs. Without a referent the
number is uninterpretable, and no additional cohort changes that — eGTEx makes
the comparison tissue-matched, it does not tell us what the ceiling is. Genome
Medicine asks for this explicitly ("comparisons with leading methylation and
variant-effect predictors"). It is the one outstanding requirement that changes
what we are allowed to claim rather than how confidently we claim it.

*What is scripted.* `scripts/23_sequence_baselines.py`, CPU-only, no GPU, no new
data:

| baseline | features | controls for |
|---|---|---|
| `composition` | GC fraction, CpG count, CpG obs/exp | is any of this better than base composition? |
| `kmer_ridge` | RC-collapsed k-mer counts, k = 1..6 (2,772 columns) | does DNABERT-2 pretraining beat classical sequence features? |

Both are exact ridge fits from streamed sufficient statistics (X'X, X'y), so the
345,359-row training split never materialises as a feature matrix; alpha is
selected on `val` by exact validation MSE computed from the same statistics.

*What makes it a fair comparison, not a strawman.* Same split CSVs, same
`M_Value_Target` column, same `centered_crop(seq, 1000)` window, same
target-CpG exclusions, and — the important one — the variant task writes
`pair_scores.csv` in scripts/19's schema, so **scripts/20 evaluates baselines and
neural models through one code path**: identical distance matching, identical
1 Mb block bootstrap, identical significance strata. No published cross-tissue
weights are involved, so no tissue handicap flatters either side.

*Two hazards, both handled.* (i) The baseline must not import the model's
dependency tree — `from training_common import centered_crop` drags in torch,
transformers and huggingface_hub for nine lines of arithmetic and makes the
comparator unrunnable on a CPU node. The two helpers are therefore duplicated
**and asserted identical to `training_common` at startup** whenever it imports.
(ii) RC-collapsed k-mer counts are exactly strand-invariant, so the RC-averaging
the neural models require is provably a no-op here; that is checked at startup
rather than stated.

*Interpretation, decided in advance.* If `kmer_ridge` matches the neural models
on variant effects, the finding is that sequence-only allelic methylation
prediction is far harder than the literature implies, and DNABERT-2 pretraining
is not what closes the gap — a real, honest, publishable benchmark result. If the
neural models win clearly, the architecture is earning its keep and we can say so
with evidence. **Both outcomes get reported. Deciding after seeing the numbers
which one to emphasise is the failure mode to avoid.**

*Tests.* `scripts/_test_sequence_baselines.py` plants a known k-mer signal in
synthetic splits, confirms the ridge path recovers it, confirms every pair-level
exclusion fires at the expected count, independently recomputes variant deltas,
and checks the emitted file carries every column scripts/20 requires.

*Still not covered by this script:* a retrained CpGenie-style CNN on our splits
(the architecture-vs-architecture comparison, ~1 GPU-hour) and the published
CpGenie/DeepCpG weights (cross-tissue, confounded, and a 2017 environment to
resurrect). Phase those after the CPU baselines return a number.

**Variant-effect synthesis (scripts/24, GENOA only, 2026-08-28).** eGTEx was
still scoring; re-run the identical command when the array lands and the
meta-analysis appears automatically.

| metric | fusion | sequence |
|---|---|---|
| signed rho | +0.152 [+0.118, +0.186] | +0.150 [+0.116, +0.182] |
| direction agreement | 0.553 [0.536, 0.571] | 0.554 [0.536, 0.571] |
| **calibration slope** | **+1.171 [+0.955, +1.435]** | +0.985 [+0.812, +1.200] |
| null control rho (p>0.5, n=13,330) | -0.0037 [-0.0205, +0.0131] | -0.0015 |

*New: the relationship is quantitative.* The calibration slope is the regression
of reported effect on predicted delta-M. Rank statistics say the ordering is
right; a slope excluding zero says predicted magnitude scales with measured
effect. **Do not read the value as calibration** -- reported effects are on an
inverse-normal scale and predictions are in M-value units, so a slope near 1 is a
coincidence of scales. OLS is attenuated by predictor error, so the true
relationship is at least this steep.

*Spatial localisation -- the second negative control.* Direction agreement by
distance (fusion, significant, non-CpG-altering):

| distance | n | direction |
|---|---|---|
| 0-50 bp | 844 | 0.568 [0.534, 0.601] |
| **50-100 bp** | 479 | **0.605 [0.559, 0.654]** |
| 100-200 bp | 761 | 0.558 [0.524, 0.594] |
| 200-300 bp | 685 | 0.545 [0.506, 0.582] |
| 300-400 bp | 642 | 0.555 [0.514, 0.598] |
| **400-501 bp** | 626 | **0.497 [0.459, 0.535]** |

Both sub-100 bp bins exclude chance; the window edge lands on it. Confounds (LD,
GC, probe properties) have no reason to decay with distance from the CpG; a
bounded receptive field must. This is an orthogonal negative control to the
association-strength null.

*Precision quintiles are FLAT -- the honest negative.* Binning the significant
pairs by |effect|/SE gives 0.521, 0.564, 0.541, 0.576, 0.566 across q1-q5, all
intervals overlapping. Only q1 covers chance. **So ~0.55-0.58 is the model's
resolution limit, not noise in the reported betas.** We cannot attribute the
modest direction agreement to a noisy reference, and should not try.

*Tissue matching is the largest single effect.* eGTEx breast leads (n=81):
rho = 0.610 [0.452, 0.731], p ~ 1e-9, 79.0% direction. GENOA blood (n=4,037):
rho = 0.152, 55.3%. **Four-fold.** Confound to resolve: the 81 are lead variants
(enriched for large, precisely measured effects) while GENOA's are all
genome-wide significant pairs, so part of the gap is lead-versus-all. The
complete eGTEx set supports lead and non-lead strata within one cohort, which
separates the two. `egtex_probe_significance.csv` carries the lead variant IDs;
80 leads fall inside the held-out window.

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

- **One 4-bp core, not fifteen factors and not even "a family."** *(sharpened
  2026-08-28 from `per_motif_coupling.csv`.)* Every one of the top eleven
  matrices contains `GGAA` or its reverse complement `TTCC`:
  ELF4 `AACCCGGAAGTG`, FEV `ACCGGAAGT`, EHF `CACTTCCTG`, ZBTB2 `ACCGGAAGTG`,
  ELF1 `CAGGAAGTG`, ELF3 `CACTTCCTG`, ZBTB11 `CACTTCCGG`, ETV1 `ACAGGAAGT`,
  ERG `ACAGGAAGTG`, GABPA `CACTTCCTGT`, FOXO1::ELK1 `ATCAACAGGAAGT`.
  Top-15 mean pairwise Jaccard of covered sets is 0.278, max 0.792.

  **ZBTB2 and ZBTB11 are the trap.** They are zinc-finger factors and read as
  independent corroboration from a different structural class. They are in the
  list because their JASPAR consensus carries the ETS core, not because
  zinc-fingers replicate the result. Writing "ETS factors plus ZBTB2/ZBTB11"
  invites a reviewer to grep the consensus column and find it in one minute.
  Write: "a single 4-bp ETS core (GGAA/TTCC), recovered through N redundant
  JASPAR matrices." The effective number of independent findings is one.
- **Recovery of known biology by an unsupervised route, not new biology.** ETS and
  GABPA sites are an established hallmark of unmethylated CpG-island promoters. The
  claim is that a model never shown a motif recovered this de novo, and that it
  holds on held-out probes in a different tissue, ancestry and platform generation.
- **The canonical panel is enriched, not missed.** The 18 known
  methylation-sensitive factors have median rank 136 of 831 — chance is 416. Say
  "recovered above chance but decisively outranked by ETS."

**The two null results belong in the same figure.** Disruption *magnitude* predicts
nothing (continuous ρ = −0.0109 [−0.0228, +0.0017]; strong vs weak median |ΔM̂|
0.0177 [0.0166, 0.0189] vs 0.0184 [0.0171, 0.0198] — overlapping, and if anything
*inverted*), and meQTL discrimination is not concentrated in strong disruptors
(AUROC 0.5918 [0.570, 0.614] vs 0.5962 [0.574, 0.620], n = 11,525 vs 9,881).
The model did not learn "breaking motifs matters" as a general rule; it learned
something signed and family-specific. Reporting only the positive would look like
fishing.

**The open question these two results jointly raise, and the cheap test for it
(2026-08-27).** Signed family-specific coupling *with* a null magnitude coupling
*and* no motif-concentrated discrimination has a simpler explanation than
"the model learned binding-site disruption": the model may have learned that
ETS-like sequence *composition* marks unmethylated regions, so perturbing toward
or away from that composition moves the prediction in the right direction without
anything resembling a binding-site mechanism. Both readings predict the signed
coupling; only the grammar reading predicts a magnitude relationship, and we do
not observe one.

The consensus-sequence reading above makes the compositional explanation more
likely, not less. A model with genuine binding-site grammar should be graded --
the worse you break the site, the larger the predicted shift. That is exactly
what Q1 tests, and Q1 is null. What survives is sensitivity to the *presence* of
a 4-bp word, which is what a k-mer model does by construction. Note also that
`motif_consensus_cpg_count` is 0 for most of the top hits, so this is not a
CpG-content artifact -- the association is to GGAA/TTCC itself.

This is directly testable and costs nothing new: **run scripts/22 on the
`kmer_ridge` baseline's pair scores** (scripts/23 emits the same schema, so point
`--scores-dir` at `results/journal/sequence_baselines/variant_scoring` and pass
`--seeds=-1` with the `=` so argparse does not read `-1` as a flag). A linear
6-mer model has no notion of a binding site whatsoever. If it reproduces the ETS
direction, the effect is compositional and must be described that way. If it does
not, the grammar reading survives a real attempt to kill it, and the claim gets
much stronger.

**ANSWERED, 2026-08-28. It is composition.** `scripts/22` run on the `kmer_ridge`
baseline (`results/journal/motif_disruption_kmer_baseline/`) reproduces the ETS
coupling **more strongly than the neural models**:

| factor | fusion rho | kmer_ridge rho |
|---|---|---|
| ELF4 | −0.266 | **−0.467** |
| FEV | −0.257 | **−0.466** |
| ERG | −0.237 | **−0.400** |
| GABPA | −0.233 | **−0.377** |
| ETV1 | −0.237 | **−0.376** |

A linear ridge over RC-collapsed k-mer counts has no attention, no pretraining
and no concept of a binding site. Standardised against each model's own null the
two are equivalent (~6 IQR-halves out in both). The ETS coupling is a response to
GGAA/TTCC *sequence content*, not learned regulatory grammar, and the manuscript
must say so.

**This generalises beyond us, and is worth one paragraph as a methodological
caution.** A PWM-disruption-coupling analysis cannot demonstrate that a sequence
model learned regulatory grammar, because a bag-of-k-mers model reproduces the
result. Any paper claiming grammar from this style of analysis needs a
composition-only control. We have one; almost nobody runs it.

**But the same run produced a dissociation that favours the neural model, and it
is the more interesting half.** Q3 meQTL discrimination:

| model | strong disruption | weak disruption |
|---|---|---|
| fusion | 0.5918 [0.570, 0.614] | 0.5962 [0.574, 0.620] |
| kmer_ridge | **0.5162** [0.496, 0.535] | **0.5090** [0.486, 0.533] |

The k-mer model is *better* at ETS coupling and *much worse* at telling real
meQTLs from null ones — 0.51 versus 0.59, near chance. The two capabilities come
apart, which means the motif coupling is not what drives meQTL discrimination and
the neural models hold something the k-mer model does not.

*Leading hypothesis, to be tested not assumed:* **bag-of-k-mers is
position-blind.** Changing one base changes the same k-mer counts wherever it
sits in the 1,000-bp window, so `kmer_ridge` cannot express "variants nearer the
CpG matter more" — the single largest real effect in the data (distance alone
gives AUROC 0.595 on GENOA). If that is the explanation, the neural advantage is
positional encoding rather than sequence grammar. **scripts/20's
distance-matched AUROC on the baseline scores tests this directly** and is the
first thing to run when `SM_baselines` lands. Do not claim the neural advantage
until that separates positional information from everything else.

Also: known methylation-sensitive factors rank 136/831 for fusion versus 232/831
for `kmer_ridge` (chance 416). Modest, but it points the same way.

The earlier framing rules stand and are now better supported: one 4-bp core, one
finding, recovery of known biology rather than new biology.

Do this before the manuscript describes the ETS result as motif-disruption
learning. As written, §7's framing rules are already correct and conservative —
the risk is not the record, it is restating it more loudly in the paper than the
evidence supports.

*Method note.* A binary inside/outside-a-motif split does not work: with the full
library scanned, **100%** of variants fall inside some occurrence at the
conventional 0.80 cutoff, leaving no background. Q1 and Q3 use a top-versus-bottom
quartile contrast on disruption magnitude, matched on exact distance and GC
quintile.

---

## A2. Retraining plan — improved training data

### Audit first: `data/audit_training_data.py` (new, 2026-08-27, CPU only)

Run this BEFORE deciding anything about retraining. It reads only the existing
split CSVs and the HM450 manifest, and answers three questions in descending
order of how much damage a bad answer does.

**1. Cross-split sequence leakage — CHECKED 2026-08-28, and the splits are clean.**
MinHash over canonical 31-mers, sketch 128:

| split | n | median | p99 | Jaccard > 0.5 | > 0.8 |
|---|---|---|---|---|---|
| val | 46,557 | 0.000 | 0.094 | 30 (0.06%) | 11 |
| **test** | **26,570** | **0.000** | **0.094** | **4 (0.02%)** | **2** |

**Four of 26,570 held-out probes share more than half their 31-mers with a
training probe.** Ninety-nine percent sit below 0.094. Held-out performance is
not memorisation, and the chromosome-blocked design does what it was meant to.
Worth one Methods sentence: the check is rarely run, and a reviewer who wonders
about paralogues or segmental duplications gets a number instead of silence.

*Original rationale, retained:* Chromosome-blocked splits stop *positional* leakage. They do
nothing about *sequence-similarity* leakage: segmental duplications, paralogues
and recent repeat families put near-identical 1,000-bp windows on different
chromosomes, and CpG-island promoters are exactly where duplications cluster. If
a meaningful share of chr8–9 test probes share most of their 31-mers with a
training probe, held-out performance is partly memorisation. Measured by MinHash
over canonical 31-mers; a median near zero is the healthy result and the tail is
what matters. `--exact` recomputes true Jaccard for flagged pairs.

*If the tail is large*, the fix is not to re-split — it is to report a
similarity-filtered test subset alongside the full one. A reviewer who asks this
question and gets a prepared answer is reassured; one who asks and gets silence
is not.

**2. Probe QC — RESOLVED 2026-08-28. My earlier claim here was wrong.**

I wrote that `data/build_training_data.py` applies none of the HM450 masks,
inferring it from a grep that found no `MASK` string in that file. The audit
settles it empirically, and the inference was wrong:

| split | n | MASK_general | MASK_snp5_common | MASK_rmsk15 | mapping / GMAF1p / nextBase |
|---|---|---|---|---|---|
| train | 345,359 | **0.0%** | 10.96% | 14.42% | 0.0% |
| val | 46,557 | **0.0%** | 11.66% | 14.13% | 0.0% |
| test | 26,570 | **0.0%** | 11.69% | 15.71% | 0.0% |

`MASK_general` is **zero in all three splits**, so masked probes never enter the
training data; the exclusion happens upstream of `build_training_data.py`. That
matches `scripts/19`, which dropped 0 of 76,893 eGTEx pairs at its own probe-QC
step. Training and scoring QC are consistent and there is nothing to fix.
`MASK_general` subsumes mapping, GMAF1p and next-base-switch, which is why those
are zero as well.

**What is real: ~11% of probes in every split carry a common SNP within 5 bp
(`MASK_snp5_common`) and ~15% overlap repeats (`MASK_rmsk15`).** Neither is
covered by `MASK_general`, and both are present in train and test alike. The
balance across splits (10.96 / 11.66 / 11.69) means they do not bias the
train/test comparison — but `MASK_snp5_common` bears directly on the variant
work, because a common SNP under the probe corrupts the measured beta in exactly
the donors whose genotype the meQTL analysis is about.

*Action, cheap and worth doing:* re-run the variant evaluation excluding
`MASK_snp5_common` probes as a sensitivity analysis. If signed rho holds, the
result is strengthened against an obvious reviewer question. No retraining —
it is a filter at evaluation time.

**3. Split comparability — chr8–9 are modestly more methylated, and the model
handles it.**

| split | n | median beta | SD(M) | beta < 0.3 | beta > 0.7 |
|---|---|---|---|---|---|
| train | 345,359 | 0.550 | 3.55 | 39.9% | 42.3% |
| val | 46,557 | 0.596 | 3.54 | 38.5% | 44.1% |
| test | 26,570 | **0.642** | 3.46 | 36.1% | **46.5%** |

Held-out chromosomes carry a median beta 0.09 above the training chromosomes and
4 points more hypermethylated probes. Spread is comparable (SD of M 3.46 vs
3.55), so this is a shift in location, not scale.

**Report this as a point in the model's favour.** A model that had learned the
training mean would be biased low on a more-methylated test set. Observed mean
signed error is **-0.002 to -0.004** across seeds — essentially zero. The model
tracks the shift rather than regressing toward the training distribution.

It is also the strongest argument for requirement 2: if chr8–9 differ this much
in composition, performance may depend on which chromosomes are held out, and
repeated splits are how that gets measured rather than assumed.

*Original rationale, retained:* Chromosome-blocked splits are not random samples. If
chr8–9 differ from the training chromosomes in methylation distribution, part of
the train/test gap is composition rather than generalisation. Reported so it can
be stated in the paper rather than discovered in review.

*Tested* against synthetic splits with ten verbatim-copied windows planted in
`test`: all ten flagged at estimated Jaccard 1.000, matched to the correct
training probe, confirmed at exact Jaccard 1.000, with zero false positives among
the 290 clean probes.

### Do not retrain until these three are answered

Retraining before the audit means rebuilding on the same unexamined foundation.
Each answer changes what "improved training data" should mean, and all three come
from one CPU job on data already on disk.


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

**BLOCKED as of 2026-08-27 — do not retry the automated path.** GSE69914's GEO
metadata carries **no group labels**. All 407 samples are titled `BCFD1` through
`BCFD407`, source name is "genomic DNA from breast sample BCFD<n>", and the only
characteristics field is "molecule subtype: bi-sulphite converted genomic DNA".
Confirmed from the series page itself, not inferred. The 50/84/263/7/4 composition
appears only in the series *summary prose*; nothing maps an identifier to a group.

`data/acquire_healthy_breast_cohort.py` is written and its composition guard
correctly refused to write anything. To finish it, someone must obtain the
per-sample annotation from the paper's supplementary material (Teschendorff et al.
Nat Commun 2016) and drop it in as a two-column CSV — identifier, group. That is a
one-time manual step, not a scripting problem. PMC blocks automated fetching, so
it needs a human with a browser.

*Cost so far:* a 1.6 GB download, since the first version of `--inspect` fetched
before reading the header. That is fixed — `--inspect` now streams the header and
stops at the table marker. Delete the archive; the URL and expected SHA-256 are in
the script.

*Recommendation:* park this. It addresses a stated limitation, not an open mentor
requirement, and requirement 3 has nothing done. Resume when the annotation table
is in hand.

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
| eGTEx Breast Mammary mQTL (all pairs) | **hg38** | no | automatic (GCS) | **in progress** — 45 GB raw, delete after prefilter |
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

---

## E. Closing entries — 29 Aug 2026

### E.1 Probe-QC sensitivity: `MASK_snp5_common`

`scripts/27_mask_snp_sensitivity.py`. No model was re-run; the `predictions.csv`
written at test time was re-scored, and the published `beta_mae` and `auc` were
reproduced from it for all 9 model-seed combinations to within 5e-4 as a check
that metric definitions had not drifted (18 values checked, 18 passed).

The HM450 manifest flags 15.40% of probes (74,764 / 485,577) as
`MASK_snp5_common` — a common SNP within 5 bp of the interrogated CpG, where the
measured beta may be partly a genotype artefact. In the held-out set this is
3,105 / 26,570 probes (11.7%).

| model | stratum | n | beta MAE | ROC-AUC |
|---|---|---|---|---|
| fusion | all | 26,570 | 0.0993 ± 0.0020 | 0.9680 ± 0.0017 |
| fusion | retained | 23,465 | 0.0993 ± 0.0021 | 0.9689 ± 0.0017 |
| fusion | excluded | 3,105 | 0.0987 ± 0.0015 | 0.9530 ± 0.0023 |
| sequence | all | 26,570 | 0.1099 ± 0.0006 | 0.9569 ± 0.0006 |
| sequence | retained | 23,465 | 0.1102 ± 0.0007 | 0.9579 ± 0.0006 |
| sequence | excluded | 3,105 | 0.1079 ± 0.0008 | 0.9362 ± 0.0010 |
| epi | all | 26,570 | 0.1395 ± 0.0001 | 0.9187 ± 0.0001 |
| epi | retained | 23,465 | 0.1409 ± 0.0001 | 0.9190 ± 0.0001 |
| epi | excluded | 3,105 | 0.1290 ± 0.0003 | 0.9096 ± 0.0001 |

**Result: the headline metrics are not propped up by genotype-affected probes.**
Excluding them leaves fusion beta MAE unchanged at 0.0993 and *raises* ROC-AUC
0.9680 → 0.9689. The flagged probes are the harder stratum, not the easier one,
which is the direction that rules out the reviewer's concern: had the model been
reading genotype artefact, those probes would have scored *better*, not worse.

**All three architectures lose discrimination on the same probes** (fusion
−0.0159, sequence −0.0217, epi −0.0094 in AUC relative to retained). A deficit
shared by a sequence-only model, a context-only model, and their fusion is a
property of the probes, not of any one model.

**Range-compression hypothesis — TESTED 30 Aug 2026, NOT CONFIRMED, and the
test is underpowered.** `scripts/29_range_compression_check.py` resamples
unflagged probes to match the flagged stratum's beta histogram and recomputes
both metrics, so the comparison is against probes that were never flagged but sit
at the same targets. Output in `results/journal/range_compression/`.

| model | ΔAUC retained→excluded | reproduced by beta-matching | ΔbetaMAE | reproduced |
|---|---|---|---|---|
| fusion | −0.0159 | 69% | −0.0006 | wrong sign |
| sequence | −0.0217 | 60% | −0.0023 | wrong sign |
| epi | −0.0094 | 211% (overshoots) | −0.0119 | 36% |

Verdict NOT EXPLAINED, max |AUC gap| 0.0104 against a 0.010 tolerance. But the
verdict is not the point: **the test cannot resolve an effect this small.**
Subsample sd is 0.003 (fusion, sequence) to 0.006 (epi), and the excluded
stratum's own AUC standard error at n = 3,105 is roughly 0.006–0.008, so combined
precision is about ±0.007 against gaps of 0.005–0.010. Every gap is inside one
standard error, and the three models do not behave alike, so the heterogeneity is
not evidence of a mechanism either.

Also note the premise barely holds: the two strata are far more similar in shape
than assumed (intermediate fraction 17.2% retained vs 18.8% excluded, sd 0.3725
vs 0.3245). The MAE difference being explained is 0.0006 for fusion — it was
over-read as a finding in the first place.

**Status: same category as E.2 — run, uninformative for a structural reason, not
written into the manuscript.** The tumour-domain question (§E.4 item) stays open
and neither limb may be asserted. The plain sensitivity finding above is
unaffected and is what the manuscript reports.

### E.2 ClinVar matched-background test — RUN, and uninformative for a structural reason

Superseded the earlier "abandoned, never run" entry: the test was rebuilt,
pre-flighted, executed on GPU (job 44784867, 15,031 variants x 4 passes x 3
seeds, 48 min) and analysed. It is reported here in full because the reason it
fails is more useful than the result.

**Design as executed.** The 35 held-out non-truncating ClinVar variants were
scored inside a pool of 14,996 synthetic comparators — single-base substitutions
drawn at random inside held-out probe windows, positions and alleles uniform,
never the protected centred CpG (`data/build_synthetic_background.py`). A
pre-flight (`data/preflight_clinvar_background.py`) confirmed before any GPU
time that 32 of 35 matched at T2 (same SBS96, same CpG effect, distance within
50 bp) and that no variant drew a material share of its background from the
ClinVar set.

**Pre-registered primary statistic: null.** 0 of 35 variants had a
matched-background tail probability below 0.05, against 1.75 expected;
binomial p = 0.42.

**The secondary statistic looked significant in the WRONG direction, and was an
artefact.** Variant-level mean percentile 37.1, z = -3.30 (secondary cohort
35.6, z = -2.67). Two confounds account for all of it:

1. **Pseudoreplication.** The 35 variants span only **6 distinct probes**, with
   **25 of them on `cg13601799` alone**. Treating them as independent counted one
   locus 25 times. Clustered by probe: mean 46.1 [23.7, 68.5], t = -0.448,
   **p = 0.67** — null.
2. **Context mismatch.** The ClinVar loci sit in markedly less active chromatin
   than the background pool: ATAC 0.004 vs 0.403, H3K4me3 4.36 vs 16.59,
   H3K27ac 0.47 vs 3.34, phyloP -1.83 vs -0.50. Matching held substitution
   class, CpG effect and distance fixed, but not genomic context.

**The power statement in the pre-registration was wrong.** It assumed n = 35.
The independent unit is the probe, so the real n is 6. At n = 6 a true mean
percentile of 70 is undetectable; roughly 80 would be needed. The declared
threshold ("65 detectable, 58 not") never applied.

**Conclusion, and why it closes the question.** The held-out ClinVar cohort
cannot support a matched-background test — not with a larger background, not
with stricter matching. Six independent loci is a structural ceiling of the
cohort, not a shortfall of this attempt. Any future version needs more held-out
probes carrying ClinVar-pathogenic non-truncating variants, which is a data
problem rather than an analysis one.

**What is reportable.** The primary statistic as pre-registered (null), the
effective sample size (6 probes, not 35 variants), and the structural reason the
test is uninformative. The apparent depletion must NOT be reported as a finding;
it is fully explained by pseudoreplication and context mismatch, and it
disappears under the correct clustering.

**Provenance.** `results/journal/clinvar_matched_background/` holds
`preregistration.json`, the cohort, the synthetic pool and its summary, the
scoring input, `clinvar_cohort_ids.json`, `scored/`, and
`clinvar_matched_background_result.json`. Note that the verdict string inside
that JSON reads INCONCLUSIVE via the enrichment-direction rule only; the
clustered analysis above is the correct reading and supersedes it.

### E.2b Breast-cancer GWAS risk-variant matched-background test — RUN, clean null

Pre-registered in `results/journal/gwas_matched_background/preregistration.json`
before any variant was scored, with two amendments logged at the time (retain
multi-allelic variants; hit bar 7->6 once the achieved n was known). Executed on
GPU, 15,028 variants x 4 passes x 3 seeds.

**Cohort.** GWAS Catalog breast-cancer associations, chr8/9 only (the held-out
chromosomes), within 500 bp of a held-out HM450 probe. 556 unique rsIDs -> 429
with a single-base risk allele -> 41 variant-probe pairs -> 35 with a reference
allele resolved from Ensembl GRCh38 -> **32 with ALT != REF**, on **32 distinct
probes, one variant each**.

**Coordinate validation.** The offset convention was determined empirically:
reference base agreed for 32/32 (100%) at offset -1, versus 21.9% at 0 and 37.5%
at +1. Had +0 been assumed, every variant would have been scored one base off
while looking perfectly valid.

**The design was sound this time**, which is what makes the null meaningful:

| check | result |
|---|---|
| independence | 32 variants / 32 probes — no pseudoreplication |
| match quality | 25/32 at T2 (same SBS96, same CpG effect, +/-50 bp) |
| comparators | min 21, median 30, max 152 |
| context confound | largest |d| = 0.36 across nine features; none material |

**Result: NULL.**

- primary: 3 of 32 with tail probability < 0.05, against 1.6 expected,
  binomial p = 0.21 (pre-registered bar: 6+ at p < 0.01)
- secondary: mean matched-background percentile **54.9 [44.8, 65.2]**,
  clustered by probe, t = +0.93, p = 0.36
- variant-level mean identical at 54.9, confirming the independence

**Interpretation, per the declared power statement.** Directionally positive but
far from significant. The interval spans a 5-point depletion to a 15-point
enrichment, so a modest true effect is not excluded — power at a true mean of 60
was 0.50. Report as null with that limitation attached, not as evidence of
absence.

**Why the null is unsurprising.** GWAS risk variants act through expression,
splicing and protein-level mechanisms as well as methylation. The 32 variants
lying within 500 bp of a held-out HM450 probe are a narrow slice of that biology,
and the model only sees local sequence.

**Consequence for the journal question.** Genome Medicine's clinical-relevance
criterion remains unmet by testing. Two pre-registered attempts, two nulls: E.2
uninformative for a structural reason, E.2b clean and properly controlled.
Remaining options are argued rather than demonstrated — a ranked resource table
for clinically actionable genes, and a VUS-interpretation framing — or a
different journal. That is a decision for the mentor conversation, not an
analysis problem.

**Provenance.** `results/journal/gwas_matched_background/` holds the
pre-registration with both amendments, `gwas_cohort.csv`,
`gwas_cohort_summary.json`, `ensembl_cache.json`, the scoring input, `scored/`
and `gwas_matched_background_result.json`. Built by `data/build_gwas_cohort.py`,
analysed by `data/gwas_matched_background.py`.

### E.6 Published-architecture head-to-head — DONE. Decisive on absolute prediction, null on variant effects

`scripts/28_published_architecture_baselines.py`. CpGenie and DeepCpG
reimplemented from released source (not from the papers, which omit the layer
tables), trained on our splits, three seeds each.

**Why reimplement.** Both are 2017 Keras/Theano/TF1 and do not install on current
hardware; more importantly the published CpGenie weights are GM12878
lymphoblastoid, not breast, so scoring them would have been a cross-tissue
strawman that favours us for the wrong reason.

**Only DeepCpG's DNA module is used, and that is a design choice, not a
handicap — challenged and verified 30 Aug 2026.** Published DeepCpG has three
modules: DNA (`CnnL2h128`, sequence only), CpG (`RnnL1`, a bidirectional GRU over
observed methylation at neighbouring CpGs across cells), and Joint. Our
implementation is `DeepCpGDnaCNN`, whose `forward(x)` takes one one-hot sequence
tensor; no second input path exists. Two reasons, both now in the manuscript
methods: the neighbour input does not exist in a bulk cohort-median design, and
it is *measured*, hence identical for REF and ALT, so it contributes exactly zero
to a predicted variant effect — the same allele-invariance that governs our own
context tower. Running full DeepCpG would raise its absolute-prediction numbers
while adding nothing to the variant comparison. CpGenie is sequence-only (1001 bp
flank, no neighbour input) and was built specifically for variant impact on
methylation and meQTL detection, making it the more apt comparator for the
variant arm.

**Faithfulness.** Layer specs read from
`CpGenie/cnn/seq_128x3_5_5_2f_simple.template` and
`deepcpg/models/dna.py::CnnL2h128`. DeepCpG's own docstring claims 4,100,000
parameters; our implementation computes 4,102,402 at the 1,000 bp input, and the
script refuses to train if the dense layer does not match 3,997,824.
Reverse-complement and crop helpers are duplicated verbatim from
`training_common` and asserted equal to it by `--verify`.

**Tuning was theirs, not ours.** CpGenie over the hyperas grid its template
declares (dropout {0.3,0.5,0.7} x lr {0.01,0.001,0.0001}); DeepCpG over dropout
{0.0,0.3,0.5}. Selected on validation beta MAE; the test split was never used for
selection. `lr=0.01` collapses CpGenie to 0.345 (constant prediction), 0.001 wins
in the interior. **Both selected the lowest dropout offered** (0.3 and 0.0) — a
boundary selection, reported in the manuscript, meaning a lower dropout than
either grid contains might serve these baselines slightly better.

| model | M MAE | beta MAE | ROC-AUC | params |
|---|---|---|---|---|
| Composition (3) | 1.9600 | 0.1954 | 0.8748 | — |
| k-mer ridge (2,772) | 1.6866 | 0.1565 | 0.9190 | — |
| Context only (9) | 1.4574 | 0.1395 | 0.9187 | — |
| **CpGenie** (reimpl.) | 1.3858 ± 0.0084 | 0.1281 ± 0.0011 | 0.9406 ± 0.0005 | 2.0 M |
| **DeepCpG** (DNA module) | 1.3412 ± 0.0092 | 0.1253 ± 0.0004 | 0.9437 ± 0.0009 | 4.1 M |
| Sequence (DNABERT-2) | 1.1941 | 0.1099 | 0.9569 | 117 M |
| **Fusion** | **1.0971** | **0.0993** | **0.9680** | 117 M |

**Monotone ordering, no ties.** Published CNNs beat the classical baselines
(CpGenie 18.1% below k-mer ridge) and are beaten by the pretrained encoder:
sequence-only alone is 12.3% below DeepCpG, fusion is **20.8% below** on beta
MAE, 18.2% on M MAE, +0.0243 AUC.

**Stated against ourselves in the manuscript:** the DNABERT-2 encoder is 117 M
parameters against 2.0 M and 4.1 M and arrives pretrained, so the margin reflects
capacity and pretraining together, not architecture alone; and both published
trunks were given our dual head, which is fairer for comparison but is not the
configuration their authors evaluated.

### E.6b Variant-effect arm of the head-to-head — 11 of 12 comparisons null

Both baselines were then scored on the identical GENOA (66,495) and eGTEx
(76,893) variant–CpG pairs through `scripts/20`, and compared to fusion by
`scripts/30_paired_model_comparison.py`.

**Why a new script was needed.** `scripts/20` draws each model its own set of
matched negatives, so part of any between-model gap is matching noise; and its
intervals are marginal, which is the wrong instrument for a difference between
estimates computed on the same variants. Script 30 builds ONE shared matched
cohort (matching depends only on the significance label and the distance, both
model-independent) and bootstraps the difference over the same blocks for both
models. It imports every metric and the matching routine from `scripts/20` by
path so the definitions cannot drift.

**This reversed a reading we had provisionally taken.** Off the unpaired
per-model numbers the ordering looked monotone (fusion 0.570 > cpgenie 0.5514 >
deepcpg 0.5444 on distance-matched AUROC). On the shared cohort fusion is 0.5604
and cpgenie 0.5567 — the gap was mostly the negative draw. Recorded because the
uncorrected version was nearly written into the manuscript.

| cohort | comparison | distance-matched AUROC | signed rho | direction |
|---|---|---|---|---|
| GENOA (n=4,037 sig) | fusion − deepcpg | **+0.0178 [+0.0035, +0.0321]** | +0.0145 [−0.0281, +0.0545] | +0.0054 [−0.0158, +0.0269] |
| GENOA | fusion − cpgenie | +0.0038 [−0.0135, +0.0199] | +0.0242 [−0.0218, +0.0683] | +0.0054 [−0.0145, +0.0242] |
| eGTEx (n=418 sig) | fusion − deepcpg | −0.0075 [−0.0483, +0.0390] | +0.0301 [−0.0728, +0.1504] | −0.0072 [−0.0566, +0.0614] |
| eGTEx | fusion − cpgenie | +0.0098 [−0.0476, +0.0629] | +0.0553 [−0.0217, +0.1556] | +0.0096 [−0.0289, +0.0584] |

One interval of twelve excludes zero. **Written up as convergence across
architectures, not as a loss**: three architectures spanning a decade recover the
same external meQTL signal at the same strength while a k-mer ridge on the same
sequence is at chance after matching (0.503), which is stronger evidence that the
external validation reflects the cohorts rather than our model than any margin
would have been. It is also what the gating analysis predicts — the context tower
is allele-invariant, so the capacity separating fusion on absolute prediction is
capacity the variant pathway does not use.

**One asymmetry recorded:** CpGenie is behind DeepCpG on absolute prediction and
not behind it on variant effects. Probe-level accuracy does not establish
variant-effect fidelity.

### E.7 Mentor requirement scorecard

| # | requirement | state |
|---|---|---|
| 1 | multi-cohort testing | **met** — GENOA (66,495 pairs), eGTEx breast, TCGA-BRCA tumour domain |
| 2 | repeated chromosome-blocked splits | **deferred by decision** — single holdout; stated in both branches of Limitations |
| 3 | stronger baselines and ablations | **met, strongly** — E.6: seven comparators including two published architectures, plus the fusion/sequence equivalence ablation. E.6b extends the head-to-head to variant effects with paired intervals |
| 4 | uncertainty calibration | **met** — RC disagreement, plus the generalisable finding that calibration must be assessed in logit space |
| 5 | ancestry analyses | **partial** — African American GENOA cohort with I² = 0%, but confounded with tissue; training-cohort stratification underpowered (84/4/1) |
| 6 | independent variant evaluation | **met, strengthened** — two cohorts, meta-analysis, null control, distance and significance gradients, distance-matched AUROC, and (E.6b) replication of the same signal by two independent published architectures |
| 7 | regulatory enrichment | **met but reinterpreted** — ETS coupling real, and reported as compositional after the k-mer control reproduced it more strongly |

Five met (three of them strongly), one partial, one deferred by decision. The two
that are not fully met are stated as such in the manuscript rather than papered
over.

### E.3 Manuscript correction pass — `main_revised.tex`

Nine edits, three of them corrections against our own earlier text:

1. **Abstract motif claim rewritten.** It asserted "a specific regulatory
   relationship the model was never shown" and survival of GC control. The
   results section had already been revised to report that a k-mer ridge
   baseline reproduces the coupling *more strongly*, i.e. that it is
   compositional. The abstract was contradicting the paper's own result.
2. **False claim behind the draft switch removed.** The `\else` branch of the
   limitations paragraph read "Evaluation uses a chromosome-holdout design with
   repeated blocked splits." Repeated splits were deferred and never run, so
   setting `\draftmodefalse` for submission would have printed a false methods
   claim that nobody would re-read. Now states the single-holdout design and
   what the intervals do and do not cover.
3. **Split status added to the case studies**, in the abstract and in the
   ranked-variants section.
4. **Two-cohort meta-analysis promoted into the abstract** (ρ = 0.178,
   0.068–0.283, p = 1.6e-3, I² = 0%, plus the GENOA null control).
5. **Probe-QC sensitivity paragraph added** to the performance section (E.1).
6. ClinVar figure removed from the outstanding list; E.2 recorded in the
   draft-status section.

**Verification.** The container lacks `lmodern`, so the PDF must be rebuilt on
the Mac. What was verified here: with `lmodern` stubbed out, the edited file and
the pristine file produce byte-identical LaTeX error profiles (7 × "Undefined x
coordinate", from figure code, in both). The edits introduce no new LaTeX errors.
Every one of the nine substitutions asserted exactly one match before applying.

### E.4 Bucket list — updated 30 Aug 2026

**Closed 30 Aug 2026**

| item | outcome |
|---|---|
| Rebuild `main_revised.pdf` | done — built in the cloud container (full TeX Live), 16 pages, zero errors, zero undefined references. Supersedes both the stale `main_revised.pdf` and `main_revised_preview.pdf` |
| Verify NCOA2 held-out status | **confirmed** — NCOA2 is at 8q13.3 and chr8 is a test chromosome, so held-out status is definitional under chromosome-blocked splitting |
| Beta-histogram check, MASK excluded stratum | **run** — E.1. Not confirmed, and underpowered relative to the effect. Not for the manuscript |
| Tumour-domain open question | **stays open** — the compression limb was tested and did not resolve; draft-status entry updated to say so |
| Baseline variant-effect evaluation | done — E.6b, both cohorts, both baselines, three seeds |
| Paired between-model intervals | done — `scripts/30_paired_model_comparison.py`; reversed a provisional reading before it reached the manuscript |
| `DILUTION GRADIENT (fusion, ...)` label bug in `scripts/20` | fixed — the header was hardcoded while the rows came from `args.models[0]`; now prints the actual model. Data in `significance_gradient.csv` was always correct |

**Open, in the order I would take them**

| # | item | note |
|---|---|---|
| 1 | **Repeated chromosome-blocked splits** | the only remaining model training, and the only unmet mentor requirement that is not a data limitation. Four additional folds at seed 42 |
| 2 | Split-status label on the case-study figure | text states it, the figure does not |
| 3 | `supplementary_package` + `reproducibility` rebuilds | stale; must run on the cluster where the source CSVs live. Now also need the E.6b outputs |
| 4 | Finish local repo cleanup, commit | `data/external/bend`, `data/__pycache__`, `scripts/__pycache__` |
| 5 | Read both `\draftmode` branches with the switch flipped | how the false "repeated blocked splits" claim survived undetected. Applies to every `\else` branch |
| 6 | Figure files are not in the local tree | `results/journal/manuscript_figures/` holds only `run_summary.json` locally, so every local build renders `\missingfigure` placeholders. Sync the PNGs from the cluster before producing a circulating PDF |

**Deferred by decision, not oversight:** nothing. Requirement 2 (repeated
chromosome-blocked splits) has moved from deferred to open item 1.

**Lost:** `data/prepare_clinvar_matched_background.py` was deleted rather than
archived during cleanup and is absent from both trees. E.2 is now the only
surviving record of that design. Recoverable on request.

### E.8 Length pass for Nature Communications — 30 Aug 2026

The draft was over on all three limits. State after the pass:

| | before | after | NC Article |
|---|---|---|---|
| abstract | 678 words | **183** | ≤200, no refs |
| main text (Intro+Results+Discussion+Availability) | 5,812 | **5,330** | ideally ≤5,000 |
| display items | 15 (8 fig, 7 tab) | **10** (7 fig, 3 tab) | ≤10 |
| Methods | 2,367 | 2,367 | excluded from the count |

**Abstract** rewritten from six numbered results to one paragraph. Dropped from
it: the ETS motif result, the NCOA2/STK11 case studies, and the two
matched-background tests as separate items (compressed to the closing sentence).
All survive in Results. **Open call for the author:** whether the motif result
deserves an abstract mention; adding it costs ~25 words from the cohort sentence.

**Moved to `\section*{Supplementary Figures and Tables}`** (renumbered S1–S5,
in-text `\ref`s resolve automatically): `tab:splits` (Methods dataset table),
`tab:gradient` (**duplicated `fig:gradient` outright** — same data on facing
columns), `tab:distance`, `tab:motifs`, `fig:top-candidate`. Nothing load-bearing
moved: Table 1, both variant tables, and the performance, discrimination,
gradient and uncertainty figures all stay in the main text.

**Cut:** the `What the evidence supports` subsection (~350 words of restatement;
its one irreplaceable sentence — what is *not* established — folded into
Limitations). `Conclusion` folded into the Discussion and trimmed. Three Results
paragraphs compressed, the largest being the convergence paragraph written the
same day.

**Second pass, same day — now inside every limit: abstract 183/200, main text
4,992/5,000, display items 10/10.** The first pass stopped at 6% over and called
that acceptable; it was not, and the second pass found that most of the excess was
redundancy rather than content:

- The motif Results subsection stated the same two null statistics twice and
  closed on "the model did not learn that disrupting motifs matters in general;
  it learned a signed, family-specific relationship" — the pre-`k`-mer-control
  framing, contradicting its own opening. Removed; three control paragraphs
  compressed to one. 573 to 264 words.
- The Discussion carried two paragraphs calling the motif result "the mechanistic
  layer the earlier draft lacked" and "unsupervised recovery from sequence alone",
  also pre-downgrade, and inconsistent with Results. Replaced by one paragraph
  stating the compositional reading and the general lesson.
- `Translational scope` restated the uncertainty section in full; compressed to a
  clause. `Data and Code Availability` listed table numbers that had moved to
  Supplementary.
- Draft scaffolding in prose ("the central addition of this revision", "the
  earlier 81-association analysis") removed — it would not have survived
  submission anyway.

**Nothing was cut for length alone.** Every deletion was a duplicate, a stale
claim, or a restatement. NCOA2/STK11 and the motif result are retained; the motif
section is shorter but says the same thing its own control supports.

**Process note — an error worth recording.** The first attempt at the display-item
move used a regex with `.*?` spanning from `\begin{table}` to the label, which
matched across intervening tables and silently deleted roughly 2,900 words of
Results. It was caught by the post-edit word count (Results 3,732 → 814), not by
the LaTeX build, which compiled cleanly. The manuscript was restored from the
committed copy and the extraction redone by enumerating minimal float blocks and
selecting by label, with assertions on block size and on one `\label` and one
`\caption` per block. **A clean compile is not evidence that a LaTeX edit was
correct.**

### E.5 PDF build state — current as of 30 Aug 2026

`main_revised.pdf` is a fresh full build of the current `main_revised.tex`:
16 pages, `latexmk -pdf` exit 0, zero errors, zero undefined references or
citations, cross-references settled over two passes. Built with full TeX Live
(Latin Modern under T1, no font substitution), so it is not the reduced-font
reading copy the previous entry described. `main_revised_preview.pdf` is now
obsolete and can be deleted.

One caveat: the figure PNGs live only on the cluster, so this build renders nine
`\missingfigure` placeholders. Every table, number and cross-reference is real.
Sync `results/journal/*/plots/` and `manuscript_figures/` before circulating.

`main.tex` remains untouched.
