# SilentMethyl — paper framework and work plan

State as of 5 Sep 2026, after the nine-tissue transfer analysis.
Companion documents: `REQUIREMENTS.md` (A2 = retraining plan),
`MELODY_COMPARISON.md` (positioning against Jin et al. 2026).

---

## 1. The claim

Everything below serves one sentence:

> **Variant effects on CpG methylation are encoded in local sequence and are
> largely tissue-invariant; epigenomic context sets the methylation baseline but
> not the response to a variant.**

This is worth stating plainly because it converts what looks like a negative
result into the paper's spine. The context tower is **allele-invariant by
construction** — the same context vector is fed for REF and ALT, so context
cannot create a variant effect, only modulate it through the gate. That design
fact predicts two things, and both are now observed:

- **R2**: fusion and sequence-only models should be indistinguishable *for
  variant effects* even though fusion wins on absolute methylation. Confirmed —
  every paired difference spans zero, across nine tissues.
- **R3**: a model trained on one tissue should therefore transfer to others
  without retraining. Confirmed — 7 of 9 tissues discriminate real mQTLs from
  distance-matched nulls, and breast is not better than the rest.

R4 is the falsification test (§4). If the claim is wrong, tissue-specific mQTLs
should be predicted as well as shared ones.

This through-line is what Melody does not have. They have a better methylation
predictor across more tissues; they make no mechanistic claim about *where
tissue-specificity lives*.

---

## 2. Results framework

### R1 — A gated fusion model predicts CpG methylation
*Question:* does combining DNABERT-2 sequence with epigenomic context beat
sequence alone and beat published architectures?
*Scripts:* `10/11/12_train_*`, `13_test_model`, `14_baselines_simple`,
`15_baselines_published`, `16_paired_model_bootstrap`
*Status:* **done.** Three arms × three seeds, vs CpGenie, DeepCpG, k-mer ridge,
composition.
*VERIFIED 5 Sep 2026* (the numbers live in
`paired_model_bootstrap/paired_model_difference_bootstrap.csv`, NOT in
`run_summary.json`, which holds metadata only). Cross-seed ensemble, 26,570 loci,
257 genomic blocks, fusion minus sequence-only:

    roc_auc    +0.010533 [+0.009090, +0.012069]   P(diff>=0) = 1.0
    beta_mae   -0.010429 [-0.011323, -0.009514]   P(diff>=0) = 0.0
    beta_rmse  -0.019041 [-0.020601, -0.017471]
    m_mae      -0.094613 [-0.102197, -0.086805]
    m_rmse     -0.150900 [-0.163059, -0.139263]

All three individual seeds agree in sign with intervals excluding zero. Against
the epigenomic-only arm the gap is an order of magnitude larger (ensemble
roc_auc +0.0507, m_mae -0.378), as expected — sequence carries the signal,
context modifies it.

*How to write it.* The gain is **consistent and significant but small**:
+0.011 AUROC and ~1.0 percentage point of beta-value MAE. Say that explicitly.
A reviewer will convert 0.011 into plain language whether or not we do, and
claiming "substantial" here is the kind of overreach that costs credibility on
the parts of the paper that are strong.

### R2 — The gain is in the baseline, not the variant response
*Question:* does the context tower contribute to *variant-effect* prediction?
*Scripts:* `20_variant_scoring`, `21_variant_evaluation`,
`22_context_stratification`
*Scripts:* also `23_context_permutation` (run 5 Sep 2026, 76,893 pairs,
identity / shuffle / median, prediction recorded before execution).
*Status:* **done, and now causal rather than merely architectural.** Corrupting
the context vector degrades predicted methylation LEVELS far more than predicted
VARIANT EFFECTS, on normalised error, Spearman and sign agreement, under both
schemes:

    scheme    quantity   MAE/SD   Spearman   sign agr.   Pearson
    shuffle   levels     0.3045     0.8610      0.8593    0.9015
    shuffle   deltas     0.1123     0.9530      0.9499    0.9276
    median    levels     0.1740     0.9772      0.9605    0.9855
    median    deltas     0.0692     0.9842      0.9744    0.9649

Normalised error is 2.7x (shuffle) and 2.5x (median) larger for levels than for
deltas. Construction counters were clean: 76,893 scoreable, 0 reference-base
mismatches, 0 CpG-altering, 0 window problems.

*State this against ourselves.* The pre-specified `deltas_preserved_more` flag
was implemented on **Pearson**, and on that metric the median scheme FAILS
(deltas 0.9649 < levels 0.9855) while shuffle passes. Report both. The reason
Pearson behaves differently is that methylation levels are bimodal with SD 3.13
M-units, so a high Pearson is cheap on levels and not comparable across the two
quantities — which is why normalised MAE, Spearman and sign agreement are the
fairer reads. Present the Pearson discrepancy in the text rather than choosing
the three metrics that agree with us; a reviewer who recomputes it will find it.

*Why this matters beyond R2.* It supplies the mechanism for R3. Variant effects
are near-invariant to what chromatin the model is shown, so a breast-context
model transferring to lung is not a surprise — it is the predicted consequence
of allele invariance. R2 and R3 stop being two results and become one argument.

*It also answers the mentor's multi-tissue demand without retraining.* Per
`why_this_substitutes_for_a_tissue_swap`: a random locus's chromatin is further
from the truth than another tissue's chromatin at the same locus, so a null here
implies a null for the tissue swap. Cite this rather than promising nine models.

*Hard constraint on the manuscript.* We may NOT claim that epigenomic context
improves variant-effect prediction anywhere in the paper. It improves level
prediction (R1) and is close to inert for deltas (R2). Any sentence implying
otherwise contradicts our own data.
*Reserved:* slot `53` for allele-specific methylation (moved from `23`, now used).

### R3 — Zero-shot transfer across nine tissues
*Question:* does a breast-trained model prioritise mQTLs in tissues it has never
seen, above a distance-matched null?
*Scripts:* `30_transfer_synthesis`, `31_transfer_discrimination`,
`32_transfer_summary`
*Status:* **done.** Distance-matched AUROC excludes 0.5 in 7/9 tissues; distance
alone pinned to exactly 0.5000; CpG-altering variants excluded (~37% of pairs,
enriched for true mQTLs, so the estimate is conservative).
*Headline:* breast (0.6141) is indistinguishable from lung (0.5880), colon
(0.6051), kidney (0.6062). **The model is not better on the tissue it was
trained on.**
*Must include:* the Melody Fig 3H tension (their tissue-matched tracks *do* win
— because they train per tissue and we do not); the prostate-vs-ovary result
killing the donor-sex explanation; Melody's independent ovary anomaly as
corroboration.

### R4 — Where transfer fails, and why
*Question:* can the model distinguish shared from tissue-specific mQTLs?
*Scripts:* `40_meqtl_tissue_specificity` (`--stage matched,chromatin`)
*Status:* **done, 5 Sep 2026. The answer is NO** — see the recorded outcome in
§4. Winner's curse, not mechanism.
*Why it matters:* this is the mentor's actual headline question, and it is the
only remaining item that produces a **biological finding** rather than a
methodological one — which is exactly what the Nature Communications bar
requires (§6).

### R5 — Mechanism and limits
*Question:* what is the model responding to, and when should it not be trusted?
*Scripts:* `50_motif_disruption`, `51_rc_uncertainty`, `52_gwas_enrichment`
*Status:* **done.** Motif disruption with a k-mer-matched null; reverse-
complement disagreement as a calibrated uncertainty signal; GWAS regulatory
enrichment on the two adequately powered tests only.
*Reserved:* slot `53` for allele-specific methylation.

### R6 — Application: variant prioritisation
*Question:* does this produce candidates a biologist would act on?
*Scripts:* `60`–`64`
*Status:* **done.** STK11 and NCOA2 retained per mentor instruction.

### Superseded
`70_mqtl_positive_control` (81 pairs) and `71_mqtl_matched_negative` (35 pairs)
are dwarfed by the nine-tissue cohort (13,744 significant pairs). Delete
together with Supplementary S2/S3 in one coordinated manuscript edit — not
before, since the text still cites them.

---

## 3. Work plan, in order

**Tier 1 — required before submission**

1. **R4 at nine tissues.** Inference-only, no GPU. Hours, not days.
2. **Repeated chromosome-blocked splits.** The only remaining training and the
   last unmet item from the original seven. Read `REQUIREMENTS.md` §A2 first —
   it records a silent-failure hazard in the positional reverse-complement swap
   that must be fixed before `TABULAR_FEATURES` is extended.
3. **Rewrite R3 in `main_revised.tex`** with the nine-tissue table, the Melody
   positioning, and the sex argument. (`main.tex` stays untouched.)
4. **Housekeeping:** regenerate `S6_active_code_sha256.txt` via
   `90_build_supplement_package.py` (merged scripts have new content); add the
   `config.json` concurrent-read note to `REQUIREMENTS.md`; force-add
   `literature_breast_variant_seeds.csv` if it is hand-curated.

**Tier 2 — high value, gated on public data**

5. **Melody head-to-head.** Zenodo record `21386471`. If checkpoints and the
   meQTL benchmark are public, running Melody on our distance-matched cohorts is
   inference-only and is the single strongest addition available — it converts
   the nearest competitor into our strongest baseline. Check this early; it may
   change how R3 is written.
6. **Context swap** (slot 23). Needs a non-breast ENCODE line with all nine
   context features. Would make R2 causal: feed lung context to a breast
   sequence and show variant effects do not move.
7. **ASM validation** (slot 53). Needs a public allele-specific methylation
   resource.
8. **eQTL / eQTM validation.** GTEx v8 eQTLs are public. eQTMs are harder.
   Lowest priority of the four — it validates relevance, not the claim.

**Not doing**

- Joint multi-tissue training. Melody published it at 39 tissues in the target
  journal; the context tower is allele-invariant so it cannot answer the
  tissue-specific-variant question anyway; and it contradicts the retraining
  scope fixed on 4 Sep.
- Naïvely pooled multi-tissue baseline — same reason.

---

## 4. R4 prediction, recorded before the run

Writing this down first so the result is a test rather than a story fitted
afterwards. `52_gwas_enrichment` already uses a `preregistration.json`; do the
same here.

If the §1 claim holds:

- **Shared mQTLs** (significant in many tissues) should be predicted *better*
  than tissue-specific ones, because shared effects are the sequence-driven
  ones.
- **Tissue-specific mQTLs** should be closer to the null — they are where
  context matters, and context cannot carry allele information in this
  architecture.
- The gap should **not** track cohort size, or it is a power artifact.
- Chromatin state should differentiate the two classes even where the model
  cannot.

If instead the model predicts tissue-specific mQTLs as well as shared ones, the
claim in §1 is wrong and R3 is measuring something more generic — say so.

### R4 OUTCOME, recorded 5 Sep 2026 — the answer is NO

72 ordered directions, 49 informative. Five tissues are too underpowered to
serve as discovery cohorts (BreastMammaryTissue, KidneyCortex, MuscleSkeletal,
Testis, WholeBlood). Under the pre-specified rule, 1 of 14 reciprocated pairs
meets it on each metric — and it is the same pair, KidneyCortex ↔ MuscleSkeletal.

That pair does not survive scrutiny, and why it doesn't is now the finding:

- **The sign of the difference separates cleanly by cohort power.** Every
  direction calling `shared` has 24–55 pairs per arm; every direction calling
  `specific` has 404–845. No overlap, on either metric.
- **In the low-power cohorts the tissue-specific arm is anti-predicted** —
  signed rho −0.3032 (Muscle→Ovary), −0.2514 (Muscle→Breast), −0.2156
  (Muscle→Kidney), −0.2155 (Muscle→Lung). The model does not merely do worse on
  those pairs; it gets their direction wrong.

That is winner's curse, not mechanism. A pair clearing p < 5e-8 in a cohort with
~130 significant pairs genome-wide, which then fails to replicate anywhere, is
more likely a false positive than a chromatin-mediated mQTL — and noise is
anti-predicted. "Shared > specific" in those cohorts therefore means "real mQTLs
beat false discoveries", which is not the hypothesis.

**The adequately powered comparisons run the other way.** Lung→Testis: specific
rho +0.2359 vs shared +0.0863. Colon→Testis: +0.2272 vs +0.0981. Where the
tissue-specific class is large enough to hold real biology, the model does as
well or better on it.

**For the manuscript:** a sequence-only variant pathway does not separate mQTLs
by mechanism. Report the power artifact explicitly rather than either apparent
effect — 8–10 of 49 directions exclude zero against ~2.5 expected, so there is
real structure, and it is a statement about which cohorts were underpowered, not
about chromatin.

This closes §1 into its final form: methylation *levels* are tissue-specific,
methylation *responses to variants* are not, and the model captures the
tissue-generic part. R2, R3 and R4 become one argument.

Consequence for §5: with no biological finding from R4, the Melody head-to-head
moves from optional to top priority.

---

## 5. Journal decision

Mentor's ladder, with where we stand:

| Target | IF | Requirement | Status |
|---|---|---|---|
| Nat Machine Intelligence | 29.8 | generalizable AI framework across several genomic tasks | not this paper |
| Nature Genetics | 25.5 | major genetic/disease discovery from large cohorts | not this paper |
| **Nature Communications** | **18.1** | cross-tissue + cross-cohort + cross-ancestry + substantial new biological findings | first three done; fourth = **R4** |
| Genome Medicine | 10.8 | clinical relevance, multiple external cohorts, comparison to leading predictors | achievable now; stronger with the Melody head-to-head |

Cross-tissue (nine eGTEx tissues), cross-cohort (GENOA + eGTEx), and
cross-ancestry (TCGA ancestry labels) are all in hand. The one missing
ingredient for NC is a biological finding, and R4 is the only remaining item
positioned to deliver one. **That makes R4 higher leverage than the chromosome
splits**, which are rigor rather than discovery.

Decide the target after R4, not before.

---

## 6. Known vulnerabilities

Write the paper knowing a reviewer will raise these.

1. **"Melody did cross-tissue meQTL prediction first."** True, Fig 3H, 17 Aug
   2026. Answer: they measure effect-size correlation among known meQTLs; we
   measure discrimination against a distance-matched null, which they never
   control for despite their own Fig 3E showing the distance gradient. Do not
   compete on Pearson r — we lose that.
2. **"Your fusion architecture doesn't help for variant effects."** True by
   design. Answer: allele invariance, stated up front in R2 as a mechanism that
   *predicts* R3. This only works if R1's absolute-methylation gain is solid —
   verify it.
3. **"Effect sizes are inverse-normal-transformed."** eGTEx effects are rank/sign
   only, no magnitude calibration. Say so; it is why direction agreement and
   rank correlation are reported rather than calibrated effect sizes.
4. **"Testis and whole blood fail."** Testis is the classic GTEx outlier for
   germline reasons and has 94 significant pairs; whole blood misses 0.5 by
   0.0004. Report both plainly; the pattern tracks power and germline biology,
   not tissue distance from breast.
5. **"37% of pairs were dropped."** CpG-altering variants, uniformly 62.3–62.5%
   retention across tissues, and it removes true mQTLs at a *higher* rate than
   background — a conservative filter. State the filter and the range in
   Methods.
