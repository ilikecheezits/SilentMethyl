# SilentMethyl — lab notes

Single running record for the project. Consolidates the former
`PAPER_FRAMEWORK.md`, `MELODY_COMPARISON.md`, `REQUIREMENTS.md` and the two
presentation outlines. `README.md` remains separate as the repository's entry
point.

Last updated 12 Sep 2026: context ablation and its full downstream rerun
complete, across both cohorts, with the context-permutation mechanism measured.

---

## 0. Status in one page

**The context source changed on 11 Sep. All six retrains and the full
downstream rerun are complete.** See §1.10.

The conclusions are unaffected and R2 is materially stronger. The swap gave a
large methylation-level gain (context arm β MAE 0.1396 → 0.1022, six of six
runs) and **no** variant-effect gain (distance-matched AUROC 0.5586 → 0.5608,
against a seed SD of 0.0117) — the controlled demonstration of allele
invariance that the paper previously argued only on derivational grounds.

Report the PAIRED fusion-minus-sequence statistics, not marginal AUROCs:
the distance-matched metric moves ~0.01 between runs of identical data, while
the paired differences are bit-identical. §1.10 has the evidence.

Every reported single-tissue number needs refreshing from
`results/journal/ablation_breast_epithelium/`. **Do not circulate figures
computed before 12 Sep 2026.**

**R8 chain, 14 Sep 2026 (§6). Task A refutes the allele-invariance claim as
currently written.** The context *embedding* is bit-identical across alleles
(confirmed), but its *gated contribution* is not, and that channel carries
**19% of the variance** of the predicted variant effect at a median 40% of the
sequence channel's magnitude. It adds almost no discriminative signal — the DNA
channel alone recovers ~99% of the model's correlation with measured effects,
which is why R2 still holds. Do not write "context cannot create a variant
effect" anywhere. The corrected wording is in §6 A.

**Task D is a positive result.** Held-out transfer error rises monotonically with
measured cross-tissue methylation variance (ρ = +0.488, +0.475 after mean-β
control; 4.4× MAE spread across deciles), and the failures are **island shores
with enhancer/polycomb chromatin** while **CpG islands with promoter chromatin**
transfer cleanly. Closest thing to a biological finding in the paper. §6 D.

**The ASM validation landed, 16 Sep 2026 (§6 J).** Job `46109189` ran clean.
Discrimination reproduces across two independently built catalogues (Rosenski
0.5625/0.5736, Do & Tycko 0.5419/0.5432, mutually inside CIs, both baselines
null). Signed agreement is now measurable and modestly positive: direction
concordance 0.590 [0.550, 0.630] on 722 SNPs. **That missed the pre-registered
0.60-0.70 band and is written up as a miss** — the model calls the methylated
allele above chance, but less well than the mQTL cohorts predicted.

**Task C** is done and came back **null**:
the gate's DNA/context *share* does not track measured cross-tissue methylation
variance (ρ = +0.042, and −0.061 once methylation level is partialled out). Do
not write that the gate detects tissue-variable loci. Task A (gate
decomposition) is running; note §6 A.1 — the specified Δ_sequence regression
measures cross-model agreement, not gating, and cannot reach R² ≈ 0.99.

The claim is methodological: here is how variant-effect prediction should be
evaluated, here is what happens to two very different models under it, and
tissue-matching buys far less than the field assumes. The biological question we
set out to answer -- can the model separate shared from tissue-specific meQTLs --
came back **no**, with an identified mechanism (winner's curse).

### Mentor requirement scorecard

| # | requirement | state |
|---|---|---|
| 1 | multi-cohort testing | **met** — GENOA (66,495 pairs), eGTEx breast, nine eGTEx tissues, TCGA-BRCA tumour |
| 2 | repeated chromosome-blocked splits | **met, 8 Sep 2026** — four folds, each retrained end to end; see §1 R1 |
| 3 | stronger baselines and ablations | **met** — CpGenie, DeepCpG, k-mer ridge, composition, Melody-MT, Melody-ST |
| 4 | uncertainty calibration | **met** — RC disagreement; calibration must be assessed in logit space |
| 5 | ancestry analyses | **partial, and bounded** — GENOA (African American) with I² = 0%, confounded with tissue and platform; within-cohort contrast impossible (84 EUR / 4 AFR / 1 SAS / 8 unlabelled of 97) |
| 6 | independent variant evaluation | **met** — two cohorts, meta-analysis, null control, distance-matched AUROC |
| 7 | regulatory enrichment | **met, reinterpreted** — ETS coupling is compositional; the k-mer control reproduced it more strongly |
| 8 | multi-tissue jointly-trained framework | **declined on architectural grounds** — see §1 R2. Context is allele-invariant; the proposed build cannot answer the question it was proposed for |
| 9 | functional validation (ASM, ATAC, TFBS, eQTL, eQTM) | **partially met, 16 Sep 2026** — ASM validated against two independent catalogues (§6 J). Discrimination replicates (0.54-0.57, both exclude 0.5); signed agreement on n=722 Do & Tycko SNPs gives direction 0.590, Spearman 0.241, directional AUROC 0.628 — these three are one signed agreement measured three ways, not independent, see §6 J. ATAC/TFBS/eQTL/eQTM remain the gap — see §3 |

### Standing constraints

- Public data only. No controlled-access sources, ever.
- `main.tex` is never revised or deleted. All manuscript work goes to
  `main_revised.tex`, additive, marked `\color{addedred}`.
- New files go in the existing directories; merge into existing scripts rather
  than proliferating new ones.
- STK11 and NCOA2 stay in the analysis (mentor instruction).
- Never write to `/tmp`.
- Keep the local repo synchronised; commit deliberately.

---

## 1. Results framework

State as of 5 Sep 2026, after the nine-tissue transfer analysis.
Companion documents: `REQUIREMENTS.md` (A2 = retraining plan),
`MELODY_COMPARISON.md` (positioning against Jin et al. 2026).

---

### 1. The claim

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

### 2. Results framework

#### R1 — A gated fusion model predicts CpG methylation
*Question:* does combining DNABERT-2 sequence with epigenomic context beat
sequence alone and beat published architectures?
*Scripts:* `10/11/12_train_*`, `13_test_model`, `14_baselines_simple`,
`15_baselines_published`, `16_paired_model_bootstrap`
*Status:* **done.** Three arms x three seeds, vs CpGenie, DeepCpG, k-mer ridge,
composition.

*REPEATED CHROMOSOME-BLOCKED FOLDS COMPLETE, 8 Sep 2026. Values refreshed for the
breast-epithelium context, 12 Sep 2026.* Four folds, each with the gate and the
context tower retrained end to end on that fold's training chromosomes; each
fold's sequence tower is its own and was reused unchanged, so the seq column is
unaffected by the context swap.

**SINGLE SEED (42) THROUGHOUT.** Fold 0 previously carried 0.0993 / 0.1099, which
was a three-seed MEAN sitting in a table of single-seed rows; the two were not
comparable. `main_revised.tex` was corrected at the time and these notes were
not. Test-set results:

    fold  test chromosomes        n        fusion b MAE  seq b MAE  d b MAE   d AUC
    0     chr8, chr9              26,570   0.0885        0.1090     -0.0205   +0.0189
    1     chr12, chr18            26,748   0.0873        0.1040     -0.0167   +0.0154
    2     chr20, chr4             26,783   0.0934        0.1064     -0.0130   +0.0153
    3     chr13, chr15, chr21     26,806   0.0862        0.1049     -0.0187   +0.0165

**The fusion gain reproduces on every split**, mean -0.0173, range -0.0130 to
-0.0205, AUC +0.0153 to +0.0189. Every split agrees in sign on both metrics.

**The published split is now the STRONGEST of the four, not a middling one.**
Like for like, -0.0205 is the largest gain in the table. The old wording --
"neither lucky nor unlucky" -- was an artefact of comparing a three-seed mean
against single-seed folds and must not be reused; say instead that the published
split is the most favourable of the four and that the gain reproduces on all of
them. Absolute performance still varies more than the gain does (0.0862-0.0934),
the expected consequence of chromosomes differing in gene density; the two most
gene-poor folds again give the LOWEST error, so the model is not carried by
promoter-dense regions.

Held-out sizes are matched by construction (26,570-26,806). Sex chromosomes stay
in training for every fold and are never evaluated -- chrY is absent in female
donors and chrX carries X-inactivation, so a fold holding either out would not
measure the same quantity as the rest.
*VERIFIED 12 Sep 2026, breast-epithelium context* (the numbers live in
`ablation_breast_epithelium/paired_model_bootstrap/paired_model_difference_bootstrap.csv`,
NOT in `run_summary.json`, which holds metadata only, and NOT in the
pre-swap `results/journal/paired_model_bootstrap/` copy, which is the 11 Aug
MCF-10A run and must not be quoted). Cross-seed ensemble, 26,570 loci,
257 genomic blocks, fusion minus sequence-only:

    roc_auc    +0.016643 [+0.014663, +0.018769]   P(diff>=0) = 1.0
    beta_mae   -0.018188 [-0.019616, -0.016756]   P(diff>=0) = 0.0
    beta_rmse  -0.032530 [-0.035247, -0.030062]   P(diff>=0) = 0.0
    m_mae      -0.156464 [-0.168563, -0.144415]   P(diff>=0) = 0.0
    m_rmse     -0.254229 [-0.279333, -0.232417]   P(diff>=0) = 0.0

All three individual seeds agree in sign, on all five metrics, with every
interval excluding zero (roc_auc +0.018868 / +0.018317 / +0.015487 for seeds
42 / 43 / 44; beta_mae -0.020442 / -0.020265 / -0.014770).

**The comparison against the epigenomic-only arm has INVERTED and the old
sentence must not be reused.** Pre-swap, the fusion-minus-epi gap was an order
of magnitude larger than fusion-minus-sequence (roc_auc +0.0507, m_mae -0.378).
With the breast-epithelium context the epi arm improves enough that the gap
collapses: ensemble roc_auc **+0.009851** [+0.007679, +0.012420] and m_mae
**-0.159558** [-0.175944, -0.144258]. Fusion now beats the epi-only arm by
*less* AUROC than it beats the sequence-only arm (+0.0099 vs +0.0166), and by
about the same M-value MAE (-0.160 vs -0.156). The two single-modality arms are
now roughly equally far from fusion. Do not write "sequence carries the signal,
context modifies it" on the strength of these numbers -- that ordering was an
artefact of the weak MCF-10A context.

*How to write it.* The gain is **consistent and significant but small**:
+0.017 AUROC and ~1.8 percentage points of beta-value MAE. Say that explicitly.
A reviewer will convert 0.017 into plain language whether or not we do, and
claiming "substantial" here is the kind of overreach that costs credibility on
the parts of the paper that are strong.

#### R2 — The gain is in the baseline, not the variant response
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

#### R3 — Zero-shot transfer across nine tissues
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

*REFRAMED 5 Sep 2026 after the Melody-ST control.* Melody-ST-Breast -- a
single-tissue breast model of a completely different architecture -- transfers
too, and beats fusion on distance-matched AUROC in four of nine tissues (7 of 9
by sign). So the claim is NOT "our architecture transfers". It is **"single-tissue
methylation models transfer, and the training tissue matters far less than
assumed"** -- a finding about the task, demonstrated in two architectures sharing
nothing but the task. That is broader and harder to attack than the version we
were writing. Do not claim SilentMethyl transfers best; it does not.

*Where fusion does win, and it is coherent:* Testis (ST-Breast falls BELOW
chance, 0.4737 [0.3594, 0.5886], rho -0.0740; paired rho +0.2030 and direction
agreement +0.1170 both DIFFERENT), Kidney (paired rho +0.2364 DIFFERENT), and the
Colon extreme tail (+0.2500 at top 0.1%, +0.0962 at top 0.5%, both DIFFERENT).
Fusion holds up in the low-powered and hardest tissues where the U-Net degrades.
Claim that, not more.

*What tissue matching is worth, measured directly:* ST-Lung vs ST-Breast on
identical Lung rows -- same architecture, same training scale, same scoring code,
only the training tissue differs -- gives AUROC -0.0054 [-0.0241, +0.0103], not
distinguishable, with the interval bounding the effect under ~0.024. Matching
helps only in the top 0.5% tail (+0.1157 [+0.0288, +0.2126]). See
section 2, part 3d of this file.
*Must include:* the Melody Fig 3H tension (their tissue-matched tracks *do* win
— because they train per tissue and we do not); the prostate-vs-ovary result
killing the donor-sex explanation; Melody's independent ovary anomaly as
corroboration.

#### R4 — Where transfer fails, and why
*Question:* can the model distinguish shared from tissue-specific mQTLs?
*Scripts:* `40_meqtl_tissue_specificity` (`--stage matched,chromatin`)
*Status:* **done, 5 Sep 2026. The answer is NO** — see the recorded outcome in
§4. Winner's curse, not mechanism.
*Why it matters:* this is the mentor's actual headline question, and it is the
only remaining item that produces a **biological finding** rather than a
methodological one — which is exactly what the Nature Communications bar
requires (§6).

#### R5 — Mechanism and limits
*Question:* what is the model responding to, and when should it not be trusted?
*Scripts:* `50_motif_disruption`, `51_rc_uncertainty`, `52_gwas_enrichment`
*Status:* **done.** Motif disruption with a k-mer-matched null; reverse-
complement disagreement as a calibrated uncertainty signal; GWAS regulatory
enrichment on the two adequately powered tests only.
*Reserved:* slot `53` for allele-specific methylation.

#### R6 — Application: variant prioritisation
*Question:* does this produce candidates a biologist would act on?
*Scripts:* `60`–`64`
*Status:* **done.** STK11 and NCOA2 retained per mentor instruction.

#### R7 — Cross-cohort, cross-ancestry, cross-platform replication
*Question:* does the variant-effect signal survive a genuinely independent cohort?
*Scripts:* `data/harmonize_genoa_meqtl.py`, `20_variant_scoring` (GENOA),
`21_variant_evaluation`, `30_transfer_synthesis`
*Status:* **done** (Aug 2026), and under the same design as the nine-tissue work:
distance-matched negatives at 10 bp tolerance, 1 Mb block bootstrap, distance-only
baseline reported alongside (~0.595), a null stratum that reaches chance.

Precision-weighted meta-analysis over independent 1 Mb LD blocks:

    fusion    rho_meta 0.1775 [0.0678, 0.2829]  p = 0.0016  I2 = 0.0  Q p = 0.449
    sequence  rho_meta 0.1698 [0.0599, 0.2757]  p = 0.0026  I2 = 0.0  Q p = 0.546

    eGTEx conditional   breast, European-dominant     81  rho 0.610 [0.452, 0.731]
    eGTEx regular       breast, European-dominant    418  rho 0.246 [0.114, 0.402]
    GENOA               whole blood, African American 4037 rho 0.152 [0.117, 0.188]
    GENOA null control  tested but null pairs           -  rho -0.004 [-0.021, +0.013]

**I2 = 0.** No detectable heterogeneity between cohorts that differ in tissue,
platform AND ancestry. The weighting is by independent LD blocks, not pair
counts -- the run summary states that pair-count weighting "would understate the
variance by roughly an order of magnitude and manufacture significance". Keep
that sentence; self-imposed conservatism is worth more than the extra stars.

*Two things it confirms without being designed to.* fusion (0.1775) and sequence
(0.1698) are indistinguishable, so **R2 replicates in an independent cohort**.
And GENOA is blood scored with breast context, so **R3 holds cross-cohort** too.

*The ancestry claim, stated exactly.* GENOA is African American and eGTEx is
European-dominant, so this is the cross-ancestry comparison -- but GENOA also
changes tissue and platform, so ancestry cannot be isolated from it.

**A within-cohort ancestry contrast was attempted and is not possible.** GDC open
ancestry calls (CCG-AIM-2020, no controlled access) applied to the TCGA-BRCA
training normals give **84 EUR, 4 AFR, 1 SAS, 8 unlabelled out of 97 donors**.
Four donors cannot support an error estimate, and `18_ancestry_stratified_error.py`
refuses to compute one. Write the limitation with those numbers in it: it shows
the alternative was checked rather than overlooked. Variant-effect stratification
by ancestry would additionally need per-donor genotypes, which are
controlled-access and deliberately not used.

*What the nine-tissue work changes.* The 9/1 presentation says ancestry and
tissue are confounded and "we can't separate which difference the model
survived". That is now one dimension narrower: R3 varies tissue alone across
nine European-dominant eGTEx cohorts, so tissue is measured on its own and
GENOA's marginal contribution is ancestry plus platform. Update the slide.

#### Superseded
`70_mqtl_positive_control` (81 pairs) and `71_mqtl_matched_negative` (35 pairs)
are dwarfed by the nine-tissue cohort (13,744 significant pairs). Delete
together with Supplementary S2/S3 in one coordinated manuscript edit — not
before, since the text still cites them.

---

### 3. Work plan, in order

Status as of 5 Sep 2026, end of day. Twelve of thirteen mentor items are closed;
the analysis phase is essentially over and the bottleneck is writing.

#### Running now
1. **Folds 1-3** (job 45290715, ~30 h remaining). Epoch 1 complete on all three,
   beta MAE 0.115-0.124, healthy. Fold 0 is the published split and is NOT
   retrained. On completion: score each fold, evaluate, and write the spread into
   R1 and Methods. The jobs self-check that the split CSVs did not change
   mid-run and exit 3 if they did -- read that line before trusting a fold.

#### Next, in order
2. **Update R3 in `main_revised.tex`** for the ST control. The subsection
   currently frames single-tissue transfer as our observation; it must become the
   architecture-independent version. Add the ST results to the Melody subsection.
3. **Mentor email.** Every number in it is now checkable. The multi-tissue
   request gets the architectural answer from R2, not a resource complaint.
4. **`references.bib`** needs a real entry for `jin2026melody`; only the
   placeholder bibitem exists.
5. **Grep `main.tex` and `main_revised.tex`** for any sentence implying
   epigenomic context improves VARIANT-EFFECT prediction. R2 forbids it.
6. **Delete the superseded `70_mqtl_positive_control` (81 pairs) and
   `71_mqtl_matched_negative` (35 pairs)** together with Supplementary S2/S3 in
   ONE coordinated manuscript edit -- not before, since the text still cites them.
7. **Commit the outstanding pieces**: `melody_env_freeze.txt` (never created),
   and the REQUIREMENTS.md notes on the selene_sdk stub, torch 2.6.0+cu124 for
   sm_70, and the margin-1 / sigmoid_first findings.

#### One decision, not yet made
8. **eQTL / eQTM colocalisation.** The only remaining substantive analysis and
   the only route to "new biological findings". Public GTEx, donors overlap
   eGTEx, inference plus joins. **Take the prior seriously before committing:**
   four disease-variant tests already returned null (ClinVar, breast GWAS,
   all-trait GWAS in both cohorts), and the honest conclusion recorded on slide 8
   is that "the model ranks methylation change, not clinical variant relevance".
   Expression sits closer to methylation than disease does, so this is a better
   bet than those were -- but it is a bet. Decide once, after the folds land.

#### Deferred, with reasons
- **Methylation-LEVEL head-to-head against Melody.** Never tested; the whole
  Melody comparison is variant effects. Needed before any claim about having the
  better methylation predictor.
- **Loyfer WGBS atlas (GSE186458) instead of TCGA.** Would remove the field-
  cancerisation limitation. Large enough (205 samples, 39 cell types, 28.2M CpGs)
  but a substantial re-plumbing.
- Pooled TCGA baseline, BEND multi-task validation (manifest entry exists, data
  MISSING -- the Nature Machine Intelligence criterion, and out of scope for
  this paper).
- **Allele-specific methylation** (slot 53) is NO LONGER deferred, 15 Sep 2026 --
  see section 6, `E (REOPENED)`. It was deferred, then dropped on a power
  calculation that used the wrong denominator and the wrong catalogue. Slot 53
  is live and awaiting authorisation.

#### Closed, do not reopen
- Multi-tissue joint training: declined on architectural grounds (R2), not
  resources. Context is allele-invariant; the proposed build cannot answer the
  question it was proposed for.
- Within-cohort ancestry stratification: impossible. 84 EUR / 4 AFR / 1 SAS /
  8 unlabelled of 97 TCGA breast normals. GENOA is the ancestry evidence.

---

### 4. R4 prediction, recorded before the run

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

#### R4 OUTCOME, recorded 5 Sep 2026 — the answer is NO

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

### 5. Journal decision

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

### 6. Known vulnerabilities

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
6. **"Your fold table compares an ensemble to single seeds."** It did. Fixed
   9 Sep 2026 — see §7.
7. **"Your power diagnostic only fires for your own model."** It did. Fixed
   9 Sep 2026 — see §7.

### 7. Pre-email audit, 9 Sep 2026

Prompted by the same question that produced the Melody R4 run: *where did we
assume a control rather than run it?* Four items were closed; the rest are
recorded as limitations.

**A. The `41` power diagnostic was one-sided — FIXED.**
`power_diagnostic()` tested only `shared.max_n < specific.min_n`, so it could
detect the confound only when the `shared` calls were the underpowered side.
That is SilentMethyl's shape. Melody's R4 failure runs the other way, so the
test stayed silent for Melody, and the silence was read as "Melody has no power
confound" — which the test never checked. Now two-sided, reports
`low_power_call`, and checks BOTH arms for anti-prediction rather than only the
tissue-specific one. A `--label` flag was added so the two runs are
distinguishable in a scrollback.

**A2. Both models re-summarised with the patched script — the answer is clean.**
Logs: `logs/r41_melody.txt`, `logs/r41_silentmethyl.txt` (cluster).

| | SilentMethyl fusion | Melody-MT |
|---|---|---|
| directions attempted / informative | 72 / 49 | 56 / 42 |
| pairs meeting rule, direction agreement | 1 of 14, favouring **shared** | 1 of 14, favouring **specific** |
| pairs meeting rule, signed rho | 1 of 14, favouring **shared** | **0 of 14** |
| low-power side | `shared` (n 24–67) | `specific` (n 24–304) |
| high-power side | `specific` (n 404–850) | `shared` (n 593–692) |
| anti-predicted arm | specific, 8 directions, −0.10 to −0.30 | shared, 2 directions, −0.1046 / −0.0357 |

Both separate cleanly by n; **both favour whichever class sits in their
least-powered cohorts; the directions are mirrored.** That mirroring is the
whole result — one model failing is a model limitation, two models failing in
opposite directions with the same power signature locates the problem in the
benchmark. Melody's anti-predicted *shared* arm would have printed nothing under
the old one-sided code, so the patch produced evidence, not just tidier output.

Written into `main_revised.tex` §`sec:tissue-specificity`, which is retitled
"Neither model separates shared from tissue-specific meQTLs" and now discloses
that our diagnostic was one-sided before the Melody comparison was run. Two
figures in the original paragraph were also corrected against the fresh run:
shared-arm cohort range 24–55 → **24–67**, specific-arm range 404–845 →
**404–850**, anti-prediction floor −0.15 → **−0.10**.

**B. Fold 0 was a three-seed mean; folds 1–3 are seed 42 — FIXED.**
The table row `0.0993 / 0.1099 / 0.9680 / 0.9569` is exactly the mean over
seeds 42/43/44 of the published-split test metrics. Folds 1–3 were trained at
seed 42 only. The rows were therefore not comparable. Seed-42-only values for
the published split are `0.0971 / 0.1090 / 0.9699 / 0.9576`.

| | ΔβMAE | ΔAUC |
|---|---|---|
| fold 0, seed 42 (correct) | −0.0119 | +0.0122 |
| fold 0, 3-seed mean (was in table) | −0.0106 | +0.0111 |
| fold 1 | −0.0098 | +0.0100 |
| fold 2 | −0.0074 | +0.0093 |
| fold 3 | −0.0111 | +0.0110 |

*The 3-seed paired-bootstrap ensemble row (−0.0104 / +0.0105) was REMOVED on
13 Sep 2026, not updated.* It was the pre-swap MCF-10A ensemble, and the new
value is −0.0182 / +0.0166 (R1) — but dropping that into this table would have
compared a new-context ensemble against four old-context folds. The folds cannot
be refreshed: `results/journal/ablation_breast_epithelium/fold{1,2,3}/` carry
`epi` and `fusion` only, and these deltas are fusion **minus sequence**, so
there is no per-fold sequence arm to difference against. Quote the ensemble from
R1, where it sits beside its own context; quote the folds from here, where every
row is seed 42 on MCF-10A. Do not mix them in one table.

This *changes a claim*. The old text said the published values sat "inside both
ranges" — i.e. the published split was unremarkable. Like-for-like, the
published split gives the **largest** gain of the four folds. `main_revised.tex`
now says so explicitly, reports the full spread (+0.0093 to +0.0122), and the
caption states that every row is seed 42. There are three legitimate estimators
of one quantity here (single-seed, mean-of-seeds, prediction-ensemble); the rule
now is that the fold table is single-seed throughout and the ensemble figure is
only ever quoted as the ensemble.

**C. NCOA2 was justified partly by "active context" — FIXED in the discussion.**
R2 says context is near-inert for allelic contrasts, so the chromatin state at
cg20699548 cannot have contributed to the predicted Δβ. The discussion now
states that the ranking is a sequence-model result and the chromatin annotation
supports only testability. The Results paragraph was left alone — it describes
the locus factually and makes no causal claim. `main.tex` untouched, per
standing instruction; its §328 carries the un-caveated version and should not
be circulated as the current text.

**D. Fold provenance guards — NOT YET VERIFIED.** Cluster-only; the fusion
rerun logs were never synced. See §3.

Recorded as limitations rather than run: Melody-ST was validated structurally
but never functionally (no published ST number exists to reproduce; the
`--random-init` control is available if wanted); no methylation-*level*
head-to-head against Melody exists, and the targets are not aligned so it is not
a cheap run; the k-mer/CpGenie/DeepCpG baselines never went through nine-tissue
transfer, though Melody-ST already showed transfer is architecture-independent
(7/9). Melody's own benchmark cannot support a distance-matched null at all —
p<1e-5 and |effect|>0.5 leaves zero negatives — which is the reason both models
were evaluated on ours, and that should be stated as a design justification
rather than left as an apparent gap.

**Still to check:** whether the winner's-curse explanation is written as
demonstrated or as hypothesised. Cohort size predicts per-dataset performance at
r=+0.323 (p=0.44) — i.e. not at all — while distance predicts at −0.946. The
data support the distance story; the power story is an interpretation and must
be worded as one.

### 8. Reference-track provenance, recovered 10 Sep 2026

`data/reference/*.bw` had **no recorded provenance at all** — `build_data.sh`
neither fetches them nor lists them among required inputs, `build_training_data.py`
opens them by hard-coded filename, and `external_manifest.json` (which records
URL, bytes and sha256 for the ClinVar VCF) had no entry for them. They were
downloaded by hand from the ENCODE portal and renamed, which loses the accession
from the filename but not from the bytes. `data/audit_reference_tracks.py`
recovers it by matching each file's md5 against the portal. All eight are hg38
natively; nothing was lifted despite the chain file in that directory.

| feature | accession | assay | output type |
|---|---|---|---|
| H3K4me3 | ENCFF548SFG | Mint-ChIP-seq | fold change over control |
| H3K27ac | ENCFF282YCX | Mint-ChIP-seq | fold change over control |
| H3K27me3 | ENCFF274LWG | Mint-ChIP-seq | fold change over control |
| H3K9me3 | ENCFF423DKY | Mint-ChIP-seq | fold change over control |
| H3K36me3 | ENCFF634LDP | Mint-ChIP-seq | fold change over control |
| H3K4me1 | ENCFF714NIL | Mint-ChIP-seq | fold change over control |
| ATAC | ENCFF021PIS (ENCSR037XNN) | **snATAC-seq** | **BAM, converted locally** |
| PhyloP | — | UCSC hg38.phyloP100way | not ENCODE |

All MCF 10A. Two things follow.

**The Methods sentence is currently wrong in two details.** The histone marks are
**Mint-ChIP-seq**, not conventional ChIP-seq. And accessibility is not "ATAC-seq
signal from ENCODE": it is a coverage track we generated ourselves from an
unreplicated single-nucleus ATAC BAM produced by a lab-custom pipeline
(ENCAN638MKH). Both are fair to use; neither is what the draft says. Fix before
submission — a reviewer who clicks the accession sees it immediately.

**The scale worry was overblown — checked and mostly dismissed.** The epi tower
feeds raw values into its first `nn.Linear` with no per-feature standardisation
(only median imputation), so scale differences do reach the model. But the
measured ranges do not single ATAC out:

    ATAC      max  90   mean@probe 0.151
    H3K4me3   max 584   mean@probe 2.735
    H3K27ac   max 245   mean@probe 0.768
    H3K36me3  max 233   mean@probe 1.378
    H3K27me3  max  49   mean@probe 0.099
    H3K9me3   max  27   mean@probe 0.143
    H3K4me1   max  31   mean@probe 0.760
    PhyloP    -20..10   mean@probe -0.243

ATAC sits in the middle of the pack on typical values and is *below* three of the
histone marks on maxima. The spread is mostly within the fold-change tracks,
which is what fold-change signal looks like. So this is a disclosure issue, not
a numerical one. Do not rebuild anything over it.

**Still not recorded:** the `bamCoverage` (or equivalent) command that produced
`ATAC_seq.bw`, and the phyloP download URL. The manifest marks the first
`CONVERSION COMMAND NOT RECORDED`. No API can recover it, and it must be
repeated identically for any tissue added later.

### 9. Joint multi-tissue model — the plan, and the first result

Assume the mentor asks for this. Design is settled. The plan below is unchanged;
the **zero-shot transfer arm has since been built and run** — see §9a.

**Source: TCGA solid-tissue-normal, not eGTEx.** GSE213478 (eGTEx methylation,
987 samples, 9 tissues, EPIC, normal donors, beta matrix public — only IDATs are
dbGaP-restricted) is the obvious alternative and is genuinely better data. It is
rejected because switching platform and source would invalidate every published
number: R1, the four folds, three seeds, CpGenie/DeepCpG, all variant scoring.
Staying on TCGA HM450 means the existing breast model **is** the single-tissue
control at zero retraining cost. Mention eGTEx to the mentor as
considered-and-rejected so it does not look unexamined.

**Tissue list**, from `data/survey_tcga_normal_cohorts.py` (10 Sep 2026). BRCA
came back at exactly 97 donors, matching the published cohort, which validates
the query:

| tissue | TCGA | donors | ENCODE |
|---|---|---|---|
| Kidney cortex | KIRC/KIRP/KICH | 205 | 7/7 |
| Breast | BRCA | 97 | 7/7 |
| Lung | LUAD+LUSC | 74 | 7/7 |
| Prostate | PRAD | 50 | 7/7 |
| Colon transverse | COAD+READ | 45 | 7/7 |
| Muscle, ovary, testis, whole blood | — | 0 | context exists, no targets |

Restrict kidney to KIRC (± KIRP): KICH is chromophobe, from the distal nephron
rather than the cortex, and the eGTEx tissue is Kidney Cortex specifically.

**The best property of this design:** the four dropped tissues all have eGTEx
mQTLs *and* ENCODE context, so they can be scored even though they cannot be
trained on. They become a genuine unseen-tissue benchmark — held out because the
data does not exist, not because we chose them. That is the Melody-G comparison
the mentor asked about, and it falls out for free.

**Budget and merging.** Training rows are probes x tissues, ceiling one cohort's
worth (418,486) to keep a run near the current ~46 h. Five tissues gives ~84k
probes each. Subsample **training only**, by cross-tissue variance — most HM450
probes are constitutively methylated everywhere and carry no tissue information.
Keep val/test at the full probe set or the numbers stop being comparable to the
published model. Hold out the **same chromosomes in every tissue**, or a probe
held out in breast re-enters through lung. Consider weighting rather than an even
split: a target median over 45 donors is a noisier label than one over 205.

**Context must be rebuilt for all five, including breast.** Section 8 shows
breast accessibility is snATAC; the other tissues have bulk ATAC-seq, which is
what the survey matched on. Repeating the current recipe would require snATAC in
kidney/lung/prostate/colon, which likely does not exist. So move all five to bulk
ATAC + the same six marks at the same output type — which changes breast's
context, and therefore requires one extra single-tissue breast run as the matched
control. One training run buys an interpretable comparison.

**Scope it to methylation levels.** Allele invariance does not change when
tissues are added; context is still identical for REF and ALT. If this is sold as
improving tissue-specific *variant* effects it will fail again for the reason
already documented in R2.

#### 9a. It was built after all — holdout_BreastEpithelium, seed 42, 13 Sep 2026

The zero-shot transfer arm is done. Trained jointly on KidneyCortex, Lung and
ColonTransverse with BreastEpithelium **entirely held out of training**, then
tested on the breast test split. Test set verified 100% BreastEpithelium
(`Tissue` column, 26,570/26,570) — no leakage.

| arm | β MAE | AUC |
|---|---|---|
| sequence-only | 0.11123 | 0.95456 |
| context-only | 0.10692 | 0.96211 |
| **fusion** | **0.09775** | **0.96933** |

n = 26,570. `results/journal/joint/holdout_BreastEpithelium/seed42/`.

**Context-only again beats sequence-only on both metrics.** Same inversion as the
paired bootstrap, now reproduced independently on a model that never saw breast
during training — so it is not an artifact of the single-tissue fit.

**Transfer penalty is small, but mind the estimator.** The 3-seed
breast-*trained* means are fusion 0.09140 and sequence 0.10990, giving +0.0064
and +0.0013. Those are the numbers to quote for the headline, but the joint model
is **seed 42 only**, so this is single-seed against mean-of-seeds — the exact mix
§3B forbids in the fold table. Like-for-like against seed 42 alone
(`ablation_breast_epithelium/seed42/`: fusion 0.08854, sequence 0.10898) the
penalties are **+0.0092** and **+0.0023**. State which estimator is in use
wherever this is quoted; do not let the two versions travel unlabelled.

**The transfer comparison is exact.** `data/datafiles_joint/holdout_
BreastEpithelium/test.csv` and `data/datafiles_breast_epithelium/test.csv` are
locus-identical: same 26,570 rows, identical `(chr, pos, probeID)` key sets,
**same row order**, and `Median_Beta` agreeing to 0.0. The joint file adds a
`Tissue` column; otherwise the schemas match. No caveat sentence is needed.

**Gate statements stay at the distribution level.** `gate_dna_share_fwd_rc_mae`
is 0.059 on test, so the per-locus DNA share is not quotable. Distribution-level
only: share mean 0.535, median 0.525, q10–q90 0.342–0.726; DNA-dominant 33.7%,
balanced 49.4%, context-dominant 16.9%.

**`per_tissue_metrics.json` reports 0.10116, and that is not a contradiction.**
`run_joint_multitissue.sbatch` selects the first `pred`-matching beta column,
which is `pred_beta_fwd` — **forward strand only**. `metrics.json` reports
forward/RC-averaged (`pred_beta_rc_avg`). Verified on the same predictions.csv:
fwd 0.10116, rc 0.10054, rc_avg 0.09775; `beta_fwd_rc_mae` 0.0503 is the
fwd-vs-rc spread that the 0.0034 gap sits inside. The per-tissue writer now
records `inference`, `pred_column` and `true_column` in the JSON, and the
existing files were backfilled — **all MAEs bit-identical**, only the labels are
new. Quote `metrics.json`.

One cosmetic note: the per-tissue split keys off `probeID` containing `__`. The
`all4` probeIDs carry it and split correctly into four tissues; the single-tissue
holdout probeIDs do not, so its one bucket is labelled `ALL` rather than
`BreastEpithelium`. Harmless — the bucket is the whole test set either way.

**The `all4` arm also landed** (45812872, 13 Sep 21:54): fusion β MAE 0.09522 /
AUC 0.97018, sequence 0.11207 / 0.95296, over all 106,278 loci. Its per-tissue
split works as intended — BreastEpithelium 0.09429, KidneyCortex 0.09513,
Lung 0.09692, ColonTransverse 0.10390 (forward-only, as labelled). It was written
by the pre-patch script, so all six joint per-tissue files have now been
backfilled with `inference` / `pred_column` / `true_column`; every `n` and
`beta_mae` was asserted bit-identical first.

---

### 10. Context ablation — breast epithelium replaces MCF-10A, 11 Sep 2026

**The published context was a cell line; the targets are primary tissue.** Six
Mint-ChIP marks from MCF-10A, a clonal immortalized line, plus an ATAC track
converted locally from a snATAC BAM (ENCFF021PIS, ENCSR037XNN) with parameters
that were never recorded. The domain mismatch was a stated limitation. ENCODE
also has primary breast epithelium with all seven marks as bulk, fold-change-
over-control bigWigs from one biosample through one pipeline — so the swap was
testable rather than merely arguable.

New context, `data/reference/BreastEpithelium/`, all checksum-verified, GRCh38:

| mark | accession | | mark | accession |
|---|---|---|---|---|
| ATAC | ENCFF665NGK | | H3K36me3 | ENCFF714QJF |
| H3K4me3 | ENCFF653CLL | | H3K9me3 | ENCFF481QEK |
| H3K4me1 | ENCFF234JZW | | H3K27ac | ENCFF085IYD |
| H3K27me3 | ENCFF212ZFW | | | |

#### Result — COMPLETE, six runs, 12 Sep 2026

Three seeds on the published split and three chromosome folds. **Every one of
the six favours breast epithelium, on both arms and both metrics.**

β MAE (lower is better):

| run | arm | MCF-10A | breast epi | Δ | | arm | MCF-10A | breast epi | Δ |
|---|---|---|---|---|---|---|---|---|---|
| seed42 | fusion | 0.0971 | 0.0885 | −0.0085 | | context | 0.1396 | 0.1022 | −0.0374 |
| seed43 | fusion | 0.0988 | 0.0901 | −0.0087 | | context | 0.1394 | 0.1022 | −0.0372 |
| seed44 | fusion | 0.1020 | 0.0956 | −0.0063 | | context | 0.1396 | 0.1023 | −0.0373 |
| fold1 | fusion | 0.0942 | 0.0873 | −0.0069 | | context | 0.1331 | 0.0980 | −0.0350 |
| fold2 | fusion | 0.0990 | 0.0934 | −0.0056 | | context | 0.1360 | 0.0985 | −0.0375 |
| fold3 | fusion | 0.0938 | 0.0862 | −0.0077 | | context | 0.1328 | 0.0987 | −0.0342 |

- **fusion** ΔMAE −0.0073 ± 0.0012, ΔAUC +0.0061 ± 0.0006
- **context-only** ΔMAE −0.0364 ± 0.0014, ΔAUC +0.0445 ± 0.0027

**It is not seed noise.** The published three-seed fusion spread is 0.0971 /
0.0988 / 0.1020 (mean 0.0993, SD 0.0025). Seed 42's ablation value of 0.0885
sits 4.3 SD below that mean, and the smallest of the six effects (fold2,
−0.0056) is still more than twice the seed SD. Six of six in the same direction
on two metrics settles it.

**Have an answer ready for the context-arm seed stability.** The ablation
context numbers are near-identical across seeds — 0.1022, 0.1022, 0.1023, with
AUC 0.9658 three times — which reads as a bug at first glance. It is not: the
PUBLISHED context arm does the same thing (0.1396, 0.1394, 0.1396). A small MLP
over 16 features with 345k training rows has almost no seed sensitivity. State
this in the paper before a reviewer asks.

#### Why it moved — mechanism, not luck

The context tower applies **no per-feature standardization**: raw bigWig values
go straight into `nn.Linear(tabular_dim*2, 128)`, with `LayerNorm` only after
that first linear. Input scale and input artifacts therefore reach the weights
directly. Two concrete defects in the MCF-10A set:

- **Blacklist artifacts in the ATAC track.** Mean signal inside ENCODE-excluded
  regions is 5.4568 against 0.001 in the flanks — roughly 5000x — with a maximum
  of 90. The locally converted track never had the exclusion applied.
- **Scale incoherence.** MCF-10A track means span 0.099–2.735 with maxima
  27–584; the ATAC track is a different quantity from the histone tracks.
  Breast epithelium means span 0.295–1.960 with maxima 8–137, and its ATAC mean
  of 1.006 sits *inside* the histone range.

Primary tissue matching primary-tissue targets is the third reason, and the one
that closes the stated limitation, but the two above are why the effect is large.

#### The asymmetry is the publishable finding

Averaged over all six runs: the context arm improved by 0.0364, fusion by
0.0073. **Only 20% of the context gain survives the gate** — roughly four fifths
of what better chromatin tracks provide is already encoded by the sequence
tower. The ratio barely moves between runs, so it is a property of the
architecture rather than of one split.

#### And it buys NOTHING for variant effects — the controlled version of R2

This is the sharpest result in the paper and it did not exist before 12 Sep.

GENOA, discrimination against the distance-matched null (2000-replicate 1 Mb
block bootstrap, distance-only forced to exactly 0.5000 by construction):

| model | context | mean AUROC | seed SD |
|---|---|---|---|
| sequence-only | none | **0.5606** | 0.0044 |
| fusion | MCF-10A | **0.5586** | 0.0117 |
| fusion | breast epithelium | **0.5608** | 0.0065 |

Per-seed, breast epithelium: 0.5624 / 0.5536 / 0.5663, ensemble 0.5645.
Per-seed, MCF-10A: 0.5452 / 0.5665 / 0.5642, ensemble 0.5700.

All three means lie within **0.0022** of each other against seed SDs of
0.004–0.012. The context swap moved variant discrimination by about one fifth
of the MCF-10A seed spread.

**So the same intervention improved methylation-level prediction by 27% on the
context arm and changed variant-effect discrimination by 0.002 AUROC.** That is
allele invariance demonstrated rather than derived: context is identical for REF
and ALT, so it can rescale a predicted effect but cannot create one or set its
direction. It also kills the obvious reviewer objection — "your chromatin data
is simply bad" — with data instead of argument. We improved the chromatin data,
verified the improvement six ways, and the variant result did not move.

Note also that **sequence-only matches both fusion models** on this benchmark.
The claim is that the three are statistically inseparable; the CIs overlap
heavily, so do NOT write that fusion underperforms sequence.

From the same table, the evaluation argument in one row: `auroc_marginal` 0.5939
against `auroc_distance_only_baseline` 0.5953 — distance alone beats the model
marginally, and signal appears only after matching (0.5623 within distance bin).
Same pattern in the CpG-altering class, 0.6253 against 0.6292. Now reproduced on
a second context.

This lands inside the ≤0.02–0.027 bound on scale/tissue matching established by
four independent routes in §2.3b/3d, corroborating that bound rather than
disturbing it.

Use it in R2 and in the §1.9 refusal of joint multi-tissue training: the reason
joint training will not deliver tissue-specific *variant* effects is
architectural, and this is the experiment that shows better context does not
rescue it.

#### eGTEx reproduces it — and forces a change in WHICH statistic we report

Both cohorts, fusion minus sequence, paired on identical pairs:

| metric | GENOA | eGTEx |
|---|---|---|
| signed rho | −0.0010 [−0.0078, +0.0051] | +0.0117 [−0.0085, +0.0320] |
| direction agreement | +0.0017 [−0.0050, +0.0079] | +0.0048 [−0.0169, +0.0234] |
| AUROC marginal | −0.0024 [−0.0060, +0.0009] | +0.0028 [−0.0118, +0.0143] |
| AUROC within distance bin | −0.0023 [−0.0058, +0.0010] | +0.0057 [−0.0094, +0.0176] |

Eight intervals, all spanning zero, in the cross-tissue cohort and the
tissue-matched one.

**REPORT THE PAIRED NUMBERS, NOT THE MARGINAL ONES.** The distance-matched
AUROC is rebuilt per evaluation (matched-cohort construction plus block
bootstrap), and it moves by ~0.01 between runs of *identical* data:

- eGTEx sequence arm, same weights and same score files, two evaluations:
  0.5781 and 0.5900 — a 0.012 gap from nothing at all.
- GENOA fusion ensemble, two runs: 0.5645 and 0.5559 — 0.0086.

The paired differences were **bit-identical** across the same two runs, because
they are computed on the same pairs rather than on a resampled cohort. So any
marginal difference below ~0.012 is unreadable, and the equivalence claim must
rest on the paired test. Quoting a 0.002 marginal difference as evidence would
be indefensible.

With that caveat, marginal distance-matched means (three seeds):

| cohort | sequence | fusion / MCF-10A | fusion / breast-epi |
|---|---|---|---|
| GENOA | 0.5606 | 0.5586 | 0.5608 |
| eGTEx | 0.5781–0.5900 (same model, two runs) | 0.5842 | 0.5883 |

#### Context permutation — the mechanism, measured

`23_context_permutation.py`, eGTEx heldout, fusion seed 42, 76,893 pairs.
Context is replaced by a shuffle or by the per-feature median and the model is
rescored. Agreement with the unperturbed run, MAE expressed in units of the
reference SD so the two quantities are comparable:

| scheme | quantity | pearson | MAE / SD | sign agreement |
|---|---|---|---|---|
| shuffle | absolute methylation (WT_M) | 0.632 | **0.526** | 0.752 |
| shuffle | variant effect (Delta_M) | 0.753 | **0.227** | 0.844 |
| median | absolute methylation (WT_M) | 0.953 | **0.353** | 0.943 |
| median | variant effect (Delta_M) | 0.862 | **0.186** | 0.836 |

**Destroying the context moves the methylation LEVEL 2.3x more than it moves the
variant EFFECT** (0.53 SD vs 0.23 SD under shuffle; 1.9x under median).

Do NOT over-claim this as "context does nothing for variant effects" — 0.23 SD
is not zero. It is exactly what allele invariance predicts: an identical context
for REF and ALT can **rescale** a predicted effect through the gate, which a
permutation will perturb, but cannot **create** one or set its direction. The
level/effect ratio is the quantitative form of that statement, and it is the
mechanistic counterpart to the statistical equivalence above.

#### Decision and cost

Swap. The 0.0086 is the tiebreaker, not the argument. The argument is that the
swap retires the domain-mismatch limitation with data instead of a caveat, and
removes an ATAC track with blacklist artifacts that a reviewer could find
unaided — much worse to defend than to have fixed. Seven fold-change tracks, one
primary biosample, one pipeline, all checksum-verified is a provenance statement
that fits in one sentence.

Cost: every single-tissue number in the paper is refreshed. Downstream
reanalyses (variant scoring, meQTL discrimination, tissue specificity) consume
model predictions and cannot start until the retrains land, but none need a GPU.

#### All six retrains completed 12 Sep 2026

Jobs 45804079–83 plus the original seed-42 run, ~14 h each, all reusing the
published sequence towers (`checkpoints_journal/seed{43,44}/sequence/` and
`checkpoints_folds/fold{1,2,3}/sequence_seed42/`). The sequence tower reads DNA
only and never sees context, so it needs no retraining — ~32 of ~46 h saved per
run, and the sequence-only baseline stays untouched and comparable.

That reuse is also why the downstream pipeline rescores **fusion only** and
links the published sequence scores in, verifying the checkpoint md5 first
(`scripts/run_ablation_analyses.sbatch`, STAGE=link).

#### Downstream reruns, launched 12 Sep 2026

Everything downstream was computed on MCF-10A models and is stale.
`scripts/run_ablation_analyses.sbatch` reruns it in four stages — link, score
(GPU array 0-5), ctxperm (GPU), analyse (RM-shared, CPU). All output lands under
`results/journal/ablation_breast_epithelium/`, so the MCF-10A results survive
for the side-by-side.

Scoring verified complete: GENOA 66,495 rows × 3 seeds, eGTEx 76,893 × 3, zero
reference-base mismatches, every task confirmed to have used the ablation
checkpoints and the breast-epithelium splits.

**Not included, and why:** 60/61/62 candidates, 22 context stratification, 63
known-variant, 64 literature screen, 91 figures. 22 and 91 hard-require the
candidate CSVs that 60 produces; 60 and 62 are GPU jobs of their own. 64 shelled
out to 63 without forwarding `--weights-template`, so it would have silently
scored the OLD MCF-10A checkpoints and reported a result that looks correct.

*Superseded 13 Sep 2026: **64 has been fixed** and now forwards both
`--weights-template` and `--split-template` (`run_scorer`, and the `--weights-
template` help text spells out the failure mode). Both must be set on the command
line for an ablation run — the defaults still point at `checkpoints_journal` and
`data/datafiles`. The chain is safe to run; `jobs/r7_ablation/64_literature.sbatch`
sets both.*

**Three silent failures this pipeline surfaced — all fixed, all worth knowing.**

1. **`40 --stage all` IGNORES `--output-dir`** and writes to its own stage
   defaults, which are the PUBLISHED paths. On 12 Sep it overwrote four MCF-10A
   result files with breast-epithelium numbers *while reporting success*; the
   only reason it was caught is that `41` then failed looking for an ablation
   directory that did not exist. Restored by rerunning `40` with defaults
   (it reads the untouched MCF-10A scores, so it regenerates exactly).
   `run_ablation_analyses.sbatch` now calls the two stages separately and ends
   every analyse run by checking whether either published path was modified in
   the last 30 minutes. Same footgun as `51 --stage all`.

   Note: `git checkout` will NOT restore these correctly — the committed
   `run_summary.json` predates the 11 Sep changes to `40` and is a different
   schema. Regenerate rather than revert.

2. **`caveats.tissue` in `21_variant_evaluation.py` was hardcoded** to "MCF-10A
   breast" and kept saying so after the swap, writing false provenance into
   every `run_summary.json`. Now derived from the weights paths recorded in the
   scoring summaries, reports "unrecorded" when it cannot tell, and shouts
   `MIXED CONTEXT SOURCES` if one directory contains both.

3. **Schema drift in `20_variant_scoring.py`.** It began writing
   `beta_ref_to_alt` and `pvalue` INLINE some time after 25 Aug; older files
   rely on `21` merging them from the input CSV. Linking old sequence scores
   beside new fusion scores produced a directory with both schemas, and every
   per-model sequence metric came back `n/a` — with no error, because the paired
   fusion-minus-sequence numbers computed fine. GENOA was affected, eGTEx was
   not. Fixed by rescoring the sequence arm (`MODELS=sequence`, ~6 GPU-hours;
   the tower is unchanged, only the column set differs). **Any pipeline that
   links old scores beside new ones needs a column check at link time.**

#### R3 settled — Lung transfer re-scored on the new context, 13 Sep 2026

The nine-tissue eGTEx transfer table was computed on MCF-10A fusion scores. The
largest well-powered tissue was re-scored (array 45849084, three seeds, all
COMPLETED) and `31_transfer_discrimination` re-run on it.

| | β AUROC, distance-matched |
|---|---|
| fusion, MCF-10A (published referent) | 0.5880 [0.5669, 0.6091] |
| fusion, breast epithelium | **0.5857 [0.5654, 0.6071]** |
| shift | **−0.0022** |

**Inside the interval — R3 carries over.** No rerun of the other eight tissues.
Methods gets one sentence: the largest well-powered tissue was re-scored on the
new context and reproduced within CI.

**The internal control is what makes this tight.** Only fusion was scored, which
is correct — the sequence tower is the published one and is not touched by a
context swap. Running the published sequence arm through the same evaluation
returned **0.5868656838 [0.5658176855, 0.6094125260]**, *bit-identical* to the
published run in value and both CI bounds. Cohort construction also reproduced
exactly: 48,340 shared pairs, 2,241 significant at p < 5e-08, 4,482 matched,
distance-only AUROC 0.5000, median distance 182 bp, 248 blocks. So cohort,
matching, blocks and bootstrap RNG are all identical and the entire −0.0022 is
attributable to the fusion context swap alone.

The two input score sets are locus-aligned: identical `Pair_UID` sets and
`beta_ref_to_alt` / `pvalue` / `abs_distance_bp` agreeing to 0.0, with predicted
Δ M correlating 0.939 — the cohort cannot move, only the predictions.

Output: `results/journal/ablation_breast_epithelium/transfer_discrimination/Lung`.
The published tree was md5-verified unchanged before and after.

**This one number also discharges the Melody head-to-heads.**
`melody_head_to_head` and `melody_st_head_to_head` read fusion from the *same*
`egtex_multitissue_scoring/by_tissue/<tissue>/heldout/fusion/` directories as
`transfer_discrimination`. Same scores, same stability. Do not rerun them on the
strength of the context swap alone.

#### Recompletion audit — where the MCF-10A → breast-epithelium swap stands

Checked 13 Sep 2026 against every directory under `results/journal/`. The queue
is **empty**; all four R6 jobs (45857185–88) finished 0:0 and every one logged
`clobber check clean`.

**Done on the new context** — training seeds 42/43/44, folds 1–3, GENOA and
eGTEx variant scoring + evaluation, context permutation, paired model bootstrap,
rc_uncertainty (+conditional), motif disruption, meQTL class chromatin,
tissue-shared meQTLs, variant-effect synthesis, GWAS regulatory enrichment,
candidates (60) and candidate comparison (62), biological context (22),
manuscript figures (91), and now Lung multitissue scoring + transfer
discrimination.

**Deliberately not rerun, because a context swap cannot move them** —
sequence_baselines, published_baselines(_egtex) (DeepCpG/CpGenie),
baseline_variant_evaluation (its models are `kmer_ridge` and `composition`; its
`fusion_vs_sequence_paired.csv` and `gate_modulation.csv` are *1-byte empty*),
melody / melody_st / melody_st_matched_vs_unmatched / tissue_shared_meqtls_melody
(Melody's own model), motif_disruption_kmer_baseline, target_qc,
training_data_audit. Eight of nine tissues in transfer_discrimination and the
Melody head-to-heads are covered by the Lung spot-check above.

**Everything on the remaining list was run on 13 Sep 2026.** The audit table that
stood here is superseded by §10a below.

#### 10a. Closing out the swap — 13 Sep 2026

Every analysis that could move under the context swap is now either finished or
queued. Two finished in-session, two more were rebuilt, and six went to the GPU
queue as `jobs/r7_ablation/` with dependencies wired so the chain completes
unattended.

**Finished — `paired_model_comparison_{genoa,egtex}` (script 31).**
Both cohorts reproduced exactly (GENOA 42,866 pairs / 4,037 significant at
5e-08; eGTEx 47,991 / 418 at the calibrated 1.483e-5). The same internal control
as the Lung run applies and is again clean: **DeepCpG and CpGenie came back
bit-identical in value and both CI bounds**, because those score files are
untouched by our context. So the whole movement is the fusion column.

| cohort | fusion, MCF-10A | fusion, breast epi | shift |
|---|---|---|---|
| GENOA | 0.5604 [0.5420, 0.5780] | 0.5618 [0.5443, 0.5787] | +0.0014 |
| eGTEx | 0.5868 [0.5380, 0.6514] | 0.5870 [0.5398, 0.6491] | +0.0002 |

Both trivially inside their referents. The conclusion is unchanged and now holds
on the new context: fusion is not distinguishable from DeepCpG or CpGenie on
distance-matched AUROC. (The GENOA tail-enrichment arm still separates from
CpGenie at top 0.5% / top 1% and from DeepCpG at top 5%, as before.)

**Finished — `rc_uncertainty_figure`.** The figure stage needs three conditional
runs and the ablation had only strata 10, so strata 20 and 50 were generated
first (`rc_uncertainty_conditional_s{20,50}`, CPU, seconds each). Note the
published `_s20`/`_s50` directories are **no longer on disk** — only the figure
that consumed them survives — so this was a rebuild from scratch, not a
re-point. Conclusion reproduces: report uncertainty on M-value error;
`boundary_distance` was tracking the compressed range of beta near 0 and 1.

**Finished — the six-job R7 chain, all exit 0:0 by 22:43.** Dependencies were
`afterok`, so a failure would have stopped the chain rather than feeding stale
inputs forward. One did fail and the guard worked: `71`'s first attempt
(45920215) died at 18 s on `FileNotFoundError` because the script default
`data/BreastMammaryTissue.regular.perm.fdr.txt` had moved to
`data/external/egtex_breast/`. Byte-identity to the published input was verified
(sha256 `604c34e4…03743da0`) before resubmitting as 45920935 — a relocation, not
a data change. **Fix the script default**, or this fires again on the next run.

| job | id | waits on |
|---|---|---|
| `70_mqtl_positive` | 45920214 | — |
| `71_mqtl_negative` | 45920215 | 70 |
| `63_known_variant` | 45920218 | — |
| `64_literature` | 45920219 | 63 |
| `22_context` | 45920224 | 70 |
| `91_figures` | 45920225 | 22 **and** 64 |

For 70 and 71 the thing that makes it a new-context run is **`--test-csv`**, not
a weights flag alone: both scripts merge `TABULAR_FEATURES` off that file, so it
must be `data/datafiles_breast_epithelium/test.csv`. The fusion arm uses
`checkpoints_ablation/breast_epithelium/seed{seed}/fusion`, the sequence arm
stays on `checkpoints_journal` — there is no ablation sequence checkpoint and
there should not be.

**What the chain retired — done.** The R6 `biological_context` and
`manuscript_figures` were STAGING runs built on MCF-10A mQTL and literature
inputs. 45920224 and 45920225 rebuilt both on new-context inputs. 91's
`run_summary.json` records nine input paths and **all nine now resolve under
`ablation_breast_epithelium/`**, so no MCF-10A product survives in the figure
chain. The cross-context caveat is retired and the figures are circulatable.
Job results: 63 → 27 model-visible pairs; 70 → 81 loci; 22 → 26,570 held-out
CpGs / 440 candidates / 81 fusion mQTL associations; 64 → 1,318 resolved SNVs →
322 candidates → 321 scored pairs; 71 → same lead input as published; 91 → 6
figures.

Every job carries `check_no_clobber`. A session-wide `find -newermt` over
`results/journal/` confirmed nothing outside the ablation subtree was written.

#### Reproduction

```bash
# build the context (CPU, ~10 min)
python -u data/build_training_data.py \
    --reference-dir data/reference/BreastEpithelium \
    --out-dir data/datafiles_breast_epithelium

# fold splits on the new context (CPU, minutes)
python -u scripts/17_chromosome_splits.py \
    --datafiles data/datafiles_breast_epithelium \
    --out-root data/datafiles_breast_epithelium/splits --folds 4

# retrain epi + gate (GPU, ~14 h each)
mkdir -p logs/ablation
sbatch --job-name=ctx-s43 --export=ALL,SEED=43 scripts/run_context_ablation.sbatch
sbatch --job-name=ctx-f1  --export=ALL,FOLD=1  scripts/run_context_ablation.sbatch
```

Outputs land in `checkpoints_ablation/breast_epithelium/<tag>/{epi,fusion}` and
`results/journal/ablation_breast_epithelium/<tag>/{epi,fusion}`.

#### Traps, recorded because they nearly fired

- **`17_chromosome_splits.py --out-root` does not follow `--datafiles`.** It
  defaults to `data/datafiles/splits` independently. Running it with only
  `--datafiles` pointed at the new context **overwrites the published fold
  splits** that `checkpoints_folds/fold{1,2,3}` were trained against. Always set
  both flags.
- **`build_training_data.py --reference-dir` without `--out-dir`** would
  overwrite `data/datafiles/`. The script now refuses this outright.
- **Fold sequence-tower reuse is only valid if the split is identical.** The
  sbatch verifies train/val/test probe sets against `data/datafiles/splits/foldN`
  before touching the GPU. Without it a mismatch completes normally while testing
  on probes the tower trained on — a result that looks excellent and is worthless.
  `build_folds` uses no RNG and the row set is identical between context builds
  (bigWigs are read after all row filtering), so this passes by construction; it
  is a guard against future edits, not against today.
- **`logs/ablation/` must exist before `sbatch`.** Slurm opens `--output` and
  `--error` before the script body runs, so a missing directory kills the job at
  ~5 s with no log to say why.
- **`--export=ALL,VAR=x`, never plain `VAR=x`.** Without `ALL`, Slurm drops the
  environment and the conda python is not found.
- **Slurm snapshots the batch script at submit time.** Editing after submission
  changes nothing. `scontrol write batch_script <jobid> -` shows what actually
  ran — use it to confirm a parameterized script was the one submitted.
- **Slurm `.err` files are not covered by `*.log`.** A 123 MB tqdm log reached a
  commit and was rejected by GitHub's 100 MB limit. `logs/` is now gitignored.
  Salvage the useful lines with
  `zcat -f <file>.err* | tr '\r' '\n' | grep -E '^20[0-9]{2}-'`.

---

## 2. Melody — positioning, reanalysis, and head-to-head

Jin, Wang, Qiao et al. "Decoding the sequence determinants of locus-specific DNA
methylation across human tissues." *Nat Commun*, accepted 5 Aug 2026, online
17 Aug 2026. doi:10.1038/s41467-026-76744-5

Read this before writing R3, the Discussion, or the cover letter. Melody was
published in our target journal three weeks before our nine-tissue analysis and
overlaps it directly. That is survivable, but only if we choose the comparison
ground deliberately instead of letting a reviewer choose it for us.

---

### 1. What Melody is

- Fully convolutional encoder–decoder (U-Net-like) over **10-kb** input windows.
- Trained on **39 normal human cell types**; three variants:
  - **Melody-ST** — one track, one cell type.
  - **Melody-MT** — multi-track, shared cross-tissue features. Best on meQTLs.
  - **Melody-G** — adds scRNA-seq foundation-model embeddings via FiLM
    conditioning to predict methylation in **unseen** cell types.
- Multi-task head: per-CpG methylation + 100-bp regional average + CpG density.
- Baselines beaten: INTERACT, CpGenie, iDNA-ABT.
- Headline methylation numbers: Melody-MT Spearman 0.723 (sampling test) and
  0.645 (test chromosome) vs 0.590 / 0.584 for the best competitor;
  AUC 0.921, AP 0.975.

### 2. The direct overlap — Figure 3H

This is the figure to worry about. Melody performs a **cross-cell-type meQTL
validation** on GTEx data across **Breast, Colon, Kidney, Lung, Muscle, Ovary,
Whole Blood** — seven of our nine tissues, from the same source. Their finding:
"related tracks typically achieve the best or second-best performance,"
i.e. the tissue-matched track predicts that tissue's meQTLs best.

So the sentence "we test whether a model trained on one tissue predicts meQTLs
in another" is **not novel as of 17 Aug 2026**. We must not write R3 as though
it were.

### 3. Where we are genuinely not comparable — and stronger

Melody's meQTL evaluation is **Pearson correlation between predicted and
observed effect, computed among meQTLs only** (Fig 3C–H; e.g. GTEx Whole Blood
r = 0.4236, n = 1334; MDSs r = 0.6253, n = 524).

That design has three gaps we already fill:

1. **No negative controls.** Melody never asks whether a variant is an meQTL at
   all — only how well the effect size correlates among variants already known
   to be meQTLs. Our primary metric is discrimination of real meQTLs from
   *tested-but-null* pairs.
2. **No distance control.** Their own Fig 3E shows performance falls with
   variant–CpG distance for every model, which is exactly the confound: distance
   alone reaches AUROC 0.60–0.69 on our cohorts. We match on distance until
   distance-only AUROC is **exactly 0.5000**, then measure what is left. Nothing
   in Melody establishes that its meQTL signal is not distance.
3. **No CpG-altering exclusion.** We drop variants that create or destroy the
   target CpG (~37% of pairs, and enriched for true meQTLs), so our numbers
   exclude the mechanically trivial cases.

**Do not compete on effect-size correlation.** Melody reports Pearson ~0.3–0.5
on GTEx tissues; our signed Spearman among significant pairs is ~0.18 (Lung).
Their 10-kb window and 39-tissue training will win that metric.

> **Superseded in part, 5 Sep 2026.** The original second half of this paragraph
> read "compete on discrimination under a distance-matched null, where they have
> no result at all." The eight-tissue head-to-head (§3c) shows we do not *win*
> there either — we tie. The distance-matched null is still where our
> contribution lives, but the claim it supports is parity-under-asymmetry, not
> superiority. Do not write the superiority sentence.

### 3b. Reanalysis of Melody's own Source Data — the strongest point we have

Computed 5 Sep 2026 from `supplemental_data/supplemental_data.xlsx` (Figure 3H
sheet) and the benchmark CSVs shipped in `meqtl/dataset/processed/GTEX/`. No
compute, no checkpoint, entirely their published numbers.

**Finding 1 — tissue identity of the model track is second-order.**
Two-way decomposition of their 8x8 cross-track matrix:

    variance explained by TARGET dataset : 85.0%
    variance explained by SOURCE track   :  1.4%
    residual (interaction, incl. matching): 13.5%

The matched track's advantage over other tracks on the same target averages
**+0.027 Pearson r**. Their claim that related tracks are "best or second-best"
holds column-wise in 5 of 8 targets, so it is not wrong -- it is just small.
State it that way; do not claim their result fails.

**Finding 2 — what the 85% actually is: distance.**
Median variant-CpG distance per benchmark, against their reported Pearson r:

    Ovary       120 bp   0.407        Kidney      473 bp   0.350
    Prostate    127 bp   0.453        Breast      478 bp   0.371
    Lung        129 bp   0.427        WholeBlood  484 bp   0.324
    Colon       153 bp   0.417        Muscle      684 bp   0.278

    Spearman -0.833 (p = 0.010)      Pearson -0.946 (p = 0.0004)
    cohort size vs r: Pearson +0.323 (p = 0.44)  -- NOT power

Roughly 89% of the variance in Melody's cross-tissue meQTL performance is
explained by how proximal each benchmark's pairs happen to be. Tissue explains
1.4%. Sample size explains nothing.

**Why this matters for our paper.** It is the strongest available motivation for
distance-matched evaluation, and it comes from the leading model rather than
from us. The argument writes itself: performance differences that look like
tissue biology are largely differences in benchmark composition, which is
exactly what a distance-matched null removes.

**Caveats to state honestly, because this is a claim about someone else's work.**
- n = 8 datasets. The correlation is strong but the sample is small, and we
  chose the predictor after seeing the pattern. Present it as a reanalysis that
  motivates our design, not as a definitive decomposition of their method.
- Melody's authors did NOT hide this: their Fig 3E explicitly shows performance
  declining with variant-CpG distance for every model tested. The criticism is
  that no one in this literature CONTROLS for it, not that they concealed it.
  Say so; it is both fairer and more credible.
- A correlation across datasets is not the same as distance driving predictions
  within a dataset. Our own within-cohort result -- distance alone reaching
  AUROC 0.60-0.69, pinned to 0.5000 by matching -- is what closes that gap. The
  two together are the argument; either alone is weaker.

**Consequence for priorities.** This reanalysis makes the head-to-head
confirmatory rather than load-bearing. Still worth doing -- "and when distance
is controlled directly, here is what happens" beats a correlation over 8 points
-- but the paper no longer depends on it.

### 3c. Head-to-head on our distance-matched cohorts — eight tissues

Run 5 Sep 2026, `scripts/31_transfer_discrimination.py`, output under
`results/journal/melody_head_to_head/<Tissue>`. Both models scored on **byte-identical
rows** (SilentMethyl's per-tissue `pair_scores.csv` is the template; only
`Predicted_Delta_M` / `Absolute_Delta_M` are replaced — `scripts/34_melody_by_tissue.py`).
Same matched negatives, same 1-Mb blocks resampled together. Testis excluded: no
testis cell type exists among Melody's 39 tracks.

**The comparison is not symmetric, and the asymmetry is the point.** Melody is
given **tissue-matched Loyfer tracks** in every tissue. SilentMethyl is
**zero-shot**: breast-trained, breast epigenomic context, no target-tissue
information of any kind. Melody has target-tissue data; we have none. Every number
below has to be read through that.

Paired difference, **fusion minus melody** (negative = Melody better):

    tissue                 n_sig   AUROC (distance-matched)          signed rho (significant)
    Lung                    2241   -0.0140 [-0.0343,+0.0052] ns      -0.0606 [-0.1094,-0.0178] MELODY
    ColonTransverse         2118   -0.0080 [-0.0342,+0.0147] ns      -0.0947 [-0.1528,-0.0406] MELODY
    Ovary                   1812   -0.0087 [-0.0368,+0.0144] ns      -0.0019 [-0.0673,+0.0570] ns
    Prostate                 825   -0.0314 [-0.0679,+0.0017] ns      -0.1268 [-0.2244,-0.0458] MELODY
    WholeBlood               183   -0.1085 [-0.1649,-0.0511] MELODY  -0.0033 [-0.1850,+0.1447] ns
    KidneyCortex             157   -0.0301 [-0.0951,+0.0367] ns      +0.2030 [+0.0166,+0.3778] OURS
    BreastMammaryTissue      154   -0.0123 [-0.1016,+0.0663] ns      -0.0592 [-0.2439,+0.0955] ns
    MuscleSkeletal           128   -0.0509 [-0.1432,+0.0369] ns      -0.1356 [-0.3829,+0.1030] ns

**Finding 1 — distance-matched AUROC is a tie in 7 of 8, with a consistent small
tilt toward Melody.** Only Whole Blood separates. But all eight point estimates
are negative (sign test p = 0.0078; median gap -0.013 to -0.030). The tissues are
not independent — overlapping CpGs and shared genome structure — so that p is
anti-conservative and should be reported as a *direction*, not a test. The honest
statement: **a zero-shot breast model is statistically indistinguishable from a
tissue-matched 39-cell-type model on distance-matched meQTL discrimination in
seven of eight tissues, while sitting consistently ~0.02 AUROC behind.**

**Finding 2 — effect-size correlation goes to Melody, as predicted.** Three
tissues favour Melody at interval level (Lung, Colon, Prostate), four tie, and
Kidney Cortex favours us (+0.2030 [+0.0166,+0.3778], n = 157 — small cohort, treat
as an observation not a claim). 7 of 8 signs favour Melody. This is their home
metric; the result is expected and should be stated as expected.

**Finding 3 — the Lung tail result did not replicate. This kills the "our tail
beats theirs" framing.** Counting tissue x threshold cells where a model's tail
enrichment excludes zero against the *distance-only* baseline:

    SilentMethyl (fusion):  7 cells  (Lung 0.1/0.5/1%, Colon 1%, Prostate 0.1/0.5/1%)
                            plus one cell where it is significantly WORSE than
                            distance (MuscleSkeletal top 5%, -0.0030 [-0.0067,-0.0002])
    Melody:                11 cells  (Lung 0.5/1%, Breast 0.5%, Colon 1%,
                                      Muscle 0.5%, Prostate 0.1/0.5/1%,
                                      WholeBlood 0.5/1/5%)

Lung was simply the first tissue run and happened to be the best case for us. The
paired fusion-minus-melody tail differences are indistinguishable in most cells and
favour Melody where they separate (Breast top 1% -0.0146, Kidney top 1% -0.0106,
WholeBlood 0.5/1/5%, Muscle top 5%). **Do not write a tail-superiority claim.**

**Finding 4 — Whole Blood is the one clear loss, and it is interpretable.** Melody
averages *five* Loyfer blood tracks there (B, T-CD3, NK, monocytes, granulocytes),
the deepest and most replicated part of the atlas, and it is the tissue where
tissue-matching plausibly buys the most. It is also the tissue where our breast
context is least applicable. Worth stating rather than hiding: it bounds how far
the parity claim generalises.

**What this does and does not license.**
- Licensed: "tissue-matching buys less than expected" — the head-to-head magnitude
  (~0.02 AUROC) is the same order as the +0.027 matched-track advantage recovered
  from their own Source Data in §3b. Two independent routes to the same number is
  the strongest thing in this document.
- Licensed: our evaluation design is the contribution. Both models were run through
  it; neither had been evaluated this way before.
- **Not licensed:** any claim that SilentMethyl outperforms Melody on meQTLs.
- **Not licensed:** "we trained a better single-nucleotide methylation predictor."
  Nothing here tests methylation *levels* head-to-head. That needs the bidirectional
  levels comparison, not this run.

**Caveat that cuts against us, not for us.** Melody contributes one released
checkpoint; our reference is three seeds averaged. Seed averaging *reduces* noise
in our scores, so it should have helped us. The consistent tilt toward Melody is
therefore not explained by the seed asymmetry.

### 3d. The Melody-ST arm -- single-tissue transfer, and what tissue matching buys

Run 5 Sep 2026. `scripts/33_melody_scoring.py` with the released single-track
checkpoints, `scripts/35_melody_st_by_tissue.py` to build the cohorts,
`scripts/31_transfer_discrimination.py` for the comparisons. Output under
`results/journal/melody_st_head_to_head/<Tissue>` and
`results/journal/melody_st_matched_vs_unmatched/Lung`.

**Why this arm exists.** Melody-MT is ONE model trained on all 39 cell types with
39 output channels, and their Methods say so directly: "Melody-MT serves as the
default model for tasks that involve multiple tissues, including meQTLs effect
prediction and cross-cell-type transfer." Their Fig 3H cross-track validation is
therefore output-head selection inside a model that saw every tissue in training.
Melody-ST is the genuinely single-tissue variant -- "trained on one cell type at a
time using a single bigWig track as supervision" -- and the paper never applies one
across tissues. That experiment is the architecture-matched control for R3, and it
did not exist until now.

#### Finding 1 -- single-tissue transfer is NOT specific to SilentMethyl

Melody-ST-Breast, applied to all nine tissues, transfers. Paired difference,
fusion minus ST-Breast, distance-matched AUROC (negative favours Melody-ST):

    Breast       -0.0378 [-0.1058, +0.0258]  ns
    Colon        -0.0228 [-0.0463, -0.0006]  ST-Breast better
    Kidney       +0.0171 [-0.0600, +0.0875]  ns
    Lung         -0.0220 [-0.0419, -0.0043]  ST-Breast better
    Muscle       -0.0163 [-0.1278, +0.0735]  ns
    Ovary        -0.0209 [-0.0406, -0.0028]  ST-Breast better
    Prostate     -0.0142 [-0.0542, +0.0258]  ns
    Testis       +0.0908 [-0.0205, +0.1921]  ns
    WholeBlood   -0.0633 [-0.1212, -0.0115]  ST-Breast better

Seven of nine favour ST-Breast, four at interval level (sign test p ~ 0.18, so
the direction is weaker than the MT comparison's 8/8). **SilentMethyl is not the
best-transferring single-tissue model here, and we must not say it is.**

This is the right result and it makes R3 stronger. Two architectures sharing
nothing but the task -- DNABERT-2 over 1 kb with a gated epigenomic tower, and a
10 kb U-Net trained on WGBS -- both reach 0.59-0.65 distance-matched AUROC in
tissues neither has seen. The finding is about **meQTL prediction**, not about
our model. Rewrite R3 accordingly: "single-tissue methylation models transfer,
and the training tissue matters far less than assumed."

#### Finding 2 -- where fusion does win, and it is a coherent pattern

    Testis   ST-Breast AUROC 0.4737 [0.3594, 0.5886] -- BELOW CHANCE; rho -0.0740
             fusion 0.5645, rho +0.1290
             paired rho +0.2030 [+0.0251, +0.4153] DIFFERENT
             direction agreement +0.1170 [+0.0332, +0.2091] DIFFERENT
    Kidney   paired rho +0.2364 [+0.1228, +0.3795] DIFFERENT
             direction agreement +0.0892 [+0.0082, +0.1816] DIFFERENT
    Colon    tail top 0.1% +0.2500 [+0.0879, +0.4087] DIFFERENT
             tail top 0.5% +0.0962 [+0.0353, +0.1667] DIFFERENT

Fusion holds up in the low-powered and hardest tissues where the U-Net degrades,
and dominates the extreme tail in colon. That is a narrower claim than "we
transfer better" and it is true. Testis is the strongest single case: the
tissue with the fewest significant pairs (94) is where ST-Breast falls below
chance and fusion does not.

#### Finding 3 -- what tissue matching buys, measured directly

ST-Lung against ST-Breast on identical Lung rows. Same architecture, same
training-data scale, same scoring code, same matched negatives, same resampled
blocks. **Only the training tissue differs.** This comparison exists in neither
paper.

    AUROC, distance-matched   -0.0054 [-0.0241, +0.0103]  not distinguishable
    signed rho (significant)  +0.0297 [-0.0198, +0.0887]  not distinguishable
    direction agreement       +0.0170 [-0.0068, +0.0420]  not distinguishable
    tail top 0.1%             +0.1458 [-0.1022, +0.3984]  not distinguishable
    tail top 0.5%             +0.1157 [+0.0288, +0.2126]  DIFFERENT
    tail top 1%               +0.0290 [-0.0218, +0.0932]  not distinguishable
    tail top 5%               -0.0074 [-0.0214, +0.0108]  not distinguishable

    marginals: ST-Lung 0.6045 [0.5833, 0.6233]   ST-Breast 0.6099 [0.5870, 0.6321]

**Tissue matching buys nothing detectable on discrimination.** The interval is
tight -- [-0.024, +0.010] -- so it also bounds the effect: whatever tissue
matching is worth, it is under about 0.024 AUROC. The point estimate slightly
favours the MISMATCHED model, which is not a claim to make, but it does rule out
matching being large.

It buys something real in one place: the **top 0.5% tail** (+0.1157
[+0.0288, +0.2126]). Matching helps you pick the very best candidates and does
not help you separate meQTLs from nulls in general. State both halves.

#### The convergence -- this is the paper's headline

Four independent routes to the same small number:

    two-way decomposition of Melody's published 8x8 matrix   source track 1.4% of variance
    matched-track advantage in that same matrix              +0.027 Pearson r
    SilentMethyl vs Melody-MT head-to-head, eight tissues    ~0.02 AUROC median
    ST-Lung vs ST-Breast, same architecture, same rows       0.005, bounded under 0.024

Their published data, our head-to-head, and their own architecture run both ways
all agree that tissue matching is worth roughly two AUROC points or less. That
is the claim, and no part of it requires us to have the better model.

### 4. The gift — Ovary

Melody, independently: *"Ovary data perform poorly, likely due to either (i)
lower data quality in ovary methylation tracks or (ii) ovary-specific motifs
being underrepresented in available meQTL datasets."*

Our nine-tissue result: **Ovary is the only tissue with a negative tail
difference** (−0.0208 [−0.1633, +0.1857]), despite being third by cohort size
(1,812 significant pairs) and having a perfectly healthy AUROC (0.5864
[0.5653, 0.6100]).

Confirmed directly in the head-to-head (§3c): run on *our* ovary cohort,
**Melody also fails to clear the distance baseline at every threshold**
(top 0.1% +0.0417 [-0.0833,+0.2283]; 0.5% +0.0124; 1% +0.0291; 5% -0.0083 — all
spanning zero), and the paired fusion-minus-melody differences are indistinguishable
throughout. Ovary is the tissue where the two models agree most completely, and
what they agree on is that neither adds anything to distance.

The ST arm adds a third: Melody-ST-Breast on our ovary cohort also fails to
clear the distance baseline at every threshold (top 0.1% +0.0000, 0.5% +0.0207,
1% +0.0353, 5% -0.0021 -- all spanning zero).

Two different architectures, different training data, different metrics, same
anomaly. That is strong evidence the ovary result is a property of the eGTEx
ovary data rather than of our model — and it lets us cite Melody as
corroboration rather than only as competition. Use this.

It also reinforces the sex argument: ovary underperforms in both papers while
prostate (male-only) is among our best, so donor sex is not the explanation in
either.

### 5. Melody-G and the "unseen tissue" claim

Melody-G predicts methylation in five held-out cell types, achieving AUC 0.663
(G1) and 0.697 (G2) against a 0.633 mean-profile baseline — improvements of
4.7% and 10.1%.

The distinction to draw, precisely: **Melody-G requires scRNA-seq from the
target cell type.** It is conditioned transfer. Our nine-tissue result is
**zero-shot** — no target-tissue data of any kind, no tissue label, no
conditioning. Those answer different questions, and ours is the one relevant to
a tissue with no molecular data available.

Note also that Melody-G's gains are on *methylation prediction*, not meQTL
effect prediction, in unseen types. Nobody has yet shown zero-shot meQTL
discrimination across tissues under a distance-matched null. That is our claim.

### 6. What R3 should say

Frame it as: a breast-trained model, given no information about the target
tissue, discriminates real meQTLs from distance-matched non-meQTLs in 7 of 9
tissues, and does so no better on breast (AUROC 0.6141) than on lung (0.5880),
colon (0.6051), or kidney (0.6062).

Then, per §3c, the head-to-head: a tissue-matched Melody is indistinguishable from
this zero-shot model in 7 of 8 tissues. R3 must not claim we beat Melody.

The "no better on its own tissue" observation is the one that is genuinely ours
and genuinely surprising — and it sits in productive tension with Melody's
Fig 3H, where tissue-matched tracks *do* win. Worth stating that tension
explicitly rather than hiding it: their models are trained per tissue and ours
is not, so the comparison localises where tissue-specificity actually lives.

### 7. Availability to check

- Zenodo record **21386471** — Melody checkpoints. If the weights and the meQTL
  benchmark are public, a head-to-head on our distance-matched cohorts is
  inference-only and cheap. This would be the single strongest addition.
- Melody web server: https://inner.wei-group.net/Melody/
- Their meQTL benchmark sources: Ólafur et al., GTEx, EPIGEN.

### 8. Shared test split

Melody uses chr10 for validation and **chr8 + chr9 for test** — identical to
ours. Say so in Methods. It makes any future head-to-head directly comparable
and pre-empts the "different splits" objection.

---

## 3. What is left

### Writing
1. `references.bib` needs a real entry for `jin2026melody`; only the placeholder
   bibitem exists in `main_revised.tex`.
2. Grep `main.tex` and `main_revised.tex` for any sentence implying epigenomic
   context improves VARIANT-EFFECT prediction. R2 forbids it.
3. Delete the superseded `70_mqtl_positive_control` (81 pairs) and
   `71_mqtl_matched_negative` (35 pairs) together with Supplementary S2/S3 in ONE
   coordinated edit — the text still cites them.
4. Sync figure PNGs from the cluster before producing a circulating PDF;
   `results/journal/manuscript_figures/` holds only JSON locally, so local builds
   render `\missingfigure` placeholders.
5. Read both `\draftmode` branches with the switch flipped. That is how a false
   "repeated blocked splits" claim survived undetected in an earlier draft.
6. Split-status label on the case-study figure — the text states it, the figure
   does not.
7. Mentor email. Every number in it is now checkable.
8. **Refresh every single-tissue number from the §1.10 rerun.** The six
   retrains are done and the downstream pipeline is running; results are under
   `results/journal/ablation_breast_epithelium/`, alongside rather than over the
   MCF-10A results. The fold table, the seed table and every variant analysis
   need their values swapped. The mentor email quotes pre-swap numbers — either
   send it before the reruns land or update it, but do not let the two drift.
9. **Fix the `caveats.tissue` string in `21_variant_evaluation.py`** — it still
   says MCF-10A and is written into every `run_summary.json`.
10. ~~**Fix `64_literature_variant_screen.py` to forward `--weights-template`**
   before running the candidate chain. As written it silently scores the old
   MCF-10A checkpoints through `63`, and the output looks correct.~~
   **Verified fixed 14 Sep 2026.** `64` forwards both `--weights-template` and
   `--split-template` to `63` (`scripts/64_literature_variant_screen.py:745-746`),
   the `--weights-template` help text spells out the failure mode, and
   `jobs/r7_ablation/64_literature.sbatch:21-22` sets both explicitly. The
   defaults still point at `checkpoints_journal` / `data/datafiles`, so both
   flags remain mandatory for an ablation run — but the defect itself is gone.
   *Do not patch this again:* this entry stayed open after the 13 Sep fix and
   nearly generated a second patch for a bug that no longer existed.
11. **Add the three-way variant-discrimination table (§1.10) to the Results.**
   Sequence-only 0.5606, fusion/MCF-10A 0.5586, fusion/breast-epi 0.5608 — all
   within 0.0022. This is the strongest form of R2 and currently exists nowhere
   in the manuscript.

### The one open scientific decision
**eQTL / eQTM colocalisation.** The only remaining substantive analysis and the
only route to "new biological findings" for the Nature Communications bar.
Public GTEx, donors overlap eGTEx, inference plus joins. Take the prior
seriously: four disease-variant tests already returned null (ClinVar, breast
GWAS, all-trait GWAS in both cohorts), and the recorded conclusion is that "the
model ranks methylation change, not clinical variant relevance". Expression sits
closer to methylation than disease does, so this is a better bet than those
were — but it is a bet. Decide once, deliberately.

### Deferred, with reasons
- **Methylation-LEVEL head-to-head against Melody.** Never tested; the entire
  Melody comparison is variant effects. Required before any claim about having
  the better methylation predictor.
- **Loyfer WGBS atlas (GSE186458) instead of TCGA.** Would remove the
  field-cancerisation limitation. Large enough (205 samples, 39 cell types,
  28.2M CpGs, hg38, public) but a substantial re-plumbing.
- Pooled TCGA baseline, BEND multi-task validation (manifest entry exists, data
  MISSING — the Nature Machine Intelligence criterion, out of scope for this
  paper).
- **Allele-specific methylation** (script slot 53) is NO LONGER deferred,
  15 Sep 2026 — see section 6, `E (REOPENED)`.

### Closed — do not reopen
- Multi-tissue joint training: declined on architectural grounds (R2), not
  resources.
- Within-cohort ancestry stratification: impossible at 4 African-ancestry donors.
  GENOA is the ancestry evidence, with its confound stated.

---

## 4. Operational notes

Hard-won details that cost time to discover. Kept because they will cost the
same time again if lost.

### Cluster
- Repo root is `/ocean/projects/med250012p/szhang37/SilentMethyl`. Not `~`.
- `sbatch --export` splits on commas — a 15-track list silently truncated to one.
  Inline lists in the script instead.
- `set -euo pipefail` plus `source ~/.bashrc` kills jobs at 3 s with
  `/etc/bashrc: BASHRCSOURCED: unbound variable`. Use the interpreter path
  directly: `PY=/jet/home/szhang37/.conda/envs/<env>/bin/python`.
- Python logging goes to **stderr**, so training progress is in `.err`, not
  `.out`. Progress bars need `tr '\r' '\n'` to read.
- **Node v005 had CUDA error 803** (driver/kernel mismatch) and silently ran
  GPU jobs on CPU at 13.3 s/it against 3.66 it/s — a 49× slowdown with no error.
  Add a `torch.cuda.is_available()` guard to every GPU sbatch so this fails in
  seconds rather than hours.
- **RM-shared caps memory at 2000 MB PER CORE.** `--mem=48G` with 8 cores is
  rejected as "Allocation requested mem-per-core higher than maximum of
  2000M/core", which Slurm then reports as `Access/permission denied` — a
  misleading message that looks like an account problem. Also pass `--gpus=0`
  or a GPU directive in the script follows the job onto a GPU-less partition.
- **Slurm snapshots the batch script at submission.** Editing the file after
  `sbatch` changes nothing for queued jobs. `scontrol write batch_script <id> -`
  prints what will actually run. Extending `TimeLimit` on a running job needs
  admin — `scontrol update` returns "Access/permission denied" for users.
- **`--array` jobs raced on `dnabert2_local/config.json`, 12 Sep 2026.**
  `patch_and_load_dnabert` in `training_common.py` rewrote that file on every
  startup with `open(path, "w")`, which truncates to zero BEFORE writing. With
  an `--array=0-5` scoring job, six processes load DNABERT-2 at the same moment
  against one shared directory; two died with

      json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

  while four identical tasks succeeded. It looks like a node fault or a corrupt
  checkpoint and is neither — it is intermittent and blames whatever node it
  lands on. Reproduced deterministically: 16 threads × 40 rounds gives 15
  failures with the old code, 0 with the fix. **Fixed** by writing only when the
  content would actually change (steady state is now read-only, so the race
  cannot occur) and by `os.replace()` for the write itself, which is atomic.
  `run_genoa_scoring.sh` and `run_egtex_scoring.sh` are both `--array=0-5` and
  carried the same exposure — any historical task that "failed on a bad node"
  was probably this.
- V100 needs torch 2.6.0+cu124; torch 2.14.0+cu130 ships no sm_70 kernels.

### Training pipeline
- Towers train separately, are frozen (and held in `.eval()`), then the gate
  trains on top. `12_train_fusion.py` takes the `.pth` **file**, not the save
  directory — it sha256's the path. Passing the directory cost three folds 32 h
  each before failing with `IsADirectoryError`.
- All three stages resume from `latest_checkpoint.pt` and save `best_weights.pth`
  atomically on every improvement, so a walltime kill loses only the epochs since
  the last improvement.
- Validation peaks at 40–60% of the schedule in every fold. Ten epochs is more
  than this needs.
- The published split is **chr10 + chr11** validation, chr8 + chr9 test — not
  chr10 alone, as an earlier version of `17_chromosome_splits.py` assumed.

### Melody reproduction
- `predict.py` shows `model(x)[0]` returns **raw logits**; `sigmoid_first()` must
  be applied before differencing. Differencing logits cost 0.15 Pearson r.
- The window is centred on the **midpoint** of the CpG and the SNP, not the CpG.
  `mut_idx = SNP_start - fetch_start - 1`. The reference base is overwritten, not
  checked — we reproduce that but count mismatches and abort past 5%.
- `--margin` must be ≥ 1 or the summation window is empty
  (their CSVs have `CpG_start == CpG_end`). Margin 1 reproduces their published
  0.407387 at r = 0.4060.
- Melody-ST checkpoints have ONE output channel: `--n-track 1 --track-index 0`.
  Resolving track names against the 39-name list would read a channel the model
  does not have.
- selene-sdk was stubbed rather than installed (2.5 GB of duplicate CUDA libs
  blew the disk quota).

### Data
- eGTEx raw regular files are **streamed** through the ±600 bp filter and never
  stored — `data/fetch_egtex_regular_slices.sh`. The 45 GB breast raw file was
  the only stored copy, from before that approach existed; it was removed
  8 Sep 2026 after its provenance (URL + sha256) was recorded in
  `external_manifest.json`.
- GENOA raw (5.3 GB) was likewise removed after harmonisation.
- Keep the `*.regular.perm.fdr.txt` files: they carry the permutation FDR that
  defines the significance thresholds.

---

## 5. Talk structure

Retained from the presentation outlines. The 5–7 minute version:

1. **Title + problem** (30 s)
2. **What we built** (45 s) — one slide, do not split
3. **Primary result** (60 s) — the baseline table
4. **It generalises: two independent cohorts** (75 s) ★ biggest slide
5. **Why you should believe it: the controls** (60 s)
6. **What the model can and cannot do** (45 s) ★ the distinctive slide
7. **Individual variants** (30 s)
8. **Limitations and next** (30 s)

The longer version adds data construction, architecture, harmonisation, and the
fusion≡sequence ablation as separate slides. Slide 6 is the one that
distinguishes this work: most talks omit what their model cannot do, and the
allele-invariance argument is more memorable than any performance number.

Now that the folds are done, slide 3 should carry the four-fold table rather
than a single split, and the 9/1 limitation "cross-ancestry and cross-tissue are
confounded" should be narrowed — the nine-tissue result measures tissue on its
own, so GENOA's marginal contribution is ancestry plus platform.

---

## 6. R8 journal chain — seven analyses (14–15 Sep 2026)

Five analyses aimed at acceptance, budgeted under 40 of the 125 GPU hours.
Single seed (42) everywhere **by design**: these characterise one trained
model's internals and external validity, not run-to-run spread. Do not add
multi-seed replication to this chain without a reason that is about the science
rather than about habit.

Order: A and C first (A is the central claim, C is free). B and D only if those
land. E only if the catalogue clears ~500 usable pairs.

Guards: `jobs/r8_journal/_common.sh`, same contract as the R6/R7 chain — clock
stamp before, `find -newermt` clobber check after, `require_cuda` to catch the
v005-style silent CPU fallback. Outputs confined to
`results/journal/ablation_breast_epithelium/` and `results/journal/joint/`.

### A. Gate decomposition — DONE 14 Sep 2026. The intended claim is REFUTED.

`scripts/54_gate_decomposition.py`, `jobs/r8_journal/54_gate_instrument.sbatch`
(array 0=genoa, 1=egtex), submitted as 45992001 at 22:19 UTC.

Smoke-tested first (`54_gate_instrument_smoke.sbatch`, 400 pairs, scratch
output): the hand-rebuilt forward pass reproduces `scripts/20` to
**max |ΔM| = 5.2e-6** against a 1e-4 threshold. Worth keeping that habit — the
reproduction check in the full run only fires at the very *end*, so without a
smoke pass a rebuild bug costs the whole 8 h walltime before it surfaces.

Throughput measured on the smoke run: **0.059 s/pair** (400 pairs in 23.5 s of
scoring, V100-32). Projected ~65 min genoa, ~75 min egtex, running in parallel
→ **~1.3 h wall, ~2.4 GPU h**, comfortably inside the 3 h estimate and the 8 h
walltime.

**Two things the analysis plan got wrong, both structural, both worth recording
before the numbers land:**

1. *The specified regression cannot reach R²≈0.99, for reasons that have nothing
   to do with gating.* Regressing Δ_fusion on Δ_sequence × gate_dna compares two
   **separately trained models** — the sequence-only arm has its own encoder and
   its own head — so it measures cross-model agreement and is capped by that. The
   CPU fallback already shows the ceiling: R² = 0.794 ungated, and *adding* the
   gate makes it slightly **worse** (0.757 with gate_dna(REF), 0.781 with the
   REF/ALT mean). On top of that the regression head is nonlinear
   (Linear→GELU→Linear), so an exact linear rescaling cannot hold by
   construction. The instrumented run therefore also computes the *within-model*
   decomposition, which is exact by construction:
   `Δ_total = Δ_DNA_channel + Δ_gate_channel`, holding the gate frozen at REF.
   That is the version that can support the intended claim.

2. *"The epi branch's contribution is bit-identical across alleles" will come
   back ≈0%, and that is correct behaviour, not a bug.* `epi` itself **is**
   bit-identical (same tabular input both alleles). Its **contribution**
   `g_epi × epi` is not, because the gate network reads `[dna, epi]` and `dna`
   moves with the allele. That residual is the *sole* channel by which context
   can touch a variant effect — small, but **not zero**. The paper must not
   claim zero. The script separates the two quantities deliberately
   (`Epi_Vector_Bit_Identical_*` vs `Epi_Contribution_Bit_Identical_*`).

   **Scale reconciliation, 14 Sep 2026 — do not quote 0.055 as the channel
   size.** A first reading paired "0.055" against "gate_epi ≈ 0.017–0.019" and
   concluded the gate swings 3× its own magnitude between alleles, which would
   have contradicted R2. It does not. Two separate errors produced that:

   - **max vs mean.** 0.055 is the single most extreme pair out of 76,893. The
     *mean* |Δg_epi| is 3.58e-4 and the median 1.06e-4. Only 104 pairs (0.14%)
     exceed 0.01 and exactly one exceeds 0.05.
   - **two different models.** gate_epi ≈ 0.017–0.019 is the **all4 joint**
     model (Task C's model: gate_dna_avg 0.0188, gate_epi_avg 0.0196). The
     model Task A scores is **breast-epithelium fusion**, whose gate_epi is
     0.0292 on validation loci and 0.0280 on the variant pairs.

   Like for like, on the breast-epithelium model: mean |Δg_epi| / gate_epi =
   **1.28%**, median 0.46%, p99 ~14%, and the 169.8% ratio occurs at one locus
   where gate_epi(REF) is itself near zero. A ~1% mean perturbation is fully
   consistent with R2's paired fusion-minus-sequence spanning zero in all eight
   tests. There is no bug here — but quote the mean or the median, never the max,
   and never across models.

Allele invariance of the DNA gate itself, from the fallback (fwd/RC-averaged,
both cohorts): |gate_dna(ALT) − gate_dna(REF)| median ~1.2e-4, p95 ~1.5e-3,
against a gate_dna level of ~0.024. Relative shift ~0.5%. The instrumented run
reports this per strand rather than averaged.

#### Result (45992001, both tasks COMPLETED 0:0, 23:24 and 23:33 UTC)

Reproduction against `scripts/20`: max |ΔM| **3.8e-6** on all 66,495 / 76,893
pairs. Channel identity `Δ_total = Δ_DNA + Δ_gate` holds to **2e-7**, so the
decomposition is exact, not fitted. Clobber checks clean. ~2.4 GPU h.

**What is confirmed.** The epi encoder output is **bit-identical across alleles**
— `Epi_Vector_Bit_Identical` True on both strands in both cohorts. The context
embedding really is allele-invariant.

**What is refuted.** The intended claim was that gated fusion merely rescales the
sequence variant effect by a near-allele-invariant scalar and "does not add an
independent context-derived effect". The gate channel is not negligible:

| | genoa | eGTEx |
|---|---|---|
| variance share, DNA channel | 0.557 | 0.555 |
| **variance share, gate channel** | **0.187** | **0.195** |
| corr(gate channel, total) | +0.729 | +0.725 |
| median \|gate channel\| / \|total\| | 0.349 | 0.346 |
| SD of gate channel (M) | 0.0404 | 0.0441 |

The shares do not sum to 1 because the two channels covary (r = +0.38–0.40);
the residual 0.25 is 2·Cov/Var.

The gated epi contribution is bit-identical in **0.005–0.009%** of pairs — i.e.
essentially never, exactly as predicted. Splitting the fused-vector perturbation
by sub-channel (from the saved L2 diagnostics, ‖e‖ ≈ 20.1):

    ‖Δ(g_epi · epi)‖   CONTEXT   median 0.0022, mean 0.0080, p95 0.0328
    g_dna(REF)·‖Δdna‖  SEQUENCE  median 0.0058, mean 0.0127, p95 0.0467
    ratio context/sequence       median 0.40, mean 0.59, p95 1.69

So the context-mediated perturbation is a **median 40%** of the sequence one, and
in ~5% of pairs it is larger. "Almost invariant scalar" does **not** imply
"negligible effect": a 0.5% gate shift multiplies a vector of norm ~20, and that
product is not small.

**Why this still reconciles with R2 — and sharpens it.** The gate channel carries
variance but almost no *information*. Signed Spearman against the measured
effect:

| | total | DNA channel only | gate channel only |
|---|---|---|---|
| genoa (`beta_genoa_ref_to_alt`) | +0.0707 | **+0.0699** | +0.0560 |
| eGTEx (`beta_ref_to_alt`) | +0.0537 | **+0.0536** | +0.0408 |

The DNA channel alone recovers ~99% of the full model's correlation. The gate
channel's apparent signal is inherited through its +0.38–0.40 correlation with
the DNA channel, not independent. That is precisely why R2's paired
fusion-minus-sequence spans zero: the context channel injects magnitude into the
variant effect without adding discriminative signal.

**The defensible claim, replacing the old one.** The context embedding is
allele-invariant, so context has no *independent* channel — the trigger is always
sequence, and with no sequence change there is no gate change and no variant
effect. But context *content* does enter the variant effect, through the gate's
dependence on the sequence embedding, at a median 40% of the sequence channel's
magnitude and ~19% of its variance. It modulates the size of a sequence-derived
effect while contributing essentially no independent information about it.

**Open, and cheap (~2.4 GPU h).** The gate channel mixes two sub-effects that the
saved counterfactuals cannot separate in M units: g_dna re-scaling `dna_ALT`
(pure sequence) and g_epi re-scaling `epi` (context content). The L2 split above
says they are comparable, but L2 is a proxy — the head is nonlinear and different
directions have different gain. One more counterfactual,
`head(g(A)_d·dna_A + g(R)_e·epi)`, closes it exactly. Worth paying for before the
claim goes in the manuscript, because the honest wording depends on which
sub-channel dominates.

#### Sub-channel split — DONE 14 Sep 2026. SEQUENCE dominates.

Ran as `45997644` (array 0=genoa, 1=egtex). **The job shows FAILED 1:0 in
`sacct` but its science is complete and correct** — see the post-mortem below
before concluding anything from the exit code. Analysed with
`54_gate_decomposition.py analyse` (CPU, zero GPU) at 22:17 UTC.

Identity `Delta_M_Gate_gdna + Delta_M_Gate_gepi = Delta_M_Gate_Channel` holds to
**8e-8 (genoa) / 1.4e-7 (eGTEx)** on both strands, so the split is exact rather
than fitted, and the run reproduces `scripts/20` to 3.8e-6 on all 66,495 /
76,893 pairs — the same figure `45992001` got.

| | genoa | eGTEx |
|---|---|---|
| variance share **of the gate channel**, g_dna (sequence self-rescaling) | **0.821** | **0.850** |
| variance share **of the gate channel**, g_epi (context content) | 0.330 | 0.314 |
| variance share of *total*, g_dna | 0.153 | 0.166 |
| variance share of *total*, g_epi | 0.0617 | 0.0613 |
| SD in M, g_dna / g_epi | 0.0366 / 0.0232 | 0.0407 / 0.0247 |
| corr(g_dna, g_epi) | −0.145 | −0.158 |

The two shares exceed 1 because the sub-channels **anti**-correlate — note the
sign flips against the +0.38–0.40 between the DNA and gate channels; do not
describe the two covariances with the same words.

**This settles the wording question.** The gate channel is dominated by the
sequence sub-channel: g_dna re-scaling `dna_ALT` is ~82–85% of its variance,
while context content is ~31–33% of the gate channel and only **~6% of total**
variant-effect variance. The L2 proxy's direction was right (it put context at a
median 40% of sequence) but L2 could not have established the ranking, because
the head is nonlinear. So: context content does enter the variant effect and is
**not zero**, but it is the minority sub-channel of a minority channel. Wording
must not promote it to a co-equal partner of sequence.

#### Post-mortem: `45997644` FAILED 1:0 with its outputs already correct

Worth reading before trusting any exit code in this chain. The script never
raised. `check_no_clobber` is the **last line** of the sbatch, and it returned 1
under `set -e` after the instrumented CSVs were written and flushed.

The guard is wall-clock (`find -newermt <stamp>`, stamped 20:03:52). Task D ran
on the **login node** and wrote `results/journal/joint/transfer_failure/*` at
20:08:11 — inside the window, outside the declared subtree. The guard cannot
tell a sibling analysis's legitimate output from this job clobbering a published
path, so it flagged Task D's own now-committed (`24c607e`) files.

`sacct` is the giveaway and the general lesson: elapsed **01:04:14** against
`45992001_0`'s 01:04:16. A job that "died early on a script error" does not run
the full hour. **Check elapsed time against the known-good run before assuming
an exit-1 job did no work** — the instinct to blame the most recently written
code cost this one a re-diagnosis, and would have cost 2.4 GPU h of needless
resubmission.

Fixed in `2fcd516`: `check_no_clobber` takes an opt-in
`--concurrent <subtree>`. Default behaviour is unchanged and still hard-fails an
*unexempted* sibling write, which was verified along with a real published-path
clobber. Exemptions are stated in the sbatch so they stay visible.

### C. Gate share vs measured tissue plasticity — DONE, null

`scripts/55_gate_plasticity.py` → `results/journal/joint/gate_plasticity/`.
Zero GPU. 46,536 probes (all4 joint validation loci, chr10+chr11), each with
`Median_Beta` measured in all four tissues.

Data note: `data/datafiles_joint/all4` assigns **one tissue per probe**, so
cross-tissue variance cannot come from the joint build. It comes from the four
per-tissue builds, each of which carries its own `val.csv` on the same two
chromosomes — a direct four-way join on probeID, 46,539 probes shared, 46,536
after matching the gate file.

**The hypothesis is null.** `gate_dna_share` vs cross-tissue variance:
**ρ = +0.042**, 95% CI [+0.019, +0.062] over 1 Mb blocks. Nonzero only because
n is large; negligible in size, and it **flips to −0.061** once mean β is
partialled out. Stratified by assigned training tissue: +0.085 / +0.041 /
+0.012 / +0.028. Attenuation cannot rescue it — fwd/RC share MAE 0.057 against
a q10–q90 range of 0.413 (~14% noise-to-range) does not turn 0.04 into a result.
Reported raw, **not disattenuated**.

*The null is real, not a Spearman artifact.* The decile table is U-shaped
(d1 median var 0.0027 → d3–d4 ~0.00002 → d6–d10 ~0.0006), and the U is fully
explained by level: `gate_dna_share` correlates **+0.396** with mean β, and
cross-tissue variance is mechanically compressed at β≈0 and β≈1 (deciles 3–4
sit at mean β ≈ 0.05). The share tracks methylation level; level induces the U.
There is no plasticity signal underneath.

**What this constrains:** the DNA/context *share* — the quantity the paper
actually interprets — carries essentially no tissue-plasticity information. Any
wording suggesting the gate discovers tissue-variable loci is unsupported.

**One real signal, but not the hypothesised one.** Both raw gates correlate
negatively with variance, and so does their sum: `gate_total` **ρ = −0.621**.
At tissue-variable loci the model shrinks *both* branches rather than
re-balancing them. Tested against the obvious alternative (prediction
difficulty): `gate_total` vs |pred err| ρ = −0.501, variance vs |pred err|
ρ = +0.530, partial(gate_total, var | pred err) = **−0.484**, partial given
level *and* error = **−0.465**. It survives both controls at ~three-quarters
strength.

Treat as a **lead, not a claim.** Gates are applied after LayerNorm, so
shrinking both gates shrinks the fused vector toward the head's bias — i.e.
toward the mean prediction. `gate_total` is therefore most naturally read as a
learned **shrinkage/confidence** term that partially tracks plasticity, not as a
plasticity detector. Anyone promoting this to a claim needs a mechanism check
first, not another correlation.

This is the same behaviour R5 sees from a different angle. `51_rc_uncertainty`
(§R5, mentor requirement 4) reads model uncertainty off forward/RC disagreement;
gate magnitude is a second, architecturally explicit route to the same quantity —
the model hedging where methylation is unstable. If the two agree per locus, the
gate is an uncertainty readout and should be described as one throughout, which
also retires any remaining temptation to read the gate as modality
*attribution*. That correlation has not been run; it is the obvious next step if
budget survives B and D.

Task C also actively supports the caveat the paper already carries: gates are
descriptive scaling, not causal attribution. The null is doing work here, not
just failing to find something.

Per the standing note, every statement here is distribution-level; no per-locus
gate share is quoted.

### D. Where zero-shot transfer fails — DONE 14 Sep 2026. Hypothesis HOLDS.

`scripts/56_transfer_failure.py` → `results/journal/joint/transfer_failure/`.
Zero GPU. 26,558 held-out breast test probes (chr8+chr9), 99.95% of the 26,570.

**Split discipline — do not conflate with Task C.** Task C's plasticity null is on
joint **validation** probes (chr10+chr11). This is on held-out **test** probes
(chr8+chr9). The two sets are **disjoint — zero probe overlap** — so the
cross-tissue variance here is recomputed from the four tissues' own `test.csv` by
the same method. Both are legitimate; they are not the same analysis and must
never be written as one.

**Error rises monotonically with measured tissue plasticity.** Spearman
ρ = **+0.488** between sequence-arm absolute β error and cross-tissue variance,
and **+0.475** after partialling out mean β — so it is not the level confound
that killed Task C. The decile table is monotone across all ten bins:

| decile | cross-tissue var | sequence MAE | fusion MAE |
|---|---|---|---|
| d1 | 0.00000 | 0.0448 | 0.0343 |
| d5 | 0.00026 | 0.0944 | 0.0815 |
| d10 | 0.02121 | **0.1965** | 0.1753 |

A **4.4× spread** in error, ordered perfectly by plasticity. Top-decile-error
probes carry **13.5×** the cross-tissue variance of the rest (0.00449 vs
0.00033).

**The failures have regulatory character, and it is coherent.** Reported as it
came out — no story was hunted for:

| annotation | top decile vs rest | direction |
|---|---|---|
| CpG-island **Shore** | 34.8% vs 24.2% (log2 **+0.53**) | enriched |
| CpG **Island** | 21.1% vs 31.0% (log2 **−0.56**) | depleted |
| Shelf | 6.1% vs 8.6% (log2 −0.48) | depleted |
| OpenSea | 38.0% vs 36.2% (log2 +0.07) | flat |
| **H3K4me1** (enhancer) | rb **+0.219**, p=1.4e-76 | enriched |
| H3K27me3 (polycomb) | rb +0.122, p=4.0e-25 | enriched |
| H3K4me3 (promoter) | rb −0.072, p=1.4e-09 | slightly depleted |
| ATAC | rb −0.056, p=2.0e-06 | slightly depleted |

Transfer fails at **island shores carrying enhancer (H3K4me1) and polycomb
(H3K27me3) chromatin**, and succeeds at **CpG islands with promoter chromatin**.
That is the expected tissue-variable/tissue-invariant division of the methylome,
recovered without supervision from prediction error alone. This is the closest
thing to a biological finding in the paper.

### E. ASM validation — SUPERSEDED 15 Sep 2026. The drop was OUR error.

**Do not act on this subsection.** It is kept because the arithmetic in it is
the arithmetic a reader will reconstruct, and the reason it is wrong is worth
recording. The corrected check is `E (REOPENED)` below, and it reverses the
decision. Two independent mistakes, either of which alone was enough to sink it:

1. **Wrong denominator.** It assumed a scored CpG must be an HM450 probe
   (26,570 / 28M = 0.095%). Nothing in the architecture requires that — see
   below.
2. **Wrong numerator.** It took the Nat Commun atlas's 34,426 ASM loci as the
   ceiling and dismissed CanASM in one line as "tumour-focused, wrong tissue
   context" without ever counting it. CanASM holds **5,003,877 SNV–CpG pairs**,
   145x the number used as the ceiling.

Catalogue check only, zero compute, no build attempted — per directive 4.

**Catalogues found (all public, all usable in principle):**

| resource | basis | scale |
|---|---|---|
| Atlas of imprinted and allele-specific DNA methylation in the human body (Nat Commun 2025) | WGBS | **34,426** ASM loci — the largest published |
| ASMdb (NAR 2022, `dna-asmdb.com`) | WGBS, 1,484 human BS-Seq datasets | open access, SNP-linked |
| CanASM (BMC Genomics 2025) | WGBS, 31 cancer types | tumour-focused, wrong tissue context |

**Why it fails, and it is structural rather than a matter of finding a better
catalogue.** Every public ASM resource is **WGBS-based**, and WGBS ASM calls sit
at arbitrary genomic CpGs. Our evaluation is locked to HM450 probes, and to the
**held-out test split only** — chr8 + chr9, 26,570 probes:

    HM450 probes total                 485,577
    our held-out test probes            26,570   =  5.47% of the array
    HM450 coverage of all human CpGs             1.73%

Expected usable CpGs = N x (26,570 / 28e6):

| genome-wide ASM CpGs N | expected on our test probes |
|---|---|
| 34,426 (largest atlas) | **33** |
| 100,000 | 95 |
| 250,000 | 237 |
| 526,910 | 500 |

Reaching the ~500-pair floor needs **N ≥ 526,910** genome-wide ASM CpGs — **15×
more than the largest published atlas**. Even granting a generous 5× enrichment
of ASM at array (regulatory) positions, the requirement is still ~105,000, or
**3.1× the largest atlas**. An array-based ASM study would avoid the 1.73%
penalty but would still lose 94.5% to the chr8/chr9 restriction, and no
array-based ASM resource of the needed scale appears to exist.

**Decision: dropped, and the mentor gets the reason.** The honest sentence is
that no adequately powered public ASM resource intersects our held-out probe
set — the limit is the chromosome-held-out design combined with array coverage,
not a lack of searching. Running it anyway would produce ~33 pairs, which is
worse than not running it, and we already carry a documented pattern of null
under-powered external results (four disease-variant tests). Slot `53` stays
reserved and unused.

### E (REOPENED). ASM validation — E1 DONE, modest positive. 15 Sep 2026

Catalogue check only, zero compute, nothing built — directive 4 still governs,
and this is the "report usable n before building" step it asks for.

#### What the old calculation got wrong

**HM450 is a training-data constraint, not an inference constraint.** The model
takes a 1,000-bp sequence window plus the context features and predicts
methylation at the central CpG. Sequence, the seven bigWig tracks and phyloP are
all genome-wide. Nothing in the forward pass asks whether the target CpG carries
an Illumina probe. Held-out-ness is preserved by the **chromosome** split, not
by the array: any CpG on chr8 or chr9 is as unseen as a probe on chr8 or chr9.

So the filter is chromosomal, and it is ~93x weaker than assumed. Measured
directly off our own reference rather than quoted:

    hg38 primary assembly (chr1-22,X,Y)        3,088,269,832 bp
      chr8 + chr9                                283,533,353 bp   =  9.181%

    CpG dinucleotides, primary assembly            29,401,360
      chr8 (1,338,200) + chr9 (1,255,728)           2,593,928     =  8.822%

CpG fraction (8.82%) is the right one to apply to an ASM catalogue, since ASM
records are CpGs and not random positions. Both agree with the mentor's ~9.2%.

    basis                        restriction              fraction
    old (Claude Code)            HM450 test probes          0.095%
    corrected                    chr8+chr9, any CpG          8.82%

#### The catalogue, counted rather than assumed

**CanASM** (BMC Genomics 2025, `bioinfor.nefu.edu.cn/CanASM/`) — the resource the
old check dismissed in a line:

    unique SNV–CpG pairs                   5,003,877
    index SNVs                             3,056,776   (2,634,406 are SNPs)
    CpGs                                   4,157,508
    samples                                      226 BS-seq, 31 cancer types,
                                                     30 tissues, matched normals
    genome build                              GRCh38   — same as ours, no liftOver
    ASM test              Fisher's exact on the 2x2 of methylated/unmethylated
                          reads by REF vs ALT allele; BH correction; per-allele
                          read coverage floor 5, ceiling 200

Breast, counted from the database's own sample table (284 GSM records embedded
in the site bundle): **15 samples with `Tissue = Breast`** — 12 cancer, 3 normal
(`GSM1279517`, `GSM4090863`, `GSM4090886`) — plus 4 breast-cancer PDX models, 2
nipple aspirate fluid, and 1 breast-cancer plasma sample. Breast is the 7th
best-represented tissue of 30.

**The 1,000-bp window is nearly free, and this is the point that decides it.**
CanASM's method section: *"For an SNV–CpG pair, overlapping reads were used for
ASM identification through Fisher's exact test."* The SNV and the CpG must be
covered by the **same reads** — that is what makes the allele assignment
possible at all. SNV–CpG separation is therefore bounded by BS-seq read/fragment
length, i.e. tens to a few hundred bp (the paper's own worked example,
`rs1883832`, has all six CpGs within 40 bp). Essentially every CanASM pair
already satisfies our 1,000-bp window. The "maybe a fifth survive" hedge is far
too conservative; the survival rate is ~1.

#### Usable n

    5,003,877 pairs x 8.822% (chr8+chr9)   ~=  441,000 pairs, all tissues
    x ~1.0 (1,000-bp window, see above)    ~=  441,000

Breast-only is the number that matters and it cannot be read off the aggregate,
because pairs are deduplicated across samples. Bounding it pessimistically: if
breast contributed only **1%** of CanASM's pairs — far below its 15/226 = 6.6%
sample share — that is 50,000 genome-wide, **~4,400 on chr8+chr9**. Nine times
the ~500 floor under an assumption chosen to be unfair to the analysis.

**Verdict: E clears the floor, by a wide margin, and is reopened.** The old
"underpowered ~15x" line is withdrawn.

#### The one thing still blocking a build

`bioinfor.nefu.edu.cn/asm/*` is **dead** as of 15 Sep 2026, 22:00-22:45 UTC.
nginx answers; the application behind it never does.

Diagnosed rather than assumed, because "the server is down" is the kind of claim
that is usually a malformed request:

    /CanASM/  and the 5.9 MB JS bundle      200, 1.2-3.9 s
    /zzz_not_a_real_path                    404 in 1.2 s   (nginx answers)
    /asm  (no trailing slash)               404 in 1.2 s   (no location match)
    /asm/zzz_not_a_real_endpoint            504 after 61 s
    /asm/snv_cpg?...  (a real endpoint)     504 after 61 s

The fourth line is decisive: a **nonexistent** path under `/asm/` behaves
identically to a real one, so nginx matches `location /asm/`, proxies, and gets
no reply for *any* path. A live backend would 404 the bogus path instantly.

The request shape was verified against the app's own axios wrapper
(`get(e,a){return r.get("https://bioinfor.nefu.edu.cn/asm"+e,{params:a})}`), and
variants were tested: `sampleType=Cancer` vs empty, `chr=chr8` vs `chr=8`, no
params, and with browser `Referer`/`Origin`/`User-Agent`/`Accept` headers. All
hang identically. It is not our request.

**Temporary or permanent — unresolved, leaning temporary.** The Wayback Machine
has exactly one `/CanASM/` snapshot (8 Apr 2025, around the preprint) and has
never crawled `/asm/`, so the archive cannot date the outage. But the site's JS
bundle has changed since that snapshot (`app.bca47efc.js` -> `app.aab82547.js`),
which means the site *is* maintained. Note also that the app sets its client
timeout to **3e6 ms (50 minutes)**, implying the developers expect these queries
to be extremely slow — while nginx gives up at 60 s. A normal browser would
therefore 504 too, so this may be chronic rather than a passing outage.

So the exact breast chr8+chr9 pair count is **not yet measured** — it is bounded,
not counted. That is an availability problem, not a power problem, and it does
not change the verdict. Retry before building; ASMdb is the fallback (below).

**ASMdb — investigated properly 15 Sep 2026. It is blocked, but not for the
reason first recorded here.**

*Correction to an earlier claim in this subsection.* It said ASMdb returns "612
rows" for a human sample, implying the resource is tiny. That was `GSM2191797`,
an atypically sparse dataset. Breast samples are two orders of magnitude larger:

    GSM1279517  Breast normal, WGBS     49,608 ASM regions
    GSM1328112  Breast MCF7,   WGBS     32,009 ASM regions
    GSM2191797  (the one first checked)     612 ASM regions

ASMdb holds **2,242 human datasets**, 1,319 of them methylation assays (918
WGBS, 253 RRBS, 82 Bisulfite-Seq, plus oxBS/TAB/hairpin variants), of which
**47 are breast-related** — including `GSM1279517`, the same normal breast WGBS
sample CanASM uses. On scale alone ASMdb would be ample: ~8.8% of ~50,000
regions is ~4,400 on chr8+chr9 from the normal breast sample by itself.

**The blocker is the SNV linkage, and it is hard.** Every queryable ASMdb table
was enumerated (`asmsqllist` = ASM regions, `asmincpgsqllist` = ASM in CpG
islands, `datasetssqllist` = sample metadata). **None carries an SNV, SNP,
allele, REF or ALT column** — the ASM table is `chrom, start, end, span,
methCount, strand`. ASM records are *regions*, not SNV–CpG pairs.

The per-species SNP calls that would supply the linkage are **not downloadable**.
Every bulk link on `/MethAelle/download` — including `Homo_sapiens.snp.tar.xz`
(listed at **20 G**) — points at `/MethAelle/underrequest`, which serves the text
*"Under Request! will published soon!"*. No form, no contact address, no data.
The ASMdb paper is from 2022 and the site's own visitor counter stopped in
Sep 2021, so "soon" has not arrived in four years.

**So ASMdb cannot produce SNV–CpG pairs**, and the earlier framing ("a build, not
a download") was too optimistic: there is nothing to build *from*, because the
allele-level data never shipped.

**One degraded variant of E remains available without CanASM**, and it should be
offered rather than silently dropped: pair a public variant catalogue (dbSNP /
1000G) to ASMdb ASM regions on chr8+chr9 and test whether SilentMethyl's
predicted |delta| is higher for variants inside ASM regions than for
distance-matched variants in matched non-ASM regions. That is an **AUROC-style
discrimination test of exactly the kind the project already runs**, and it is
genuinely informative. But it **cannot** deliver direction concordance or signed
Spearman — the two statistics the mentor named first — because ASMdb never
exposes which allele is methylated. Treat it as a fallback worth one paragraph,
not as a replacement for the CanASM analysis.

#### THE RESOURCE THAT ACTUALLY UNBLOCKS E — counted, not bounded, 15 Sep 2026

**We were never restricted to the two catalogues the mentor named.** The
Rosenski/Dor/Kaplan atlas (*Atlas of imprinted and allele-specific DNA
methylation in the human body*, Nat Commun 2025, `PMC11897249`,
doi 10.1038/s41467-025-57433-1) publishes its ASM calls as **open-access
supplementary tables**. No server, no request form, no gate. Downloaded and
verified on disk:

    https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-57433-1/
      MediaObjects/41467_2025_57433_MOESM4_ESM.xlsx    15.5 MB  (Tables S2-S16)
      MediaObjects/41467_2025_57433_MOESM3_ESM.zip     32.4 MB  (Data S1, 531 MB tsv)

Sheet **`S3. ASM SNPs`** is the one that matters: **55,271 ASM SNP loci**, with
`chrom`, `SNP pos (hg19)`, `dbSNP id`, `Alleles`, `ASM bimodal region (hg19)`,
per-sample Fisher exact p, BH-adjusted p, and the sample names the call fired in.

**This is the same GSE186458 atlas already listed in §3 as "deferred, public,
hg38".** We had it on the deferred list for a different purpose and never
connected it to E.

**Usable n, measured end-to-end** — parsed, lifted hg19->hg38 with the project's
existing `data/reference/hg19ToHg38.over.chain.gz` (99.6% clean), then CpGs
counted directly in `data/hg38.fa` within 1,000 bp of each SNP and inside its
ASM region:

    chr8 + chr9, ALL tissues        5,489 ASM SNPs  ->  136,065 SNV-CpG pairs
    chr8 + chr9, breast epithelium    390 ASM SNPs  ->   12,550 SNV-CpG pairs
                                   (median 19 and 26 CpGs per SNP respectively)

**Breast alone clears the ~500 floor by 25x.** All-tissue clears it by 270x.
The 9.98% chr8+chr9 share of the 55,271 loci matches the 8.8-9.2% expectation
from CpG and bp fractions, which is a useful sanity check on the whole
denominator argument.

**The tissue match is better than CanASM's, not worse.** Seven purified breast
samples, all normal: `Breast-Basal-Epithelial` x4 (`Z000000V6`, `Z000000VG`,
`Z000000VL`, `Z0000043E`) and `Breast-Luminal-Epithelial` x3 (`Z000000V2`,
`Z000000VJ`, `Z000000VN`). CanASM's breast is 12 tumour / 3 normal of 15.
**Our model's context is breast epithelium**, so purified normal breast
epithelium is the closest tissue match any ASM resource offers us.

#### The limitation, and it decides which statistics are available

**The supplementary publishes significance, not direction.** `S3`/`S4` carry
Fisher p and FDR per sample but **no signed allelic methylation difference** and
no per-allele beta. `Data S1` is a binary 0/1 presence matrix of bimodal regions
by sample, not methylation values. So from these tables alone:

    AUROC, ASM SNPs vs distance-matched non-ASM SNPs      AVAILABLE, well powered
    direction concordance                                 NOT available
    signed Spearman                                       NOT available

The mentor named direction concordance and signed Spearman first. Getting them
needs a per-allele methylation value, which means either CanASM (server dead) or
computing it ourselves from GSE186458's read-level `pat` files plus donor
genotypes — a genuine build, not a download.

**So E splits into two tiers, and the first is unblocked right now:**

- **E1 — discrimination, zero blockers.** 12,550 breast / 136,065 all-tissue
  chr8+chr9 pairs. Does predicted |delta| separate ASM SNV-CpG pairs from
  distance-matched non-ASM pairs? Same protocol as GENOA/eGTEx: 10 bp matching
  tolerance, 1 Mb block bootstrap, distance-only baseline alongside. ~3 GPU h.
  This is a real external validation and it is the one we can actually run.
- **E2 — signed agreement.** Needs CanASM to come back, or a GSE186458 build.
  Do not promise it.

**Recommendation: run E1, state E2 as a limitation.** E1 answers the mentor's
underlying question — does the model recognise allele-specific methylation it
was never trained on — and the tissue match is better than the resource he
suggested. Report the OOD-position caveat exactly as recorded above.

#### Protocol, when it is authorised

Same as GENOA/eGTEx, no new machinery:

- direction concordance, signed Spearman, AUROC vs distance-matched negatives;
- 10 bp matching tolerance, 1 Mb block bootstrap, distance-only baseline
  reported alongside;
- **not** exact effect-size agreement — the mentor is right that the scale
  mismatch (allelic methylation difference within one individual vs our predicted
  delta) makes agreement in magnitude the wrong target;
- prioritise the 3 normal breast samples; report cancer and normal separately,
  since 12 of 15 breast samples are tumour and tumour methylation is not a clean
  stand-in for the breast-epithelium context the model carries.

#### E1 BUILT AND SUBMITTED — `46096838`, 15 Sep 2026

Scope note: this is the **authorised** build of E1, not a scope expansion. E was
reopened on the corrected power calculation above; the mentor asked for ASM
validation; this is it.

**Scripts (slot 53, reserved for ASM since the R2 framework was written):**

    53_asm_build.py      atlas -> scoreable pairs + context      CPU, done
    53b_asm_score.py     frozen-checkpoint inference             GPU
    53c_asm_evaluate.py  AUROC, block bootstrap, baselines       CPU
    jobs/r8_journal/53_asm_score.sbatch                          both stages

**Script 20 is NOT touched.** It reads target CpGs out of the split CSVs and then
hard-fails any probeID absent from the HM450 manifest — both assumptions are
exactly what this analysis exists to escape. `53b` path-loads script 20 for
`build_model` / `score_chunk` (the idiom script 23 already uses), so numerics are
identical to GENOA/eGTEx — FP32, forward/RC averaging, phyloP swap on the RC
context — while every frozen analysis depending on script 20 is unaffected.

**Window arithmetic — a real error caught during the build.** The model window is
`seq[2000:3000]`, so the target C sits at index 499 and the variant must fall at
offset **-499 to +500**: an asymmetric +/-500 bp, **not** +/-1000. The "12,550
breast pairs" figure recorded earlier in this subsection used +/-1000 and is
therefore roughly 2x too generous. The build uses the correct range; the counts
below supersede it.

**Counts as built** (`data/external/asm_atlas/scoring/build_summary.json`):

    ASM SNP loci, all chromosomes          55,203
    lifted hg19 -> hg38                    55,079   (99.8%, chain already in repo)
    on chr8+chr9                            5,489
    SNV-CpG pairs constructed              100,274
      positive  (CpG in its ASM region)     82,747
      negative  (bimodal, not ASM)           3,455
      negative  (background)                14,072
      dropped   (in some OTHER ASM region)   4,190
    after 1:1 distance matching             35,054 rows, 2 contrasts
    unique CpGs / unique variants      27,510 / 4,796

**Two contrasts, and the second one is why this is worth believing.**

    positive_vs_bimodal_non_asm   3,455 + 3,455   CONTROLLED -- lead with this
    positive_vs_background       14,072 + 14,072   weaker, less specific

ASM regions are a subset of the atlas's *bimodal* methylation regions —
CpG-dense, intermediate-methylation, enhancer-like. Against plain background CpGs
the model could separate them by recognising that regional character alone, with
nothing allele-specific involved, and a reviewer will say exactly that. The
bimodal-but-not-ASM tier holds the character fixed. **If background separates and
bimodal does not, the honest conclusion is that the model recognises region
class, not ASM** — `53c` prints that verdict itself so it cannot be quietly spun.

**A matching bug, caught and fixed.** The first build kept all 82,747 positives
against 17,527 negatives instead of pairing 1:1. Rewritten as greedy 1:1 with a
two-tier preference (same variant first, then same chromosome), each negative
consumed at most once, and **unmatched positives dropped**. Balance after the fix:

    contrast                      label   n        median |dist|   mean |dist|
    positive_vs_background          0   14,072         339           319.2
    positive_vs_background          1   14,072         339           318.5
    positive_vs_bimodal_non_asm     0    3,455         352           328.3
    positive_vs_bimodal_non_asm     1    3,455         351           327.5

So the distance-only baseline should pin at 0.5. If it does not, the matching
failed and neither AUROC is interpretable — `53c` checks this and says so.

**Smoke test, CPU, 64 pairs: all 64 scoreable, every rejection counter zero.**
The one that matters is `reference_base_mismatch: 0` — an off-by-one between the
build's `cpg_pos0`, the variant coordinate and the hg38 sequence would have sent
essentially every row into that bucket. It did not.

**Breast is thin in the controlled contrast** — 309 positives / 276 negatives
(585 pairs). Reportable with wide intervals; the powered result is the all-tissue
bimodal contrast at 3,455 per side. Do not lead with the breast number.

**Still not measured, and not by choice.** The atlas tables carry significance and
sample membership but **no signed allelic methylation difference and no
per-allele beta**. Direction concordance and signed Spearman — the two statistics
the mentor named first — are therefore unavailable from this source. That is E2,
and it needs CanASM (dead) or a build on GSE186458's read-level data. **Do not
substitute |delta| agreement for direction agreement.**

#### E1 RESULT — `46096838`, 15 Sep 2026. It works, and it is modest.

**Job state first, because `sacct` will mislead you.** `46096838` reads
**FAILED 1:0** with elapsed **01:09:50**. The science completed and every output
is on disk. This is the `45997644` pattern exactly: full-length elapsed, correct
outputs, guard tripped at the final line.

**Cause, and it was self-inflicted.** `check_no_clobber` fired on six files:

    results/journal/ablation_breast_epithelium/context_ladder/VERDICT_CORRECTION.md
    results/journal/manuscript_figures_r8/{run_summary.json, fig_r8_1..5.png}

All were written **from the login node while the job was running** — the R8
figures (subsection H) and the script-23 correction sidecar. Both live under
`results/journal/` and outside the job's declared subtree, so the guard did
exactly its job. This is the *same* mistake recorded in `_common.sh`'s own
comment about 14 Sep, made again. **Do not write anything under
`results/journal/` while a job is in flight**, or declare the subtree with
`--concurrent`. The outputs were not resubmitted; nothing is wrong with them.

#### The numbers

Absolute predicted M-scale effect (`Absolute_Delta_M`), AUROC against
distance-matched negatives, 1 Mb block bootstrap, 2,000 replicates:

    contrast / stratum                      arm        AUROC   95% CI          excl 0.5
    positive_vs_bimodal_non_asm  all tissues fusion    0.5625  [0.533, 0.593]   yes
    positive_vs_bimodal_non_asm  all tissues sequence  0.5736  [0.545, 0.601]   yes
    positive_vs_bimodal_non_asm  all tissues distance  0.5017  [0.478, 0.524]   -
    positive_vs_bimodal_non_asm  breast      fusion    0.5879  [0.483, 0.683]   NO
    positive_vs_bimodal_non_asm  breast      sequence  0.6045  [0.512, 0.688]   yes
    positive_vs_background       all tissues fusion    0.5762  [0.554, 0.598]   yes
    positive_vs_background       all tissues sequence  0.5820  [0.556, 0.605]   yes
    positive_vs_background       breast      fusion    0.5726  [0.520, 0.644]   yes
    positive_vs_background       breast      sequence  0.5845  [0.522, 0.664]   yes

**Matching is sound.** Worst distance-only baseline deviation from 0.5 is 0.0376,
and the all-tissue baselines sit at 0.5016 and 0.5017. The headline AUROCs are
not distance leaking back in.

#### What this licenses, and what it does not

**1. The controlled contrast separates. This is the result.** Against
bimodal-but-not-ASM CpGs — holding the CpG-dense, intermediate-methylation,
enhancer-like regional character fixed — both arms exclude 0.5 on all tissues
(fusion 0.5625, sequence 0.5736). So the model is **not** merely recognising
region class. It carries some genuine signal about which CpGs are
allele-specifically methylated, at positions no array probe covers and on
chromosomes it never saw. That is a real external validation.

**2. It is weak, and must be reported as weak.** AUROC 0.56-0.57. Write it as
*modest but reproducible discrimination*, never as "the model predicts ASM".
A reviewer comparing 0.56 against the mQTL work's numbers will not be generous
to an overclaim.

**3. Sequence >= fusion in every single stratum.** 0.5736 vs 0.5625 all-tissue
controlled; the same ordering in all four. The intervals overlap heavily so this
is not a significant difference, but the direction is consistent and it is an
**independent confirmation of the R2 hard constraint**: epigenomic context does
not improve variant-effect prediction. E1 was not designed to test that and does
it anyway, in a third cohort type. Worth one sentence in 3.4.

**4. Breast alone is underpowered, exactly as flagged before the run.** In the
controlled contrast the breast fusion interval spans 0.5 ([0.483, 0.683],
p(<=0.5) = 0.0525) on n=585. The sequence arm clears it, barely. **Do not lead
with a breast-specific ASM number.** The powered result is all-tissue.

#### Wording for 3.4

> On held-out chromosomes, at CpGs outside the training array entirely,
> SilentMethyl's predicted variant effects distinguish allele-specifically
> methylated SNV-CpG pairs from distance-matched pairs in bimodal regions that
> are not allele-specific (AUROC 0.56-0.57, 95% CI excluding 0.5; distance-only
> baseline 0.502). The effect is modest, the sequence-only arm matches or exceeds
> the fusion arm, and the breast-restricted subset is underpowered.

State the two limitations in the same paragraph: **positional
out-of-distribution** (trained only at HM450 positions, enriched at promoters and
islands) and **no signed statistics** (the catalogue publishes significance, not
per-allele methylation, so direction concordance and signed Spearman are not
available — see E2).

Outputs: `results/journal/asm_validation/`. Figure:
`manuscript_figures_r8/fig_r8_4_asm_discrimination.png`. Supplement: S11, now
complete at 5 files.

#### The caveat that must appear in the output, not just here

The model was trained **only at HM450 positions**, which are enriched at
promoters and CpG islands. Scoring arbitrary ASM CpGs is **out of distribution
with respect to training positions** — a different thing from being out of
distribution in sequence or in chromatin, and it should be named that precisely.
It is a limitation to state, and it is also a generalisation test worth having:
a model that holds up at CpGs the array never covered is a stronger claim than
one evaluated only where it was fit. Report the position-class breakdown
(island/shore/shelf/open sea, and HM450-covered vs not) alongside the headline
so a reader can see how much of the result comes from array-like positions.

**Budget: ~3 GPU h, unchanged. Slot `53` is now live, not reserved.**

### B. Context dose-response ladder — DONE 15 Sep 2026. `46007255` analysed.

`scripts/23_context_permutation.py`, `jobs/r8_journal/23_ctx_ladder.sbatch`,
eGTEx heldout, 76,893 pairs, four rungs, seed 42:

    rung 1  identity      native context (control)
    rung 2  shuffle       another probe's context, same tissue
    rung 3  tissue_Lung   the SAME locus's context in a different tissue
    rung 4  xtissue_mean  per-locus mean across all four tissues

**Status: COMPLETED 0:0, elapsed 4:53:44, ~4.9 GPU h as projected. Analysed
15 Sep 2026 — see RESULT below.** This was the last compute in the project.

**It is single-shot — there is no separate analyse step.** On success the job
writes, under `results/journal/ablation_breast_epithelium/context_ladder/`:
`agreement_with_identity.csv`, `level_accuracy_by_scheme.csv`,
`pair_scores_{identity,shuffle,tissue_Lung,xtissue_mean}.csv`, and
`run_summary.json`. Confirm `sacct -j 46007255` is COMPLETED 0:0 **and** that
the log's per-rung guards fired (below) before reading any number.

#### RESULT — read 15 Sep 2026. Guards clean. The banner is WRONG, and here is why.

**Guards, verified before reading any number** (per the standing instruction):

    train split contributed      0 target probes     <- split discipline clean
    val   split contributed      0 target probes
    test  split contributed 20,634 target probes
    all four tissue builds resolved; 20,632 of 20,634 present in ALL four (99.99%)
    tissue_Lung   context replaced 76,884 / 76,893 rows (100.0%)
    xtissue_mean  context replaced 76,884 / 76,893 rows (100.0%)
    PhyloP max shift, both rungs   0.000e+00 (expected exactly 0)
    clobber check clean; nothing written outside the context_ladder subtree

(The notes predicted 26,558/26,570 probe coverage. The actual target count is
20,634 — 26,570 is the full test probe set, and only 20,634 of those carry
scoreable variants. The ratio is as expected; the earlier figure was the wrong
denominator, harmless here.)

**Level accuracy against truth — a clean dose-response** (`level_accuracy_by_scheme.csv`,
20,634 probes):

    rung            beta MAE   vs identity   beta Pearson
    identity          0.0899        --           0.9261
    xtissue_mean      0.0916      +1.9%          0.9246
    tissue_Lung       0.1029     +14.5%          0.9069
    shuffle           0.2260    +151.4%          0.5634

The ladder behaves exactly as designed: the more realistic the substituted
context, the smaller the cost. **The cross-tissue mean context costs 1.9%** — a
locus's average context across four tissues is very nearly as good as its own.
That is a result in its own right and it supports R3: if averaged context is
almost free, tissue-specific context is carrying little for level prediction.

#### The banner fired at full scale — and it is an artefact. Do not act on it.

The job printed, on all 76,893 pairs, the same verdict the smoke did:

> At least one scheme moved the deltas as much as the levels.
> The allele-invariance argument does NOT hold empirically here.

**The banner is implemented on Pearson only** — visible in the log's own
`levels r= / deltas r=` lines. §1 R2 already records why Pearson is the wrong
metric here: methylation levels are bimodal with SD 3.18 M-units, so a high
Pearson on levels is cheap and not comparable across the two quantities.

Pooled agreement with identity, all four metrics:

    rung          normMAE_lev  normMAE_del  ratio | Spear_l  Spear_d | sign_l  sign_d | Pear_l  Pear_d
    shuffle          0.5260       0.2266    2.32x |  0.6180   0.7755 | 0.7522  0.8440 | 0.6324  0.7528
    tissue_Lung      0.1577       0.1154    1.37x |  0.9581   0.9564 | 0.9596  0.9473 | 0.9709  0.9527
    xtissue_mean     0.0774       0.0650    1.19x |  0.9865   0.9856 | 0.9767  0.9737 | 0.9927  0.9827

Normalised MAE favours the deltas at **every** rung (2.32x, 1.37x, 1.19x).
Spearman, sign agreement and Pearson favour the deltas **only under shuffle**,
and reverse weakly at the two tissue rungs. That reversal is what trips the
banner — and it is why this could not be settled from the pooled table alone.

**This kills the §1 R2 defence as currently written.** That text says the
dissociation holds "on normalised error, Spearman and sign agreement". At the
realistic rungs, Spearman and sign agreement no longer support it. The sentence
must change regardless of how the rest resolves.

#### Why the reversal is not evidence against the dissociation

Two independent checks, and the second is the one that settles it.

**1. The rank metrics have no dynamic range left at rungs 3 and 4.** At
`xtissue_mean`, levels Spearman is 0.9865 and deltas 0.9856 — a gap of 0.0009,
with both quantities essentially unmoved. A correlation cannot resolve which of
two nearly-unchanged quantities changed less. Only normalised error retains
resolution in that regime, and it keeps the same sign throughout.

**2. Stratifying by effect size — and the stratifier matters enormously.**

Binning on the model's own `|Predicted_Delta_M|` from the identity run appears
to show the dissociation *reversing* for small deltas (ratio 0.10x-0.28x in the
smallest quintile, rising monotonically to 1.37x-2.00x in the largest). **That is
regression to the mean and must not be reported.** Selecting pairs whose
identity-run delta happened to land near zero guarantees the other rung's delta
sits relatively further away, purely by selection.

Re-binning on the **observed** effect size `beta_ref_to_alt`, which is
model-independent and therefore bias-free, removes the artefact completely.
Reproduce both with `scripts/58_ladder_effect_size_stratification.py`
(`--stratifier observed` for the result, `--stratifier predicted` to regenerate
the artefact) ->
`results/journal/ablation_breast_epithelium/context_ladder_stratified/`:

    quintile of observed |beta|   shuffle   tissue_Lung   xtissue_mean
      Q1 smallest                  2.09x       1.20x         1.06x
      Q2                           2.17x       1.24x         1.08x
      Q3                           2.13x       1.22x         1.09x
      Q4                           2.31x       1.34x         1.15x
      Q5 largest                   2.56x       1.63x         1.40x
    (ratio = normMAE_levels / normMAE_deltas, within-quintile SD; >1 = deltas
     move LESS. Delta sign agreement is flat and high across quintiles:
     ~0.84 shuffle, ~0.95 tissue_Lung, ~0.97 xtissue_mean.)

**The dissociation holds at every quintile of every rung, and is strongest for
the largest observed effects.** State the trend carefully: it is **not strictly
monotone** — only `xtissue_mean` increases at every step; `shuffle` (2.09, 2.17,
2.13, 2.31, 2.56) and `tissue_Lung` (1.20, 1.24, 1.22, 1.34, 1.63) wobble across
the lower three quintiles. The honest description is **flat across Q1-Q3, then
rising clearly in Q4 and Q5**, with Q5 vs Q1 of 2.56 vs 2.09, 1.63 vs 1.20, and
1.40 vs 1.06. The monotonicity flag in `58`'s summary records this per rung; do
not write "monotonic" in the manuscript.

Even stated that way it is the strongest form of the R2 result the project has
produced: the variants with the largest *measured* effects are the ones whose
predicted effects are most robust to having their context replaced, and the
dissociation never reverses anywhere.

#### What R2 must now say, and must not say

**Must not say:** the blanket allele-invariance claim, and the current "on
normalised error, Spearman and sign agreement" phrasing. Both are now
contradicted by the pooled table.

**Must say**, and this is well supported:

- corrupting context degrades predicted **levels** more than predicted **variant
  effects**, on scale-fair normalised error, at all three perturbation rungs;
- the separation is **dose-dependent** — 2.32x under a random context, 1.37x
  under another tissue's context at the same locus, 1.19x under the cross-tissue
  mean — and shrinks as the perturbation becomes realistic;
- the separation is **largest for the largest observed effects** (Q5 vs Q1 at
  every rung), so it is not a small-effect noise artefact — but it is flat across
  the bottom three quintiles and must not be called monotonic;
- **report the Pearson/Spearman reversal at rungs 3-4 explicitly**, with the
  saturation explanation. A reviewer who recomputes it will find it, and the same
  transparency argument already applied to the original Pearson discrepancy in
  §1 R2 applies here. Do not quietly select the metric that agrees with us.

**The honest one-line version:** *context sets the methylation baseline and the
predicted variant effect is comparatively robust to it, increasingly so for
variants with larger measured effects — but under realistic context substitution
both quantities move so little that rank-based metrics cannot separate them.*

#### On the banner itself — FIXED 15 Sep 2026

`23_context_permutation.py`'s verdict banner tested a Pearson inequality that
§1 R2 had already documented as the wrong comparison, so the next person to run
the script would have got the same wrong verdict. Now fixed:

- the verdict is decided on **normalised MAE** (each quantity over its own SD);
- Pearson is still printed, and the banner **says explicitly when the two
  disagree and why** rather than hiding the disagreement;
- the bare `deltas_preserved_more` key is **removed**, not redefined. Anything
  assuming the old Pearson semantics now fails loudly instead of silently reading
  a different verdict under the same name. Replaced by
  `deltas_preserved_more_normalised_mae` and `deltas_preserved_more_pearson`,
  plus a `verdict_metric` field in `run_summary.json` explaining the choice.

Replayed against the frozen `46007255` table, the corrected logic flips the
verdict from *"the allele-invariance argument does NOT hold"* to *deltas survive
better in every scheme*, with the Pearson disagreement noted — matching
subsection B's independent analysis.

**The job was NOT re-run** (~4.9 GPU h, and no data would change — only the
derived verdict). **`run_summary.json` in `context_ladder/` was NOT edited**; it
records what the job actually produced. Instead
`results/journal/ablation_breast_epithelium/context_ladder/VERDICT_CORRECTION.md`
sits next to it and explains the discrepancy, so a reader who opens the frozen
JSON or the job log cannot take the old verdict at face value.

#### Why `45998301` failed, and what the fix was

Genuine `NameError: TABULAR_FEATURES` at line 123 in `load_tissue_contexts`.
Script 23 path-loads `scripts/20` as its scorer but **never imported
`training_common`**, so four names were undefined: `TABULAR_FEATURES`,
`MISSING_FEATURES`, `PHYLOP_1`, `PHYLOP_2` (an AST scan confirms those are the
complete set — no others lurk).

Rungs 1 and 2 never reach any of them: `load_tissue_contexts` sits behind
`needs_tissue`, and the PhyloP guard behind the `scheme.startswith("tissue_")`
branch. **That is the whole reason the ladder ran fine for two rungs and broke
the moment rungs 3 and 4 were added** — not a flaw in the new rung logic, which
is sound. Fixed in `2fcd516` with the same `sys.path` + `from training_common`
idiom scripts 20 and 54 already use.

#### Smoke test `46006922` — COMPLETED 0:0, all guards fired

400 pairs / 105 probes, scratch output. The guards are the point of the smoke,
so record that they actually triggered rather than merely not crashing:

    tissue_Lung   context replaced 400/400 (100.0%)   PhyloP max shift 0.000e+00
    xtissue_mean  context replaced 400/400 (100.0%)   PhyloP max shift 0.000e+00
    all four tissue builds resolved 105/105 probes, shared in ALL four

PhyloP shift of exactly zero matters: conservation is not tissue-specific, so a
nonzero shift would mean the swap is reading the wrong columns and every rung
below it is meaningless. Coverage at full scale will be **26,558 of 26,570
probes (99.95%)** present in all four tissue builds — checked directly, so the
`moved >= 0.5` assertion has enormous headroom.

Level MAE on the smoke's 105 probes, for orientation only — **n is far too small
to read as the result**: identity 0.0912, xtissue_mean 0.0907, tissue_Lung
0.0959, shuffle 0.1074.

#### A flag on the smoke's verdict banner — RESOLVED, see RESULT above

The smoke printed: *"At least one scheme moved the deltas as much as the levels.
The allele-invariance argument does NOT hold empirically here. Do not claim it;
report this instead."* On 105 probes that is not a finding, and the full run
re-evaluates it from scratch. It is flagged here only because it can touch the
same R2 sentences as the sub-channel split, which is exactly the churn
directive 2 exists to prevent. **Read it off `46007255`, not off the smoke.**

**RESOLVED 15 Sep 2026.** The banner did fire at full scale, and it is a Pearson
artefact — see `RESULT` above. The dissociation holds. The smoke was not
misleading about *whether* the banner would fire, only about what it means.

#### If `46007255` fails — MOOT, it completed 0:0. Kept for the clobber-guard idiom.

Check `sacct` elapsed first, per the `45997644` post-mortem above: a full-length
elapsed with outputs on disk means the science landed and only a guard tripped.
The clobber check now declares its subtree, and nothing else should be writing
under `results/journal/` — but if a sibling analysis is running, exempt it with
`check_no_clobber "$STAMP" "${OUT}" --concurrent <subtree>` rather than dropping
the guard.

### F. Where the fusion gain concentrates — DONE 15 Sep 2026, zero GPU

`scripts/57_fusion_gain_stratified.py` ->
`results/journal/ablation_breast_epithelium/fusion_gain_stratified/`.
Breast-epithelium held-out test set, 26,570 probes, chr8+chr9 — the same probe
set as Task D, so the two are directly comparable. Seeds 42/43/44 ensembled to
match `22_context_stratification.py`; ATAC and H3K27ac reproduce that script's
frozen `fusion_gain_by_context.csv` to the fourth decimal, which is the
regression test. 2,000 replicates, 1 Mb block bootstrap, per stratum.

This is the analysis Results 3.2 needs and did not have. It extends the existing
stratification in two directions: **all seven tracks** (not just ATAC and
H3K27ac), and a **paired AUROC difference** alongside the paired beta MAE.

**Every stratum's fusion gain is significant.** All 33 strata have both the beta
MAE difference and the AUROC difference with 95% block-bootstrap intervals
excluding zero. There is no region where context fails to help. The finding is
about *how much*, not *whether*.

#### Read the relative column, not the absolute one

Absolute beta-MAE gain is capped by the sequence-only error in the stratum, so a
stratum that starts worse can post the biggest absolute gain with no greater
fractional benefit. The two columns rank the strata differently and that
disagreement is the substance of 3.2:

    stratum                     n      seq MAE   abs gain     rel [95% CI]      dAUROC
    all held-out CpGs       26,570      0.1079    -0.0182    16.9%             +0.0166
    ---- CpG island context ----
    Island                   7,972      0.0793    -0.0146    18.4%            +0.0178
    Shore                    6,713      0.1333    -0.0230    17.3%            +0.0281
    Shelf                    2,217      0.0979    -0.0123    12.6%            +0.0275
    Open sea                 9,668      0.1161    -0.0192    16.5%            +0.0399
    ---- top vs bottom quartile, selected tracks ----
    H3K27me3  Q1 low         6,643      0.0861    -0.0206    23.9% [22.2,25.4] +0.0111
    H3K27me3  Q4 high        6,643      0.1321    -0.0136    10.3% [ 9.0,11.5] +0.0171
    H3K27ac   Q4 high        6,643      0.0767    -0.0175    22.9% [20.7,25.0] +0.0107
    ATAC      Q4 high        6,643      0.0802    -0.0180    22.4% [20.6,24.1] +0.0152
    ATAC      Q1 low         6,642      0.1104    -0.0167    15.1%            +0.0292
    H3K4me3   Q4 high        6,643      0.0583    -0.0131    22.4%            +0.0264
    H3K4me1   Q4 high        6,643      0.1424    -0.0241    16.9%            +0.0229
    H3K4me1   Q1 low         6,642      0.0853    -0.0173    20.3%            +0.0056

#### The mentor's two specific questions, answered

**Shores — yes on absolute, no on relative, and the distinction matters.** Shores
carry the largest absolute gain of the four island classes (-0.0230). But shores
also carry the largest sequence-only error (0.1333, vs 0.0793 at islands), and in
fractional terms they are unremarkable: 17.3%, below islands at 18.4%. Write the
shore result as *shores are where the most absolute error is removed*, and do not
let that slide into *shores are where context matters most*. It is not supported.

**High ATAC and high H3K27ac — yes, and this is the clean version of the claim.**
Top-quartile ATAC (22.4%) and top-quartile H3K27ac (22.9%) sit at the top of
their own ladders, each with an interval clear of the quartile below. The mentor's
hypothesis holds, but only in the relative column; in absolute terms both are
middling (-0.0180, -0.0175), which is why the existing
`fusion_gain_by_context.csv` looked flat across ATAC quartiles and concealed this.

**The counter-current worth reporting.** ΔAUROC runs the *opposite* way in those
same strata: ATAC Q4 high is the weakest AUROC gain of its ladder (+0.0152 vs
+0.0292 at Q1), and H3K27ac Q4 high is the weakest of its (+0.0107 vs +0.0243 at
Q3). Open sea has the largest AUROC gain of any island class (+0.0399) while
being middling on beta MAE. Open, accessible chromatin is where context sharpens
the *value* the most and the *call* the least — sequence alone already gets the
binary call right there. Both directions are significant; report both.

#### Against Task D — a partial parallel, and it must not be forced

Task D found transfer failure concentrating at **shores** (log2 enrichment
+0.525), **H3K4me1** (rank-biserial +0.219) and **H3K27me3** (+0.122), with
islands depleted (-0.556). Testing each against the fusion gain:

    Task D: transfer fails at ...   fusion gain there                  pairing
    Shore                           largest absolute of 4 classes       HOLDS
    H3K4me1 high                    -0.0241, largest of all 33 strata   HOLDS
    H3K27me3 high                   10.3% relative, SMALLEST of all 33  INVERTS

Two of three hold and the third inverts cleanly. The polycomb inversion is
monotone across the whole ladder — H3K27me3 Q1 23.9% [22.2, 25.4] down to Q4
10.3% [9.0, 11.5], non-overlapping, a 2.3x gradient — so it is not noise and not
a boundary artefact.

**So the coherent-pairing sentence is available for shores and enhancer
chromatin, and is false for polycomb.** The honest framing: context rescues
sequence exactly where methylation is tissue-plastic *in the enhancer direction*
(H3K4me1, shores), and rescues it least at polycomb-marked CpGs, which are the
regions where transfer fails AND context does not help. Those are simply hard for
both arms — neither the sequence window nor the seven tracks carry what is needed.
That is a more interesting sentence than the forced parallel, and it is what the
data says. Do not write "context helps most exactly where sequence alone
transfers worst" without the polycomb exception attached.

#### Seed sensitivity

Seed 42 alone (the standing single-seed constraint) vs the 42/43/44 ensemble:
relative reductions shift by at most 0.031 and uniformly *upward* (the ensemble
helps the sequence arm slightly more), ΔAUROC by at most 0.015, and the stratum
ranking is preserved (Spearman 0.94). Every conclusion above is seed-robust.
Ensemble is reported because it matches the frozen `22` artifact; `--seeds 42`
reproduces the single-seed version.

### G. Results section structure — the mentor's proposal, and the mapping

Recorded 15 Sep 2026. This is the **target structure for `main_revised.tex`**.
`main_revised.tex` is NOT restructured yet — that happens in the one wording pass
(directive 2). This subsection exists so the pass knows where everything lands.

#### The mentor's proposal, verbatim

> 3.1 Multimodal integration improves CpG methylation prediction across genomic partitions
> Include the original sequence-only/context-only/fusion comparison together with the additional chromosome-block splits.
>
> 3.2 Epigenomic context provides genomic-region-specific predictive gains
> Show where fusion contributes most, including CpG shores and regions with high ATAC-seq or H3K27ac signal.
>
> 3.3 Single-nucleotide perturbation reveals sequence-driven methylation responses
> Introduce the WT–MUT framework and include the context-swapping analysis here.
>
> 3.4 Independent validation of variant-associated methylation effects
> Include the current mQTL validation and, if successful, the ASM validation. The Melody comparison could be presented as a secondary benchmark or moved to the Supplementary Results.
>
> 3.5 Variant-associated methylation responses are organized by genomic context and CpG proximity
> Include promoter/TSS versus UTR versus gene-body differences, variant-to-CpG distance, and related regulatory-context analyses.
>
> 3.6 Biological applications of SilentMethyl
> Use NCOA2 as the synonymous-variant prioritization example and STK11 as an example showing that the framework can also be applied to clinically relevant nonsynonymous variants. If the eQTL analysis produces useful results, it could also be integrated into these case studies.

#### Why this structure is safe to adopt

**3.3 is the paper's central finding, and he got there independently.** He wrote
"single-nucleotide perturbation reveals sequence-driven methylation responses"
without having seen Task A or Task B. That is our dissociation claim — variant
effects are sequence-driven, context sets the baseline — arrived at from the
analyses he *had* seen. The framing risk flagged earlier is gone: the dissociation
story fits this structure rather than fighting it.

#### Mapping

    3.1  Multimodal integration ... across genomic partitions
         R1     sequence / context / fusion + published baselines
                scripts 10,11,12,13,14,15,16
                results/journal/{seed42,seed43,seed44}, paired_model_bootstrap,
                published_baselines, sequence_baselines
         NEW    chromosome-block splits, script 17 -> results/journal/folds/
                (fold1..3, whole-chromosome blocked, probe-disjointness asserted)
         also   R1's honesty constraint: 0.017 beta MAE is NOT "substantial";
                see section 1 R1. Carry that wording into 3.1.

    3.2  Epigenomic context provides genomic-region-specific gains
         F      NEW, script 57 -> ablation_breast_epithelium/fusion_gain_stratified/
                all seven tracks + paired AUROC, 1 Mb block bootstrap.
                THIS IS THE ANALYSIS 3.2 WAS MISSING.
         R2a    script 22 -> biological_context/fusion_gain_by_context.csv
                (genomic-region rows: promoter/UTR/gene body/intergenic)
         caveat report the RELATIVE column when comparing strata; the shore
                result is absolute-only and must not be written as "context
                matters most at shores". See F above.

    3.3  Single-nucleotide perturbation reveals sequence-driven responses
         R2     the dissociation itself — scripts 20,21
                context permutation (identity/shuffle/median), script 23
         B      four-rung context ladder, 46007255 COMPLETED 0:0 (4:53:44),
                -> ablation_breast_epithelium/context_ladder/
                This is the "context-swapping analysis" he asks for.
         A      gate decomposition + sub-channel split, script 54
                -> gate_decomposition/. Sequence self-rescaling is 82-85% of the
                gate channel's variance. The allele-invariance claim is WRONG as
                originally written; 3.3 must carry the corrected version.
         C      gate share vs measured tissue plasticity, script 55 — NULL.
                Report as a null or drop to Supplementary; do not oversell.
         hard   the R2 constraint stands: we may NOT claim epigenomic context
                improves variant-effect prediction anywhere in the paper.

    3.4  Independent validation of variant-associated effects
         R7     GENOA + eGTEx replication, cross-cohort/ancestry/platform
                rho_meta 0.1775 [0.0678, 0.2829], I2 = 0
         R2b    mQTL positive control + distance-matched negative, scripts 70,71
         E      ASM validation — REOPENED, see E (REOPENED) above. Not yet run.
                If it lands it belongs here; if it does not, 3.4 stands without it.
         Melody head-to-head, scripts 33,34,35 -> secondary benchmark or
                Supplementary, per his suggestion. Section 2 of these notes has
                the full positioning and must not be flattened into one line —
                the Fig 3H tension and the ovary gift both need to survive.

    3.5  Responses organized by genomic context and CpG proximity
         R2c    script 22 -> variant_response_by_context.csv (promoter/TSS, UTR,
                gene body, intergenic) and variant_response_by_distance.csv
                (0-50, 51-100, 101-250, 251-500 bp)
         R5     motif disruption with k-mer-matched null, script 50
                RC-disagreement as calibrated uncertainty, script 51
                GWAS regulatory enrichment (two powered tests only), script 52

    3.6  Biological applications
         R6     scripts 60-64 -> candidates/, known_variant_application/,
                literature_variant_screen/
                NCOA2 = synonymous prioritisation; STK11 = nonsynonymous.
                Both retained per mentor instruction.
         eQTL   not run. Only integrate if it produces something.

#### The one real gap in his structure — R3, R4 and Task D have no home

He proposed 3.1-3.6 without having seen Task D, and the structure has **no slot
for cross-tissue transfer**. That is not a small omission:

- **R3** — zero-shot transfer across nine tissues, and its reframing: single-tissue
  methylation models transfer, and the training tissue matters far less than
  assumed, demonstrated in two architectures (ours and Melody-ST) sharing nothing
  but the task. Distance-matched AUROC excludes 0.5 in 7/9 tissues.
- **R4** — can the model separate tissue-specific from shared mQTLs? Answer NO
  (§4). Winner's curse, not mechanism. A recorded, pre-specified falsification
  test that failed, which we report.
- **D** — where transfer fails: island shores with H3K4me1 and H3K27me3, islands
  with promoter chromatin succeed. These notes call it "the closest thing to a
  biological finding in the paper", and F above now pairs against it.

**Recommendation, not a decision.** Widen 3.1 to "across genomic partitions **and
tissues**" and let it carry R3, or add a section between 3.3 and 3.4 for transfer.
The second is cleaner: transfer is a generalisation result, not a benchmarking
result, and burying it in 3.1 next to the baseline table will lose it. Task D
then sits with R3, and F (3.2) references it across the section boundary.

**Raise this with the mentor rather than silently filing R3/R4/D somewhere.** He
has not seen Task D, and where transfer lands changes what 3.1 and 3.4 each claim.

### H. Figures and supplement for R8 — DONE 15 Sep 2026, zero GPU

The R8 analyses had **no figures and no supplement entries at all**. Both now
exist. Neither of the frozen builders (91, 90) was modified.

#### Figures — `scripts/92_build_r8_figures.py` -> `results/journal/manuscript_figures_r8/`

    fig_r8_1_fusion_gain_relative.png   Task F   -> Results 3.2
    fig_r8_2_context_ladder.png         Task B   -> Results 3.3
    fig_r8_3_transfer_vs_gain.png       D x F    -> Results 3.2/3.3 boundary
    fig_r8_4_asm_discrimination.png     Task E1  -> Results 3.4  [awaits 46096838]
    fig_r8_5_gate_decomposition.png     Task A   -> Results 3.3

**Why a new script rather than extending 91.** Script 91 ends with
`keep_manuscript_outputs()`, which **deletes every file in its output directory
that is not on its own hard-coded whitelist**. Extending it would mean editing
that whitelist and re-running a frozen builder across published outputs. Script
92 writes to its own directory; 91 is untouched.

**An earlier claim in this session was wrong and is withdrawn.** I said the
manuscript figure `fusion_gain_by_epigenomic_context.png` was stale because it
was built on MCF-10A. The published `results/journal/manuscript_figures/` copy
is indeed MCF-10A, but a correct breast-epithelium rebuild already exists at
`results/journal/ablation_breast_epithelium/manuscript_figures/` (13 Sep). The
real gap was never a stale figure; it was that **R8 had no figures**.

**Two drawing decisions, both made to stop a figure overstating the result:**

1. **Figure 1 leads with RELATIVE gain, not absolute.** Absolute gain is bounded
   by the sequence-only error in each stratum, so plotting it ranks strata by how
   badly they start. On the absolute measure shores look like where context
   matters most; on the relative one they are 10th of 32 and islands beat them.
2. **Figure 3 does not force the D-F pairing.** It draws both measures for the
   three strata Task D flagged, with each stratum's rank out of 32:

        stratum          absolute gain    relative gain
        Shore              rank  3/32       rank 10/32
        H3K4me1 high       rank  1/32       rank 12/32
        H3K27me3 high      rank 30/32       rank 32/32

   So the pairing holds for 2 of 3 on absolute and for **none** on relative, and
   H3K27me3 inverts on both. The figure says so in its own panel titles. The
   sentence "context helps most exactly where sequence alone transfers worst" is
   defensible only on the absolute measure and only with the polycomb exception
   attached — and it should probably not be written at all.

#### Supplement — `scripts/93_build_r8_supplement.py` -> `results/supplementary_package_r8/`

Sections **S7-S11**, continuing the published S1-S6 numbering. Same layout as
script 90 (numbered section folders, README naming every file, SHA256SUMS.txt),
built into a separate directory so script 90 and the published sections are not
touched. Merge the two at submission.

    S7   gate decomposition + sub-channel split        1 file
    S8   context ladder + effect-size stratification   6 files
    S9   transfer failure                              2 files
    S10  fusion gain by region                         3 files
    S11  ASM validation                                2 files, PARTIAL

**S8 deliberately ships the artefact table.** `ladder_effect_size_stratification
_predicted.csv` is the invalid predicted-delta binning, included and named
`_ARTEFACT` so the regression-to-the-mean trap is on the record rather than
rediscovered. The README says it is not a result.

**S11 is partial until `46096838` lands** — three of its five files do not exist
yet. The builder reports missing inputs instead of failing, and the README states
which sections are incomplete, so a partial package cannot be mistaken for a
finished one. **Rerun both 92 and 93 after the job completes.**

#### What is NOT done, and is not figures or supplement

- **E2** — signed statistics. Not buildable from the atlas; CanASM still 504.
- **`23_context_permutation.py`'s verdict banner** still tests a Pearson
  inequality this file already documents as the wrong comparison. Its outputs are
  correct, so this is a recorded lead, not a defect. Not fixed: scope.
- **Everything else remaining is prose** plus the mentor's decision on where
  R3/R4/D sit in the 3.1-3.6 structure (subsection G).

### I. ASM validation moves to Do & Tycko 2020 — E1 + E2 from ONE catalogue

Recorded 15 Sep 2026, **before the scoring job was submitted**. The prediction
below is pre-specified deliberately, following the §4 precedent for R4: E2 has a
real chance of landing near the null, and writing the expectation down first is
what makes either outcome reportable rather than rationalised afterwards.

#### Why the source changed

The Nat Commun 2025 atlas (subsection E, REOPENED) publishes ASM **significance**
but no signed allelic difference, which capped that analysis at discrimination
and made direction concordance and signed Spearman impossible. I reported those
two as impossible. **That was wrong** — not impossible, just unavailable from the
catalogue I had picked.

**Do C, Dumont ELP, Salas M, ... Tycko B. Genome Biology 21:153 (2020)**,
doi 10.1186/s13059-020-02059-3, Additional file 3 "Table S2" publishes, per ASM
index SNP:

    snp avg avg meth read ref    % methylation on REF-allele reads
    snp avg avg meth read alt    % methylation on ALT-allele reads
    snp avg diff alt ref         signed ALT - REF, percentage points

Keyed REF->ALT, the same convention as the model's MUT-minus-WT delta. **All
three statistics the mentor named are therefore computable from this one
catalogue**, which is why it is now the sole ASM source. 17,931 ASM index SNPs
over 15,112 DMRs, 13 tissues including `mammary`.

#### Build, and the checks that matter

`scripts/53d_tycko_build.py` ->
`data/external/asm_tycko_gb2020/scoring/`. Provenance in that directory's
`SOURCE.txt`.

    Table S2 rows, all chromosomes              17,931
    chr8+chr9, resolved to GRCh38 via Ensembl    1,831
    multi-allelic, EXCLUDED                      1,109
    biallelic analysis set                         722   all FDR < 0.05
      observed sign split                    328 pos / 393 neg
      annotated mammary                             53
      median |observed effect|                    45.8 pp
    DMR liftOver hg19->hg38                        722 kept, 0 unmapped
    ASM CpG pairs (positives)                    9,822   median ~13 CpGs / SNP
    distance-matched negatives                   5,373
    unique CpGs needing context                 12,999

**Two traps, both handled explicitly:**

1. **Mixed coordinate builds.** `dmr` is hg19; SNP positions are absent from the
   table entirely (rsID only) and come from Ensembl REST, which returns
   **GRCh38**. So regions need lifting and positions must NOT be lifted. Verified
   on rs67165842: DMR reads 1:100015940, GRCh38 is 99550369. Reversing this would
   mis-place every variant while still "working".
2. **Sign convention.** If a site's alleles are swapped relative to hg38, the
   sign of `diff alt ref` must flip with them, and a silent error there inverts
   the whole direction result. The build re-derives REF from hg38 and negates the
   effect when needed. **It fired zero times: all 722 REF alleles agree with
   hg38** (`ref_matches_hg38: 722`, `alleles_swapped_sign_flipped: 0`,
   `ref_matches_neither_dropped: 0`). The convention is confirmed, not assumed.

**Multi-allelic exclusion is deliberate and costly** — 1,109 of 1,831 sites.
Ensembl gives REF plus several ALTs and Table S2 does not say which ALT its
methylation refers to. Guessing (e.g. taking the minor allele) would score a
different substitution than the one measured and could invert the sign. 722 is
sufficient for all three statistics, so the conservative exclusion was taken.

#### Unit of analysis for E2 — do not get this wrong

The published effect is **per SNP, averaged over the CpGs in its ASM DMR**. So
the prediction is aggregated the same way: the mean predicted delta over that
SNP's scored DMR CpGs. Comparing one CpG's prediction against a DMR-wide
measurement is a unit mismatch and would understate agreement.

#### PREDICTION, recorded before the run

Stated so the result cannot be reinterpreted to fit afterwards.

- **Direction concordance: 0.60-0.70, clearly above 0.5.** The mQTL work already
  shows the model gets variant-effect direction right well above chance in two
  cohorts, and these are large effects (median 45.8 pp) which should be the
  easiest to call. Below 0.55 would be a genuine surprise and a real limit on the
  paper's central claim.
- **Signed Spearman: +0.15 to +0.35.** Bounded above by the scale mismatch and by
  the DMR-averaging; GENOA's rho_meta was 0.1775 on a cleaner comparison.
- **AUROC positive vs negative: 0.62-0.72**, tracking direction concordance.
- **Sequence arm >= fusion arm**, as in every previous variant-effect comparison
  and as the R2 constraint requires.
- **Mammary (n=53) will not be separately significant.** Reported as a stratum
  only; do not lead with it.

**If direction concordance lands at or below ~0.55, that is a result and gets
reported as one.** It would mean the model ranks *which* CpGs are
allele-specifically methylated without predicting *which allele* carries the
methylation — a real limit on "variant effects are sequence-encoded", and far
better found by us now than by a reviewer.

#### What happens to the Nat Commun E1 result

**Not deleted, and not yet demoted.** `results/journal/asm_validation/` stands.
If the Tycko discrimination lands near the atlas's 0.5625/0.5736 that is
independent replication across two catalogues built by different groups from
different data, which is worth more than either alone. If they diverge, that
needs explaining before either goes in the paper. **Decide after the number
exists**, not before.

#### Job

`jobs/r8_journal/53d_tycko_score.sbatch` — 15,195 pairs x 2 arms, seed 42, FP32,
~30 min estimated. Evaluation is `scripts/53e_tycko_evaluate.py`, in the same job.

**Nothing may be written under `results/journal/` while it is in flight.** That
is what failed `46096838` on the clobber guard with the science already correct,
and `45997644` before it. The sbatch carries the warning in its header.

### J. Tycko ASM validation — E1 + E2 RESULT. `46109189` COMPLETED. 16 Sep 2026

The job the §I prediction was written for. **It ran clean end-to-end**, unlike
the two before it: `COMPLETED`, `00:29:52`, exit `0:0`, and the run's own guard
logged `clobber check clean: nothing outside results/journal/asm_validation_tycko
was written`. `grep "PUBLISHED-PATH CLOBBER"` on the `.err` returns 0 hits. 53e
also ran inside the job and did **not** seize on the input load as its CPU smoke
test had — no rerun was needed. All outputs in
`results/journal/asm_validation_tycko/`; `tycko_pair_scores.csv`
(sha `bfe5b0bd…`) is the input to the evaluation summary.

    15,195 pairs x 2 arms scored, 0 dropped
    scoreable 15,195 / reference_base_mismatch 0 / outside_model_window 0
    seed 42, FP32, V100, fusion = breast_epithelium/seed42, sequence = journal/seed42

#### E2 — the three statistics the mentor named, scored against §I

Unit is one ASM index SNP, prediction = mean predicted delta over that SNP's
scored DMR CpGs, as §I required. n = 722 SNPs, 328 observed-positive. CIs are
2,000-replicate bootstrap over 179 1-Mb genomic blocks.

    statistic                    predicted (§I)   fusion    sequence   verdict
    -- signed agreement: SAME 722 SNPs, same checkpoints, NOT independent --
    direction concordance        0.60 - 0.70      0.5900    0.5914     MISSED, low
                                                  [0.550,   [0.554,
                                                   0.630]    0.629]
    directional AUROC            0.62 - 0.72      0.6277    0.6308     HIT
      (§I called this "AUROC                      [0.584,   [0.588,
       positive vs negative")                      0.669]    0.671]
    signed Spearman              +0.15 - +0.35    0.2410    0.2503     HIT
                                                  [0.180,   [0.193,
                                                   0.299]    0.306]
    -- structurally independent of the three above --
    sequence arm >= fusion arm   yes              — sequence higher on all 3 —  HIT
    mammary (n=53) not sep. sig. yes              p 0.152-0.196, CIs span null  HIT

**Do NOT write this up as "four of five independent predictions hit."** That is
technically true and will not survive a careful reader. Three of the five are
signed-agreement statistics computed on the same 722 SNPs from the same two
checkpoints, so they cannot confirm one another:

- **direction concordance** — `sign(pred) x sign(obs)`, discards both magnitudes;
- **directional AUROC** — `predicted magnitude x sign(obs)`: the *same label* as
  direction concordance, differing only in that the predicted margin is kept;
- **signed Spearman** — `rank(pred) x rank(obs)`, keeps both.

Direction concordance and the directional AUROC are the tightest pair — **one
quantity measured two ways.** Spearman is a third view of the same signed
agreement, not independent corroboration of it. The two genuinely independent
predictions are the arm ordering and the mammary power call.

The defensible framing, and the one to use: **two independent predictions, plus
one quantity measured three ways.**

#### Why the twin split — direction missed, its continuous version hit

State this rather than leave it to be discovered. Direction concordance landed
0.01 *below* its band while the directional AUROC landed *inside* its own, and a
reader who notices the two are the same quantity will ask how.

The mechanism: **a sign error costs direction concordance a full count however
small the margin, but costs the AUROC almost nothing when the misranked SNP sits
near the middle of the predicted ordering.** The two therefore diverge exactly
when sign errors concentrate on SNPs the model predicts weakly. The split is not
an inconsistency — it says the model gets *marginal* effect directions wrong more
often than large ones, which is the expected failure mode and the same
dose-dependence subsection B found.

**Half of that is evidenced and half is structural — do not blur them.**
Restricting to `|observed effect| >= 20 pp` raises direction concordance from
0.590 to 0.6055 (fusion) and 0.591 to 0.6125 (sequence), into the predicted band:
direct evidence that sign errors concentrate on small *measured* effects. The
*predicted*-margin half of the mechanism was never measured, and measuring it
would be a new analysis. Write the evidenced half, and label the stratum post-hoc.

**The miss is a miss and is not re-banded.** 0.590/0.591 against a pre-specified
floor of 0.60 is short by about a point, not a collapse. §I set 0.55 as the
threshold below which the result would be "a real limit on the paper's central
claim"; the observed value is above that, and its bootstrap CI excludes 0.5
(`P_LE_Half = 0.0`, CI low 0.550 fusion / 0.554 sequence). The model calls the
methylated allele better than chance, reliably, but less well than the mQTL
cohorts led us to predict. The honest sentence is **"above chance and below what
we expected"**, not "as predicted".

The sequence-arm ordering held on all three statistics — but by margins of
0.0014, 0.0093 and 0.0031, far too small to be a separation, and across coupled
statistics, so it is **one** observation and not three. Report it as "sequence is
not worse", consistent with R2, not as a sequence advantage.

**Mammary, n = 53, is a stratum and never the headline.** Direction 0.5660 both
arms, every CI spanning the null (direction p = 0.152 fusion / 0.196 sequence).
This is exactly the pre-specified expectation — underpowered, reported, not led
with.

**One stratum was NOT pre-registered and is flagged as such**: restricting to
`|observed effect| >= 20 pp` (n = 578) raises direction to 0.6055/0.6125,
Spearman to 0.2575/0.2670, directional AUROC to 0.6516/0.6566. Every statistic moves the
right way with effect size, which is the same dose-dependence subsection B found.
It supports the B wording, but it is post-hoc and must be labelled post-hoc.

**Magnitudes are not compared and no calibration is claimed.** Observed effects
are percentage points of methylation; predicted deltas are on the model's M
scale. Only sign, rank and AUROC are used — this is recorded in the summary
JSON's `scale_note` as well.

#### E1 — does Tycko's discrimination reproduce the Rosenski atlas?

    catalogue                    contrast                       fusion   sequence
    Rosenski/Dor/Kaplan 2025     pos vs bimodal non-ASM         0.5625   0.5736
      (n=6,910, 242 blocks)                             CI   [.533,.593] [.545,.601]
    Do & Tycko 2020              pos vs distance-matched non-DMR 0.5419  0.5432
      (n=10,746, 179 blocks)                            CI   [.508,.580] [.502,.585]
    distance-only baseline       Tycko 0.5030 [.480,.531] p=0.39
                                 Rosenski 0.5017 [.478,.524] p=0.45

**It reproduces. Both catalogues stay.** The Rosenski point estimates (0.5625,
0.5736) sit inside the Tycko confidence intervals, and the Tycko estimates sit
inside the Rosenski intervals; both exclude 0.5 (Tycko p = 0.008 fusion / 0.017
sequence). The distance-only baseline is null in both, so neither result is
carried by how the negatives were positioned. Two catalogues, built by different
groups from different sequencing data with different ASM tests, landing on
0.54-0.57 with sequence >= fusion in both, is **independent replication**, and
that is worth more than either number alone.

The Tycko estimate is the lower of the two, by ~0.02-0.03. That is well inside
overlapping CIs and does not need explaining away, but the likely reason is the
negatives: Rosenski's headline contrast uses bimodal non-ASM CpGs — sites that
look methylation-variable but are not allele-specific — while Tycko's are
distance-matched non-DMR CpGs. Neither is the harder control in an obvious
direction, and we should not claim one is.

**No mentor decision is needed on retiring Rosenski.** §I made that conditional
on divergence; there is none.

#### What this changes for the write-up

1. **E2 is no longer a limitation.** §I's closing bullet in the "what the wording
   pass must fold in" list says "E2 stated as a limitation" — that is now stale.
   E2 exists, with all three statistics, and must be written as a result.
2. **E1's number in the R8 table stays as the Rosenski figure**, now with Tycko
   as independent replication alongside it.
3. **The direction-concordance miss goes in the paper as a miss.** 0.59 with a CI
   of [0.55, 0.63] is a real, modest, above-chance result, and the pre-registered
   0.60-0.70 band is what makes it reportable in that form.
4. Out-of-distribution caveat carries over unchanged: the model was trained only
   at HM450 positions, and every CpG scored here is an arbitrary genomic CpG.

### Standing directives for the rest of R8 (14 Sep 2026)

1. **Scope is FROZEN**: A + sub-channel split, B, C, D — and, added 15 Sep 2026
   on mentor feedback, **E (reopened)** and **F (fusion-gain stratification)**.
   Both are justified rather than scope creep: E was dropped on our own
   arithmetic error, and F is the analysis Results 3.2 requires and lacked.
   **After these, scope is frozen again.** If B, D or F turns up something
   interesting, record it as a lead here and **do not chase it**.
2. **ONE wording pass, at the very end**, after the split, B and D have all
   landed. Not after the split. B's rung 4 can touch the same R2 sentences and two
   passes is exactly the churn being avoided.
3. **Wording for the partial-correlation result**: write it as *consistent in sign
   across two independent cohorts, small in magnitude, and in eGTEx only
   marginally separable from zero* (CI [+0.0017, +0.0168]). Do **not** write
   "confirmed in both cohorts" — a reviewer reading a +0.0017 lower bound against
   that phrasing will not be generous.
4. **E** — REVISED 15 Sep 2026, the original wording contained the error.
   The old text said "after intersecting with test probes", which is what caused
   the 93x power miscalculation. **The restriction is chr8+chr9, not HM450.**
   Catalogue check done (E REOPENED above): usable n clears the ~500 floor by
   roughly an order of magnitude even under pessimistic assumptions.
   **Still zero compute, still nothing built.** The scoring run needs explicit
   authorisation, and CanASM's server must come back up first.
5. `main.tex` is never touched. Corrections go to `main_revised.tex` only.
6. **`main_revised.tex` is not restructured yet.** Subsection G records the target
   3.1–3.6 structure and the mapping; the restructure happens inside the single
   wording pass, not before it.
7. **Single seed** remains the standing constraint for anything new. Where an
   existing frozen artifact is seed-ensembled (script 22, and F which must
   reproduce it), match the artifact and report the single-seed sensitivity
   alongside — F does this.

### Where this stands — 15 Sep 2026, B analysed

**`46007255` landed: COMPLETED 0:0, elapsed 4:53:44.** Task B is done and all
four rungs wrote. That was the last GPU compute in the project as planned; the
only compute that could follow is E's scoring run, which is not authorised yet.

| task | state |
|---|---|
| A. gate decomposition | DONE, `45992001` COMPLETED 0:0, committed |
| A. sub-channel split | DONE, `45997644` (FAILED 1:0 on the guard only — outputs correct, analysed, committed) |
| B. context ladder | **DONE and ANALYSED** 15 Sep 2026 — dissociation holds, verdict banner is a Pearson artefact |
| C. gate plasticity | DONE, null, zero GPU |
| D. transfer failure | DONE, hypothesis holds, zero GPU |
| E1. ASM discrimination | **DONE, AND REPLICATED** — Rosenski `46096838` 0.5625 fusion / 0.5736 sequence; Do & Tycko `46109189` 0.5419 / 0.5432. Mutually inside CIs, both exclude 0.5, both baselines null (subsection J) |
| E2. ASM signed statistics | **DONE** `46109189` COMPLETED 0:0, 16 Sep 2026. n=722 SNPs. Direction 0.590/0.591 (pre-registered 0.60-0.70 — MISSED low, still excludes 0.5), signed Spearman 0.241/0.250 HIT, directional AUROC 0.628/0.631 HIT — one signed agreement measured three ways, NOT independent (subsection J) |
| F. fusion gain by region | **DONE** 15 Sep 2026, zero GPU, script 57 |

Budget: ~22.9 GPU h spent of 125 (17.5 + B's 4.9 + E's 0.5).

**All compute is finished. `46109189` was the last job — it ran clean, and it is the
only R8 job that neither tripped the clobber guard nor needed a rerun.**

**What the wording pass must now fold in.** Every input has landed:

1. the sub-channel split's sequence-dominance result (A);
2. B's four-rung ladder, **and the resolution of the allele-invariance banner**:
   it fires at full scale but is a Pearson artefact. R2's current "on normalised
   error, Spearman and sign agreement" phrasing is now false and MUST be
   rewritten; the supported claim is dose-dependent and strengthens with observed
   effect size. Subsection B has the exact wording to use and to avoid;
3. D's transfer-failure localisation;
4. **F's fusion-gain stratification**, including the absolute-vs-relative
   distinction and the polycomb inversion against D (subsection F);
5. **the 3.1–3.6 restructure** (subsection G), including the R3/R4/D gap;
6. **the ASM result, E1 AND E2** (subsection J, superseding the E-REOPENED
   framing). E1 is modest, ~0.54-0.57, and now **replicated across two
   independent catalogues**. E2 is **no longer a limitation** — it exists, with
   all three statistics the mentor named. Write the direction-concordance miss
   (0.59 against a pre-registered 0.60-0.70) as a miss.

**B is read in and R2's required change is now known** (subsection B). The
wording pass is no longer blocked on compute — every input has landed. Its
remaining dependency is a decision from the mentor on where R3/R4/D sit in the
3.1-3.6 structure (subsection G).

No further jobs are to be submitted without authorisation. **There is no code left
to write and no job left to run.** Everything remaining is prose: the single
wording pass, whose one open dependency is the mentor's decision on where R3/R4/D
sit in the 3.1-3.6 structure (subsection G).
