# SilentMethyl — lab notes

Single running record for the project. Consolidates the former
`PAPER_FRAMEWORK.md`, `MELODY_COMPARISON.md`, `REQUIREMENTS.md` and the two
presentation outlines. `README.md` remains separate as the repository's entry
point.

Last updated 9 Sep 2026, after the repeated chromosome-blocked folds completed.

---

## 0. Status in one page

**Every analysis is finished. What remains is writing.**

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
| 9 | functional validation (ASM, ATAC, TFBS, eQTL, eQTM) | **the one real gap** — see §3 |

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

*REPEATED CHROMOSOME-BLOCKED FOLDS COMPLETE, 8 Sep 2026.* Four folds, each with
both towers and the gate retrained end to end on that fold's training
chromosomes. Test-set results:

    fold  test chromosomes        n        fusion b MAE  seq b MAE  d b MAE   d AUC
    0     chr8, chr9              26,570   0.0993        0.1099     -0.0106   +0.0111
    1     chr12, chr18            26,748   0.0942        0.1040     -0.0098   +0.0100
    2     chr20, chr4             26,783   0.0990        0.1064     -0.0074   +0.0093
    3     chr13, chr15, chr21     26,806   0.0938        0.1049     -0.0111   +0.0110

**The fusion gain reproduces on every split.** The published -0.0106 sits inside
the fold range (-0.0074 to -0.0111), so the original split was neither lucky nor
unlucky. Absolute performance varies more than the gain does (0.0938-0.0993),
which is the expected consequence of chromosomes differing in gene density; the
two most gene-poor folds gave the LOWEST error, so the model is not carried by
promoter-dense regions.

Held-out sizes are matched by construction (26,570-26,806). Sex chromosomes stay
in training for every fold and are never evaluated -- chrY is absent in female
donors and chrX carries X-inactivation, so a fold holding either out would not
measure the same quantity as the rest.
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
- **Allele-specific methylation** (slot 53), pooled TCGA baseline, BEND
  multi-task validation (manifest entry exists, data MISSING -- the Nature
  Machine Intelligence criterion, and out of scope for this paper).

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
| 3-seed paired bootstrap (ensemble) | −0.0104 | +0.0105 |

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

### 9. Joint multi-tissue model — the standing plan

Assume the mentor asks for this. Design is settled; nothing has been built.

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
- **Allele-specific methylation** (script slot 53), pooled TCGA baseline, BEND
  multi-task validation (manifest entry exists, data MISSING — the Nature
  Machine Intelligence criterion, out of scope for this paper).

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
