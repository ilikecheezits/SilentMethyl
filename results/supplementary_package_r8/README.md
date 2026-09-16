# SilentMethyl supplementary data, sections S7-S11

Companion to the published S1-S6 package built by
`scripts/90_build_supplement_package.py`. These sections cover the R8
analyses. Merge the two directories at submission.

Generated 2026-09-16T11:11:33.827686+00:00.

## S7 — Gate decomposition of the fusion model's variant effect (Task A)

Directory: `Supplementary_Data_S7_Gate_Decomposition/`

- **S7_gate_decomposition_summary.json** — Variance decomposition of the fusion variant effect into DNA and gate channels, both cohorts, with the allele-invariance diagnostics.

## S8 — Context dose-response ladder and effect-size stratification (Task B)

Directory: `Supplementary_Data_S8_Context_Ladder/`

- **S8_ladder_agreement_with_native_context.csv** — Agreement of each perturbation rung with the native-context run, for absolute methylation and for variant effects.
- **S8_ladder_level_accuracy.csv** — Held-out methylation accuracy under each context substitution.
- **S8_ladder_run_summary.json** — Provenance and guard counters for the four-rung ladder.
- **S8_ladder_by_observed_effect_size.csv** — Ladder dissociation stratified by OBSERVED effect size (beta_ref_to_alt), the bias-free stratifier.
- **S8_ladder_by_predicted_effect_size_ARTEFACT.csv** — The same stratification binned on the model's OWN predicted effect. Included only to document the regression-to-the-mean artefact it produces; it is NOT a result.
- **S8_ladder_stratification_summary.json** — Provenance for the effect-size stratification.

## S9 — Where zero-shot cross-tissue transfer fails (Task D)

Directory: `Supplementary_Data_S9_Transfer_Failure/`

- **S9_transfer_failure_summary.json** — Error-versus-plasticity deciles, CpG-island enrichment and chromatin contrasts for the top error decile.
- **S9_transfer_failure_per_probe.csv** — Per-probe held-out errors and cross-tissue variance.

## S10 — Where the fusion gain concentrates, by genomic region (Task F)

Directory: `Supplementary_Data_S10_Fusion_Gain_By_Region/`

- **S10_fusion_gain_stratified.csv** — Paired fusion-minus-sequence beta MAE and AUROC by CpG-island context and by all seven chromatin tracks, with 1 Mb block-bootstrap intervals. Compare strata on Relative_Beta_MAE_Reduction, not the absolute column.
- **S10_fusion_gain_summary.json** — Provenance and the absolute-versus-relative caveat.
- **S10_fusion_gain_by_context_prior.csv** — The earlier two-track stratification, retained so the newer all-track table can be checked against it.

## S11 — Allele-specific methylation validation (Task E1)

Directory: `Supplementary_Data_S11_ASM_Validation/`

- **S11_asm_discrimination.csv** — AUROC separating ASM SNV-CpG pairs from distance-matched non-ASM pairs, per contrast, stratum and model arm, with block-bootstrap intervals.
- **S11_asm_matching_balance.csv** — Variant-to-CpG distance balance between matched positives and negatives.
- **S11_asm_evaluation_summary.json** — Provenance and the distance-baseline sanity check. This catalogue publishes ASM significance but no signed allelic difference, so direction concordance and signed Spearman are not computable here; they are in S12, from Do & Tycko 2020.
- **S11_asm_build_summary.json** — How the ASM pairs were constructed: liftOver counters, window arithmetic, positive/negative tier definitions and counts.
- **S11_asm_source_provenance.txt** — Citation, download URLs and contents of the source ASM catalogue.

## S12 — Allele-specific methylation, signed validation and replication (Tasks E1 + E2, Do & Tycko 2020)

Directory: `Supplementary_Data_S12_ASM_Signed_Validation/`

- **S12_asm_signed_agreement.csv** — Direction concordance, signed Spearman and AUROC_Directional per ASM index SNP (n=722), by stratum and model arm, with 1 Mb block-bootstrap intervals. The unit is one SNP: the prediction is the mean predicted delta over the CpGs scored inside that SNP's ASM DMR, matching how the published effect was averaged. AUROC_Directional is NOT case-versus-control: every SNP in it is an ASM SNP and the classes are the sign of the measured ALT-REF difference, so it is a continuous-margin restatement of direction concordance and is not independent of it. All three columns are one signed agreement measured three ways. The mammary stratum (n=53) is underpowered and is not a headline. The |effect| >= 20 pp stratum is post-hoc and was not pre-registered.
- **S12_asm_discrimination_replication.csv** — AUROC_Detection: discrimination of ASM CpGs from distance-matched non-DMR CpGs in the second, independent catalogue, with the distance-only null baseline. This asks which SITE is allele-specifically methylated and is a different question from S12's AUROC_Directional; the two numbers are not comparable to each other. Compare against S11: the two catalogues' estimates lie inside each other's confidence intervals.
- **S12_asm_signed_evaluation_summary.json** — Provenance, bootstrap settings, and the scale caveat: observed effects are percentage points of methylation and predicted deltas are on the model's M scale, so only sign, rank and AUROC are compared and no magnitude calibration is claimed.
- **S12_asm_score_summary.json** — Scoring provenance: checkpoints and their hashes, seed, and the input-validation counters (reference-base mismatches, off-window and non-CpG-centred pairs, all zero).
- **S12_asm_build_summary.json** — How the SNP-CpG pairs were built: the hg19 DMR liftOver, the GRCh38 SNP positions from Ensembl (deliberately NOT lifted), the multi-allelic exclusion, and the REF-versus-hg38 sign-convention check.
- **S12_asm_source_provenance.txt** — Citation, download URL and contents of the Do & Tycko 2020 catalogue.

## Two cautions carried from the analyses

1. **S10**: absolute fusion gain is bounded by the sequence-only error in
   each stratum, so it ranks strata by how badly they start. Compare
   strata on the relative reduction.
2. **S11**: the source catalogue publishes significance but no signed
   allelic methylation difference, so direction concordance and signed
   Spearman are not available and are not reported.

