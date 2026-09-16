# SilentMethyl supplementary data, sections S7-S11

Companion to the published S1-S6 package built by
`scripts/90_build_supplement_package.py`. These sections cover the R8
analyses. Merge the two directories at submission.

Generated 2026-09-16T00:13:52.495520+00:00.

## INCOMPLETE PACKAGE

The following files were not present when this package was built, so
their sections are absent or partial. Rebuild once they exist:

- `results/journal/asm_validation/asm_discrimination.csv` (section S11)
- `results/journal/asm_validation/asm_matching_balance.csv` (section S11)
- `results/journal/asm_validation/evaluation_summary.json` (section S11)

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

- **S11_asm_build_summary.json** — How the ASM pairs were constructed: liftOver counters, window arithmetic, positive/negative tier definitions and counts.
- **S11_asm_source_provenance.txt** — Citation, download URLs and contents of the source ASM catalogue.

## Two cautions carried from the analyses

1. **S10**: absolute fusion gain is bounded by the sequence-only error in
   each stratum, so it ranks strata by how badly they start. Compare
   strata on the relative reduction.
2. **S11**: the source catalogue publishes significance but no signed
   allelic methylation difference, so direction concordance and signed
   Spearman are not available and are not reported.

