# Figure build: reconciliation against current exports

Written 19 September 2026 alongside `scripts/95_build_revision_figures.py`.

Every panel is drawn from the current primary-breast-epithelium tree
(`results/journal/ablation_breast_epithelium/`, plus the sequence-only arms the MCF-10A →
primary-epithelium swap did not touch, plus two reruns described in §3). The superseded
`results/journal/<analysis>/` copies are not used anywhere.

**The corrections in §1 and §2 have been applied** to `SilentMethyl_EXPANDED.tex` and
`supplementary_revision.tex`. `main.tex` was not touched. §5 lists what remains open.

## 0. Date check

The two trees are cleanly separated in time, which confirms which is current:

| Export | Superseded copy | Current copy |
|---|---|---|
| `context_permutation/agreement_with_identity.csv` | 5 Sep 2026 | **12 Sep 2026** |
| `literature_variant_screen/stk11_case_study_figure_values.csv` | 20 Aug 2026 | **13 Sep 2026** |
| `variant_effect_synthesis/all_strata.csv` | 28 Aug 2026 | **12 Sep 2026** |
| `genoa_variant_evaluation/fusion_vs_sequence_paired.csv` | 25 Aug 2026 | **12 Sep 2026** |
| GWAS enrichment | 30 Aug 2026 | **12 Sep 2026** (GENOA only) |

`repro_check/abl/` holds an independent 16 September rerun of much of the ablation chain. Where it
overlaps, it reproduces the published ABL values **bit for bit** — GWAS share 0.470968
(0.369177–0.556294), STK11 +0.089031 / −0.052045, the synthesis table and the paired-difference
table are all identical. The current values below are therefore confirmed twice.

## 1. Draft numbers corrected from the superseded tree

Each was traced by matching the draft's digits against both trees; in every case the draft's value
was an exact match for the superseded file.

### 1.1 Context perturbation — Fig. 4d and Supplementary Table S3

Normalised deviation = `mae / sd_reference`, n = 76,893 pairs, seed 42.

| Quantity | Was (superseded) | Now (current) |
|---|---:|---:|
| Shuffled context, levels | 0.3045 | **0.5260** |
| Shuffled context, effects | 0.1123 | **0.2266** |
| Median context, levels | 0.1740 | **0.3530** |
| Median context, effects | 0.0692 | **0.1860** |

Table S3's Spearman, sign-agreement and Pearson columns came from the same superseded file and were
replaced wholesale (e.g. shuffle/levels Spearman 0.8610 → **0.6180**, Pearson 0.9015 → **0.6324**).
The qualitative claim survives — levels still move about twice as much as effects under both
schemes — but the magnitudes roughly double, which makes the model look *more* context-sensitive,
not less.

### 1.2 STK11 case study — Fig. 5c

| Item | Was | Now |
|---|---:|---:|
| rs2145420809 / cg16601904 | +0.0774 | **+0.0890** |
| rs148928808 / cg08681293 | −0.0613 | **−0.0520** |
| Distance to cg08681293 | 204 bp | **205 bp** |
| Scored nonsynonymous pool | 322 | **321** |

The offsets come from `Signed_Offset_From_Target_CpG_C` (−1 and +205). The CpG-class annotations in
the draft are correct against the current export: cg16601904 is Shelf, cg08681293 is Shore, both on
the training split, both on NM_000455.5.

### 1.3 Cross-cohort agreement — Fig. 3a

| Cohort / endpoint | Was | Now |
|---|---|---|
| eGTEx signed ρ | 0.246 (0.114–0.402) | **0.236 (0.098–0.392)** |
| GENOA signed ρ | 0.152 (0.118–0.186) | **0.149 (0.112–0.185)** |
| eGTEx direction | 59.6% (54.7–65.2) | **59.6% (54.2–65.1)** |
| GENOA direction | 55.3% (53.6–57.1) | **55.6% (53.8–57.2)** |

### 1.4 Paired fusion-minus-sequence differences — Fig. 4a

| Endpoint | Was | Now (GENOA) |
|---|---|---|
| signed ρ | 0.0026 (−0.0020–0.0070) | **−0.0010 (−0.0078–0.0051)** |
| direction agreement | −0.0007 (−0.0064–0.0051) | **0.0017 (−0.0050–0.0079)** |
| distance-matched AUROC | −0.0021 (−0.0053–0.0011) | **−0.0023 (−0.0058–0.0010)** |

Two point estimates change sign, but every interval still spans zero, so "no clear added signal"
is unchanged and the panel keeps that title. The current eGTEx paired export is a separate and
larger set (+0.0117, +0.0048, +0.0057); the panel is labelled GENOA so the two are never blended.

### 1.5 Shared-cohort baseline comparison

| Contrast | Was | Now |
|---|---|---|
| fusion − DeepCpG, matched AUROC | 0.018 (0.004–0.032) | **0.019 (0.005–0.033)** |
| fusion − CpGenie, matched AUROC | 0.004 (−0.014–0.020) | **0.005 (−0.013–0.021)** |

### 1.6 Supplementary Table S1 — the M-value column

The draft's β and ROC-AUC columns were already current, but the $M$ MAE column for the context and
fusion rows was carried over from the pre-swap run (where context β was 0.1395, not 0.1022):

| Row | $M$ MAE was | now |
|---|---:|---:|
| Context | 1.4574 ± 0.0023 | **1.1763 ± 0.0005** |
| Fusion | 1.0971 ± 0.0192 | **1.0341 ± 0.0289** |

Sequence (1.1941 ± 0.0070) is identical in both trees, as expected: the sequence tower never reads
the context columns and was not retrained for the swap.

The table also **mixed two SD conventions**: the CpGenie and DeepCpG rows used population SD
(ddof = 0) while the sequence and fusion rows used sample SD (ddof = 1). The whole table is now
sample SD across seeds 42–44, and the caption says so. This is why CpGenie's displayed SDs moved
(0.0011 → 0.0013, 0.0084 → 0.0103) although its data did not.

## 2. Reruns performed, with authorisation, to replace unusable exports

Both were written to `repro_check/`, never into a published tree.

### 2.1 GWAS annotation control — Fig. 5d

Two problems, not one:

- The draft's GENOA figure **0.596 (0.417–0.752) matches no export** in the repository. Those digits
  are the eGTEx direction agreement from §1.3, so it looks like a transcription slip.
- More seriously, the published `ABL/gwas_regulatory_enrichment` run used `mqtl_significance: null`
  — every tested pair, roughly 71% of which have no measured methylation effect. Script 52's own
  help text calls that "the wrong population for a regulatory claim", and both the main text and
  the supplement describe the p < 0.05 **nominal** design instead. The only nominal runs in the
  repository scored from the pre-swap trees.

Both cohorts were therefore rescored against the current variant scores with
`--mqtl-significance 0.05`, into `repro_check/gwas_nominal_current/{egtex,genoa}/`. The build step
reproduced the draft's labelled-pair counts exactly (510 eGTEx / 423 variants / 455 probes; 573
GENOA / 435 variants / 533 probes), which confirms the design was read correctly — labelling
depends on variant identity, not on the model.

| Cohort | Was | Now (current context, nominal) |
|---|---|---|
| eGTEx top 5% | 0.490 (0.289–0.673) | **0.510 (0.271–0.679)**, n = 51 in tail |
| GENOA top 5% | 0.596 (0.417–0.752) | **0.561 (0.402–0.728)**, n = 57 in tail |

Both still span the matched null of 0.5, so the null result stands. Figure 5d now shows both
cohorts again rather than the GENOA-only version I built before the rerun.

### 2.2 Baseline variant evaluation — Fig. 3b

`results/journal/baseline_variant_evaluation/` contained **two** ensemble rows per model with
different values and block counts (k-mer 0.5075 / 245 blocks and 0.5038 / 246 blocks; composition
0.4629 / 246 and 0.4648 / 243), in both `matched_negative_auroc.csv` and `primary_metrics.csv`.
Script 21 emits one row per model-seed, so the file holds two runs. The draft's 0.5055
(0.4934–0.5172) matches neither.

Script 21 was rerun on `results/journal/sequence_baselines/variant_scoring` (seed −1, 2,000 block
bootstraps, seed 42 RNG) into `repro_check/baseline_variant_evaluation/`, giving one row per model:

| Model | Unmatched AUROC | Distance-matched AUROC |
|---|---|---|
| k-mer ridge | 0.5098 (0.4980–0.5219) | **0.5063 (0.4930–0.5191)** |
| composition | 0.4621 (0.4496–0.4739) | **0.4582 (0.4428–0.4734)** |

The draft's k-mer figure is now 0.5063 (0.4930–0.5191). Figure 3b reads the clean export.

## 3. Corrections to my own earlier reporting

- **The gate-channel partial Spearman intervals do exist.** I reported them as absent; that was
  wrong — my probe truncated the JSON key list before reaching
  `partial_spearman_gate_given_dna_95ci`. They are present in both the published and rerun
  summaries. The draft's intervals were nonetheless slightly stale, and are now corrected to
  **0.0152 (0.0066–0.0247)** in GENOA and **0.0091 (0.0017–0.0160)** in eGTEx. Figure 4c draws them.
- **Figure 3d used the wrong Rosenski stratum.** I had taken
  `positive_vs_background | all tissues` (28,144 rows, 261 blocks). The documented design, and the
  one Supplementary Table S2 reports, is `positive_vs_bimodal_non_asm | all tissues` (6,910 rows,
  242 blocks). The panel now uses it and matches Table S2 exactly: fusion 0.5625, sequence 0.5736,
  distance-only 0.5017 (0.478–0.524). Table S2's Do–Tycko row was already correct.
- **Figure 3b originally mixed export families.** It paired a shared-cohort fusion value with a
  per-seed sequence value. It is now built entirely from the ensemble family, so unmatched, matched
  and distance-only entries are mutually comparable. CpGenie and DeepCpG exist only in the
  shared-cohort export and are left to the text rather than mixed into the panel.

## 4. Checks that passed against current data, unchanged

Fold MAEs 0.0873 / 0.0934 / 0.0862; fusion three-seed mean β MAE 0.0914 and 16.8% reduction over
sequence; composition 0.1954, k-mer 0.1565, CpGenie 0.1281, DeepCpG 0.1253; relative gains 22.4%
high-ATAC and 22.9% high-H3K27ac, shores not the largest class, the high-H3K27me3 reversal;
gate-channel variance shares 0.187 / 0.195 and context-rescaling 0.0617 / 0.0613; marginal
DNA-channel ρ 0.0699; Do–Tycko signed ρ 0.2410 and direction 0.590 with its missed 0.60–0.70 band;
fusion unmatched 0.600, matched 0.556, distance-only 0.595 and its realized 0.5 after matching;
66 matched comparators; NCOA2 −0.1539 at 9 bp on the test split; Table S2 in full; the DeepCpG
27.1% arithmetic.

The fold-0 conflict flagged in FIGURE_GUIDE.md is not a conflict: seed 42 alone is 0.108981 and the
across-seed mean is 0.109897. Figure 2b is a seed-42 panel and uses the former; the Results text
quotes the latter and is right as written.

## 5. Still open

- **Bootstrap counts differ by analysis** — 500 draws in `variant_effect_synthesis` and the GWAS
  runs, 2,000 in `genoa_variant_evaluation`, `fusion_gain_stratified` and the ASM evaluations,
  10,000 variant-cluster draws in the eGTEx positive control, 5,000 match-set draws in the matched
  negative. No caption states a single count for a whole figure; each panel's count is in its
  source-data row.
- **The two reruns live in `repro_check/`.** If you want them to be the published record, they need
  installing into `results/journal/` and adding to `reproducibility/published_outputs_sha256.txt`,
  and the two commands belong in REPRODUCE.md §5.6 beside the existing GENOA GWAS call.
- **The duplicated rows in `results/journal/baseline_variant_evaluation/` are still there.** I did
  not overwrite the published file; the clean rerun sits beside it.
- **STK11 chromatin-quartile claims are unverified.** The draft says the two probes sit in the
  second/third ATAC and third/second H3K27ac quartiles. The raw signals are in the export (0.5071
  and 2.4026 ATAC; 1.3999 and 1.5012 H3K27ac) but the quartile boundaries depend on which
  population defines them, which the export does not record.
- **`main.tex` is unchanged**, so its copies of these numbers still carry the superseded values.

## 6. Panel sources

| Panel | Export |
|---|---|
| 2a | `ABL/paired_model_bootstrap/model_metrics_recomputed.csv`, `sequence_baselines/absolute_prediction_metrics.csv`, `published_baselines/{cpgenie,deepcpg}/seed*/metrics.json` |
| 2b | `ABL/fold{1,2,3}/fusion/metrics.json`, `folds/fold{1,2,3}/sequence/metrics.json`, seed-42 rows of 2a |
| 2c | `ABL/fusion_gain_stratified/fusion_gain_stratified.csv` |
| 2d | `joint/transfer_failure/transfer_failure_summary.json` and `_per_probe.csv` |
| 3a | `ABL/variant_effect_synthesis/all_strata.csv` |
| 3b | `ABL/genoa_variant_evaluation/{primary_metrics,matched_negative_auroc,matching_balance}.csv`; `repro_check/baseline_variant_evaluation/` for k-mer and composition |
| 3c | `ABL/genoa_variant_evaluation/significance_gradient.csv` |
| 3d | `asm_validation/asm_discrimination.csv` (bimodal non-ASM stratum), `asm_validation_tycko/tycko_e{1,2}_*.csv` |
| 4a | `ABL/genoa_variant_evaluation/fusion_vs_sequence_paired.csv` |
| 4b | schematic; no data |
| 4c | `ABL/gate_decomposition/gate_decomposition_summary.json` |
| 4d | `ABL/context_permutation/agreement_with_identity.csv` |
| 5a, 5b | `ABL/candidates/{top_candidate_case_study,candidate_matched_background_statistics,top_candidate_matched_comparators_long}.csv` |
| 5c | `ABL/literature_variant_screen/{stk11_case_study_figure_values,literature_variant_predictions_ranked}.csv` |
| 5d | `repro_check/gwas_nominal_current/{genoa,egtex}/run_summary.json` |
| S1 | `ABL/rc_uncertainty/*` |
| S2 | `ABL/motif_disruption/per_motif_coupling.csv`, `motif_disruption_kmer_baseline/per_motif_coupling.csv` |
| S3 | `ABL/target_qc/{test_coverage_per_probe,coverage_bin_metrics}.csv` |
| S4 | `ABL/egtex_mqtl_positive_control/mqtl_positive_control_metrics.csv`, `ABL/egtex_mqtl_matched_negative/matched_negative_metrics.csv`, `ABL/genoa_variant_evaluation/distance_bins.csv`, `asm_validation_tycko/tycko_e2_signed_agreement.csv` |

The sequence-only rows (`folds/*/sequence`, `published_baselines`, `sequence_baselines`,
`baseline_variant_evaluation`) sit outside the ablation tree deliberately: those models never read
the `Ref_*` context columns, so the context swap does not apply to them.

Per-panel source data with SHA-256 checksums is in `figures/source_data/`, indexed by
`figures/source_data/source_data_index.csv`.
