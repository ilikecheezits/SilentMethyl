# MCF-10A audit — final, 16 Sep 2026

Two passes on 16 Sep 2026. The first was static (grep, hashes, recorded inputs)
and fixed nothing. The second, recorded here, re-verified every claim it relies
on, fixed all of bucket (b), added the supplement guard, and left bucket (a)
unfixed pending sign-off. Grep scope: every `.py .sbatch .sh .json .csv .md .tex
.txt .tsv` in the repository and `results/` (logs excluded), case-insensitive, for
`MCF`, `MCF-10A`, `MCF10A`, `MCF 10A`, `Mint-ChIP`, and the MCF-10A accessions
ENCFF548SFG, ENCFF282YCX, ENCFF274LWG, ENCFF423DKY, ENCFF634LDP, ENCFF714NIL,
ENCFF021PIS, ENCSR037XNN and ENCAN638MKH. PDFs were checked through `pdftotext`.
PNG titles cannot be grepped and were checked from the plotting code. 63 files
hit.

**Standard applied:** a number may appear in the paper only if it traces back to
breast-epithelium inputs, or to a model that reads no context at all.

## 0. The two context sets

| set | accessions | biosample | status |
|---|---|---|---|
| `data/reference/{ATAC_seq,H3K*}.bw` | ENCFF548SFG, ENCFF282YCX, ENCFF274LWG, ENCFF423DKY, ENCFF634LDP, ENCFF714NIL; ATAC converted from ENCFF021PIS | MCF 10A (Mint-ChIP, snATAC) | superseded 11 Sep 2026 |
| `data/reference/BreastEpithelium/*.bw` | ENCFF665NGK, ENCFF653CLL, ENCFF085IYD, ENCFF212ZFW, ENCFF481QEK, ENCFF714QJF, ENCFF234JZW | breast epithelium (ChIP/ATAC FC) | **published**; 7/7 local md5 = portal md5, re-resolved live 16 Sep |

## 1. Which arms consume context (verified 16 Sep 2026)

| arm | reads context? | retrained on breast epithelium? | weights |
|---|---|---|---|
| sequence-only | **no**: `SequenceOnlyModel.forward(input_ids, attention_mask)` | no, deliberately | `checkpoints_journal/seed*/sequence`, `checkpoints_folds/fold*/sequence_seed42` |
| context-only | yes | yes | `checkpoints_ablation/breast_epithelium/*/epi` |
| fusion | yes | yes | `checkpoints_ablation/breast_epithelium/*/fusion` |
| k-mer, composition, CpGenie, DeepCpG, Melody | no | n/a | unchanged |

Per-column MD5 over all rows of `train/val/test.csv`, `testing_data.csv` and
`testing_data_test_only.csv` in `data/datafiles/` vs `data/datafiles_breast_epithelium/`:
the only differing columns are the seven `Ref_*_Signal` tracks and some
`_Missing` flags. All ten FASTAs are byte-identical. The sequence arm has no path
to MCF-10A.

---

## 2. Bucket (a): wrong numbers — NOT EMPTY. Nothing in it was fixed.

None of these is in a submitted or circulated artefact that this repository
knows of. All are in the manuscript draft, its figure copies, or build scripts
whose output did not yet exist.

| # | item | evidence | fix prepared | status |
|---|---|---|---|---|
| A1 | the nine main-text figure copies at the repo root (`model_incremental_performance.png` … `uncertainty_scale_stability.pdf`), and hence `main_revised.pdf` | md5 of each root file equals the pre-swap build and differs from the breast-epithelium build (9/9) | `scripts/94_collect_manuscript_figures.sh` now maps to `ablation_breast_epithelium/`; tested into `repro_check/root_figures` (9/9 found) | **not run over the root copies**, awaiting sign-off |
| A2 | `main_revised.tex`: context-dependent numbers are pre-swap (e.g. fusion β MAE 0.0993 = MCF-10A mean of 0.0971/0.0988/0.1020; breast epithelium is 0.0914) and the text says "MCF-10A" at lines 194, 363, 1174 | spot-checked against both `metrics.json` trees; last written 9 Sep, before the swap | full line table in the appendix | manuscript rewrite, not done here |
| A3 | `ablation…/manuscript_figures/top_candidate_matched_background.png` carries the label `NCOA2: −0.1798 ± 0.0198` (the pre-swap three-seed value) over breast-epithelium data | the literal was hardcoded at `scripts/91_build_manuscript_figures.py:704` | 91 now derives the label. On pre-swap inputs it regenerates the published pre-swap figure **byte-identically**. On breast-epithelium inputs it reads `NCOA2: −0.1539 (seed 42 only)`, and the other 5 figures and 3 side outputs are byte-identical to the published build | corrected figure built in scratch, **not installed**. **A three-seed breast-epithelium NCOA2 value does not exist**, and nothing was substituted |
| A4 | `RESULTS_REVISED.md` R3: eight non-Lung tissue AUROCs carried over from the MCF-10A fusion model | stated in its own header | GPU rescoring command in `REPRODUCE.md` §10 | GPU job |
| A5 | **new in this pass:** `results/journal/target_qc/coverage_{error_correlations,threshold_metrics,bin_metrics}.csv`, packaged as Supplementary S5, were computed from MCF-10A epi and fusion predictions (epi β MAE 0.1396) | 01_target_qc defaults to `results/journal/seed42/{model}/predictions.csv` | breast-epithelium rerun (2 min, CPU) in `repro_check/abl/target_qc`; sequence rows byte-identical, epi 0.1022 / fusion 0.0885 | install into `ablation…/target_qc` was **blocked** (published tree); awaiting sign-off |
| A6 | `scripts/90_build_supplement_package.py` (S1–S6) packaged the pre-swap trees and the MCF-10A build manifests | its manifest | repointed to breast epithelium, guard added (§5) | fixed in code; no S1–S6 package had been built |

Untraced manuscript numbers (sources not found in either tree): tumour
shifted-target MAE 0.0975 → 0.1180 (lines 588–592), and direction by precision
quintile 0.521–0.566 (line 1209).

## 3. Bucket (b): wrong labels — all fixed

| what | fix |
|---|---|
| 15 `run_summary_{fusion,sequence}_seed*.json` under `ablation…/{genoa,egtex}_variant_scoring/heldout/` and `ablation…/egtex_multitissue_scoring/Lung/heldout/`: `tissue_caveat` said "MCF-10A breast tracks" | rewritten in place with the now-derived caveat, plus a `relabelled` note. Every other key was verified unchanged. Weights confirmed from the `Weights_Path` column; Lung split template confirmed from `logs/egtex_mt_scoring` |
| `ablation…/{genoa,egtex}_variant_evaluation/run_summary.json`: `context_source` "unrecorded" | rewritten: "primary breast epithelium (ENCODE, …)" |
| `published_baselines/variant_evaluation_genoa/run_summary.json`: CpGenie/DeepCpG described as having MCF-10A context | rewritten: "none (cpgenie, deepcpg are sequence-only models …)" |
| `baseline_variant_evaluation/run_summary.json`: "unrecorded" | rewritten: "none (composition, kmer_ridge …)" |
| `ablation…/biological_context/plots/fusion_gain_by_context.png`: panels titled "MCF-10A ATAC/H3K27ac signal" | 22 derives the label from `--test-path`. It was rerun into scratch: all tables reproduced (one CSV differs only in a float's last printed digit), and the relabelled PNG was installed. Original backed up outside the repo |
| sources that would re-stale future runs: `20_variant_scoring.py` (hardcoded caveat), `21` (CpGenie/DeepCpG not context-free), `22`, `40` (docstring and banner), `build_{training,testing}_data.py` help, `acquire_multitissue_inputs.py` (docstring and the TRACK_SET template) | all fixed. 20 now also **refuses** a fusion checkpoint scored against a different context build |
| `data/reference/TRACK_SET.md` said the MCF-10A tracks are "THE PUBLISHED SINGLE-TISSUE MODEL"; `BreastEpithelium/`, `Lung/`, `ColonTransverse/` and `KidneyCortex/TRACK_SET.md` said "both are MCF-10A" | rewritten: the root set is superseded; BreastEpithelium is the published context, with accession, experiment and md5 |
| `README.md` §2, the "MCF 10A" reference row, and the MCF-10A reproduction path | README rewritten to point to `REPRODUCE.md` |

19 JSON files were relabelled, and no numeric value changed. Backups are at
`/ocean/projects/med250012p/szhang37/SM_scratch/json_backup/`.

## 4. Bucket (c): correct historical record

- Pre-swap results directly under `results/journal/` (epi/fusion arms of
  `seed*/`, `folds/`; `biological_context`, `candidates`, `context_permutation`,
  `egtex_mqtl_*`, `{genoa,egtex}_variant_{scoring,evaluation}`,
  `egtex_multitissue_scoring` fusion, `gwas_enrichment_*`, `known_variant_application`,
  `literature_variant_screen`, `manuscript_figures`, `meqtl_class_chromatin`,
  `motif_disruption`, `paired_model_*`, `rc_uncertainty*`, `target_qc`,
  `tissue_shared_meqtls`, `transfer_discrimination`, `variant_effect_synthesis`,
  `melody_head_to_head`, `melody_st_head_to_head`). Their MCF-10A labels are accurate.
- `checkpoints_journal/seed*/{epi,fusion}`, `checkpoints_folds/fold*/{epi,fusion}_seed42`,
  `data/reference/*.bw` (root), the `Ref_*` columns of `data/datafiles/`.
- `data/external/external_manifest.json` `reference_tracks.root` (provenance of the MCF-10A files).
- `main.tex` (the original manuscript; never modified).
- Code comments and job-file warnings that describe the MCF-10A hazard (`jobs/r7_ablation/64_literature.sbatch`,
  `jobs/r8_journal/{_common.sh,23_ctx_ladder.sbatch,54_gate_instrument.sbatch}`,
  `scripts/run_ablation_analyses.sbatch`, `scripts/run_context_ablation.sbatch`,
  `scripts/run_egtex_multitissue_scoring.sh`, `scripts/21_variant_evaluation.py`,
  `data/{audit_reference_tracks,audit_multitissue_builds,survey_tcga_normal_cohorts}.py`).
- `LAB_NOTES.md`, `RESULTS_REVISED.md` (comparison tables that label MCF-10A as such), this file.

**Still read downstream:** only by A1, A3 and A5 above (the manuscript's figure
copies and S5), plus the context-free sequence arms that the published tree reuses
by design. With 94 and 90 repointed, no build script reads a bucket (c)
context-dependent product.

## 5. The four named suspects

1. **`91:704` NCOA2 literal.** Confirmed (a), A3. The script now reads the value
   from its inputs. No three-seed breast-epithelium value exists: `60_candidates.sbatch`
   ran seed 42 only.
2. **Stale strings in existing `run_summary.json`.** 19 files relabelled (§3). The
   generator (`20`) is fixed, so a rescore cannot reintroduce them.
3. **R6 staging runs of `22`/`91`.** Their outputs no longer exist separately: R7
   jobs 45920224/45920225 rewrote `ablation…/biological_context` and
   `ablation…/manuscript_figures` in place on 13 Sep. `manuscript_figures/run_summary.json`
   records 9 inputs, all under `ablation_breast_epithelium/`. Nothing reads a
   staging output. The two job files that would recreate the hybrid
   (`jobs/r6_ablation/{22_context,91_figures}.sbatch`) were **deleted** (git
   history keeps them), along with their logs.
4. **Input trace of manuscript-facing artefacts:**

| artefact | built by | inputs | resolves under breast epithelium? |
|---|---|---|---|
| main figures (source builds) | 91 (R7 job 45920225), 21, 50, 51 | 9 recorded inputs, all `ablation_breast_epithelium/` | yes (the label: A3) |
| main figures (root copies used by the .tex) | 94 | pre-swap builds | **no** (A1) |
| R8 figures `manuscript_figures_r8` | 92 | `ablation…/{fusion_gain_stratified,context_ladder,gate_decomposition}`, `joint/transfer_failure`, `asm_validation` | yes |
| S7 gate decomposition | 93 | `ablation…/gate_decomposition` ← ablation fusion + sequence pair scores | yes |
| S8 context ladder | 93 | `ablation…/context_ladder(_stratified)` ← `checkpoints_ablation/…/seed42/fusion`, BE splits, per-tissue tracks | yes |
| S9 transfer failure | 93 | `joint/transfer_failure` ← `joint/holdout_BreastEpithelium/seed42` ← `datafiles_joint` composed from `datafiles_breast_epithelium` | yes |
| S10 fusion gain | 93 | `ablation…/fusion_gain_stratified`, `ablation…/biological_context` | yes |
| S11/S12 ASM | 93 | `asm_validation(_tycko)` ← `checkpoints_ablation/…/seed42/fusion` + `checkpoints_journal/seed42/sequence`; features from `data/reference/BreastEpithelium` (53/53d default and build summary) | yes |
| S1–S6 | 90 | repointed to `ablation…` and `data/datafiles_breast_epithelium` (A6); S5 awaits A5 | yes after A5 |
| tables in `main_revised.tex` | — | see appendix | pending rewrite (A2) |

The rebuilt S7–S12 package is byte-identical to `results/supplementary_package_r8`
(23/23 data files) and passes the guard.

## 6. The guard

`scripts/93_build_r8_supplement.py` (shared by 90) fails the build and deletes
the package if (1) any packaged file matches the MCF-10A marker pattern, or (2)
any packaged source lies outside breast-epithelium or context-free paths. Check (2)
is what catches an unlabelled pre-swap number. Tested 16 Sep 2026:

| test | result |
|---|---|
| clean rebuild | passes, `0 MCF-10A marker(s)` |
| `MCF-10A` appended to a packaged CSV, `--check-package` | exit 3, hit reported with file:line |
| source containing `MCF 10A Mint-ChIP ENCFF548SFG`, in-build | exit 3, package deleted |
| pre-swap source with no marker (`results/journal/candidates/…`), in-build | exit 3, package deleted |

## 7. What references `data/datafiles/` (not deleted)

Still needed: `data/build_testing_data.py` reads the frozen GDC response
from `data/datafiles/gdc_tcga_brca_synonymous_raw.json.gz` in every build;
`scripts/run_context_ablation.sbatch` compares fold probe sets against the split
the reused sequence tower recorded (`data/datafiles/splits/foldN` for ours).
Defaults only (every documented command overrides them): `01, 13, 14, 15, 18, 20,
21, 22, 23, 40, 50, 60, 62, 63, 64, 70, 71, 90` (docstring), `data/{build_genoa_scoring_input,
harmonize_egtex_mqtl,audit_training_data,build_tcga_ancestry_labels}.py`.
Records: `checkpoints_journal/*/run_config.json`, `checkpoints_folds/*/run_config.json`,
`reproducibility/{processed_data_sha256.txt,data_purity_audit.json}`, LAB_NOTES,
RESULTS_REVISED.

## 8. main.tex

Not staged and unchanged against HEAD at the start of this pass. It had been
staged for deletion by the cleanup session `5ecdf559`
(`git rm -r -q _archive && git rm -q main.tex mentor_email_draft.md && …`) and
restored from HEAD by the audit session `d0503b5d`.

---

## Appendix: manuscript line table (from the static pass, spot-checked)

### A2. `main_revised.tex`: every context-dependent number is pre-swap

The file was last written on **9 Sep 2026**, and the context changed on **11 Sep**. Spot check: the
headline fusion β MAE 0.0993 is the MCF-10A three-seed mean
(0.0971/0.0988/0.1020). Breast epithelium gives 0.0914 (0.0885/0.0901/0.0956). The
fold table's 0.0971/0.0942/0.0990/0.0938 are the MCF-10A fold values. Breast
epithelium is 0.0885/0.0873/0.0934/0.0862.

| lines | claim | breast-epithelium source | status |
|---|---|---|---|
| 101–102 | abstract MAE 0.0993, AUC 0.968 | `ablation…/seed4x/fusion/metrics.json` | replaceable |
| 194, 363, 1174 | "seven MCF-10A attributes", "MCF-10A inputs", MCF-10A limitation | — | wording; the domain-mismatch limitation no longer applies |
| 445–452, tab:performance 492–494, 500–502 | Context and Fusion columns; % reductions; fusion−sequence CI | `ablation…/seed4x/{epi,fusion}`, `ablation…/paired_model_bootstrap` | replaceable. Composition, k-mer, CpGenie, DeepCpG, and Sequence columns are context-free and stay |
| 530, tab:folds 539–542, 547–572 | fold fusion MAE/AUC, gain ranges, "published split largest" | `ablation…/seed42/fusion`, `ablation…/fold{1,2,3}/fusion` | replaceable. The **qualitative claims must be re-checked** (e.g. which fold has the largest gain) |
| 577–580 | fusion gain by region/island/ATAC/H3K27ac | `ablation…/biological_context/fusion_gain_by_context.csv` | replaceable. Values roughly double (e.g. shore 0.0230), and "largest in highest ATAC quartile" **no longer holds** (Q4 0.0180 < Q2 0.0191) |
| 588–592 | tumour shifted-target 0.0975 → 0.1180 | **no source found** in either tree | **untraced**, fixed predictions from pre-swap fusion |
| 599–617, tab:genoa 634–638, tab:cohorts 663–669, 644–646, 675–684 | eGTEx/GENOA ρ, direction, calibration, meta-analysis, gradient | `ablation…/{genoa,egtex}_variant_evaluation`, `ablation…/variant_effect_synthesis`, `ablation…/egtex_mqtl_{positive_control,matched_negative}` | replaceable (3 seeds exist). Sequence column and distance-only AUROC are context-free |
| 706–714, 729–733 | matched AUROC 0.570; fusion vs DeepCpG/CpGenie | `ablation…/genoa_variant_evaluation`, `ablation…/paired_model_comparison_genoa` | replaceable. k-mer 0.503 is context-free |
| 740–782, tab:ctxperm 760–763 | context permutation | `ablation…/context_permutation` | replaceable |
| 787–809 | nine-tissue transfer AUROCs (breast 0.614, colon 0.605, kidney 0.606, lung 0.588, ovary …) | **Lung only** (`ablation…/transfer_discrimination/Lung`) | **GAP: 8 of 9 tissues have no breast-epithelium value** |
| 811–820, 893–951 | Melody-MT and Melody-ST head-to-heads (SilentMethyl column) | none on breast epithelium | **GAP** (depends on the 8 tissues) |
| 824–868 | tissue-specific vs shared meQTLs (SilentMethyl arm) | `ablation…/tissue_shared_meqtls` (GENOA + eGTEx cohorts) | replaceable only if the manuscript's cohorts match that run. The 72-direction / 9-tissue version has **no** breast-epithelium source |
| 958–980 | fusion − sequence equivalence; ρ = 0.981 among 440 candidates | `ablation…/{genoa,egtex}_variant_evaluation/fusion_vs_sequence_paired.csv` | equivalence replaceable. The **candidate concordance has 1 seed only** |
| 984–992, fig uncertainty | RC-disagreement uncertainty | `ablation…/rc_uncertainty*` | replaceable |
| 1010–1028, tab:motifs 1370–1375 | motif coupling, k-mer vs fusion AUROC 0.516 vs 0.592 | `ablation…/motif_disruption` | replaceable. The k-mer column is context-free |
| 1058–1066 | 440 candidates: cross-seed ρ 0.681–0.699, region/distance medians | `ablation…/candidates`, `ablation…/biological_context` | **GAP: cross-seed statistics need 3 seeds; only seed 42 exists** |
| 1071–1081 | NCOA2 Δβ = −0.1798 ± 0.0198, "exceeded all 66 matched" | `ablation…/candidates` (seed 42: −0.1539, still rank 1) | **GAP: no three-seed breast-epithelium value exists.** Nothing substituted |
| 1089–1105 | STK11 Δβ 0.0774 / −0.0613 | `ablation…/literature_variant_screen` (3 seeds) | replaceable |
| 1122–1130 | GWAS top-5% share | `ablation…/gwas_regulatory_enrichment` | replaceable |
| 1147–1172 | Discussion restates ρ 0.178, AUROC 0.570, shores/ATAC claim | as above | follows the replacements |
| 1209 | direction by precision quintile 0.521–0.566 | not located | **untraced** |
| tab:gradient 1329–1331, tab:distance 1347–1352 | GENOA gradient and distance bins | `ablation…/genoa_variant_evaluation/{significance_gradient,distance_bins}.csv` | replaceable |
| tab:splits 1312–1314 | loci counts, mean β | context-free | OK |
| 1383–1412 | supplementary figures | see A1 | wrong files |

