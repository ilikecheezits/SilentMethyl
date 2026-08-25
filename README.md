# SilentMethyl Repository Guide

This repository contains the active end-to-end SilentMethyl workflow used in the current checkout. The scripts and wrappers in this tree assume the project root is the repository root itself, and the canonical evaluation uses seeds 42, 43, and 44 with chromosome 8--9 held out for testing and chromosome 10--11 reserved for validation in the data build step.

Use the commands below from the repository root.

## 1. Environment setup

```bash
cd /path/to/SilentMethyl

conda create --name silentmethyl python=3.10 -y
conda activate silentmethyl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If your cluster requires a module load before Conda is available, do that first:

```bash
command -v conda || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
```

The repo is set up around the current Slurm scripts, which use `module load anaconda3` and, on Bridges-2-style systems, `module load cuda/12.4.0` before activating the environment. The site-specific `ROOT=` lines and scheduler directives in the shell scripts must be adjusted to the local checkout and cluster.

Check the relevant scripts before submitting jobs:

```bash
grep -nE '^#SBATCH|^ROOT=|module load|conda activate' \
  build_data.sh run_epi.sh run_baseline.sh run_multimodal.sh run_experiments.sh
```

## 2. Required project inputs

The current repo expects the processed and reference data files under `data/` and `data/reference/`. At a minimum, verify the files below before building the processed cohort:

```bash
for path in \
  data/hg38.fa \
  data/hg38.fa.fai \
  data/HM450.hg38.manifest.tsv.gz \
  data/HM450.hg38.manifest.CpGIsland.tsv.gz \
  data/TCGA-BRCA.methylation450.tsv.gz \
  data/BreastMammaryTissue.regular.perm.fdr.txt \
  data/egtex_breast_mqtl_heldout.csv \
  data/egtex_breast_mqtl_heldout_qc.csv \
  data/egtex_breast_mqtl_model_visible.csv \
  data/datafiles/gdc_tcga_brca_synonymous_raw.json.gz \
  data/reference/ATAC_seq.bw \
  data/reference/H3K27ac.bw \
  data/reference/H3K27me3.bw \
  data/reference/H3K36me3.bw \
  data/reference/H3K4me1.bw \
  data/reference/H3K4me3.bw \
  data/reference/H3K9me3.bw \
  data/reference/gencode.v44.annotation.gtf.gz \
  data/reference/hg38.phyloP100way.bw
do
  test -s "$path" || { echo "MISSING: $path" >&2; exit 1; }
done
echo "All required inputs are present."
```

The project also expects the GDC raw response cache at:

```text
data/datafiles/gdc_tcga_brca_synonymous_raw.json.gz
```

If you are starting from a fresh checkout, fetch and verify the public reference files and the released project-derived tables before proceeding. The scripts make assumptions about the exact file layout shown in the repo and validate them aggressively.

## 3. Build processed data and purity audit

Create the log directories used by the project:

```bash
mkdir -p logs/data_build logs/training logs/testing \
  logs/experiments logs/reproducibility reproducibility
```

Run the data build on a Slurm-enabled cluster:

```bash
sbatch build_data.sh
```

The wrapper performs three phases:

1. build training/validation splits with `data/build_training_data.py`
2. build the testing cohort with `data/build_testing_data.py`
3. audit purity and leakage with `data/audit_data_purity.py`

If the wrapper is not compatible with the local scheduler, run the equivalent commands directly:

```bash
python -u data/build_training_data.py \
  --data-dir data \
  --val-chroms chr10 chr11 \
  --test-chroms chr8 chr9

python -u data/build_testing_data.py \
  --data-dir data

python -u data/audit_data_purity.py \
  --data-dir data/datafiles \
  --output data/datafiles/data_purity_audit.json
```

The build should emit the split CSVs and manifests under `data/datafiles/` and complete with a valid `data/datafiles/data_purity_audit.json`.

## 4. Train the three journal models

The pipeline trains one context-only model, one sequence-only model, and one gated fusion model for each seed. The repo uses seeds 42, 43, and 44 and expects the `SEED` environment variable to be set for each job.

```bash
submit_training_seed() {
  seed="$1"

  epi_job=$(sbatch --parsable \
    --export="ALL,SEED=${seed}" \
    --job-name="SM_epi_s${seed}" \
    --output="logs/training/epi_seed${seed}_%j.out" \
    --error="logs/training/epi_seed${seed}_%j.err" \
    run_epi.sh)
  epi_job=${epi_job%%;*}

  seq_job=$(sbatch --parsable \
    --export="ALL,SEED=${seed}" \
    --job-name="SM_seq_s${seed}" \
    --output="logs/training/sequence_seed${seed}_%j.out" \
    --error="logs/training/sequence_seed${seed}_%j.err" \
    run_baseline.sh)
  seq_job=${seq_job%%;*}

  fusion_job=$(sbatch --parsable \
    --dependency="afterok:${epi_job}:${seq_job}" \
    --export="ALL,SEED=${seed}" \
    --job-name="SM_gate_s${seed}" \
    --output="logs/training/fusion_seed${seed}_%j.out" \
    --error="logs/training/fusion_seed${seed}_%j.err" \
    run_multimodal.sh)
  fusion_job=${fusion_job%%;*}

  echo "seed=${seed} context=${epi_job} sequence=${seq_job} fusion=${fusion_job}"
}

submit_training_seed 42
submit_training_seed 43
submit_training_seed 44
```

Check that all nine final checkpoint files exist:

```bash
for seed in 42 43 44; do
  for model in epi sequence fusion; do
    test -s "checkpoints_journal/seed${seed}/${model}/best_weights.pth" \
      || { echo "MISSING: seed${seed}/${model}" >&2; exit 1; }
  done
done
echo "All checkpoints are present."
```

## 5. Evaluate held-out chromosomes

After training, evaluate each saved checkpoint on the held-out test chromosomes:

```bash
for seed in 42 43 44; do
  python -u scripts/02_test_epi_journal.py \
    --seed "$seed" \
    --weights_path "checkpoints_journal/seed${seed}/epi/best_weights.pth" \
    --output_dir "results/journal/seed${seed}/epi" \
    2>&1 | tee "logs/testing/epi_seed${seed}.log"

  python -u scripts/02_test_sequence_journal.py \
    --seed "$seed" \
    --weights_path "checkpoints_journal/seed${seed}/sequence/best_weights.pth" \
    --output_dir "results/journal/seed${seed}/sequence" \
    2>&1 | tee "logs/testing/sequence_seed${seed}.log"

  python -u scripts/02_test_fusion_journal.py \
    --seed "$seed" \
    --weights_path "checkpoints_journal/seed${seed}/fusion/best_weights.pth" \
    --output_dir "results/journal/seed${seed}/fusion" \
    2>&1 | tee "logs/testing/fusion_seed${seed}.log"
done
```

Validate the expected output JSON metrics:

```bash
for seed in 42 43 44; do
  for model in epi sequence fusion; do
    python -m json.tool "results/journal/seed${seed}/${model}/metrics.json" >/dev/null || exit 1
  done
done
echo "All metrics files are valid JSON."
```

## 6. Run the downstream analysis workflow

The main experiment wrapper is:

```bash
sbatch run_experiments.sh 42 43 44
```

This wrapper performs the target QC, positive-control mQTL analysis, candidate scoring, and candidate stability workflows, and it checks the generated JSON outputs before exiting.

After the wrapper succeeds, run the remaining downstream analyses used by the repo:

```bash
python -u scripts/07_compare_candidate_models.py --seeds 42 43 44
python -u scripts/08_paired_model_bootstrap.py \
  --seeds 42 43 44 \
  --models epi sequence fusion \
  --block-size-bp 1000000 \
  --bootstrap-replicates 5000
python -u scripts/09_mqtl_matched_negative_control.py \
  --seeds 42 43 44 \
  --models fusion sequence \
  --maf-caliper 0.05 \
  --bootstrap-replicates 5000 \
  --permutation-replicates 10000
python -u scripts/10_tcga_participant_matrix_audit.py \
  --matrix-source "UCSC Xena GDC hub" \
  --matrix-source-id "TCGA-BRCA.methylation450.tsv.gz" \
  --matrix-source-url "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.methylation450.tsv.gz" \
  --matrix-data-type "DNA methylation beta values" \
  --matrix-processing "Matrix used as distributed; SilentMethyl selected sample-type-11 columns and computed the available-sample median per probe; upstream normalization was not independently reconstructed"
python -u scripts/12_biological_context_analysis.py \
  --seeds 42 43 44 \
  --models epi sequence fusion \
  --block-size-bp 1000000 \
  --bootstrap-replicates 2000
python -u scripts/13_build_manuscript_figures.py
```

## 7. Active application workflow: script 14

The active application example is the known-variant workflow implemented in `scripts/14_known_variant_application.py`. This script scores a user-supplied candidate SNV (or the built-in MLH1 example) against every eligible model-visible HM450 CpG in the frozen model window, keeps the nearest model-visible target as the primary display locus, and writes the rank/annotation summary under `results/journal/known_variant_application/`.

Run the default built-in example:

```bash
python -u scripts/14_known_variant_application.py \
  --seeds 42 43 44 \
  --output-dir results/journal/known_variant_application
```

Run a custom SNV list from a CSV containing at least `Variant_ID, Gene, chr, Position_1based, Ref, Alt`:

```bash
python -u scripts/14_known_variant_application.py \
  --variant-csv path/to/your_variants.csv \
  --seeds 42 43 44 \
  --output-dir results/journal/known_variant_application
```

This script is the supported demonstration workflow for disease- or gene-linked application examples. It is not a benchmark validation step and it deliberately uses the existing model visibility rules rather than a literature-only ranking.

## 8. Legacy literature screen: script 15

The literature-based variant screen in `scripts/15_literature_variant_screen.py` is historical and optional. It was used to assemble a broad breast-cancer literature/ClinVar candidate pool, filter it against the project's existing benchmark and HM450 window rules, and then hand the eligible SNVs to script 14 for scoring. This script is not part of the default analysis workflow and should only be run when reproducing the retired screen or generating historical discovery panels.

If you are restoring the legacy files from an archived branch or release tarball, copy them into place before running:

```bash
cp literature_variant_screen/15_literature_variant_screen.py scripts/
cp literature_variant_screen/literature_breast_variant_seeds.csv scripts/
```

Then export your NCBI email and run the preparation step (which only assembles and audits candidates without scoring them):

```bash
export NCBI_EMAIL="your_email@example.edu"

python -u scripts/15_literature_variant_screen.py \
  --prepare-only \
  2>&1 | tee logs/experiments/15_literature_variant_screen.log
```

To run the full historical screen after the preparation step, omit `--prepare-only` and allow the script to query ClinVar / PubMed and score the eligible candidates using the active known-variant application workflow. This mode is discovery-oriented and should be treated as a historical screening tool, not as a primary validation analysis.

Do not use smoke-test row limits or automatic mixed precision for the reportable candidate or mQTL analyses.

## 9. Confirm the primary outputs

The repo expects these files to exist and be valid JSON when the analysis is complete:

```bash
for path in \
  results/journal/target_qc/hm450_manifest_audit.json \
  results/journal/egtex_mqtl_positive_control/run_summary.json \
  results/journal/candidates/candidate_analysis_summary.json \
  results/journal/candidates/stability/stability_summary.json \
  results/journal/candidates/model_comparison/sequence_vs_fusion_summary.json \
  results/journal/paired_model_bootstrap/run_summary.json \
  results/journal/egtex_mqtl_matched_negative/run_summary.json \
  results/journal/biological_context/run_summary.json \
  results/journal/manuscript_figures/run_summary.json \
  reproducibility/tcga_participant_matrix_audit.json
do
  test -s "$path" || { echo "MISSING: $path" >&2; exit 1; }
  python -m json.tool "$path" >/dev/null || exit 1
done
echo "Primary outputs are present and valid."
```

The expected result sizes for the journal analysis are:

- 26,570 held-out CpGs on chromosomes 8--9
- 81 positive-control eGTEx breast mQTL associations
- 440 model-visible candidate rows after probe QC
- 35 matched significant and nonsignificant lead pairs
- 97 selected type-11 TCGA normal samples

## 8. Build the supplementary package

Once the main results are complete, generate the submission package:

```bash
python -u scripts/11_build_supplement_package.py --replace
```

Then verify the package:

```bash
cd supplementary_package
sha256sum -c SHA256SUMS.txt
```

The package includes S1–S6 tables plus two supplementary figures
(`information_source_gains` and `fusion_gain_by_genomic_region`). The six
main-text figures are not duplicated in the package.

## 9. Recommended execution order

1. Create the Python environment and install dependencies.
2. Verify the required input files and reference resources.
3. Build the processed data and purity audit.
4. Train the context-only, sequence-only, and fusion models for seeds 42, 43, and 44.
5. Run the held-out evaluation scripts for all nine checkpoints.
6. Run `run_experiments.sh 42 43 44`.
7. Run scripts 07 through 10, then 12 and 13.
8. Build and verify the supplementary package.

This is the current repository workflow reflected in the active scripts and shell wrappers in this checkout.

## 10. Repository status

- The active benchmark workflow is the frozen-model held-out evaluation and the downstream analysis pipeline anchored by scripts 02, 03, 04, 07, 08, 09, 10, 12, and 13.
- The codebase is expected to run from the repository root, with the current Slurm wrappers and per-seed checkpoint conventions preserved.
