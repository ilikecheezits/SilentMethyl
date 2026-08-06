# SilentMethyl Reproduction Guide

Run all commands from the repository root on a Linux system. The reported
models use seeds `42`, `43`, and `44`, with chromosomes 10--11 for validation
and chromosomes 8--9 for testing.

## 1. Clone and install

```bash
git clone https://github.com/ilikecheezits/SilentMethyl.git
cd SilentMethyl
```

If the directory already exists, use a different destination instead of
overwriting it:

```bash
git clone https://github.com/ilikecheezits/SilentMethyl.git SilentMethyl-reproduction
cd SilentMethyl-reproduction
```

Make Conda available. On Bridges-2 this requires `module load anaconda3`; many
systems already provide Conda and should skip that command.

```bash
command -v conda || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create --name silentmethyl python=3.10 -y
conda activate silentmethyl
python -m pip install -r requirements.txt
python -m pip check
```

`pip check` should report `No broken requirements found.` CUDA may appear
unavailable on a login node even when it is available inside GPU jobs.

The supplied batch files contain Bridges-2 `#SBATCH`, module, and project-root
settings. Before submission, inspect and adapt only those site-specific lines:

```bash
grep -nE '^#SBATCH|^ROOT=|module load|conda activate' \
  build_data.sh run_epi.sh run_baseline.sh run_multimodal.sh run_experiments.sh
```

The batch-file `ROOT` must point to the checkout being reproduced. Other
clusters may use different partitions, GPU requests, CUDA modules, or no module
commands at all.

## 2. Acquire input data

```bash
mkdir -p data/reference data/source/atac data/datafiles reproducibility
```

### 2.1 Genome, methylation, annotation, and conservation

```bash
wget -c -O data/hg38.fa.gz \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gzip -dc data/hg38.fa.gz > data/hg38.fa

python - <<'PY'
from pathlib import Path
import pysam

fasta = Path("data/hg38.fa")
pysam.faidx(str(fasta))
index = Path(str(fasta) + ".fai")
if not index.is_file() or index.stat().st_size == 0:
    raise SystemExit(f"Missing FASTA index: {index}")
print(index)
PY

wget -c -O data/HM450.hg38.manifest.tsv.gz \
  https://zhouserver.research.chop.edu/InfiniumAnnotation/current/HM450/HM450.hg38.manifest.tsv.gz
echo "668a11ea624b3e645aab963dca048bdd4628ae5d297732867c91525c831cf191  data/HM450.hg38.manifest.tsv.gz" \
  | sha256sum -c -

wget -c -O data/TCGA-BRCA.methylation450.tsv.gz \
  https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.methylation450.tsv.gz
echo "71f7a02dd9ff849f43e05c6e54a9b8266349c9697601dc0450c9dc30f47679db  data/TCGA-BRCA.methylation450.tsv.gz" \
  | sha256sum -c -

wget -c -O data/reference/hg38.phyloP100way.bw \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw

wget -c -O data/reference/gencode.v44.annotation.gtf.gz \
  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
```

The TCGA matrix is the UCSC Xena GDC-hub matrix used by the project. The
pipeline selects 97 sample-type-11 columns and computes the available-sample
median at each CpG.

### 2.2 ENCODE reference tracks

Download one released GRCh38 `fold change over control` BigWig from each
experiment and save it under the required name.

| Required path | ENCODE experiment |
|---|---|
| `data/reference/H3K27ac.bw` | `ENCSR685JSL` |
| `data/reference/H3K27me3.bw` | `ENCSR884WUC` |
| `data/reference/H3K36me3.bw` | `ENCSR362VGZ` |
| `data/reference/H3K4me1.bw` | `ENCSR585PIL` |
| `data/reference/H3K4me3.bw` | `ENCSR291QUA` |
| `data/reference/H3K9me3.bw` | `ENCSR049WGG` |

Experiment pages use this form:

```text
https://www.encodeproject.org/experiments/ENCSR685JSL/
```

Record the selected `ENCFF...` file accession and ENCODE MD5. Check each local
file with the corresponding published MD5:

```bash
echo '<ENCODE_MD5>  data/reference/H3K27ac.bw' | md5sum -c -
```

The ATAC track uses filtered GRCh38 BAM `ENCFF021PIS` from experiment
`ENCSR037XNN` and CPM-normalized 50-bp bins:

```bash
wget -c -O data/source/atac/ENCFF021PIS.bam \
  https://www.encodeproject.org/files/ENCFF021PIS/@@download/ENCFF021PIS.bam

python - <<'PY'
from pathlib import Path
import pysam

bam = Path("data/source/atac/ENCFF021PIS.bam")
index = Path(str(bam) + ".bai")
if not index.exists():
    pysam.index("-@", "8", str(bam))
print(index)
PY

bamCoverage \
  --bam data/source/atac/ENCFF021PIS.bam \
  --outFileName data/reference/ATAC_seq.bw \
  --outFileFormat bigwig \
  --binSize 50 \
  --normalizeUsing CPM \
  --numberOfProcessors 8
```

The archived modeling track has SHA-256
`b256de0d7c517809733601dfe2093b2d4458ea8ad94d3ccb309be327f5838578`.
Different deepTools versions can produce a signal-equivalent BigWig with
different bytes, so record `bamCoverage --version`.

### 2.3 eGTEx inputs

```bash
wget -c -O data/BreastMammaryTissue.regular.perm.fdr.txt \
  https://storage.googleapis.com/egtex/methylation/epic-arrays/mQTLs/BreastMammaryTissue.regular.perm.fdr.txt
```

The public release must also contain these frozen project-derived files:

```text
data/egtex_breast_mqtl_heldout.csv
data/egtex_breast_mqtl_heldout_qc.csv
data/egtex_breast_mqtl_model_visible.csv
```

Verify all required inputs:

```bash
for path in \
  data/hg38.fa \
  data/hg38.fa.fai \
  data/HM450.hg38.manifest.tsv.gz \
  data/TCGA-BRCA.methylation450.tsv.gz \
  data/BreastMammaryTissue.regular.perm.fdr.txt \
  data/egtex_breast_mqtl_heldout.csv \
  data/egtex_breast_mqtl_heldout_qc.csv \
  data/egtex_breast_mqtl_model_visible.csv \
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

Do not continue to data construction while any required input is missing.

## 3. Build processed data

For an exact frozen reproduction, place the released GDC response cache at:

```text
data/datafiles/gdc_tcga_brca_synonymous_raw.json.gz
```

Create log directories:

```bash
mkdir -p logs/data_build logs/training logs/testing \
  logs/experiments logs/reproducibility reproducibility
```

On a configured Slurm cluster, run the complete data build and purity audit:

```bash
DATA_JOB=$(sbatch --parsable build_data.sh)
echo "$DATA_JOB"
```

If the wrapper is not compatible with the local scheduler, run its three core
commands inside a sufficiently large CPU allocation:

```bash
python -u data/build_training_data.py \
  --data-dir data \
  --val-chroms chr10 chr11 \
  --test-chroms chr8 chr9 \
  2>&1 | tee logs/data_build/build_training_data.log

python -u data/build_testing_data.py \
  --data-dir data \
  2>&1 | tee logs/data_build/build_testing_data.log

python -u data/audit_data_purity.py \
  --data-dir data/datafiles \
  --output data/datafiles/data_purity_audit.json \
  2>&1 | tee logs/reproducibility/data_purity_audit.log
```

Do not use `--refresh-gdc` for the frozen reproduction. The audit should report
zero hard errors and zero warnings.

```bash
python -m json.tool data/datafiles/data_purity_audit.json >/dev/null
```

The build should create `train.csv`, `val.csv`, `test.csv`, their FASTA views,
the candidate tables, and the associated JSON manifests under
`data/datafiles/`.

## 4. Train three seeds

The training wrappers read seed number from the exported `SEED` variable. Do
not use a Slurm array index as the seed. For each seed, context and sequence
training run in parallel, and fusion starts only after both succeed.

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

The programs resume from `latest_checkpoint.pt` when it exists. A clean
from-scratch reproduction starts without old checkpoint directories.

Confirm that all nine final checkpoints exist:

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

Run these commands inside a GPU allocation:

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

Do not pass `--max_rows` for reportable results. Verify all metrics files:

```bash
for seed in 42 43 44; do
  for model in epi sequence fusion; do
    python -m json.tool \
      "results/journal/seed${seed}/${model}/metrics.json" >/dev/null \
      || exit 1
  done
done
echo "All held-out metrics are valid JSON."
```

## 6. Run downstream analyses

After all checkpoints and held-out predictions exist, run target QC, the mQTL
positive control, candidate scoring, and candidate stability:

```bash
sbatch run_experiments.sh 42 43 44
```

`run_experiments.sh` takes seeds as positional arguments. After that job
finishes, run the remaining analyses in the same environment:

```bash
python -u scripts/07_compare_candidate_models.py \
  --seeds 42 43 44 \
  2>&1 | tee logs/experiments/07_candidate_model_comparison.log

python -u scripts/08_paired_model_bootstrap.py \
  --seeds 42 43 44 \
  --models epi sequence fusion \
  --block-size-bp 1000000 \
  --bootstrap-replicates 5000 \
  2>&1 | tee logs/experiments/08_paired_model_bootstrap.log

python -u scripts/09_mqtl_matched_negative_control.py \
  --seeds 42 43 44 \
  --models fusion sequence \
  --maf-caliper 0.05 \
  --bootstrap-replicates 5000 \
  --permutation-replicates 10000 \
  2>&1 | tee logs/experiments/09_mqtl_matched_negative.log

python -u scripts/10_tcga_participant_matrix_audit.py \
  --matrix-source "UCSC Xena GDC hub" \
  --matrix-source-id "TCGA-BRCA.methylation450.tsv.gz" \
  --matrix-source-url "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.methylation450.tsv.gz" \
  --matrix-data-type "DNA methylation beta values" \
  --matrix-processing "Matrix used as distributed; SilentMethyl selected sample-type-11 columns and computed the available-sample median per probe; upstream normalization was not independently reconstructed" \
  2>&1 | tee logs/reproducibility/10_tcga_participant_matrix_audit.log
```

Do not use smoke-test row limits or automatic mixed precision for reportable
candidate or mQTL results.

## 7. Confirm the primary outputs

```bash
for path in \
  results/journal/target_qc/hm450_manifest_audit.json \
  results/journal/egtex_mqtl_positive_control/run_summary.json \
  results/journal/candidates/candidate_analysis_summary.json \
  results/journal/candidates/stability/stability_summary.json \
  results/journal/candidates/model_comparison/sequence_vs_fusion_summary.json \
  results/journal/paired_model_bootstrap/run_summary.json \
  results/journal/egtex_mqtl_matched_negative/run_summary.json \
  reproducibility/tcga_participant_matrix_audit.json
do
  test -s "$path" || { echo "MISSING: $path" >&2; exit 1; }
  python -m json.tool "$path" >/dev/null || exit 1
done
echo "Primary outputs are present and valid."
```

Expected cohort sizes are:

- 26,570 held-out CpGs on chromosomes 8--9;
- 81 positive-control mQTL associations representing 70 unique variants;
- 440 model-visible candidate rows after probe QC;
- 35 matched significant/high-q mQTL pairs;
- 97 selected TCGA type-11 columns from 97 unique participants.

## 8. Build the supplementary package

```bash
python -u scripts/11_build_supplement_package.py --replace \
  2>&1 | tee logs/reproducibility/11_build_supplement_package.log

(
  cd supplementary_package
  sha256sum -c SHA256SUMS.txt
)
```

The package should contain numbered data folders S1--S6,
`supplement_manifest.csv`, and `SHA256SUMS.txt`. Do not use
`--allow-missing-required` for a submission package.

## 9. Execution order

1. Install the environment.
2. Download and verify every required input.
3. Build and audit processed data.
4. Train all three model types for seeds 42, 43, and 44.
5. Test all nine checkpoints on chromosomes 8--9.
6. Run `run_experiments.sh 42 43 44`.
7. Run scripts 07--10.
8. Build and verify the supplementary package.