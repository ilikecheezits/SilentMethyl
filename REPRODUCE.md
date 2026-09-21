# Reproducing SilentMethyl

Run the steps in order from the repository root. The Slurm commands use the
Bridges-2 cluster configuration; they are not local Mac commands.

The local checkout contains scripts, summaries, figures and source data. The
cluster working directory also contains checkpoints, downloaded inputs and full
prediction tables. On the cluster, check existing files before rebuilding them.
For a fresh checkout, download, copy or generate the required files first.

Checksum checks compare your files with the recorded versions. `OK` means a file
matches. Wait for each required job to finish before starting a dependent step.

### Files already on the cluster

| Path | Contents |
|---|---|
| `checkpoints_journal/`, `checkpoints_folds/`, `checkpoints_joint/` | Trained weights and run settings |
| `dnabert2_local/` | Downloaded DNABERT-2 model and tokenizer |
| `data/reference/`, `data/targets/`, `data/external/` | Reference tracks, methylation matrices and external datasets |
| `data/datafiles_breast_epithelium/` | Main breast training, validation and test splits |
| `data/datafiles_multitissue/`, `data/datafiles_joint/` | Per-tissue and joint-model splits |
| `results/journal/ablation_breast_epithelium/` | Current breast-epithelium predictions and analyses |
| `repro_check/` | Outputs from separate reproduction checks |

Use the checksum checks below to verify these files before reusing them.

## 1. Models and checkpoint paths

| Model | Uses `Ref_*` context? | Checkpoint or result path |
|---|---|---|
| sequence-only (DNABERT-2) | no; trained on `data/datafiles/` | `checkpoints_journal/seed*/sequence`, `checkpoints_folds/fold*/sequence_seed42` |
| context-only | yes; breast epithelium | `checkpoints_journal/seed*/epi`, `checkpoints_folds/fold*/epi_seed42` |
| gated fusion | yes; breast epithelium | `checkpoints_journal/seed*/fusion`, `checkpoints_folds/fold*/fusion_seed42` |
| k-mer, composition, CpGenie, DeepCpG | no | `results/journal/{sequence_baselines,published_baselines}` |

The sequence model does not use context, so its weights were reused after the
switch to primary breast-epithelium tracks. Context and fusion models were
retrained. Section 5.3 checks that fusion uses the matching sequence weights.

## 2. Environment

Create the environment and install the dependencies:

```bash
conda create --name silentmethyl python=3.10.20 -y
conda activate silentmethyl
python -m pip install --upgrade pip
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

Set the Python path used by the job scripts:

```bash
export PY=$HOME/.conda/envs/silentmethyl/bin/python
export SILENTMETHYL_PY=$PY
export SBATCH_EXPORT=ALL
$PY --version
```

Set `<project-dir>` to your project directory, then download DNABERT-2:

```bash
export HF_HOME=<project-dir>/.cache/huggingface
$PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("zhihan1996/DNABERT-2-117M", local_dir="dnabert2_local")
EOF
```

## 3. Estimated compute time

Training uses about 350 V100 GPU-hours. Later GPU steps use less than 40
GPU-hours, and CPU analyses take about one hour. These estimates exclude queue
and download time.

## 4. Input data

### 4.1 Genome, array and reference resources

```bash
mkdir -p data/reference
curl -fL https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip > data/hg38.fa
$PY -c "import pysam; pysam.faidx('data/hg38.fa')"
curl -fL -o data/HM450.hg38.manifest.tsv.gz \
  https://zhouserver.research.chop.edu/InfiniumAnnotation/20180909/HM450/HM450.hg38.manifest.tsv.gz
curl -fL -o data/HM450.hg38.manifest.CpGIsland.tsv.gz \
  https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/HM450/HM450.hg38.manifest.CpGIsland.tsv.gz
curl -fL -o data/TCGA-BRCA.methylation450.tsv.gz \
  https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.methylation450.tsv.gz
curl -fL -o data/reference/hg38.phyloP100way.bw \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw
curl -fL -o data/reference/gencode.v44.annotation.gtf.gz \
  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
curl -fL -o data/reference/hg19ToHg38.over.chain.gz \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
curl -fL --retry 10 --retry-delay 60 --connect-timeout 30 -o data/reference/ENCFF356LFX.bed.gz \
  https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz
```

Check the downloads below; expect eight `OK` results:
```bash
grep -E "  data/(hg38\.fa|HM450|TCGA-BRCA|reference/(hg38\.phyloP|gencode|ENCFF))" \
  reproducibility/reference_audit_breast_epithelium.txt | sha256sum -c -
```

### 4.2 Context tracks (ENCODE)

The first command previews the breast-epithelium downloads. The second downloads
the files. The final command restores the saved acquisition plan.

```bash
$PY -u data/acquire_multitissue_inputs.py --tissues BreastEpithelium --context-only \
    --picks data/multitissue_picks.tsv --plan data/multitissue_acquisition_plan.json
$PY -u data/acquire_multitissue_inputs.py --tissues BreastEpithelium --context-only \
    --picks data/multitissue_picks.tsv --plan data/multitissue_acquisition_plan.json --apply
git checkout data/multitissue_acquisition_plan.json
```

After completing Section 4.4, check all reference files; expect 16 `OK` results:
```bash
sha256sum -c reproducibility/reference_audit_breast_epithelium.txt
```

### 4.3 Cohorts and catalogues

Start the GENOA download on a login node:
```bash
nohup $PY -u data/acquire_external_cohorts.py --only genoa_meqtl > genoa_meqtl.log 2>&1 &
```

Download the other cohorts and catalogues, and submit the eGTEx preprocessing jobs:

```bash
mkdir -p data/external/egtex_breast logs/data_build
curl -fL -o data/external/egtex_breast/BreastMammaryTissue.mQTLs.regular.txt.gz \
  https://storage.googleapis.com/egtex/methylation/epic-arrays/mQTLs/BreastMammaryTissue.mQTLs.regular.txt.gz
curl -fL -o data/external/egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt \
  https://storage.googleapis.com/egtex/methylation/epic-arrays/mQTLs/BreastMammaryTissue.regular.perm.fdr.txt
sbatch -J egtex_breast_slice -p RM-shared -N1 -n1 -c 4 --mem=8000M -t 06:00:00 \
  -o logs/data_build/egtex_breast_slice_%j.log \
  --wrap "$PY -u data/harmonize_egtex_mqtl.py --prefilter --prefiltered data/external/egtex_breast/egtex_breast_within600.tsv.gz"
sbatch -J egtex_regular -p RM-shared -N1 -n1 -c 6 --mem=12000M -t 24:00:00 \
  -o logs/data_build/egtex_regular_%j.log --wrap "bash data/fetch_egtex_regular_slices.sh"
sbatch -J egtex_multi -p RM-shared -N1 -n1 -c 2 --mem=4000M -t 04:00:00 \
  -o logs/data_build/egtex_multi_%j.log --wrap "bash data/fetch_egtex_multitissue.sh"

mkdir -p data/external/gwas_catalog
curl -fL -o gwas.zip \
  https://ftp.ebi.ac.uk/pub/databases/gwas/releases/2026/08/24/gwas-catalog-associations_ontology-annotated-full.zip
unzip -o -p gwas.zip '*.tsv' > data/external/gwas_catalog/gwas-catalog-download-associations-alt-full.tsv
rm gwas.zip

mkdir -p data/external/jaspar
curl -fL -o data/external/jaspar/JASPAR_CORE_vertebrates_non-redundant_pfms_jaspar.txt \
  https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt

$PY -u data/acquire_external_cohorts.py --only clinvar

mkdir -p data/external/asm_atlas_natcommun2025 data/external/asm_tycko_gb2020
curl -fL -o data/external/asm_atlas_natcommun2025/41467_2025_57433_MOESM4_ESM.xlsx \
  'https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-57433-1/MediaObjects/41467_2025_57433_MOESM4_ESM.xlsx'
curl -fL -o data/external/asm_tycko_gb2020/13059_2020_2059_MOESM3_ESM.xlsx \
  'https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-020-02059-3/MediaObjects/13059_2020_2059_MOESM3_ESM.xlsx'
```

Expect four `OK` results below. Also check the eGTEx logs for `FAILED`:
```bash
sha256sum -c - <<'SUMS'
d4dbd19fe6d9c2748b6064812b01e7ee2d75d4659279c46d2ec41c18adf8a1ff  data/external/gwas_catalog/gwas-catalog-download-associations-alt-full.tsv
2a6c7c24afb0614ed418c9f02b68845adebf38f80f6cebabbcd5de804eaacb59  data/external/jaspar/JASPAR_CORE_vertebrates_non-redundant_pfms_jaspar.txt
111c5fa049b93a9e732a6eedece1260527d9eae8809fea2d4bde777c48690349  data/external/asm_atlas_natcommun2025/41467_2025_57433_MOESM4_ESM.xlsx
a90c0cb743db32b953958ffb2fd2b10bafad46fe632ec1804fdd4d53e263a9c7  data/external/asm_tycko_gb2020/13059_2020_2059_MOESM3_ESM.xlsx
SUMS
```

Once all 22 GENOA chromosome files have downloaded and no `.part` files remain,
submit the harmonization job:
```bash
ls data/external/genoa_meqtl/meQTL_summarystat_chr*.txt.gz | wc -l
ls data/external/genoa_meqtl/*.part 2>/dev/null
sbatch --export=ALL data/run_harmonize_genoa.sh
```
Wait for all harmonization tasks to finish, then merge their outputs:

```bash
$PY -u data/merge_genoa_harmonized.py --dir data/external/genoa_meqtl/harmonized
```

### 4.4 Saved input files

Extract the archived inputs and API responses used in the analyses, then check
the input checksums:

```bash
tar xzf reproducibility/frozen_inputs.tar.gz
sha256sum -c reproducibility/frozen_inputs_sha256.txt
tar xzf reproducibility/literature_screen_api_cache.tar.gz
```

## 5. Build and run the analyses

Set these variables in every new shell:
```bash
export PY=$HOME/.conda/envs/silentmethyl/bin/python
export SILENTMETHYL_PY=$PY
export SBATCH_EXPORT=ALL
export ABL=results/journal/ablation_breast_epithelium
export DATA=data/datafiles_breast_epithelium
export SPLITS='data/datafiles_breast_epithelium/{split}.csv'
export FUSION_W='checkpoints_journal/seed{seed}/fusion/best_weights.pth'
export SEQ_W='checkpoints_journal/seed{seed}/sequence/best_weights.pth'
mkdir -p logs/{data_build,training,folds,ablation,ablation_analyses,baselines,joint,single_tissue,mechanism,egtex_mt_scoring,melody,melody_st}
```

### 5.1 Training data and splits

```bash
sbatch --export=ALL,OUT_DIR=data/datafiles_breast_epithelium build_data.sh
```

For multi-tissue training, the commands below also need the Lung, KidneyCortex
and ColonTransverse reference tracks and methylation matrices at the listed
paths. These files are present on the cluster. Section 4.2 downloads breast
tracks only, so a fresh checkout still needs the other tissues' inputs.
```bash
B="$PY -u data/build_training_data.py --data-dir data --m-value-precision float32 --val-chroms chr10 chr11 --test-chroms chr8 chr9"
R="-p RM-shared -N1 -n1 -c 32 --mem=64000M -t 06:00:00"
sbatch $R -J build_Lung -o logs/data_build/build_Lung_%j.out --wrap "$B --reference-dir data/reference/Lung \
    --methylation data/targets/TCGA-LUAD.methylation450.tsv.gz data/targets/TCGA-LUSC.methylation450.tsv.gz \
    --out-dir data/datafiles_multitissue/Lung"
sbatch $R -J build_Kidney -o logs/data_build/build_KidneyCortex_%j.out --wrap "$B --reference-dir data/reference/KidneyCortex \
    --methylation data/targets/TCGA-KIRC.methylation450.tsv.gz \
    --out-dir data/datafiles_multitissue/KidneyCortex"
sbatch $R -J build_Colon -o logs/data_build/build_ColonTransverse_%j.out --wrap "$B --reference-dir data/reference/ColonTransverse \
    --methylation data/targets/TCGA-COAD.methylation450.tsv.gz \
    --out-dir data/datafiles_multitissue/ColonTransverse"
```
After the breast, lung, kidney and colon builds finish, combine their splits:

```bash
$PY -u data/compose_multitissue_splits.py --out-dir data/datafiles_joint/all4 --seed 42
$PY -u data/compose_multitissue_splits.py --out-dir data/datafiles_joint/holdout_BreastEpithelium --seed 42 --holdout BreastEpithelium
```

Check the built files; a successful check produces no output:
```bash
grep -E " data/datafiles_(breast_epithelium|multitissue|joint)/" reproducibility/published_outputs_sha256.txt \
  | grep -vE "/(training_data_manifest|candidate_cohort_manifest|composition_manifest|splits_summary)\.json$" \
  | sha256sum -c --quiet
```

Build the variant-scoring inputs in this order:
```bash
ls data/datafiles_breast_epithelium/{train,val,test}.csv
$PY -u data/build_genoa_scoring_input.py \
  --pairs data/external/genoa_meqtl/harmonized/genoa_model_visible_pairs.csv.gz \
  --fasta data/hg38.fa --split-dir data/datafiles_breast_epithelium \
  --output-dir data/external/genoa_meqtl/scoring
$PY -u data/harmonize_egtex_mqtl.py \
  --prefiltered data/external/egtex_breast/egtex_breast_within600.tsv.gz \
  --split-dir data/datafiles_breast_epithelium \
  --output-dir data/external/egtex_breast/scoring
$PY -u data/egtex_significance_threshold.py \
  --perm data/external/egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt \
  --scoring-dir data/external/egtex_breast/scoring \
  --output-dir data/external/egtex_breast/scoring --write-probe-table
SPLIT_DIR=data/datafiles_breast_epithelium bash data/harmonize_egtex_multitissue.sh
```

Check the scoring inputs; expect five `OK` results:
```bash
grep -E ' data/external/(genoa_meqtl|egtex_breast)/' reproducibility/published_outputs_sha256.txt \
  | grep -v '_summary\.json$' | sha256sum -c
```

### 5.2 Training (GPU)

To train from scratch, start below. To reuse existing weights, skip to
**Existing checkpoints**. They are present on the cluster at the paths in
Section 1; copy the complete directories when using another machine.
Run **Sequence baselines** in either case.

```bash
for s in 42 43 44; do sbatch --job-name=SM_seq_s$s --export=ALL,SEED=$s,DATA=$DATA run_baseline.sh; done
sbatch --array=1-3 --export=ALL,SPLIT_ROOT=$DATA/splits,STAGES=sequence scripts/run_folds.sbatch
```

After the sequence-model jobs finish, train the context and fusion models and test the sequence models:
```bash
for s in 42 43 44; do sbatch --job-name=ctx-s$s --export=ALL,SEED=$s scripts/run_context_ablation.sbatch; done
for f in 1 2 3;    do sbatch --job-name=ctx-f$f --export=ALL,FOLD=$f scripts/run_context_ablation.sbatch; done
for s in 42 43 44; do
  sbatch --partition=GPU-shared --gpus=v100-32:1 --cpus-per-task=5 --mem=48G --time=01:00:00 \
      --job-name=seqtest_s$s --output=logs/training/seqtest_s${s}_%j.out \
      --wrap "$PY -u scripts/13_test_model.py --model sequence --seed $s --test_path $DATA/test.csv \
              --weights_path checkpoints_journal/seed$s/sequence/best_weights.pth --output_dir $ABL/seed$s/sequence"
done
```

Train the joint multi-tissue models:
```bash
for cfg in all4 holdout_BreastEpithelium; do
  j=$(sbatch --parsable --export=ALL,CONFIG=$cfg,SEED=42,STAGE=seq --time=47:00:00 scripts/run_joint_multitissue.sbatch)
  sbatch --dependency=afterok:$j --export=ALL,CONFIG=$cfg,SEED=42,STAGE=fuse --time=24:00:00 scripts/run_joint_multitissue.sbatch
done
```

**Sequence baselines:**
```bash
j=$(sbatch --parsable --array=0-1 --export=ALL,DATA=$DATA scripts/run_baseline_grid.sh)
sbatch --dependency=afterok:$j --export=ALL,DATA=$DATA scripts/run_baseline_seeds.sh
sbatch -p RM-shared -N1 -n1 -c 32 --mem=64000M -t 03:00:00 -J baselines_simple \
    -o logs/baselines/simple_%j.out --wrap "$PY -u scripts/14_baselines_simple.py --task both \
      --split-template $SPLITS \
      --input-csv data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv \
      --output-dir results/journal/sequence_baselines"
```

**Existing checkpoints:** use the directories already on the cluster, or copy them to the same paths. Include `checkpoints_joint/` for joint-model evaluation. Every check below should return `OK`:
```bash
grep -E ' checkpoints_(journal|folds|joint)/' reproducibility/published_outputs_sha256.txt | sha256sum -c
```

Evaluate the checkpoints. Run each block separately, starting with the shared settings:
```bash
G="--partition=GPU-shared --gpus=v100-32:1 --cpus-per-task=5 --mem=48G"
T="$PY -u scripts/13_test_model.py"
```
```bash
for s in 42 43 44; do for m in sequence epi fusion; do
  W=checkpoints_journal/seed$s/$m/best_weights.pth
  O=$ABL/seed$s/$m
  sbatch $G --time=01:00:00 -J test_${m}_s$s -o logs/training/test_${m}_s${s}_%j.out \
    --wrap "$T --model $m --seed $s --test_path $DATA/test.csv --weights_path $W --output_dir $O"
done; done
```
```bash
for f in 1 2 3; do for m in epi fusion; do
  W=checkpoints_folds/fold$f/${m}_seed42/best_weights.pth
  O=$ABL/fold$f/$m
  X=$DATA/splits/fold$f/test.csv
  sbatch $G --time=01:00:00 -J test_${m}_f$f -o logs/folds/test_${m}_f${f}_%j.out \
    --wrap "$T --model $m --seed 42 --test_path $X --weights_path $W --output_dir $O"
done; done
```
```bash
for c in all4 holdout_BreastEpithelium; do for m in fusion epi sequence; do
  sbatch $G --time=03:00:00 -J test_${m}_$c -o logs/joint/test_${m}_${c}_%j.out --wrap "$T --model $m --seed 42 \
      --test_path data/datafiles_joint/$c/test.csv --weights_path checkpoints_joint/$c/seed42/$m/best_weights.pth \
      --output_dir results/journal/joint/$c/seed42/$m"
done; done
```

Expect 45 `OK` results and three `FAILED` results for
`seed*/sequence/metrics.json`. The recorded difference is the `test_path` field;
check any other mismatch before continuing:
```bash
P=' results/journal/(ablation_breast_epithelium/(seed4[234]|fold[123])|joint/[^/]+/seed42)/'
grep -E "$P[a-z]+/[a-z_]*(predictions\.csv|metrics\.json)\$" \
  reproducibility/published_outputs_sha256.txt | sha256sum -c
```

### 5.3 Variant scoring (GPU)

Check the sequence weights used by each fusion model; expect `42 OK`, `43 OK` and `44 OK`:
```bash
for s in 42 43 44; do
  want=$(grep -o '"sequence_weights_sha256": *"[0-9a-f]*"' checkpoints_journal/seed$s/fusion/run_config.json | grep -o '[0-9a-f]\{64\}' | head -1)
  got=$(sha256sum checkpoints_journal/seed$s/sequence/best_weights.pth | cut -c1-64)
  [ "$want" = "$got" ] && echo "$s OK" || echo "$s MISMATCH"
done
```

```bash
sbatch --array=0-5 --export=ALL,STAGE=score,MODELS=fusion scripts/run_ablation_analyses.sbatch
sbatch --array=0-5 --export=ALL,STAGE=score,MODELS=sequence scripts/run_ablation_analyses.sbatch
sbatch --partition=GPU-shared --gpus=v100-32:1 --cpus-per-task=5 --mem=48G --time=12:00:00 \
    -J ctxperm -o logs/ablation_analyses/ctxperm_%j.out \
    --wrap "$PY -u scripts/23_context_permutation.py --schemes identity shuffle median \
      --input-csv data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv \
      --stratum heldout --seed 42 --weights-template $FUSION_W --split-template $SPLITS \
      --device cuda --batch-size 32 --output-dir $ABL/context_permutation"
sbatch --array=0-2 --export=ALL,INPUT_CSV=data/external/egtex_multitissue/scoring/Lung/egtex_scoring_input_heldout.csv,OUT_DIR=$ABL/egtex_multitissue_scoring/Lung,WEIGHTS_TEMPLATE='checkpoints_journal/seed{seed}/{model}/best_weights.pth',SPLIT_TEMPLATE="$SPLITS" \
    --output=logs/egtex_mt_scoring/score_%a_%A.out --error=logs/egtex_mt_scoring/score_%a_%A.err scripts/run_egtex_multitissue_scoring.sh
```

After scoring finishes, expect 15 `OK` results in the first check. After `ctxperm` finishes, expect four in the second:
```bash
P=' results/journal/ablation_breast_epithelium/(genoa_variant_scoring|egtex_variant_scoring|egtex_multitissue_scoring/Lung)/'
grep -E "$P.*pair_scores\.csv\$" reproducibility/published_outputs_sha256.txt | sha256sum -c
```
```bash
grep ' results/journal/ablation_breast_epithelium/context_permutation/' reproducibility/published_outputs_sha256.txt \
  | grep -v 'summary\.json$' | sha256sum -c
```

Score variants with CpGenie and DeepCpG after their baseline weights are ready.
On the cluster, these are in `results/journal/published_baselines/{cpgenie,deepcpg}/seed*/weights.pth`.
Keep `DATA` set:
```bash
echo "DATA=$DATA"
for a in cpgenie deepcpg; do
  for c in "genoa_meqtl/scoring/genoa_scoring_input_heldout.csv heldout" \
           "egtex_breast/scoring/egtex_scoring_input_heldout.csv heldout_egtex"; do
    set -- $c
    sbatch --partition=GPU-shared --gpus=v100-32:1 --cpus-per-task=5 --mem=32G --time=01:00:00 \
        --job-name=bl_${a}_$2 --output=logs/baselines/score_${a}_$2_%j.out \
        --wrap "$PY -u scripts/15_baselines_published.py --arch $a --score-pairs --pairs data/external/$1 \
                --stratum $2 --test $DATA/test.csv --output-dir results/journal/published_baselines"
  done
done
```

After all four `bl_*` jobs finish, move the eGTEx scores to the expected directory. Expect no `git status` output:
```bash
src=results/journal/published_baselines/variant_scoring/heldout_egtex
dst=results/journal/published_baselines_egtex/variant_scoring/heldout
ls $src/cpgenie/seed44 $src/deepcpg/seed44 && mkdir -p $dst && cp -r $src/. $dst/ && rm -r $src
git status --short results/journal/published_baselines*
```

Score the combined eGTEx input with the sequence model:
```bash
sbatch --array=3-5 --export=ALL,SPLIT_TEMPLATE="$SPLITS" scripts/run_egtex_multitissue_scoring.sh
```
After the array finishes, separate its predictions by tissue:

```bash
$PY -u data/split_predictions_by_tissue.py --models sequence
```

### 5.4 Candidates, mQTL controls and variant applications (GPU)

After Section 5.3 finishes, submit these jobs:
```bash
j60=$(sbatch --parsable jobs/single_tissue/60_candidates.sbatch)
sbatch --dependency=afterok:$j60 jobs/single_tissue/62_comparison.sbatch
j70=$(sbatch --parsable jobs/single_tissue/70_mqtl_positive.sbatch)
sbatch --dependency=afterok:$j70 jobs/single_tissue/71_mqtl_negative.sbatch
j63=$(sbatch --parsable jobs/single_tissue/63_known_variant.sbatch)
sbatch --dependency=afterok:$j63 jobs/single_tissue/64_literature.sbatch
```

A job may print `[!] other jobs ran in this tree meanwhile` and exit 0 if another
job wrote files at the same time. Check that the listed paths belong to the other
jobs. After the jobs finish, the check below should print `ALL OK`:
```bash
P=' results/journal/ablation_breast_epithelium/(candidates|known_variant_application|egtex_mqtl_positive_control|egtex_mqtl_matched_negative|literature_variant_screen)/'
X='summary\.json$|hm450_probe_overlap_audit\.json$|top_candidate_case_study\.csv$|stk11_case_study_figure_values\.csv$'
grep -E "$P" reproducibility/published_outputs_sha256.txt | grep -vE "$X" | sha256sum -c --quiet && echo ALL OK
```

### 5.5 Gate components, context changes and ASM validation

Build the ASM inputs on CPU; expect four `OK` results:
```bash
$PY -u scripts/53_asm_build.py --asm-table data/external/asm_atlas_natcommun2025/41467_2025_57433_MOESM4_ESM.xlsx \
    --chain data/reference/hg19ToHg38.over.chain.gz --genome data/hg38.fa --reference-dir data/reference/BreastEpithelium \
    --phylop data/reference/hg38.phyloP100way.bw --imputation $DATA/feature_imputation.json \
    --bimodal-bed data/external/asm_atlas/bimodal_chr8_chr9.bed --distance-tolerance-bp 10 --seed 42 \
    --output-dir data/external/asm_atlas/scoring
$PY -u scripts/53d_tycko_build.py --table data/external/asm_tycko_gb2020/13059_2020_2059_MOESM3_ESM.xlsx \
    --ensembl-cache data/external/asm_tycko_gb2020/ensembl_grch38_chr8_chr9_cache.csv \
    --chain data/reference/hg19ToHg38.over.chain.gz --genome data/hg38.fa --reference-dir data/reference/BreastEpithelium \
    --phylop data/reference/hg38.phyloP100way.bw --imputation $DATA/feature_imputation.json \
    --distance-tolerance-bp 10 --seed 42 --output-dir data/external/asm_tycko_gb2020/scoring
grep -E ' data/external/asm_(atlas|tycko_gb2020)/scoring/' reproducibility/published_outputs_sha256.txt \
  | grep -v 'build_summary\.json$' | sha256sum -c
```

After Sections 5.3 and 5.4 finish, run the GPU analyses:
```bash
sbatch jobs/mechanism/53_asm_score.sbatch
sbatch jobs/mechanism/53d_tycko_score.sbatch
sbatch jobs/mechanism/23_ctx_ladder.sbatch
sbatch --array=0-1 jobs/mechanism/54_gate_instrument.sbatch
```

After those jobs finish, check the outputs; expect `ALL OK`:
```bash
P=' results/journal/(ablation_breast_epithelium/(gate_decomposition|context_ladder)|asm_validation|asm_validation_tycko)/'
grep -E "$P" reproducibility/published_outputs_sha256.txt | grep -v 'summary\.json$' | sha256sum -c --quiet && echo ALL OK
```

### 5.6 CPU analyses, figures and supplements

To save checks in a separate directory, first run
`export OUT=repro_check/abl OUTJ=repro_check/journal OUTR=repro_check/results`.
Otherwise, these commands write to the standard results directories.

With the separate paths, script 93 reports four S8–S10 CSVs as `MISSING`, and
script 90 reports `INCOMPLETE`. These messages are expected for this check.

<!-- cpu-validate:begin -->
```bash
OUT=${OUT:-$ABL}; OUTJ=${OUTJ:-results/journal}; OUTR=${OUTR:-results}
mkdir -p "$OUT" "$OUTJ" "$OUTR"

$PY -u scripts/16_paired_model_bootstrap.py --seeds 42 43 44 --models epi sequence fusion \
    --prediction-template "$ABL/seed{seed}/{model}/predictions.csv" \
    --block-size-bp 1000000 --bootstrap-replicates 5000 --random-seed 42 --output-dir "$OUT/paired_model_bootstrap"
$PY -u scripts/22_context_stratification.py --test-path $DATA/test.csv \
    --prediction-template "$ABL/seed{seed}/{model}/predictions.csv" \
    --cpg-island-annotation data/HM450.hg38.manifest.CpGIsland.tsv.gz --gencode-gtf data/reference/gencode.v44.annotation.gtf.gz \
    --candidate-path "$ABL/candidates/candidate_matched_background_statistics.csv" \
    --mqtl-path "$ABL/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv" \
    --seeds 42 43 44 --models epi sequence fusion --block-size-bp 1000000 --bootstrap-replicates 2000 --random-seed 20260814 \
    --output-dir "$OUT/biological_context"
$PY -u scripts/01_target_qc.py --matrix data/TCGA-BRCA.methylation450.tsv.gz --test $DATA/test.csv \
    --prediction-template "$ABL/seed42/{model}/predictions.csv" --manifest data/HM450.hg38.manifest.tsv.gz \
    --split-csvs $DATA/train.csv $DATA/val.csv $DATA/test.csv --output-dir "$OUT/target_qc"

$PY -u scripts/21_variant_evaluation.py --scores-dir "$ABL/genoa_variant_scoring" --cohort GENOA --stratum heldout \
    --models fusion sequence --seeds 42 43 44 --significance 5e-8 --n-boot 2000 --match-tolerance 10 --match-ratio 1 \
    --random-seed 42 --output-dir "$OUT/genoa_variant_evaluation"
$PY -u scripts/21_variant_evaluation.py --scores-dir "$ABL/egtex_variant_scoring" --cohort eGTEx --stratum heldout \
    --models fusion sequence --seeds 42 43 44 --significance 1.483e-5 --n-boot 2000 --match-tolerance 10 --match-ratio 1 \
    --random-seed 42 --output-dir "$OUT/egtex_variant_evaluation"
$PY -u scripts/30_transfer_synthesis.py --cohort "GENOA:$ABL/genoa_variant_scoring:5e-8" \
    --cohort "eGTEx:$ABL/egtex_variant_scoring:1.483e-5" --stratum heldout --models fusion sequence --seeds 42 43 44 \
    --n-boot 500 --random-seed 42 --output-dir "$OUT/variant_effect_synthesis"
$PY -u scripts/31_transfer_discrimination.py --reference "$ABL/genoa_variant_scoring::fusion::42,43,44" \
    --compare "results/journal/published_baselines/variant_scoring::deepcpg::42,43,44" \
    --compare "results/journal/published_baselines/variant_scoring::cpgenie::42,43,44" \
    --output-dir "$OUT/paired_model_comparison_genoa"
$PY -u scripts/31_transfer_discrimination.py --reference "$ABL/egtex_variant_scoring::fusion::42,43,44" \
    --compare "results/journal/published_baselines_egtex/variant_scoring::deepcpg::42,43,44" \
    --compare "results/journal/published_baselines_egtex/variant_scoring::cpgenie::42,43,44" \
    --significance 1.483e-5 --output-dir "$OUT/paired_model_comparison_egtex"

$PY -u scripts/31_transfer_discrimination.py --reference "$ABL/egtex_multitissue_scoring/Lung::fusion::42,43,44" \
    --compare "results/journal/egtex_multitissue_scoring/by_tissue/Lung::sequence::42,43,44" \
    --stratum heldout --n-boot 500 --match-tolerance 10 --match-ratio 1 --random-seed 42 \
    --output-dir "$OUT/transfer_discrimination/Lung"

$PY -u scripts/40_meqtl_tissue_specificity.py --stage matched --cohort "GENOA:$ABL/genoa_variant_scoring:5e-8" \
    --cohort "eGTEx:$ABL/egtex_variant_scoring:1.483e-5" --split-template "$SPLITS" --model fusion --seeds 42 43 44 \
    --n-boot 1000 --random-seed 42 --output-dir "$OUT/tissue_shared_meqtls"
$PY -u scripts/40_meqtl_tissue_specificity.py --stage chromatin --cohort "GENOA:$ABL/genoa_variant_scoring:5e-8" \
    --cohort "eGTEx:$ABL/egtex_variant_scoring:1.483e-5" --split-template "$SPLITS" --model fusion --seeds 42 43 44 \
    --n-boot 1000 --random-seed 42 --output-dir "$OUT/meqtl_class_chromatin"
$PY -u scripts/41_tissue_specificity_summary.py --results "$OUT/tissue_shared_meqtls" --out "$OUT/tissue_shared_meqtls" \
    --label "SilentMethyl (breast-epithelium context)"

$PY -u scripts/51_rc_uncertainty.py --stage base --results-root "$ABL" --seeds 42 43 44 --models sequence fusion epi \
    --bootstrap-replicates 2000 --random-seed 20260824 --output-dir "$OUT/rc_uncertainty"
for k in 10 20 50; do
  sfx=$([ $k = 10 ] && echo "" || echo "_s$k")
  $PY -u scripts/51_rc_uncertainty.py --stage conditional --results-root "$ABL" --seeds 42 43 44 \
      --models sequence fusion epi --strata $k --output-dir "$OUT/rc_uncertainty_conditional$sfx"
done
$PY -u scripts/51_rc_uncertainty.py --stage figure --runs "10:$OUT/rc_uncertainty_conditional" \
    "20:$OUT/rc_uncertainty_conditional_s20" "50:$OUT/rc_uncertainty_conditional_s50" --output-dir "$OUT/rc_uncertainty_figure"
$PY -u scripts/52_gwas_enrichment.py --build --gwas data/external/gwas_catalog/gwas-catalog-download-associations-alt-full.tsv \
    --scores-dir "$ABL/genoa_variant_scoring" --stratum heldout --model fusion --seeds 42 43 44 \
    --output-dir "$OUT/gwas_regulatory_enrichment"
$PY -u scripts/52_gwas_enrichment.py --analyse --scores-dir "$ABL/genoa_variant_scoring" --stratum heldout \
    --model fusion --seeds 42 43 44 --output-dir "$OUT/gwas_regulatory_enrichment"
$PY -u scripts/50_motif_disruption.py --scores-dir "$ABL/genoa_variant_scoring" --stratum heldout --model fusion \
    --seeds 42 43 44 --split-template "$SPLITS" \
    --jaspar data/external/jaspar/JASPAR_CORE_vertebrates_non-redundant_pfms_jaspar.txt \
    --n-boot 500 --random-seed 42 --output-dir "$OUT/motif_disruption"

$PY -u scripts/57_fusion_gain_stratified.py --test-path $DATA/test.csv \
    --prediction-template "$ABL/seed{seed}/{model}/predictions.csv" \
    --cpg-island-annotation data/HM450.hg38.manifest.CpGIsland.tsv.gz --seeds 42 43 44 \
    --block-size-bp 1000000 --bootstrap-replicates 2000 --random-seed 20260915 --output-dir "$OUT/fusion_gain_stratified"
for strat in observed predicted; do
  $PY -u scripts/58_ladder_effect_size_stratification.py --ladder-dir "$ABL/context_ladder" \
      --stratifier $strat --output-dir "$OUT/context_ladder_stratified"
done
if [ "$OUT" != "$ABL" ]; then
  mkdir -p "$OUT/gate_decomposition" && cp -r "$ABL/gate_decomposition/genoa" "$ABL/gate_decomposition/egtex" "$OUT/gate_decomposition/"
fi
$PY -u scripts/54_gate_decomposition.py analyse --cohorts genoa egtex --seed 42 --bootstrap-draws 1000 \
    --pair-scores-root "$ABL/{cohort}_variant_scoring/heldout" --output-dir "$OUT/gate_decomposition"
$PY -u scripts/55_gate_plasticity.py --gates-csv checkpoints_joint/all4/seed42/fusion/best_validation_gates.csv \
    --joint-val-csv data/datafiles_joint/all4/val.csv --seed 42 --bootstrap-draws 1000 --output-dir "$OUTJ/joint/gate_plasticity"
$PY -u scripts/56_transfer_failure.py --pred-root results/journal/joint/holdout_BreastEpithelium/seed42 \
    --cgi-manifest data/HM450.hg38.manifest.CpGIsland.tsv.gz --breast-test-csv $DATA/test.csv --seed 42 \
    --bootstrap-draws 400 --output-dir "$OUTJ/joint/transfer_failure"
$PY -u scripts/53c_asm_evaluate.py --scores-csv results/journal/asm_validation/asm_pair_scores.csv \
    --block-size-bp 1000000 --bootstrap-replicates 2000 --random-seed 20260915 --output-dir "$OUTJ/asm_validation"
$PY -u scripts/53e_tycko_evaluate.py --scores-csv results/journal/asm_validation_tycko/tycko_pair_scores.csv \
    --block-size-bp 1000000 --bootstrap-replicates 2000 --random-seed 20260915 --strong-effect-pp 20.0 \
    --output-dir "$OUTJ/asm_validation_tycko"

$PY -u scripts/91_build_manuscript_figures.py \
    --metrics-path "$OUT/paired_model_bootstrap/model_metrics_recomputed.csv" \
    --context-path "$OUT/biological_context/fusion_gain_by_context.csv" \
    --variant-context-path "$OUT/biological_context/variant_response_by_context.csv" \
    --variant-distance-path "$OUT/biological_context/variant_response_by_distance.csv" \
    --mqtl-path "$ABL/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv" \
    --candidate-path "$ABL/candidates/candidate_matched_background_statistics.csv" \
    --candidate-seed-path "$ABL/candidates/candidate_seed_scores_long.csv" \
    --comparator-path "$ABL/candidates/top_candidate_matched_comparators_long.csv" \
    --case-study-output "$OUT/candidates/top_candidate_case_study.csv" \
    --literature-variant-path "$ABL/literature_variant_screen/literature_variant_predictions_ranked.csv" \
    --output-dir "$OUT/manuscript_figures" \
    --stk11-case-output "$OUT/literature_variant_screen/stk11_case_study_figure_values.csv" \
    --genomic-region-output "$OUT/biological_context/plots/fusion_gain_by_genomic_region.png" \
    --information-gains-output "$OUT/biological_context/plots/information_source_gains.png"
$PY -u scripts/92_build_r8_figures.py \
    --fusion-gain "$OUT/fusion_gain_stratified/fusion_gain_stratified.csv" \
    --ladder-agreement "$ABL/context_ladder/agreement_with_identity.csv" \
    --ladder-levels "$ABL/context_ladder/level_accuracy_by_scheme.csv" \
    --transfer-failure "$OUTJ/joint/transfer_failure/transfer_failure_summary.json" \
    --asm-discrimination "$OUTJ/asm_validation/asm_discrimination.csv" \
    --gate-decomposition "$OUT/gate_decomposition/gate_decomposition_summary.json" \
    --output-dir "$OUTJ/manuscript_figures_r8"

$PY -u scripts/93_build_r8_supplement.py --output-dir "$OUTR/supplementary_package_r8"
$PY -u scripts/90_build_supplement_package.py --project-root . --output-dir "$OUTR/supplementary_package" --allow-missing-required
```
<!-- cpu-validate:end -->

### 5.7 Melody (optional, not in the manuscript)

```bash
sbatch --export=ALL,MELODY_PY=/path/to/melody/env/bin/python scripts/run_melody_union.sbatch
sbatch --export=ALL,MELODY_PY=/path/to/melody/env/bin/python scripts/run_melody_st.sbatch
```

### 5.8 Figures and manuscript

Rebuild the figures and source data. Expect no `git status` output if they match
the saved versions:
```bash
[ -d repro_check/abl ] && cp -rn repro_check/abl/. "$ABL"/
$PY -u scripts/95_build_revision_figures.py \
    --repo-root . \
    --figures-output figures \
    --source-data-output figures/source_data
git status --short figures
```


## 6. Interpretation and package checks
Check that the supplement package uses the intended context data:

```bash
$PY -u scripts/93_build_r8_supplement.py --check-package results/supplementary_package_r8
```

## 7. Compare with the saved outputs

### 7.1 Checksums

After running Section 5.6 with outputs in `repro_check/`, check for differences:
```bash
H=reproducibility/published_outputs_sha256.txt
sed 's#  results/journal/ablation_breast_epithelium/#  repro_check/abl/#' $H | grep '  repro_check/abl/' \
  | sha256sum -c --ignore-missing 2>/dev/null | grep -v ': OK$'
sed 's#  results/journal/#  repro_check/journal/#' $H \
  | grep -E '  repro_check/journal/(asm_validation|asm_validation_tycko|joint|manuscript_figures_r8)/' \
  | sha256sum -c --ignore-missing 2>/dev/null | grep -v ': OK$'
sed 's#  results/supplementary_package_r8/#  repro_check/results/supplementary_package_r8/#' $H \
  | grep '  repro_check/results/supplementary_package_r8/' \
  | sha256sum -c --ignore-missing 2>/dev/null | grep -v ': OK$'
```

Expected differences include `*summary.json`, `decision.json`,
`preregistration.json`, PDF `CreationDate`, and the R8 package's
`SHA256SUMS.txt`/`package_summary.json`. Small numerical differences are also
expected: the last decimal place in `fusion_gain_stratified.csv` and about
1e-6 in sequence-model rows of `transfer_discrimination/Lung/*.csv`.

After changing a tracked script, update the code checksum list. All checks should return `OK`:
```bash
git ls-files -- '*.py' '*.sh' '*.sbatch' | grep -v '^overleaf/' | sort | xargs sha256sum \
  > reproducibility/analysis_code_sha256.txt
sha256sum -c reproducibility/analysis_code_sha256.txt
```

### 7.2 Reference metrics for retrained models

Use these values as a reference; retrained results should be close to about three decimal places.

| run | context-only β MAE / AUC | sequence-only β MAE / AUC | fusion β MAE / AUC |
|---|---|---|---|
| seed 42 | 0.1022 / 0.9658 | 0.1090 / 0.9576 | 0.0885 / 0.9765 |
| seed 43 | 0.1022 / 0.9658 | 0.1103 / 0.9568 | 0.0901 / 0.9751 |
| seed 44 | 0.1023 / 0.9658 | 0.1104 / 0.9562 | 0.0956 / 0.9717 |
| fold 1 | 0.0980 / 0.9707 | — | 0.0873 / 0.9773 |
| fold 2 | 0.0985 / 0.9679 | — | 0.0934 / 0.9724 |
| fold 3 | 0.0987 / 0.9703 | — | 0.0862 / 0.9779 |
