# Reproducing SilentMethyl from a fresh clone

This is the single, complete path from `git clone` to every published number,
figure and supplementary table. Every command is written with explicit flags and
runs from the repository root. Where a step is too expensive to rerun, the command
is still given in full, together with the checksum or metric that lets you confirm
the shipped output is what that command produces.

The **published model** uses primary **breast-epithelium** chromatin context
(ENCODE, seven fold-change bigWigs). An earlier model used MCF-10A context. That
model was superseded on 11 Sep 2026 and survives only as a historical record
(section 9). Nothing in this document produces or reads an MCF-10A product.

Contents

1. [Which arms consume context](#1-which-arms-consume-context)
2. [Environment](#2-environment)
3. [Hardware and time budget](#3-hardware-and-time-budget)
4. [Input data](#4-input-data)
5. [Build order](#5-build-order)
6. [Published-path traps](#6-published-path-traps)
7. [Verifying against the shipped outputs](#7-verifying-against-the-shipped-outputs)
8. [What was executed to validate this document](#8-what-was-executed-to-validate-this-document)
9. [The superseded MCF-10A record](#9-the-superseded-mcf-10a-record)
10. [Known gaps](#10-known-gaps)

---

## 1. Which arms consume context

SilentMethyl has three arms. Only two of them read the seven chromatin-context
columns (`Ref_{ATAC,H3K4me3,H3K27ac,H3K27me3,H3K9me3,H3K36me3,H3K4me1}_Signal` and
their `_Missing` flags):

| arm | model class | reads context columns? | retrained for breast epithelium? | published weights |
|---|---|---|---|---|
| sequence-only | `SequenceOnlyModel` (DNABERT-2), `forward(input_ids, attention_mask)` | **no** | **no, and it must not be** | `checkpoints_journal/seed{42,43,44}/sequence`, `checkpoints_folds/fold{1,2,3}/sequence_seed42` |
| context-only | `EpigeneticOnlyModel`, `forward(tab, tab_missing)` | yes | yes | `checkpoints_ablation/breast_epithelium/{seed42,seed43,seed44,fold1,fold2,fold3}/epi` |
| gated fusion | `FusionModel` (sequence tower + context tower + gate) | yes | yes | `checkpoints_ablation/breast_epithelium/{seed42,seed43,seed44,fold1,fold2,fold3}/fusion` |
| k-mer ridge, composition, CpGenie, DeepCpG | sequence baselines | no | n/a | `results/journal/{sequence_baselines,published_baselines}` |
| Melody-MT / Melody-ST | authors' checkpoints | no (Melody's own tracks) | n/a | external |

**Why the sequence towers live under `checkpoints_journal/` and are not
contaminated.** They were trained from `data/datafiles/`, the build made with the
superseded MCF-10A tracks. Two facts, both verified on 16 Sep 2026, make that
irrelevant:

1. The model cannot see context. `SequenceOnlyModel.forward` takes token ids and
   an attention mask only (`scripts/training_common.py`). No tabular tensor
   reaches it.
2. The inputs it does see are byte-identical between builds. Between
   `data/datafiles/` and `data/datafiles_breast_epithelium/`, every FASTA is
   byte-identical. Across `train.csv`, `val.csv`, `test.csv`, `testing_data.csv`
   and `testing_data_test_only.csv`, a per-column hash shows that the only
   differing columns are the seven `Ref_*_Signal` tracks and some of their
   `_Missing` flags. Sequences, targets, probe IDs, row order and phyloP are
   identical.

A context swap therefore cannot change a context-blind model. The towers were not
retrained on purpose, and every breast-epithelium fusion model is built on top of
them. A path list that contains `checkpoints_journal/…/sequence` is **not**
evidence of MCF-10A contamination. The epi and fusion weights under
`checkpoints_journal/` and `checkpoints_folds/` **are** MCF-10A products, and
nothing published reads them.

---

## 2. Environment

| | |
|---|---|
| Python | **3.10.20** |
| conda env | `silentmethyl` |
| PyTorch | 2.6.0 + CUDA 12.4 (`torch==2.6.0+cu124`) |
| pins | `requirements.txt` (transformers 5.8.0, pandas 2.3.3, numpy 2.2.6, pyBigWig 0.3.25, …) |
| full record | `reproducibility/environment_snapshot.txt` |

```bash
conda create --name silentmethyl python=3.10.20 -y
conda activate silentmethyl
python -m pip install --upgrade pip
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

**`python` is not on `PATH` on a Bridges-2 login node** (only `/usr/bin/python3`,
which lacks pandas and torch). Always use the environment's interpreter by full
path. Every command below is written as `$PY`:

```bash
export PY=$HOME/.conda/envs/silentmethyl/bin/python   # adjust to your conda prefix
export SILENTMETHYL_PY=$PY                            # every sbatch/sh job honours this
$PY --version                                         # Python 3.10.20
```

DNABERT-2 backbone, read offline from `dnabert2_local/` by every GPU script:

```bash
$PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("zhihan1996/DNABERT-2-117M", local_dir="dnabert2_local")
EOF
```

The Melody comparison (`scripts/33_melody_scoring.py`, `scripts/run_melody_*.sbatch`)
needs the authors' repository cloned to `data/external/melody/Melody_repo` and its
own environment (frozen in `data/external/melody/melody_env_freeze.txt`), exported
as `MELODY_PY`.

Slurm directives (`--partition`, `--gpus=v100-32:1`, `module load`) are Bridges-2
specific. Adjust them for your scheduler. Submit every job from the repository
root with `--export=ALL,...`: without `ALL`, Slurm drops the environment and the
interpreter is not found. Create each `logs/<dir>` before `sbatch`, because Slurm
opens `--output` before the script runs, and a missing directory kills the job in
~5 s with no log.

---

## 3. Hardware and time budget

Measured wall times from the published runs (`sacct`), Bridges-2:

| step | hardware | wall time | count |
|---|---|---|---|
| input downloads | network, ~70 GB | hours | once |
| context build (`build_data.sh`) | 32 CPU, 64 GB RAM | **~25 min** (validated 16 Sep: see §8) | once |
| **sequence tower, published split** | 1× V100-32 GPU | **~26 h each** | ×3 seeds |
| **sequence tower, folds 1–3** | 1× V100-32 GPU | **~32 h each** | ×3 folds |
| **context + fusion towers** (`run_context_ablation.sbatch`) | 1× V100-32 GPU | **~13–14 h each** | ×6 (3 seeds + 3 folds) |
| **joint multi-tissue model** (seq stage + fuse stage) | 1× V100-32 GPU | **~34 h + ~15 h** | ×2 configs |
| CpGenie/DeepCpG grid + seeds | 1× V100 | 19 min + ~7 min each | 2 + 6 |
| variant scoring, GENOA / eGTEx | 1× V100 | ~1.1 h per model×seed | ×12 |
| eGTEx Lung transfer scoring | 1× V100 | ~1.2 h | ×3 |
| context permutation / dose-response ladder | 1× V100 | ~4 h / ~4.9 h | once each |
| gate instrumentation | 1× V100 | ~1 h | ×2 cohorts |
| candidate + application GPU jobs (60, 62, 63, 64, 70, 71) | 1× V100 | 1–11 min each | once each |
| ASM scoring (Rosenski, Do & Tycko) | 1× V100 | ~0.5 h | ×2 |
| Melody-MT / Melody-ST scoring | 1× V100 | 25 min / 3.7 h | once each |
| all CPU analyses, figures, supplements (§5.6) | 4–8 CPU | ~1 h total (motif scan dominates) | once |

**The GPU trainings total roughly 350 GPU-hours.** Everything after training is
under 40 GPU-hours plus about an hour of CPU.

---

## 4. Input data

Everything is public. Nothing controlled-access is used anywhere. After
downloading, verify with `data/external/external_manifest.json` (URL, SHA-256,
bytes and build for each cohort) and the checksum files named below.

### 4.1 Genome, array and reference resources

| path | source | how to get it |
|---|---|---|
| `data/hg38.fa`, `data/hg38.fa.fai` | UCSC hg38 | `hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz`, then `gunzip`, `samtools faidx` |
| `data/HM450.hg38.manifest.tsv.gz`, `data/HM450.hg38.manifest.CpGIsland.tsv.gz` | Zhou et al. InfiniumAnnotation | `zwdzwd.github.io/InfiniumAnnotation` (HM450, hg38) |
| `data/TCGA-BRCA.methylation450.tsv.gz` | UCSC Xena GDC hub | `https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.methylation450.tsv.gz` |
| `data/reference/hg38.phyloP100way.bw` | UCSC | `hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw` |
| `data/reference/gencode.v44.annotation.gtf.gz` | GENCODE v44 | `ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/` |
| `data/reference/hg19ToHg38.over.chain.gz` | UCSC liftOver | `hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/` |
| `data/reference/ENCFF356LFX.bed.gz` | **ENCODE GRCh38 exclusion list** (Amemiya et al. 2019) | `https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz` |

SHA-256 of all of these (and of the seven context tracks) as used:
`reproducibility/reference_audit_breast_epithelium.txt`.

The exclusion list is **not a context track and is not applied** to the published
context. The breast-epithelium bigWigs are used exactly as ENCODE released them.
It is read only by `data/profile_bigwig.py` (track audits) and by
`data/acquire_multitissue_inputs.py --blacklist` when converting BAMs.

### 4.2 The seven breast-epithelium context tracks (the published context)

`data/reference/BreastEpithelium/`, biosample *breast epithelium*, GRCh38, output
type *fold change over control*, all released:

| feature | file | accession | experiment | ENCODE md5 |
|---|---|---|---|---|
| ATAC-seq | `ATAC_seq.bw` | ENCFF665NGK | ENCSR955JSO | `7effdce98e749c91009b74c6eac225cd` |
| H3K4me3 | `H3K4me3.bw` | ENCFF653CLL | | `0744ffdccc18c4bbba4f1054eb6cf0d7` |
| H3K27ac | `H3K27ac.bw` | ENCFF085IYD | ENCSR081OTO | `5ddb4795f5bcd88dd75ebb874a33199d` |
| H3K27me3 | `H3K27me3.bw` | ENCFF212ZFW | ENCSR134LLK | `6c4830851a98900cd213db67ed57a892` |
| H3K9me3 | `H3K9me3.bw` | ENCFF481QEK | | `36b8acf0bedfb712692d8117b1f971e0` |
| H3K36me3 | `H3K36me3.bw` | ENCFF714QJF | ENCSR793QCL | `706a5be148d96239ca0ba4cb819af011` |
| H3K4me1 | `H3K4me1.bw` | ENCFF234JZW | | `cbfcb57dbc1de468d185392839f8775b` |

Full records, including every experiment ID: `data/reference/BreastEpithelium/TRACK_SET.md`
and `data/multitissue_acquisition_plan.json`.

```bash
# resolve against the live portal and write a plan (no download); checks md5s, assay, assembly
$PY -u data/acquire_multitissue_inputs.py --tissues BreastEpithelium --context-only \
    --picks data/multitissue_picks.tsv --plan data/multitissue_acquisition_plan.json
# download (10.4 GB), md5-verified on arrival
$PY -u data/acquire_multitissue_inputs.py --tissues BreastEpithelium --context-only \
    --picks data/multitissue_picks.tsv --plan data/multitissue_acquisition_plan.json --apply
```

**Joint multi-tissue model only:** the same script with `--tissues Lung
ColonTransverse KidneyCortex` fetches the other three context sets into
`data/reference/<Tissue>/` (each has its own `TRACK_SET.md`) and their TCGA
normal-tissue targets into `data/targets/` (`TCGA-{KIRC,LUAD,LUSC,COAD,READ}`).

### 4.3 Cohorts and catalogues

| input | path | source | fetch / build |
|---|---|---|---|
| GENOA meQTL summary statistics (hg19) | `data/external/genoa_meqtl/` | GENOA, public summary statistics | `$PY -u data/acquire_external_cohorts.py --only genoa_meqtl`; then `sbatch data/run_harmonize_genoa.sh` (array over chr1–22, liftOver to hg38); then `$PY -u data/merge_genoa_harmonized.py --dir data/external/genoa_meqtl/harmonized` |
| GENOA scoring input | `data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv` | derived | `$PY -u data/build_genoa_scoring_input.py --pairs data/external/genoa_meqtl/harmonized/genoa_model_visible_pairs.csv.gz --fasta data/hg38.fa --split-dir data/datafiles_breast_epithelium --output-dir data/external/genoa_meqtl/scoring` |
| eGTEx Breast Mammary Tissue mQTL | `data/external/egtex_breast/` | eGTEx (GTEx portal, open access) | `bash data/fetch_egtex_regular_slices.sh` (streams ~45 GB, keeps the ±600 bp slice); `BreastMammaryTissue.regular.perm.fdr.txt` from the GTEx portal eGTEx mQTL bundle |
| eGTEx scoring input | `data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv` | derived | `$PY -u data/harmonize_egtex_mqtl.py --prefiltered data/external/egtex_breast/egtex_breast_within600.tsv.gz --split-dir data/datafiles_breast_epithelium --output-dir data/external/egtex_breast/scoring` |
| eGTEx significance threshold 1.483e-5 | constant | derived | `$PY -u data/egtex_significance_threshold.py --perm data/external/egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt --scoring-dir data/external/egtex_breast/scoring --output-dir repro_check/egtex_threshold` |
| eGTEx, eight transfer tissues | `data/external/egtex_multitissue/` | eGTEx | `bash data/fetch_egtex_multitissue.sh`; `SPLIT_DIR=data/datafiles_breast_epithelium bash data/harmonize_egtex_multitissue.sh` |
| GWAS Catalog, all associations | `data/external/gwas_catalog/gwas-catalog-download-associations-alt-full.tsv` | EBI GWAS Catalog | download from `ebi.ac.uk/gwas/docs/file-downloads`; verify `sha256sum -c data/external/gwas_catalog/SHA256SUMS.txt` (the catalogue is updated weekly, so a new download will differ) |
| JASPAR CORE vertebrates, non-redundant | `data/external/jaspar/` | JASPAR | `$PY -u data/acquire_external_cohorts.py --only jaspar` |
| ClinVar (literature screen) | `data/external/clinvar/clinvar_GRCh38.vcf.gz` | NCBI | `$PY -u data/acquire_external_cohorts.py --only clinvar` |
| Rosenski et al. 2025 ASM atlas | `data/external/asm_atlas_natcommun2025/` | Nat Commun 16:2141 supplementary | URLs, sheet names and header rows in `SOURCE.txt` |
| Do & Tycko 2020 ASM catalogue | `data/external/asm_tycko_gb2020/` | Genome Biol 21:153 supplementary | URL and sheet in `SOURCE.txt` |
| TCGA ancestry calls | `data/external/tcga_ancestry/` | Carrot-Zhang et al. 2020 (public) | `$PY -u data/build_tcga_ancestry_labels.py` |

`$PY -u data/acquire_external_cohorts.py --check` lists every source and probes
each endpoint. Sources without a deterministic URL are reported as MANUAL, with
the exact destination path.

### 4.4 Frozen inputs shipped in the repository

These cannot be re-derived from a live service today and give the same bytes, so
the exact files used are committed:

| file | why frozen |
|---|---|
| `reproducibility/frozen_inputs.tar.gz` → `data/datafiles/gdc_tcga_brca_synonymous_raw.json.gz` | GDC API response for the TCGA-BRCA synonymous-variant pool. GDC data releases change it. `data/build_testing_data.py` reads it from this fixed path in every context build |
| … → `data/external/asm_atlas/bimodal_chr8_chr9.bed` | chr8/chr9 bimodal regions extracted from Rosenski MOESM3 (531 MB zip). **The extraction code was not preserved** |
| … → `data/external/asm_tycko_gb2020/ensembl_grch38_chr8_chr9_cache.csv` | rsID → GRCh38 positions from the Ensembl REST API. **The resolver code was not preserved** |
| … → `data/egtex_breast_mqtl_{heldout,heldout_qc,model_visible}.csv` | eGTEx positive-control cohort as used by `70` and `64` |
| … → `data/external/egtex_multitissue/scoring/union_scoring_input_heldout.csv` | union of the eight tissues' scoring inputs for Melody-MT. **The builder was not preserved** |
| `reproducibility/literature_screen_api_cache.tar.gz` | NCBI/ClinVar API responses behind the literature screen (`64 --offline`) |

```bash
tar xzf reproducibility/frozen_inputs.tar.gz
sha256sum -c reproducibility/frozen_inputs_sha256.txt
tar xzf reproducibility/literature_screen_api_cache.tar.gz     # -> data/cache/literature_variant_screen/
```

---

## 5. Build order

Set these once per shell. Every command below uses them.

```bash
export PY=$HOME/.conda/envs/silentmethyl/bin/python
export SILENTMETHYL_PY=$PY
export ABL=results/journal/ablation_breast_epithelium    # published breast-epithelium results
export DATA=data/datafiles_breast_epithelium              # published context build
export SPLITS='data/datafiles_breast_epithelium/{split}.csv'
export FUSION_W='checkpoints_ablation/breast_epithelium/seed{seed}/fusion/best_weights.pth'
export SEQ_W='checkpoints_journal/seed{seed}/sequence/best_weights.pth'
mkdir -p logs/{data_build,training,folds,ablation,ablation_analyses,baselines,joint,r6_ablation,r7_ablation,r8_journal,egtex_mt_scoring}
```

### 5.1 Reference context → training data → splits (CPU, ~25 min)

```bash
sbatch --export=ALL,OUT_DIR=data/datafiles_breast_epithelium build_data.sh
```

This runs, with every flag explicit:
`data/build_training_data.py --data-dir data --reference-dir data/reference/BreastEpithelium --out-dir $DATA --m-value-precision float64 --val-chroms chr10 chr11 --test-chroms chr8 chr9`,
then `data/build_testing_data.py --data-dir data --reference-dir data/reference/BreastEpithelium --out-dir $DATA`,
then `data/audit_data_purity.py --data-dir $DATA --output $DATA/data_purity_audit.json`,
then `scripts/17_chromosome_splits.py --datafiles $DATA --out-root $DATA/splits --folds 4`,
and finally writes `$DATA/SHA256SUMS.txt`.

**Verify:** `sha256sum -c` against the `data/datafiles_breast_epithelium/` lines of
`reproducibility/published_outputs_sha256.txt` (see §7). The build is deterministic,
and the §8 validation reproduced every CSV and FASTA byte for byte.

Joint model only, one build per extra tissue (same script, different targets and context):

```bash
$PY -u data/build_training_data.py --data-dir data --reference-dir data/reference/Lung \
    --methylation data/targets/TCGA-LUAD.methylation450.tsv.gz data/targets/TCGA-LUSC.methylation450.tsv.gz \
    --out-dir data/datafiles_multitissue/Lung --m-value-precision float32 --val-chroms chr10 chr11 --test-chroms chr8 chr9
$PY -u data/build_training_data.py --data-dir data --reference-dir data/reference/KidneyCortex \
    --methylation data/targets/TCGA-KIRC.methylation450.tsv.gz \
    --out-dir data/datafiles_multitissue/KidneyCortex --m-value-precision float32 --val-chroms chr10 chr11 --test-chroms chr8 chr9
$PY -u data/build_training_data.py --data-dir data --reference-dir data/reference/ColonTransverse \
    --methylation data/targets/TCGA-COAD.methylation450.tsv.gz data/targets/TCGA-READ.methylation450.tsv.gz \
    --out-dir data/datafiles_multitissue/ColonTransverse --m-value-precision float32 --val-chroms chr10 chr11 --test-chroms chr8 chr9
$PY -u data/compose_multitissue_splits.py --out-dir data/datafiles_joint/all4 --seed 42
$PY -u data/compose_multitissue_splits.py --out-dir data/datafiles_joint/holdout_BreastEpithelium --seed 42 --holdout BreastEpithelium
```

### 5.2 Training (GPU, ~350 GPU-hours total)

**⏱ ~26 h × 3. Sequence towers, published split:**
```bash
for s in 42 43 44; do sbatch --job-name=SM_seq_s$s --export=ALL,SEED=$s,DATA=$DATA run_baseline.sh; done
```

**⏱ ~32 h × 3. Sequence towers, folds 1–3** (fold 0 *is* the published split):
```bash
sbatch --array=1-3 --export=ALL,SPLIT_ROOT=$DATA/splits,STAGES=sequence scripts/run_folds.sbatch
```

**⏱ ~14 h × 6. Context and fusion towers on breast epithelium**, after the matching sequence tower exists:
```bash
for s in 42 43 44; do sbatch --job-name=ctx-s$s --export=ALL,SEED=$s scripts/run_context_ablation.sbatch; done
for f in 1 2 3;    do sbatch --job-name=ctx-f$f --export=ALL,FOLD=$f scripts/run_context_ablation.sbatch; done
```
Each job trains `epi` then `fusion` into `checkpoints_ablation/breast_epithelium/<tag>/`
and tests both into `$ABL/<tag>/{epi,fusion}`. Test the sequence arm into the same tree:
```bash
for s in 42 43 44; do
  $PY -u scripts/13_test_model.py --model sequence --seed $s --test_path $DATA/test.csv \
      --weights_path checkpoints_journal/seed$s/sequence/best_weights.pth --output_dir $ABL/seed$s/sequence
done
```

**⏱ ~49 h × 2. Joint multi-tissue model** (R8 gate plasticity, transfer failure):
```bash
for cfg in all4 holdout_BreastEpithelium; do
  j=$(sbatch --parsable --export=ALL,CONFIG=$cfg,SEED=42,STAGE=seq --time=47:00:00 scripts/run_joint_multitissue.sbatch)
  sbatch --dependency=afterok:$j --export=ALL,CONFIG=$cfg,SEED=42,STAGE=fuse --time=24:00:00 scripts/run_joint_multitissue.sbatch
done
```

**Sequence baselines** (context-free):
```bash
j=$(sbatch --parsable --array=0-1 --export=ALL,DATA=$DATA scripts/run_baseline_grid.sh)
sbatch --dependency=afterok:$j --export=ALL,DATA=$DATA scripts/run_baseline_seeds.sh
$PY -u scripts/14_baselines_simple.py --task both --split-template "$SPLITS" \
    --input-csv data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv \
    --output-dir results/journal/sequence_baselines
```

**Verify training without retraining:** every `best_weights.pth` SHA-256 is in
`reproducibility/published_outputs_sha256.txt`. GPU kernels are nondeterministic,
so a retrained checkpoint will not match bitwise. Compare its held-out metrics
instead (§7.2): they agree to about the third decimal.

### 5.3 Variant scoring (GPU)

```bash
# the unchanged sequence arm: verify the tower, then link its predictions into the tree (seconds, CPU)
STAGE=link bash scripts/run_ablation_analyses.sbatch
# fusion, GENOA + eGTEx x 3 seeds (~1.1 h each)
S=$(sbatch --parsable --array=0-5 --export=ALL,STAGE=score,MODELS=fusion scripts/run_ablation_analyses.sbatch)
# sequence rescored with the current scorer so both arms share one column schema
sbatch --array=0-5 --export=ALL,STAGE=score,MODELS=sequence scripts/run_ablation_analyses.sbatch
# context permutation, fusion seed 42 (~4 h)
sbatch --time=12:00:00 --export=ALL,STAGE=ctxperm scripts/run_ablation_analyses.sbatch
# eGTEx Lung transfer, fusion x 3 seeds (~1.2 h each)
sbatch --array=0-2 --export=ALL,INPUT_CSV=data/external/egtex_multitissue/scoring/Lung/egtex_scoring_input_heldout.csv,OUT_DIR=$ABL/egtex_multitissue_scoring/Lung,WEIGHTS_TEMPLATE='checkpoints_ablation/breast_epithelium/seed{seed}/{model}/best_weights.pth',SPLIT_TEMPLATE="$SPLITS" \
    --output=logs/egtex_mt_scoring/score_%a_%A.out --error=logs/egtex_mt_scoring/score_%a_%A.err scripts/run_egtex_multitissue_scoring.sh
```
The `ctxperm` stage runs `scripts/23_context_permutation.py --input-csv data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv --stratum heldout --seed 42 --weights-template "$FUSION_W" --split-template "$SPLITS" --device cuda --batch-size 32 --output-dir $ABL/context_permutation`.
The published run used the schemes that existed then (`identity shuffle median`). To
reproduce it exactly with the current script, add `--schemes identity shuffle median`.

### 5.4 Candidates, positive controls, applications (GPU, minutes)

```bash
j60=$(sbatch --parsable jobs/r6_ablation/60_candidates.sbatch)
sbatch --dependency=afterok:$j60 jobs/r6_ablation/62_comparison.sbatch
j70=$(sbatch --parsable jobs/r7_ablation/70_mqtl_positive.sbatch)
sbatch --dependency=afterok:$j70 jobs/r7_ablation/71_mqtl_negative.sbatch
j63=$(sbatch --parsable jobs/r7_ablation/63_known_variant.sbatch)
sbatch --dependency=afterok:$j63 jobs/r7_ablation/64_literature.sbatch     # needs the literature cache (§4.4); runs --offline-equivalent from cache
```
The job files hold the full flag sets. Every job stamps the clock and fails
afterwards if it wrote anything outside `$ABL`. As published, `60_candidates.sbatch`
scores **seed 42 only** (see §10).

### 5.5 R8: gate mechanism, dose-response ladder, ASM validation (GPU)

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

sbatch jobs/r8_journal/53_asm_score.sbatch          # Rosenski: 53b scoring + 53c evaluation
sbatch jobs/r8_journal/53d_tycko_score.sbatch       # Do & Tycko: 53b scoring + 53e evaluation
sbatch jobs/r8_journal/23_ctx_ladder.sbatch         # dose-response ladder (~4.9 h)
sbatch --array=0-1 jobs/r8_journal/54_gate_instrument.sbatch
```
`53b_asm_score.py` reads `checkpoints_ablation/breast_epithelium/seed42/fusion` and
`checkpoints_journal/seed42/sequence` by default. Both are recorded in
`results/journal/asm_validation*/score_summary.json`.

### 5.6 CPU analyses, figures and supplements (~1 h)

This block is written for a fresh clone, where it writes to the published
locations: `OUT=$ABL`, `OUTJ=results/journal` for the few outputs that live outside
the ablation tree (`joint/`, `asm_validation*/`, `manuscript_figures_r8/`), and
`OUTR=results` for the supplement packages. To check a rerun against the shipped
outputs without overwriting them, set `OUT=repro_check/abl OUTJ=repro_check/journal
OUTR=repro_check/results`. Every step reads GPU products from their published
locations and writes only under the three output roots. This exact block was
executed that way on 16 Sep 2026 (§8).

<!-- cpu-validate:begin -->
```bash
OUT=${OUT:-$ABL}; OUTJ=${OUTJ:-results/journal}; OUTR=${OUTR:-results}
mkdir -p "$OUT" "$OUTJ" "$OUTR"

# R1: paired model comparison and context stratification
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

# R2: variant effects, GENOA and eGTEx
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

# R3: transfer to eGTEx Lung (the only non-breast tissue rescored on breast epithelium, see §10)
$PY -u scripts/31_transfer_discrimination.py --reference "$ABL/egtex_multitissue_scoring/Lung::fusion::42,43,44" \
    --compare "results/journal/egtex_multitissue_scoring/by_tissue/Lung::sequence::42,43,44" \
    --stratum heldout --n-boot 500 --match-tolerance 10 --match-ratio 1 --random-seed 42 \
    --output-dir "$OUT/transfer_discrimination/Lung"

# R4: shared vs tissue-specific meQTLs. Always run the two stages separately (see §6)
$PY -u scripts/40_meqtl_tissue_specificity.py --stage matched --cohort "GENOA:$ABL/genoa_variant_scoring:5e-8" \
    --cohort "eGTEx:$ABL/egtex_variant_scoring:1.483e-5" --split-template "$SPLITS" --model fusion --seeds 42 43 44 \
    --n-boot 1000 --random-seed 42 --output-dir "$OUT/tissue_shared_meqtls"
$PY -u scripts/40_meqtl_tissue_specificity.py --stage chromatin --cohort "GENOA:$ABL/genoa_variant_scoring:5e-8" \
    --cohort "eGTEx:$ABL/egtex_variant_scoring:1.483e-5" --split-template "$SPLITS" --model fusion --seeds 42 43 44 \
    --n-boot 1000 --random-seed 42 --output-dir "$OUT/meqtl_class_chromatin"
$PY -u scripts/41_tissue_specificity_summary.py --results "$OUT/tissue_shared_meqtls" --out "$OUT/tissue_shared_meqtls" \
    --label "SilentMethyl (breast-epithelium context)"

# R5: uncertainty, GWAS, motifs
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

# R8: fusion-gain stratification, ladder stratification, gate decomposition, joint model, ASM
$PY -u scripts/57_fusion_gain_stratified.py --test-path $DATA/test.csv \
    --prediction-template "$ABL/seed{seed}/{model}/predictions.csv" \
    --cpg-island-annotation data/HM450.hg38.manifest.CpGIsland.tsv.gz --seeds 42 43 44 \
    --block-size-bp 1000000 --bootstrap-replicates 2000 --random-seed 20260915 --output-dir "$OUT/fusion_gain_stratified"
for strat in observed predicted; do
  $PY -u scripts/58_ladder_effect_size_stratification.py --ladder-dir "$ABL/context_ladder" \
      --stratifier $strat --output-dir "$OUT/context_ladder_stratified"
done
if [ "$OUT" != "$ABL" ]; then   # analyse reads the GPU-instrumented pairs from its own output dir
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

# Figures. 91 DELETES every file in --output-dir that is not one of its own figures:
# never point --output-dir at a directory that holds anything else.
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

# Supplements. Both builds run the MCF-10A guard (§6) and fail if it fires.
$PY -u scripts/93_build_r8_supplement.py --output-dir "$OUTR/supplementary_package_r8"
$PY -u scripts/90_build_supplement_package.py --project-root . --output-dir "$OUTR/supplementary_package" --allow-missing-required
```
<!-- cpu-validate:end -->

Scripts 92 and 93 read their other inputs from `results/journal/` defaults, which
are the published breast-epithelium locations. 93 takes no input flags; its
manifest is fixed in the script and every source is under `$ABL`,
`results/journal/{joint,asm_validation,asm_validation_tycko}` or `data/external/`.

`90_build_supplement_package.py` (S1–S6) needs `--allow-missing-required` until
the gaps in §10 are closed. Without it, the build stops and lists what is missing.

### 5.7 Melody comparison (context-free on the Melody side)

```bash
sbatch --export=ALL,MELODY_PY=/path/to/melody/env/bin/python scripts/run_melody_union.sbatch
sbatch --export=ALL,MELODY_PY=/path/to/melody/env/bin/python scripts/run_melody_st.sbatch
```

### 5.8 Manuscript

```bash
./scripts/94_collect_manuscript_figures.sh     # copies the nine main-text figures to the repo root
pdflatex main_revised.tex && bibtex main_revised && pdflatex main_revised.tex && pdflatex main_revised.tex
```
`workflow.pdf` (Fig. 1) is hand-drawn and has no generating script.

---

## 6. Published-path traps

Every one of these has fired or nearly fired. Most are now refused in code; the rest are listed so you can pass the flag.

| trap | consequence | status |
|---|---|---|
| `17_chromosome_splits.py` without **both** `--datafiles` and `--out-root` | `--out-root` used to default to `data/datafiles/splits` independently, so the published fold splits were overwritten | **both required**; refuses an `--out-root` that already holds folds unless `--overwrite` |
| `build_training_data.py` / `build_testing_data.py` defaults | `--reference-dir` defaulted to the MCF-10A tracks and `--out-dir` to `data/datafiles/` | **both required**; refuses an existing build unless `--overwrite` |
| `20_variant_scoring.py` / `23_context_permutation.py` `--split-template` default `data/datafiles/{split}.csv` | new fusion weights scored against old context: a hybrid that looks normal | 20 **refuses** a fusion checkpoint whose context differs from the split template's; always pass `--split-template "$SPLITS"` |
| `--weights-template` default `checkpoints_journal/…` in 20, 23, 60, 63, 64, 70, 71 | scores the superseded MCF-10A fusion | always pass `$FUSION_W` (the job files do) |
| `40_meqtl_tissue_specificity.py --stage all`, `51_rc_uncertainty.py --stage all` | ignored `--output-dir` and wrote to published paths (overwrote four results on 12 Sep) | **refused** when combined with `--output-dir`; run stages separately |
| `91_build_manuscript_figures.py --output-dir` | deletes every non-whitelisted file in that directory | send side outputs (`--stk11-case-output` etc.) elsewhere |
| `64_literature_variant_screen.py` | shells out to 63; older versions did not forward `--weights-template` | fixed; pass `--weights-template "$FUSION_W"` (job file does) |
| `60_candidate_background.py --seeds` default `[42]` | single-seed candidates | pass `--seeds 42 43 44` (see §10) |
| `70`/`71` `--test-csv` default `data/datafiles/test.csv` | tabular features merged from the MCF-10A build | pass `--test-csv $DATA/test.csv` (job files do) |
| defaults of 01, 14, 15, 16, 18, 21, 22, 30, 31, 40, 50, 51, 52, 60–64, 70, 71, 91 | read or write pre-swap `results/journal/<analysis>/` | every command in §5 passes explicit paths |
| `run_egtex_multitissue_scoring.sh` | `OUT_DIR` defaults to the pre-swap directory | refuses `WEIGHTS_TEMPLATE` without `SPLIT_TEMPLATE`, and alternative weights into the default `OUT_DIR` |
| `12_train_fusion.py --sequence_weights/--epi_weights` | take the `.pth` **file**, not the directory | job files check first |
| `run_baseline.sh`, `run_folds.sbatch`, `run_context_ablation.sbatch` | retraining over published towers | **refuse** if `best_weights.pth` exists |
| `build_data.sh` | wrote over `data/datafiles/` and the published audit files | `OUT_DIR` required; writes nothing outside it |
| Slurm `--export=VAR=x` without `ALL`; missing `logs/<dir>` | interpreter not found; job dies in 5 s with no log | always `--export=ALL,…`; `mkdir -p logs/...` (§5 preamble) |

**MCF-10A guard.** `scripts/93_build_r8_supplement.py` (and, through it, `90`)
refuses to produce a package if any packaged file contains an MCF-10A marker
(`MCF 10A` in any spelling, `Mint-ChIP`, or any accession of the seven MCF-10A
tracks), or if any packaged source lies outside a breast-epithelium or
context-free path. A number carries no marker, so the second check is the one
that catches a pre-swap table. Check an existing package:

```bash
$PY -u scripts/93_build_r8_supplement.py --check-package results/supplementary_package_r8
```

---

## 7. Verifying against the shipped outputs

### 7.1 Checksums

`reproducibility/published_outputs_sha256.txt` holds the SHA-256 of every file
under `results/journal/ablation_breast_epithelium/`, `asm_validation*/`, `joint/`,
`manuscript_figures_r8/`, `results/supplementary_package_r8/`, all 36
`best_weights.pth`, the published context builds and the scoring inputs, as of
16 Sep 2026.

```bash
grep ' data/datafiles_breast_epithelium/' reproducibility/published_outputs_sha256.txt | sha256sum -c --quiet
```

CPU stages are deterministic. Compare a rerun into `repro_check/abl` file by
file (strip the prefix):

```bash
sed 's#  results/journal/ablation_breast_epithelium/#  repro_check/abl/#' reproducibility/published_outputs_sha256.txt \
  | grep '  repro_check/abl/' | sha256sum -c 2>/dev/null | grep -v ': OK$'
```

`run_summary.json` files embed a timestamp and their own output path, so expect
those to differ. §8 lists every other expected difference.

### 7.2 Metrics for the GPU stages

GPU training and scoring reproduce to about three decimals, not bitwise.
Published held-out test metrics (chr8+chr9, 26,570 CpGs; `$ABL/<tag>/<arm>/metrics.json`):

| run | context-only β MAE / AUC | sequence-only β MAE / AUC | fusion β MAE / AUC |
|---|---|---|---|
| seed 42 | 0.1022 / 0.9658 | 0.1090 / 0.9576 | 0.0885 / 0.9765 |
| seed 43 | 0.1022 / 0.9658 | 0.1103 / 0.9568 | 0.0901 / 0.9751 |
| seed 44 | 0.1023 / 0.9658 | 0.1104 / 0.9562 | 0.0956 / 0.9717 |
| fold 1 | 0.0980 / 0.9707 | — | 0.0873 / 0.9773 |
| fold 2 | 0.0985 / 0.9679 | — | 0.0934 / 0.9724 |
| fold 3 | 0.0987 / 0.9703 | — | 0.0862 / 0.9779 |

Distance-matched AUROC (`$ABL/{genoa,egtex}_variant_evaluation/matched_negative_auroc.csv`, seeds 42/43/44):

| cohort | fusion | sequence |
|---|---|---|
| GENOA (n = 8,074) | 0.5610 / 0.5569 / 0.5706 | 0.5671 / 0.5609 / 0.5572 |
| eGTEx breast (n = 836) | 0.5823 / 0.5952 / 0.5873 | 0.6045 / 0.5529 / 0.6126 |

Candidate chain: 440 model-visible candidates, NCOA2 (chr8:70148394 G>A,
cg20699548) rank 1 at Δβ = −0.1539 (seed 42). Positive control: 81 held-out
eGTEx associations. Matched negative: 35 pairs. TCGA: 97 type-11 normal samples.

---

## 8. What was executed to validate this document

On 16 Sep 2026, on Bridges-2, from this commit's code:

| step | how | result |
|---|---|---|
| §2 environment | `$PY --version`; imports | Python 3.10.20, torch 2.6.0+cu124, transformers 5.8.0, pandas 2.3.3 |
| §4.2 context tracks | resolve-only run of the command as written, `--plan` to scratch; md5 of local files | live portal returns the same 7 accessions and md5s; 7/7 local files match |
| §4.3 external check | `acquire_external_cohorts.py --check` | **GENOA endpoint timed out** (TimeoutError); run stopped after 300 s |
| §5.1 context build | `sbatch --export=ALL,OUT_DIR=repro_check/datafiles_breast_epithelium,VERIFY_AGAINST=data/datafiles_breast_epithelium build_data.sh` (job `46189068`, with `--m-value-precision float64`) | **pass**: COMPLETED 0:0; purity audit PASS (0 errors, 0 warnings); 4 chromosome-blocked folds written; output **byte-identical** to `data/datafiles_breast_epithelium` |
| §5.6 CPU block | the block above, extracted verbatim and run with `OUT=repro_check/abl` (job `46187803`) | **pass**: COMPLETED 0:0; all 32 script invocations (16, 22, 01, 21, 30, 31, 40, 41, 51, 52, 50, 57, 58, 54, 55, 56, 53c, 53e, 91, 92, 93, 90) ran with no traceback; MCF-10A guard clean (23 and 55 sources). `repro_check/abl/target_qc` is byte-identical to the installed `$ABL/target_qc` (6/6 files). Two caveats below |
| — S1–S6 package (90) | same job | **status INCOMPLETE**: 6 required files missing. S1 ×3 (`candidates/stability/*`, the seed-42-only gap in §10) and S6 ×3 (`reproducibility/{data_purity_audit_breast_epithelium.json,analysis_code_sha256.txt,processed_data_sha256_breast_epithelium.txt}`). **S6 closed 17 Sep 2026**: the three files were generated (purity audit PASS, byte-identical to the rebuild's; data hashes match the rebuild's `SHA256SUMS.txt`), and a rerun of 90 leaves only the three S1 files missing (60 files included). R8 package S7–S12: 23 files, complete |
| — published-tree writes | the job's clobber check | 91 rewrote `$ABL/candidates/top_candidate_case_study.csv`; content is the seed-42 NCOA2 label (−0.1539). The file is git-ignored, so the previous bytes cannot be compared. **Cause and fix:** `--case-study-path` is an *output* despite its name — inside `plot_candidate` the parameter is `case_output` — and the block pointed it at `$ABL` instead of `$OUT`. The block above now passes `"$OUT/candidates/top_candidate_case_study.csv"`, so a check run no longer writes into the published tree. The flag is now renamed `--case-study-output`, matching `--stk11-case-output` / `--genomic-region-output` |
| script 91 regression | 91 on the **pre-swap** inputs with the new label code | all 7 outputs byte-identical to the published pre-swap build, including the NCOA2 label |
| guard | injected `MCF-10A` into a packaged CSV; marker in a source; pre-swap source with no marker | all three refused (exit 3, package deleted); clean package passes |
| trap guards | 17 without `--out-root`; builders without paths; 17/builders into existing builds; 40/51 `--stage all --output-dir` | all refused |

Not executed (documented only): downloads (§4.3 beyond the check), every GPU step
in §5.2–5.5 and §5.7, the joint-model data builds, LaTeX.

---

## 9. The superseded MCF-10A record

Kept for the record, read by nothing published:

- tracks: `data/reference/{ATAC_seq,H3K*}.bw` (see `data/reference/TRACK_SET.md`)
- build: `data/datafiles/` (still needed for one file: the frozen GDC cache path, §4.4)
- weights: `checkpoints_journal/seed*/{epi,fusion}`, `checkpoints_folds/fold*/{epi,fusion}_seed42`
- results: every analysis directory directly under `results/journal/` except
  `ablation_breast_epithelium/`, `asm_validation*/`, `joint/`, `manuscript_figures_r8/`
  and the context-free baselines (`sequence_baselines`, `published_baselines*`,
  `melody*`, `baseline_variant_evaluation`, `motif_disruption_kmer_baseline`,
  `training_data_audit`, and the sequence arms of `seed*/`, `folds/`,
  `*_variant_scoring/`, `egtex_multitissue_scoring/`)
- the job files that produced them are in git history before commit `ff5050e`'s successor

`reproducibility/MCF10A_AUDIT.md` is the full audit of where each MCF-10A product sits.

---

## 10. Known gaps

These are not reproducible from this repository today, or the published
breast-epithelium value does not exist yet. Each is either a queued GPU job or a
manuscript decision.

| gap | what is missing | to close it |
|---|---|---|
| **candidate chain is seed 42 only** | S1 cross-seed statistics, `candidates/stability/`, NCOA2 mean ± SD | GPU: `sbatch --export=ALL jobs/r6_ablation/60_candidates.sbatch` with `--seeds 42 43 44` added, then `61_candidate_stability.py --seeds 42 43 44 --scores-template "$ABL/candidates/seed{seed}/candidate_scores.csv" --output-dir $ABL/candidates/stability`, then 62, 22, 91 |
| **eight eGTEx tissues other than Lung** | R3 transfer AUROCs and Melody head-to-heads on breast-epithelium fusion | GPU: `run_egtex_multitissue_scoring.sh` over `union_scoring_input_heldout.csv` with the §5.3 overrides, then `data/split_predictions_by_tissue.py`, `31`, `32`, `34`, `35`, `40` |
| frozen intermediates without code | bimodal BED, Ensembl rsID cache, multi-tissue union input | shipped in `reproducibility/frozen_inputs.tar.gz` with SHA-256 |
| GENOA download endpoint | timed out during validation | URL and SHA-256 in `data/external/external_manifest.json` |
| baseline eGTEx scoring layout | `published_baselines_egtex/variant_scoring/heldout/` was scored with `15 --score-pairs --stratum heldout_egtex --pairs data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv --output-dir results/journal/published_baselines` and then moved; the move is inferred from `scoring_summary.json`, not recorded | rerun and move as described |
| `workflow.pdf` | hand-drawn | supply |
