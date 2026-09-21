# SilentMethyl

SilentMethyl predicts CpG methylation from a 1,000-bp DNA sequence and reference
epigenomic data. It combines a DNABERT-2 sequence model with a context MLP through
gated fusion, then estimates variant effects as MUT − WT.

The published model uses seven chromatin tracks from **primary breast
epithelium** and phyloP conservation scores. Code supports the manuscript in
`main.tex` and `supplementary.tex`.

## Getting started

- [REPRODUCE.md](REPRODUCE.md): setup, data downloads, training, evaluation and file checks.
- `data/reference/*/TRACK_SET.md`: sources, assays and checksums for the context tracks, written by `data/acquire_multitissue_inputs.py` when the tracks are downloaded.
- `figures/source_data/`: data for individual figure panels; see `source_data_index.csv`.

The run commands use Slurm on Bridges-2. The cluster working directory includes
model checkpoints, downloaded inputs and full prediction tables that are absent
from the local checkout. To work from a fresh checkout, download, copy or rebuild
these files using REPRODUCE.md.

## Results

The test set contains 26,570 CpGs on held-out chromosomes 8 and 9. Values with
± show the mean and standard deviation across seeds 42–44.

| Model | β MAE | ROC-AUC |
|---|---|---|
| composition | 0.1954 | 0.8748 |
| *k*-mer ridge | 0.1565 | 0.9190 |
| CpGenie (retrained) | 0.1281 ± 0.0013 | 0.9406 ± 0.0006 |
| DeepCpG (retrained) | 0.1253 ± 0.0005 | 0.9437 ± 0.0011 |
| sequence-only | 0.1099 ± 0.0008 | 0.9569 ± 0.0007 |
| context-only | 0.1022 ± 0.0000 | 0.9658 ± 0.0000 |
| **gated fusion** | **0.0914 ± 0.0037** | **0.9744 ± 0.0025** |

Fusion gave the best baseline methylation predictions. Context-only also
outperformed the retrained CpGenie and DeepCpG models across seeds.

**Variant prediction showed a smaller benefit.** Fusion showed no significant
improvement over sequence alone in paired GENOA and eGTEx comparisons. Context
can change variant predictions through the fusion gates even when REF and ALT
use the same context. These changes did not clearly improve external accuracy.

**External agreement was modest.** In GENOA, distance alone reached AUROC 0.595;
distance matching reduced fusion AUROC from 0.600 to 0.556. Signed Spearman
correlation was 0.236 (0.098–0.392) in eGTEx breast and 0.149 (0.112–0.185) in
GENOA whole blood. Direction agreement was 59.6% and 55.6%.

**ASM provided a complementary test.** Fusion AUROC was 0.5625 in Rosenski and
0.5419 in Do–Tycko; sequence-only AUROC was 0.5736 and 0.5432. Distance alone
performed near chance. On 722 Do–Tycko SNPs, direction agreement was 0.590
(0.550–0.630), below the expected range of 0.60–0.70 recorded before analysis.

**Transfer error was higher at tissue-variable CpGs.** Error increased with
cross-tissue methylation variance, with a 4.4-fold MAE difference between the
lowest and highest variance deciles. The largest errors were enriched at CpG
shores and loci with higher H3K4me1/H3K27me3 signals.

Detailed results are in `results/journal/ablation_breast_epithelium/` and
`figures/source_data/`.

## Models and checkpoints

The sequence model uses DNA only. Context-only and fusion models use the
reference features. Sequence weights were reused after the context source
changed; context and fusion models were retrained on breast-epithelium tracks.
REPRODUCE.md Section 5.3 checks that the sequence weights match.

| Model | Main checkpoint path | Additional chromosome splits |
|---|---|---|
| sequence-only | `checkpoints_journal/seed*/sequence` | `checkpoints_folds/fold*/sequence_seed42` |
| context-only | `checkpoints_journal/seed*/epi` | `checkpoints_folds/fold*/epi_seed42` |
| fusion | `checkpoints_journal/seed*/fusion` | `checkpoints_folds/fold*/fusion_seed42` |

These paths exist on the cluster. Joint-model weights are in
`checkpoints_joint/{all4,holdout_BreastEpithelium}/seed42/`. CpGenie and DeepCpG
weights are under `results/journal/published_baselines/`.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Training, scoring, evaluation and figure scripts, plus job launchers |
| `jobs/single_tissue/` | Slurm jobs for candidates, mQTL controls and variant applications |
| `jobs/mechanism/` | Slurm jobs for ASM, gate components and context changes |
| `data/` | Download and preprocessing scripts; the cluster also holds inputs and built splits |
| `checkpoints_*/` | Trained model weights and run settings on the cluster |
| `dnabert2_local/` | Downloaded DNABERT-2 model and tokenizer on the cluster |
| `figures/` | Main and supplementary figures, with panel data in `source_data/` |
| `reproducibility/` | Environment details, input checks, archived inputs and file checksums |
| `results/` | Metrics and summaries; the cluster also holds full predictions and analysis tables |
| `repro_check/` | Outputs from separate reproduction checks on the cluster |
| `archive/` | Older analysis summaries on the cluster |
| `logs/` | Job output and error logs |
| `build_data.sh` | Builds training data and chromosome splits |
| `run_baseline.sh` | Trains the sequence model |

Use `results/journal/ablation_breast_epithelium/` for current breast-epithelium
results. Other result folders include older runs; check their run settings before
combining outputs.

