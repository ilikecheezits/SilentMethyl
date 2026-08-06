#!/bin/bash
#SBATCH --job-name=SM_experiments
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/experiments/experiments_%j.out
#SBATCH --error=logs/experiments/experiments_%j.err

set -euo pipefail
umask 027

ROOT=/ocean/projects/med250012p/szhang37/SilentMethyl
cd "$ROOT"

if (( $# == 0 )); then
  echo "Usage: sbatch run_experiments.sh SEED [SEED ...]" >&2
  echo "Example: sbatch run_experiments.sh 42 43 44" >&2
  exit 2
fi

SEEDS=("$@")
declare -A SEEN=()
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "[!] Seed must be a non-negative integer: $seed" >&2
    exit 2
  fi
  if [[ -n "${SEEN[$seed]:-}" ]]; then
    echo "[!] Duplicate seed: $seed" >&2
    exit 2
  fi
  SEEN[$seed]=1
done

module load anaconda3
module load cuda/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

mkdir -p logs/experiments results/journal

for required in \
  scripts/03_target_qc.py \
  scripts/04_mqtl_positive_control.py \
  scripts/05_matched_background.py \
  scripts/06_stability.py \
  data/datafiles/test.csv \
  data/datafiles/testing_data_test_only.csv \
  data/HM450.hg38.manifest.tsv.gz \
  data/TCGA-BRCA.methylation450.tsv.gz \
  data/egtex_breast_mqtl_heldout_qc.csv \
  results/journal/seed42/epi/predictions.csv \
  results/journal/seed42/sequence/predictions.csv \
  results/journal/seed42/fusion/predictions.csv; do
  if [[ ! -s "$required" ]]; then
    echo "[!] Missing required file: $required" >&2
    exit 2
  fi
done

for seed in "${SEEDS[@]}"; do
  for checkpoint in \
    "checkpoints_journal/seed${seed}/sequence/best_weights.pth" \
    "checkpoints_journal/seed${seed}/fusion/best_weights.pth"; do
    if [[ ! -s "$checkpoint" ]]; then
      echo "[!] Missing checkpoint for seed ${seed}: $checkpoint" >&2
      exit 2
    fi
  done
done

export PYTHONHASHSEED=20260810
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[*] Host: $(hostname)"
echo "[*] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[*] Git commit: $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "[*] Experiment seeds: ${SEEDS[*]}"
nvidia-smi || true

echo "[*] Stage 3: target-quality audit"
python -u scripts/03_target_qc.py

echo "[*] Stage 4: frozen eGTEx breast mQTL positive control"
python -u scripts/04_mqtl_positive_control.py \
  --seeds "${SEEDS[@]}"

echo "[*] Stage 5: somatic candidate scoring and matched background"
python -u scripts/05_matched_background.py \
  --seeds "${SEEDS[@]}"

echo "[*] Stage 6: forward/RC and cross-seed candidate stability"
python -u scripts/06_stability.py \
  --seeds "${SEEDS[@]}"

python -m json.tool \
  results/journal/target_qc/hm450_manifest_audit.json >/dev/null
python -m json.tool \
  results/journal/egtex_mqtl_positive_control/run_summary.json >/dev/null
python -m json.tool \
  results/journal/candidates/candidate_analysis_summary.json >/dev/null
python -m json.tool \
  results/journal/candidates/stability/stability_summary.json >/dev/null

echo "[✓] All experiments complete for seeds: ${SEEDS[*]}"