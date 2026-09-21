#!/bin/bash
#SBATCH --job-name=acquire_mt
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/data_build/acquire_mt_%j.out
#SBATCH --error=logs/data_build/acquire_mt_%j.err
# Slurm job: download the multi-tissue targets and context tracks on a CPU node.

set -eo pipefail

ROOT="${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
mkdir -p logs/data_build

PY="${SILENTMETHYL_PY:-python}"

echo "=== host $(hostname)  $(date -u +%FT%TZ) ==="

if ! $PY -u data/acquire_multitissue_inputs.py --network-check; then
  echo "STOP: no outbound HTTPS from this compute node." >&2
  echo "Run the download on a login node under nohup instead -- see the" >&2
  echo "header of this file." >&2
  exit 4
fi

echo "--- resolving ---"
$PY -u data/acquire_multitissue_inputs.py

echo "--- downloading ---"
$PY -u data/acquire_multitissue_inputs.py --apply

echo "--- disk ---"
du -sh data/reference/* data/targets 2>/dev/null || true

echo "=== done $(date -u +%FT%TZ) ==="
echo "Next: record provenance for each new tissue, e.g."
echo "  python -u data/audit_reference_tracks.py --write-manifest \\"
echo "      --reference-dir data/reference/Lung"
