#!/bin/bash
# Download the multi-tissue targets and context tracks on a CPU node.
#
# Why a batch job rather than a login node
# ---------------------------------------
# This pulls tens of GB. Login nodes are shared and enforce per-process limits,
# so a long transfer can be killed part-way -- and PSC asks that sustained work
# not run there. The download is I/O bound and needs no GPU, so RM-shared with a
# couple of cores is the right home for it.
#
# BUT: compute nodes do not always have outbound HTTPS. Check before submitting,
# from a login node:
#
#   srun --partition=RM-shared --ntasks=1 --time=00:05:00 \
#        python -u data/acquire_multitissue_inputs.py --network-check
#
# If that reports both hosts reachable, submit this. If it does not, run the
# download on a login node instead, under nohup so a disconnect does not kill
# it, and rely on the resume behaviour if it is interrupted:
#
#   nohup python -u data/acquire_multitissue_inputs.py --apply \
#         > logs/data_build/acquire_multitissue.log 2>&1 &
#
# Either way the transfer is resumable: an interrupted file leaves a .partial
# and the next run continues from that offset, so re-running is always safe.
#
#   sbatch data/acquire_multitissue.sbatch
#SBATCH --job-name=acquire_mt
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/data_build/acquire_mt_%j.out
#SBATCH --error=logs/data_build/acquire_mt_%j.err
set -eo pipefail

ROOT=/ocean/projects/med250012p/szhang37/SilentMethyl
cd "$ROOT"
mkdir -p logs/data_build

PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python

echo "=== host $(hostname)  $(date -u +%FT%TZ) ==="

# Fail immediately and legibly if this node cannot reach the sources, rather
# than eight hours of connection timeouts.
if ! $PY -u data/acquire_multitissue_inputs.py --network-check; then
  echo "STOP: no outbound HTTPS from this compute node." >&2
  echo "Run the download on a login node under nohup instead -- see the" >&2
  echo "header of this file." >&2
  exit 4
fi

# Resolve first and keep the plan, so what was fetched is recorded even if the
# transfer is interrupted later.
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
