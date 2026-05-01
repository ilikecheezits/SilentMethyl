#!/bin/bash
# ==============================================================================
# Slurm Job Configuration
# ==============================================================================
#SBATCH --job-name=SilentMethyl_Train
#SBATCH --output=logs/training_output_%j.log   # Standard output/error log (%j is jobID)
#SBATCH --error=logs/training_error_%j.log     # Separate file for errors
#SBATCH --partition=gpu                        # Target the GPU nodes (adjust for your cluster)
#SBATCH --gres=gpu:1                           # Request 1 GPU (e.g., A100 or V100)
#SBATCH --cpus-per-task=8                      # Request 8 CPU threads (Matches num_workers)
#SBATCH --mem=64G                              # Request 64GB of RAM (To handle dataset generation)
#SBATCH --time=12:00:00                        # Max time limit (12 hours)

# ==============================================================================
# 1. Environment Setup
# ==============================================================================
echo "========================================"
echo "Starting Job: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "========================================"

# Load your Anaconda/Miniconda module (uncomment and edit based on your cluster's setup)
# module load anaconda/2023a
# source activate silentmethyl_env

# Ensure the output directories exist
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs

# ==============================================================================
# 2. Build the Training Data
# ==============================================================================
echo "[*] Step 1: Executing 01_build_data.py..."

# Note: Update data/raw/... paths to exactly where your raw data sits on the cluster
python scripts/01_build_data.py \
    --matrix_path data/raw/Patient_Omics_Matrix.csv \
    --dict_path data/raw/Sequence_Dictionary.csv \
    --base_dir data/processed

if [ $? -ne 0 ]; then
    echo "[!] ERROR: Data build failed. Aborting pipeline."
    exit 1
fi
echo "[✓] Step 1 Complete."

# ==============================================================================
# 3. Train the Model & Secure Weights
# ==============================================================================
echo "[*] Step 2: Executing 02_train_model.py..."

python scripts/02_train_model.py \
    --matrix_path data/processed/cleaned_training_data.csv \
    --dict_path data/raw/Sequence_Dictionary.csv \
    --region_vocab_path data/processed/region_vocab.pkl \
    --island_vocab_path data/processed/island_vocab.pkl \
    --save_dir ./checkpoints \
    --batch_size 32 \
    --num_workers 8 \
    --epochs 20 \
    --lr 1e-4

if [ $? -ne 0 ]; then
    echo "[!] ERROR: Model training failed."
    exit 1
fi

echo "[✓] Step 2 Complete. Model weights successfully saved to ./checkpoints/"
echo "Pipeline finished at $(date)"