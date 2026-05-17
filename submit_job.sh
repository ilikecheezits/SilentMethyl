#!/bin/bash
#SBATCH --job-name=SilentMethyl_Train
#SBATCH --output=logs/training_output_%j.log
#SBATCH --error=logs/training_error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "========================================"
echo "Starting Job: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "========================================"

# --- 1. ENVIRONMENT SETUP ---
# Load Python module if required by your cluster
# module load python/3.10 

# Activate your virtual environment
# If you uploaded your .venv, use this. If you use Conda, use 'conda activate'
source .venv/bin/activate

# Ensure directories exist
mkdir -p data/processed checkpoints logs

# (Optional) Prevent HuggingFace download warnings/limits
# export HF_TOKEN=your_token_here

# --- 2. BUILD THE TRAINING DATA ---
echo "[*] Step 1: Executing 01_build_data.py..."

# FIXED: Pointing to the REAL matrix and dictionary
python scripts/01_build_data.py \
    --matrix_path data/raw/Final_Discovery_Dataset_MultiOmnics.csv \
    --dict_path data/raw/DNA_Sequence_Dictionary.csv \
    --base_dir data/processed

if [ $? -ne 0 ]; then
    echo "[!] ERROR: Data build failed. Aborting pipeline."
    exit 1
fi

# --- 3. TRAIN THE MODEL ---
echo "[*] Step 2: Executing 02_train_model.py..."

# FIXED: Updated vocab paths to match the actual output of Step 1
python scripts/02_train_model.py \
    --matrix_path data/processed/cleaned_training_data.csv \
    --dict_path data/raw/DNA_Sequence_Dictionary.csv \
    --region_vocab_path data/processed/SilentMethyl_RegionVocab.pkl \
    --island_vocab_path data/processed/SilentMethyl_IslandVocab.pkl \
    --save_dir ./checkpoints \
    --batch_size 32 \
    --num_workers 8 \
    --epochs 20 \
    --lr 1e-4

if [ $? -ne 0 ]; then
    echo "[!] ERROR: Model training failed."
    exit 1
fi

echo "========================================"
echo "Pipeline finished at $(date)"
echo "========================================"