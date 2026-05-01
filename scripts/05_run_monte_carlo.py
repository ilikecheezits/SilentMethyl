import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
import argparse
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.architecture import SilentMethylModel
from model.lora_utils import inject_lora_adapters, load_lora_weights
from inference.engine import predict_epigenetics

def main():
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulation for model stability analysis.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the inference data CSV.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing scalers and vocabs.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M", help="Hugging Face model path.")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of the sample to analyze.")
    parser.add_argument("--n_runs", type=int, default=200, help="Number of Monte Carlo runs.")
    parser.add_argument("--save_dir", default=".", help="Directory to save the output plot.")
    args = parser.parse_args()

    print("--- MONTE CARLO STABILITY ANALYSIS ---")

    # =========================================
    # Environment Setup
    # =========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================
    # Load Data and Artifacts
    # =========================================
    print("[*] Loading data and artifacts...")
    df_test = pd.read_csv(args.data_path)
    sample = df_test.iloc[args.sample_index]

    scaler_age = joblib.load(os.path.join(args.base_dir, "Scaler_Age.pkl"))
    scaler_seq = joblib.load(os.path.join(args.base_dir, "Scaler_Seq.pkl"))
    REGION_VOCAB = joblib.load(os.path.join(args.base_dir, "SilentMethyl_RegionVocab.pkl"))
    ISLAND_VOCAB = joblib.load(os.path.join(args.base_dir, "SilentMethyl_IslandVocab.pkl"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # =========================================
    # Model Rebuilding and Loading
    # =========================================
    print("[*] Rebuilding model and loading weights...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model = load_lora_weights(model, args.weights_path, device)
    
    # =========================================
    # Monte Carlo Simulation
    # =========================================
    print(f"[*] Running {args.n_runs} Monte Carlo runs for sample {args.sample_index}...")
    
    # Enable dropout
    model.train()

    wt_predictions = []
    mut_predictions = []

    for _ in range(args.n_runs):
        wt_prob = predict_epigenetics(
            sample['Healthy_5000bp_DNA'], sample['Age'], sample['Gene_Region'],
            sample['CpG_Island_Status'], model, tokenizer, scaler_age, scaler_seq,
            REGION_VOCAB, ISLAND_VOCAB, device
        )
        wt_predictions.append(wt_prob)

        mut_prob = predict_epigenetics(
            sample['Mutated_5000bp_DNA'], sample['Age'], sample['Gene_Region'],
            sample['CpG_Island_Status'], model, tokenizer, scaler_age, scaler_seq,
            REGION_VOCAB, ISLAND_VOCAB, device
        )
        mut_predictions.append(mut_prob)

    # =========================================
    # Visualization
    # =========================================
    plt.figure(figsize=(12, 6))
    sns.kdeplot(wt_predictions, color='blue', label='Wild-Type Stability', fill=True, alpha=0.3)
    sns.kdeplot(mut_predictions, color='red', label='Mutated Stability', fill=True, alpha=0.3)
    
    wt_mean = np.mean(wt_predictions)
    mut_mean = np.mean(mut_predictions)
    
    plt.axvline(wt_mean, color='blue', linestyle='--', label=f'WT Mean: {wt_mean:.3f}')
    plt.axvline(mut_mean, color='red', linestyle='--', label=f'Mut Mean: {mut_mean:.3f}')
    
    plt.title(f'Monte Carlo Dropout Simulation (n={args.n_runs}) for {sample["Gene"]}')
    plt.xlabel('Predicted Methylation Beta')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, 'monte_carlo_stability.png')
    plt.savefig(plot_path)
    print(f"[✓] Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
