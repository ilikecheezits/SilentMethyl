import pandas as pd
import torch
import os
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.architecture import SilentMethylModel
from data.dataset import GenomicVariantDataset
from data.preprocess import preprocess_inference_data
from model.lora_utils import inject_lora_adapters, load_lora_weights

def main():
    parser = argparse.ArgumentParser(description="Run mass inference on mutated data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the inference data CSV.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory for artifacts and results.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M", help="Hugging Face model path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers.")
    args = parser.parse_args()

    print("--- RUNNING THE SILENT MUTATION INFERENCE ENGINE ---")

    # =========================================
    # Environment Setup
    # =========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.base_dir, exist_ok=True)

    # =========================================
    # Load Data and Artifacts
    # =========================================
    print(f"[*] Loading test dataset: {args.data_path}")
    df_test = pd.read_csv(args.data_path)

    print("[*] Loading Frozen Scalers and Vocabularies...")
    scaler_age = joblib.load(os.path.join(args.base_dir, "Scaler_Age.pkl"))
    scaler_seq = joblib.load(os.path.join(args.base_dir, "Scaler_Seq.pkl"))
    REGION_VOCAB = joblib.load(os.path.join(args.base_dir, "SilentMethyl_RegionVocab.pkl"))
    ISLAND_VOCAB = joblib.load(os.path.join(args.base_dir, "SilentMethyl_IslandVocab.pkl"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # =========================================
    # Preprocess Inference Data
    # =========================================
    print("[*] Preprocessing inference data...")
    df_test = preprocess_inference_data(df_test, scaler_age, scaler_seq)

    # =========================================
    # Create Dataset and DataLoader
    # =========================================
    inference_dataset = GenomicVariantDataset(df_test, tokenizer, REGION_VOCAB, ISLAND_VOCAB)
    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # =========================================
    # Model Rebuilding
    # =========================================
    print("[*] Rebuilding LoRA Architecture...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model = load_lora_weights(model, args.weights_path, device)
    model.eval()

    # =========================================
    # Dual-Pass Inference
    # =========================================
    print("[*] Running Dual-Pass Inference Loop (Wild-Type vs Mutated)...")
    results = []

    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Analyzing Variants"):
            region_idx = batch['region_idx'].to(device)
            island_idx = batch['island_idx'].to(device)
            
            _, wt_reg_logits = model(batch['wt_input_ids'].to(device), batch['wt_attention_mask'].to(device), batch['wt_num_tensor'].to(device), region_idx, island_idx, batch['wt_tata_idx'].to(device))
            wt_prob = torch.clamp(torch.sigmoid(wt_reg_logits), 0.0, 1.0)

            _, mut_reg_logits = model(batch['mut_input_ids'].to(device), batch['mut_attention_mask'].to(device), batch['mut_num_tensor'].to(device), region_idx, island_idx, batch['mut_tata_idx'].to(device))
            mut_prob = torch.clamp(torch.sigmoid(mut_reg_logits), 0.0, 1.0)
            
            raw_wt_prob = torch.sigmoid(wt_reg_logits)
            raw_mut_prob = torch.sigmoid(mut_reg_logits)
            relative_shift_pct = ((mut_prob - wt_prob) / (wt_prob + 1e-9)) * 100

            for i in range(len(batch['mutation_id'])):
                results.append({
                    'Mutation_ID': batch['mutation_id'][i],
                    'Gene': batch['gene'][i],
                    'HGVSp_Protein_Notation': batch['hgvsp'][i],
                    'True_Mutated_Beta': batch['true_beta'][i].item(),
                    'WT_Prob': wt_prob[i].item(),
                    'Mut_Prob': mut_prob[i].item(),
                    'Absolute_Delta_P': (mut_prob[i] - wt_prob[i]).item(),
                    'Relative_Shift_%': relative_shift_pct[i].item(),
                    'Raw_Neural_WT': raw_wt_prob[i].item(),
                    'Raw_Neural_Mut': raw_mut_prob[i].item(),
                    'Raw_Neural_Delta': (raw_mut_prob[i] - raw_wt_prob[i]).item()
                })

    df_results = pd.DataFrame(results)

    # =========================================
    # Statistical Analysis
    # =========================================
    print("[*] Performing statistical analysis and FDR correction...")
    delta_p = df_results['Absolute_Delta_P'].values
    empirical_sigma = 0.0058  # From negative control analysis in notebook
    z_scores = delta_p / empirical_sigma
    p_values = norm.sf(np.abs(z_scores)) * 2
    
    reject_null, pvals_corrected, _, _ = multipletests(p_values, alpha=0.01, method='fdr_bh')
    df_results['FDR_Pval'] = pvals_corrected
    df_results['Significant'] = reject_null
    
    # =========================================
    # Save and Display Results
    # =========================================
    OUTPUT_PATH = os.path.join(args.base_dir, "Final_Mutation_Inference_Results_Enhanced.csv")
    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Inference results saved to {OUTPUT_PATH}")

    print("" + "="*80)
    print("🧬 TOP REAL TCGA SYNONYMOUS VARIANTS (Sorted by RELATIVE Epigenetic Impact):")
    print("="*80)
    display_cols = ['Mutation_ID', 'Gene', 'WT_Prob', 'Mut_Prob', 'Absolute_Delta_P', 'Relative_Shift_%', 'Raw_Neural_Delta']
    print(df_results.iloc[3:].sort_values(by="Relative_Shift_%", key=abs, ascending=False)[display_cols].head(10))

    print("" + "="*60)
    print("🏆 GOLDEN CONTROL GROUND TRUTH RESULTS:")
    print("="*60)
    print(df_results.head(3))
    
    # =========================================
    # Volcano Plot
    # =========================================
    print("[*] Generating volcano plot...")
    plt.figure(figsize=(10, 7))
    neg_log10_p = -np.log10(np.clip(pvals_corrected, 1e-300, 1.0))
    
    plt.scatter(delta_p[~reject_null], neg_log10_p[~reject_null], color='grey', alpha=0.3, s=10)
    plt.scatter(delta_p[reject_null], neg_log10_p[reject_null], color='#FF1493', alpha=0.8, s=20)
    
    plt.axhline(y=-np.log10(0.01), color='white', linestyle='--')
    plt.axvline(x=0, color='white', linewidth=0.5)
    
    plt.title('SilentMethyl ISM: Genome-Wide Volcano Plot')
    plt.xlabel('Relative Probability Shift (ΔP)')
    plt.ylabel('-Log10 (FDR-Corrected p-value)')
    
    plot_path = os.path.join(args.base_dir, 'volcano_plot.png')
    plt.savefig(plot_path)
    print(f"[✓] Volcano plot saved to {plot_path}")

if __name__ == "__main__":
    main()
