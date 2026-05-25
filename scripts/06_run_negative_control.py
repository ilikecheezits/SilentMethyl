import pandas as pd
import torch
import os
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model.architecture import SilentMethylModel, perform_triton_surgery
from model.lora_utils import inject_lora_adapters

# Standard Genetic Code Map for synonymous generation
CODON_MAP = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'
}

def get_synonymous_mutation(sequence, pos):
    """Finds a valid synonymous mutation at a specific position."""
    codon_start = pos - (pos % 3)
    original_codon = sequence[codon_start:codon_start+3]
    if len(original_codon) != 3 or original_codon not in CODON_MAP:
        return None

    original_aa = CODON_MAP[original_codon]
    valid_codons = [c for c, aa in CODON_MAP.items() if aa == original_aa and c != original_codon]
    
    if not valid_codons:
        return None
        
    new_codon = random.choice(valid_codons)
    mutation_idx_in_codon = pos % 3
    return new_codon[mutation_idx_in_codon]

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Logit Distribution Test.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to cleaned inference data CSV.")
    parser.add_argument("--base_dir", type=str, default="./checkpoints", help="Base directory for artifacts.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to trained model weights.")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Number of Monte Carlo iterations.")
    args = parser.parse_args()

    print("--- RUNNING MONTE CARLO NULL DISTRIBUTION TEST ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Load the dataset (excluding our Golden Controls so they don't bias the background)
    df = pd.read_csv(args.data_path)
    df_real = df[~df['Mutation_ID'].str.startswith('GOLDEN')].copy()
    
    # 2. Rebuild Architecture
    print("[*] Initializing AI Architecture...")
    REGION_VOCAB = joblib.load(os.path.join("data", "processed", "SilentMethyl_RegionVocab.pkl"))
    ISLAND_VOCAB = joblib.load(os.path.join("data", "processed", "SilentMethyl_IslandVocab.pkl"))
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
    model.eval()

    print(f"[*] Generating {args.n_permutations} Synonymous Permutations across random patient backgrounds...")

    logit_delta_results = []
    attempts = 0
    
    with torch.no_grad():
        pbar = tqdm(total=args.n_permutations)
        while len(logit_delta_results) < args.n_permutations and attempts < (args.n_permutations * 10):
            attempts += 1
            
            # Sample a random patient background
            row = df_real.sample(1).iloc[0]
            wt_seq_full = str(row['Healthy_5000bp_DNA']).upper()
            wt_center = wt_seq_full[2000:3000] if len(wt_seq_full) >= 3000 else wt_seq_full
            
            # --- FIX: SAFE BOUNDARIES ---
            # Prevent IndexError by bounding the random choice strictly to the actual sequence length
            upper_bound = min(600, len(wt_center) - 4)
            lower_bound = min(400, max(0, upper_bound - 1))
            pos = random.randint(lower_bound, upper_bound)
            new_base = get_synonymous_mutation(wt_center, pos)

            if new_base:
                # Build mutant sequence
                seq_list = list(wt_center)
                seq_list[pos] = new_base
                mut_center = "".join(seq_list)

                # Tokenize
                wt_inputs = tokenizer(wt_center, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                mut_inputs = tokenizer(mut_center, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                
                # Extract numericals safely
                # Extract numericals safely (Now including mRNA_Z!)
                num_tensor = torch.tensor([[
                    row.get('Age', 0), row.get('mRNA_Z', 0), row.get('WT_GC_Content', 0), row.get('WT_CpG_Count', 0), 
                    row.get('WT_CpG_OE_Ratio', 0), row.get('WT_GC_Skew', 0), row.get('WT_Shore_Asymmetry', 0), 
                    row.get('WT_FOXA1_Motifs', 0), row.get('WT_GATA3_Motifs', 0), row.get('WT_AP1_Motifs', 0), 
                    row.get('WT_CTCF_Motifs', 0), row.get('WT_SP1_Motifs', 0), row.get('WT_TpG_CpA_Clock', 0), 
                    row.get('WT_Poly_A_Tracts', 0), row.get('WT_Alu_Proxy', 0), row.get('WT_G4_Quadruplex_Proxy', 0), 
                    row.get('WT_ERE_Motifs', 0), row.get('WT_E_Box_Motifs', 0), row.get('WT_YY1_Motifs', 0), 
                    row.get('WT_HRE_Motifs', 0)
                ]], dtype=torch.float32).to(device)
                
                region_idx = torch.tensor([REGION_VOCAB.get(row.get('Gene_Region', 'Unknown'), 0)], dtype=torch.long).to(device)
                island_idx = torch.tensor([ISLAND_VOCAB.get(row.get('CpG_Island_Status', 'Unknown'), 0)], dtype=torch.long).to(device)
                tata_idx = torch.tensor([row.get('WT_TATA_Box_Present', 0)], dtype=torch.long).to(device)

                # Run WT Inference
                out_wt = model(wt_inputs['input_ids'].to(device), wt_inputs['attention_mask'].to(device), num_tensor, region_idx, island_idx, tata_idx)
                # Safely parse tuple return
                debug_wt = out_wt[1] if isinstance(out_wt[1], dict) else out_wt[0]

                # Run MUT Inference
                out_mut = model(mut_inputs['input_ids'].to(device), mut_inputs['attention_mask'].to(device), num_tensor, region_idx, island_idx, tata_idx)
                debug_mut = out_mut[1] if isinstance(out_mut[1], dict) else out_mut[0]
                
                # --- THE PURE ESTIMATOR ---
                wt_dna_logit = debug_wt['dna_logits'][0].item()
                mut_dna_logit = debug_mut['dna_logits'][0].item()
                dna_weight = debug_wt['dna_weight_val']
                
                true_logit_delta = (mut_dna_logit - wt_dna_logit) * dna_weight
                logit_delta_results.append(true_logit_delta)
                
                pbar.update(1)
        pbar.close()

    # 6. Statistical Proof & Plotting
    mean_noise = np.mean(logit_delta_results)
    std_noise = np.std(logit_delta_results)
    
    # Calculate Statistical Boundaries
    p_01_bound = 2.58 * std_noise
    p_001_bound = 3.29 * std_noise

    print("\n--- MONTE CARLO STATISTICAL TOPOLOGY ---")
    print(f"Empirical Mean (μ): {mean_noise:.6f}")
    print(f"Empirical Std Dev (σ): {std_noise:.6f}")
    print(f"p < 0.01 Boundary (2.58σ): ±{p_01_bound:.6f}")
    print(f"p < 0.001 Boundary (3.29σ): ±{p_001_bound:.6f}")

    # Plot the True Delta Z Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(logit_delta_results, bins=50, kde=True, color='#00FFFF', stat='density')
    
    # Add Statistical Boundaries
    plt.axvline(0, color='white', linestyle='-', linewidth=2, label='Zero Shift (Mean)')
    plt.axvline(p_01_bound, color='#FF3333', linestyle='--', linewidth=1.5, label=f'p < 0.01 (+{p_01_bound:.4f})')
    plt.axvline(-p_01_bound, color='#FF3333', linestyle='--', linewidth=1.5)
    plt.axvline(p_001_bound, color='#FF00FF', linestyle=':', linewidth=1.5, label=f'p < 0.001 (+{p_001_bound:.4f})')
    plt.axvline(-p_001_bound, color='#FF00FF', linestyle=':', linewidth=1.5)

    # Styling
    plt.title(f'Monte Carlo Null Distribution of Logit Difference (ΔZ)\nn={args.n_permutations} Synonymous Mutations', pad=15)
    plt.xlabel('Unbiased Causal Shift (ΔZ)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.style.use('dark_background')
    plt.tight_layout()
    
    plot_path = os.path.join(args.base_dir, 'Monte_Carlo_Null_Distribution.png')
    plt.savefig(plot_path, dpi=300)
    
    # Save the raw data
    csv_path = os.path.join(args.base_dir, 'Monte_Carlo_Raw_Logits.csv')
    pd.DataFrame({'Logit_Delta': logit_delta_results}).to_csv(csv_path, index=False)
    
    print(f"[✓] Visual proof saved to {plot_path}")
    print(f"[✓] Raw data saved to {csv_path}")

if __name__ == "__main__":
    main()
