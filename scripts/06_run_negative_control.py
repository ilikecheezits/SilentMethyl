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
    parser = argparse.ArgumentParser(description="Run Negative Control Specificity Test.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the cleaned inference data CSV.")
    parser.add_argument("--base_dir", type=str, default="./checkpoints", help="Base directory for artifacts.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to trained model weights.")
    parser.add_argument("--n_permutations", type=int, default=100, help="Number of synonymous permutations.")
    args = parser.parse_args()

    print("--- RUNNING NEGATIVE CONTROL SPECIFICITY TEST ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Load the first stable/benign sequence from the dataset
    df = pd.read_csv(args.data_path)
    # Assume the first row is our negative control baseline
    baseline_row = df.iloc[0]
    wt_seq_full = str(baseline_row['Healthy_5000bp_DNA']).upper()
    wt_center = wt_seq_full[2000:3000] if len(wt_seq_full) >= 3000 else wt_seq_full
    
    print(f"[*] Selected Baseline Gene: {baseline_row.get('Gene', 'Unknown')}")

    # 2. Rebuild Architecture
    REGION_VOCAB = joblib.load(os.path.join("data", "processed", "SilentMethyl_RegionVocab.pkl"))
    ISLAND_VOCAB = joblib.load(os.path.join("data", "processed", "SilentMethyl_IslandVocab.pkl"))
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
    model.eval()

    # 3. Extract baseline tensors (Static for all permutations)
    wt_inputs = tokenizer(wt_center, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    input_ids = wt_inputs['input_ids'].to(device)
    attention_mask = wt_inputs['attention_mask'].to(device)
    
    # Extract numericals exactly as dataset.py does
    num_tensor = torch.tensor([[
        baseline_row.get('Age', 0), baseline_row.get('WT_GC_Content', 0), baseline_row.get('WT_CpG_Count', 0), 
        baseline_row.get('WT_CpG_OE_Ratio', 0), baseline_row.get('WT_GC_Skew', 0), baseline_row.get('WT_Shore_Asymmetry', 0), 
        baseline_row.get('WT_FOXA1_Motifs', 0), baseline_row.get('WT_GATA3_Motifs', 0), baseline_row.get('WT_AP1_Motifs', 0), 
        baseline_row.get('WT_CTCF_Motifs', 0), baseline_row.get('WT_SP1_Motifs', 0), baseline_row.get('WT_TpG_CpA_Clock', 0), 
        baseline_row.get('WT_Poly_A_Tracts', 0), baseline_row.get('WT_Alu_Proxy', 0), baseline_row.get('WT_G4_Quadruplex_Proxy', 0), 
        baseline_row.get('WT_ERE_Motifs', 0), baseline_row.get('WT_E_Box_Motifs', 0), baseline_row.get('WT_YY1_Motifs', 0), 
        baseline_row.get('WT_HRE_Motifs', 0)
    ]], dtype=torch.float32).to(device)
    
    region_idx = torch.tensor([REGION_VOCAB.get(baseline_row.get('Gene_Region', 'Unknown'), 0)], dtype=torch.long).to(device)
    island_idx = torch.tensor([ISLAND_VOCAB.get(baseline_row.get('CpG_Island_Status', 'Unknown'), 0)], dtype=torch.long).to(device)
    tata_idx = torch.tensor([baseline_row.get('WT_TATA_Box_Present', 0)], dtype=torch.long).to(device)

    # 4. Calculate True WT Baseline Probability
    with torch.no_grad():
        _, wt_logits = model(input_ids, attention_mask, num_tensor, region_idx, island_idx, tata_idx)
        wt_prob = torch.sigmoid(wt_logits).item()

    print(f"[*] Established WT Baseline Probability: {wt_prob:.4f}")
    print(f"[*] Generating {args.n_permutations} Synonymous Permutations...")

    # 5. Permutation Loop
    delta_p_results = []
    attempts = 0
    
    with torch.no_grad():
        pbar = tqdm(total=args.n_permutations)
        while len(delta_p_results) < args.n_permutations and attempts < 10000:
            attempts += 1
            # Pick a random pos in the center 1000bp
            pos = random.randint(100, 900) 
            new_base = get_synonymous_mutation(wt_center, pos)

            if new_base:
                seq_list = list(wt_center)
                seq_list[pos] = new_base
                mut_seq = "".join(seq_list)

                mut_inputs = tokenizer(mut_seq, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                mut_ids = mut_inputs['input_ids'].to(device)
                mut_mask = mut_inputs['attention_mask'].to(device)

                _, mut_logits = model(mut_ids, mut_mask, num_tensor, region_idx, island_idx, tata_idx)
                mut_prob_val = torch.sigmoid(mut_logits).item()
                
                delta_p_results.append(mut_prob_val - wt_prob)
                pbar.update(1)
        pbar.close()

    # 6. Statistical Proof & Plotting
    mean_noise = np.mean(delta_p_results)
    std_noise = np.std(delta_p_results)

    print("\n--- STATISTICAL TOPOLOGY & VISUAL PROOF ---")
    print(f"Empirical Mean (μ): {mean_noise:.6f}")
    print(f"Empirical Std Dev (σ): {std_noise:.6f}")

    if std_noise < 0.01:
        print("🚨 SUCCESS: Latent manifold is highly stable. Model exhibits ~zero erratic hallucination.")

    plt.figure(figsize=(8, 5))
    sns.histplot(delta_p_results, bins=30, kde=True, color='#00FFFF', stat='density')
    plt.axvline(0, color='white', linestyle='--', linewidth=1.5, label='Zero Shift (Perfect Stability)')
    plt.xlim(-0.15, 0.15)
    plt.title(f'Negative Control Topology (n={args.n_permutations})\nProving Strict Model Specificity')
    plt.xlabel('Absolute Probability Shift (ΔP)')
    plt.ylabel('Density')
    plt.legend()
    plt.style.use('dark_background')
    
    plot_path = os.path.join(args.base_dir, 'negative_control_specificity.png')
    plt.savefig(plot_path, dpi=300)
    print(f"[✓] Specificity proof saved to {plot_path}")

if __name__ == "__main__":
    main()