import os
import sys
import argparse
import math
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

sys.path.append(os.path.join(os.getcwd(), 'src'))
from model.architecture import SilentMethylModel, perform_triton_surgery
from model.lora_utils import inject_lora_adapters

def enable_mc_dropout(model):
    """Forces all Dropout layers to remain active during evaluation for MC Sampling."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo In-Silico Mutagenesis")
    parser.add_argument("--mutation_matrix", type=str, required=True)
    parser.add_argument("--weights_path", type=str, default="SilentMethyl_Best_Weights.pth")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mc_samples", type=int, default=30, help="Number of stochastic forward passes.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing MC-Dropout Engine on {device} | N={args.mc_samples}")

    ism_df = pd.read_csv(args.mutation_matrix)
    REGION_VOCAB = joblib.load("data/processed/SilentMethyl_RegionVocab.pkl")
    ISLAND_VOCAB = joblib.load("data/processed/SilentMethyl_IslandVocab.pkl")

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
    
    # Enable eval mode for LayerNorms, but force Dropout active for MC
    model.eval()
    enable_mc_dropout(model)

    results = []
    num_batches = math.ceil(len(ism_df) / args.batch_size)

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="MC-Dropout Batches"):
            batch_df = ism_df.iloc[i * args.batch_size : (i + 1) * args.batch_size]

            wt_seqs, mut_seqs, num_feats_list, reg_idx_list, isl_idx_list, tata_idx_list = [], [], [], [], [], []

            for _, row in batch_df.iterrows():
                # Extract your standard 20 features here (condensed for space)
                num_feats_list.append([row.get(col, 0) for col in ['Age', 'WT_mRNA_ZScore', 'WT_GC_Content', 'WT_CpG_Count', 'WT_CpG_OE_Ratio', 'WT_GC_Skew', 'WT_Shore_Asymmetry', 'WT_FOXA1_Motifs', 'WT_GATA3_Motifs', 'WT_AP1_Motifs', 'WT_CTCF_Motifs', 'WT_SP1_Motifs', 'WT_TpG_CpA_Clock', 'WT_Poly_A_Tracts', 'WT_Alu_Proxy', 'WT_G4_Quadruplex_Proxy', 'WT_ERE_Motifs', 'WT_E_Box_Motifs', 'WT_YY1_Motifs', 'WT_HRE_Motifs']])
                reg_idx_list.append(REGION_VOCAB.get(str(row.get('Gene_Region', 'Unknown')), 0))
                isl_idx_list.append(ISLAND_VOCAB.get(str(row.get('CpG_Island_Status', 'Unknown')), 0))
                tata_idx_list.append(int(row.get('WT_TATA_Box_Present', 0)))

                mid = len(str(row.get('Healthy_5000bp_DNA', ''))) // 2
                wt_seqs.append(str(row.get('Healthy_5000bp_DNA', ''))[mid-500:mid+500].upper())
                mut_seqs.append(str(row.get('Mutated_5000bp_DNA', ''))[mid-500:mid+500].upper())

            # Tensors
            wt_num_feats = torch.tensor(num_feats_list, dtype=torch.float32).to(device)
            reg_idx = torch.tensor(reg_idx_list, dtype=torch.long).to(device)
            isl_idx = torch.tensor(isl_idx_list, dtype=torch.long).to(device)
            tata_idx = torch.tensor(tata_idx_list, dtype=torch.long).to(device)

            wt_tokens = tokenizer(wt_seqs, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            mut_tokens = tokenizer(mut_seqs, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)

            # MC Sampling Loop
            batch_deltas = [[] for _ in range(len(batch_df))]
            
            for _ in range(args.mc_samples):
                _, wt_debug = model(wt_tokens['input_ids'], wt_tokens['attention_mask'], wt_num_feats, reg_idx, isl_idx, tata_idx)
                _, mut_debug = model(mut_tokens['input_ids'], mut_tokens['attention_mask'], wt_num_feats, reg_idx, isl_idx, tata_idx)
                
                wt_dna = wt_debug['dna_logits'].reshape(-1).cpu().numpy()
                mut_dna = mut_debug['dna_logits'].reshape(-1).cpu().numpy()
                
                # Handle scaler dimension properly
                w_val = wt_debug['dna_weight_val']
                if isinstance(w_val, torch.Tensor):
                    w_dna = w_val.reshape(-1).cpu().numpy()
                    if len(w_dna) == 1: w_dna = np.repeat(w_dna, len(batch_df))
                else:
                    w_dna = np.full(len(batch_df), float(w_val))
                
                for j in range(len(batch_df)):
                    batch_deltas[j].append((mut_dna[j] - wt_dna[j]) * w_dna[j])

            # Statistical Aggregation
            for j in range(len(batch_df)):
                deltas = np.array(batch_deltas[j])
                mean_delta = np.mean(deltas)
                var_delta = np.var(deltas)
                std_delta = np.std(deltas) + 1e-8 # Prevent div by zero
                
                # Calculate Epistemic Confidence (Z-score of the shift against the model's own variance)
                confidence_z = abs(mean_delta) / std_delta

                results.append({
                    'Mutation_ID': batch_df.iloc[j]['Mutation_ID'],
                    'Gene': batch_df.iloc[j].get('Gene', 'Unknown'),
                    'MC_Mean_Logit_Delta': mean_delta,
                    'MC_Std_Delta': std_delta,
                    'Epistemic_Confidence': confidence_z
                })

    df_res = pd.DataFrame(results)
    # Filter out noisy predictions: We only want shifts where the Mean is at least 3 standard deviations above the noise
    df_res = df_res[df_res['Epistemic_Confidence'] > 3.0].sort_values(by='MC_Mean_Logit_Delta', key=abs, ascending=False)
    
    out_path = os.path.join(args.save_dir, "MC_Confident_Mutations.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\n[✓] MC Inference complete! High-confidence variants saved to {out_path}")

if __name__ == "__main__":
    main()