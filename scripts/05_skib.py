import os
import sys
import argparse
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoConfig

sys.path.append(os.path.join(os.getcwd(), 'src'))
from model.architecture import SilentMethylModel, perform_triton_surgery
from model.lora_utils import inject_lora_adapters

def main():
    parser = argparse.ArgumentParser(description="Run In-Silico Mutagenesis on Synonymous Variants")
    parser.add_argument("--mutation_matrix", type=str, required=True, help="Path to your specific mutation testing data (CSV).")
    parser.add_argument("--weights_path", type=str, default="SilentMethyl_Best_Weights.pth", help="Path to trained weights.")
    parser.add_argument("--region_vocab", type=str, default="data/processed/SilentMethyl_RegionVocab.pkl", help="Path to region vocab.")
    parser.add_argument("--island_vocab", type=str, default="data/processed/SilentMethyl_IslandVocab.pkl", help="Path to island vocab.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save the ISM results.")
    args = parser.parse_args()

    # --- SETUP RIGOROUS LOGGING ---
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] ISM Engine Online. Running on: {device}")

    logger.info(f"[*] Loading Data: {args.mutation_matrix}")
    ism_df = pd.read_csv(args.mutation_matrix)
    ism_df = ism_df.dropna(subset=['Healthy_5000bp_DNA', 'Mutated_5000bp_DNA']).reset_index(drop=True)

    REGION_VOCAB = joblib.load(args.region_vocab)
    ISLAND_VOCAB = joblib.load(args.island_vocab)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    logger.info("[*] Initializing AI Architecture...")
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model) 
    model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
    model.eval()
    logger.info("[✓] Brain Loaded Successfully.")

    results = []
    logger.info("\n[*] Commencing Variant Effect Prediction (VEP)...")
    
    with torch.no_grad():
        for idx, row in tqdm(ism_df.iterrows(), total=len(ism_df), desc="Scanning"):
            
            # --- FROZEN METADATA VECTOR ---
            wt_num_feats = torch.tensor([[
                row.get('Age', 0), row.get('WT_mRNA_ZScore', row.get('mRNA_Z', 0)),
                row.get('WT_GC_Content', 0), row.get('WT_CpG_Count', 0),
                row.get('WT_CpG_OE_Ratio', 0), row.get('WT_GC_Skew', 0),
                row.get('WT_Shore_Asymmetry', 0), row.get('WT_FOXA1_Motifs', 0),
                row.get('WT_GATA3_Motifs', 0), row.get('WT_AP1_Motifs', 0),
                row.get('WT_CTCF_Motifs', 0), row.get('WT_SP1_Motifs', 0), 
                row.get('WT_TpG_CpA_Clock', 0), row.get('WT_Poly_A_Tracts', 0), 
                row.get('WT_Alu_Proxy', 0), row.get('WT_G4_Quadruplex_Proxy', 0),
                row.get('WT_ERE_Motifs', 0), row.get('WT_E_Box_Motifs', 0), 
                row.get('WT_YY1_Motifs', 0), row.get('WT_HRE_Motifs', 0)
            ]], dtype=torch.float32).to(device)

            reg_idx = torch.tensor([REGION_VOCAB.get(str(row.get('Gene_Region', 'Unknown')), 0)], dtype=torch.long).to(device)
            isl_idx = torch.tensor([ISLAND_VOCAB.get(str(row.get('CpG_Island_Status', 'Unknown')), 0)], dtype=torch.long).to(device)
            tata_idx = torch.tensor([int(row.get('WT_TATA_Box_Present', 0))], dtype=torch.long).to(device)

            # --- RUN WILD-TYPE ---
            wt_seq = str(row.get('Healthy_5000bp_DNA', '')).upper()
            mid_wt = len(wt_seq) // 2
            start_wt = max(0, mid_wt - 500)
            wt_center = wt_seq[start_wt:start_wt+1000]
            
            wt_tokens = tokenizer(wt_center, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            wt_logits, wt_debug = model(
                wt_tokens['input_ids'].to(device), wt_tokens['attention_mask'].to(device), 
                wt_num_feats, reg_idx, isl_idx, tata_idx
            )
            wt_prob = torch.sigmoid(wt_logits).item()

            # --- RUN MUTATED ---
            mut_seq = str(row.get('Mutated_5000bp_DNA', '')).upper()
            mid_mut = len(mut_seq) // 2
            start_mut = max(0, mid_mut - 500)
            mut_center = mut_seq[start_mut:start_mut+1000]

            mut_tokens = tokenizer(mut_center, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            mut_logits, mut_debug = model(
                mut_tokens['input_ids'].to(device), mut_tokens['attention_mask'].to(device), 
                wt_num_feats, reg_idx, isl_idx, tata_idx
            )
            mut_prob = torch.sigmoid(mut_logits).item()
            
            # --- 🚨 RIGOROUS INSPECTION LOGGING (Shows exactly what shifted) 🚨 ---
            mut_id = row.get('Mutation_ID', f"Var_{idx}")
            if idx < 3 or "CTRL" in mut_id.upper():  # Only print first few and controls to avoid console flood
                logger.info(f"\n[🔍 DEBUG] Inspecting: {mut_id}")
                logger.info(f"   -> Meta Logits (WT vs MUT): {wt_debug['meta_logits'][0].item():.4f} vs {mut_debug['meta_logits'][0].item():.4f}  <-- THESE MUST BE IDENTICAL!")
                logger.info(f"   -> DNA Logits  (WT vs MUT): {wt_debug['dna_logits'][0].item():.4f} vs {mut_debug['dna_logits'][0].item():.4f}  <-- THIS IS THE TRUE DELTA")
                logger.info(f"   -> Active DNA Weight Scaler : {wt_debug['dna_weight_val']:.4f}")
                logger.info(f"   -> Active Meta Weight Scaler: {wt_debug['meta_weight_val']:.4f}")

            delta_p = mut_prob - wt_prob
            results.append({
                'Mutation_ID': mut_id,
                'Gene': row.get('Gene', 'Unknown'),
                'WT_Prob': wt_prob,
                'Mut_Prob': mut_prob,
                'Raw_Delta': delta_p,
                'Relative_Shift_%': (delta_p / wt_prob * 100) if wt_prob > 0 else 0.0,
                'Abs_Delta': abs(delta_p)
            })

    # 6. SAVE RESULTS
    results_df = pd.DataFrame(results).sort_values(by='Abs_Delta', ascending=False)
    out_file = os.path.join(args.save_dir, "Top_Synonymous_Mutations_Impact.csv")
    results_df.to_csv(out_file, index=False)

    print("\n" + "="*80)
    print("🚨 TOP 15 SYNONYMOUS VARIANTS (MASSIVE EPIGENETIC DISRUPTIONS) 🚨")
    print("="*80)
    display_cols = ['Mutation_ID', 'Gene', 'WT_Prob', 'Mut_Prob', 'Raw_Delta', 'Relative_Shift_%']
    print(results_df.head(15)[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()