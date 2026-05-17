import os
import sys
import argparse
import math
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import mean_absolute_error

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
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sequences to process simultaneously on the GPU.")
    args = parser.parse_args()

    # --- SETUP RIGOROUS LOGGING ---
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] ISM Engine Online. Running on: {device} with Batch Size: {args.batch_size}")

    # 1. LOAD DATA
    logger.info(f"[*] Loading Data: {args.mutation_matrix}")
    ism_df = pd.read_csv(args.mutation_matrix)

    # 2. LOAD VOCABS
    REGION_VOCAB = joblib.load(args.region_vocab)
    ISLAND_VOCAB = joblib.load(args.island_vocab)

    # 3. INITIALIZE MODEL & TOKENIZER
    logger.info("[*] Initializing AI Architecture...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)

    # 4. LOAD WEIGHTS
    model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
    model.eval()
    logger.info("[✓] Brain Loaded Successfully.")

    # 5. RUN ISM OVER ALL VARIANTS (BATCHED)
    results = []
    actual_patient_betas = []
    predicted_patient_betas = []
    
    logger.info("\n[*] Commencing HIGH-THROUGHPUT Variant Effect Prediction (VEP)...")
    
    num_batches = math.ceil(len(ism_df) / args.batch_size)

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing Batches"):
            batch_df = ism_df.iloc[i * args.batch_size : (i + 1) * args.batch_size]

            wt_seqs, mut_seqs = [], []
            num_feats_list, reg_idx_list, isl_idx_list, tata_idx_list = [], [], [], []

            for _, row in batch_df.iterrows():
                # Extract numerical features
                num_feats_list.append([
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
                ])

                reg_idx_list.append(REGION_VOCAB.get(str(row.get('Gene_Region', 'Unknown')), 0))
                isl_idx_list.append(ISLAND_VOCAB.get(str(row.get('CpG_Island_Status', 'Unknown')), 0))
                tata_idx_list.append(int(row.get('WT_TATA_Box_Present', 0)))

                # Extract WT Sequence
                wt_seq = str(row.get('Healthy_5000bp_DNA', '')).upper()
                mid_wt = len(wt_seq) // 2
                start_wt = max(0, mid_wt - 500)
                wt_seqs.append(wt_seq[start_wt:start_wt+1000])

                # Extract Mutated Sequence
                mut_seq = str(row.get('Mutated_5000bp_DNA', '')).upper()
                mid_mut = len(mut_seq) // 2
                start_mut = max(0, mid_mut - 500)
                mut_seqs.append(mut_seq[start_mut:start_mut+1000])

            # Convert to tensors
            wt_num_feats = torch.tensor(num_feats_list, dtype=torch.float32).to(device)
            reg_idx = torch.tensor(reg_idx_list, dtype=torch.long).to(device)
            isl_idx = torch.tensor(isl_idx_list, dtype=torch.long).to(device)
            tata_idx = torch.tensor(tata_idx_list, dtype=torch.long).to(device)

            # Tokenize batches (padding dynamically applied up to max_length)
            wt_tokens = tokenizer(wt_seqs, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            mut_tokens = tokenizer(mut_seqs, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)

            # Run Inferences in Parallel
            wt_logits, wt_debug = model(wt_tokens['input_ids'], wt_tokens['attention_mask'], wt_num_feats, reg_idx, isl_idx, tata_idx)
            mut_logits, mut_debug = model(mut_tokens['input_ids'], mut_tokens['attention_mask'], wt_num_feats, reg_idx, isl_idx, tata_idx)

            # Flatten output tensors safely
            wt_probs = torch.sigmoid(wt_logits).reshape(-1).cpu().numpy()
            mut_probs = torch.sigmoid(mut_logits).reshape(-1).cpu().numpy()

            wt_dna_logits = wt_debug['dna_logits'].reshape(-1).cpu().numpy()
            mut_dna_logits = mut_debug['dna_logits'].reshape(-1).cpu().numpy()
            wt_meta_logits = wt_debug['meta_logits'].reshape(-1).cpu().numpy()
            mut_meta_logits = mut_debug['meta_logits'].reshape(-1).cpu().numpy()

            # Helper to extract scalar weights regardless of tensor shape
            def extract_weight(weight_val, batch_len):
                if isinstance(weight_val, torch.Tensor):
                    w = weight_val.reshape(-1).cpu().numpy()
                    if len(w) == 1 and batch_len > 1:
                        return np.repeat(w, batch_len)
                    return w
                return np.full(batch_len, float(weight_val))

            dna_weights = extract_weight(wt_debug['dna_weight_val'], len(batch_df))
            meta_weights = extract_weight(wt_debug['meta_weight_val'], len(batch_df))

            # Store Results
            for j in range(len(batch_df)):
                row = batch_df.iloc[j]
                mut_id = row.get('Mutation_ID', f"Var_{i * args.batch_size + j}")
                true_beta = row.get('True_Mutated_Beta', np.nan)

                wt_p = wt_probs[j]
                mut_p = mut_probs[j]
                delta_p = mut_p - wt_p

                wt_dna_log = wt_dna_logits[j]
                mut_dna_log = mut_dna_logits[j]
                w_dna = dna_weights[j]

                wt_meta_log = wt_meta_logits[j]
                mut_meta_log = mut_meta_logits[j]
                w_meta = meta_weights[j]

                true_logit_delta = (mut_dna_log - wt_dna_log) * w_dna

                # Print debug telemetry only for the Golden Controls or the very first couple of variants
                if "CTRL" in str(mut_id).upper() or (i == 0 and j < 3):
                    logger.info(f"\n[🔍 DEBUG] Inspecting: {mut_id}")
                    logger.info(f"   -> Meta Logits (WT vs MUT): {wt_meta_log:.4f} vs {mut_meta_log:.4f}  <-- THESE MUST BE IDENTICAL!")
                    logger.info(f"   -> DNA Logits  (WT vs MUT): {wt_dna_log:.4f} vs {mut_dna_log:.4f}  <-- THIS IS THE TRUE DELTA")
                    logger.info(f"   -> Active DNA Weight Scaler : {w_dna:.4f}")
                    logger.info(f"   -> Active Meta Weight Scaler: {w_meta:.4f}")

                results.append({
                    'Mutation_ID': mut_id,
                    'Gene': row.get('Gene', 'Unknown'),
                    'True_Beta': true_beta,
                    'WT_Prob': wt_p,
                    'Mut_Prob': mut_p,
                    'Raw_Delta': delta_p,
                    'Relative_Shift_%': (delta_p / wt_p * 100) if wt_p > 0 else 0.0,
                    'Abs_Delta': abs(delta_p),
                    'Logit_Delta': true_logit_delta,
                    'Abs_Logit_Delta': abs(true_logit_delta)
                })

                if not str(mut_id).startswith("GOLDEN") and not pd.isna(true_beta):
                    actual_patient_betas.append(true_beta)
                    predicted_patient_betas.append(mut_p)

    # 6. SAVE RESULTS
    results_df = pd.DataFrame(results).sort_values(by='Abs_Logit_Delta', ascending=False)
    out_file = os.path.join(args.save_dir, "Top_Synonymous_Mutations_Impact.csv")
    results_df.to_csv(out_file, index=False)
    logger.info(f"\n[*] Done! Results saved to {out_file}")

    # Print the new true Logit Leaderboard
    print("\n" + "="*80)
    print("🚨 TOP SYNONYMOUS VARIANTS (SORTED BY TRUE LOGIT DELTA ΔZ) 🚨")
    print("="*80)
    print(f"{len(results_df)} variants analyzed. Displaying top 15 with highest absolute logit shifts:\n")
    display_df = results_df[['Mutation_ID', 'Gene', 'Logit_Delta', 'WT_Prob', 'Mut_Prob']].head(15)
    print(display_df.to_string(index=False))

    # --- THE FINAL MAE REVEAL ---
    if len(actual_patient_betas) > 0:
        final_mae = mean_absolute_error(actual_patient_betas, predicted_patient_betas)
        print("\n" + "="*80)
        print("🎯 MODEL ACCURACY ON UNSEEN TCGA PATIENTS 🎯")
        print("="*80)
        print(f"-> Evaluated {len(actual_patient_betas)} real clinical mutations.")
        print(f"-> Mean Absolute Error (MAE): {final_mae:.4f}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
