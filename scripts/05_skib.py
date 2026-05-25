import os
import sys
import argparse
import math
import torch
import joblib
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.join(os.getcwd(), 'src'))
from model.architecture import SilentMethylModel, perform_triton_surgery
from model.lora_utils import inject_lora_adapters

def get_motif_count(seq, motifs):
    return sum(len(re.findall(m, seq)) for m in motifs)

def main():
    parser = argparse.ArgumentParser(description="Run In-Silico Mutagenesis on Synonymous Variants")
    parser.add_argument("--mutation_matrix", type=str, required=True, help="Path to mutation data.")
    parser.add_argument("--weights_path", type=str, default="SilentMethyl_Best_Weights.pth", help="Path to weights.")
    parser.add_argument("--region_vocab", type=str, default="data/processed/SilentMethyl_RegionVocab.pkl")
    parser.add_argument("--island_vocab", type=str, default="data/processed/SilentMethyl_IslandVocab.pkl")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] ISM Engine Online. Running on: {device}")

    ism_df = pd.read_csv(args.mutation_matrix)
    REGION_VOCAB = joblib.load(args.region_vocab)
    ISLAND_VOCAB = joblib.load(args.island_vocab)

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
    model.eval()

    results = []
    logger.info("[!] Anti-Leakage Patch V2 Active: DIFFERENTIAL TABULAR RECALCULATION ON ALL FEATURES.")
    
    num_batches = math.ceil(len(ism_df) / args.batch_size)

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing Batches"):
            batch_df = ism_df.iloc[i * args.batch_size : (i + 1) * args.batch_size]

            wt_seqs, mut_seqs = [], []
            wt_num_feats_list, mut_num_feats_list = [], []
            reg_idx_list, isl_idx_list, tata_idx_list = [], [], []

            for _, row in batch_df.iterrows():
                wt_seq_full = str(row.get('Healthy_5000bp_DNA', '')).upper()
                mut_seq_full = str(row.get('Mutated_5000bp_DNA', '')).upper()

                # Slice for Transformer
                mid_wt = len(wt_seq_full) // 2
                wt_seqs.append(wt_seq_full[max(0, mid_wt - 500):max(0, mid_wt - 500)+1000])
                mut_seqs.append(mut_seq_full[max(0, mid_wt - 500):max(0, mid_wt - 500)+1000])

                # --- 1. BASELINE WT FEATURES ---
                wt_feats = [
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
                ]
                wt_num_feats_list.append(wt_feats)

                # --- 2. DIFFERENTIAL MUTATION CALCULATOR ---
                L = max(1, len(wt_seq_full))
                
                # Biometrics Delta
                d_gc = (mut_seq_full.count('G') + mut_seq_full.count('C')) / L - (wt_seq_full.count('G') + wt_seq_full.count('C')) / L
                d_cpg = mut_seq_full.count('CG') - wt_seq_full.count('CG')
                
                g_w, c_w = wt_seq_full.count('G'), wt_seq_full.count('C')
                g_m, c_m = mut_seq_full.count('G'), mut_seq_full.count('C')
                d_skew = ((g_m - c_m) / max(1, g_m + c_m)) - ((g_w - c_w) / max(1, g_w + c_w))
                
                d_oe = ((mut_seq_full.count('CG') * L) / max(1, g_m * c_m)) - ((wt_seq_full.count('CG') * L) / max(1, g_w * c_w))

                # Motif Deltas (Core strings)
                def d_motif(motifs): return get_motif_count(mut_seq_full, motifs) - get_motif_count(wt_seq_full, motifs)

                # --- 3. APPLY DELTAS TO MUTATED TENSOR ---
                mut_feats = wt_feats.copy()
                mut_feats[2] += d_gc                                        # GC_Content
                mut_feats[3] += d_cpg                                       # CpG_Count
                mut_feats[4] += d_oe                                        # CpG_OE_Ratio
                mut_feats[5] += d_skew                                      # GC_Skew
                mut_feats[7] += d_motif(['GTAAACA', 'TGTTTAC'])             # FOXA1
                mut_feats[8] += d_motif(['GATA', 'TATC'])                   # GATA3
                mut_feats[9] += d_motif(['TGACTCA', 'TGAGTCA'])             # AP1
                mut_feats[10] += d_motif(['CCGCG', 'CGCGG'])                # CTCF
                mut_feats[11] += d_motif(['GGGCGG', 'CCGCCC'])              # SP1
                mut_feats[12] += d_motif(['TG', 'CA'])                      # TpG_CpA
                mut_feats[13] += d_motif(['AAAA', 'TTTT'])                  # Poly_A
                mut_feats[15] += d_motif(['GGG'])                           # G4 Proxy
                mut_feats[16] += d_motif(['AGGTC', 'GACCT'])                # ERE
                mut_feats[17] += d_motif(['CAGCTG', 'CACGTG'])              # E_Box
                mut_feats[18] += d_motif(['CCAT', 'ATGG'])                  # YY1
                mut_feats[19] += d_motif(['TACGTG', 'CACGTA'])              # HRE
                
                mut_num_feats_list.append(mut_feats)

                # Categoricals
                reg_idx_list.append(REGION_VOCAB.get(str(row.get('Gene_Region', 'Unknown')), 0))
                isl_idx_list.append(ISLAND_VOCAB.get(str(row.get('CpG_Island_Status', 'Unknown')), 0))
                tata_idx_list.append(int(row.get('WT_TATA_Box_Present', 0)))

            # To Tensors
            wt_num_feats = torch.tensor(wt_num_feats_list, dtype=torch.float32).to(device)
            mut_num_feats = torch.tensor(mut_num_feats_list, dtype=torch.float32).to(device)
            
            reg_idx = torch.tensor(reg_idx_list, dtype=torch.long).to(device)
            isl_idx = torch.tensor(isl_idx_list, dtype=torch.long).to(device)
            tata_idx = torch.tensor(tata_idx_list, dtype=torch.long).to(device)

            wt_tokens = tokenizer(wt_seqs, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            mut_tokens = tokenizer(mut_seqs, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)

            wt_logits, wt_debug = model(wt_tokens['input_ids'], wt_tokens['attention_mask'], wt_num_feats, reg_idx, isl_idx, tata_idx)
            mut_logits, mut_debug = model(mut_tokens['input_ids'], mut_tokens['attention_mask'], mut_num_feats, reg_idx, isl_idx, tata_idx)

            wt_probs = torch.sigmoid(wt_logits).reshape(-1).cpu().numpy()
            mut_probs = torch.sigmoid(mut_logits).reshape(-1).cpu().numpy()

            for j in range(len(batch_df)):
                row = batch_df.iloc[j]
                mut_id = row.get('Mutation_ID', f"Var_{i * args.batch_size + j}")
                
                delta_p = mut_probs[j] - wt_probs[j]
                true_logit_delta = mut_debug['dna_logits'].reshape(-1)[j].item() - wt_debug['dna_logits'].reshape(-1)[j].item()

                results.append({
                    'Mutation_ID': mut_id,
                    'Gene': row.get('Gene', 'Unknown'),
                    'Logit_Delta': true_logit_delta,
                    'Abs_Logit_Delta': abs(true_logit_delta)
                })

    results_df = pd.DataFrame(results).sort_values(by='Abs_Logit_Delta', ascending=False)
    out_file = os.path.join(args.save_dir, "Top_Synonymous_Mutations_Impact.csv")
    results_df.to_csv(out_file, index=False)
    print("\n" + "="*80)
    print("🚨 TOP SYNONYMOUS VARIANTS (SORTED BY TRUE LOGIT DELTA ΔZ) 🚨")
    print("="*80)
    print(f"{len(results_df)} variants analyzed. Displaying top 15 with highest absolute logit shifts:\n")
    display_df = results_df[['Mutation_ID', 'Gene', 'Logit_Delta']].head(15)
    print(display_df.to_string(index=False))
    logger.info(f"\n[*] Done! ALL tabular features dynamically updated. Results saved.")

if __name__ == "__main__":
    main()
