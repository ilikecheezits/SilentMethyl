import os
import re
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pyfaidx import Fasta
from transformers import AutoTokenizer, AutoConfig, AutoModel
import logging
import random
import scipy.stats as stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TEST_CSV_PATH = 'data/datafiles/testing_data.csv'
HG38_PATH = 'data/hg38.fa' 
MODEL_WEIGHTS = 'checkpoints_baseline/baseline_best_weights.pth' 
BASE_DIR = 'results/baseline'
WINDOW_SIZE = 1000
MC_SAMPLES = 1000  

os.makedirs(BASE_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"[*] Using device: {DEVICE}")

# =============================================================================
# 2. ARCHITECTURE DEFINITIONS (SYNCED TO TRAINING)
# =============================================================================
def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local"):
    logging.info("--- Performing DNABERT-2 Surgery & Patching ---")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        from huggingface_hub import snapshot_download
        cache_path = snapshot_download(model_path)
        for item in os.listdir(cache_path):
            src = os.path.join(cache_path, item)
            dst = os.path.join(local_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        triton_file = os.path.join(local_dir, "flash_attn_triton.py")
        if os.path.exists(triton_file):
            with open(triton_file, "w") as f:
                f.write("def __getattr__(name):\n    return None\n")

        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        config_data["use_flash_attn"] = False
        if "pad_token_id" not in config_data or config_data["pad_token_id"] is None:
            config_data["pad_token_id"] = 0 
            
        with open(config_path, "w") as f:
            json.dump(config_data, f)
    
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    config.output_attentions = True 
    base_model = AutoModel.from_config(config, trust_remote_code=True)
    return config, base_model

class BaselineDNABert(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M"):
        super(BaselineDNABert, self).__init__()
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size
        
        # FIXED: Synced architecture layers!
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        self.classification_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.regression_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        
        # FIXED: Synced forward pass
        hidden_states_t = hidden_states.permute(0, 2, 1)
        spatial_features = F.relu(self.spatial_conv(hidden_states_t)).permute(0, 2, 1)
        
        attn_weights = self.attention_pool(spatial_features).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        pooled_output = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)

        class_logits = self.classification_head(pooled_output)
        m_value_pred = self.regression_head(pooled_output)
        return class_logits, m_value_pred

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
def m_value_to_beta(m_val):
    m_val = np.clip(m_val, -20, 20) # Prevent overflow warnings
    return (2 ** m_val) / (1 + (2 ** m_val))

def parse_mutation_id(mut_id):
    mut_str = str(mut_id).upper()
    if mut_str == 'NAN': return None, None, None
    match = re.search(r'(\d+)\s*([ACGT])\s*>\s*([ACGT])', mut_str)
    if match: return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None

def get_anchored_sequence(genome, chrom, cpg_pos, window=1000):
    half_window = window // 2
    start_idx = (cpg_pos - 1) - half_window
    end_idx = start_idx + window
    return str(genome[chrom][start_idx:end_idx]).upper()

def apply_mutation(seq, cpg_pos, mut_pos, alt_base, window=1000):
    half_window = window // 2
    rel_idx = half_window + (mut_pos - cpg_pos)
    if rel_idx < 0 or rel_idx >= len(seq): return seq 
    return seq[:rel_idx] + alt_base + seq[rel_idx+1:]

@torch.no_grad()
def batch_inference(model, tokenizer, sequences, batch_size=32):
    all_m = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        encodings = tokenizer(batch, truncation=True, max_length=WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            _, m_preds = model(encodings['input_ids'], encodings['attention_mask'])
            all_m.extend(m_preds.cpu().flatten().tolist())
    return np.array(all_m)

# =============================================================================
# 4. MAIN INFERENCE PIPELINE
# =============================================================================
def main():
    logging.info("[*] Initializing Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BaselineDNABert().to(DEVICE)

    if os.path.exists(MODEL_WEIGHTS):
        # Strict=True ensures we catch architecture mismatches
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True)
        logging.info("[+] Weights loaded successfully.")
    else:
        logging.warning("[!] Weights file not found. Ensure MODEL_WEIGHTS path is correct.")
        return
    model.eval()

    logging.info(f"[*] Loading Genome and Test Data...")
    genome = Fasta(HG38_PATH)
    df = pd.read_csv(TEST_CSV_PATH)
    df_sample = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)

    logging.info("[*] Phase 1: Screening tagged mutations in CSV...")
    valid_rows, wt_seqs, mut_seqs = [], [], []

    for idx, row in df_sample.iterrows():
        chrom, cpg_pos, mut_id = str(row['chr']).strip(), int(row['pos']), row['Mutation_ID']
        if chrom not in genome: continue
        
        mut_pos, ref_base, alt_base = parse_mutation_id(mut_id)
        if not mut_pos: continue
            
        wt_seq = get_anchored_sequence(genome, chrom, cpg_pos, WINDOW_SIZE)
        mut_seq = apply_mutation(wt_seq, cpg_pos, mut_pos, alt_base, WINDOW_SIZE)
        
        valid_rows.append(row)
        wt_seqs.append(wt_seq)
        mut_seqs.append(mut_seq)

    logging.info(f"    -> Running batch inference on {len(valid_rows)} pairs...")
    wt_m_vals = batch_inference(model, tokenizer, wt_seqs)
    mut_m_vals = batch_inference(model, tokenizer, mut_seqs)

    wt_betas = m_value_to_beta(wt_m_vals)
    mut_betas = m_value_to_beta(mut_m_vals)
    deltas = mut_betas - wt_betas

    # Grab the top 10 most impactful mutations instead of just 3
    top_k = min(10, len(deltas))
    top_indices = np.argsort(np.abs(deltas))[-top_k:][::-1]

    logging.info(f"[*] Phase 2: Generating {MC_SAMPLES} background perturbations for Top {top_k} Targets...")
    bases = ['A', 'C', 'G', 'T']
    
    # Store results for the CSV export
    mc_results = []

    for rank, best_idx in enumerate(top_indices):
        target_row = valid_rows[best_idx]
        target_wt_seq = wt_seqs[best_idx]
        target_tagged_delta = deltas[best_idx]
        target_gene = str(target_row['Gene']) if pd.notna(target_row['Gene']) else f"Intergenic_{target_row['chr']}"
        
        logging.info(f"    -> Target #{rank+1}: {target_gene} ({target_row['Mutation_ID']}) | Delta: {target_tagged_delta:.4f}")
        
        mc_mutated_seqs = []
        for _ in range(MC_SAMPLES):
            rand_idx = random.choice([i for i in range(50, WINDOW_SIZE - 50) if i not in [500, 501]])
            orig_base = target_wt_seq[rand_idx]
            alt_base = random.choice([b for b in bases if b != orig_base])
            mc_seq = target_wt_seq[:rand_idx] + alt_base + target_wt_seq[rand_idx+1:]
            mc_mutated_seqs.append(mc_seq)

        mc_mut_m_vals = batch_inference(model, tokenizer, mc_mutated_seqs)
        mc_deltas = m_value_to_beta(mc_mut_m_vals) - wt_betas[best_idx]
        
        # Calculate robust statistics
        noise_mean = np.mean(mc_deltas)
        noise_std = np.std(mc_deltas)
        z_score = (target_tagged_delta - noise_mean) / (noise_std + 1e-9)
        
        # Empirical P-Value (two-tailed)
        p_val = np.sum(np.abs(mc_deltas) >= np.abs(target_tagged_delta)) / MC_SAMPLES
        
        mc_results.append({
            'Rank': rank + 1,
            'Gene': target_gene,
            'Mutation_ID': target_row['Mutation_ID'],
            'Target_CpG': target_row['probeID'],
            'Predicted_Delta_Beta': round(target_tagged_delta, 4),
            'MC_Background_Mean': round(noise_mean, 4),
            'MC_Background_STD': round(noise_std, 4),
            'Z_Score': round(z_score, 2),
            'P_Value': p_val
        })

        # Visualization for Top 3 only to avoid spamming images
        if rank < 3:
            plt.figure(figsize=(10, 6))
            sns.histplot(mc_deltas, bins=50, kde=True, color='#94a3b8', edgecolor='black', label="Random Sequence Background")

            plt.axvline(0, color='black', linestyle='-', linewidth=1.5)
            plt.axvline(noise_std, color='gray', linestyle=':')
            plt.axvline(-noise_std, color='gray', linestyle=':')

            plt.axvline(target_tagged_delta, color='#ef4444' if target_tagged_delta > 0 else '#22c55e', 
                        linestyle='-', linewidth=3, label=f"Tagged Variant (\u0394={target_tagged_delta:.4f})")

            # Add Stats Box
            stats_text = f"Z-Score: {z_score:.2f}\nP-Value: {p_val:.4f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)

            plt.title(f'Target #{rank+1}: Background Noise vs. Tagged Variant\nGene: {target_gene} | Probe: {target_row["probeID"]}', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Shift in Methylation (\u0394\u03B2)', fontsize=12)
            plt.ylabel('Count of Random Background Mutations', fontsize=12)
            plt.legend()
            plt.tight_layout()

            output_file = f'single_probe_mc_top{rank+1}_{target_gene}.png'
            plt.savefig(os.path.join(BASE_DIR, output_file), dpi=300)
            plt.close() 
        
    # Export the statistical summary CSV
    df_export = pd.DataFrame(mc_results)
    csv_out = os.path.join(BASE_DIR, 'monte_carlo_statistics.csv')
    df_export.to_csv(csv_out, index=False)
    logging.info(f"[✓] Statistical summary saved to {csv_out}")
    logging.info("[✓] Top 3 Monte Carlo figures generated and saved successfully!")

if __name__ == "__main__":
    main()