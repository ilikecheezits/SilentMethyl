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
from transformers import AutoTokenizer, AutoConfig, AutoModel
import logging
import random
import scipy.stats as stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TEST_CSV_PATH = 'data/datafiles/testing_data.csv'
WT_SHAPE_PATH = 'data/datafiles/wt_3d_shapes.tsv'
MUT_SHAPE_PATH = 'data/datafiles/mut_3d_shapes.tsv'
MODEL_WEIGHTS = 'checkpoints_multimodal/best_weights.pth' 
BASE_DIR = 'results/multimodal'
SEQ_WINDOW_SIZE = 1000
SHAPE_WINDOW_SIZE = 100
MC_SAMPLES = 1000  

os.makedirs(BASE_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"[*] Using device: {DEVICE}")

TABULAR_FEATURES = [
    'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
    'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Ref_H3K36me3_Signal', 
    'Ref_H3K4me1_Signal', 'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
]

# =============================================================================
# 2. ARCHITECTURE DEFINITIONS
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
    config.output_attentions = False 
    base_model = AutoModel.from_config(config, trust_remote_code=True)
    return config, base_model

class GatedFusionModel(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M", tabular_dim=9):
        super(GatedFusionModel, self).__init__()
        
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size 
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256)
        )
        self.shape_cnn = nn.Sequential(
            nn.Conv1d(28, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveMaxPool1d(1) 
        )
        self.shape_fc = nn.Sequential(nn.Linear(128, 512), nn.LayerNorm(512), nn.GELU())

        self.norm_dna = nn.LayerNorm(768)
        self.norm_epi = nn.LayerNorm(768)
        self.gate_network = nn.Sequential(
            nn.Linear(768 * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 2), nn.Sigmoid() 
        )
        
        self.classification_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.regression_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))        

    def forward(self, tab, tab_mask, input_ids, attention_mask, shape, shape_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        hidden_states_t = hidden_states.permute(0, 2, 1)
        spatial_features = F.relu(self.spatial_conv(hidden_states_t)).permute(0, 2, 1)
        attn_weights = self.attention_pool(spatial_features).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=-1)
        dna_embeddings = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)

        tab_in = torch.cat([tab, tab_mask], dim=1)
        tab_out = self.tab_mlp(tab_in) 
        shape_in = torch.cat([shape, shape_mask], dim=1)
        shape_out = self.shape_cnn(shape_in).squeeze(-1)
        shape_out = self.shape_fc(shape_out) 
        epi_embeddings = torch.cat((tab_out, shape_out), dim=1) 
        
        dna_norm = self.norm_dna(dna_embeddings)
        epi_norm = self.norm_epi(epi_embeddings)
        concat_features = torch.cat([dna_norm, epi_norm], dim=1)
        gates = self.gate_network(concat_features)
        
        gate_dna = gates[:, 0].unsqueeze(1) 
        gate_epi = gates[:, 1].unsqueeze(1) 
        fused_embeddings = (dna_norm * gate_dna) + (epi_norm * gate_epi) 
        
        class_logits = self.classification_head(fused_embeddings)
        m_value_pred = self.regression_head(fused_embeddings)
        
        return class_logits, m_value_pred, gate_dna, gate_epi

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
def m_value_to_beta(m_val):
    m_val = np.clip(m_val, -20, 20)
    return (2 ** m_val) / (1 + (2 ** m_val))

def parse_mutation_id(mut_id):
    mut_str = str(mut_id).upper()
    if mut_str == 'NAN': return None, None, None
    match = re.search(r'(\d+)\s*([ACGT])\s*>\s*([ACGT])', mut_str)
    if match: return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None

@torch.no_grad()
def batch_inference(model, tokenizer, sequences, tab_tensor, tab_mask, shape_tensor, shape_mask, batch_size=32):
    all_m = []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        b_len = len(batch_seqs)
        
        encodings = tokenizer(batch_seqs, truncation=True, max_length=SEQ_WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
        
        # Expand target tensors to match batch size
        b_tab = tab_tensor.repeat(b_len, 1).to(DEVICE)
        b_tab_mask = tab_mask.repeat(b_len, 1).to(DEVICE)
        b_shape = shape_tensor.repeat(b_len, 1, 1).to(DEVICE)
        b_shape_mask = shape_mask.repeat(b_len, 1, 1).to(DEVICE)

        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            _, m_preds, _, _ = model(b_tab, b_tab_mask, encodings['input_ids'], encodings['attention_mask'], b_shape, b_shape_mask)
            all_m.extend(m_preds.cpu().flatten().tolist())
            
    return np.array(all_m)

# =============================================================================
# 4. MAIN INFERENCE PIPELINE
# =============================================================================
def main():
    logging.info("[*] Initializing Gated Fusion Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = GatedFusionModel().to(DEVICE)

    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True)
        logging.info("[+] Multimodal Weights loaded successfully.")
    else:
        logging.warning(f"[!] Weights file not found: {MODEL_WEIGHTS}")
        return
    model.eval()

    logging.info(f"[*] Loading Multimodal Test Data (Separate WT and Mutated Shapes)...")
    df = pd.read_csv(TEST_CSV_PATH)
    wt_shapes = pd.read_csv(WT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values
    mut_shapes = pd.read_csv(MUT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values

    logging.info("[*] Phase 1: Screening tagged mutations in CSV...")
    mask = df['probeID'].str.startswith('cg')
    df = df[mask].reset_index(drop=True)
    wt_shapes = wt_shapes[mask.values]
    mut_shapes = mut_shapes[mask.values]
    
    valid_rows, wt_seqs, mut_seqs = [], [], []
    valid_tabs, valid_tab_masks = [], []
    valid_wt_shapes, valid_wt_shape_masks = [], []
    valid_mut_shapes, valid_mut_shape_masks = [], []

    for idx, row in df.iterrows():
        mut_id = row['GDC_Genomic_DNA_Change']
        mut_pos, ref_base, alt_base = parse_mutation_id(mut_id)
        if not mut_pos: continue
        
        wt_full = str(row['Healthy_5000bp_DNA']).upper()
        mut_full = str(row['Mutated_5000bp_DNA']).upper()
        
        wt_seq = wt_full[2000:3000]
        mut_seq = mut_full[2000:3000]

        assert wt_seq[499:501] == "CG", f"Expected 'CG' at 500-501 in WT, got '{wt_seq[499:501]}'"
        if wt_seq == mut_seq: continue
        
        # Epigenetics
        tab_raw = row[TABULAR_FEATURES].values.astype(np.float32)
        tab_t = torch.tensor(tab_raw).unsqueeze(0)
        tab_m = ~torch.isnan(tab_t)
        tab_t = torch.nan_to_num(tab_t, nan=0.0)
        
        # Shapes
        wt_shape_flat = wt_shapes[idx]
        wt_shape_t = torch.tensor(wt_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE)
        wt_shape_m = ~torch.isnan(wt_shape_t)
        wt_shape_t = torch.nan_to_num(wt_shape_t, nan=0.0)

        mut_shape_flat = mut_shapes[idx]
        mut_shape_t = torch.tensor(mut_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE)
        mut_shape_m = ~torch.isnan(mut_shape_t)
        mut_shape_t = torch.nan_to_num(mut_shape_t, nan=0.0)

        valid_rows.append(row)
        wt_seqs.append(wt_seq)
        mut_seqs.append(mut_seq)
        valid_tabs.append(tab_t)
        valid_tab_masks.append(tab_m)
        valid_wt_shapes.append(wt_shape_t)
        valid_wt_shape_masks.append(wt_shape_m)
        valid_mut_shapes.append(mut_shape_t)
        valid_mut_shape_masks.append(mut_shape_m)

    logging.info(f"    -> Extracted {len(valid_rows)} fully valid target pairs.")

    # =========================================================================
    # PHASE 1.5: SCORE EVERYTHING TO FIND THE TOP 10 DRIVERS (MEMORY SAFE)
    # =========================================================================
    logging.info(f"[*] Phase 1.5: Scoring ALL {len(valid_rows)} targets to find the True Top Drivers...")
    all_wt_m = []
    all_mut_m = []
    batch_size = 16  # Lowered to 16 for VRAM safety
    
    with torch.no_grad(): # <-- Prevents the memory leak!
        for i in tqdm(range(0, len(valid_rows), batch_size), desc="Scoring Targets"):
            b_wt_seqs = wt_seqs[i:i+batch_size]
            b_mut_seqs = mut_seqs[i:i+batch_size]
            
            # Dynamically build batches for multimodal inputs
            b_tab = torch.cat(valid_tabs[i:i+batch_size], dim=0).to(DEVICE)
            b_tab_m = torch.cat(valid_tab_masks[i:i+batch_size], dim=0).to(DEVICE)
            b_wt_shape = torch.cat(valid_wt_shapes[i:i+batch_size], dim=0).to(DEVICE)
            b_wt_shape_m = torch.cat(valid_wt_shape_masks[i:i+batch_size], dim=0).to(DEVICE)
            b_mut_shape = torch.cat(valid_mut_shapes[i:i+batch_size], dim=0).to(DEVICE)
            b_mut_shape_m = torch.cat(valid_mut_shape_masks[i:i+batch_size], dim=0).to(DEVICE)
            
            wt_encs = tokenizer(b_wt_seqs, truncation=True, max_length=SEQ_WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
            mut_encs = tokenizer(b_mut_seqs, truncation=True, max_length=SEQ_WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                _, wt_preds, _, _ = model(b_tab, b_tab_m, wt_encs['input_ids'], wt_encs['attention_mask'], b_wt_shape, b_wt_shape_m)
                _, mut_preds, _, _ = model(b_tab, b_tab_m, mut_encs['input_ids'], mut_encs['attention_mask'], b_mut_shape, b_mut_shape_m)
                
            all_wt_m.extend(wt_preds.cpu().flatten().tolist())
            all_mut_m.extend(mut_preds.cpu().flatten().tolist())
            
    torch.cuda.empty_cache() # Clear cache just to be safe
        
    # Convert M-values to Betas to find the true biological Delta
    all_wt_betas = m_value_to_beta(np.array(all_wt_m))
    all_mut_betas = m_value_to_beta(np.array(all_mut_m))
    deltas = all_mut_betas - all_wt_betas
    
    # SORT by the absolute strongest impact
    top_k = min(10, len(deltas))
    top_indices = np.argsort(np.abs(deltas))[-top_k:][::-1]

    # =========================================================================
    # PHASE 2: RUN MONTE CARLO ON THE TRUE TOP DRIVERS
    # =========================================================================
    logging.info(f"[*] Phase 2: Generating {MC_SAMPLES} background perturbations for Top {top_k} Targets...")
    bases = ['A', 'C', 'G', 'T']
    mc_results = []
    
    for rank, best_idx in enumerate(top_indices):
        target_row = valid_rows[best_idx]
        target_wt_seq = wt_seqs[best_idx]
        
        t_tab = valid_tabs[best_idx]
        t_tab_m = valid_tab_masks[best_idx]
        t_wt_shape = valid_wt_shapes[best_idx]
        t_wt_shape_m = valid_wt_shape_masks[best_idx]
        
        wt_beta = all_wt_betas[best_idx]
        target_tagged_delta = deltas[best_idx]
        target_gene = str(target_row['Gene']) if pd.notna(target_row['Gene']) else f"Intergenic_{target_row['chr']}"
        
        logging.info(f"    -> Processing Target #{rank+1}: {target_gene} ({target_row['GDC_Genomic_DNA_Change']}) | Tagged Delta: {target_tagged_delta:.4f}")
        
        mc_mutated_seqs = []
        for _ in range(MC_SAMPLES):
            rand_idx = random.choice([j for j in range(0, SEQ_WINDOW_SIZE)])
            orig_base = target_wt_seq[rand_idx]
            alt_base = random.choice([b for b in bases if b != orig_base])
            mc_seq = target_wt_seq[:rand_idx] + alt_base + target_wt_seq[rand_idx+1:SEQ_WINDOW_SIZE]
            mc_mutated_seqs.append(mc_seq)

        # Batch process MC against the WT structural/epigenetic backbone
        mc_mut_m_vals = batch_inference(model, tokenizer, mc_mutated_seqs, t_tab, t_tab_m, t_wt_shape, t_wt_shape_m, batch_size=32)
        mc_deltas = m_value_to_beta(mc_mut_m_vals) - wt_beta
        
        noise_mean = np.mean(mc_deltas)
        noise_std = np.std(mc_deltas)
        z_score = (target_tagged_delta - noise_mean) / (noise_std + 1e-9)
        p_val = np.sum(np.abs(mc_deltas) >= np.abs(target_tagged_delta)) / MC_SAMPLES
        
        mc_results.append({
            'Rank': rank + 1,
            'Gene': target_gene,
            'GDC_Genomic_DNA_Change': target_row['GDC_Genomic_DNA_Change'],
            'Target_CpG': target_row['probeID'],
            'Predicted_Delta_Beta': round(target_tagged_delta, 4),
            'MC_Background_Mean': round(noise_mean, 4),
            'MC_Background_STD': round(noise_std, 4),
            'Z_Score': round(z_score, 2),
            'P_Value': p_val
        })

        if rank < 3:
            # --- Histogram ---
            plt.figure(figsize=(10, 6))
            sns.histplot(mc_deltas, bins=50, kde=True, color='#94a3b8', edgecolor='black', label="Random Sequence Background")
            plt.axvline(0, color='black', linestyle='-', linewidth=1.5)
            plt.axvline(noise_std, color='gray', linestyle=':')
            plt.axvline(-noise_std, color='gray', linestyle=':')
            plt.axvline(target_tagged_delta, color='#ef4444' if target_tagged_delta > 0 else '#22c55e', 
                        linestyle='-', linewidth=3, label=f"Tagged Variant (\u0394={target_tagged_delta:.4f})")

            stats_text = f"Z-Score: {z_score:.2f}\nP-Value: {p_val:.4f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)

            plt.title(f'Multimodal Target #{rank+1}: Background Noise vs. Tagged Variant\nGene: {target_gene} | Probe: {target_row["probeID"]}', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Shift in Methylation (\u0394\u03B2)', fontsize=12)
            plt.ylabel('Count of Random Background Mutations', fontsize=12)
            plt.legend()
            plt.tight_layout()

            output_file = f'fusion_single_probe_mc_top{rank+1}_{target_gene}.png'
            plt.savefig(os.path.join(BASE_DIR, output_file), dpi=300)
            plt.close() 

            # --- Q-Q Plot ---
            stat, p_value_normality = stats.shapiro(mc_deltas)
            logging.info(f"        -> Normality (Shapiro-Wilk) for {target_gene}: Stat={stat:.4f}, p-value={p_value_normality:.4e}")
            
            plt.figure(figsize=(6, 6))
            stats.probplot(mc_deltas, dist="norm", plot=plt)
            plt.title(f'Gated Fusion Q-Q Plot: MC Background for {target_gene}')
            plt.xlabel('Theoretical Normal Quantiles')
            plt.ylabel('Empirical Monte Carlo Quantiles (\u0394\u03B2)')
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, f'fusion_qq_plot_top{rank+1}_{target_gene}.png'), dpi=300)
            plt.close()
        
    df_export = pd.DataFrame(mc_results)
    csv_out = os.path.join(BASE_DIR, 'fusion_monte_carlo_statistics.csv')
    df_export.to_csv(csv_out, index=False)
    logging.info(f"[✓] Statistical summary saved to {csv_out}")

if __name__ == "__main__":
    main()
