import os
import re
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoConfig, AutoModel
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TEST_CSV_PATH = 'data/datafiles/testing_data.csv'
WT_SHAPE_PATH = 'data/datafiles/wt_3d_shapes.tsv'
MUT_SHAPE_PATH = 'data/datafiles/mut_3d_shapes.tsv'
MODEL_WEIGHTS = 'checkpoints_multimodal/best_weights.pth' 
OUTPUT_DIR = 'results/multimodal/long_range_validation'

TARGET_GENES = ['MSRA', 'DDC']
SEQ_WINDOW_SIZE = 1000
SHAPE_WINDOW_SIZE = 100
NUCLEOSOME_LENGTH = 147 # Standard biological base-pair length for DNA wrapping

TABULAR_FEATURES = [
    'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
    'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Ref_H3K36me3_Signal', 
    'Ref_H3K4me1_Signal', 'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 2. ARCHITECTURE DEFINITIONS
# =============================================================================
def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local"):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        from huggingface_hub import snapshot_download
        cache_path = snapshot_download(model_path)
        for item in os.listdir(cache_path):
            src = os.path.join(cache_path, item)
            dst = os.path.join(local_dir, item)
            if os.path.isdir(src): shutil.copytree(src, dst, dirs_exist_ok=True)
            else: shutil.copy2(src, dst)
        triton_file = os.path.join(local_dir, "flash_attn_triton.py")
        if os.path.exists(triton_file):
            with open(triton_file, "w") as f: f.write("def __getattr__(name):\n    return None\n")
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, "r") as f: config_data = json.load(f)
        config_data["use_flash_attn"] = False
        if "pad_token_id" not in config_data or config_data["pad_token_id"] is None:
            config_data["pad_token_id"] = 0 
        with open(config_path, "w") as f: json.dump(config_data, f)
    
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    config.output_attentions = False 
    base_model = AutoModel.from_config(config, trust_remote_code=True)
    return config, base_model

class SaliencyFusionModel(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M", tabular_dim=9):
        super(SaliencyFusionModel, self).__init__()
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
        
        # EXTRACTING THE WEIGHTS
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
        
        m_value_pred = self.regression_head(fused_embeddings)
        return m_value_pred, attn_weights

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

# =============================================================================
# 4. MAIN SCRIPT
# =============================================================================
def main():
    logging.info("[*] Loading Data...")
    df = pd.read_csv(TEST_CSV_PATH)
    wt_shapes = pd.read_csv(WT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values
    mut_shapes = pd.read_csv(MUT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values

    logging.info("[*] Loading Saliency Fusion Model...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = SaliencyFusionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()

    # Find the target genes
    target_indices = []
    for gene in TARGET_GENES:
        match = df[df['Gene'].str.contains(gene, na=False, case=False)]
        if not match.empty:
            target_indices.append(match.index[0])

    for idx in target_indices:
        row = df.iloc[idx]
        gene_name = str(row['Gene'])
        mut_id = row['GDC_Genomic_DNA_Change']
        mut_pos, _, _ = parse_mutation_id(mut_id)
        cpg_pos = int(row['pos'])
        
        logging.info(f"\n[*] Processing Long-Range Saliency Map for: {gene_name}")
        
        wt_full = str(row['Healthy_5000bp_DNA']).upper()
        mut_full = str(row['Mutated_5000bp_DNA']).upper()
        
        wt_1000bp = wt_full[2000:3000]
        mut_1000bp = mut_full[2000:3000]

        offset = (mut_pos - 1) - cpg_pos
        mut_idx_in_1000 = 499 + offset

        tab_raw = row[TABULAR_FEATURES].values.astype(np.float32)
        tab_t = torch.tensor(tab_raw).unsqueeze(0).to(DEVICE)
        tab_m = ~torch.isnan(tab_t)
        tab_t = torch.nan_to_num(tab_t, nan=0.0)

        wt_shape_flat = wt_shapes[idx]
        wt_shape_t = torch.tensor(wt_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE).to(DEVICE)
        wt_shape_m = ~torch.isnan(wt_shape_t)
        wt_shape_t = torch.nan_to_num(wt_shape_t, nan=0.0)

        mut_shape_flat = mut_shapes[idx]
        mut_shape_t = torch.tensor(mut_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE).to(DEVICE)
        mut_shape_m = ~torch.isnan(mut_shape_t)
        mut_shape_t = torch.nan_to_num(mut_shape_t, nan=0.0)

        logging.info("    -> Running Strict Inference on WT and MUT States...")
        wt_encs = tokenizer([wt_1000bp], truncation=True, max_length=SEQ_WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
        mut_encs = tokenizer([mut_1000bp], truncation=True, max_length=SEQ_WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                wt_m_pred, wt_attn = model(tab_t, tab_m, wt_encs['input_ids'], wt_encs['attention_mask'], wt_shape_t, wt_shape_m)
                mut_m_pred, _ = model(tab_t, tab_m, mut_encs['input_ids'], mut_encs['attention_mask'], mut_shape_t, mut_shape_m)
                
        wt_beta = m_value_to_beta(wt_m_pred.item())
        mut_beta = m_value_to_beta(mut_m_pred.item())
        delta_beta = mut_beta - wt_beta

        logging.info("    -> Mapping Tokens to Physical Base Pairs...")
        tokens = tokenizer.convert_ids_to_tokens(wt_encs['input_ids'][0])
        weights = wt_attn[0].cpu().numpy()

        valid_indices = [i for i, t in enumerate(tokens) if t not in ['[PAD]', '[CLS]', '[SEP]']]
        valid_tokens = [tokens[i].replace('##', '') for i in valid_indices]
        valid_weights = weights[valid_indices]

        bp_starts = []
        token_widths = []
        current_bp = 0
        
        for t in valid_tokens:
            bp_starts.append(current_bp)
            t_len = len(t)
            token_widths.append(t_len)
            current_bp += t_len

        logging.info(f"    -> Plotting Saliency Map with Nucleosome Periodicity Intervals...")
        plt.figure(figsize=(14, 6))

        plt.bar(bp_starts, valid_weights, width=token_widths, align='edge', color='#94a3b8', alpha=0.9, edgecolor='none')

        # Highlight the exact Mutation base pair
        plt.axvline(mut_idx_in_1000, color='#ef4444', linewidth=2.5, label='Distant Mutation Locus')
        
        # Highlight the Target CpG Island (Center of Window)
        plt.axvspan(495, 505, color='#3b82f6', alpha=0.3, label='Target CpG Island')

        # Plot Nucleosome Wrapping Periodicity (147bp intervals from the mutation)
        max_intervals = 3
        for i in range(1, max_intervals + 1):
            left_interval = mut_idx_in_1000 - (NUCLEOSOME_LENGTH * i)
            right_interval = mut_idx_in_1000 + (NUCLEOSOME_LENGTH * i)
            
            if left_interval >= 0:
                plt.axvline(left_interval, color='#f97316', linewidth=1.5, linestyle=':', alpha=0.8, 
                            label='Nucleosome Structural Phasing (~147bp)' if i == 1 else "")
            if right_interval <= 1000:
                plt.axvline(right_interval, color='#f97316', linewidth=1.5, linestyle=':', alpha=0.8)

        stats_text = (
            f"Verified Gated Fusion Inference:\n"
            f"Wild-Type \u03B2: {wt_beta:.4f}\n"
            f"Mutated \u03B2: {mut_beta:.4f}\n"
            f"Actual \u0394\u03B2 Shift: {delta_beta:.4f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#ef4444')
        plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12, 
                       fontweight='bold', verticalalignment='top', bbox=props)

        plt.title(f'Neural Network Sequence Attention Profile for {gene_name}\n(Visualizing 3D Nucleosome Phasing)', fontsize=16, fontweight='bold')
        plt.xlabel('Genomic Coordinate (Base Pairs)', fontsize=12)
        plt.ylabel('Attention Pooling Weight (Importance Score)', fontsize=12)
        
        plt.xlim(0, 1000)
        
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f'{gene_name}_saliency_nucleosome_phasing.png')
        plt.savefig(plot_path, dpi=300)
        logging.info(f"[✓] Complete! Saliency Map saved to {plot_path}")

if __name__ == "__main__":
    main()
