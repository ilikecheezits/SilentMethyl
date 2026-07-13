import os
import random
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
from pyjaspar import jaspardb
from Bio.Seq import Seq
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
OUTPUT_DIR = 'results/multimodal/chd5_case_study'

TARGET_GENE = 'CHD5'
TARGET_MUT_ID = 'chr1:g.6128067C>T'

SEQ_WINDOW_SIZE = 1000
SHAPE_WINDOW_SIZE = 100
MC_SAMPLES = 500
SCORE_THRESHOLD = 5.0 # Conservative filter for real binding sites

TABULAR_FEATURES = [
    'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
    'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Ref_H3K36me3_Signal', 
    'Ref_H3K4me1_Signal', 'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"[*] Using device: {DEVICE}")

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

@torch.no_grad()
def batch_inference(model, tokenizer, sequences, tab_tensor, tab_mask, shape_tensor, shape_mask, batch_size=32):
    all_m = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        b_len = len(batch_seqs)
        
        encodings = tokenizer(batch_seqs, truncation=True, max_length=SEQ_WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
        b_tab = tab_tensor.repeat(b_len, 1).to(DEVICE)
        b_tab_mask = tab_mask.repeat(b_len, 1).to(DEVICE)
        b_shape = shape_tensor.repeat(b_len, 1, 1).to(DEVICE)
        b_shape_mask = shape_mask.repeat(b_len, 1, 1).to(DEVICE)

        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            _, m_preds, _, _ = model(b_tab, b_tab_mask, encodings['input_ids'], encodings['attention_mask'], b_shape, b_shape_mask)
            all_m.extend(m_preds.cpu().flatten().tolist())
            
    return np.array(all_m)

# =============================================================================
# 4. JASPAR SETUP (GLOBAL SCAN)
# =============================================================================
logging.info("[*] Initializing JASPAR and fetching ALL Vertebrate Motifs...")
jdb = jaspardb(release='JASPAR2022') 
all_motifs = jdb.fetch_motifs(collection='CORE')

motifs = []
for m in all_motifs:
    if hasattr(m, 'tax_group') and m.tax_group:
        if 'vertebrate' in str(m.tax_group).lower():
            motifs.append(m)

if not motifs:
    motifs = all_motifs

logging.info(f"[*] Precomputing PSSMs for {len(motifs)} Transcription Factors...")
pssm_dict = {}
for motif in motifs:
    pwm = motif.counts.normalize(pseudocounts=0.5)
    pssm_dict[motif.name] = pwm.log_odds()

def get_max_disruption(wt_seq_41bp, mut_seq_41bp):
    """Scans all vertebrate TFs and returns the delta score of the most disrupted motif."""
    wt_seq = Seq(wt_seq_41bp)
    mut_seq = Seq(mut_seq_41bp)
    
    wt_seq_rc = wt_seq.reverse_complement()
    mut_seq_rc = mut_seq.reverse_complement()
    
    max_disruption = 0
    best_delta = 0.0
    
    for tf_name, pssm in pssm_dict.items():
        try:
            wt_fwd = max(pssm.calculate(wt_seq))
            wt_rev = max(pssm.calculate(wt_seq_rc))
            best_wt = max(wt_fwd, wt_rev)
            
            mut_fwd = max(pssm.calculate(mut_seq))
            mut_rev = max(pssm.calculate(mut_seq_rc))
            best_mut = max(mut_fwd, mut_rev)
            
            # Only consider it a disruption if it was a real binding site to begin with
            if max(best_wt, best_mut) > SCORE_THRESHOLD:
                disruption = abs(best_wt - best_mut)
                if disruption > max_disruption:
                    max_disruption = disruption
                    best_delta = best_mut - best_wt
        except Exception:
            continue
            
    return best_delta

# =============================================================================
# 5. MAIN SCRIPT
# =============================================================================
def main():
    logging.info("[*] Loading Data...")
    df = pd.read_csv(TEST_CSV_PATH)
    wt_shapes = pd.read_csv(WT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values
    mut_shapes = pd.read_csv(MUT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values

    try:
        chd5_idx = df.index[df['GDC_Genomic_DNA_Change'] == TARGET_MUT_ID].tolist()[0]
        chd5_row = df.iloc[chd5_idx]
    except IndexError:
        raise ValueError(f"[!] Target mutation {TARGET_MUT_ID} not found in {TEST_CSV_PATH}")

    cpg_pos = int(chd5_row['pos'])
    mut_pos = 6128067 
    offset = mut_pos - cpg_pos
    mut_idx_in_5000 = 2500 + offset
    mut_idx_in_1000 = 500 + offset

    wt_full_seq = str(chd5_row['Healthy_5000bp_DNA']).upper()
    mut_full_seq = str(chd5_row['Mutated_5000bp_DNA']).upper()
    
    wt_1000bp = wt_full_seq[2000:3000]
    mut_1000bp = mut_full_seq[2000:3000] 

    assert wt_1000bp[mut_idx_in_1000] == 'C', "WT center base is incorrect!"

    tab_raw = chd5_row[TABULAR_FEATURES].values.astype(np.float32)
    tab_t = torch.tensor(tab_raw).unsqueeze(0).to(DEVICE)
    tab_m = ~torch.isnan(tab_t)
    tab_t = torch.nan_to_num(tab_t, nan=0.0)

    wt_shape_flat = wt_shapes[chd5_idx]
    wt_shape_t = torch.tensor(wt_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE).to(DEVICE)
    wt_shape_m = ~torch.isnan(wt_shape_t)
    wt_shape_t = torch.nan_to_num(wt_shape_t, nan=0.0)

    mut_shape_flat = mut_shapes[chd5_idx]
    mut_shape_t = torch.tensor(mut_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE).to(DEVICE)
    mut_shape_m = ~torch.isnan(mut_shape_t)
    mut_shape_t = torch.nan_to_num(mut_shape_t, nan=0.0)

    logging.info("[*] Loading Gated Fusion Model...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = GatedFusionModel().to(DEVICE)
    
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True)
    else:
        raise FileNotFoundError(f"[!] Model weights not found at {MODEL_WEIGHTS}")
    model.eval()

    logging.info(f"[*] Generating {MC_SAMPLES} Monte Carlo permutations...")
    mc_1000bp_seqs = []
    mc_41bp_seqs = []
    bases = ['A', 'C', 'G', 'T']

    wt_41bp = wt_full_seq[mut_idx_in_5000 - 20 : mut_idx_in_5000 + 21]

    wt_m_val = batch_inference(model, tokenizer, [wt_1000bp], tab_t, tab_m, wt_shape_t, wt_shape_m, batch_size=1)[0]
    wt_beta = m_value_to_beta(wt_m_val)

    for _ in range(MC_SAMPLES):
        rand_offset = random.randint(-20, 20)
        rand_idx_in_1000 = mut_idx_in_1000 + rand_offset
        orig_base = wt_1000bp[rand_idx_in_1000]
        alt_base = random.choice([b for b in bases if b != orig_base])
        
        mc_1000bp = wt_1000bp[:rand_idx_in_1000] + alt_base + wt_1000bp[rand_idx_in_1000+1:]
        mc_1000bp_seqs.append(mc_1000bp)
        
        rand_idx_in_41 = 20 + rand_offset
        mc_41bp = wt_41bp[:rand_idx_in_41] + alt_base + wt_41bp[rand_idx_in_41+1:]
        mc_41bp_seqs.append(mc_41bp)

    logging.info(f"[*] Globally scoring Monte Carlo Sequences against all {len(motifs)} Vertebrate Motifs...")
    mc_motif_deltas = []
    for mc_seq in tqdm(mc_41bp_seqs, desc="Scanning JASPAR"):
        delta = get_max_disruption(wt_41bp, mc_seq)
        mc_motif_deltas.append(delta)
    mc_motif_deltas = np.array(mc_motif_deltas)

    logging.info("[*] Scoring Monte Carlo Sequences through Gated Fusion Model...")
    mc_m_vals = batch_inference(model, tokenizer, mc_1000bp_seqs, tab_t, tab_m, wt_shape_t, wt_shape_m, batch_size=32)
    mc_betas = m_value_to_beta(mc_m_vals)
    mc_beta_deltas = np.array(mc_betas - wt_beta)

    mut_41bp = mut_full_seq[mut_idx_in_5000 - 20 : mut_idx_in_5000 + 21]
    chd5_motif_delta = get_max_disruption(wt_41bp, mut_41bp)

    chd5_m_val = batch_inference(model, tokenizer, [mut_1000bp], tab_t, tab_m, mut_shape_t, mut_shape_m, batch_size=1)[0]
    chd5_beta_delta = m_value_to_beta(chd5_m_val) - wt_beta

    # =========================================================================
    # 6. STATISTICAL THRESHOLDING & PLOTTING
    # =========================================================================
    logging.info("[*] Calculating statistical thresholds and plotting quadrants...")
    
    noise_mean = np.mean(mc_beta_deltas)
    noise_std = np.std(mc_beta_deltas)
    upper_bound = noise_mean + (2 * noise_std)
    lower_bound = noise_mean - (2 * noise_std)
    
    sig_mask = (mc_beta_deltas > upper_bound) | (mc_beta_deltas < lower_bound)
    
    avg_jaspar_sig = np.mean(np.abs(mc_motif_deltas[sig_mask])) if np.sum(sig_mask) > 0 else 0
    avg_jaspar_insig = np.mean(np.abs(mc_motif_deltas[~sig_mask]))
    
    plt.figure(figsize=(10, 7))
    
    sns.scatterplot(x=mc_motif_deltas[~sig_mask], y=mc_beta_deltas[~sig_mask], 
                    color='#94a3b8', alpha=0.5, s=50, edgecolor='none', label='Background Noise (< 2\u03C3)')
    
    sns.scatterplot(x=mc_motif_deltas[sig_mask], y=mc_beta_deltas[sig_mask], 
                    color='#f97316', alpha=0.9, s=70, edgecolor='black', linewidth=0.5, label='Significant Shift (> 2\u03C3)')

    plt.scatter(x=chd5_motif_delta, y=chd5_beta_delta, color='#ef4444', s=350, marker='*', 
                edgecolor='black', linewidth=1.5, zorder=5, label='True CHD5 Mutation (C>T)')

    z = np.polyfit(mc_motif_deltas, mc_beta_deltas, 1)
    p = np.poly1d(z)
    plt.plot(mc_motif_deltas, p(mc_motif_deltas), "k--", alpha=0.7, label=f'Global Trend (Slope: {z[0]:.4f})')

    plt.axvline(0, color='black', linewidth=1, alpha=0.3)
    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.axhline(upper_bound, color='#f97316', linewidth=1.5, linestyle=':', alpha=0.6)
    plt.axhline(lower_bound, color='#f97316', linewidth=1.5, linestyle=':', alpha=0.6)

    stats_text = (
        f"Statistical Proof of Motif Dependency:\n"
        f"Avg Max Disruption (Noise): {avg_jaspar_insig:.2f} \u0394 Log-Odds\n"
        f"Avg Max Disruption (Sig Shifts): {avg_jaspar_sig:.2f} \u0394 Log-Odds"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                   verticalalignment='bottom', bbox=props)

    plt.title(f'CHD5 Biological Validation: Network \u0394\u03B2 vs. Max JASPAR Disruption\n(Scanning All Vertebrate Motifs)', fontsize=14, fontweight='bold')
    plt.xlabel(f'Max JASPAR Motif Score Disruption (\u0394 Log-Odds)', fontsize=12)
    plt.ylabel('Model Predicted Methylation Shift (\u0394\u03B2)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'chd5_jaspar_mc_correlation.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"[✓] Complete! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
