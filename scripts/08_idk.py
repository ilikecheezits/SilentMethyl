import os
import re
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
import scipy.signal as signal
import logging
import random
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =============================================================================
# 1. CONFIGURATION & REPRODUCIBILITY
# =============================================================================
SEED = 42

def set_seed(seed):
    """Locks down all random number generators for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

TEST_CSV_PATH = 'data/datafiles/testing_data.csv'
WT_SHAPE_PATH = 'data/datafiles/wt_3d_shapes.tsv'
MODEL_WEIGHTS = 'checkpoints_multimodal/best_weights.pth' 

TARGET_GENES = ['MSRA', 'DDC']
SEQ_WINDOW_SIZE = 1000
SHAPE_WINDOW_SIZE = 100
MC_SAMPLES = 5000  # High sample count needed for rigorous statistics
BIN_SIZE = 10      # 10bp bins for smooth mathematical modeling

TABULAR_FEATURES = [
    'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
    'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Ref_H3K36me3_Signal', 
    'Ref_H3K4me1_Signal', 'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
]

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
def batch_inference(model, tokenizer, sequences, tab_tensor, tab_mask, shape_tensor, shape_mask, batch_size=64):
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

def calculate_periodicity(distances, shifts, bin_size=10):
    # Bin the data to find the biological envelope (99th percentile capacity)
    bins = np.arange(0, 500 + bin_size, bin_size)
    envelope = []
    
    for i in range(len(bins)-1):
        mask = (np.array(distances) >= bins[i]) & (np.array(distances) < bins[i+1])
        if np.sum(mask) > 0:
            envelope.append(np.percentile(shifts[mask], 99))
        else:
            envelope.append(0)
            
    envelope = np.array(envelope)
    
    # Detrend the signal (remove the general distance decay slope) to isolate the wave
    detrended_envelope = signal.detrend(envelope)
    
    # Fourier Transform: Calculate the Power Spectral Density
    frequencies, psd = signal.periodogram(detrended_envelope, fs=1.0/bin_size)
    
    # Ignore DC component
    valid_idx = frequencies > 0
    frequencies = frequencies[valid_idx]
    psd = psd[valid_idx]
    
    # Find the dominant repeating frequency
    max_idx = np.argmax(psd)
    dominant_period = 1.0 / frequencies[max_idx]
    max_power = psd[max_idx]
    
    return dominant_period, max_power

# =============================================================================
# 4. MAIN EXPERIMENT
# =============================================================================
def main():
    logging.info("[*] Loading Data & Model for Mathematical Periodicity Proof...")
    df = pd.read_csv(TEST_CSV_PATH)
    wt_shapes = pd.read_csv(WT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = GatedFusionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()

    target_indices = []
    for gene in TARGET_GENES:
        match = df[df['Gene'].str.contains(gene, na=False, case=False)]
        if not match.empty:
            target_indices.append(match.index[0])

    for idx in target_indices:
        row = df.iloc[idx]
        gene_name = str(row['Gene'])
        logging.info(f"\n=======================================================")
        logging.info(f"[*] Analyzing 3D Spatial Periodicity for: {gene_name}")
        logging.info(f"=======================================================")
        
        wt_full = str(row['Healthy_5000bp_DNA']).upper()
        wt_1000bp = wt_full[2000:3000]

        tab_raw = row[TABULAR_FEATURES].values.astype(np.float32)
        tab_t = torch.tensor(tab_raw).unsqueeze(0).to(DEVICE)
        tab_m = ~torch.isnan(tab_t)
        tab_t = torch.nan_to_num(tab_t, nan=0.0)

        wt_shape_flat = wt_shapes[idx]
        wt_shape_t = torch.tensor(wt_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE).to(DEVICE)
        wt_shape_m = ~torch.isnan(wt_shape_t)
        wt_shape_t = torch.nan_to_num(wt_shape_t, nan=0.0)

        # Baseline beta
        wt_m_val = batch_inference(model, tokenizer, [wt_1000bp], tab_t, tab_m, wt_shape_t, wt_shape_m, batch_size=1)[0]
        wt_beta = m_value_to_beta(wt_m_val)

        logging.info(f"    -> Running {MC_SAMPLES} Monte Carlo perturbations across sequence...")
        bases = ['A', 'C', 'G', 'T']
        mc_seqs, mc_distances = [], []
        
        for _ in range(MC_SAMPLES):
            rand_idx = random.randint(0, 999)
            orig = wt_1000bp[rand_idx]
            alt = random.choice([b for b in bases if b != orig])
            mc_seq = wt_1000bp[:rand_idx] + alt + wt_1000bp[rand_idx+1:]
            
            mc_seqs.append(mc_seq)
            mc_distances.append(abs(rand_idx - 500))  # Distance from center CpG

        mc_m_vals = batch_inference(model, tokenizer, mc_seqs, tab_t, tab_m, wt_shape_t, wt_shape_m, batch_size=64)
        mc_abs_deltas = np.abs(m_value_to_beta(mc_m_vals) - wt_beta)

        logging.info("    -> Computing Fourier Power Spectral Density (PSD)...")
        # 1. Calculate True Periodicity
        true_period, true_power = calculate_periodicity(mc_distances, mc_abs_deltas, bin_size=BIN_SIZE)
        
        logging.info(f"    [+] DOMINANT STRUCTURAL PERIOD DETECTED: {true_period:.1f} bp")

        # 2. Rigorous Permutation Test
        logging.info("    -> Running Statistical Permutation Test (1,000 Shuffles)...")
        PERMUTATIONS = 1000
        null_powers = []
        
        for _ in range(PERMUTATIONS):
            shuffled_distances = np.random.permutation(mc_distances)
            _, null_power = calculate_periodicity(shuffled_distances, mc_abs_deltas, bin_size=BIN_SIZE)
            null_powers.append(null_power)
            
        null_powers = np.array(null_powers)
        p_val = np.sum(null_powers >= true_power) / PERMUTATIONS

        logging.info(f"    [+] STATISTICAL SIGNIFICANCE (P-Value): {p_val:.4f}")
        
        if 130 <= true_period <= 160 and p_val < 0.05:
            logging.info("    [VERIFIED] The network's predictive capacity mathematically oscillates at the physical nucleosome wrapping frequency (~147bp)!")
        else:
            logging.warning("    [FAILED] No significant nucleosome periodicity found.")

if __name__ == "__main__":
    main()
