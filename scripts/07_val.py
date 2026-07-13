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
from pyjaspar import jaspardb
from Bio.Seq import Seq
import random
import logging
import scipy.stats as stats
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

# Targets highlighted by the reviewer for long-range validation
TARGET_GENES = ['MSRA', 'DDC']

SEQ_WINDOW_SIZE = 1000
SHAPE_WINDOW_SIZE = 100
MC_SAMPLES_EXP1 = 5000  # High sample count needed for a smooth distance-decay curve

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TABULAR_FEATURES = [
    'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
    'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Ref_H3K36me3_Signal', 
    'Ref_H3K4me1_Signal', 'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
]

# =============================================================================
# 2. EXACT INFERENCE ARCHITECTURE (From 03_experiments_multimodal.py)
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
        b_tab = tab_tensor.repeat(b_len, 1).to(DEVICE)
        b_tab_mask = tab_mask.repeat(b_len, 1).to(DEVICE)
        b_shape = shape_tensor.repeat(b_len, 1, 1).to(DEVICE)
        b_shape_mask = shape_mask.repeat(b_len, 1, 1).to(DEVICE)
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            _, m_preds, _, _ = model(b_tab, b_tab_mask, encodings['input_ids'], encodings['attention_mask'], b_shape, b_shape_mask)
            all_m.extend(m_preds.cpu().flatten().tolist())
    return np.array(all_m)

# =============================================================================
# 3. EXPERIMENT PIPELINES
# =============================================================================
def main():
    logging.info("[*] Loading Multimodal Test Data & Model...")
    df = pd.read_csv(TEST_CSV_PATH)
    wt_shapes = pd.read_csv(WT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values
    mut_shapes = pd.read_csv(MUT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = GatedFusionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()

    # Find the target genes or the longest range mutations if names don't match
    target_indices = []
    for gene in TARGET_GENES:
        match = df[df['Gene'].str.contains(gene, na=False, case=False)]
        if not match.empty:
            target_indices.append(match.index[0])
    
    if not target_indices:
        logging.warning("[!] Specific genes not found. Defaulting to largest physical distance targets.")
        df['Distance'] = abs(df['pos'] - df['GDC_Genomic_DNA_Change'].apply(lambda x: parse_mutation_id(x)[0] if parse_mutation_id(x)[0] else 0))
        target_indices = df.nlargest(2, 'Distance').index.tolist()

    for idx in target_indices:
        row = df.iloc[idx]
        gene_name = str(row['Gene'])
        mut_id = row['GDC_Genomic_DNA_Change']
        mut_pos, ref_base, alt_base = parse_mutation_id(mut_id)
        cpg_pos = int(row['pos'])
        
        physical_distance = abs(mut_pos - cpg_pos)
        logging.info(f"\n=======================================================")
        logging.info(f"[*] Processing Target: {gene_name} | Distance: {physical_distance}bp")
        logging.info(f"=======================================================")
        
        wt_full = str(row['Healthy_5000bp_DNA']).upper()
        mut_full = str(row['Mutated_5000bp_DNA']).upper()
        wt_1000bp = wt_full[2000:3000]
        mut_1000bp = mut_full[2000:3000]

        offset = (mut_pos - 1) - cpg_pos
        mut_idx_in_1000 = 499 + offset

        # Epigenetics Tensor
        tab_raw = row[TABULAR_FEATURES].values.astype(np.float32)
        tab_t = torch.tensor(tab_raw).unsqueeze(0).to(DEVICE)
        tab_m = ~torch.isnan(tab_t)
        tab_t = torch.nan_to_num(tab_t, nan=0.0)

        # Shape Tensors
        wt_shape_flat = wt_shapes[idx]
        wt_shape_t = torch.tensor(wt_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE).to(DEVICE)
        wt_shape_m = ~torch.isnan(wt_shape_t)
        wt_shape_t = torch.nan_to_num(wt_shape_t, nan=0.0)

        mut_shape_flat = mut_shapes[idx]
        mut_shape_t = torch.tensor(mut_shape_flat).view(1, 14, SHAPE_WINDOW_SIZE).to(DEVICE)
        mut_shape_m = ~torch.isnan(mut_shape_t)
        mut_shape_t = torch.nan_to_num(mut_shape_t, nan=0.0)

        # Baseline inference (Verified Logic)
        wt_m_val = batch_inference(model, tokenizer, [wt_1000bp], tab_t, tab_m, wt_shape_t, wt_shape_m, batch_size=1)[0]
        wt_beta = m_value_to_beta(wt_m_val)
        
        mut_m_val = batch_inference(model, tokenizer, [mut_1000bp], tab_t, tab_m, mut_shape_t, mut_shape_m, batch_size=1)[0]
        target_delta = m_value_to_beta(mut_m_val) - wt_beta

        # ---------------------------------------------------------------------
        # EXPERIMENT 1: DISTANCE-DECAY CURVE (The "Physics" Proof)
        # ---------------------------------------------------------------------
        logging.info(f"[*] EXP 1: Computing Distance-Decay Physics ({MC_SAMPLES_EXP1} Samples)...")
        bases = ['A', 'C', 'G', 'T']
        mc_seqs, mc_distances = [], []
        
        for _ in range(MC_SAMPLES_EXP1):
            rand_idx = random.randint(0, 999)
            orig = wt_1000bp[rand_idx]
            alt = random.choice([b for b in bases if b != orig])
            mc_seq = wt_1000bp[:rand_idx] + alt + wt_1000bp[rand_idx+1:]
            
            mc_seqs.append(mc_seq)
            # Distance from center CpG (index 500)
            mc_distances.append(abs(rand_idx - 500))

        mc_m_vals = batch_inference(model, tokenizer, mc_seqs, tab_t, tab_m, wt_shape_t, wt_shape_m, batch_size=64)
        mc_abs_deltas = np.abs(m_value_to_beta(mc_m_vals) - wt_beta)

        # Bin distances
        bins = np.arange(0, 550, 50)
        max_deltas = []
        for i in range(len(bins)-1):
            mask = (np.array(mc_distances) >= bins[i]) & (np.array(mc_distances) < bins[i+1])
            if np.sum(mask) > 0:
                # 99th percentile represents maximum biological capacity
                max_deltas.append(np.percentile(mc_abs_deltas[mask], 99))
            else:
                max_deltas.append(0)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=mc_distances, y=mc_abs_deltas, color='#94a3b8', alpha=0.3, s=15, label="Random MC Mutation")
        plt.plot(bins[:-1] + 25, max_deltas, color='#f97316', linewidth=2.5, marker='o', label="99th Percentile Capacity (Decay Curve)")
        plt.scatter(physical_distance, abs(target_delta), color='#ef4444', s=250, marker='*', edgecolor='black', zorder=5, label=f"True {gene_name} Variant")
        
        plt.title(f'{gene_name}: Epigenetic Dysregulation Distance-Decay Law', fontsize=14, fontweight='bold')
        plt.xlabel('Physical Distance from Target CpG (Base Pairs)', fontsize=12)
        plt.ylabel('Absolute Predicted Methylation Shift (|\u0394\u03B2|)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'exp1_distance_decay_{gene_name}.png'), dpi=300)

        # ---------------------------------------------------------------------
        # EXPERIMENT 2: PHYLO-P EVOLUTIONARY CONSERVATION
        # ---------------------------------------------------------------------
        logging.info("[*] EXP 2: Checking PhyloP-100way Evolutionary Conservation...")
        # Get background distribution from entire test dataset
        bg_phylop = df['Target_Base_PhyloP_100way_1'].dropna().values
        target_phylop = row['Target_Base_PhyloP_100way_1']
        
        p_val_phylop = np.sum(bg_phylop >= target_phylop) / len(bg_phylop)
        
        plt.figure(figsize=(8, 5))
        sns.kdeplot(bg_phylop, fill=True, color='#94a3b8', label="Dataset Background Conservation")
        plt.axvline(target_phylop, color='#ef4444', linewidth=3, linestyle='--', label=f"{gene_name} Locus (PhyloP: {target_phylop:.2f})")
        plt.title(f'{gene_name}: Evolutionary Conservation (PhyloP-100way)', fontsize=14, fontweight='bold')
        plt.xlabel('PhyloP Conservation Score (Higher = More Conserved)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        stats_text = f"P-Value: {p_val_phylop:.4e}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'exp2_phylop_{gene_name}.png'), dpi=300)

        # ---------------------------------------------------------------------
        # EXPERIMENT 3: 3D LOOPING ANCHOR MOTIF SCAN (CTCF / RAD21 / YY1)
        # ---------------------------------------------------------------------
        logging.info("[*] EXP 3: Orthogonal Motif Scan for 3D Looping Master Regulators...")
        jdb = jaspardb(release='JASPAR2022') 
        # Fetch known architectural looping proteins
        loop_factors = ['CTCF', 'RAD21', 'YY1', 'ZNF143', 'SMC3']
        loop_motifs = [m for m in jdb.fetch_motifs(collection='CORE') if m.name.upper() in loop_factors]
        
        # Extract 41bp window precisely around the mutation
        start_idx = mut_idx_in_1000 - 20
        end_idx = mut_idx_in_1000 + 21
        if start_idx >= 0 and end_idx <= 1000:
            wt_41bp = wt_1000bp[start_idx:end_idx]
            mut_41bp = mut_1000bp[start_idx:end_idx]
            
            wt_seq_obj = Seq(wt_41bp)
            mut_seq_obj = Seq(mut_41bp)
            
            print(f"\n--- 3D Architectural Disruption Analysis ({gene_name}) ---")
            for motif in loop_motifs:
                pwm = motif.counts.normalize(pseudocounts=0.5).log_odds()
                
                wt_score = max(max(pwm.calculate(wt_seq_obj)), max(pwm.calculate(wt_seq_obj.reverse_complement())))
                mut_score = max(max(pwm.calculate(mut_seq_obj)), max(pwm.calculate(mut_seq_obj.reverse_complement())))
                
                if max(wt_score, mut_score) > 3.0: # Base threshold to report
                    print(f"[{motif.name}] WT Score: {wt_score:.2f} | MUT Score: {mut_score:.2f} | Disruption: {abs(wt_score - mut_score):.2f}")
        else:
            logging.warning("Mutation is too close to window edge to extract 41bp context.")

    logging.info("[✓] All long-range validation experiments complete!")

if __name__ == "__main__":
    main()
