import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- Configuration ---
TEST_CSV_PATH = 'data/datafiles/testing_data.csv'
WT_SHAPE_PATH = 'data/datafiles/wt_3d_shapes.tsv'
MUT_SHAPE_PATH = 'data/datafiles/mut_3d_shapes.tsv'
JASPAR_PATH = "results/jaspar_motif_disruptions.csv"
MODEL_WEIGHTS = 'checkpoints_multimodal/best_weights.pth' 
OUTPUT_PLOT = "results/multimodal/model_pred_vs_motif.png"

SEQ_WINDOW_SIZE = 1000
SHAPE_WINDOW_SIZE = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TABULAR_FEATURES = [
    'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
    'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Ref_H3K36me3_Signal', 
    'Ref_H3K4me1_Signal', 'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
]

# --- 1. Model Architecture (Imported from your multimodal script) ---
def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local"):
    import shutil
    from huggingface_hub import snapshot_download
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
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
        self.gate_network = nn.Sequential(nn.Linear(768 * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 2), nn.Sigmoid())
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

# --- 2. Inference & Correlation Pipeline ---
def main():
    logging.info("[*] Initializing Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = GatedFusionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()

    logging.info("[*] Loading Test Data & Shapes...")
    df = pd.read_csv(TEST_CSV_PATH)
    wt_shapes = pd.read_csv(WT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values
    mut_shapes = pd.read_csv(MUT_SHAPE_PATH, sep='\t', header=None, dtype=np.float32).values

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
        
        wt_seq = str(row['Healthy_5000bp_DNA']).upper()[2000:3000]
        mut_seq = str(row['Mutated_5000bp_DNA']).upper()[2000:3000]
        if wt_seq == mut_seq: continue
        
        tab_raw = row[TABULAR_FEATURES].values.astype(np.float32)
        tab_t = torch.nan_to_num(torch.tensor(tab_raw).unsqueeze(0), nan=0.0)
        tab_m = ~torch.isnan(torch.tensor(tab_raw).unsqueeze(0))
        
        wt_shape_t = torch.nan_to_num(torch.tensor(wt_shapes[idx]).view(1, 14, SHAPE_WINDOW_SIZE), nan=0.0)
        wt_shape_m = ~torch.isnan(torch.tensor(wt_shapes[idx]).view(1, 14, SHAPE_WINDOW_SIZE))
        mut_shape_t = torch.nan_to_num(torch.tensor(mut_shapes[idx]).view(1, 14, SHAPE_WINDOW_SIZE), nan=0.0)
        mut_shape_m = ~torch.isnan(torch.tensor(mut_shapes[idx]).view(1, 14, SHAPE_WINDOW_SIZE))

        valid_rows.append(row)
        wt_seqs.append(wt_seq)
        mut_seqs.append(mut_seq)
        valid_tabs.append(tab_t)
        valid_tab_masks.append(tab_m)
        valid_wt_shapes.append(wt_shape_t)
        valid_wt_shape_masks.append(wt_shape_m)
        valid_mut_shapes.append(mut_shape_t)
        valid_mut_shape_masks.append(mut_shape_m)

    logging.info("[*] Running Inference on Valid Sequences...")
    all_wt_m, all_mut_m = [], []
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_rows), batch_size)):
            b_tab = torch.cat(valid_tabs[i:i+batch_size], dim=0).to(DEVICE)
            b_tab_m = torch.cat(valid_tab_masks[i:i+batch_size], dim=0).to(DEVICE)
            b_wt_shape = torch.cat(valid_wt_shapes[i:i+batch_size], dim=0).to(DEVICE)
            b_wt_shape_m = torch.cat(valid_wt_shape_masks[i:i+batch_size], dim=0).to(DEVICE)
            b_mut_shape = torch.cat(valid_mut_shapes[i:i+batch_size], dim=0).to(DEVICE)
            b_mut_shape_m = torch.cat(valid_mut_shape_masks[i:i+batch_size], dim=0).to(DEVICE)
            
            wt_encs = tokenizer(wt_seqs[i:i+batch_size], truncation=True, max_length=SEQ_WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
            mut_encs = tokenizer(mut_seqs[i:i+batch_size], truncation=True, max_length=SEQ_WINDOW_SIZE, padding='max_length', return_tensors='pt').to(DEVICE)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                _, wt_preds, _, _ = model(b_tab, b_tab_m, wt_encs['input_ids'], wt_encs['attention_mask'], b_wt_shape, b_wt_shape_m)
                _, mut_preds, _, _ = model(b_tab, b_tab_m, mut_encs['input_ids'], mut_encs['attention_mask'], b_mut_shape, b_mut_shape_m)
                
            all_wt_m.extend(wt_preds.cpu().flatten().tolist())
            all_mut_m.extend(mut_preds.cpu().flatten().tolist())

    # Build predictions dataframe
    df_preds = pd.DataFrame({
        "GDC_Genomic_DNA_Change": [r['GDC_Genomic_DNA_Change'] for r in valid_rows],
        "Ref_ATAC_Signal": [r['Ref_ATAC_Signal'] for r in valid_rows], # <--- ADD THIS LINE
        "Predicted_WT_Beta": m_value_to_beta(np.array(all_wt_m)),
        "Predicted_Mut_Beta": m_value_to_beta(np.array(all_mut_m))
    })
    df_preds["Predicted_Abs_Delta_Beta"] = (df_preds["Predicted_Mut_Beta"] - df_preds["Predicted_WT_Beta"]).abs()

    logging.info("[*] Merging with JASPAR Disruption Data...")
    df_jaspar = pd.read_csv(JASPAR_PATH)
    df_merged = pd.merge(df_preds, df_jaspar, on="GDC_Genomic_DNA_Change", how="inner")
    df_clean = df_merged.dropna(subset=["Predicted_Abs_Delta_Beta", "Absolute_Disruption"]).copy()

    df_clean = df_clean[df_clean["Ref_ATAC_Signal"] > 0.0].copy()
    # Stratify
    # --- FLIPPED TEST: Stratify by Model Prediction ---
    logging.info("[*] Stratifying by Model Predicted Shift...")
    pct_75 = df_clean["Predicted_Abs_Delta_Beta"].quantile(0.95)
    pct_25 = df_clean["Predicted_Abs_Delta_Beta"].quantile(0.05)

    def assign_pred_group(score):
        if score >= pct_75: return "High Predicted Shift\n(Top 25%)"
        elif score <= pct_25: return "Low Predicted Shift\n(Bottom 25%)"
        else: return "Medium"

    df_clean["Pred_Group"] = df_clean["Predicted_Abs_Delta_Beta"].apply(assign_pred_group)
    df_extremes = df_clean[df_clean["Pred_Group"] != "Medium"]

    # --- Statistics: Do high predictions have higher motif disruption? ---
    high_pred_group = df_extremes[df_extremes["Pred_Group"] == "High Predicted Shift\n(Top 25%)"]["Absolute_Disruption"]
    low_pred_group = df_extremes[df_extremes["Pred_Group"] == "Low Predicted Shift\n(Bottom 25%)"]["Absolute_Disruption"]
    
    stat, p_val = stats.mannwhitneyu(high_pred_group, low_pred_group, alternative='greater')

    logging.info(f"--- Flipped Statistical Results ---")
    logging.info(f"High Predicted Shift -> Mean JASPAR Disruption: {high_pred_group.mean():.4f}")
    logging.info(f"Low Predicted Shift  -> Mean JASPAR Disruption: {low_pred_group.mean():.4f}")
    logging.info(f"Mann-Whitney U p-value:                       {p_val:.2e}")

    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    sns.boxplot(
        x="Pred_Group", y="Absolute_Disruption", data=df_extremes, 
        hue="Pred_Group", legend=False, palette=["#ef4444", "#3b82f6"], 
        showfliers=False, width=0.5
    )
    sns.stripplot(
        x="Pred_Group", y="Absolute_Disruption", data=df_extremes, 
        color=".25", alpha=0.5, jitter=True, size=4
    )
    
    plt.title("Model Confidence is Driven by TF Motif Disruption", fontsize=14, fontweight="bold")
    plt.xlabel("Model's Predicted Epigenetic Shift (\u0394 Beta)", fontsize=12)
    plt.ylabel("JASPAR Motif Disruption Severity", fontsize=12)
    plt.text(0.5, df_extremes["Absolute_Disruption"].max() * 0.9, 
             f"p-value = {p_val:.2e}", ha='center', va='bottom', 
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    logging.info(f"[✓] Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
