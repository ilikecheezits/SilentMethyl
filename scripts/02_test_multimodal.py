import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import logging
import os
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import snapshot_download, hf_hub_download
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve
from tqdm import tqdm

# =========================================
# 1. Multi-Modal Dataset (Synced)
# =========================================
class MultiModalLateFusionDataset(Dataset):
    def __init__(self, df, shape_data_array, tokenizer, seq_window_size=1000, shape_window_size=100):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_window_size = seq_window_size
        self.shape_window_size = shape_window_size
        self.shape_data = shape_data_array

        assert len(self.df) == len(self.shape_data), "CRITICAL ERROR: CSV and TSV row counts do not match!"
        
        self.tabular_features = [
            'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
            'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 
            'Ref_H3K36me3_Signal', 'Ref_H3K4me1_Signal',
            'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
        ]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tabular Data
        tab_raw = row[self.tabular_features].values.astype(np.float32)
        tab_tensor = torch.tensor(tab_raw)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)
        
        # Sequence Data
        full_sequence = str(row['Healthy_5000bp_DNA']).upper()
        true_c_idx = len(full_sequence) // 2
        start_idx = true_c_idx - (self.seq_window_size // 2)
        end_idx = start_idx + self.seq_window_size
        
        if start_idx < 0: sequence = full_sequence[:self.seq_window_size]
        elif end_idx > len(full_sequence): sequence = full_sequence[-self.seq_window_size:]
        else: sequence = full_sequence[start_idx : end_idx]
            
        encoding = self.tokenizer(sequence, truncation=True, max_length=self.seq_window_size, padding='max_length', return_tensors='pt')
        
        # Shape Data
        shape_flat = self.shape_data[idx]
        shape_tensor = torch.tensor(shape_flat).view(14, self.shape_window_size)
        shape_mask = ~torch.isnan(shape_tensor)
        shape_tensor = torch.nan_to_num(shape_tensor, nan=0.0)

        # Targets
        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'tab': tab_tensor, 'tab_mask': tab_mask.float(),
            'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
            'shape': shape_tensor, 'shape_mask': shape_mask.float(),
            'm_value': m_value, 'beta_value': torch.tensor(beta, dtype=torch.float32),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32)
        }

# =========================================
# 2. Triton Neutralizer & Gated Architecture
# =========================================
def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local_inference"):
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
        
        # MODEL X: DNA
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size 
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        # MODEL Y: Epi & Shape
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256)
        )
        self.shape_cnn = nn.Sequential(
            nn.Conv1d(28, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveMaxPool1d(1) 
        )
        self.shape_fc = nn.Sequential(nn.Linear(128, 512), nn.LayerNorm(512), nn.GELU())

        # Gated Fusion
        self.norm_dna = nn.LayerNorm(768)
        self.norm_epi = nn.LayerNorm(768)
        self.gate_network = nn.Sequential(
            nn.Linear(768 * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 2), nn.Sigmoid() 
        )
        
        # Heads
        self.classification_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.regression_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))        

    def forward(self, tab, tab_mask, input_ids, attention_mask, shape, shape_mask):
        # 1. DNA Embeddings
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        hidden_states_t = hidden_states.permute(0, 2, 1)
        spatial_features = F.relu(self.spatial_conv(hidden_states_t)).permute(0, 2, 1)
        attn_weights = self.attention_pool(spatial_features).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=-1)
        dna_embeddings = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)

        # 2. Epi Embeddings
        tab_in = torch.cat([tab, tab_mask], dim=1)
        tab_out = self.tab_mlp(tab_in) 
        shape_in = torch.cat([shape, shape_mask], dim=1)
        shape_out = self.shape_cnn(shape_in).squeeze(-1)
        shape_out = self.shape_fc(shape_out) 
        epi_embeddings = torch.cat((tab_out, shape_out), dim=1) 
        
        # 3. Gated Fusion
        dna_norm = self.norm_dna(dna_embeddings)
        epi_norm = self.norm_epi(epi_embeddings)
        concat_features = torch.cat([dna_norm, epi_norm], dim=1)
        gates = self.gate_network(concat_features)
        
        gate_dna = gates[:, 0].unsqueeze(1) 
        gate_epi = gates[:, 1].unsqueeze(1) 
        fused_embeddings = (dna_norm * gate_dna) + (epi_norm * gate_epi) 
        
        # 4. Predictions
        class_logits = self.classification_head(fused_embeddings)
        m_value_pred = self.regression_head(fused_embeddings)
        
        return class_logits, m_value_pred, gate_dna, gate_epi

# Helper to convert Predicted M-value back to Beta for biological interpretability
def m_to_beta(m_val):
    m_val = np.clip(m_val, -20, 20)
    return (2 ** m_val) / (1 + (2 ** m_val))

# =========================================
# 3. Execution Logic
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv_path", type=str, required=True, default="data/datafiles/test.csv")
    parser.add_argument("--test_shape_tsv", type=str, required=True, default="data/datafiles/test_3d_shapes.tsv")
    parser.add_argument("--weights_path", type=str, required=True, default="checkpoints_multimodal_gated/gated_fusion_best.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_window_size", type=int, default=1000)
    parser.add_argument("--shape_window_size", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    logging.info("[*] Loading Test Data...")
    df = pd.read_csv(args.test_csv_path)
    shapes = pd.read_csv(args.test_shape_tsv, sep='\t', header=None, dtype=np.float32).values
    
    mask = df['probeID'].str.startswith('cg')
    df = df[mask].reset_index(drop=True)
    shapes = shapes[mask.values]
    
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    test_dataset = MultiModalLateFusionDataset(df, shapes, tokenizer, args.seq_window_size, args.shape_window_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = GatedFusionModel().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True), strict=True)
    model.eval()

    all_m_true, all_m_pred = [], []
    all_beta_true = []
    all_binary_true, all_binary_prob = [], []
    gate_dna_vals, gate_epi_vals = [], []

    logging.info("[*] Running inference to gather plot data...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            shape, shape_mask = batch['shape'].to(device), batch['shape_mask'].to(device)
            
            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred, gate_dna, gate_epi = model(tab, tab_mask, input_ids, attention_mask, shape, shape_mask)
                
            all_m_true.extend(batch['m_value'].float().numpy().flatten().tolist())
            all_m_pred.extend(m_value_pred.cpu().float().numpy().flatten().tolist())
            all_beta_true.extend(batch['beta_value'].float().numpy().flatten().tolist())
            all_binary_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())
            all_binary_true.extend(batch['binary_state'].float().numpy().flatten().tolist())
            
            gate_dna_vals.extend(gate_dna.cpu().float().numpy().flatten().tolist())
            gate_epi_vals.extend(gate_epi.cpu().float().numpy().flatten().tolist())

    # Calculate Metrics cleanly
    m_true, m_pred = np.array(all_m_true), np.array(all_m_pred)
    beta_true = np.array(all_beta_true)
    beta_pred = m_to_beta(m_pred)
    
    test_rmse_m = np.sqrt(mean_squared_error(m_true, m_pred))
    test_mae_m  = mean_absolute_error(m_true, m_pred)
    test_rmse_b = np.sqrt(mean_squared_error(beta_true, beta_pred))
    test_mae_b  = mean_absolute_error(beta_true, beta_pred)
    test_auc = roc_auc_score(all_binary_true, all_binary_prob)
    
    avg_dna_weight = np.mean(gate_dna_vals)
    avg_epi_weight = np.mean(gate_epi_vals)

    logging.info("\n========================================")
    logging.info(f"RMSE (M-Value): {test_rmse_m:.4f} | MAE (M-Value): {test_mae_m:.4f}")
    logging.info(f"RMSE (Beta)   : {test_rmse_b:.4f} | MAE (Beta)   : {test_mae_b:.4f}")
    logging.info(f"AUC (Binary)  : {test_auc:.4f}")
    logging.info("----------------------------------------")
    logging.info(f"Average DNA Gate Weight : {avg_dna_weight:.4f}")
    logging.info(f"Average Epi Gate Weight : {avg_epi_weight:.4f}")
    logging.info("========================================\n[*] Generating Figures...")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 1. DENSITY SCATTER PLOT
    plt.figure(figsize=(7, 6))
    plt.hexbin(beta_true, beta_pred, gridsize=40, cmap='Blues', bins='log', mincnt=5)
    plt.plot([0, 1], [0, 1], color='#ef4444', linestyle='--', linewidth=2, label='Ideal Fit')
    plt.title('Gated Fusion Density Scatter: Predicted vs. True Beta', fontweight='bold')
    plt.xlabel('True Methylation Fraction (Beta)')
    plt.ylabel('Predicted Methylation Fraction (Beta)')
    plt.colorbar(label='Log10(Count)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_1_fusion_density_scatter.png', dpi=300)
    plt.close()

    # 2A. SIGNED ERROR (RESIDUAL) DISTRIBUTION
    signed_errors = beta_pred - beta_true
    plt.figure(figsize=(7, 5))
    sns.histplot(signed_errors, bins=60, kde=True, color='#3b82f6', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.title('Signed Error Distribution (Predicted - True Beta)', fontweight='bold')
    plt.xlabel('Signed Error (\u0394\u03B2)')
    plt.ylabel('Probe Count')
    
    mean_signed = np.mean(signed_errors)
    textstr = f'Beta MAE: {test_mae_b:.4f}\nMean Signed Error: {mean_signed:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    plt.savefig('fig_2a_fusion_signed_error.png', dpi=300)
    plt.close()

    # 2B. ABSOLUTE ERROR DISTRIBUTION
    absolute_errors = np.abs(beta_pred - beta_true)
    plt.figure(figsize=(7, 5))
    sns.histplot(absolute_errors, bins=60, kde=True, color='#f59e0b', edgecolor='black')
    plt.axvline(test_mae_b, color='#ef4444', linestyle='--', linewidth=2, label=f'Beta MAE ({test_mae_b:.4f})')
    plt.title('Absolute Error Distribution |Predicted - True Beta|', fontweight='bold')
    plt.xlabel('Absolute Error Magnitude')
    plt.ylabel('Probe Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_2b_fusion_absolute_error.png', dpi=300)
    plt.close()

    # 3. BIMODAL DISTRIBUTION OVERLAY
    plt.figure(figsize=(7, 5))
    sns.kdeplot(beta_true, color='#10b981', fill=True, alpha=0.4, label='True Targets')
    sns.kdeplot(beta_pred, color='#6366f1', fill=True, alpha=0.4, label='Model Predictions')
    plt.title('Biological Bimodal Topology Recovery', fontweight='bold')
    plt.xlabel('DNA Methylation Fraction (Beta)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_3_fusion_bimodal.png', dpi=300)
    plt.close()

    # 4. ROC CURVE
    fpr, tpr, _ = roc_curve(all_binary_true, all_binary_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='#ef4444', lw=2, label=f'AUC = {test_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.title('Chromatin State Classification (ROC)', fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('fig_4_fusion_roc.png', dpi=300)
    plt.close()
    
    # 5. CALIBRATION CURVE
    prob_true, prob_pred = calibration_curve(all_binary_true, all_binary_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', color='#8b5cf6', lw=2, label='Gated Fusion Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    plt.title('Reliability Diagram (Calibration)', fontweight='bold')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_5_fusion_calibration.png', dpi=300)
    plt.close()

    logging.info("[✓] Successfully saved all 5 Gated Fusion publication figures to disk!")

if __name__ == "__main__":
    main()
