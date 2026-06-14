import torch
import torch.nn as nn
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
# 1. Dataset & Model Classes (Kept exactly as yours for perfect alignment)
# =========================================
class SequenceOnlyBaselineDataset(Dataset):
    def __init__(self, df, tokenizer, window_size=5000, debug_cpg=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.debug_cpg = debug_cpg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_sequence = str(row['Healthy_5000bp_DNA']).upper()
        seq_len = len(full_sequence)
        
        approx_center = seq_len // 2
        search_radius = 50 
        search_start = max(0, approx_center - search_radius)
        search_end = min(seq_len, approx_center + search_radius)
        search_window = full_sequence[search_start : search_end]
        
        if "CG" in search_window:
            cg_local_idx = search_window.index("CG")
            true_c_idx = search_start + cg_local_idx
        else:
            true_c_idx = approx_center 
            
        half_window = self.window_size // 2
        start_idx = true_c_idx - half_window
        end_idx = start_idx + self.window_size
        
        if start_idx < 0: sequence = full_sequence[:self.window_size]
        elif end_idx > seq_len: sequence = full_sequence[-self.window_size:]
        else: sequence = full_sequence[start_idx : end_idx]
            
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.window_size, 
            padding='max_length',
            return_tensors='pt'
        )
        
        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        binary_state = torch.tensor(row['Binary_State_Target'], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'm_value': m_value,
            'binary_state': binary_state,
            'probe_id': row['probeID']
        }

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
    config.output_attentions = True 
    base_model = AutoModel.from_config(config, trust_remote_code=True)
    return config, base_model

class BaselineDNABert(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M"):
        super(BaselineDNABert, self).__init__()
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size
        
        self.classification_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.regression_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        class_logits = self.classification_head(pooled_output)
        m_value_pred = self.regression_head(pooled_output)
        return class_logits, m_value_pred, None

# Helper to convert M-value back to Beta for biological interpretability
def m_to_beta(m_val):
    return (2 ** m_val) / (1 + (2 ** m_val))

# =========================================
# 2. Execution Logic
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to testing CSV")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to baseline_best_weights.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    df = pd.read_csv(args.test_data_path)
    df = df[df['probeID'].str.startswith('cg')].reset_index(drop=True)
    
    # Sample down to 10k max for plotting speed and clarity (optional, can remove)
    if len(df) > 10000: df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    test_dataset = SequenceOnlyBaselineDataset(df, tokenizer, window_size=args.window_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = BaselineDNABert().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    model.eval()

    all_m_true, all_m_pred = [], []
    all_binary_true, all_binary_prob = [], []

    logging.info("[*] Running inference to gather plot data...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            m_value_target = batch['m_value'].to(device)
            binary_target = batch['binary_state'].to(device)
            
            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred, _ = model(input_ids, attention_mask)
                
            all_m_true.extend(m_value_target.cpu().float().numpy().flatten().tolist())
            all_m_pred.extend(m_value_pred.cpu().float().numpy().flatten().tolist())
            all_binary_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())
            all_binary_true.extend(binary_target.cpu().float().numpy().flatten().tolist())

    # Calculate Metrics in M-value (for math) and Beta-value (for biology)
    m_true, m_pred = np.array(all_m_true), np.array(all_m_pred)
    beta_true, beta_pred = m_to_beta(m_true), m_to_beta(m_pred)
    
    test_rmse_m = np.sqrt(mean_squared_error(m_true, m_pred))
    test_mae_m  = mean_absolute_error(m_true, m_pred)
    test_rmse_b = np.sqrt(mean_squared_error(beta_true, beta_pred))
    
    test_auc = roc_auc_score(all_binary_true, all_binary_prob)

    logging.info("\n========================================")
    logging.info(f"RMSE (M-Value): {test_rmse_m:.4f} | RMSE (Beta): {test_rmse_b:.4f}")
    logging.info(f"MAE (M-Value) : {test_mae_m:.4f}")
    logging.info(f"AUC (Binary)  : {test_auc:.4f}")
    logging.info("========================================\n[*] Generating Figures...")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 1. DENSITY SCATTER PLOT
    plt.figure(figsize=(7, 6))
    plt.hexbin(beta_true, beta_pred, gridsize=40, cmap='Blues', bins='log', mincnt=1)
    plt.plot([0, 1], [0, 1], color='#ef4444', linestyle='--', linewidth=2, label='Ideal Fit')
    plt.title('Baseline Density Scatter: Predicted vs. True $\\beta$', fontweight='bold')
    plt.xlabel('True Methylation Fraction ($\\beta$)')
    plt.ylabel('Predicted Methylation Fraction ($\\beta$)')
    plt.colorbar(label='$\log_{10}(\text{Count})$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_1_density_scatter.png', dpi=300)
    plt.close()

    # =========================================================================
    # 2. SIGNED ERROR (RESIDUAL) DISTRIBUTION
    # =========================================================================
    signed_errors = beta_pred - beta_true
    
    plt.figure(figsize=(7, 5))
    sns.histplot(signed_errors, bins=60, kde=True, color='#3b82f6', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.title('Signed Error Distribution (Predicted - True)', fontweight='bold')
    plt.xlabel('Signed Error (\u0394\u03B2)')
    plt.ylabel('Probe Count')
    
    # Add text box with MAE and Mean Error stats
    mean_signed = np.mean(signed_errors)
    textstr = f'MAE: {test_mae_m:.4f}\nMean Signed Error: {mean_signed:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    plt.savefig('fig_2a_signed_error.png', dpi=300)
    plt.close()

    # =========================================================================
    # 2B. ABSOLUTE ERROR DISTRIBUTION
    # =========================================================================
    absolute_errors = np.abs(beta_pred - beta_true)
    
    plt.figure(figsize=(7, 5))
    # Using a KDE and histogram focused on the positive domain
    sns.histplot(absolute_errors, bins=60, kde=True, color='#f59e0b', edgecolor='black')
    plt.axvline(test_mae_m, color='#ef4444', linestyle='--', linewidth=2, label=f'MAE ({test_mae_m:.4f})')
    plt.title('Absolute Error Distribution |Predicted - True|', fontweight='bold')
    plt.xlabel('Absolute Error Magnitude')
    plt.ylabel('Probe Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_2b_absolute_error.png', dpi=300)
    plt.close()

    # 3. BIMODAL DISTRIBUTION OVERLAY
    plt.figure(figsize=(7, 5))
    sns.kdeplot(beta_true, color='#10b981', fill=True, alpha=0.4, label='True Targets')
    sns.kdeplot(beta_pred, color='#6366f1', fill=True, alpha=0.4, label='Model Predictions')
    plt.title('Biological Bimodal Topology Recovery', fontweight='bold')
    plt.xlabel('DNA Methylation Fraction ($\\beta$)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_3_bimodal.png', dpi=300)
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
    plt.savefig('fig_4_roc.png', dpi=300)
    plt.close()
    
    # 5. CALIBRATION CURVE
    prob_true, prob_pred = calibration_curve(all_binary_true, all_binary_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', color='#8b5cf6', lw=2, label='DNABERT-2 Baseline')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    plt.title('Reliability Diagram (Calibration)', fontweight='bold')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_5_calibration.png', dpi=300)
    plt.close()

    logging.info("[✓] Successfully saved all 5 publication figures to disk!")

if __name__ == "__main__":
    main()
