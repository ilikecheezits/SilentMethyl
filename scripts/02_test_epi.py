import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from tqdm import tqdm

# =========================================
# 1. Pure Epigenetic Dataset
# =========================================
class PureEpigeneticDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tabular_features = [
            'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
            'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 
            'Ref_H3K36me3_Signal', 'Ref_H3K4me1_Signal',
            'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
        ]

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        tab_raw = row[self.tabular_features].values.astype(np.float32)
        tab_tensor = torch.tensor(tab_raw)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)

        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'tab': tab_tensor, 'tab_mask': tab_mask.float(),
            'm_value': m_value, 'beta_value': torch.tensor(beta, dtype=torch.float32),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32)
        }

# =========================================
# 2. Pure Epi Architecture
# =========================================
class PureEpigeneticNN(nn.Module):
    def __init__(self, tabular_dim=9):
        super(PureEpigeneticNN, self).__init__()
        
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256)
        )
        
        self.epi_proj = nn.Linear(256, 768)
        
        self.classification_head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )        

    def forward(self, tab, tab_mask):
        tab_in = torch.cat([tab, tab_mask], dim=1)
        tab_out = self.tab_mlp(tab_in)
        epi_features = self.epi_proj(tab_out)
        
        class_logits = self.classification_head(epi_features)
        m_value_pred = self.regression_head(epi_features)
        return class_logits, m_value_pred

def m_to_beta(m_val):
    m_val = np.clip(m_val, -20, 20)
    return (2 ** m_val) / (1 + (2 ** m_val))

# =========================================
# 3. Execution Logic
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv_path", type=str, required=True, default="data/datafiles/test.csv")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="pure_epi")
    parser.add_argument("--batch_size", type=int, default=512) 
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"[*] Starting Evaluation for Pure Epigenomic Model")
    
    logging.info("[*] Loading Test CSV...")
    df = pd.read_csv(args.test_csv_path)
    df = df[df['probeID'].str.startswith('cg')].reset_index(drop=True)
    
    test_dataset = PureEpigeneticDataset(df)
    # High batch size because MLP is extremely lightweight
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = PureEpigeneticNN(tabular_dim=9).to(device)
    
    logging.info(f"[*] Loading strict weights from {args.weights_path}")
    state_dict = torch.load(args.weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_m_true, all_m_pred = [], []
    all_beta_true = []
    all_binary_true, all_binary_prob = [], []

    logging.info("[*] Running inference to gather plot data...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {args.run_name}"):
            tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
            
            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred = model(tab, tab_mask)
                
            all_m_true.extend(batch['m_value'].float().numpy().flatten().tolist())
            all_m_pred.extend(m_value_pred.cpu().float().numpy().flatten().tolist())
            all_beta_true.extend(batch['beta_value'].float().numpy().flatten().tolist())
            all_binary_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())
            all_binary_true.extend(batch['binary_state'].float().numpy().flatten().tolist())

    # Calculate Metrics
    m_true, m_pred = np.array(all_m_true), np.array(all_m_pred)
    beta_true = np.array(all_beta_true)
    beta_pred = m_to_beta(m_pred)
    
    test_rmse_m = np.sqrt(mean_squared_error(m_true, m_pred))
    test_mae_m  = mean_absolute_error(m_true, m_pred)
    test_rmse_b = np.sqrt(mean_squared_error(beta_true, beta_pred))
    test_mae_b  = mean_absolute_error(beta_true, beta_pred)
    test_auc = roc_auc_score(all_binary_true, all_binary_prob)
    
    logging.info("\n========================================")
    logging.info(f"FINAL RESULTS: PURE EPIGENOMIC MODEL")
    logging.info(f"RMSE (M-Value): {test_rmse_m:.4f} | MAE (M-Value): {test_mae_m:.4f}")
    logging.info(f"RMSE (Beta)   : {test_rmse_b:.4f} | MAE (Beta)   : {test_mae_b:.4f}")
    logging.info(f"AUC (Binary)  : {test_auc:.4f}")
    logging.info("========================================\n[*] Generating Figures...")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    prefix = args.run_name

    # 1. DENSITY SCATTER PLOT
    plt.figure(figsize=(7, 6))
    plt.hexbin(beta_true, beta_pred, gridsize=40, cmap='Blues', bins='log', mincnt=5)
    plt.plot([0, 1], [0, 1], color='#ef4444', linestyle='--', linewidth=2, label='Ideal Fit')
    plt.title(f'Pure Epigenomic Model Density Scatter', fontweight='bold')
    plt.xlabel('True Methylation Fraction (Beta)')
    plt.ylabel('Predicted Methylation Fraction (Beta)')
    plt.colorbar(label='Log10(Count)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_fig_1_density_scatter.png', dpi=300)
    plt.close()

    # 2A. SIGNED ERROR (RESIDUAL) DISTRIBUTION
    signed_errors = beta_pred - beta_true
    plt.figure(figsize=(7, 5))
    sns.histplot(signed_errors, bins=60, kde=True, color='#3b82f6', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.title(f'Signed Error Distribution (Pure Epi)', fontweight='bold')
    plt.xlabel('Signed Error (\u0394\u03B2)')
    plt.ylabel('Probe Count')
    
    mean_signed = np.mean(signed_errors)
    textstr = f'Beta MAE: {test_mae_b:.4f}\nMean Signed Error: {mean_signed:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    plt.savefig(f'{prefix}_fig_2a_signed_error.png', dpi=300)
    plt.close()

    # 2B. ABSOLUTE ERROR DISTRIBUTION
    absolute_errors = np.abs(beta_pred - beta_true)
    plt.figure(figsize=(7, 5))
    sns.histplot(absolute_errors, bins=60, kde=True, color='#f59e0b', edgecolor='black')
    plt.axvline(test_mae_b, color='#ef4444', linestyle='--', linewidth=2, label=f'Beta MAE ({test_mae_b:.4f})')
    plt.title(f'Absolute Error Distribution (Pure Epi)', fontweight='bold')
    plt.xlabel('Absolute Error Magnitude')
    plt.ylabel('Probe Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_fig_2b_absolute_error.png', dpi=300)
    plt.close()

    # 3. BIMODAL DISTRIBUTION OVERLAY
    plt.figure(figsize=(7, 5))
    sns.kdeplot(beta_true, color='#10b981', fill=True, alpha=0.4, label='True Targets')
    sns.kdeplot(beta_pred, color='#6366f1', fill=True, alpha=0.4, label='Model Predictions')
    plt.title(f'Biological Bimodal Topology Recovery (Pure Epi)', fontweight='bold')
    plt.xlabel('DNA Methylation Fraction (Beta)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_fig_3_bimodal.png', dpi=300)
    plt.close()

    # 4. ROC CURVE
    fpr, tpr, _ = roc_curve(all_binary_true, all_binary_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='#ef4444', lw=2, label=f'AUC = {test_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.title(f'Chromatin State Classification (Pure Epi)', fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'{prefix}_fig_4_roc.png', dpi=300)
    plt.close()
    
    # 5. CALIBRATION CURVE
    prob_true, prob_pred = calibration_curve(all_binary_true, all_binary_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', color='#8b5cf6', lw=2, label=f'Pure Epi')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    plt.title(f'Reliability Diagram (Pure Epi)', fontweight='bold')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_fig_5_calibration.png', dpi=300)
    plt.close()

    logging.info(f"[✓] Successfully saved all 5 Pure Epigenomic figures to disk!")

if __name__ == "__main__":
    main()