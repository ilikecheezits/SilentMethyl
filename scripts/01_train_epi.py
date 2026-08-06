import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import random
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve

# =========================================
# 0. Strict Reproducibility
# =========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        
        # Process Tabular Epigenomics
        tab_raw = row[self.tabular_features].values.astype(np.float32)
        tab_tensor = torch.tensor(tab_raw)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)
        
        # Targets
        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'tab': tab_tensor, 
            'tab_mask': tab_mask.float(),
            'm_value': m_value, 
            'beta_value': torch.tensor(beta, dtype=torch.float32),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32)
        }

# =========================================
# 2. Pure Epi Architecture
# =========================================
class PureEpigeneticNN(nn.Module):
    def __init__(self, tabular_dim=9):
        super(PureEpigeneticNN, self).__init__()
        
        # Epigenomic Feature Extractor
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        
        # Project up to 768 to perfectly align with DNABERT-2 later
        self.epi_proj = nn.Linear(256, 768)
        
        # Prediction Heads (Taking 768 dims)
        self.classification_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )        

    def forward(self, tab, tab_mask):
        tab_in = torch.cat([tab, tab_mask], dim=1)
        tab_out = self.tab_mlp(tab_in)
        epi_features = self.epi_proj(tab_out)
        
        class_logits = self.classification_head(epi_features)
        m_value_pred = self.regression_head(epi_features)
        return class_logits, m_value_pred

# =========================================
# 3. Training Loop
# =========================================
def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/datafiles/train.csv")
    parser.add_argument("--val_path", type=str, default="data/datafiles/val.csv")
    parser.add_argument("--save_dir", default="checkpoints_epi_only")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3) 
    args = parser.parse_args()
    
    writer = SummaryWriter(log_dir="runs/phase1_pure_epi")
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] Starting Pure Epigenetic NN Training on {device}")

    # Load Data
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    
    # Filter for CpG probes
    train_df = train_df[train_df['probeID'].str.startswith('cg')].reset_index(drop=True)
    val_df = val_df[val_df['probeID'].str.startswith('cg')].reset_index(drop=True)

    train_dataset = PureEpigeneticDataset(train_df)
    val_dataset = PureEpigeneticDataset(val_df)    
    
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PureEpigeneticNN(tabular_dim=9).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_huber = nn.HuberLoss(delta=1.345)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_mae = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
            m_value_target = batch['m_value'].to(device).view(-1, 1)
            binary_target = batch['binary_state'].to(device).view(-1, 1)

            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred = model(tab, tab_mask)
                loss = criterion_bce(class_logits, binary_target) + criterion_huber(m_value_pred, m_value_target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0
        all_beta_true, all_beta_prob, all_binary_true = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [VAL]"):
                tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
                m_value_target = batch['m_value'].to(device).view(-1, 1)
                beta_target = batch['beta_value'].to(device).view(-1, 1)
                binary_target = batch['binary_state'].to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda'):
                    class_logits, m_value_pred = model(tab, tab_mask)
                    loss = criterion_bce(class_logits, binary_target) + criterion_huber(m_value_pred, m_value_target)
                    
                val_loss += loss.item()
                all_beta_true.extend(beta_target.cpu().float().numpy().flatten().tolist())
                all_beta_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())
                all_binary_true.extend(binary_target.cpu().float().numpy().flatten().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_beta_mae  = mean_absolute_error(all_beta_true, all_beta_prob)
        
        unique_classes = set(all_binary_true)
        val_auc = roc_auc_score(all_binary_true, all_beta_prob) if len(unique_classes) == 2 else float('nan')

        logger.info(f"\n--- EPOCH {epoch} SUMMARY ---")
        logger.info(f"  Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Beta MAE: {val_beta_mae:.4f} | AUC: {val_auc:.4f}")
        
        writer.add_scalar("Val/Epoch_Total_Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/Beta_MAE", val_beta_mae, epoch)
        writer.add_scalar("Val/AUC", val_auc, epoch)
        
        if val_beta_mae < best_val_mae:
            best_val_mae = val_beta_mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_weights.pth"))
            logger.info(f"[★] New Best Epi Model (Beta MAE: {best_val_mae:.4f}) saved!")

if __name__ == "__main__":
    main()
