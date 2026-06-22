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
# 1. Pure Epigenetic Dataset (No DNA Sequence)
# =========================================
class PureEpigeneticDataset(Dataset):
    def __init__(self, df, shape_data_array, shape_window_size=100):
        self.df = df.reset_index(drop=True)
        self.shape_window_size = shape_window_size
        self.shape_data = shape_data_array

        assert len(self.df) == len(self.shape_data), "CRITICAL ERROR: CSV and TSV row counts do not match!"
        
        # Ensure these match your new 100bp localized feature names!
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
        
        # Process Tabular (100bp local averages)
        tab_raw = row[self.tabular_features].values.astype(np.float32)
        tab_tensor = torch.tensor(tab_raw)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)
        
        # Process 3D Shape
        shape_flat = self.shape_data[idx]
        shape_tensor = torch.tensor(shape_flat).view(14, self.shape_window_size)
        shape_mask = ~torch.isnan(shape_tensor)
        shape_tensor = torch.nan_to_num(shape_tensor, nan=0.0)

        # Targets
        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'tab': tab_tensor, 
            'tab_mask': tab_mask.float(),
            'shape': shape_tensor, 
            'shape_mask': shape_mask.float(),
            'm_value': m_value, 
            'beta_value': torch.tensor(beta, dtype=torch.float32),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32)
        }

# =========================================
# 2. Pure NN Architecture (Perfect 768-Dim Alignment)
# =========================================
class PureEpigeneticNN(nn.Module):
    def __init__(self, tabular_dim=9):
        super(PureEpigeneticNN, self).__init__()
        
        # Tabular Extractor (Outputs 256)
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
        
        # 3D Shape Extractor (Outputs 512)
        self.shape_cnn = nn.Sequential(
            nn.Conv1d(in_channels=28, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1) 
        )
        self.shape_fc = nn.Sequential(
            nn.Linear(128, 512), 
            nn.LayerNorm(512), 
            nn.GELU()
        )
        
        # Pure NN Fusion Head (256 + 512 = exactly 768 input dims)
        # Matches BaselineDNABert heads EXACTLY: 768 -> 256 -> 1
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

    def forward(self, tab, tab_mask, shape, shape_mask):
        # 1. Process Epigenetics
        tab_in = torch.cat([tab, tab_mask], dim=1)
        tab_out = self.tab_mlp(tab_in)
        
        # 2. Process Spatial 3D Bends
        shape_in = torch.cat([shape, shape_mask], dim=1)
        shape_out = self.shape_cnn(shape_in).squeeze(-1)
        shape_out = self.shape_fc(shape_out)

        # 3. WE WANT THIS IF WE ARE DOING A MULTIMODAL FUSION OR LATENT ENSEMBLE
        fused_features = torch.cat((tab_out, shape_out), dim=1) # [Batch_Size, 768]
        
        class_logits = self.classification_head(fused_features)
        m_value_pred = self.regression_head(fused_features)
        
        return class_logits, m_value_pred

# =========================================
# 3. Training Loop
# =========================================
def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/datafiles/train.csv")
    parser.add_argument("--val_path", type=str, default="data/datafiles/val.csv")
    parser.add_argument("--train_shape_tsv", type=str, required=True)
    parser.add_argument("--val_shape_tsv", type=str, required=True)
    parser.add_argument("--save_dir", default="checkpoints_pure_nn")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3) 
    parser.add_argument("--shape_window_size", type=int, default=100)
    args = parser.parse_args()
    
    writer = SummaryWriter(log_dir="runs/phase1_pure_nn")
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] Starting Pure Epigenetic NN Training on {device}")

    # Load Data
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    train_shapes = pd.read_csv(args.train_shape_tsv, sep='\t', header=None, dtype=np.float32).values
    val_shapes = pd.read_csv(args.val_shape_tsv, sep='\t', header=None, dtype=np.float32).values
    
    # Filter for CpG probes only
    train_mask = train_df['probeID'].str.startswith('cg')
    val_mask = val_df['probeID'].str.startswith('cg')
    train_df = train_df[train_mask].reset_index(drop=True)
    train_shapes = train_shapes[train_mask.values]
    val_df = val_df[val_mask].reset_index(drop=True)
    val_shapes = val_shapes[val_mask.values]

    train_dataset = PureEpigeneticDataset(train_df, train_shapes, args.shape_window_size)
    val_dataset = PureEpigeneticDataset(val_df, val_shapes, args.shape_window_size)    
    
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model & Optimizer
    model = PureEpigeneticNN(tabular_dim=9).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Warmup Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_huber = nn.HuberLoss(delta=1.345)
    scaler = torch.amp.GradScaler('cuda')
    
    start_epoch = 1
    best_val_mae = float('inf')
    global_step = 0

    # --- RESUME LOGIC ---
    latest_ckpt_path = os.path.join(args.save_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_ckpt_path):
        logger.info(f"[*] Found interrupted run. Restoring state...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        global_step = (start_epoch - 1) * len(train_loader)
        logger.info(f"[✓] Fast-forwarding to Epoch {start_epoch}...")

    # --- MAIN LOOP ---
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
            shape, shape_mask = batch['shape'].to(device), batch['shape_mask'].to(device)
            m_value_target = batch['m_value'].to(device).view(-1, 1)
            binary_target = batch['binary_state'].to(device).view(-1, 1)

            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred = model(tab, tab_mask, shape, shape_mask)
                loss_bce = criterion_bce(class_logits, binary_target)
                loss_huber = criterion_huber(m_value_pred, m_value_target)
                loss = loss_bce + loss_huber

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            global_step += 1
            
            writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Learning_Rate", scheduler.get_last_lr()[0], global_step)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0
        all_m_true, all_m_pred, all_beta_true, all_beta_prob, all_binary_true = [], [], [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [VAL]"):
                tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
                shape, shape_mask = batch['shape'].to(device), batch['shape_mask'].to(device)
                m_value_target = batch['m_value'].to(device).view(-1, 1)
                beta_target = batch['beta_value'].to(device).view(-1, 1)
                binary_target = batch['binary_state'].to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda'):
                    class_logits, m_value_pred = model(tab, tab_mask, shape, shape_mask)
                    loss_bce = criterion_bce(class_logits, binary_target)
                    loss_huber = criterion_huber(m_value_pred, m_value_target)
                    batch_loss = loss_bce + loss_huber
                    
                val_loss += batch_loss.item()
                all_m_true.extend(m_value_target.cpu().float().numpy().flatten().tolist())
                all_m_pred.extend(m_value_pred.cpu().float().numpy().flatten().tolist())
                all_beta_true.extend(beta_target.cpu().float().numpy().flatten().tolist())
                all_beta_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())
                all_binary_true.extend(binary_target.cpu().float().numpy().flatten().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_m_rmse = np.sqrt(mean_squared_error(all_m_true, all_m_pred))
        val_m_mae  = mean_absolute_error(all_m_true, all_m_pred)
        val_beta_rmse = np.sqrt(mean_squared_error(all_beta_true, all_beta_prob))
        val_beta_mae  = mean_absolute_error(all_beta_true, all_beta_prob)
        
        unique_classes = set(all_binary_true)
        val_auc = roc_auc_score(all_binary_true, all_beta_prob) if len(unique_classes) == 2 else float('nan')

        logger.info(f"\n--- EPOCH {epoch} SUMMARY ---")
        logger.info(f"  Train Loss : {train_loss / len(train_loader):.4f} | Val Loss : {avg_val_loss:.4f}")
        logger.info(f"  [M-Value]    MAE={val_m_mae:.4f} | [Beta] MAE={val_beta_mae:.4f} | AUC={val_auc:.4f}")
        
        writer.add_scalar("Val/Epoch_Total_Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/Beta_MAE", val_beta_mae, epoch)
        writer.add_scalar("Val/AUC", val_auc, epoch)
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_mae': best_val_mae
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f"latest_checkpoint.pt"))

        if val_beta_mae < best_val_mae:
            best_val_mae = val_beta_mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"pure_nn_best.pth"))
            logger.info(f"[★] New Best Model (Beta MAE: {best_val_mae:.4f}) saved!")

if __name__ == "__main__":
    main()