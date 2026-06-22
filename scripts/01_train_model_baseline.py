import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import shutil
import json
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download, hf_hub_download
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
# 1. Custom Baseline Dataset (STATICALLY ANCHORED)
# =========================================
class SequenceOnlyBaselineDataset(Dataset):
    def __init__(self, df, tokenizer, window_size=5000):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.window_size = window_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row['Healthy_5000bp_DNA']).upper()
        
        # Simple static cropping based on assumed perfect upstream centering
        true_c_idx = len(seq) // 2
        start_idx = true_c_idx - (self.window_size // 2)
        end_idx = start_idx + self.window_size
        
        if start_idx < 0: sequence = seq[:self.window_size]
        elif end_idx > len(seq): sequence = seq[-self.window_size:]
        else: sequence = seq[start_idx : end_idx]
            
        encoding = self.tokenizer(
            sequence, truncation=True, max_length=self.window_size, 
            padding='max_length', return_tensors='pt'
        )
        
        # Enforce strict binary state for BCE Loss
        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'm_value': torch.tensor(row['M_Value_Target'], dtype=torch.float32),
            'beta_value': torch.tensor(beta, dtype=torch.float32),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32),
            'probe_id': row['probeID']
        }

# =========================================
# 2. The Triton Neutralizer & Model Loader
# =========================================
def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local"):
    logging.info("--- Performing DNABERT-2 Surgery & Patching ---")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        logging.info("1. Downloading raw repo files...")
        cache_path = snapshot_download(model_path)
        for item in os.listdir(cache_path):
            src = os.path.join(cache_path, item)
            dst = os.path.join(local_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        logging.info("2. Neutralizing broken Triton Flash Attention...")
        triton_file = os.path.join(local_dir, "flash_attn_triton.py")
        if os.path.exists(triton_file):
            with open(triton_file, "w") as f:
                f.write("def __getattr__(name):\n    return None\n")

        logging.info("3. Patching config.json...")
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        config_data["use_flash_attn"] = False
        if "pad_token_id" not in config_data or config_data["pad_token_id"] is None:
            config_data["pad_token_id"] = 0
            
        with open(config_path, "w") as f:
            json.dump(config_data, f)
    else:
        logging.info("Patched directory already exists. Skipping surgery.")

    logging.info("4. Initializing blank model via from_config (Bypassing meta device manager)...")
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    config.output_attentions = True 
    base_model = AutoModel.from_config(config, trust_remote_code=True)

    logging.info("5. Manually mapping pre-trained weights to CPU...")
    weights_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict, strict=False)
    
    logging.info("✓ Surgery complete. Model safely loaded.")
    return config, base_model

# =========================================
# 3. Baseline Dual-Head Architecture
# =========================================
class BaselineDNABert(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M"):
        super(BaselineDNABert, self).__init__()
        
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size
        
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        
        hidden_states_t = hidden_states.permute(0, 2, 1)
        spatial_features = F.relu(self.spatial_conv(hidden_states_t)).permute(0, 2, 1)
        
        attn_weights = self.attention_pool(spatial_features).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=-1)
         
        #WE WANT THIS IF WE ARE DOING A MUTLIMODAL FUSION 
        pooled_output = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)

        class_logits = self.classification_head(pooled_output)
        m_value_pred = self.regression_head(pooled_output)
        attentions = outputs[-1] if isinstance(outputs, tuple) and len(outputs) > 1 else getattr(outputs, 'attentions', None)
        return class_logits, m_value_pred, attentions

# =========================================
# 4. Training Loop
# =========================================
def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/datafiles/train.csv")
    parser.add_argument("--val_path", type=str, default="data/datafiles/val.csv")
    parser.add_argument("--save_dir", default="checkpoints_baseline")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--grad_accum_steps", type=int, default=8) 
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--window_size", type=int, default=1000, help="Size of centered DNA sequence crop")
    args = parser.parse_args()
    
    writer = SummaryWriter(log_dir="runs/phase1_baseline")
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] Starting Training on {device} | Window Size: {args.window_size}bp")

    logger.info("[*] Loading Pre-Split Baseline Datasets...")
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    
    train_before = len(train_df)
    val_before = len(val_df)
    
    train_df = train_df[train_df['probeID'].str.startswith('cg')].reset_index(drop=True)
    val_df = val_df[val_df['probeID'].str.startswith('cg')].reset_index(drop=True)
    
    print(f'[*] Train filtering: {train_before - len(train_df)} non-CpG removed, {len(train_df)} remaining.')
    print(f'[*] Val filtering: {val_before - len(val_df)} non-CpG removed, {len(val_df)} remaining.')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    train_dataset = SequenceOnlyBaselineDataset(train_df, tokenizer, window_size=args.window_size)
    val_dataset = SequenceOnlyBaselineDataset(val_df, tokenizer, window_size=args.window_size)
    
    # Strictly Seeded DataLoader
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    model = BaselineDNABert(args.model_path).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # Locked in from the golden file
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_huber = nn.HuberLoss(delta=1.345)
    scaler = torch.amp.GradScaler('cuda')
    
    # --- RESUME LOGIC (SLURM SAFETY) ---
    start_epoch = 1
    best_val_mae = float('inf')
    global_step = 0
    latest_ckpt_path = os.path.join(args.save_dir, "latest_checkpoint.pt")
    
    if os.path.exists(latest_ckpt_path):
        logger.info(f"[*] Found interrupted run at {latest_ckpt_path}. Restoring state...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        global_step = (start_epoch - 1) * (len(train_loader) // args.grad_accum_steps)
        logger.info(f"[✓] Successfully resumed! Fast-forwarding to Epoch {start_epoch}...")
    # -----------------------------------

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            m_value_target = batch['m_value'].to(device).view(-1, 1)
            binary_target = batch['binary_state'].to(device).view(-1, 1)

            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred, attentions = model(input_ids, attention_mask)
                loss_bce = criterion_bce(class_logits, binary_target)
                loss_huber = criterion_huber(m_value_pred, m_value_target)
                
                loss_sparsity = 0.0
                if attentions is not None:
                    last_layer_attn = attentions[-1] 
                    loss_sparsity = torch.mean(torch.abs(last_layer_attn))
                
                loss = loss_bce + loss_huber + (0.01 * loss_sparsity)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += (loss.item() * args.grad_accum_steps)

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scale_before <= scaler.get_scale():
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                writer.add_scalar("Train/Total_Loss", loss.item() * args.grad_accum_steps, global_step)
                writer.add_scalar("Train/Huber_Loss", loss_huber.item(), global_step)
                writer.add_scalar("Train/BCE_Loss", loss_bce.item(), global_step)
                writer.add_scalar("Train/Learning_Rate", scheduler.get_last_lr()[0], global_step)
                pbar.set_postfix({'Loss': f"{loss.item() * args.grad_accum_steps:.4f}"})

        model.eval()
        val_loss = 0.0
        all_m_true, all_m_pred = [], []
        all_beta_true, all_beta_prob = [], []
        all_binary_true = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [VAL]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                m_value_target = batch['m_value'].to(device).view(-1, 1)
                beta_target = batch['beta_value'].to(device).view(-1, 1)
                binary_target = batch['binary_state'].to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda'):
                    class_logits, m_value_pred, _ = model(input_ids, attention_mask)
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
        
        # Calculate M-Value Metrics
        val_m_rmse = np.sqrt(mean_squared_error(all_m_true, all_m_pred))
        val_m_mae  = mean_absolute_error(all_m_true, all_m_pred)
        
        # Calculate Beta Metrics
        val_beta_rmse = np.sqrt(mean_squared_error(all_beta_true, all_beta_prob))
        val_beta_mae  = mean_absolute_error(all_beta_true, all_beta_prob)
        
        unique_classes = set(all_binary_true)
        if len(unique_classes) == 2:
            val_auc = roc_auc_score(all_binary_true, all_beta_prob)
        else:
            val_auc = float('nan')
            logger.warning(f"[!] AUC skipped -- only class(es) {unique_classes} present in val set")

        m_pred_arr = np.array(all_m_pred)
        beta_prob_arr = np.array(all_beta_prob)

        logger.info(f"\n--- EPOCH {epoch} SUMMARY ---")
        logger.info(f"  Train Loss : {train_loss / len(train_loader):.4f}")
        logger.info(f"  Val Loss   : {avg_val_loss:.4f}")
        logger.info(f"  [M-Value]        RMSE={val_m_rmse:.4f}  MAE={val_m_mae:.4f}")
        logger.info(f"  [M-Value]        pred range=[{m_pred_arr.min():.3f}, {m_pred_arr.max():.3f}]  mean={m_pred_arr.mean():.3f}  std={m_pred_arr.std():.3f}")
        logger.info(f"  [Beta-Value]     RMSE={val_beta_rmse:.4f}  MAE={val_beta_mae:.4f}")
        logger.info(f"  [Classification] AUC={val_auc:.4f}")
        logger.info(f"  [Classification] prob range=[{beta_prob_arr.min():.3f}, {beta_prob_arr.max():.3f}]  mean={beta_prob_arr.mean():.3f}  std={beta_prob_arr.std():.3f}")
        
        # Core TensorBoard Scalars
        writer.add_scalar("Val/Epoch_Total_Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/M_Value_RMSE", val_m_rmse, epoch)
        writer.add_scalar("Val/M_Value_MAE", val_m_mae, epoch)
        writer.add_scalar("Val/Beta_RMSE", val_beta_rmse, epoch)
        writer.add_scalar("Val/Beta_MAE", val_beta_mae, epoch)
        writer.add_scalar("Val/AUC", val_auc, epoch)
        
        # Histograms
        writer.add_histogram("Distributions/M_Value_Predictions", m_pred_arr, epoch)
        writer.add_histogram("Distributions/M_Value_True", np.array(all_m_true), epoch)
        writer.add_histogram("Distributions/Beta_Probabilities", beta_prob_arr, epoch)
        
        # Graph 1: M-Value Regression Scatter Plot
        fig_scatter_m, ax_scatter_m = plt.subplots(figsize=(8, 8))
        ax_scatter_m.scatter(all_m_true, all_m_pred, alpha=0.3, edgecolors='none')
        min_m = min(min(all_m_true), min(all_m_pred))
        max_m = max(max(all_m_true), max(all_m_pred))
        ax_scatter_m.plot([min_m, max_m], [min_m, max_m], 'r--', lw=2, label="Perfect Prediction")
        ax_scatter_m.set_xlabel("True M-Value")
        ax_scatter_m.set_ylabel("Predicted M-Value")
        ax_scatter_m.set_title(f"Epoch {epoch} M-Value Accuracy")
        ax_scatter_m.legend()
        writer.add_figure("Plots/M_Value_Scatter", fig_scatter_m, epoch)
        plt.close(fig_scatter_m)
        
        # Graph 2: Beta-Value Scatter Plot
        fig_scatter_beta, ax_scatter_beta = plt.subplots(figsize=(8, 8))
        ax_scatter_beta.scatter(all_beta_true, all_beta_prob, alpha=0.3, edgecolors='none', color='green')
        ax_scatter_beta.plot([0, 1], [0, 1], 'r--', lw=2, label="Perfect Prediction")
        ax_scatter_beta.set_xlabel("True Beta-Value")
        ax_scatter_beta.set_ylabel("Predicted Beta Probability")
        ax_scatter_beta.set_title(f"Epoch {epoch} Beta-Value Accuracy")
        ax_scatter_beta.legend()
        writer.add_figure("Plots/Beta_Scatter", fig_scatter_beta, epoch)
        plt.close(fig_scatter_beta)
        
        # Graph 3: Classification ROC Curve
        if not np.isnan(val_auc):
            fpr, tpr, _ = roc_curve(all_binary_true, all_beta_prob)
            fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {val_auc:.4f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'Epoch {epoch} Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            writer.add_figure("Plots/ROC_Curve", fig_roc, epoch)
            plt.close(fig_roc)

        # SAVE LOGIC (Continuous Checkpointing + Best Tracker)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_mae': best_val_mae
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f"latest_checkpoint_{epoch + 1}.pt"))
        logger.info(f"[✓] Full Training State backed up for Epoch {epoch}.")

        if val_beta_mae < best_val_mae:
            best_val_mae = val_beta_mae
            best_save_path = os.path.join(args.save_dir, f"baseline_best_weights_{epoch + 1}.pth")
            torch.save(model.state_dict(), best_save_path)
            logger.info(f"[★] New Best Model (Beta MAE: {best_val_mae:.4f}) saved!")

if __name__ == "__main__":
    main()
