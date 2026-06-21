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
# 1. The Triton Neutralizer & Model Loader
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
# 2. Multi-Modal Dataset (STATICALLY ANCHORED)
# =========================================
class MultiModalLateFusionDataset(Dataset):
    def __init__(self, df, shape_data_array, tokenizer, seq_window_size=1000, shape_window_size=100):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_window_size = seq_window_size
        self.shape_window_size = shape_window_size
        self.shape_data = shape_data_array

        logging.info(f"[✓] Verification: CSV Rows = {len(self.df)}, TSV Array Rows = {len(self.shape_data)}")
        assert len(self.df) == len(self.shape_data), "CRITICAL ERROR: CSV and TSV row counts do not match!"
        
        self.tabular_features = [
            'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
            'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 
            'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. SPATIAL ANCHOR (Tabular Epigenetics) ---
        tab_raw = row[self.tabular_features].values.astype(np.float32)
        tab_tensor = torch.tensor(tab_raw)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)
        
        # --- 2. PHYSICAL ANCHOR (Sequence - Exact Baseline Crop) ---
        full_sequence = str(row['Healthy_5000bp_DNA']).upper()
        true_c_idx = len(full_sequence) // 2
        start_idx = true_c_idx - (self.seq_window_size // 2)
        end_idx = start_idx + self.seq_window_size
        
        if start_idx < 0: sequence = full_sequence[:self.seq_window_size]
        elif end_idx > len(full_sequence): sequence = full_sequence[-self.seq_window_size:]
        else: sequence = full_sequence[start_idx : end_idx]
            
        encoding = self.tokenizer(
            sequence, truncation=True, max_length=self.seq_window_size, 
            padding='max_length', return_tensors='pt'
        )
        
        # --- 3. GEOMETRIC ANCHOR (3D Shape) ---
        shape_flat = self.shape_data[idx]
        shape_tensor = torch.tensor(shape_flat).view(14, self.shape_window_size)
        shape_mask = ~torch.isnan(shape_tensor)
        shape_tensor = torch.nan_to_num(shape_tensor, nan=0.0)

        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'tab': tab_tensor,
            'tab_mask': tab_mask.float(),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'shape': shape_tensor,
            'shape_mask': shape_mask.float(),
            'm_value': m_value,
            'beta_value': torch.tensor(beta, dtype=torch.float32),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32),
            'probe_id': row['probeID']
        }

# =========================================
# 3. Robust Late Fusion Architecture
# =========================================
class SilentMethylModel(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M", tabular_dim=7):
        super(SilentMethylModel, self).__init__()
        
        # A. Spatial Anchor (Epigenetic MLP with Norm to prevent collapse)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32)
        )
        
        # B. Physical Anchor (Baseline sequence logic + Compression)
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size
        
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        self.text_fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # C. Geometric Anchor (Deepened 1D-CNN)
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
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        # D. Fusion Synthesis Heads
        self.classification_head = nn.Sequential(
            nn.Linear(224, 256),
            nn.GELU(),
            nn.Dropout(0.3), 
            nn.Linear(256, 1)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(224, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, tab, tab_mask, input_ids, attention_mask, shape, shape_mask):
        # 1. Epigenetics
        tab_in = torch.cat([tab, tab_mask], dim=1)
        tab_out = self.tab_mlp(tab_in)
        
        # 2. Sequence (Exact Baseline Match)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        
        hidden_states_t = hidden_states.permute(0, 2, 1)
        spatial_features = F.relu(self.spatial_conv(hidden_states_t)).permute(0, 2, 1)
        
        attn_weights = self.attention_pool(spatial_features).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        seq_pooled = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)
        text_out = self.text_fc(seq_pooled)
        
        # 3. 3D Shape Topology
        shape_in = torch.cat([shape, shape_mask], dim=1)
        shape_out = self.shape_cnn(shape_in).squeeze(-1)
        shape_out = self.shape_fc(shape_out)

        # 4. Asymmetric Modality Dropout (Forces network to learn all modalities)
        if self.training:
            rand_val = torch.rand(1).item()
            # 10% chance to drop epigenetics/shape (rely on DNA)
            if rand_val < 0.10: 
                tab_out = torch.zeros_like(tab_out)
                shape_out = torch.zeros_like(shape_out)
            # 15% chance to drop DNA entirely! (Forces reliance on epigenetics/shape)
            elif rand_val < 0.25: 
                text_out = torch.zeros_like(text_out)
        
        # 5. Fused Synthesis (32 + 128 + 64 = 224)
        fused_features = torch.cat((tab_out, text_out, shape_out), dim=1)
        class_logits = self.classification_head(fused_features)
        m_value_pred = self.regression_head(fused_features)
        
        attentions = bert_out[-1] if isinstance(bert_out, tuple) and len(bert_out) > 1 else getattr(bert_out, 'attentions', None)
        return class_logits, m_value_pred, attentions

# =========================================
# 4. Training Loop
# =========================================
def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/datafiles/train.csv")
    parser.add_argument("--val_path", type=str, default="data/datafiles/val.csv")
    parser.add_argument("--train_shape_tsv", type=str, required=True, help="Must match train.csv perfectly")
    parser.add_argument("--val_shape_tsv", type=str, required=True, help="Must match val.csv perfectly")
    parser.add_argument("--save_dir", default="./checkpoints_multimodal")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--grad_accum_steps", type=int, default=8) 
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq_window_size", type=int, default=1000)
    parser.add_argument("--shape_window_size", type=int, default=100)
    args = parser.parse_args()
    
    writer = SummaryWriter(log_dir="runs/phase2_multimodal")
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] Starting Multimodal Training on {device} | Seq Window: {args.seq_window_size}bp | Shape Window: {args.shape_window_size}bp")

    logger.info("[*] Loading Pre-Split CSV and TSV Data...")
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    
    train_shapes = pd.read_csv(args.train_shape_tsv, sep='\t', header=None, dtype=np.float32).values
    val_shapes = pd.read_csv(args.val_shape_tsv, sep='\t', header=None, dtype=np.float32).values
    
    train_mask = train_df['probeID'].str.startswith('cg')
    val_mask = val_df['probeID'].str.startswith('cg')
    
    train_df = train_df[train_mask].reset_index(drop=True)
    train_shapes = train_shapes[train_mask.values]
    
    val_df = val_df[val_mask].reset_index(drop=True)
    val_shapes = val_shapes[val_mask.values]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    train_dataset = MultiModalLateFusionDataset(train_df, train_shapes, tokenizer, args.seq_window_size, args.shape_window_size)
    val_dataset = MultiModalLateFusionDataset(val_df, val_shapes, tokenizer, args.seq_window_size, args.shape_window_size)    
    
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SilentMethylModel(args.model_path, tabular_dim=7).to(device)
    
    bert_params = []
    new_head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "bert" in name: bert_params.append(param)
        else: new_head_params.append(param)

    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': 5e-5},     
        {'params': new_head_params, 'lr': 1e-3}    
    ], weight_decay=1e-4)
    
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_huber = nn.HuberLoss(delta=1.345)
    scaler = torch.amp.GradScaler('cuda')
    
    # --- RESUME LOGIC ---
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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        
        for step, batch in enumerate(pbar):
            tab = batch['tab'].to(device)
            tab_mask = batch['tab_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            shape = batch['shape'].to(device)
            shape_mask = batch['shape_mask'].to(device)
            
            m_value_target = batch['m_value'].to(device).view(-1, 1)
            binary_target = batch['binary_state'].to(device).view(-1, 1)

            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred, attentions = model(
                    tab, tab_mask, input_ids, attention_mask, shape, shape_mask
                )
                
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
                writer.add_scalar("Train/Learning_Rate_Heads", optimizer.param_groups[1]['lr'], global_step)
                pbar.set_postfix({'Loss': f"{loss.item() * args.grad_accum_steps:.4f}"})

        model.eval()
        val_loss = 0.0
        all_m_true, all_m_pred = [], []
        all_beta_true, all_beta_prob = [], []
        all_binary_true = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [VAL]"):
                tab = batch['tab'].to(device)
                tab_mask = batch['tab_mask'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                shape = batch['shape'].to(device)
                shape_mask = batch['shape_mask'].to(device)
                
                m_value_target = batch['m_value'].to(device).view(-1, 1)
                beta_target = batch['beta_value'].to(device).view(-1, 1)
                binary_target = batch['binary_state'].to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda'):
                    class_logits, m_value_pred, _ = model(
                        tab, tab_mask, input_ids, attention_mask, shape, shape_mask
                    )
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
        if len(unique_classes) == 2:
            val_auc = roc_auc_score(all_binary_true, all_beta_prob)
        else:
            val_auc = float('nan')

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
        
        writer.add_scalar("Val/Epoch_Total_Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/M_Value_RMSE", val_m_rmse, epoch)
        writer.add_scalar("Val/M_Value_MAE", val_m_mae, epoch)
        writer.add_scalar("Val/Beta_RMSE", val_beta_rmse, epoch)
        writer.add_scalar("Val/Beta_MAE", val_beta_mae, epoch)
        writer.add_scalar("Val/AUC", val_auc, epoch)
        
        writer.add_histogram("Distributions/M_Value_Predictions", m_pred_arr, epoch)
        writer.add_histogram("Distributions/M_Value_True", np.array(all_m_true), epoch)
        writer.add_histogram("Distributions/Beta_Probabilities", beta_prob_arr, epoch)
        
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
        
        fig_scatter_beta, ax_scatter_beta = plt.subplots(figsize=(8, 8))
        ax_scatter_beta.scatter(all_beta_true, all_beta_prob, alpha=0.3, edgecolors='none', color='green')
        ax_scatter_beta.plot([0, 1], [0, 1], 'r--', lw=2, label="Perfect Prediction")
        ax_scatter_beta.set_xlabel("True Beta-Value")
        ax_scatter_beta.set_ylabel("Predicted Beta Probability")
        ax_scatter_beta.set_title(f"Epoch {epoch} Beta-Value Accuracy")
        ax_scatter_beta.legend()
        writer.add_figure("Plots/Beta_Scatter", fig_scatter_beta, epoch)
        plt.close(fig_scatter_beta)
        
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
            best_save_path = os.path.join(args.save_dir, f"multimodal_best_weights.pth")
            torch.save(model.state_dict(), best_save_path)
            logger.info(f"[★] New Best Model (Beta MAE: {best_val_mae:.4f}) saved!")

if __name__ == "__main__":
    main()