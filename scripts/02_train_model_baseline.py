import os
import json
import shutil
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download, hf_hub_download
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from peft import LoraConfig, get_peft_model

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
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'm_value': torch.tensor(row['M_Value_Target'], dtype=torch.float32),
            'beta_value': torch.tensor(row['Median_Beta'], dtype=torch.float32),
            'probe_id': row['probeID']
        }

def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="dnabert2_local"):
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
    base_model = AutoModel.from_config(config, trust_remote_code=True)

    weights_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
    base_model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
    return config, base_model

class BaselineDNABert(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M"):
        super(BaselineDNABert, self).__init__()
        self.config, base_model = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size
        
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["Wqkv"],
            lora_dropout=0.1,
            bias="none"
        )
        self.bert = get_peft_model(base_model, lora_config)
        
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        self.classification_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.regression_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        
        hidden_states_t = hidden_states.permute(0, 2, 1)
        spatial_features = F.relu(self.spatial_conv(hidden_states_t)).permute(0, 2, 1)
        
        attn_weights = self.attention_pool(spatial_features).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        pooled_output = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)
        
        class_logits = self.classification_head(pooled_output)
        m_value_pred = self.regression_head(pooled_output)
        
        return class_logits, m_value_pred

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_term * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()
    
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/datafiles/train.csv")
    parser.add_argument("--val_path", type=str, default="data/datafiles/val.csv")
    parser.add_argument("--save_dir", default="checkpoints_baseline")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--grad_accum_steps", type=int, default=4) 
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    tb_log_dir = os.path.join(args.save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=tb_log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_stats_path = os.path.join(args.save_dir, "validation_statistics.csv")
    if not os.path.exists(val_stats_path):
        with open(val_stats_path, "w") as f:
            f.write("Epoch,Train_Loss,Val_Loss,M_Value_RMSE,Beta_MAE,Binarized_AUC\n")
    
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    train_df = train_df[train_df['probeID'].str.startswith('cg')].reset_index(drop=True)
    val_df = val_df[val_df['probeID'].str.startswith('cg')].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    train_dataset = SequenceOnlyBaselineDataset(train_df, tokenizer, window_size=args.window_size)
    val_dataset = SequenceOnlyBaselineDataset(val_df, tokenizer, window_size=args.window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = BaselineDNABert(args.model_path).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    criterion_focal = FocalLossWithLogits(alpha=0.5, gamma=2.0)
    criterion_huber = nn.HuberLoss(delta=1.345)
    
    # 1.0 ensures regression (continuous distance) is just as critical as binary accuracy
    lambda_reg = 1.0 
    scaler = torch.amp.GradScaler('cuda')
    
    # MAE-Driven Early Stopping Setup (Lower is better)
    best_val_mae = float('inf')
    epochs_no_improve = 0
    global_step = 0 

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        for step, batch in enumerate(pbar):
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            m_value_target = batch['m_value'].to(device).view(-1, 1)
            beta_target = batch['beta_value'].to(device).view(-1, 1)

            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred = model(input_ids, attention_mask)
                loss_clf = criterion_focal(class_logits, beta_target)
                loss_huber = criterion_huber(m_value_pred, m_value_target)
                loss = (loss_clf + (lambda_reg * loss_huber)) / args.grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += (loss.item() * args.grad_accum_steps)

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                writer.add_scalar("Train/Total_Loss", loss.item() * args.grad_accum_steps, global_step)
                pbar.set_postfix({'Loss': f"{loss.item() * args.grad_accum_steps:.4f}"})

        model.eval()
        val_loss = 0.0
        all_m_true, all_m_pred, all_beta_true, all_beta_prob = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [VAL]"):
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                m_value_target = batch['m_value'].to(device).view(-1, 1)
                beta_target = batch['beta_value'].to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda'):
                    class_logits, m_value_pred = model(input_ids, attention_mask)
                    batch_loss = criterion_focal(class_logits, beta_target) + (lambda_reg * criterion_huber(m_value_pred, m_value_target))
                    
                val_loss += batch_loss.item()
                all_m_true.extend(m_value_target.cpu().float().numpy().flatten().tolist())
                all_m_pred.extend(m_value_pred.cpu().float().numpy().flatten().tolist())
                all_beta_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())
                all_beta_true.extend(beta_target.cpu().float().numpy().flatten().tolist())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_rmse = np.sqrt(mean_squared_error(all_m_true, all_m_pred))
        val_mae_beta = mean_absolute_error(all_beta_true, all_beta_prob)
        
        binary_true_eval = [1 if b > 0.5 else 0 for b in all_beta_true]
        try:
            val_auc = roc_auc_score(binary_true_eval, all_beta_prob)
        except ValueError:
            val_auc = float('nan')

        logging.info(f"\n--- EPOCH {epoch} SUMMARY ---")
        logging.info(f"  Train Loss : {avg_train_loss:.4f} | Val Loss : {avg_val_loss:.4f}")
        logging.info(f"  Regression : M-Value RMSE={val_rmse:.4f}")
        logging.info(f"  Classify   : Beta MAE={val_mae_beta:.4f} | Binarized AUC={val_auc:.4f}")
        
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/RMSE_M", val_rmse, epoch)
        writer.add_scalar("Val/MAE_Beta", val_mae_beta, epoch)
        writer.add_scalar("Val/AUC", val_auc, epoch)
        
        with open(val_stats_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_rmse:.4f},{val_mae_beta:.4f},{val_auc:.4f}\n")
        
        # MAE-DRIVEN SAVING LOGIC (Minimizing error)
        if val_mae_beta < best_val_mae:
            best_val_mae = val_mae_beta
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "baseline_best_weights.pth"))
            logging.info(f"[✓] New Best Model Saved! (Beta MAE: {best_val_mae:.4f})")
        else:
            epochs_no_improve += 1
            logging.info(f"[!] Validation Beta MAE did not improve. Early stopping counter: {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                logging.info(f"[X] Early stopping triggered after {epoch} epochs. Training halted to prevent overfitting.")
                break

if __name__ == "__main__":
    main()