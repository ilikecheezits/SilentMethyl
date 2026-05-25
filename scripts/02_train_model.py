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
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download, hf_hub_download
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from peft import LoraConfig, get_peft_model

# =========================================
# 1. The Triton Neutralizer & Model Loader
# =========================================
def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local"):
    """
    Bypasses the 'device meta' PyTorch bug by downloading the remote code locally,
    disabling the broken Triton Flash Attention, and manually loading weights to CPU.
    """
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
# 2. Multi-Modal Dataset (DYNAMICALLY ANCHORED)
# =========================================
class MultiModalLateFusionDataset(Dataset):
    def __init__(self, df, shape_tsv, tokenizer, window_size=101, debug_cpg=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.debug_cpg = debug_cpg

        logging.info(f"[*] Loading 3D shape tensor from {shape_tsv}...")
        # header=None ensures pandas counts all 370,109 lines as data
        self.shape_data = pd.read_csv(shape_tsv, sep='\t', header=None, dtype=np.float32).values
        
        logging.info(f"[✓] Verification: CSV Rows = {len(self.df)}, TSV Rows = {len(self.shape_data)}")
        assert len(self.df) == len(self.shape_data), "CRITICAL ERROR: CSV and TSV row counts do not match!"

        self.tabular_features = [
            'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
            'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Target_Base_PhyloP_100way'
        ]
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. SPATIAL ANCHOR (Tabular Epigenetics with Mask) ---
        tab_raw = row[self.tabular_features].values.astype(np.float32)
        tab_tensor = torch.tensor(tab_raw)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)
        
        # --- 2. PHYSICAL ANCHOR (DNABERT-2 Sequence) ---
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
            if self.debug_cpg:
                raise ValueError(f"CRITICAL: No 'CG' found for index {idx}. Window: {search_window}")
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
        
        # --- 3. PHYSICAL ANCHOR (3D DNA Shape with Mask) ---
        shape_flat = self.shape_data[idx]
        shape_tensor = torch.tensor(shape_flat).view(14, self.window_size)
        shape_mask = ~torch.isnan(shape_tensor)
        shape_tensor = torch.nan_to_num(shape_tensor, nan=0.0)

        # Targets
        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        binary_state = torch.tensor(row['Binary_State_Target'], dtype=torch.float32)
        
        return {
            'tab': tab_tensor,
            'tab_mask': tab_mask.float(),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'shape': shape_tensor,
            'shape_mask': shape_mask.float(),
            'm_value': m_value,
            'binary_state': binary_state,
            'probe_id': row['probeID']
        }

# =========================================
# 3. Late Fusion Multi-Modal Architecture
# =========================================
class SilentMethylModel(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M", tabular_dim=6):
        super(SilentMethylModel, self).__init__()
        
        # A. Spatial Anchor (Epigenetic MLP)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )
        
        # B. Physical Anchor (DNABERT-2 with PEFT/LoRA)
        self.config, base_bert = patch_and_load_dnabert(model_path)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.05,
            bias="none"
        )
        self.bert = get_peft_model(base_bert, lora_config)
        self.bert.print_trainable_parameters()
        
        self.text_fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.GELU()
        )
        
        # C. Physical Anchor (1D-CNN for 3D Shape)
        self.shape_cnn = nn.Sequential(
            nn.Conv1d(in_channels=28, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1) 
        )
        self.shape_fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU()
        )
        
        # D. Late Fusion Synthesis Heads (Match Baseline Outputs)
        self.classification_head = nn.Sequential(
            nn.Linear(224, 256),
            nn.GELU(),
            nn.Dropout(0.2), 
            nn.Linear(256, 1)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(224, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, tab, tab_mask, input_ids, attention_mask, shape, shape_mask):
        # 1. Process Spatial Topology
        tab_in = torch.cat([tab, tab_mask], dim=1)
        tab_out = self.tab_mlp(tab_in)
        
        # 2. Process Sequence
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        seq_pooled = sum_embeddings / sum_mask
        text_out = self.text_fc(seq_pooled)
        
        # 3. Process 3D Geometry
        shape_in = torch.cat([shape, shape_mask], dim=1)
        shape_out = self.shape_cnn(shape_in).squeeze(-1)
        shape_out = self.shape_fc(shape_out)

        # 4. EXPLICIT MODALITY DROPOUT
        if self.training:
            rand_val = torch.rand(1).item()
            if rand_val < 0.15: 
                tab_out = torch.zeros_like(tab_out)
            elif rand_val < 0.30: 
                text_out = torch.zeros_like(text_out)
                shape_out = torch.zeros_like(shape_out)
        
        # 5. Synthesize
        fused_features = torch.cat((tab_out, text_out, shape_out), dim=1)
        class_logits = self.classification_head(fused_features)
        m_value_pred = self.regression_head(fused_features)
        
        if isinstance(bert_out, tuple):
            attentions = bert_out[-1] if len(bert_out) > 1 else None
        else:
            attentions = bert_out.attentions if hasattr(bert_out, 'attentions') else None
        
        return class_logits, m_value_pred, attentions

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_term = alpha_t * focal_term
            
        loss = focal_term * bce_loss
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

# =========================================
# 4. Training Loop
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/baseline_phase1_final.csv")
    parser.add_argument("--shape_tsv", type=str, default="training_wild_type_3d_shapes.tsv")
    parser.add_argument("--save_dir", default="./checkpoints_multimodal")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--grad_accum_steps", type=int, default=8) 
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--window_size", type=int, default=101, help="Must match 3D Shape geometry")
    parser.add_argument("--debug_cpg", action="store_true", help="Assert the exact center of sequence is CG")
    args = parser.parse_args()
    
    writer = SummaryWriter(log_dir="runs/phase2_multimodal")
    global_step = 0 
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] Starting Multimodal Training on {device} | Window Size: {args.window_size}bp")

    logger.info("[*] Loading CSV Data...")
    df = pd.read_csv(args.data_path)
    df = df.sample(frac=1)
    before = len(df)
    df = df[df['probeID'].str.startswith('cg')].reset_index(drop=True)
    print(f'[*] Filtered non-CpG probes: {before - len(df)} removed, {len(df)} remaining.')
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    train_dataset = MultiModalLateFusionDataset(train_df, args.shape_tsv, tokenizer, window_size=args.window_size, debug_cpg=args.debug_cpg)
    val_dataset = MultiModalLateFusionDataset(val_df, args.shape_tsv, tokenizer, window_size=args.window_size, debug_cpg=args.debug_cpg)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SilentMethylModel(args.model_path).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    criterion_focal = FocalLossWithLogits(alpha=0.5, gamma=2.0)
    criterion_huber = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
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
                
                loss_focal = criterion_focal(class_logits, binary_target)
                loss_huber = criterion_huber(m_value_pred, m_value_target)
                
                loss_sparsity = 0.0
                if attentions is not None:
                    last_layer_attn = attentions[-1] 
                    loss_sparsity = torch.mean(torch.abs(last_layer_attn))
                
                loss = loss_focal + loss_huber + (0.01 * loss_sparsity)
                loss = loss / args.grad_accum_steps

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
                writer.add_scalar("Train/MSE_Loss", loss_huber.item(), global_step)
                writer.add_scalar("Train/Focal_Loss", loss_focal.item(), global_step)
                writer.add_scalar("Train/Learning_Rate", scheduler.get_last_lr()[0], global_step)
                pbar.set_postfix({'Loss': f"{loss.item() * args.grad_accum_steps:.4f}"})

        model.eval()
        val_loss = 0.0
        all_m_true, all_m_pred = [], []
        all_binary_true, all_binary_prob = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [VAL]"):
                tab = batch['tab'].to(device)
                tab_mask = batch['tab_mask'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                shape = batch['shape'].to(device)
                shape_mask = batch['shape_mask'].to(device)
                
                m_value_target = batch['m_value'].to(device).view(-1, 1)
                binary_target = batch['binary_state'].to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda'):
                    class_logits, m_value_pred, _ = model(
                        tab, tab_mask, input_ids, attention_mask, shape, shape_mask
                    )
                    loss_focal = criterion_focal(class_logits, binary_target)
                    loss_huber = criterion_huber(m_value_pred, m_value_target)
                    batch_loss = loss_focal + loss_huber
                    
                val_loss += batch_loss.item()
                all_m_true.extend(m_value_target.cpu().float().numpy().flatten().tolist())
                all_m_pred.extend(m_value_pred.cpu().float().numpy().flatten().tolist())
                all_binary_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())
                all_binary_true.extend(binary_target.cpu().float().numpy().flatten().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_rmse = np.sqrt(mean_squared_error(all_m_true, all_m_pred))
        val_mae  = mean_absolute_error(all_m_true, all_m_pred)

        unique_classes = set(all_binary_true)
        if len(unique_classes) == 2:
            val_auc = roc_auc_score(all_binary_true, all_binary_prob)
        else:
            val_auc = float('nan')
            logger.warning(f"[!] AUC skipped -- only class(es) {unique_classes} present in val set")

        m_pred_arr = np.array(all_m_pred)
        prob_arr   = np.array(all_binary_prob)

        logger.info(f"\n--- EPOCH {epoch} SUMMARY ---")
        logger.info(f"  Train Loss : {train_loss / len(train_loader):.4f}")
        logger.info(f"  Val Loss   : {avg_val_loss:.4f}")
        logger.info(f"  [Regression]     RMSE={val_rmse:.4f}  MAE={val_mae:.4f}")
        logger.info(f"  [Regression]     pred range=[{m_pred_arr.min():.3f}, {m_pred_arr.max():.3f}]  mean={m_pred_arr.mean():.3f}  std={m_pred_arr.std():.3f}")
        logger.info(f"  [Regression]     true range=[{min(all_m_true):.3f}, {max(all_m_true):.3f}]  mean={np.mean(all_m_true):.3f}")
        logger.info(f"  [Classification] AUC={val_auc:.4f}")
        logger.info(f"  [Classification] prob range=[{prob_arr.min():.3f}, {prob_arr.max():.3f}]  mean={prob_arr.mean():.3f}  std={prob_arr.std():.3f}")
        
        writer.add_scalar("Val/Epoch_Total_Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/RMSE", val_rmse, epoch)
        writer.add_scalar("Val/AUC", val_auc, epoch)
        
        writer.add_histogram("Distributions/Predictions", m_pred_arr, epoch)
        writer.add_histogram("Distributions/True_Targets", np.array(all_m_true), epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "multimodal_best_weights.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"[✓] New Best Model Saved!")

if __name__ == "__main__":
    main()
