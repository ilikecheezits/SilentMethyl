import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm
import numpy as np
import os
import shutil
import json
import random
import pandas as pd
import argparse
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

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
    
    weights_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict, strict=False)
    return config, base_model

# =========================================
# 2. Multimodal Dataset with Zero-Masking Ablation
# =========================================
class MultimodalDataset(Dataset):
    def __init__(self, df, shape_data_array, tokenizer, seq_window_size=1000, shape_window_size=100, ablation_mode="none"):
        self.df = df.reset_index(drop=True)
        self.shape_window_size = shape_window_size
        self.seq_window_size = seq_window_size
        self.shape_data = shape_data_array
        self.tokenizer = tokenizer
        self.ablation_mode = ablation_mode

        assert len(self.df) == len(self.shape_data), "CRITICAL ERROR: CSV and TSV row counts do not match!"
        
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
        
        # --- Modality 1: Sequence ---
        full_sequence = str(row.get('Sequence', row.get('Healthy_5000bp_DNA', ''))).upper()
        true_c_idx = len(full_sequence) // 2
        start_idx = true_c_idx - (self.seq_window_size // 2)
        end_idx = start_idx + self.seq_window_size
        
        if start_idx < 0: sequence = full_sequence[:self.seq_window_size]
        elif end_idx > len(full_sequence): sequence = full_sequence[-self.seq_window_size:]
        else: sequence = full_sequence[start_idx : end_idx]
            
        encoded = self.tokenizer(sequence, truncation=True, max_length=self.seq_window_size, padding='max_length', return_tensors='pt')
        input_ids = encoded['input_ids'].flatten()
        attention_mask = encoded['attention_mask'].flatten()

        # --- Modality 2: Epigenomics (Tabular) ---
        tab_raw = row[self.tabular_features].values.astype(np.float32)
        tab_tensor = torch.tensor(tab_raw)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)
        
        # --- Modality 3: 3D Shape ---
        shape_flat = self.shape_data[idx]
        shape_tensor = torch.tensor(shape_flat).view(14, self.shape_window_size)
        shape_mask = ~torch.isnan(shape_tensor)
        shape_tensor = torch.nan_to_num(shape_tensor, nan=0.0)

        # --- ABLATION LOGIC: ZERO-MASKING ---
        if self.ablation_mode == 'no_shape':
            shape_tensor = torch.zeros_like(shape_tensor)
            shape_mask = torch.zeros_like(shape_mask)
        elif self.ablation_mode == 'no_epi':
            tab_tensor = torch.zeros_like(tab_tensor)
            tab_mask = torch.zeros_like(tab_mask)

        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            'tab': tab_tensor, 'tab_mask': tab_mask.float(),
            'shape': shape_tensor, 'shape_mask': shape_mask.float(),
            'm_value': m_value, 'beta_value': torch.tensor(beta, dtype=torch.float32),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32)
        }

# =========================================
# 3. Dynamic Gated Fusion Architecture (Aligned for Checkpoints)
# =========================================
class GatedFusionModel(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M", tabular_dim=9, disable_gating=False):
        super(GatedFusionModel, self).__init__()
        self.disable_gating = disable_gating
        
        # --- TOWER X: DNA Sequence ---
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size 
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        # --- TOWER Y: Epigenetic & Shape ---
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256)
        )
        self.shape_cnn = nn.Sequential(
            nn.Conv1d(in_channels=28, out_channels=64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveMaxPool1d(1) 
        )
        self.shape_fc = nn.Sequential(nn.Linear(128, 512), nn.LayerNorm(512), nn.GELU())

        # --- FUSION BLOCK ---
        self.norm_dna = nn.LayerNorm(768)
        self.norm_epi = nn.LayerNorm(768)
        
        if self.disable_gating:
            self.concat_proj = nn.Linear(1536, 768)
        else:
            self.gate_network = nn.Sequential(
                nn.Linear(1536, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, 2), nn.Sigmoid() 
            )
        
        # --- HEADS ---
        self.classification_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.regression_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))        

    def forward(self, tab, tab_mask, input_ids, attention_mask, shape, shape_mask):
        # 1. DNA
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        spatial_features = F.relu(self.spatial_conv(hidden_states.permute(0, 2, 1))).permute(0, 2, 1)
        attn_weights = F.softmax(self.attention_pool(spatial_features).squeeze(-1).masked_fill(attention_mask == 0, -1e4), dim=-1)
        dna_embeddings = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)

        # 2. Context
        tab_out = self.tab_mlp(torch.cat([tab, tab_mask], dim=1))
        shape_out = self.shape_fc(self.shape_cnn(torch.cat([shape, shape_mask], dim=1)).squeeze(-1))
        epi_embeddings = torch.cat((tab_out, shape_out), dim=1)
        
        # 3. Fusion
        dna_norm = self.norm_dna(dna_embeddings)
        epi_norm = self.norm_epi(epi_embeddings)
        concat_features = torch.cat([dna_norm, epi_norm], dim=1)
        
        if self.disable_gating:
            fused_embeddings = self.concat_proj(concat_features)
        else:
            gates = self.gate_network(concat_features)
            fused_embeddings = (dna_norm * gates[:, 0].unsqueeze(1)) + (epi_norm * gates[:, 1].unsqueeze(1))
        
        class_logits = self.classification_head(fused_embeddings)
        m_value_pred = self.regression_head(fused_embeddings)
        
        return class_logits, m_value_pred

# =========================================
# 4. Training Loop (with Ancestor Injection & Freezing)
# =========================================
def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/datafiles/train.csv")
    parser.add_argument("--val_path", type=str, default="data/datafiles/val.csv")
    parser.add_argument("--train_shape_tsv", type=str, required=True)
    parser.add_argument("--val_shape_tsv", type=str, required=True)
    parser.add_argument("--baseline_weights", type=str, required=True)
    parser.add_argument("--pure_nn_weights", type=str, required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq_window_size", type=int, default=1000)
    parser.add_argument("--shape_window_size", type=int, default=100)
    parser.add_argument("--ablation_mode", type=str, default="none", choices=["none", "no_shape", "no_epi"])
    parser.add_argument("--disable_gating", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", os.path.basename(args.save_dir)))
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[*] Ablation Mode: {args.ablation_mode.upper()} | Gating Disabled: {args.disable_gating}")

    # Load Data
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
    train_dataset = MultimodalDataset(train_df, train_shapes, tokenizer, args.seq_window_size, args.shape_window_size, args.ablation_mode)
    val_dataset = MultimodalDataset(val_df, val_shapes, tokenizer, args.seq_window_size, args.shape_window_size, args.ablation_mode)    
    
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = GatedFusionModel(args.model_path, tabular_dim=9, disable_gating=args.disable_gating).to(device)

    # --- WEIGHT INJECTION ---
    logger.info("[*] Injecting Ancestor Weights (Baseline DNABERT & Pure NN)...")
    baseline_ckpt = torch.load(args.baseline_weights, map_location=device, weights_only=False)
    if 'model_state_dict' in baseline_ckpt: baseline_ckpt = baseline_ckpt['model_state_dict']
    model.load_state_dict(baseline_ckpt, strict=False)

    nn_ckpt = torch.load(args.pure_nn_weights, map_location=device, weights_only=False)
    if 'model_state_dict' in nn_ckpt: nn_ckpt = nn_ckpt['model_state_dict']
    model.load_state_dict(nn_ckpt, strict=False)

    # --- THE PERMANENT FREEZE ---
    logger.info("--- LOCKING ANCESTORS: ONLY TRAINING FUSION LOGIC ---")
    trainable_params = []
    for name, param in model.named_parameters():
        if "norm_" in name or "gate_network" in name or "concat_proj" in name or "head" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False 

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_huber = nn.HuberLoss(delta=1.345)
    scaler = torch.amp.GradScaler('cuda')
    
    optimizer = optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-4)
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad() 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        for step, batch in enumerate(pbar):
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
            shape, shape_mask = batch['shape'].to(device), batch['shape_mask'].to(device)
            m_value_target = batch['m_value'].to(device).view(-1, 1)
            binary_target = batch['binary_state'].to(device).view(-1, 1)

            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred = model(tab, tab_mask, input_ids, attention_mask, shape, shape_mask)
                loss = criterion_bce(class_logits, binary_target) + criterion_huber(m_value_pred, m_value_target)
                loss = loss / args.grad_accum_steps 

            scaler.scale(loss).backward()
            train_loss += (loss.item() * args.grad_accum_steps)

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scale_before <= scaler.get_scale(): scheduler.step()
                optimizer.zero_grad()
            
            pbar.set_postfix({'Loss': f"{loss.item() * args.grad_accum_steps:.4f}"})

        model.eval()
        val_loss = 0.0
        all_beta_true, all_beta_prob = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [VAL]"):
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
                shape, shape_mask = batch['shape'].to(device), batch['shape_mask'].to(device)
                m_value_target = batch['m_value'].to(device).view(-1, 1)
                beta_target = batch['beta_value'].to(device).view(-1, 1)
                binary_target = batch['binary_state'].to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda'):
                    class_logits, m_value_pred = model(tab, tab_mask, input_ids, attention_mask, shape, shape_mask)
                    
                all_beta_true.extend(beta_target.cpu().float().numpy().flatten().tolist())
                all_beta_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())

        val_beta_mae = mean_absolute_error(all_beta_true, all_beta_prob)
        logger.info(f"Epoch {epoch} | Beta MAE: {val_beta_mae:.4f}")
        writer.add_scalar("Val/Beta_MAE", val_beta_mae, epoch)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"weights_ep{epoch}.pth"))

if __name__ == "__main__":
    main()
