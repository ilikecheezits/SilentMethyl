import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import os
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
# 1. Multimodal Dataset with Zero-Masking Ablation
# =========================================
class MultimodalDataset(Dataset):
    def __init__(self, df, shape_data_array, tokenizer, max_length=1000, shape_window_size=100, ablation_mode="none"):
        self.df = df.reset_index(drop=True)
        self.shape_window_size = shape_window_size
        self.shape_data = shape_data_array
        self.tokenizer = tokenizer
        self.max_length = max_length
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
        seq = str(row.get('Sequence', '')) 
        encoded = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

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

        # Targets
        m_value = torch.tensor(row['M_Value_Target'], dtype=torch.float32)
        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tab': tab_tensor, 
            'tab_mask': tab_mask.float(),
            'shape': shape_tensor, 
            'shape_mask': shape_mask.float(),
            'm_value': m_value, 
            'beta_value': torch.tensor(beta, dtype=torch.float32),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32)
        }

# =========================================
# 2. Multimodal Gated Architecture
# =========================================
class SilentMethylModel(nn.Module):
    def __init__(self, tabular_dim=9, disable_gating=False):
        super(SilentMethylModel, self).__init__()
        self.disable_gating = disable_gating

        # 1. Sequence Branch (DNABERT-2)
        self.dnabert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.seq_proj = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1),
            nn.GELU()
        )
        self.seq_attention = nn.Sequential(
            nn.Linear(768, 1),
            nn.Tanh()
        )

        # 2. Epigenomic Branch (Outputs 256)
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
        
        # 3. 3D Shape Branch (Outputs 512)
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

        # 4. Fusion Components
        if self.disable_gating:
            self.concat_proj = nn.Linear(1536, 768)
        else:
            self.gate_w = nn.Linear(1536, 2)
        
        # 5. Prediction Heads
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

    def forward(self, input_ids, attention_mask, tab, tab_mask, shape, shape_mask):
        # Sequence
        outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
        seq_embeds = outputs[0]  # [B, Seq_Len, 768]
        seq_embeds = self.seq_proj(seq_embeds.transpose(1, 2)).transpose(1, 2)
        attn_weights = torch.softmax(self.seq_attention(seq_embeds), dim=1)
        x_seq = torch.sum(seq_embeds * attn_weights, dim=1) # [B, 768]

        # Epigenomics
        tab_in = torch.cat([tab, tab_mask], dim=1)
        x_epi = self.tab_mlp(tab_in) # [B, 256]
        
        # Shape
        shape_in = torch.cat([shape, shape_mask], dim=1)
        x_shape = self.shape_cnn(shape_in).squeeze(-1)
        x_shape = self.shape_fc(x_shape) # [B, 512]

        # Context (Epi + Shape)
        x_context = torch.cat([x_epi, x_shape], dim=1) # [B, 768]

        # Fusion
        if self.disable_gating:
            h = torch.cat([x_seq, x_context], dim=1)
            z = self.concat_proj(h)
        else:
            h = torch.cat([x_seq, x_context], dim=1)
            g = torch.sigmoid(self.gate_w(h))
            z = g[:, 0:1] * x_seq + g[:, 1:2] * x_context

        # Heads
        class_logits = self.classification_head(z)
        m_value_pred = self.regression_head(z)
        
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
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5) # Lower LR for transformer fine-tuning
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
    
    # Filter for CpG probes
    train_mask = train_df['probeID'].str.startswith('cg')
    val_mask = val_df['probeID'].str.startswith('cg')
    train_df = train_df[train_mask].reset_index(drop=True)
    train_shapes = train_shapes[train_mask.values]
    val_df = val_df[val_mask].reset_index(drop=True)
    val_shapes = val_shapes[val_mask.values]

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    train_dataset = MultimodalDataset(train_df, train_shapes, tokenizer, 1000, args.shape_window_size, args.ablation_mode)
    val_dataset = MultimodalDataset(val_df, val_shapes, tokenizer, 1000, args.shape_window_size, args.ablation_mode)    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SilentMethylModel(tabular_dim=9, disable_gating=args.disable_gating).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_huber = nn.HuberLoss(delta=1.345)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_mae = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
            shape, shape_mask = batch['shape'].to(device), batch['shape_mask'].to(device)
            m_value_target = batch['m_value'].to(device).view(-1, 1)
            binary_target = batch['binary_state'].to(device).view(-1, 1)

            with torch.amp.autocast('cuda'):
                class_logits, m_value_pred = model(input_ids, attention_mask, tab, tab_mask, shape, shape_mask)
                loss = criterion_bce(class_logits, binary_target) + criterion_huber(m_value_pred, m_value_target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0
        all_beta_true, all_beta_prob = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [VAL]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
                shape, shape_mask = batch['shape'].to(device), batch['shape_mask'].to(device)
                m_value_target = batch['m_value'].to(device).view(-1, 1)
                beta_target = batch['beta_value'].to(device).view(-1, 1)
                binary_target = batch['binary_state'].to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda'):
                    class_logits, m_value_pred = model(input_ids, attention_mask, tab, tab_mask, shape, shape_mask)
                    loss = criterion_bce(class_logits, binary_target) + criterion_huber(m_value_pred, m_value_target)
                    
                val_loss += loss.item()
                all_beta_true.extend(beta_target.cpu().float().numpy().flatten().tolist())
                all_beta_prob.extend(torch.sigmoid(class_logits).cpu().float().numpy().flatten().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_beta_mae  = mean_absolute_error(all_beta_true, all_beta_prob)

        logger.info(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Beta MAE: {val_beta_mae:.4f}")
        
        writer.add_scalar("Val/Beta_MAE", val_beta_mae, epoch)

        if val_beta_mae < best_val_mae:
            best_val_mae = val_beta_mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_weights.pth"))

if __name__ == "__main__":
    main()