import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import logging
import os
import json
import shutil
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import snapshot_download, hf_hub_download
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from tqdm import tqdm

# =========================================
# 1. Dataset & Model Classes (Copied from Baseline)
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

def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local"):
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    config.output_attentions = True 
    base_model = AutoModel.from_config(config, trust_remote_code=True)
    return config, base_model

class BaselineDNABert(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M"):
        super(BaselineDNABert, self).__init__()
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size
        
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
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        class_logits = self.classification_head(pooled_output)
        m_value_pred = self.regression_head(pooled_output)
        return class_logits, m_value_pred, None

# =========================================
# 2. Execution Logic
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the completely unseen testing CSV")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to your baseline_best_weights.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=5000, help="Must match what you trained the baseline with")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"[*] Initializing UNSEEN TEST EVALUATION on {device}...")
    
    # Load 100% of the Test Data (No splitting!)
    df = pd.read_csv(args.test_data_path)
    before = len(df)
    df = df[df['probeID'].str.startswith('cg')].reset_index(drop=True)
    logging.info(f"[*] Loaded {len(df)} testing samples (Filtered out {before - len(df)} non-CpGs).")

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    test_dataset = SequenceOnlyBaselineDataset(df, tokenizer, window_size=args.window_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load Model and Weights
    logging.info("[*] Initializing Baseline Model...")
    model = BaselineDNABert().to(device)
    
    logging.info(f"[*] Injecting trained weights from {args.weights_path}...")
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    model.eval()

    all_m_true, all_m_pred = [], []
    all_binary_true, all_binary_prob = [], []

    logging.info("[*] Running inference on unseen data...")
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

    # Calculate Final Metrics
    test_rmse = np.sqrt(mean_squared_error(all_m_true, all_m_pred))
    test_mae  = mean_absolute_error(all_m_true, all_m_pred)
    
    unique_classes = set(all_binary_true)
    test_auc = roc_auc_score(all_binary_true, all_binary_prob) if len(unique_classes) == 2 else float('nan')

    logging.info("\n========================================")
    logging.info("FINAL UNSEEN TEST METRICS (BASELINE)")
    logging.info("========================================")
    logging.info(f"RMSE : {test_rmse:.4f}")
    logging.info(f"MAE  : {test_mae:.4f}")
    logging.info(f"AUC  : {test_auc:.4f}")
    logging.info("========================================")

if __name__ == "__main__":
    main()
