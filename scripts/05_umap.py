import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import logging
import os
import re
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import snapshot_download
from tqdm import tqdm
import umap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =========================================
# 1. Dataset Loader & Helpers
# =========================================
class MultiModalLateFusionDataset(Dataset):
    def __init__(self, df, shape_data_array, tokenizer, seq_window_size=1000, shape_window_size=100):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_window_size = seq_window_size
        self.shape_window_size = shape_window_size
        self.shape_data = shape_data_array

        self.tabular_features = [
            'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
            'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Ref_H3K36me3_Signal', 
            'Ref_H3K4me1_Signal', 'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
        ]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        tab_raw = row[self.tabular_features].values.astype(np.float32)
        tab_tensor = torch.tensor(tab_raw)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)
        
        full_sequence = str(row['Healthy_5000bp_DNA']).upper()
        true_c_idx = len(full_sequence) // 2
        start_idx = true_c_idx - (self.seq_window_size // 2)
        end_idx = start_idx + self.seq_window_size
        
        if start_idx < 0: sequence = full_sequence[:self.seq_window_size]
        elif end_idx > len(full_sequence): sequence = full_sequence[-self.seq_window_size:]
        else: sequence = full_sequence[start_idx : end_idx]
            
        encoding = self.tokenizer(sequence, truncation=True, max_length=self.seq_window_size, padding='max_length', return_tensors='pt')
        
        shape_flat = self.shape_data[idx]
        shape_tensor = torch.tensor(shape_flat).view(14, self.shape_window_size)
        shape_mask = ~torch.isnan(shape_tensor)
        shape_tensor = torch.nan_to_num(shape_tensor, nan=0.0)

        beta = float(row['Median_Beta'])
        binary_state = 1.0 if beta > 0.5 else 0.0
        
        return {
            'tab': tab_tensor, 'tab_mask': tab_mask.float(),
            'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
            'shape': shape_tensor, 'shape_mask': shape_mask.float(),
            'binary_state': torch.tensor(binary_state, dtype=torch.float32)
        }

def m_value_to_beta(m_val):
    m_val = np.clip(m_val, -20, 20)
    return (2 ** m_val) / (1 + (2 ** m_val))

def parse_mutation_id(mut_id):
    mut_str = str(mut_id).upper()
    if mut_str == 'NAN': return None, None, None
    match = re.search(r'(\d+)\s*([ACGT])\s*>\s*([ACGT])', mut_str)
    if match: return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None

# =========================================
# 2. Gated Fusion Model
# =========================================
def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local"):
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
    return config, base_model

class GatedFusionModel(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M", tabular_dim=9):
        super(GatedFusionModel, self).__init__()
        
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size 
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256)
        )
        self.shape_cnn = nn.Sequential(
            nn.Conv1d(28, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveMaxPool1d(1) 
        )
        self.shape_fc = nn.Sequential(nn.Linear(128, 512), nn.LayerNorm(512), nn.GELU())

        self.norm_dna = nn.LayerNorm(768)
        self.norm_epi = nn.LayerNorm(768)
        self.gate_network = nn.Sequential(
            nn.Linear(768 * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 2), nn.Sigmoid() 
        )
        
        self.classification_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.regression_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))        

    def forward(self, tab, tab_mask, input_ids, attention_mask, shape, shape_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        hidden_states_t = hidden_states.permute(0, 2, 1)
        spatial_features = F.relu(self.spatial_conv(hidden_states_t)).permute(0, 2, 1)
        attn_weights = self.attention_pool(spatial_features).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=-1)
        dna_embeddings = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)

        tab_in = torch.cat([tab, tab_mask], dim=1)
        tab_out = self.tab_mlp(tab_in) 
        shape_in = torch.cat([shape, shape_mask], dim=1)
        shape_out = self.shape_cnn(shape_in).squeeze(-1)
        shape_out = self.shape_fc(shape_out) 
        epi_embeddings = torch.cat((tab_out, shape_out), dim=1) 
        
        dna_norm = self.norm_dna(dna_embeddings)
        epi_norm = self.norm_epi(epi_embeddings)
        concat_features = torch.cat([dna_norm, epi_norm], dim=1)
        gates = self.gate_network(concat_features)
        
        gate_dna = gates[:, 0].unsqueeze(1) 
        gate_epi = gates[:, 1].unsqueeze(1) 
        fused_embeddings = (dna_norm * gate_dna) + (epi_norm * gate_epi) 
        
        class_logits = self.classification_head(fused_embeddings)
        m_value_pred = self.regression_head(fused_embeddings)
        
        return class_logits, m_value_pred, gate_dna, gate_epi, fused_embeddings

# =========================================
# 3. Execution Logic
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv_path", type=str, default="data/datafiles/test.csv")
    parser.add_argument("--test_shape_tsv", type=str, default="data/datafiles/test_3d_shapes.tsv")
    parser.add_argument("--wt_shape_tsv", type=str, default="data/datafiles/wt_3d_shapes.tsv")
    parser.add_argument("--mut_shape_tsv", type=str, default="data/datafiles/mut_3d_shapes.tsv")
    parser.add_argument("--weights_path", type=str, default="checkpoints_multimodal/best_weights.pth")
    parser.add_argument("--save_dir", type=str, default="results/multimodal_gated")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_window_size", type=int, default=1000)
    parser.add_argument("--shape_window_size", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info("[*] Loading Data...")
    df = pd.read_csv(args.test_csv_path)
    shapes = pd.read_csv(args.test_shape_tsv, sep='\t', header=None, dtype=np.float32).values
    wt_shapes = pd.read_csv(args.wt_shape_tsv, sep='\t', header=None, dtype=np.float32).values
    mut_shapes = pd.read_csv(args.mut_shape_tsv, sep='\t', header=None, dtype=np.float32).values
    
    mask = df['probeID'].str.startswith('cg')
    df = df[mask].reset_index(drop=True)
    shapes = shapes[mask.values]
    wt_shapes = wt_shapes[mask.values]
    mut_shapes = mut_shapes[mask.values]
    
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    test_dataset = MultiModalLateFusionDataset(df, shapes, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info("[*] Loading Model...")
    model = GatedFusionModel().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True), strict=True)
    model.eval()

    # --- Phase 1: Global Embeddings ---
    all_embeddings = []
    all_states = []

    logging.info("[*] Phase 1: Extracting Global Latent Representations...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Global Inference"):
            tab, tab_mask = batch['tab'].to(device), batch['tab_mask'].to(device)
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            shape, shape_mask = batch['shape'].to(device), batch['shape_mask'].to(device)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                _, _, _, _, fused_emb = model(tab, tab_mask, input_ids, attention_mask, shape, shape_mask)
                
            all_embeddings.append(fused_emb.cpu().float().numpy())
            all_states.extend(batch['binary_state'].numpy().flatten().tolist())

    X = np.vstack(all_embeddings)
    y = np.array(all_states)
    state_labels = ["Hypomethylated (<0.5 Beta)" if val == 0.0 else "Hypermethylated (>0.5 Beta)" for val in y]

    logging.info(f"[*] Fitting UMAP on {X.shape[0]} samples (Dimensions: {X.shape[1]} -> 2)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(X)

    # --- Phase 2: Top Drivers Trajectories ---
    logging.info("[*] Phase 2: Isolating Top Driver Mutations for Overlay...")
    target_data = []
    
    tabular_features = ['Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 'Ref_H3K36me3_Signal', 'Ref_H3K4me1_Signal', 'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2']
    
    for idx, row in df.iterrows():
        mut_id = row['GDC_Genomic_DNA_Change']
        mut_pos, _, _ = parse_mutation_id(mut_id)
        if not mut_pos: continue
        
        wt_full = str(row['Healthy_5000bp_DNA']).upper()
        mut_full = str(row['Mutated_5000bp_DNA']).upper()
        wt_seq = wt_full[2000:3000]
        mut_seq = mut_full[2000:3000]
        
        if wt_seq == mut_seq: continue
        
        tab_raw = row[tabular_features].values.astype(np.float32)
        tab_t = torch.tensor(tab_raw).unsqueeze(0)
        tab_m = ~torch.isnan(tab_t)
        tab_t = torch.nan_to_num(tab_t, nan=0.0)
        
        wt_shape_t = torch.nan_to_num(torch.tensor(wt_shapes[idx]).view(1, 14, args.shape_window_size), nan=0.0)
        wt_shape_m = ~torch.isnan(wt_shape_t)
        mut_shape_t = torch.nan_to_num(torch.tensor(mut_shapes[idx]).view(1, 14, args.shape_window_size), nan=0.0)
        mut_shape_m = ~torch.isnan(mut_shape_t)
        
        target_data.append({
            'gene': str(row['Gene']) if pd.notna(row['Gene']) else f"Intergenic_{row['chr']}",
            'wt_seq': wt_seq, 'mut_seq': mut_seq,
            'tab': tab_t, 'tab_m': tab_m,
            'wt_shape': wt_shape_t, 'wt_shape_m': wt_shape_m,
            'mut_shape': mut_shape_t, 'mut_shape_m': mut_shape_m
        })
        
    logging.info(f"    -> Extracted {len(target_data)} mutated pairs. Scoring to find Top 5 trajectories...")
    
    trajectory_results = []
    with torch.no_grad():
        for item in target_data:
            wt_enc = tokenizer(item['wt_seq'], truncation=True, max_length=args.seq_window_size, padding='max_length', return_tensors='pt').to(device)
            mut_enc = tokenizer(item['mut_seq'], truncation=True, max_length=args.seq_window_size, padding='max_length', return_tensors='pt').to(device)
            
            tab, tab_m = item['tab'].to(device), item['tab_m'].to(device)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                _, wt_pred, _, _, wt_emb = model(tab, tab_m, wt_enc['input_ids'], wt_enc['attention_mask'], item['wt_shape'].to(device), item['wt_shape_m'].to(device))
                _, mut_pred, _, _, mut_emb = model(tab, tab_m, mut_enc['input_ids'], mut_enc['attention_mask'], item['mut_shape'].to(device), item['mut_shape_m'].to(device))
                
            wt_beta = m_value_to_beta(wt_pred.item())
            mut_beta = m_value_to_beta(mut_pred.item())
            
            trajectory_results.append({
                'gene': item['gene'],
                'delta': np.abs(mut_beta - wt_beta),
                'wt_emb': wt_emb.cpu().float().numpy(),
                'mut_emb': mut_emb.cpu().float().numpy()
            })
            
    # Sort and grab top 5
    trajectory_results.sort(key=lambda x: x['delta'], reverse=True)
    top_5 = trajectory_results[:5]

    # Map the top 5 embeddings into the UMAP space
    top_wt_embs = np.vstack([x['wt_emb'] for x in top_5])
    top_mut_embs = np.vstack([x['mut_emb'] for x in top_5])
    
    wt_2d = reducer.transform(top_wt_embs)
    mut_2d = reducer.transform(top_mut_embs)

    # --- Phase 3: Plotting ---
    logging.info("[*] Plotting Integrated Latent Space with Trajectories...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 8))
    
    # 1. Background Scatter (Faded)
    sns.scatterplot(
        x=embedding_2d[:, 0], 
        y=embedding_2d[:, 1], 
        hue=state_labels, 
        palette={"Hypomethylated (<0.5 Beta)": "#3b82f6", "Hypermethylated (>0.5 Beta)": "#ef4444"},
        alpha=0.15, # Faded alpha
        edgecolor=None,
        s=15,
        legend=True
    )
    
    # 2. Overlay Trajectories
    for i, item in enumerate(top_5):
        # Draw Arrow
        plt.annotate(
            '', xy=(mut_2d[i, 0], mut_2d[i, 1]), xytext=(wt_2d[i, 0], wt_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color='black', lw=2, shrinkA=0, shrinkB=0)
        )
        # Highlight Start and End points
        plt.scatter(wt_2d[i, 0], wt_2d[i, 1], color='#22c55e', s=80, zorder=5, edgecolor='black', marker='o', label='Wild-Type State' if i==0 else "")
        plt.scatter(mut_2d[i, 0], mut_2d[i, 1], color='#ef4444', s=120, zorder=5, edgecolor='black', marker='*', label='Mutated State' if i==0 else "")
        
        # Add Gene label slightly offset
        plt.text(mut_2d[i, 0] + 0.2, mut_2d[i, 1] + 0.2, item['gene'], fontsize=10, fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.title('Gated Fusion Latent Space: Global Manifold & Mutation Trajectories', fontweight='bold', fontsize=15)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Fix legend placement