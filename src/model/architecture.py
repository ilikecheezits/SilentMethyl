import os
import shutil
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from huggingface_hub import snapshot_download

def perform_triton_surgery():
    """
    Downloads the DNABERT-2 model, patches it to disable Triton, and returns the path to the patched model.
    """
    print("[*] 1. Downloading raw files locally to perform surgery...")
    model_path = "zhihan1996/DNABERT-2-117M"
    model_cache_path = snapshot_download(model_path)

    local_model_dir = "./dnabert2_local"
    if os.path.exists(local_model_dir):
        shutil.rmtree(local_model_dir)
    os.makedirs(local_model_dir, exist_ok=True)

    for item in os.listdir(model_cache_path):
        src = os.path.join(model_cache_path, item)
        dst = os.path.join(local_model_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print("[*] 2. Neutralizing Triton Flash Attention & Patching Config...")
    triton_file = os.path.join(local_model_dir, "flash_attn_triton.py")
    if os.path.exists(triton_file):
        with open(triton_file, "w") as f:
            f.write("def __getattr__(name):\n    return None\n")

    config_path = os.path.join(local_model_dir, "config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)

    config_data["pad_token_id"] = 0
    config_data["use_flash_attn"] = False

    with open(config_path, "w") as f:
        json.dump(config_data, f)
        
    return local_model_dir

class SilentMethylModel(nn.Module):
    def __init__(self, patched_config, region_vocab_size, island_vocab_size, meta_dim=20):
        super().__init__()
        
        # 1. DNA BRANCH (The Residual)
        self.bert = AutoModel.from_config(patched_config, trust_remote_code=True)
        self.seq_out_dim = self.bert.config.hidden_size
        self.dna_head = nn.Sequential(
            nn.Linear(self.seq_out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1) 
        )
        
        # 2. METADATA BRANCH (The Global Mean)
        self.reg_emb = nn.Embedding(region_vocab_size, 16)
        self.isl_emb = nn.Embedding(island_vocab_size, 16)
        self.tata_emb = nn.Embedding(2, 4)
        
        self.meta_total_dim = meta_dim + 16 + 16 + 4
        self.meta_head = nn.Sequential(
            nn.Linear(self.meta_total_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        # 3. LEARNABLE SCALERS
        self.dna_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.meta_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx, dna_only=False):
        # A. Sequence Processing with Mean Pooling
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling is mathematically better for finding motifs anywhere in the 1kb
        seq_emb = torch.topk(outputs[0], k=3, dim=1).values.mean(dim=1)
        dna_logits = self.dna_head(seq_emb)

        # WARMUP TRAP: If in warmup, return DNA predictions immediately
        if dna_only:
            return dna_logits, dna_logits

        # B. Metadata Processing
        reg_e = self.reg_emb(reg_idx)
        isl_e = self.isl_emb(isl_idx)
        tata_e = self.tata_emb(tata_idx)
        meta_input = torch.cat([num_feats, reg_e, isl_e, tata_e], dim=1)
        meta_logits = self.meta_head(meta_input)

        # C. ADDITIVE DECOMPOSITION
        combined = (self.dna_weight * dna_logits) + (self.meta_weight * meta_logits)
        return combined, combined
