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

    local_model_dir = "/content/DNABERT-2-Fixed"
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
            f.write("def __getattr__(name):")
    return None

    config_path = os.path.join(local_model_dir, "config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)

    config_data["pad_token_id"] = 0
    config_data["use_flash_attn"] = False

    with open(config_path, "w") as f:
        json.dump(config_data, f)
        
    return local_model_dir

class SilentMethylModel(nn.Module):
    def __init__(self, patched_config, region_vocab_size, island_vocab_size, meta_dim=19):
        super().__init__()
        self.bert = AutoModel.from_config(patched_config, trust_remote_code=True)
        self.seq_out_dim = self.bert.config.hidden_size

        self.reg_emb = nn.Embedding(num_embeddings=region_vocab_size, embedding_dim=8)
        self.isl_emb = nn.Embedding(num_embeddings=island_vocab_size, embedding_dim=8)
        self.tata_emb = nn.Embedding(num_embeddings=2, embedding_dim=4)

        self.meta_out_dim = meta_dim + 8 + 8 + 4
        self.seq_proj = nn.Linear(self.seq_out_dim, 128)
        self.meta_proj = nn.Sequential(
            nn.Linear(self.meta_out_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )

        self.classifier = nn.Linear(128, 1)
        self.regressor = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            seq_emb = outputs[0][:, 0, :]
        else:
            seq_emb = outputs.last_hidden_state[:, 0, :]

        seq_proj = self.seq_proj(seq_emb)
        reg_embedded = self.reg_emb(reg_idx)
        isl_embedded = self.isl_emb(isl_idx)
        tata_embedded = self.tata_emb(tata_idx)

        meta_concat = torch.cat([num_feats, reg_embedded, isl_embedded, tata_embedded], dim=1)
        meta_gate = torch.sigmoid(self.meta_proj(meta_concat))

        fused = seq_proj + (seq_proj * meta_gate)

        class_logits = self.classifier(fused)
        reg_logits = self.regressor(fused)

        return class_logits, reg_logits
