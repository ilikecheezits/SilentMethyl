from __future__ import annotations

import os
import re
import json
import shutil
import math
import random
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoConfig, AutoModel

warnings_filter = logging.getLogger("transformers")
warnings_filter.setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TEST_CSV_PATH = "data/datafiles/testing_data.csv"
MODEL_WEIGHTS = "checkpoints_seq_epi_fusion/best_weights.pth"
BASE_DIR = "results/seq_epi_stability"

SEQ_WINDOW_SIZE = 1000
MC_PASSES = 50
BATCH_SIZE = 32

TABULAR_FEATURES = [
    "Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal", 
    "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal", 
    "Ref_H3K4me1_Signal", "Target_Base_PhyloP_100way_1", "Target_Base_PhyloP_100way_2"
]

os.makedirs(BASE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"[*] Using device: {DEVICE}")


# =============================================================================
# 2. ARCHITECTURE DEFINITION
# =============================================================================
def patch_and_load_dnabert(model_path="zhihan1996/DNABERT-2-117M", local_dir="./dnabert2_local_inference"):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        from huggingface_hub import snapshot_download
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


class SequenceEpiFusionModel(nn.Module):
    def __init__(self, model_path="zhihan1996/DNABERT-2-117M", tabular_dim=9):
        super().__init__()
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size 
        
        self.spatial_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        
        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256)
        )
        self.epi_proj = nn.Linear(256, 768)
        
        self.norm_dna = nn.LayerNorm(768)
        self.norm_epi = nn.LayerNorm(768)
        self.gate_network = nn.Sequential(
            nn.Linear(1536, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 2), nn.Sigmoid()
        )
        
        self.classification_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.regression_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))        

    def forward(self, tab, tab_mask, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        spatial_features = F.relu(self.spatial_conv(hidden_states.permute(0, 2, 1))).permute(0, 2, 1)
        
        attn_scores = self.attention_pool(spatial_features).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        dna_embeddings = torch.sum(spatial_features * attn_weights.unsqueeze(-1), dim=1)

        tab_out = self.tab_mlp(torch.cat([tab, tab_mask], dim=1))
        epi_embeddings = self.epi_proj(tab_out) 
        
        dna_norm = self.norm_dna(dna_embeddings)
        epi_norm = self.norm_epi(epi_embeddings)
        
        gates = self.gate_network(torch.cat([dna_norm, epi_norm], dim=1))
        gate_dna, gate_epi = gates[:, 0].unsqueeze(1), gates[:, 1].unsqueeze(1)
        
        fused_embeddings = (dna_norm * gate_dna) + (epi_norm * gate_epi)
        m_value_pred = self.regression_head(fused_embeddings)
        
        return None, m_value_pred, gate_dna, gate_epi


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
def m_to_beta(m_value: torch.Tensor) -> torch.Tensor:
    """Stable transformation from M-value to Beta-value."""
    return torch.sigmoid(m_value * math.log(2.0))

def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Activates ONLY dropout layers to preserve eval mode (e.g. LayerNorm)."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def set_inference_seed(seed: int) -> None:
    """Locks down randomness for paired forward passes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def absolute_effect_ranks(delta: np.ndarray) -> np.ndarray:
    """Returns 1-based ranks prioritizing the largest absolute effects."""
    order = np.argsort(-np.abs(delta))
    ranks = np.empty(len(delta), dtype=np.int64)
    ranks[order] = np.arange(1, len(delta) + 1)
    return ranks

RC_TABLE = str.maketrans("ACGTN", "TGCAN")
def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC_TABLE)[::-1]

# =============================================================================
# 4. PAIRED FORWARD PASSES
# =============================================================================
@torch.inference_mode()
def deterministic_forward(model, wt_ids, wt_mask, mut_ids, mut_mask, tab, tab_m):
    """Clean inference pass with no dropout."""
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        _, wt_m, wt_g_seq, wt_g_epi = model(tab, tab_m, wt_ids, wt_mask)
        _, mut_m, mut_g_seq, mut_g_epi = model(tab, tab_m, mut_ids, mut_mask)
        
    wt_beta = m_to_beta(wt_m)
    mut_beta = m_to_beta(mut_m)
    return wt_beta, mut_beta, wt_g_seq, wt_g_epi, mut_g_seq, mut_g_epi


@torch.inference_mode()
def paired_mc_forward(model, wt_ids, wt_mask, mut_ids, mut_mask, tab, tab_m, seed: int):
    """Enforces identical stochastic dropout masks for WT and Mutant."""
    set_inference_seed(seed)
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        _, wt_m, wt_g_seq, wt_g_epi = model(tab, tab_m, wt_ids, wt_mask)

    set_inference_seed(seed) # Reset seed to replicate exact dropout masks
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        _, mut_m, mut_g_seq, mut_g_epi = model(tab, tab_m, mut_ids, mut_mask)

    wt_beta = m_to_beta(wt_m)
    mut_beta = m_to_beta(mut_m)
    
    return {
        "delta_beta": (mut_beta - wt_beta).cpu().float().numpy().flatten(),
        "wt_gate_seq": wt_g_seq.cpu().float().numpy().flatten(),
        "wt_gate_epi": wt_g_epi.cpu().float().numpy().flatten(),
        "mut_gate_seq": mut_g_seq.cpu().float().numpy().flatten(),
        "mut_gate_epi": mut_g_epi.cpu().float().numpy().flatten(),
    }


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================
def main():
    logging.info("[*] Initializing Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = SequenceEpiFusionModel().to(DEVICE)
    
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True), strict=True)

    # --- A. DATA PREPARATION ---
    logging.info("[*] Loading Cohort and computing pre-tokenizations...")
    df = pd.read_csv(TEST_CSV_PATH)
    df = df[df['probeID'].astype(str).str.startswith('cg')].reset_index(drop=True)
    
    # Persistent Identifiers
    df["Variant_UID"] = df["GDC_Genomic_DNA_Change"].astype(str) + "|" + df["probeID"].astype(str)
    
    wt_seqs, mut_seqs = [], []
    tabs, tab_masks = [], []
    
    for idx, row in df.iterrows():
        wt_full = str(row.get("Healthy_5000bp_DNA", "")).upper()
        mut_full = str(row.get("Mutated_5000bp_DNA", "")).upper()
        wt_seqs.append(wt_full[2000:3000])
        mut_seqs.append(mut_full[2000:3000])
        
        tab_raw = row[TABULAR_FEATURES].to_numpy(dtype=np.float32)
        tab_t = torch.tensor(tab_raw, dtype=torch.float32)
        tab_m = ~torch.isnan(tab_t)
        tabs.append(torch.nan_to_num(tab_t, nan=0.0))
        tab_masks.append(tab_m.float())
        
    tabs = torch.stack(tabs)
    tab_masks = torch.stack(tab_masks)
    
    # Pretokenize Once (Store on CPU to save VRAM)
    wt_encoded = tokenizer(wt_seqs, truncation=True, max_length=1000, padding="max_length", return_tensors="pt")
    mut_encoded = tokenizer(mut_seqs, truncation=True, max_length=1000, padding="max_length", return_tensors="pt")

    n_samples = len(df)
    logging.info(f"[*] Pre-tokenized {n_samples} full-cohort pairs.")

    # --- B. DETERMINISTIC BASELINE ---
    logging.info("[*] Running Deterministic Baseline Sweep...")
    model.eval()
    
    det_deltas, det_wt_g_seq, det_wt_g_epi, det_mut_g_seq, det_mut_g_epi = [], [], [], [], []
    
    for i in tqdm(range(0, n_samples, BATCH_SIZE), desc="Deterministic"):
        b_wt_ids = wt_encoded['input_ids'][i:i+BATCH_SIZE].to(DEVICE)
        b_wt_mask = wt_encoded['attention_mask'][i:i+BATCH_SIZE].to(DEVICE)
        b_mut_ids = mut_encoded['input_ids'][i:i+BATCH_SIZE].to(DEVICE)
        b_mut_mask = mut_encoded['attention_mask'][i:i+BATCH_SIZE].to(DEVICE)
        b_tab = tabs[i:i+BATCH_SIZE].to(DEVICE)
        b_tab_m = tab_masks[i:i+BATCH_SIZE].to(DEVICE)
        
        wt_b, mut_b, w_gs, w_ge, m_gs, m_ge = deterministic_forward(
            model, b_wt_ids, b_wt_mask, b_mut_ids, b_mut_mask, b_tab, b_tab_m
        )
        
        det_deltas.extend((mut_b - wt_b).cpu().float().numpy().flatten())
        det_wt_g_seq.extend(w_gs.cpu().float().numpy().flatten())
        det_wt_g_epi.extend(w_ge.cpu().float().numpy().flatten())
        det_mut_g_seq.extend(m_gs.cpu().float().numpy().flatten())
        det_mut_g_epi.extend(m_ge.cpu().float().numpy().flatten())
        
    det_deltas = np.array(det_deltas)
    det_ranks = absolute_effect_ranks(det_deltas)
    det_abs = np.abs(det_deltas)
    det_top10 = set(np.argsort(-det_abs)[:10])

    # --- C. REVERSE-COMPLEMENT DIAGNOSTIC ---
    logging.info("[*] Running Reverse-Complement Diagnostic Pass...")
    rc_wt_seqs = [reverse_complement(s) for s in wt_seqs]
    rc_mut_seqs = [reverse_complement(s) for s in mut_seqs]
    rc_wt_encoded = tokenizer(rc_wt_seqs, truncation=True, max_length=1000, padding="max_length", return_tensors="pt")
    rc_mut_encoded = tokenizer(rc_mut_seqs, truncation=True, max_length=1000, padding="max_length", return_tensors="pt")
    
    rc_deltas = []
    for i in tqdm(range(0, n_samples, BATCH_SIZE), desc="RC Sweep"):
        b_wt_ids = rc_wt_encoded['input_ids'][i:i+BATCH_SIZE].to(DEVICE)
        b_wt_mask = rc_wt_encoded['attention_mask'][i:i+BATCH_SIZE].to(DEVICE)
        b_mut_ids = rc_mut_encoded['input_ids'][i:i+BATCH_SIZE].to(DEVICE)
        b_mut_mask = rc_mut_encoded['attention_mask'][i:i+BATCH_SIZE].to(DEVICE)
        b_tab = tabs[i:i+BATCH_SIZE].to(DEVICE)
        b_tab_m = tab_masks[i:i+BATCH_SIZE].to(DEVICE)
        
        wt_b, mut_b, _, _, _, _ = deterministic_forward(model, b_wt_ids, b_wt_mask, b_mut_ids, b_mut_mask, b_tab, b_tab_m)
        rc_deltas.extend((mut_b - wt_b).cpu().float().numpy().flatten())
        
    rc_deltas = np.array(rc_deltas)
    rc_ranks = absolute_effect_ranks(rc_deltas)
    rc_rho = spearmanr(det_abs, np.abs(rc_deltas)).statistic
    rc_top10 = set(np.argsort(-np.abs(rc_deltas))[:10])
    
    rc_df = pd.DataFrame({
        "Variant_UID": df["Variant_UID"],
        "Gene": df["Gene"],
        "Deterministic_Delta": det_deltas,
        "RC_Delta": rc_deltas,
        "Deterministic_Rank": det_ranks,
        "RC_Rank": rc_ranks
    })
    rc_df.to_csv(os.path.join(BASE_DIR, "reverse_complement_consistency.csv"), index=False)
    logging.info(f"    -> RC Spearman Rho: {rc_rho:.4f} | Top-10 Overlap: {len(det_top10 & rc_top10)/10.0:.2%}")

    # --- D. 50 MONTE CARLO DROPOUT PASSES ---
    logging.info(f"[*] Starting {MC_PASSES} Monte Carlo Dropout Passes...")
    enable_mc_dropout(model)
    
    mc_deltas = np.zeros((MC_PASSES, n_samples))
    mc_wt_g_seq = np.zeros((MC_PASSES, n_samples))
    mc_wt_g_epi = np.zeros((MC_PASSES, n_samples))
    mc_mut_g_seq = np.zeros((MC_PASSES, n_samples))
    mc_mut_g_epi = np.zeros((MC_PASSES, n_samples))
    mc_ranks = np.zeros((MC_PASSES, n_samples), dtype=np.int64)
    
    pass_summary_rows = []

    for pass_idx in range(MC_PASSES):
        pass_d, pass_wgs, pass_wge, pass_mgs, pass_mge = [], [], [], [], []
        
        for batch_idx, i in enumerate(range(0, n_samples, BATCH_SIZE)):
            b_wt_ids = wt_encoded['input_ids'][i:i+BATCH_SIZE].to(DEVICE)
            b_wt_mask = wt_encoded['attention_mask'][i:i+BATCH_SIZE].to(DEVICE)
            b_mut_ids = mut_encoded['input_ids'][i:i+BATCH_SIZE].to(DEVICE)
            b_mut_mask = mut_encoded['attention_mask'][i:i+BATCH_SIZE].to(DEVICE)
            b_tab = tabs[i:i+BATCH_SIZE].to(DEVICE)
            b_tab_m = tab_masks[i:i+BATCH_SIZE].to(DEVICE)
            
            # Unique seed per pass AND batch to guarantee independent noise
            batch_seed = 10_000 + pass_idx * 100_000 + batch_idx
            
            res = paired_mc_forward(
                model, b_wt_ids, b_wt_mask, b_mut_ids, b_mut_mask, b_tab, b_tab_m, batch_seed
            )
            
            pass_d.extend(res["delta_beta"])
            pass_wgs.extend(res["wt_gate_seq"])
            pass_wge.extend(res["wt_gate_epi"])
            pass_mgs.extend(res["mut_gate_seq"])
            pass_mge.extend(res["mut_gate_epi"])

        mc_deltas[pass_idx] = np.array(pass_d)
        mc_wt_g_seq[pass_idx] = np.array(pass_wgs)
        mc_wt_g_epi[pass_idx] = np.array(pass_wge)
        mc_mut_g_seq[pass_idx] = np.array(pass_mgs)
        mc_mut_g_epi[pass_idx] = np.array(pass_mge)
        
        # Rank the current pass
        pass_ranks = absolute_effect_ranks(mc_deltas[pass_idx])
        mc_ranks[pass_idx] = pass_ranks
        
        # Cohort Level Stability Math
        pass_abs = np.abs(mc_deltas[pass_idx])
        rho = spearmanr(det_abs, pass_abs).statistic
        pass_top10 = set(np.argsort(-pass_abs)[:10])
        top10_overlap = len(det_top10 & pass_top10) / 10.0
        
        pass_summary_rows.append({
            "Pass": pass_idx + 1,
            "Spearman_Absolute_Delta": rho,
            "Top10_Overlap": top10_overlap,
        })
        
        if (pass_idx + 1) % 10 == 0:
            logging.info(f"    -> Completed {pass_idx + 1}/{MC_PASSES} passes. " 
                         f"(Latest Rho: {rho:.3f}, Top10 Overlap: {top10_overlap:.1f})")

    # --- E. STATISTICAL AGGREGATION & EXPORT ---
    logging.info("[*] Aggregating Stochastic Posteriors & Normalizing Gates...")
    
    np.savez_compressed(
        os.path.join(BASE_DIR, "mc_dropout_raw_predictions.npz"),
        delta_beta=mc_deltas,
        absolute_ranks=mc_ranks,
        wt_gate_seq=mc_wt_g_seq,
        wt_gate_epi=mc_wt_g_epi,
        mut_gate_seq=mc_mut_g_seq,
        mut_gate_epi=mc_mut_g_epi,
    )
    pd.DataFrame(pass_summary_rows).to_csv(os.path.join(BASE_DIR, "mc_dropout_pass_summary.csv"), index=False)
    
    # 1. Delta Beta Stats
    delta_median = np.median(mc_deltas, axis=0)
    delta_q05 = np.quantile(mc_deltas, 0.05, axis=0)
    delta_q95 = np.quantile(mc_deltas, 0.95, axis=0)
    delta_std = np.std(mc_deltas, axis=0, ddof=1)
    
    det_sign = np.sign(det_deltas)
    sign_consistency = np.mean(np.sign(mc_deltas) == det_sign[None, :], axis=0)
    
    # 2. Rank Stats
    rank_median = np.median(mc_ranks, axis=0)
    rank_q05 = np.quantile(mc_ranks, 0.05, axis=0)
    rank_q95 = np.quantile(mc_ranks, 0.95, axis=0)
    top10_frequency = np.mean(mc_ranks <= 10, axis=0)
    top20_frequency = np.mean(mc_ranks <= 20, axis=0)

    # 3. Gate Stats (Normalized descriptive ratio)
    EPS = 1e-8
    wt_sequence_ratio = mc_wt_g_seq / (mc_wt_g_seq + mc_wt_g_epi + EPS)
    mut_sequence_ratio = mc_mut_g_seq / (mc_mut_g_seq + mc_mut_g_epi + EPS)

    df_results = pd.DataFrame({
        "Variant_UID": df["Variant_UID"],
        "Gene": df["Gene"],
        "Deterministic_Delta_Beta": det_deltas,
        "Deterministic_Absolute_Rank": det_ranks,
        "MC_Delta_Median": delta_median,
        "MC_Delta_P05": delta_q05,
        "MC_Delta_P95": delta_q95,
        "MC_Delta_STD": delta_std,
        "Sign_Consistency": sign_consistency,
        "MC_Rank_Median": rank_median,
        "MC_Rank_P05": rank_q05,
        "MC_Rank_P95": rank_q95,
        "Top10_Frequency": top10_frequency,
        "Top20_Frequency": top20_frequency,
        "WT_Sequence_Gate_Median": np.median(mc_wt_g_seq, axis=0),
        "WT_Epigenomic_Gate_Median": np.median(mc_wt_g_epi, axis=0),
        "WT_Sequence_Ratio_Median": np.median(wt_sequence_ratio, axis=0),
        "Mutant_Sequence_Ratio_Median": np.median(mut_sequence_ratio, axis=0),
    })

    # Sort final CSV deterministically by absolute baseline delta
    df_results = df_results.sort_values(by="Deterministic_Absolute_Rank").reset_index(drop=True)
    df_results.to_csv(os.path.join(BASE_DIR, "mc_dropout_variant_stability.csv"), index=False)
    
    logging.info("[✓] Experiment Complete. Outputs securely written to: " + BASE_DIR)

if __name__ == "__main__":
    main()