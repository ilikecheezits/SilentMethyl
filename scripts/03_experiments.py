from __future__ import annotations

import json
import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from matched_synonymous_null import (
    PROTECTED_CPG_INDICES,
    annotate_variant,
    compute_matched_null_statistics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TEST_CSV_PATH = "data/datafiles/testing_data.csv"
MODEL_WEIGHTS = "checkpoints_baseline/best_weights.pth"
BASE_DIR = "results/baseline_matched_null"
WINDOW_SIZE = 1000
TOP_K = 10
MIN_MATCHED_CONTROLS = 20
MAX_MATCHED_CONTROLS = 1000
RANDOM_SEED = 42
INFERENCE_BATCH_SIZE = 32

os.makedirs(BASE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("[*] Using device: %s", DEVICE)


# =============================================================================
# 2. ARCHITECTURE DEFINITIONS
# =============================================================================
def patch_and_load_dnabert(
    model_path: str = "zhihan1996/DNABERT-2-117M",
    local_dir: str = "./dnabert2_local",
):
    logging.info("--- Performing DNABERT-2 Surgery & Patching ---")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        from huggingface_hub import snapshot_download

        cache_path = snapshot_download(model_path)
        for item in os.listdir(cache_path):
            src = os.path.join(cache_path, item)
            dst = os.path.join(local_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        triton_file = os.path.join(local_dir, "flash_attn_triton.py")
        if os.path.exists(triton_file):
            with open(triton_file, "w", encoding="utf-8") as handle:
                handle.write("def __getattr__(name):\n    return None\n")

        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        config_data["use_flash_attn"] = False
        if config_data.get("pad_token_id") is None:
            config_data["pad_token_id"] = 0
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config_data, handle)

    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    config.output_attentions = True
    base_model = AutoModel.from_config(config, trust_remote_code=True)
    return config, base_model


class BaselineDNABert(nn.Module):
    def __init__(self, model_path: str = "zhihan1996/DNABERT-2-117M"):
        super().__init__()
        self.config, self.bert = patch_and_load_dnabert(model_path)
        hidden_size = self.config.hidden_size
        self.spatial_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        spatial_features = F.relu(
            self.spatial_conv(hidden_states.permute(0, 2, 1))
        ).permute(0, 2, 1)
        attention_scores = self.attention_pool(spatial_features).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e4)
        attention_weights = F.softmax(attention_scores, dim=-1)
        pooled_output = torch.sum(
            spatial_features * attention_weights.unsqueeze(-1), dim=1
        )
        return (
            self.classification_head(pooled_output),
            self.regression_head(pooled_output),
        )


# =============================================================================
# 3. INFERENCE AND DATA HELPERS
# =============================================================================
def m_value_to_beta(m_values: np.ndarray) -> np.ndarray:
    clipped = np.clip(m_values, -20, 20)
    powers = np.power(2.0, clipped)
    return powers / (1.0 + powers)


@torch.no_grad()
def batch_inference(model, tokenizer, sequences: list[str]) -> np.ndarray:
    predictions: list[float] = []
    for start in range(0, len(sequences), INFERENCE_BATCH_SIZE):
        batch = sequences[start:start + INFERENCE_BATCH_SIZE]
        encodings = tokenizer(
            batch,
            truncation=True,
            max_length=WINDOW_SIZE,
            padding="max_length",
            return_tensors="pt",
        ).to(DEVICE)
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            _, m_predictions = model(
                encodings["input_ids"], encodings["attention_mask"]
            )
        predictions.extend(m_predictions.cpu().float().flatten().tolist())
    return np.asarray(predictions, dtype=float)


def build_unique_variant_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate centered inputs and retain one row per variant--CpG pair."""
    records: list[dict] = []
    wt_sequences: list[str] = []
    mutant_sequences: list[str] = []
    seen_uids: set[str] = set()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating sSNVs"):
        wt_full = str(row.get("Healthy_5000bp_DNA", "")).upper()
        mut_full = str(row.get("Mutated_5000bp_DNA", "")).upper()
        if len(wt_full) < 3000 or len(mut_full) < 3000:
            continue

        wt_sequence = wt_full[2000:3000]
        mutant_sequence = mut_full[2000:3000]

        try:
            metadata = annotate_variant(row, wt_sequence, mutant_sequence, WINDOW_SIZE)
        except ValueError as error:
            logging.warning(
                "[skip] %s | %s",
                row.get("GDC_Genomic_DNA_Change", "unknown mutation"),
                error,
            )
            continue

        if metadata["Mutation_Window_Index_0based"] in PROTECTED_CPG_INDICES:
            continue
        uid = metadata["Variant_UID"]
        if uid in seen_uids:
            continue
        seen_uids.add(uid)

        metadata["Source_Row_Index"] = int(row.name)
        records.append(metadata)
        wt_sequences.append(wt_sequence)
        mutant_sequences.append(mutant_sequence)

    return pd.DataFrame(records), wt_sequences, mutant_sequences


def export_top_control_relations(
    scored: pd.DataFrame,
    control_indices_by_uid: dict[str, np.ndarray],
    top_indices: np.ndarray,
    output_path: str,
) -> None:
    relations: list[dict] = []
    for rank, target_index in enumerate(top_indices, start=1):
        target = scored.iloc[target_index]
        control_indices = control_indices_by_uid[str(target["Variant_UID"])]
        for control_index in control_indices:
            control = scored.iloc[int(control_index)]
            relations.append(
                {
                    "Target_Rank": rank,
                    "Target_Variant_UID": target["Variant_UID"],
                    "Target_Gene": target["Gene"],
                    "Target_Delta_Beta": target["Predicted_Delta_Beta"],
                    "Matching_Tier": target["Matched_Null_Tier"],
                    "Control_Variant_UID": control["Variant_UID"],
                    "Control_Gene": control["Gene"],
                    "Control_Delta_Beta": control["Predicted_Delta_Beta"],
                    "Control_SBS96": control["Canonical_SBS96"],
                    "Control_CpG_Effect": control["CpG_Effect"],
                    "Control_Distance_From_CpG": control[
                        "Absolute_Distance_From_Target_CpG"
                    ],
                }
            )
    pd.DataFrame(relations).to_csv(output_path, index=False)


def plot_matched_null(
    target: pd.Series,
    control_deltas: np.ndarray,
    rank: int,
    output_path: str,
) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        control_deltas,
        bins=min(50, max(10, int(np.sqrt(len(control_deltas)) * 2))),
        kde=len(control_deltas) >= 20,
        color="#94a3b8",
        edgecolor="black",
        label="Matched observed synonymous controls",
    )
    plt.axvline(0, color="black", linewidth=1.3)
    observed = float(target["Predicted_Delta_Beta"])
    plt.axvline(
        observed,
        color="#ef4444" if observed > 0 else "#22c55e",
        linewidth=3,
        label=f"Observed somatic sSNV (Δβ={observed:.4f})",
    )
    stats_text = (
        f"Controls: {int(target['Matched_Control_Count'])}\n"
        f"Tier: {target['Matched_Null_Tier']}\n"
        f"Percentile: {target['Matched_Null_Absolute_Effect_Percentile']:.1f}\n"
        f"Empirical p: {target['Matched_Empirical_P']:.4g}\n"
        f"BH q: {target['Matched_BH_Q']:.4g}"
    )
    plt.gca().text(
        0.03,
        0.97,
        stats_text,
        transform=plt.gca().transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    plt.title(
        f"Baseline matched synonymous null — rank {rank}\n"
        f"{target['Gene']} | {target['GDC_Genomic_DNA_Change']} | "
        f"probe {target['Target_CpG']}"
    )
    plt.xlabel("Predicted methylation sensitivity (Δβ)")
    plt.ylabel("Matched control count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    logging.info("[*] Initializing baseline model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    model = BaselineDNABert().to(DEVICE)

    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")
    model.load_state_dict(
        torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True),
        strict=True,
    )
    model.eval()

    df = pd.read_csv(TEST_CSV_PATH)
    if "probeID" in df.columns:
        df = df[df["probeID"].astype(str).str.startswith("cg")].reset_index(drop=True)

    metadata, wt_sequences, mutant_sequences = build_unique_variant_cohort(df)
    if metadata.empty:
        raise RuntimeError("No valid centered single-nucleotide variant pairs were found.")
    logging.info(
        "[*] Retained %d unique synonymous variant--CpG pairs.", len(metadata)
    )

    wt_m_values = batch_inference(model, tokenizer, wt_sequences)
    mutant_m_values = batch_inference(model, tokenizer, mutant_sequences)
    wt_betas = m_value_to_beta(wt_m_values)
    mutant_betas = m_value_to_beta(mutant_m_values)

    metadata["Predicted_WT_M"] = wt_m_values
    metadata["Predicted_Mutant_M"] = mutant_m_values
    metadata["Predicted_WT_Beta"] = wt_betas
    metadata["Predicted_Mutant_Beta"] = mutant_betas
    metadata["Predicted_Delta_Beta"] = mutant_betas - wt_betas
    metadata["Absolute_Delta_Beta"] = metadata["Predicted_Delta_Beta"].abs()

    scored, control_indices_by_uid = compute_matched_null_statistics(
        metadata,
        delta_column="Predicted_Delta_Beta",
        min_controls=MIN_MATCHED_CONTROLS,
        max_controls=MAX_MATCHED_CONTROLS,
        random_seed=RANDOM_SEED,
    )
    scored["Absolute_Delta_Beta_Rank"] = scored["Absolute_Delta_Beta"].rank(
        ascending=False, method="min"
    ).astype(int)

    ranked_indices = np.argsort(
        scored["Absolute_Delta_Beta"].to_numpy(dtype=float)
    )[::-1]
    top_count = min(TOP_K, len(scored))
    top_indices = ranked_indices[:top_count]

    scored_sorted = scored.iloc[ranked_indices].reset_index(drop=True)
    all_results_path = os.path.join(BASE_DIR, "all_variant_matched_null_statistics.csv")
    scored_sorted.to_csv(all_results_path, index=False)
    logging.info("[✓] Saved full-cohort matched-null results to %s", all_results_path)

    top_results = scored.iloc[top_indices].copy().reset_index(drop=True)
    top_results.to_csv(
        os.path.join(BASE_DIR, "top_variant_matched_null_statistics.csv"), index=False
    )

    export_top_control_relations(
        scored,
        control_indices_by_uid,
        top_indices,
        os.path.join(BASE_DIR, "top_variant_matched_controls_long.csv"),
    )

    for rank, target_index in enumerate(top_indices[:3], start=1):
        target = scored.iloc[int(target_index)]
        control_indices = control_indices_by_uid[str(target["Variant_UID"])]
        if len(control_indices) == 0:
            continue
        control_deltas = scored.iloc[control_indices]["Predicted_Delta_Beta"].to_numpy()
        safe_gene = str(target["Gene"]).replace("/", "_").replace(" ", "_")
        plot_matched_null(
            target,
            control_deltas,
            rank,
            os.path.join(BASE_DIR, f"matched_null_top{rank}_{safe_gene}.png"),
        )

    logging.info(
        "[✓] Finished. The centered CpG at indices 499--500 was protected in "
        "both observed and control comparisons."
    )


if __name__ == "__main__":
    main()
