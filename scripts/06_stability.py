from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

DEFAULT_TEST_CSV_PATH = "data/datafiles/testing_data.csv"
DEFAULT_MODEL_WEIGHTS = "checkpoints_seq_epi_fusion/best_weights.pth"
DEFAULT_BASE_DIR = "results/seq_epi_stability"

SEQ_WINDOW_SIZE = 1000
CENTER_C_INDEX = 499
CENTER_G_INDEX = 500
PROTECTED_CPG_INDICES = {CENTER_C_INDEX, CENTER_G_INDEX}
DNA_BASES = set("ACGT")

TABULAR_FEATURES = [
    "Ref_ATAC_Signal",
    "Ref_H3K4me3_Signal",
    "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal",
    "Ref_H3K9me3_Signal",
    "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal",
    "Target_Base_PhyloP_100way_1",
    "Target_Base_PhyloP_100way_2",
]

PHYLOP_1_INDEX = TABULAR_FEATURES.index("Target_Base_PhyloP_100way_1")
PHYLOP_2_INDEX = TABULAR_FEATURES.index("Target_Base_PhyloP_100way_2")

RC_TABLE = str.maketrans("ACGTN", "TGCAN")
BASE_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


# =============================================================================
# 1. MODEL
# =============================================================================
def patch_and_load_dnabert(
    model_path: str = "zhihan1996/DNABERT-2-117M",
    local_dir: str = "./dnabert2_local_inference",
):
    """Create the DNABERT-2 architecture used by the saved fusion checkpoint."""
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
    config.output_attentions = False
    base_model = AutoModel.from_config(config, trust_remote_code=True)
    return config, base_model


class SequenceEpiFusionModel(nn.Module):
    def __init__(
        self,
        model_path: str = "zhihan1996/DNABERT-2-117M",
        tabular_dim: int = 9,
    ):
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

        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
        )
        self.epi_proj = nn.Linear(256, 768)

        self.norm_dna = nn.LayerNorm(768)
        self.norm_epi = nn.LayerNorm(768)
        self.gate_network = nn.Sequential(
            nn.Linear(1536, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

        # Keep both heads so strict checkpoint loading matches training exactly.
        self.classification_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, tab, tab_mask, input_ids, attention_mask):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = (
            bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        )

        spatial_features = F.relu(
            self.spatial_conv(hidden_states.permute(0, 2, 1))
        ).permute(0, 2, 1)

        attn_scores = self.attention_pool(spatial_features).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        dna_embeddings = torch.sum(
            spatial_features * attn_weights.unsqueeze(-1),
            dim=1,
        )

        tab_out = self.tab_mlp(torch.cat([tab, tab_mask], dim=1))
        epi_embeddings = self.epi_proj(tab_out)

        dna_norm = self.norm_dna(dna_embeddings)
        epi_norm = self.norm_epi(epi_embeddings)

        gates = self.gate_network(torch.cat([dna_norm, epi_norm], dim=1))
        gate_dna = gates[:, 0].unsqueeze(1)
        gate_epi = gates[:, 1].unsqueeze(1)

        fused_embeddings = dna_norm * gate_dna + epi_norm * gate_epi
        class_logits = self.classification_head(fused_embeddings)
        m_value_pred = self.regression_head(fused_embeddings)

        return class_logits, m_value_pred, gate_dna, gate_epi


# =============================================================================
# 2. HELPERS
# =============================================================================
def m_to_beta(m_value: torch.Tensor) -> torch.Tensor:
    """Inverse M-value transform: beta = sigmoid(M * ln 2)."""
    return torch.sigmoid(m_value * math.log(2.0))


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable only Dropout modules while leaving the rest in evaluation mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def set_inference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def absolute_effect_ranks(delta: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.abs(delta))
    ranks = np.empty(len(delta), dtype=np.int64)
    ranks[order] = np.arange(1, len(delta) + 1)
    return ranks


def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC_TABLE)[::-1]


def single_difference(wt: str, mut: str) -> tuple[int, str, str]:
    if len(wt) != len(mut):
        raise ValueError("WT and mutant sequences have different lengths.")

    differences = [
        index
        for index, (wt_base, mut_base) in enumerate(zip(wt, mut))
        if wt_base != mut_base
    ]
    if len(differences) != 1:
        raise ValueError(
            f"Expected exactly one WT/mutant difference, found {len(differences)}."
        )

    index = differences[0]
    ref = wt[index]
    alt = mut[index]
    if ref not in DNA_BASES or alt not in DNA_BASES:
        raise ValueError(f"Non-ACGT mutation at index {index}: {ref}>{alt}.")

    return index, ref, alt


def build_valid_cohort(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], torch.Tensor, torch.Tensor]:
    """Reproduce the matched-null cohort eligibility rules.

    Requirements:
      * probe ID begins with cg
      * 1000-bp WT and mutant windows
      * intact centered CpG in both inputs
      * exactly one noncentral SNV
      * unique variant-probe UID
    """
    required_columns = {
        "probeID",
        "GDC_Genomic_DNA_Change",
        "Healthy_5000bp_DNA",
        "Mutated_5000bp_DNA",
        *TABULAR_FEATURES,
    }
    missing = sorted(required_columns - set(raw_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = raw_df[
        raw_df["probeID"].astype(str).str.startswith("cg")
    ].copy()

    records: list[dict] = []
    wt_sequences: list[str] = []
    mut_sequences: list[str] = []
    tab_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    seen_uids: set[str] = set()

    skipped = {
        "short_sequence": 0,
        "wrong_window_length": 0,
        "centered_cpg": 0,
        "not_single_snv": 0,
        "central_mutation": 0,
        "duplicate_uid": 0,
    }

    for source_index, row in tqdm(
        working.iterrows(),
        total=len(working),
        desc="Validating candidate cohort",
    ):
        wt_full = str(row["Healthy_5000bp_DNA"]).upper()
        mut_full = str(row["Mutated_5000bp_DNA"]).upper()

        if len(wt_full) < 3000 or len(mut_full) < 3000:
            skipped["short_sequence"] += 1
            continue

        wt_sequence = wt_full[2000:3000]
        mut_sequence = mut_full[2000:3000]

        if (
            len(wt_sequence) != SEQ_WINDOW_SIZE
            or len(mut_sequence) != SEQ_WINDOW_SIZE
        ):
            skipped["wrong_window_length"] += 1
            continue

        if (
            wt_sequence[CENTER_C_INDEX:CENTER_G_INDEX + 1] != "CG"
            or mut_sequence[CENTER_C_INDEX:CENTER_G_INDEX + 1] != "CG"
        ):
            skipped["centered_cpg"] += 1
            continue

        try:
            mutation_index, ref, alt = single_difference(
                wt_sequence,
                mut_sequence,
            )
        except ValueError:
            skipped["not_single_snv"] += 1
            continue

        if mutation_index in PROTECTED_CPG_INDICES:
            skipped["central_mutation"] += 1
            continue

        uid = (
            f"{row['GDC_Genomic_DNA_Change']}|"
            f"{row['probeID']}"
        )
        if uid in seen_uids:
            skipped["duplicate_uid"] += 1
            continue
        seen_uids.add(uid)

        tab_raw = row[TABULAR_FEATURES].to_numpy(dtype=np.float32)
        tab_tensor = torch.tensor(tab_raw, dtype=torch.float32)
        tab_mask = ~torch.isnan(tab_tensor)
        tab_tensor = torch.nan_to_num(tab_tensor, nan=0.0)

        record = row.to_dict()
        record.update(
            {
                "Variant_UID": uid,
                "Source_Row_Index": int(source_index),
                "Mutation_Window_Index_0based": int(mutation_index),
                "Ref": ref,
                "Alt": alt,
            }
        )

        records.append(record)
        wt_sequences.append(wt_sequence)
        mut_sequences.append(mut_sequence)
        tab_rows.append(tab_tensor)
        mask_rows.append(tab_mask.float())

    if not records:
        raise RuntimeError("No valid variant pairs remained after filtering.")

    cohort = pd.DataFrame(records).reset_index(drop=True)
    tabs = torch.stack(tab_rows)
    tab_masks = torch.stack(mask_rows)

    logging.info(
        "[+] Valid cohort: %d unique variant-probe pairs; skipped=%s",
        len(cohort),
        skipped,
    )

    return cohort, wt_sequences, mut_sequences, tabs, tab_masks


def build_rc_tabular(
    tabs: torch.Tensor,
    tab_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform ordered target-base features for reverse-complement input.

    Reverse complementation maps original sequence position 499 to 500 and
    original position 500 to 499. Therefore the two ordered target-base
    PhyloP values, and their missingness indicators, must be exchanged.
    """
    rc_tabs = tabs.clone()
    rc_masks = tab_masks.clone()

    phylo_1_values = tabs[:, PHYLOP_1_INDEX].clone()
    phylo_2_values = tabs[:, PHYLOP_2_INDEX].clone()
    rc_tabs[:, PHYLOP_1_INDEX] = phylo_2_values
    rc_tabs[:, PHYLOP_2_INDEX] = phylo_1_values

    phylo_1_masks = tab_masks[:, PHYLOP_1_INDEX].clone()
    phylo_2_masks = tab_masks[:, PHYLOP_2_INDEX].clone()
    rc_masks[:, PHYLOP_1_INDEX] = phylo_2_masks
    rc_masks[:, PHYLOP_2_INDEX] = phylo_1_masks

    return rc_tabs, rc_masks


def validate_reverse_complement_pairs(
    wt_sequences: list[str],
    mut_sequences: list[str],
    rc_wt_sequences: list[str],
    rc_mut_sequences: list[str],
) -> None:
    if not (
        len(wt_sequences)
        == len(mut_sequences)
        == len(rc_wt_sequences)
        == len(rc_mut_sequences)
    ):
        raise ValueError("Original and RC sequence collections are misaligned.")

    for index, (wt, mut, rc_wt, rc_mut) in enumerate(
        zip(
            wt_sequences,
            mut_sequences,
            rc_wt_sequences,
            rc_mut_sequences,
        )
    ):
        original_position, original_ref, original_alt = single_difference(wt, mut)
        rc_position, rc_ref, rc_alt = single_difference(rc_wt, rc_mut)

        if rc_wt[CENTER_C_INDEX:CENTER_G_INDEX + 1] != "CG":
            raise AssertionError(f"RC WT lost centered CpG at row {index}.")
        if rc_mut[CENTER_C_INDEX:CENTER_G_INDEX + 1] != "CG":
            raise AssertionError(f"RC mutant lost centered CpG at row {index}.")

        expected_rc_position = SEQ_WINDOW_SIZE - 1 - original_position
        if rc_position != expected_rc_position:
            raise AssertionError(
                f"RC mutation index mismatch at row {index}: "
                f"expected {expected_rc_position}, found {rc_position}."
            )

        if rc_ref != BASE_COMPLEMENT[original_ref]:
            raise AssertionError(
                f"RC reference allele mismatch at row {index}: "
                f"{original_ref} should become {BASE_COMPLEMENT[original_ref]}, "
                f"found {rc_ref}."
            )
        if rc_alt != BASE_COMPLEMENT[original_alt]:
            raise AssertionError(
                f"RC alternate allele mismatch at row {index}: "
                f"{original_alt} should become {BASE_COMPLEMENT[original_alt]}, "
                f"found {rc_alt}."
            )


def tokenize_sequences(tokenizer, sequences: list[str]):
    return tokenizer(
        sequences,
        truncation=True,
        max_length=SEQ_WINDOW_SIZE,
        padding="max_length",
        return_tensors="pt",
    )


def autocast_context(device: torch.device):
    return torch.amp.autocast(
        device_type=device.type,
        enabled=(device.type == "cuda"),
    )


# =============================================================================
# 3. INFERENCE
# =============================================================================
@torch.inference_mode()
def deterministic_forward(
    model,
    wt_ids,
    wt_mask,
    mut_ids,
    mut_mask,
    tab,
    tab_mask,
    device,
):
    with autocast_context(device):
        _, wt_m, wt_gate_seq, wt_gate_epi = model(
            tab,
            tab_mask,
            wt_ids,
            wt_mask,
        )
        _, mut_m, mut_gate_seq, mut_gate_epi = model(
            tab,
            tab_mask,
            mut_ids,
            mut_mask,
        )

    return {
        "wt_beta": m_to_beta(wt_m).cpu().float().numpy().flatten(),
        "mut_beta": m_to_beta(mut_m).cpu().float().numpy().flatten(),
        "wt_gate_seq": wt_gate_seq.cpu().float().numpy().flatten(),
        "wt_gate_epi": wt_gate_epi.cpu().float().numpy().flatten(),
        "mut_gate_seq": mut_gate_seq.cpu().float().numpy().flatten(),
        "mut_gate_epi": mut_gate_epi.cpu().float().numpy().flatten(),
    }


@torch.inference_mode()
def paired_mc_forward(
    model,
    wt_ids,
    wt_mask,
    mut_ids,
    mut_mask,
    tab,
    tab_mask,
    seed: int,
    device,
):
    # Resetting the seed gives the WT and mutant the same dropout realization.
    set_inference_seed(seed)
    with autocast_context(device):
        _, wt_m, wt_gate_seq, wt_gate_epi = model(
            tab,
            tab_mask,
            wt_ids,
            wt_mask,
        )

    set_inference_seed(seed)
    with autocast_context(device):
        _, mut_m, mut_gate_seq, mut_gate_epi = model(
            tab,
            tab_mask,
            mut_ids,
            mut_mask,
        )

    wt_beta = m_to_beta(wt_m)
    mut_beta = m_to_beta(mut_m)

    return {
        "delta_beta": (mut_beta - wt_beta).cpu().float().numpy().flatten(),
        "wt_gate_seq": wt_gate_seq.cpu().float().numpy().flatten(),
        "wt_gate_epi": wt_gate_epi.cpu().float().numpy().flatten(),
        "mut_gate_seq": mut_gate_seq.cpu().float().numpy().flatten(),
        "mut_gate_epi": mut_gate_epi.cpu().float().numpy().flatten(),
    }


def run_deterministic_sweep(
    model,
    wt_encoded,
    mut_encoded,
    tabs,
    tab_masks,
    batch_size: int,
    device,
    description: str,
) -> dict[str, np.ndarray]:
    outputs = {
        "wt_beta": [],
        "mut_beta": [],
        "wt_gate_seq": [],
        "wt_gate_epi": [],
        "mut_gate_seq": [],
        "mut_gate_epi": [],
    }

    n_samples = len(tabs)
    for start in tqdm(
        range(0, n_samples, batch_size),
        desc=description,
    ):
        stop = start + batch_size
        batch_result = deterministic_forward(
            model=model,
            wt_ids=wt_encoded["input_ids"][start:stop].to(device),
            wt_mask=wt_encoded["attention_mask"][start:stop].to(device),
            mut_ids=mut_encoded["input_ids"][start:stop].to(device),
            mut_mask=mut_encoded["attention_mask"][start:stop].to(device),
            tab=tabs[start:stop].to(device),
            tab_mask=tab_masks[start:stop].to(device),
            device=device,
        )
        for key in outputs:
            outputs[key].extend(batch_result[key])

    arrays = {
        key: np.asarray(values, dtype=float)
        for key, values in outputs.items()
    }
    arrays["delta_beta"] = arrays["mut_beta"] - arrays["wt_beta"]
    arrays["absolute_rank"] = absolute_effect_ranks(arrays["delta_beta"])
    return arrays


def run_reverse_complement_diagnostic(
    model,
    tokenizer,
    cohort: pd.DataFrame,
    wt_sequences: list[str],
    mut_sequences: list[str],
    tabs: torch.Tensor,
    tab_masks: torch.Tensor,
    deterministic: dict[str, np.ndarray],
    batch_size: int,
    device,
    base_dir: Path,
) -> dict[str, float]:
    logging.info("[*] Running corrected reverse-complement diagnostic...")

    rc_wt_sequences = [reverse_complement(sequence) for sequence in wt_sequences]
    rc_mut_sequences = [reverse_complement(sequence) for sequence in mut_sequences]

    validate_reverse_complement_pairs(
        wt_sequences,
        mut_sequences,
        rc_wt_sequences,
        rc_mut_sequences,
    )

    # Critical correction: exchange ordered target-base PhyloP features.
    rc_tabs, rc_tab_masks = build_rc_tabular(tabs, tab_masks)

    rc_wt_encoded = tokenize_sequences(tokenizer, rc_wt_sequences)
    rc_mut_encoded = tokenize_sequences(tokenizer, rc_mut_sequences)

    model.eval()
    rc = run_deterministic_sweep(
        model=model,
        wt_encoded=rc_wt_encoded,
        mut_encoded=rc_mut_encoded,
        tabs=rc_tabs,
        tab_masks=rc_tab_masks,
        batch_size=batch_size,
        device=device,
        description="Corrected RC sweep",
    )

    det_delta = deterministic["delta_beta"]
    rc_delta = rc["delta_beta"]
    det_abs = np.abs(det_delta)
    rc_abs = np.abs(rc_delta)
    det_rank = deterministic["absolute_rank"]
    rc_rank = rc["absolute_rank"]

    absolute_rho = float(spearmanr(det_abs, rc_abs).statistic)
    signed_rho = float(spearmanr(det_delta, rc_delta).statistic)
    sign_agreement = float(np.mean(np.sign(det_delta) == np.sign(rc_delta)))

    overlaps: dict[int, float] = {}
    for k in (10, 20, 50, 100):
        effective_k = min(k, len(cohort))
        det_top = set(np.argsort(-det_abs)[:effective_k])
        rc_top = set(np.argsort(-rc_abs)[:effective_k])
        overlaps[k] = len(det_top & rc_top) / effective_k

    rc_table = pd.DataFrame(
        {
            "Variant_UID": cohort["Variant_UID"],
            "Gene": cohort.get("Gene", pd.Series(["Unknown"] * len(cohort))),
            "Mutation_Window_Index_0based": cohort[
                "Mutation_Window_Index_0based"
            ],
            "RC_Mutation_Window_Index_0based": (
                SEQ_WINDOW_SIZE
                - 1
                - cohort["Mutation_Window_Index_0based"].to_numpy()
            ),
            "Deterministic_Delta_Beta": det_delta,
            "RC_Delta_Beta": rc_delta,
            "Deterministic_Absolute_Rank": det_rank,
            "RC_Absolute_Rank": rc_rank,
            "Sign_Retained": np.sign(det_delta) == np.sign(rc_delta),
            "Absolute_Delta_Ratio_RC_to_Original": (
                rc_abs / np.maximum(det_abs, 1e-12)
            ),
        }
    ).sort_values("Deterministic_Absolute_Rank")

    rc_table.to_csv(
        base_dir / "reverse_complement_consistency.csv",
        index=False,
    )

    summary = {
        "Absolute_Effect_Spearman_Rho": absolute_rho,
        "Signed_Effect_Spearman_Rho": signed_rho,
        "Sign_Agreement": sign_agreement,
        "Top10_Overlap": overlaps[10],
        "Top20_Overlap": overlaps[20],
        "Top50_Overlap": overlaps[50],
        "Top100_Overlap": overlaps[100],
    }
    pd.DataFrame([summary]).to_csv(
        base_dir / "reverse_complement_summary.csv",
        index=False,
    )

    print("\n" + "=" * 64)
    print("CORRECTED REVERSE-COMPLEMENT DIAGNOSTIC")
    print("=" * 64)
    print(f"Absolute-effect Spearman rho: {absolute_rho:.4f}")
    print(f"Signed-effect Spearman rho:   {signed_rho:.4f}")
    print(f"Overall sign agreement:       {sign_agreement:.2%}")
    print("\nTop-K overlaps:")
    for k in (10, 20, 50, 100):
        print(f"  Top {k}: {overlaps[k]:.2%}")

    target_uids = {
        "MSRA": "chr8:g.10428264C>T|cg14264678",
        "DDC": "chr7:g.50543912G>A|cg05346287",
        "CHD5": "chr1:g.6128067C>T|cg12135344",
    }

    print("\nTarget tracking:")
    uid_array = cohort["Variant_UID"].to_numpy()
    for gene, uid in target_uids.items():
        matches = np.flatnonzero(uid_array == uid)
        if len(matches) != 1:
            print(f"  {gene}: expected one row for {uid}, found {len(matches)}")
            continue

        index = int(matches[0])
        print(
            f"  {gene}: "
            f"Det={det_delta[index]:.4f}, "
            f"RC={rc_delta[index]:.4f}, "
            f"Det_Rank={det_rank[index]}, "
            f"RC_Rank={rc_rank[index]}, "
            f"Sign_Retained={np.sign(det_delta[index]) == np.sign(rc_delta[index])}"
        )

    print("=" * 64 + "\n")
    return summary


def run_mc_dropout(
    model,
    cohort: pd.DataFrame,
    wt_encoded,
    mut_encoded,
    tabs: torch.Tensor,
    tab_masks: torch.Tensor,
    deterministic: dict[str, np.ndarray],
    mc_passes: int,
    batch_size: int,
    device,
    base_dir: Path,
) -> None:
    logging.info("[*] Running %d matched MC-dropout passes...", mc_passes)
    enable_mc_dropout(model)

    n_samples = len(cohort)
    mc_deltas = np.zeros((mc_passes, n_samples), dtype=np.float32)
    mc_ranks = np.zeros((mc_passes, n_samples), dtype=np.int64)

    mc_wt_gate_seq = np.zeros((mc_passes, n_samples), dtype=np.float32)
    mc_wt_gate_epi = np.zeros((mc_passes, n_samples), dtype=np.float32)
    mc_mut_gate_seq = np.zeros((mc_passes, n_samples), dtype=np.float32)
    mc_mut_gate_epi = np.zeros((mc_passes, n_samples), dtype=np.float32)

    deterministic_abs = np.abs(deterministic["delta_beta"])
    deterministic_top10 = set(np.argsort(-deterministic_abs)[:10])
    pass_summary_rows: list[dict] = []

    for pass_index in tqdm(range(mc_passes), desc="MC dropout passes"):
        pass_delta: list[float] = []
        pass_wt_gate_seq: list[float] = []
        pass_wt_gate_epi: list[float] = []
        pass_mut_gate_seq: list[float] = []
        pass_mut_gate_epi: list[float] = []

        for batch_index, start in enumerate(
            range(0, n_samples, batch_size)
        ):
            stop = start + batch_size
            batch_seed = 10_000 + pass_index * 100_000 + batch_index

            result = paired_mc_forward(
                model=model,
                wt_ids=wt_encoded["input_ids"][start:stop].to(device),
                wt_mask=wt_encoded["attention_mask"][start:stop].to(device),
                mut_ids=mut_encoded["input_ids"][start:stop].to(device),
                mut_mask=mut_encoded["attention_mask"][start:stop].to(device),
                tab=tabs[start:stop].to(device),
                tab_mask=tab_masks[start:stop].to(device),
                seed=batch_seed,
                device=device,
            )

            pass_delta.extend(result["delta_beta"])
            pass_wt_gate_seq.extend(result["wt_gate_seq"])
            pass_wt_gate_epi.extend(result["wt_gate_epi"])
            pass_mut_gate_seq.extend(result["mut_gate_seq"])
            pass_mut_gate_epi.extend(result["mut_gate_epi"])

        mc_deltas[pass_index] = np.asarray(pass_delta)
        mc_wt_gate_seq[pass_index] = np.asarray(pass_wt_gate_seq)
        mc_wt_gate_epi[pass_index] = np.asarray(pass_wt_gate_epi)
        mc_mut_gate_seq[pass_index] = np.asarray(pass_mut_gate_seq)
        mc_mut_gate_epi[pass_index] = np.asarray(pass_mut_gate_epi)

        pass_ranks = absolute_effect_ranks(mc_deltas[pass_index])
        mc_ranks[pass_index] = pass_ranks

        pass_abs = np.abs(mc_deltas[pass_index])
        rho = float(spearmanr(deterministic_abs, pass_abs).statistic)
        pass_top10 = set(np.argsort(-pass_abs)[:10])
        top10_overlap = len(deterministic_top10 & pass_top10) / 10.0

        pass_summary_rows.append(
            {
                "Pass": pass_index + 1,
                "Spearman_Absolute_Delta": rho,
                "Top10_Overlap": top10_overlap,
            }
        )

    np.savez_compressed(
        base_dir / "mc_dropout_raw_predictions.npz",
        delta_beta=mc_deltas,
        absolute_ranks=mc_ranks,
        wt_gate_seq=mc_wt_gate_seq,
        wt_gate_epi=mc_wt_gate_epi,
        mut_gate_seq=mc_mut_gate_seq,
        mut_gate_epi=mc_mut_gate_epi,
    )

    pass_summary = pd.DataFrame(pass_summary_rows)
    pass_summary.to_csv(
        base_dir / "mc_dropout_pass_summary.csv",
        index=False,
    )

    deterministic_delta = deterministic["delta_beta"]
    deterministic_sign = np.sign(deterministic_delta)
    epsilon = 1e-8

    stability = pd.DataFrame(
        {
            "Variant_UID": cohort["Variant_UID"],
            "Gene": cohort.get("Gene", pd.Series(["Unknown"] * len(cohort))),
            "Deterministic_Delta_Beta": deterministic_delta,
            "Deterministic_Absolute_Rank": deterministic["absolute_rank"],
            "MC_Delta_Median": np.median(mc_deltas, axis=0),
            "MC_Delta_P05": np.quantile(mc_deltas, 0.05, axis=0),
            "MC_Delta_P95": np.quantile(mc_deltas, 0.95, axis=0),
            "MC_Delta_STD": np.std(mc_deltas, axis=0, ddof=1),
            "Sign_Consistency": np.mean(
                np.sign(mc_deltas) == deterministic_sign[None, :],
                axis=0,
            ),
            "MC_Rank_Median": np.median(mc_ranks, axis=0),
            "MC_Rank_P05": np.quantile(mc_ranks, 0.05, axis=0),
            "MC_Rank_P95": np.quantile(mc_ranks, 0.95, axis=0),
            "Top10_Frequency": np.mean(mc_ranks <= 10, axis=0),
            "Top20_Frequency": np.mean(mc_ranks <= 20, axis=0),
            "Deterministic_WT_Sequence_Gate": deterministic[
                "wt_gate_seq"
            ],
            "Deterministic_WT_Epigenomic_Gate": deterministic[
                "wt_gate_epi"
            ],
            "Deterministic_Mutant_Sequence_Gate": deterministic[
                "mut_gate_seq"
            ],
            "Deterministic_Mutant_Epigenomic_Gate": deterministic[
                "mut_gate_epi"
            ],
            "WT_Sequence_Gate_Median": np.median(
                mc_wt_gate_seq,
                axis=0,
            ),
            "WT_Epigenomic_Gate_Median": np.median(
                mc_wt_gate_epi,
                axis=0,
            ),
            "WT_Sequence_Ratio_Median": np.median(
                mc_wt_gate_seq
                / (mc_wt_gate_seq + mc_wt_gate_epi + epsilon),
                axis=0,
            ),
            "Mutant_Sequence_Ratio_Median": np.median(
                mc_mut_gate_seq
                / (mc_mut_gate_seq + mc_mut_gate_epi + epsilon),
                axis=0,
            ),
        }
    ).sort_values("Deterministic_Absolute_Rank")

    stability.to_csv(
        base_dir / "mc_dropout_variant_stability.csv",
        index=False,
    )

    logging.info(
        "[+] MC median absolute-rank Spearman rho: %.4f",
        pass_summary["Spearman_Absolute_Delta"].median(),
    )
    logging.info(
        "[+] MC median top-10 overlap: %.2f%%",
        100.0 * pass_summary["Top10_Overlap"].median(),
    )


# =============================================================================
# 4. MAIN
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Corrected reverse-complement and MC-dropout stability analysis "
            "for the sequence-epigenomic fusion model."
        )
    )
    parser.add_argument(
        "--test-csv",
        default=DEFAULT_TEST_CSV_PATH,
    )
    parser.add_argument(
        "--model-weights",
        default=DEFAULT_MODEL_WEIGHTS,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_BASE_DIR,
    )
    parser.add_argument(
        "--model-path",
        default="zhihan1996/DNABERT-2-117M",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--rc-only",
        action="store_true",
        help="Run deterministic + corrected RC diagnostic, then stop.",
    )
    parser.add_argument(
        "--skip-rc",
        action="store_true",
        help="Skip the RC diagnostic and run only deterministic + MC dropout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("[*] Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = SequenceEpiFusionModel(
        model_path=args.model_path,
        tabular_dim=len(TABULAR_FEATURES),
    ).to(device)

    checkpoint = torch.load(
        args.model_weights,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    raw_df = pd.read_csv(args.test_csv)
    cohort, wt_sequences, mut_sequences, tabs, tab_masks = build_valid_cohort(
        raw_df
    )

    # Save the exact ordered cohort so every later analysis can reuse it.
    cohort.to_csv(base_dir / "stability_eligible_cohort.csv", index=False)

    wt_encoded = tokenize_sequences(tokenizer, wt_sequences)
    mut_encoded = tokenize_sequences(tokenizer, mut_sequences)

    logging.info("[*] Running deterministic reference sweep...")
    deterministic = run_deterministic_sweep(
        model=model,
        wt_encoded=wt_encoded,
        mut_encoded=mut_encoded,
        tabs=tabs,
        tab_masks=tab_masks,
        batch_size=args.batch_size,
        device=device,
        description="Deterministic",
    )

    deterministic_table = pd.DataFrame(
        {
            "Variant_UID": cohort["Variant_UID"],
            "Gene": cohort.get("Gene", pd.Series(["Unknown"] * len(cohort))),
            "Predicted_WT_Beta": deterministic["wt_beta"],
            "Predicted_Mutant_Beta": deterministic["mut_beta"],
            "Predicted_Delta_Beta": deterministic["delta_beta"],
            "Absolute_Delta_Beta_Rank": deterministic["absolute_rank"],
        }
    ).sort_values("Absolute_Delta_Beta_Rank")
    deterministic_table.to_csv(
        base_dir / "deterministic_predictions.csv",
        index=False,
    )

    if not args.skip_rc:
        run_reverse_complement_diagnostic(
            model=model,
            tokenizer=tokenizer,
            cohort=cohort,
            wt_sequences=wt_sequences,
            mut_sequences=mut_sequences,
            tabs=tabs,
            tab_masks=tab_masks,
            deterministic=deterministic,
            batch_size=args.batch_size,
            device=device,
            base_dir=base_dir,
        )

    if args.rc_only:
        logging.info("[+] RC-only run completed.")
        return

    run_mc_dropout(
        model=model,
        cohort=cohort,
        wt_encoded=wt_encoded,
        mut_encoded=mut_encoded,
        tabs=tabs,
        tab_masks=tab_masks,
        deterministic=deterministic,
        mc_passes=args.mc_passes,
        batch_size=args.batch_size,
        device=device,
        base_dir=base_dir,
    )

    logging.info("[+] Stability analysis completed. Outputs: %s", base_dir)


if __name__ == "__main__":
    main()

