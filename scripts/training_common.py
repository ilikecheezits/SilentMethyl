from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


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

MISSING_FEATURES = [f"{name}_Missing" for name in TABULAR_FEATURES]
PHYLOP_1 = "Target_Base_PhyloP_100way_1"
PHYLOP_2 = "Target_Base_PhyloP_100way_2"
PHYLOP_1_MISSING = f"{PHYLOP_1}_Missing"
PHYLOP_2_MISSING = f"{PHYLOP_2}_Missing"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def centered_crop(seq: str, window_size: int) -> str:
    seq = seq.upper()
    if window_size <= 0 or window_size > len(seq):
        raise ValueError(f"window_size={window_size} is invalid for sequence length {len(seq)}")
    # The 5-kb builder places the target C at index 2499 and G at 2500.
    # For an even crop this keeps them at indices window/2-1 and window/2.
    center_right = len(seq) // 2
    start = center_right - (window_size // 2)
    end = start + window_size
    return seq[start:end]


def m_to_beta_tensor(m_value: torch.Tensor) -> torch.Tensor:
    # M = log2(beta / (1-beta)), hence beta = sigmoid(M * ln 2).
    return torch.sigmoid(m_value * math.log(2.0))


def m_to_beta_numpy(m_value: np.ndarray) -> np.ndarray:
    x = np.asarray(m_value, dtype=np.float64) * math.log(2.0)
    # Numerically stable logistic.
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def deterministic_rc_choice(seed: int, epoch: int, idx: int, probability: float) -> bool:
    """Reproducible pseudo-random orientation choice for one sample in one epoch.

    This avoids dependence on DataLoader worker RNG state and makes epoch-level
    resume exactly reproducible. Dataset length remains N, not 2N.
    """
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    # Stable integer seed independent of Python hash randomization.
    local_seed = (
        (int(seed) * 0x9E3779B185EBCA87)
        ^ (int(epoch) * 0xC2B2AE3D27D4EB4F)
        ^ (int(idx) * 0x165667B19E3779F9)
    ) & ((1 << 64) - 1)
    return random.Random(local_seed).random() < probability


def _require_columns(df: pd.DataFrame, columns, name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_split_dataframe(df: pd.DataFrame, expected_split: str, name: str) -> None:
    base_cols = [
        "probeID",
        "Healthy_5000bp_DNA",
        "Median_Beta",
        "M_Value_Target",
        "Binary_State_Target",
        "Split",
    ] + TABULAR_FEATURES + MISSING_FEATURES
    _require_columns(df, base_cols, name)

    if not df["probeID"].astype(str).str.startswith("cg").all():
        bad = int((~df["probeID"].astype(str).str.startswith("cg")).sum())
        raise ValueError(f"{name} contains {bad} non-cg probe IDs")
    if not df["Split"].eq(expected_split).all():
        counts = df["Split"].value_counts(dropna=False).to_dict()
        raise ValueError(f"{name} has unexpected Split values: {counts}")
    if df["probeID"].duplicated().any():
        raise ValueError(f"{name} contains duplicate probe IDs")
    beta = pd.to_numeric(df["Median_Beta"], errors="coerce")
    if beta.isna().any() or ((beta < 0) | (beta > 1)).any():
        raise ValueError(f"{name} contains invalid Median_Beta values")


class OrientationAwareDataset(Dataset):
    """One row per locus, with stochastic RC view only during training.

    For training, exactly one orientation is returned per __getitem__. For
    validation/testing, both forward and RC views are returned for deterministic
    test-time averaging. Regional epigenomic features stay unchanged; the two
    ordered central-base PhyloP values and their missingness indicators swap.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer=None,
        seq_window_size: int = 1000,
        training: bool = False,
        rc_probability: float = 0.5,
        seed: int = 42,
        include_sequence: bool = True,
        include_context: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_window_size = int(seq_window_size)
        self.training = bool(training)
        self.rc_probability = float(rc_probability)
        self.seed = int(seed)
        self.epoch = 0
        self.include_sequence = bool(include_sequence)
        self.include_context = bool(include_context)

        if not (0.0 <= self.rc_probability <= 1.0):
            raise ValueError("rc_probability must be in [0, 1]")
        if self.include_sequence and tokenizer is None:
            raise ValueError("tokenizer is required when include_sequence=True")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.df)

    def _context(self, row: pd.Series, use_rc: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        values = row[TABULAR_FEATURES].to_numpy(dtype=np.float32, copy=True)
        missing = row[MISSING_FEATURES].to_numpy(dtype=np.float32, copy=True)

        if not np.isfinite(values).all():
            raise ValueError("Non-finite tabular value encountered after preprocessing")
        if not np.isin(missing, [0.0, 1.0]).all():
            raise ValueError("Missingness indicators must be binary 0/1")

        if use_rc:
            # Ordered central-base PhyloP positions reverse under RC.
            values[-2], values[-1] = values[-1], values[-2]
            missing[-2], missing[-1] = missing[-1], missing[-2]

        return torch.from_numpy(values), torch.from_numpy(missing)

    def _encode_sequence(self, row: pd.Series, use_rc: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = centered_crop(str(row["Healthy_5000bp_DNA"]), self.seq_window_size)
        if use_rc:
            seq = reverse_complement(seq)

        encoded = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.seq_window_size,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"].flatten(), encoded["attention_mask"].flatten()

    def _one_view(self, row: pd.Series, use_rc: bool) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {}
        if self.include_sequence:
            ids, mask = self._encode_sequence(row, use_rc)
            item["input_ids"] = ids
            item["attention_mask"] = mask
        if self.include_context:
            tab, tab_missing = self._context(row, use_rc)
            item["tab"] = tab
            item["tab_missing"] = tab_missing
        return item

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        beta = float(row["Median_Beta"])
        item: Dict[str, torch.Tensor] = {
            "m_value": torch.tensor(float(row["M_Value_Target"]), dtype=torch.float32),
            "beta_value": torch.tensor(beta, dtype=torch.float32),
            "binary_state": torch.tensor(float(row["Binary_State_Target"]), dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
        }

        if self.training:
            use_rc = deterministic_rc_choice(self.seed, self.epoch, idx, self.rc_probability)
            item.update(self._one_view(row, use_rc))
            item["use_rc"] = torch.tensor(float(use_rc), dtype=torch.float32)
            return item

        fwd = self._one_view(row, False)
        rc = self._one_view(row, True)
        for key, value in fwd.items():
            item[f"{key}_fwd"] = value
        for key, value in rc.items():
            item[f"{key}_rc"] = value
        return item


class SequenceEncoder(nn.Module):
    def __init__(self, model_path: str, local_dir: str = "./dnabert2_local") -> None:
        super().__init__()
        self.config, self.bert = patch_and_load_dnabert(model_path, local_dir)
        hidden_size = int(self.config.hidden_size)
        self.hidden_size = hidden_size
        self.spatial_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        spatial = F.relu(self.spatial_conv(hidden.permute(0, 2, 1))).permute(0, 2, 1)
        scores = self.attention_pool(spatial).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = F.softmax(scores, dim=-1)
        return torch.sum(spatial * weights.unsqueeze(-1), dim=1)


class EpigeneticEncoder(nn.Module):
    def __init__(self, tabular_dim: int = 9, hidden_size: int = 768) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
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
        self.epi_proj = nn.Linear(256, self.hidden_size)

    def forward(self, tab: torch.Tensor, tab_missing: torch.Tensor) -> torch.Tensor:
        x = torch.cat([tab, tab_missing], dim=1)
        return self.epi_proj(self.tab_mlp(x))


class DualHeads(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.classification_head(features), self.regression_head(features)


class SequenceOnlyModel(nn.Module):
    def __init__(self, model_path: str, local_dir: str = "./dnabert2_local") -> None:
        super().__init__()
        self.sequence_encoder = SequenceEncoder(model_path, local_dir)
        self.heads = DualHeads(self.sequence_encoder.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        features = self.sequence_encoder(input_ids, attention_mask)
        return self.heads(features)


class EpigeneticOnlyModel(nn.Module):
    def __init__(self, hidden_size: int = 768, tabular_dim: int = 9) -> None:
        super().__init__()
        self.epi_encoder = EpigeneticEncoder(tabular_dim, hidden_size)
        self.heads = DualHeads(hidden_size)

    def forward(self, tab: torch.Tensor, tab_missing: torch.Tensor):
        features = self.epi_encoder(tab, tab_missing)
        return self.heads(features)


class FusionModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        fusion_mode: str = "gated",
        tabular_dim: int = 9,
        local_dir: str = "./dnabert2_local",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"gated", "concat"}:
            raise ValueError("fusion_mode must be 'gated' or 'concat'")
        self.fusion_mode = fusion_mode
        self.sequence_encoder = SequenceEncoder(model_path, local_dir)
        hidden_size = self.sequence_encoder.hidden_size
        self.epi_encoder = EpigeneticEncoder(tabular_dim, hidden_size)
        self.norm_dna = nn.LayerNorm(hidden_size)
        self.norm_epi = nn.LayerNorm(hidden_size)

        if fusion_mode == "gated":
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_size * 2, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 2),
                nn.Sigmoid(),
            )
            head_dim = hidden_size
        else:
            # Literal direct-concatenation comparator: normalized DNA and context
            # embeddings are concatenated and sent straight to the common heads.
            head_dim = hidden_size * 2

        self.heads = DualHeads(head_dim)

    def forward(
        self,
        tab: torch.Tensor,
        tab_missing: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        dna = self.norm_dna(self.sequence_encoder(input_ids, attention_mask))
        epi = self.norm_epi(self.epi_encoder(tab, tab_missing))
        joined = torch.cat([dna, epi], dim=1)

        if self.fusion_mode == "gated":
            gates = self.gate_network(joined)
            fused = dna * gates[:, 0:1] + epi * gates[:, 1:2]
            class_logits, m_pred = self.heads(fused)
            return class_logits, m_pred, gates

        class_logits, m_pred = self.heads(joined)
        return class_logits, m_pred, None


def patch_and_load_dnabert(
    model_path: str = "zhihan1996/DNABERT-2-117M",
    local_dir: str = "./dnabert2_local",
):
    """Load DNABERT-2 pretrained weights without the HF meta-device path.

    DNABERT-2's custom model can fail under ``AutoModel.from_pretrained`` on
    some PyTorch/Transformers combinations with a meta-vs-CPU ALiBi error.
    We therefore instantiate on CPU with ``AutoModel.from_config`` (the same
    workaround used by the older working cluster script), then manually load
    the released checkpoint.

    Crucially, the released checkpoint may prefix the base-model weights with
    ``bert.`` while the instantiated ``BertModel`` expects unprefixed keys.
    The loader below selects the prefix normalization with maximal parameter
    overlap and refuses to continue unless *every named model parameter* is
    initialized from the pretrained checkpoint.  This prevents the previous
    silent random-backbone failure caused by ``strict=False``.
    """
    logging.info("--- Performing DNABERT-2 cluster-safe pretrained loading ---")
    local_dir = str(local_dir)

    # Materialize a local copy once.
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        logging.info("Downloading DNABERT-2 repository into %s", local_dir)
        cache_path = snapshot_download(model_path)
        for item in os.listdir(cache_path):
            src = os.path.join(cache_path, item)
            dst = os.path.join(local_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    # Cluster-safe Triton/FlashAttention neutralization.
    triton_file = os.path.join(local_dir, "flash_attn_triton.py")
    if os.path.exists(triton_file):
        with open(triton_file, "w") as handle:
            handle.write("def __getattr__(name):\n    return None\n")

    config_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing DNABERT config: {config_path}")
    with open(config_path) as handle:
        config_data = json.load(handle)
    config_data["use_flash_attn"] = False
    if config_data.get("pad_token_id") is None:
        config_data["pad_token_id"] = 0
    with open(config_path, "w") as handle:
        json.dump(config_data, handle)

    # from_config avoids the DNABERT-2 meta-device/ALiBi failure seen with
    # AutoModel.from_pretrained on the Bridges-2 software stack.
    config = AutoConfig.from_pretrained(
        local_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    config.output_attentions = False
    base_model = AutoModel.from_config(config, trust_remote_code=True)

    # Prefer the already-materialized checkpoint; download only as a fallback.
    local_weights = os.path.join(local_dir, "pytorch_model.bin")
    if os.path.exists(local_weights):
        weights_path = local_weights
    else:
        weights_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")

    raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        raw_state = raw_state["state_dict"]
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported DNABERT checkpoint payload type: {type(raw_state)!r}")

    model_state = base_model.state_dict()
    parameter_names = {name for name, _ in base_model.named_parameters()}

    # The released checkpoint has appeared with different wrapper prefixes
    # across versions.  Pick the normalization that matches the greatest
    # number of actual base-model parameters instead of assuming one format.
    prefix_candidates = ("", "bert.", "module.", "module.bert.", "model.", "model.bert.")

    def remap_with_prefix(prefix: str):
        mapped = {}
        for key, value in raw_state.items():
            new_key = key[len(prefix):] if prefix and key.startswith(prefix) else key
            # Do not let a non-prefixed key overwrite a genuinely stripped key.
            if new_key not in mapped:
                mapped[new_key] = value
        return mapped

    best_prefix = None
    best_mapped = None
    best_score = (-1, -1)
    for prefix in prefix_candidates:
        mapped = remap_with_prefix(prefix)
        param_overlap = sum(name in mapped for name in parameter_names)
        total_overlap = sum(name in mapped for name in model_state)
        score = (param_overlap, total_overlap)
        if score > best_score:
            best_score = score
            best_prefix = prefix
            best_mapped = mapped

    assert best_mapped is not None

    # The released DNABERT-2 checkpoint does not contain the optional BERT
    # pooler parameters.  This project never uses pooled_output from DNABERT-2;
    # SequenceEncoder builds its own Conv1D + learned attention pooling over
    # last_hidden_state.  Therefore these two randomly initialized, unused
    # pooler parameters are the only acceptable missing parameters.
    allowed_missing_parameters = {
        "pooler.dense.weight",
        "pooler.dense.bias",
    }

    # Require every non-pooler pretrained model parameter to be present and
    # shape-compatible.  Any other missing/mismatched parameter is fatal.
    missing_parameters = []
    mismatched_parameters = []
    for name in sorted(parameter_names):
        if name not in best_mapped:
            missing_parameters.append(name)
            continue
        expected_shape = tuple(model_state[name].shape)
        found_shape = tuple(best_mapped[name].shape)
        if expected_shape != found_shape:
            mismatched_parameters.append((name, found_shape, expected_shape))

    disallowed_missing = [
        name for name in missing_parameters if name not in allowed_missing_parameters
    ]
    if disallowed_missing or mismatched_parameters:
        raise RuntimeError(
            "DNABERT-2 pretrained backbone could not be loaded safely. "
            f"Selected prefix normalization={best_prefix!r}; "
            f"matched_parameters={best_score[0]}/{len(parameter_names)}; "
            f"allowed_missing_pooler={sorted(set(missing_parameters) & allowed_missing_parameters)}; "
            f"disallowed_missing={len(disallowed_missing)}; "
            f"mismatched_parameters={len(mismatched_parameters)}; "
            f"disallowed_missing[:5]={disallowed_missing[:5]}; "
            f"mismatched[:3]={mismatched_parameters[:3]}"
        )

    # Load only keys belonging to this base model. Extra task-head keys in a
    # wrapped checkpoint, if any, are intentionally ignored.
    compatible_state = {
        key: value
        for key, value in best_mapped.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    missing_after, unexpected_after = base_model.load_state_dict(compatible_state, strict=False)

    # Missing non-parameter buffers can be recreated by the model implementation.
    # The only permitted missing parameters are the unused optional pooler.
    missing_parameter_after = [key for key in missing_after if key in parameter_names]
    disallowed_missing_after = [
        key for key in missing_parameter_after if key not in allowed_missing_parameters
    ]
    if disallowed_missing_after:
        raise RuntimeError(
            "DNABERT-2 load left required model parameters uninitialized: "
            f"{disallowed_missing_after[:10]}"
        )
    if unexpected_after:
        raise RuntimeError(f"Unexpected keys after filtered DNABERT load: {unexpected_after[:10]}")

    ignored_raw = len(raw_state) - len(compatible_state)
    loaded_parameter_count = len(parameter_names) - len(missing_parameter_after)
    logging.info(
        "DNABERT-2 pretrained backbone loaded successfully: %d/%d parameters loaded; "
        "prefix normalization=%r; allowed missing pooler=%s; ignored raw keys=%d; "
        "missing non-parameter buffers=%d",
        loaded_parameter_count,
        len(parameter_names),
        best_prefix,
        sorted(missing_parameter_after),
        ignored_raw,
        len([key for key in missing_after if key not in parameter_names]),
    )
    return config, base_model


def get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def get_dnabert_hidden_size(model_path: str, local_dir: str = "./dnabert2_local") -> int:
    # Prefer patched local config when available to ensure architecture agreement.
    config_source = local_dir if os.path.exists(os.path.join(local_dir, "config.json")) else model_path
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    return int(config.hidden_size)


def load_model_state(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    return payload


def load_submodule_from_checkpoint(
    target_module: nn.Module,
    checkpoint_path: str,
    prefix: str,
    map_location="cpu",
) -> None:
    state = load_model_state(checkpoint_path, map_location=map_location)
    wanted = {}
    prefix_dot = prefix + "."
    for key, value in state.items():
        if key.startswith(prefix_dot):
            wanted[key[len(prefix_dot):]] = value
    if not wanted:
        raise ValueError(f"No keys with prefix '{prefix_dot}' found in {checkpoint_path}")
    missing, unexpected = target_module.load_state_dict(wanted, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Submodule load mismatch for {prefix}: missing={missing}, unexpected={unexpected}"
        )


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def make_amp(device: torch.device):
    enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    return scaler, enabled


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def regression_and_classification_metrics(
    beta_true,
    m_true,
    m_pred_avg,
    class_prob_avg,
    binary_true,
    beta_pred_avg: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    beta_true = np.asarray(beta_true, dtype=np.float64)
    m_true = np.asarray(m_true, dtype=np.float64)
    m_pred_avg = np.asarray(m_pred_avg, dtype=np.float64)
    class_prob_avg = np.asarray(class_prob_avg, dtype=np.float64)
    binary_true = np.asarray(binary_true, dtype=np.int64)
    if beta_pred_avg is None:
        beta_pred_avg = m_to_beta_numpy(m_pred_avg)
    else:
        beta_pred_avg = np.asarray(beta_pred_avg, dtype=np.float64)

    metrics = {
        "m_rmse": float(np.sqrt(mean_squared_error(m_true, m_pred_avg))),
        "m_mae": float(mean_absolute_error(m_true, m_pred_avg)),
        "beta_rmse": float(np.sqrt(mean_squared_error(beta_true, beta_pred_avg))),
        "beta_mae": float(mean_absolute_error(beta_true, beta_pred_avg)),
    }
    metrics["auc"] = (
        float(roc_auc_score(binary_true, class_prob_avg))
        if len(np.unique(binary_true)) == 2
        else float("nan")
    )
    return metrics


def orientation_agreement_metrics(
    beta_fwd: np.ndarray,
    beta_rc: np.ndarray,
    class_prob_fwd: np.ndarray,
    class_prob_rc: np.ndarray,
) -> Dict[str, float]:
    beta_fwd = np.asarray(beta_fwd, dtype=np.float64)
    beta_rc = np.asarray(beta_rc, dtype=np.float64)
    class_prob_fwd = np.asarray(class_prob_fwd, dtype=np.float64)
    class_prob_rc = np.asarray(class_prob_rc, dtype=np.float64)

    def corr(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "beta_fwd_rc_mae": float(np.mean(np.abs(beta_fwd - beta_rc))),
        "beta_fwd_rc_max_abs": float(np.max(np.abs(beta_fwd - beta_rc))),
        "beta_fwd_rc_pearson": corr(beta_fwd, beta_rc),
        "classprob_fwd_rc_mae": float(np.mean(np.abs(class_prob_fwd - class_prob_rc))),
        "classprob_fwd_rc_pearson": corr(class_prob_fwd, class_prob_rc),
    }


def save_run_config(save_dir: str, config: Dict) -> None:
    path = Path(save_dir) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
