#!/usr/bin/env python3
"""Candidate scoring plus matched observed-synonymous background analysis.

Primary inference is exactly aligned with the final SilentMethyl evaluation:
for every checkpoint and every WT/MUT sequence, evaluate both the forward and
reverse-complement orientations, swap the two ordered target-base PhyloP
features (and their explicit missingness indicators) under RC, convert each
orientation's M-value to beta, average beta across orientations, and only then
compute mutant - wild-type delta beta.

The script accepts any explicit list of trained seeds.  ``--seeds 42`` is a
valid single-seed run.  Later, ``--seeds 42 43 44`` reruns the same analysis
without changing code. When multiple seeds are supplied, background comparison
uses the cross-seed mean RC-averaged delta beta; with one seed, that mean is
simply the one available model's delta beta.

The comparator pool contains other observed synonymous candidates selected by
progressively broader similarity-priority tiers. Results are descriptive
background percentiles and tail probabilities, not a random-mutation null or a
formal hypothesis test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: str | Path) -> Path:
    """Locate the project without requiring a particular launch directory."""
    start = Path(start).resolve()
    candidates = [start, *start.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "scripts" / "training_common.py").is_file():
            return candidate
    return start


def coerce_boolean(series: pd.Series, column_name: str = "value") -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True, "t": True, "1": True, "yes": True,
        "false": False, "f": False, "0": False, "no": False,
    }
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        examples = series[parsed.isna()].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"Cannot parse {column_name} as boolean; examples={examples}"
        )
    return parsed.astype(bool)

PROJECT_ROOT = find_project_root(SCRIPT_DIR)
PROJECT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if PROJECT_SCRIPTS_DIR.is_dir() and str(PROJECT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPTS_DIR))

from training_common import (
    FusionModel,
    MISSING_FEATURES,
    TABULAR_FEATURES,
    autocast_context,
    centered_crop,
    get_tokenizer,
    load_model_state,
    m_to_beta_tensor,
    reverse_complement,
    set_seed,
)
from matched_background_utils import (
    CENTER_C_INDEX,
    PROTECTED_CPG_INDICES,
    annotate_variant,
    compute_matched_background_statistics,
)

LOGGER = logging.getLogger("silentmethyl.candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score held-out SilentMethyl candidates with RC-averaged journal checkpoints"
    )
    parser.add_argument(
        "--input-csv",
        default="data/datafiles/testing_data_test_only.csv",
        help="Clean held-out candidate cohort emitted by build_testing_data.py.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Explicit trained seeds to analyze, e.g. --seeds 42 or --seeds 42 43 44.",
    )
    parser.add_argument(
        "--weights-template",
        default="checkpoints_journal/seed{seed}/fusion/best_weights.pth",
        help="Python format string containing {seed}.",
    )
    parser.add_argument("--output-dir", default="results/journal/candidates")
    parser.add_argument("--hm450-manifest", default="data/HM450.hg38.manifest.tsv.gz")
    parser.add_argument(
        "--include-masked-probes", action="store_true",
        help="Include MASK_general=True probes. By default they are excluded from the primary cohort.",
    )
    parser.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local-model-dir", default="./dnabert2_local")
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Enable CUDA mixed-precision inference. Disabled by default because candidate "
            "effects are differences between nearly identical WT/MUT predictions and should "
            "be scored in FP32 for the primary counterfactual analysis."
        ),
    )
    parser.add_argument(
        "--min-comparators", "--min-controls", dest="min_comparators", type=int, default=20
    )
    parser.add_argument(
        "--max-comparators", "--max-controls", dest="max_comparators", type=int, default=1000
    )
    parser.add_argument(
        "--background-random-seed", "--null-random-seed",
        dest="background_random_seed", type=int, default=42,
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--plot-top-k", type=int, default=3)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="0 uses all eligible held-out candidates; positive values are smoke-test only.",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def validate_seed_list(seeds: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        seed = int(seed)
        if seed in seen:
            continue
        seen.add(seed)
        result.append(seed)
    if not result:
        raise ValueError("At least one seed is required")
    return result


def validate_candidate_table(df: pd.DataFrame, input_path: str) -> None:
    required = {
        "probeID",
        "Healthy_5000bp_DNA",
        "Mutated_5000bp_DNA",
        *TABULAR_FEATURES,
        *MISSING_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    if "Model_Split" not in df.columns:
        raise ValueError(
            "Candidate file lacks Model_Split. Use the cleaned journal cohort and the held-out "
            "testing_data_test_only.csv file."
        )
    bad_split = ~df["Model_Split"].astype(str).eq("test")
    if bad_split.any():
        counts = df["Model_Split"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Primary candidate input must be held-out only; Model_Split counts={counts}")

    for feature in TABULAR_FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{feature} contains non-finite values; candidate inputs must already use "
                "train-derived imputation."
            )
    for feature in MISSING_FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{feature} must be an explicit binary missingness indicator")


def apply_probe_qc(df: pd.DataFrame, manifest_path: str, include_masked: bool) -> pd.DataFrame:
    """Attach MASK_general and make unmasked probes the default primary cohort."""
    manifest = pd.read_csv(manifest_path, sep="\t", compression="infer", low_memory=False)
    probe_col = next((c for c in ("probeID", "IlmnID", "Name") if c in manifest.columns), None)
    if probe_col is None or "MASK_general" not in manifest.columns:
        raise ValueError(f"{manifest_path} requires a probe ID column and MASK_general")
    mapping = manifest[[probe_col, "MASK_general"]].rename(columns={probe_col: "probeID"}).copy()
    mapping["probeID"] = mapping["probeID"].astype(str)
    mapping["HM450_MASK_general"] = coerce_boolean(mapping.pop("MASK_general"), "MASK_general")
    if mapping["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probe IDs in {manifest_path}")
    out = df.copy()
    out["probeID"] = out["probeID"].astype(str)
    out = out.merge(mapping, on="probeID", how="left", validate="many_to_one")
    if out["HM450_MASK_general"].isna().any():
        missing = int(out["HM450_MASK_general"].isna().sum())
        raise ValueError(f"{missing} candidate rows are absent from the HM450 manifest")
    if not include_masked:
        before = len(out)
        out = out[~out["HM450_MASK_general"].astype(bool)].copy()
        LOGGER.info("HM450 probe QC retained %d/%d candidate rows", len(out), before)
    return out


def build_model_visible_cohort(
    raw_df: pd.DataFrame,
    window_size: int,
) -> tuple[pd.DataFrame, list[str], list[str], torch.Tensor, torch.Tensor]:
    """Filter to candidates whose unique SNV is actually visible in the model crop."""
    records: list[dict] = []
    wt_sequences: list[str] = []
    mut_sequences: list[str] = []
    tabs: list[torch.Tensor] = []
    missing_rows: list[torch.Tensor] = []
    seen_uids: set[str] = set()
    skipped = {
        "invalid_sequence": 0,
        "mutation_outside_model_crop": 0,
        "invalid_or_noncentral_snv": 0,
        "duplicate_uid": 0,
    }

    for source_index, row in tqdm(
        raw_df.iterrows(), total=len(raw_df), desc="Validating model-visible candidates"
    ):
        try:
            wt = centered_crop(str(row["Healthy_5000bp_DNA"]), window_size)
            mut = centered_crop(str(row["Mutated_5000bp_DNA"]), window_size)
        except ValueError:
            skipped["invalid_sequence"] += 1
            continue

        differences = [i for i, (a, b) in enumerate(zip(wt, mut)) if a != b]
        if len(differences) == 0:
            skipped["mutation_outside_model_crop"] += 1
            continue
        if len(differences) != 1 or differences[0] in PROTECTED_CPG_INDICES:
            skipped["invalid_or_noncentral_snv"] += 1
            continue

        try:
            metadata = annotate_variant(row, wt, mut, window_size=window_size)
        except ValueError:
            skipped["invalid_or_noncentral_snv"] += 1
            continue

        uid = str(metadata["Variant_UID"])
        if uid in seen_uids:
            skipped["duplicate_uid"] += 1
            continue
        seen_uids.add(uid)

        record = row.to_dict()
        record.update(metadata)
        record["Source_Row_Index"] = int(source_index)
        record["Model_Visible"] = True
        records.append(record)
        wt_sequences.append(wt)
        mut_sequences.append(mut)
        tabs.append(torch.tensor(row[TABULAR_FEATURES].to_numpy(dtype=np.float32), dtype=torch.float32))
        missing_rows.append(
            torch.tensor(row[MISSING_FEATURES].to_numpy(dtype=np.float32), dtype=torch.float32)
        )

    if not records:
        raise RuntimeError("No held-out candidates have a valid SNV inside the 1000-bp model crop")

    LOGGER.info("Model-visible cohort: %d candidates; skipped=%s", len(records), skipped)
    return (
        pd.DataFrame(records).reset_index(drop=True),
        wt_sequences,
        mut_sequences,
        torch.stack(tabs),
        torch.stack(missing_rows),
    )


def make_rc_context(
    tab: torch.Tensor,
    missing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rc_tab = tab.clone()
    rc_missing = missing.clone()
    rc_tab[:, -2], rc_tab[:, -1] = tab[:, -1].clone(), tab[:, -2].clone()
    rc_missing[:, -2], rc_missing[:, -1] = missing[:, -1].clone(), missing[:, -2].clone()
    return rc_tab, rc_missing


@torch.inference_mode()
def infer_sequences(
    model: FusionModel,
    tokenizer,
    sequences: list[str],
    tab: torch.Tensor,
    missing: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, np.ndarray]:
    m_values: list[np.ndarray] = []
    beta_values: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    for start in tqdm(range(0, len(sequences), batch_size), desc="Inference", leave=False):
        stop = min(len(sequences), start + batch_size)
        encoded = tokenizer(
            sequences[start:stop],
            truncation=True,
            max_length=len(sequences[start]),
            padding="max_length",
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        batch_tab = tab[start:stop].to(device)
        batch_missing = missing[start:stop].to(device)

        with autocast_context(device, bool(use_amp and device.type == "cuda")):
            _, m_pred, gate = model(batch_tab, batch_missing, ids, attention_mask)

        # Primary candidate scoring must genuinely remain FP32 because Delta beta is
        # obtained by subtracting nearly identical WT and mutant predictions. Merely
        # disabling autocast is insufficient if a model was instantiated in a lower
        # dtype by its Hugging Face config, so fail loudly instead of silently
        # quantizing the counterfactual effects.
        if not use_amp and m_pred.dtype != torch.float32:
            raise RuntimeError(
                f"FP32 candidate inference requested, but regression output dtype is {m_pred.dtype}. "
                "The model must be converted to float32 before scoring."
            )
        m_pred_for_beta = m_pred.float()
        beta_pred = m_to_beta_tensor(m_pred_for_beta)

        m_values.append(m_pred_for_beta.detach().cpu().numpy().reshape(-1))
        beta_values.append(beta_pred.detach().cpu().float().numpy().reshape(-1))
        gates.append(gate.detach().cpu().float().numpy())

    return {
        "m": np.concatenate(m_values),
        "beta": np.concatenate(beta_values),
        "gates": np.concatenate(gates, axis=0),
    }


def _gate_share(gates: np.ndarray) -> np.ndarray:
    denom = gates[:, 0] + gates[:, 1]
    return gates[:, 0] / np.clip(denom, 1e-12, None)


def score_seed(
    seed: int,
    weights_path: str,
    cohort: pd.DataFrame,
    wt_sequences: list[str],
    mut_sequences: list[str],
    tabs: torch.Tensor,
    missing: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer,
) -> pd.DataFrame:
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Seed {seed} checkpoint not found: {weights_path}")

    set_seed(seed)
    model = FusionModel(
        args.model_path,
        fusion_mode="gated",
        tabular_dim=len(TABULAR_FEATURES),
        local_dir=args.local_model_dir,
    )
    state = load_model_state(weights_path, map_location="cpu")
    floating_checkpoint_dtypes = sorted(
        {str(value.dtype) for value in state.values() if torch.is_tensor(value) and value.is_floating_point()}
    )
    LOGGER.info("Seed %d checkpoint floating dtypes: %s", seed, floating_checkpoint_dtypes)
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Strict fusion checkpoint mismatch for seed {seed}: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    # Force the full model to FP32 for primary counterfactual inference. This is
    # intentionally stronger than only disabling autocast because some HF custom
    # configurations can instantiate modules in their configured torch_dtype.
    if not args.amp:
        model = model.float()
    model = model.to(device)
    model.eval()

    floating_model_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    )
    LOGGER.info("Seed %d model floating parameter dtypes after device move: %s", seed, floating_model_dtypes)
    if not args.amp and floating_model_dtypes != ["torch.float32"]:
        raise RuntimeError(
            "FP32 candidate inference requested, but model contains non-FP32 floating parameters: "
            f"{floating_model_dtypes}"
        )

    rc_tabs, rc_missing = make_rc_context(tabs, missing)
    wt_rc = [reverse_complement(sequence) for sequence in wt_sequences]
    mut_rc = [reverse_complement(sequence) for sequence in mut_sequences]

    LOGGER.info("Seed %d: scoring WT forward", seed)
    wt_fwd = infer_sequences(model, tokenizer, wt_sequences, tabs, missing, args.batch_size, device, args.amp)
    LOGGER.info("Seed %d: scoring WT RC", seed)
    wt_rc_out = infer_sequences(model, tokenizer, wt_rc, rc_tabs, rc_missing, args.batch_size, device, args.amp)
    LOGGER.info("Seed %d: scoring MUT forward", seed)
    mut_fwd = infer_sequences(model, tokenizer, mut_sequences, tabs, missing, args.batch_size, device, args.amp)
    LOGGER.info("Seed %d: scoring MUT RC", seed)
    mut_rc_out = infer_sequences(model, tokenizer, mut_rc, rc_tabs, rc_missing, args.batch_size, device, args.amp)

    wt_beta_avg = (wt_fwd["beta"] + wt_rc_out["beta"]) / 2.0
    mut_beta_avg = (mut_fwd["beta"] + mut_rc_out["beta"]) / 2.0
    delta_fwd = mut_fwd["beta"] - wt_fwd["beta"]
    delta_rc = mut_rc_out["beta"] - wt_rc_out["beta"]
    delta_avg = mut_beta_avg - wt_beta_avg

    wt_gate_avg = (wt_fwd["gates"] + wt_rc_out["gates"]) / 2.0
    mut_gate_avg = (mut_fwd["gates"] + mut_rc_out["gates"]) / 2.0

    output = cohort.copy()
    output["Seed"] = int(seed)
    output["Weights_Path"] = str(weights_path)
    output["Weights_SHA256"] = sha256_file(weights_path)
    output["WT_M_FWD"] = wt_fwd["m"]
    output["WT_M_RC"] = wt_rc_out["m"]
    output["WT_M_RC_Avg"] = (wt_fwd["m"] + wt_rc_out["m"]) / 2.0
    output["WT_Beta_FWD"] = wt_fwd["beta"]
    output["WT_Beta_RC"] = wt_rc_out["beta"]
    output["WT_Beta_RC_Avg"] = wt_beta_avg
    output["MUT_M_FWD"] = mut_fwd["m"]
    output["MUT_M_RC"] = mut_rc_out["m"]
    output["MUT_M_RC_Avg"] = (mut_fwd["m"] + mut_rc_out["m"]) / 2.0
    output["MUT_Beta_FWD"] = mut_fwd["beta"]
    output["MUT_Beta_RC"] = mut_rc_out["beta"]
    output["MUT_Beta_RC_Avg"] = mut_beta_avg
    output["Delta_Beta_FWD"] = delta_fwd
    output["Delta_Beta_RC"] = delta_rc
    output["Predicted_Delta_Beta"] = delta_avg
    output["Absolute_Delta_Beta"] = np.abs(delta_avg)
    output["Delta_Beta_RC_Absolute_Difference"] = np.abs(delta_fwd - delta_rc)
    output["Delta_Beta_RC_Sign_Agree"] = (
        np.sign(delta_fwd) == np.sign(delta_rc)
    ).astype(int)

    for prefix, gate_array in (
        ("WT_Gate_FWD", wt_fwd["gates"]),
        ("WT_Gate_RC", wt_rc_out["gates"]),
        ("WT_Gate_Avg", wt_gate_avg),
        ("MUT_Gate_FWD", mut_fwd["gates"]),
        ("MUT_Gate_RC", mut_rc_out["gates"]),
        ("MUT_Gate_Avg", mut_gate_avg),
    ):
        output[f"{prefix}_DNA"] = gate_array[:, 0]
        output[f"{prefix}_EPI"] = gate_array[:, 1]
        output[f"{prefix}_DNA_Share"] = _gate_share(gate_array)

    output["Absolute_Delta_Beta_Rank_Within_Seed"] = (
        output["Absolute_Delta_Beta"].rank(ascending=False, method="min").astype(int)
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def aggregate_seeds(seed_scores: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    seeds = sorted(seed_scores["Seed"].unique().tolist())
    grouped = seed_scores.groupby("Variant_UID", sort=False)
    rows: list[dict] = []

    for uid, group in grouped:
        group = group.sort_values("Seed")
        if group["Seed"].nunique() != len(seeds):
            raise RuntimeError(f"Candidate {uid} is missing one or more requested seed scores")
        delta = group["Predicted_Delta_Beta"].to_numpy(dtype=float)
        signs = np.sign(delta)
        nonzero = signs[signs != 0]
        if len(nonzero):
            _, counts = np.unique(nonzero, return_counts=True)
            sign_consistency = float(np.max(counts) / len(nonzero))
        else:
            sign_consistency = 1.0

        ranks = group["Absolute_Delta_Beta_Rank_Within_Seed"].to_numpy(dtype=float)
        row = {
            "Variant_UID": uid,
            "Seed_Count": int(len(seeds)),
            "Seeds": ",".join(str(seed) for seed in seeds),
            "Predicted_Delta_Beta": float(np.mean(delta)),
            "Predicted_Delta_Beta_Mean": float(np.mean(delta)),
            "Predicted_Delta_Beta_SD": float(np.std(delta, ddof=0)),
            "Predicted_Delta_Beta_Median": float(np.median(delta)),
            "Predicted_Delta_Beta_Min": float(np.min(delta)),
            "Predicted_Delta_Beta_Max": float(np.max(delta)),
            "Delta_Beta_Sign_Consistency": sign_consistency,
            "Mean_Absolute_Delta_Beta": float(np.mean(np.abs(delta))),
            "Mean_Within_Seed_Rank": float(np.mean(ranks)),
            "SD_Within_Seed_Rank": float(np.std(ranks, ddof=0)),
            "Best_Within_Seed_Rank": int(np.min(ranks)),
            "Worst_Within_Seed_Rank": int(np.max(ranks)),
            "Top10_Seed_Frequency": float(np.mean(ranks <= 10)),
            "Top20_Seed_Frequency": float(np.mean(ranks <= 20)),
            "Mean_Delta_RC_Absolute_Difference": float(
                group["Delta_Beta_RC_Absolute_Difference"].mean()
            ),
            "Delta_RC_Sign_Agreement_Fraction": float(group["Delta_Beta_RC_Sign_Agree"].mean()),
            "WT_Gate_DNA_Mean": float(group["WT_Gate_Avg_DNA"].mean()),
            "WT_Gate_EPI_Mean": float(group["WT_Gate_Avg_EPI"].mean()),
            "WT_Gate_DNA_Share_Mean": float(group["WT_Gate_Avg_DNA_Share"].mean()),
            "MUT_Gate_DNA_Mean": float(group["MUT_Gate_Avg_DNA"].mean()),
            "MUT_Gate_EPI_Mean": float(group["MUT_Gate_Avg_EPI"].mean()),
            "MUT_Gate_DNA_Share_Mean": float(group["MUT_Gate_Avg_DNA_Share"].mean()),
        }
        rows.append(row)

    aggregate = pd.DataFrame(rows)
    base = cohort.drop_duplicates("Variant_UID").copy()
    aggregate = base.merge(aggregate, on="Variant_UID", how="inner", validate="one_to_one")
    aggregate["Absolute_Delta_Beta"] = aggregate["Predicted_Delta_Beta"].abs()
    aggregate["Absolute_Delta_Beta_Rank"] = (
        aggregate["Absolute_Delta_Beta"].rank(ascending=False, method="min").astype(int)
    )
    return aggregate.sort_values(
        ["Absolute_Delta_Beta", "Variant_UID"], ascending=[False, True]
    ).reset_index(drop=True)


def export_top_comparator_relations(
    scored: pd.DataFrame,
    comparator_indices_by_uid: dict[str, np.ndarray],
    top_indices: np.ndarray,
) -> pd.DataFrame:
    relations: list[dict] = []
    for rank, target_index in enumerate(top_indices, start=1):
        target = scored.iloc[int(target_index)]
        for comparator_index in comparator_indices_by_uid[str(target["Variant_UID"])]:
            comparator = scored.iloc[int(comparator_index)]
            relations.append(
                {
                    "Target_Rank": rank,
                    "Target_Variant_UID": target["Variant_UID"],
                    "Target_Gene": target["Gene"],
                    "Target_Delta_Beta": target["Predicted_Delta_Beta"],
                    "Matching_Tier": target["Matched_Background_Tier"],
                    "Comparator_Variant_UID": comparator["Variant_UID"],
                    "Comparator_Gene": comparator["Gene"],
                    "Comparator_Delta_Beta": comparator["Predicted_Delta_Beta"],
                    "Comparator_SBS96": comparator["Canonical_SBS96"],
                    "Comparator_CpG_Effect": comparator["CpG_Effect"],
                    "Comparator_Distance_From_CpG": comparator["Absolute_Distance_From_Target_CpG"],
                }
            )
    return pd.DataFrame(relations)


def plot_matched_background(
    target: pd.Series,
    comparators: np.ndarray,
    rank: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(50, max(10, int(np.sqrt(max(1, len(comparators))) * 2)))
    ax.hist(comparators, bins=bins, alpha=0.75)
    ax.axvline(0, linewidth=1.0)
    ax.axvline(float(target["Predicted_Delta_Beta"]), linewidth=2.0)
    ax.set_xlabel("RC-averaged predicted methylation sensitivity (delta beta)")
    ax.set_ylabel("Matched observed synonymous comparators")
    ax.set_title(
        f"Rank {rank}: {target['Gene']} | descriptive tail="
        f"{target['Matched_Empirical_Tail_Probability']:.4g}"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    seeds = validate_seed_list(args.seeds)
    if args.window_size != 1000:
        raise ValueError(
            "The journal candidate protocol is locked to the trained 1000-bp model window; "
            "do not change --window-size for the primary analysis."
        )

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    raw_df = pd.read_csv(input_path)
    input_row_count_before_probe_qc = len(raw_df)
    validate_candidate_table(raw_df, str(input_path))
    raw_df = apply_probe_qc(raw_df, args.hm450_manifest, args.include_masked_probes)
    if args.max_candidates > 0:
        raw_df = raw_df.head(args.max_candidates).copy()
        LOGGER.warning("Smoke-test mode: limiting input to %d rows", len(raw_df))

    cohort, wt_sequences, mut_sequences, tabs, missing = build_model_visible_cohort(
        raw_df, args.window_size
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_csv(cohort, out / "eligible_heldout_model_visible_cohort.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s | seeds=%s", device, seeds)
    LOGGER.info("Candidate inference precision: %s", "CUDA AMP" if args.amp and device.type == "cuda" else "FP32")
    tokenizer = get_tokenizer(args.model_path)

    per_seed_scores: list[pd.DataFrame] = []
    checkpoint_manifest = []
    for seed in seeds:
        weights_path = args.weights_template.format(seed=seed)
        seed_df = score_seed(
            seed,
            weights_path,
            cohort,
            wt_sequences,
            mut_sequences,
            tabs,
            missing,
            args,
            device,
            tokenizer,
        )
        seed_out = out / f"seed{seed}"
        atomic_csv(seed_df, seed_out / "candidate_scores.csv")
        per_seed_scores.append(seed_df)
        checkpoint_manifest.append(
            {
                "seed": int(seed),
                "weights_path": str(weights_path),
                "sha256": sha256_file(weights_path),
            }
        )

    long_scores = pd.concat(per_seed_scores, ignore_index=True)
    atomic_csv(long_scores, out / "candidate_seed_scores_long.csv")

    aggregate = aggregate_seeds(long_scores, cohort)
    scored_unsorted, comparator_indices = compute_matched_background_statistics(
        aggregate,
        delta_column="Predicted_Delta_Beta",
        min_comparators=args.min_comparators,
        max_comparators=args.max_comparators,
        random_seed=args.background_random_seed,
    )
    scored = scored_unsorted.sort_values(
        ["Absolute_Delta_Beta", "Variant_UID"], ascending=[False, True]
    ).reset_index(drop=True)
    scored["Absolute_Delta_Beta_Rank"] = np.arange(1, len(scored) + 1)

    atomic_csv(scored, out / "candidate_matched_background_statistics.csv")
    atomic_csv(scored.head(args.top_k), out / "top_candidates.csv")

    top_indices_original = np.array(
        [
            scored_unsorted.index[scored_unsorted["Variant_UID"].eq(uid)][0]
            for uid in scored.head(args.top_k)["Variant_UID"]
        ],
        dtype=int,
    )
    comparators_long = export_top_comparator_relations(
        scored_unsorted, comparator_indices, top_indices_original
    )
    atomic_csv(comparators_long, out / "top_candidate_matched_comparators_long.csv")

    # Plots retrieve comparator rows by immutable UID, so sorting is harmless.
    for rank, target in scored.head(args.plot_top_k).iterrows():
        uid = str(target["Variant_UID"])
        original_index = int(scored_unsorted.index[scored_unsorted["Variant_UID"].eq(uid)][0])
        comparator_values = scored_unsorted.iloc[comparator_indices[uid]]["Predicted_Delta_Beta"].to_numpy(dtype=float)
        plot_matched_background(
            target, comparator_values, rank + 1,
            out / "plots" / f"matched_background_rank{rank + 1}.png",
        )

    summary = {
        "input_csv": str(input_path),
        "requested_seeds": seeds,
        "seed_count": len(seeds),
        "inference_precision": "cuda_amp" if args.amp and device.type == "cuda" else "fp32_forced_and_dtype_audited",
        "cross_seed_aggregation": "mean RC-averaged delta beta" if len(seeds) > 1 else "single available seed",
        "cross_seed_stability_inference_available": len(seeds) > 1,
        "candidate_rows_input_before_probe_qc": int(input_row_count_before_probe_qc),
        "candidate_rows_after_probe_qc": int(len(raw_df)),
        "hm450_manifest": str(args.hm450_manifest),
        "masked_probes_included": bool(args.include_masked_probes),
        "candidate_rows_model_visible": int(len(cohort)),
        "primary_ranking": "absolute value of cross-seed mean RC-averaged mutant-minus-WT beta shift",
        "background_interpretation": (
            "descriptive observed-synonymous comparators selected by similarity-priority tiers; "
            "not a random-mutation null or calibrated p-value"
        ),
        "matched_background_delta_column": "Predicted_Delta_Beta",
        "min_comparators": int(args.min_comparators),
        "max_comparators": int(args.max_comparators),
        "background_random_seed": int(args.background_random_seed),
        "checkpoints": checkpoint_manifest,
        "minimum_empirical_tail_probability": _json_float(
            scored["Matched_Empirical_Tail_Probability"].min()
        ),
        "output_rows": int(len(scored)),
    }
    atomic_json(summary, out / "candidate_analysis_summary.json")
    LOGGER.info(
        "Finished candidate background analysis: n=%d seeds=%s min_tail=%s",
        len(scored),
        seeds,
        summary["minimum_empirical_tail_probability"],
    )


if __name__ == "__main__":
    main()
