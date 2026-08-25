#!/usr/bin/env python3
"""Compare significant and matched nonsignificant eGTEx leads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm


LOGGER = logging.getLogger("silentmethyl.mqtl_matched_negative")
DNA_BASES = frozenset("ACGT")
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def find_project_root(start: str | Path) -> Path:
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare model-derived absolute allelic effects between matched "
            "significant and nonsignificant eGTEx breast lead associations"
        )
    )
    parser.add_argument(
        "--lead-mqtl-file",
        default=str(PROJECT_ROOT / "data" / "BreastMammaryTissue.regular.perm.fdr.txt"),
    )
    parser.add_argument(
        "--test-csv",
        default=str(PROJECT_ROOT / "data" / "datafiles" / "test.csv"),
    )
    parser.add_argument(
        "--hm450-manifest",
        default=str(PROJECT_ROOT / "data" / "HM450.hg38.manifest.tsv.gz"),
    )
    parser.add_argument(
        "--positive-control-script",
        default=str(PROJECT_ROOT / "scripts" / "04_mqtl_positive_control.py"),
        help="Current positive-control script whose audited inference implementation is reused.",
    )
    parser.add_argument("--models", nargs="+", choices=("fusion", "sequence"), default=("fusion", "sequence"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--positive-q-threshold", type=float, default=0.05)
    parser.add_argument("--negative-q-threshold", type=float, default=0.50)
    parser.add_argument("--comparators-per-positive", type=int, default=1)
    parser.add_argument(
        "--minimum-comparators-per-positive",
        type=int,
        default=1,
        help="Compatibility option; the globally optimized primary design requires a value of 1.",
    )
    parser.add_argument(
        "--maf-caliper",
        type=float,
        default=0.05,
        help="Maximum absolute MAF difference for a valid same-chromosome matched pair.",
    )
    parser.add_argument("--matching-seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    parser.add_argument("--statistics-seed", type=int, default=20260812)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local-model-dir", default=str(PROJECT_ROOT / "dnabert2_local"))
    parser.add_argument(
        "--fusion-weights-template",
        default=str(PROJECT_ROOT / "checkpoints_journal" / "seed{seed}" / "fusion" / "best_weights.pth"),
    )
    parser.add_argument(
        "--sequence-weights-template",
        default=str(PROJECT_ROOT / "checkpoints_journal" / "seed{seed}" / "sequence" / "best_weights.pth"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "journal" / "egtex_mqtl_matched_negative"),
    )
    parser.add_argument(
        "--max-positive",
        type=int,
        default=0,
        help="Smoke test only: cap eligible significant leads before matching; 0 uses all.",
    )
    return parser.parse_args()


def unique_in_order(values: Iterable[int | str]) -> list:
    output, seen = [], set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def validate_args(args: argparse.Namespace) -> None:
    args.seeds = [int(value) for value in unique_in_order(args.seeds)]
    args.models = [str(value) for value in unique_in_order(args.models)]
    if not args.seeds or not args.models:
        raise ValueError("At least one model and seed are required")
    if not 0 < args.positive_q_threshold < args.negative_q_threshold <= 1:
        raise ValueError("Require 0 < positive q threshold < negative q threshold <= 1")
    if args.comparators_per_positive != 1 or args.minimum_comparators_per_positive != 1:
        raise ValueError("The globally optimized primary design requires one comparator per positive")
    if not 0 < args.maf_caliper <= 0.5:
        raise ValueError("--maf-caliper must lie in (0, 0.5]")
    if min(args.bootstrap_replicates, args.permutation_replicates, args.max_positive) < 0:
        raise ValueError("Replicate counts and --max-positive cannot be negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def import_positive_control(path: str | Path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Positive-control implementation not found: {path}")
    spec = importlib.util.spec_from_file_location("silentmethyl_mqtl_positive_control", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def canonical_sbs6(ref: str, alt: str) -> str:
    ref, alt = ref.upper(), alt.upper()
    if ref in {"A", "G"}:
        ref, alt = COMPLEMENT[ref], COMPLEMENT[alt]
    return f"{ref}>{alt}"


def titv_class(ref: str, alt: str) -> str:
    return "transition" if {ref.upper(), alt.upper()} in ({"A", "G"}, {"C", "T"}) else "transversion"


def prepare_eligible_leads(args: argparse.Namespace, pc) -> tuple[pd.DataFrame, dict]:
    usecols = ["cpg_id", "variant_id", "maf", "slope", "slope_se", "pval_nominal", "pval_permuted", "qval"]
    lead = pd.read_csv(args.lead_mqtl_file, sep="\t", usecols=usecols, low_memory=False)
    input_rows = len(lead)
    for column in ("maf", "slope", "slope_se", "pval_nominal", "pval_permuted", "qval"):
        lead[column] = pd.to_numeric(lead[column], errors="coerce")

    parsed = lead["variant_id"].astype(str).str.extract(
        r"^(chr[^_]+)_(\d+)_([ACGT])_([ACGT])_b38$"
    )
    parsed.columns = ["var_chr", "var_pos1", "ref", "alt"]
    lead = pd.concat([lead, parsed], axis=1)
    lead = lead.dropna(subset=["cpg_id", "var_chr", "var_pos1", "ref", "alt", "maf", "qval"])
    lead["var_pos1"] = lead["var_pos1"].astype(np.int64)
    lead["var_pos0"] = lead["var_pos1"] - 1
    lead["var_chr"] = lead["var_chr"].map(pc.normalize_chromosome)
    lead["ref"] = lead["ref"].str.upper()
    lead["alt"] = lead["alt"].str.upper()

    test = pd.read_csv(args.test_csv)
    pc.validate_split_dataframe(test, "test", args.test_csv)
    test = test.rename(columns={"chr": "cpg_chr", "pos": "cpg_pos0"})
    merged = lead.merge(
        test,
        left_on="cpg_id",
        right_on="probeID",
        how="inner",
        validate="one_to_one",
    )
    merged["cpg_chr"] = merged["cpg_chr"].map(pc.normalize_chromosome)
    merged["cpg_pos0"] = pd.to_numeric(merged["cpg_pos0"], errors="raise").astype(np.int64)
    merged = merged[merged["var_chr"].eq(merged["cpg_chr"])].copy()
    merged["offset_from_cpg_C"] = merged["var_pos0"] - merged["cpg_pos0"]
    merged["absolute_distance_bp"] = merged["offset_from_cpg_C"].abs()
    merged = merged[
        merged["offset_from_cpg_C"].between(-pc.MODEL_TARGET_C_INDEX, pc.MODEL_WINDOW_SIZE - pc.MODEL_TARGET_C_INDEX - 1)
        & ~merged["offset_from_cpg_C"].isin(pc.PROTECTED_OFFSETS)
    ].copy()

    sequence_ok = []
    for row in merged.itertuples(index=False):
        sequence = str(row.Healthy_5000bp_DNA).upper()
        index = pc.FULL_TARGET_C_INDEX + int(row.offset_from_cpg_C)
        sequence_ok.append(
            len(sequence) == pc.FULL_SEQUENCE_LENGTH
            and sequence[pc.FULL_TARGET_C_INDEX : pc.FULL_TARGET_C_INDEX + 2] == "CG"
            and 0 <= index < len(sequence)
            and sequence[index] == str(row.ref)
        )
    merged["hg38_ref_match"] = sequence_ok
    merged = merged[merged["hg38_ref_match"]].copy()
    merged["cpg_pos0"] = merged["cpg_pos0"].astype(np.int64)
    merged["split"] = "test"
    merged["model_visible"] = True
    merged["target_cpg_variant"] = False
    merged["sbs6"] = [canonical_sbs6(r, a) for r, a in zip(merged["ref"], merged["alt"])]
    merged["titv"] = [titv_class(r, a) for r, a in zip(merged["ref"], merged["alt"])]
    merged["true_median_beta"] = pd.to_numeric(merged["Median_Beta"], errors="raise")

    merged["association_class"] = np.where(
        merged["qval"] < args.positive_q_threshold,
        "significant",
        np.where(merged["qval"] >= args.negative_q_threshold, "nonsignificant", "excluded_intermediate"),
    )
    merged = merged[merged["association_class"] != "excluded_intermediate"].copy()
    annotated, probe_audit = pc.annotate_hm450_probe_geometry(merged, args.hm450_manifest)
    before_probe = len(annotated)
    annotated = annotated[
        ~annotated["hm450_mask_general"].astype(bool)
        & ~annotated["variant_overlaps_probe_or_extension_conservative"].astype(bool)
    ].copy()
    annotated["significant_label"] = annotated["association_class"].eq("significant").astype(int)

    audit = {
        "lead_rows_input": int(input_rows),
        "eligible_before_conservative_probe_filter": int(before_probe),
        "eligible_after_conservative_probe_filter": int(len(annotated)),
        "significant_after_filter": int(annotated["significant_label"].sum()),
        "nonsignificant_after_filter": int((annotated["significant_label"] == 0).sum()),
        "positive_q_threshold": float(args.positive_q_threshold),
        "negative_q_threshold": float(args.negative_q_threshold),
        "probe_audit_before_filter": probe_audit,
    }
    return annotated.reset_index(drop=True), audit


def match_leads(eligible: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = eligible[eligible["significant_label"] == 1].copy()
    negatives = eligible[eligible["significant_label"] == 0].copy()
    if args.max_positive:
        positives = positives.sort_values(["qval", "cpg_id"]).head(args.max_positive).copy()
        LOGGER.warning("SMOKE TEST: retaining only %d significant leads", len(positives))
    if positives.empty or negatives.empty:
        raise ValueError("Significant or nonsignificant eligible lead pool is empty")

    positives = positives.sort_values(["cpg_chr", "cpg_id"]).reset_index(drop=True)
    negatives = negatives.sort_values(["cpg_chr", "cpg_id"]).reset_index(drop=True)
    n_positive, n_negative = len(positives), len(negatives)
    invalid_cost = 1_000_000.0
    unmatched_cost = 100.0
    cost = np.full((n_positive, n_negative + n_positive), invalid_cost, dtype=float)
    rng = np.random.default_rng(args.matching_seed)

    for i, positive in positives.iterrows():
        same_chromosome = negatives["cpg_chr"].eq(positive["cpg_chr"]).to_numpy()
        maf_difference = (negatives["maf"] - positive["maf"]).abs().to_numpy(dtype=float)
        valid = same_chromosome & (maf_difference <= args.maf_caliper)
        distance_difference = (
            negatives["absolute_distance_bp"] - positive["absolute_distance_bp"]
        ).abs().to_numpy(dtype=float)
        beta_difference = (
            negatives["true_median_beta"] - positive["true_median_beta"]
        ).abs().to_numpy(dtype=float)
        substitution_penalty = np.where(
            negatives["ref"].eq(positive["ref"]).to_numpy()
            & negatives["alt"].eq(positive["alt"]).to_numpy(),
            0.0,
            np.where(
                negatives["sbs6"].eq(positive["sbs6"]).to_numpy(),
                0.20,
                np.where(negatives["titv"].eq(positive["titv"]).to_numpy(), 0.50, 0.80),
            ),
        )
        pair_cost = (
            4.0 * maf_difference / args.maf_caliper
            + distance_difference / 500.0
            + beta_difference / 0.50
            + substitution_penalty
            + rng.uniform(0.0, 1e-9, n_negative)
        )
        cost[i, :n_negative] = np.where(valid, pair_cost, invalid_cost)
        cost[i, n_negative + i] = unmatched_cost

    row_indices, column_indices = linear_sum_assignment(cost)
    assignments = [
        (int(i), int(j), float(cost[i, j]))
        for i, j in zip(row_indices, column_indices)
        if j < n_negative and cost[i, j] < unmatched_cost
    ]
    selected_rows: list[dict] = []
    set_rows: list[dict] = []
    for positive_index, negative_index, distance in assignments:
        positive = positives.iloc[positive_index]
        negative = negatives.iloc[negative_index]
        if positive["ref"] == negative["ref"] and positive["alt"] == negative["alt"]:
            tier = "O1_exact_substitution"
        elif positive["sbs6"] == negative["sbs6"]:
            tier = "O2_SBS6"
        elif positive["titv"] == negative["titv"]:
            tier = "O3_TiTv"
        else:
            tier = "O4_chromosome_only"
        match_set_id = f"match_{len(set_rows) + 1:04d}"
        positive_row = positive.to_dict()
        positive_row.update(
            match_set_id=match_set_id,
            match_role="significant",
            match_tier="positive_anchor",
            match_distance=0.0,
        )
        negative_row = negative.to_dict()
        negative_row.update(
            match_set_id=match_set_id,
            match_role="nonsignificant_comparator",
            match_tier=tier,
            match_distance=distance,
        )
        selected_rows.extend([positive_row, negative_row])
        set_rows.append(
            {
                "match_set_id": match_set_id,
                "positive_cpg_id": positive["cpg_id"],
                "positive_variant_id": positive["variant_id"],
                "comparator_cpg_id": negative["cpg_id"],
                "comparator_variant_id": negative["variant_id"],
                "comparator_count": 1,
                "matching_tier": tier,
                "absolute_maf_difference": float(abs(positive["maf"] - negative["maf"])),
                "absolute_distance_difference_bp": float(
                    abs(positive["absolute_distance_bp"] - negative["absolute_distance_bp"])
                ),
                "absolute_baseline_beta_difference": float(
                    abs(positive["true_median_beta"] - negative["true_median_beta"])
                ),
                "optimization_cost": distance,
            }
        )

    if not set_rows:
        raise ValueError("No complete matched sets were constructed")
    matched = pd.DataFrame(selected_rows).reset_index(drop=True)
    sets = pd.DataFrame(set_rows)
    LOGGER.info(
        "Globally matched %d/%d significant leads to %d/%d unique nonsignificant comparators "
        "under same-chromosome MAF caliper %.3f",
        len(sets),
        n_positive,
        int((matched["significant_label"] == 0).sum()),
        n_negative,
        args.maf_caliper,
    )
    return matched, sets


def build_sequences_and_context(matched: pd.DataFrame, pc):
    wt_sequences, alt_sequences = [], []
    mutation_indices_5kb, mutation_indices_1kb = [], []
    for row in matched.itertuples(index=False):
        wt_full = str(row.Healthy_5000bp_DNA).upper()
        offset = int(row.offset_from_cpg_C)
        full_index = pc.FULL_TARGET_C_INDEX + offset
        crop_index = pc.MODEL_TARGET_C_INDEX + offset
        alt_full = wt_full[:full_index] + str(row.alt) + wt_full[full_index + 1 :]
        wt = pc.centered_crop(wt_full, pc.MODEL_WINDOW_SIZE)
        alt = pc.centered_crop(alt_full, pc.MODEL_WINDOW_SIZE)
        differences = [i for i, (a, b) in enumerate(zip(wt, alt)) if a != b]
        if differences != [crop_index] or wt[crop_index] != row.ref or alt[crop_index] != row.alt:
            raise ValueError(f"REF/ALT reconstruction failed for {row.cpg_id}/{row.variant_id}")
        if wt[pc.MODEL_TARGET_C_INDEX : pc.MODEL_TARGET_G_INDEX + 1] != "CG":
            raise ValueError(f"Reference crop lost central CpG for {row.cpg_id}")
        if alt[pc.MODEL_TARGET_C_INDEX : pc.MODEL_TARGET_G_INDEX + 1] != "CG":
            raise ValueError(f"ALT crop altered central CpG for {row.cpg_id}")
        wt_sequences.append(wt)
        alt_sequences.append(alt)
        mutation_indices_5kb.append(full_index)
        mutation_indices_1kb.append(crop_index)

    context_array = matched[pc.TABULAR_FEATURES].to_numpy(dtype=np.float32, copy=True)
    missing_array = matched[pc.MISSING_FEATURES].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(context_array).all() or not np.isin(missing_array, [0.0, 1.0]).all():
        raise ValueError("Invalid context values or missingness indicators")
    cohort = matched.copy()
    cohort["mutation_index_5000_0based"] = mutation_indices_5kb
    cohort["mutation_index_1000_0based"] = mutation_indices_1kb
    cohort["slope_effect_allele"] = "ALT"
    cohort["slope_alt_aligned"] = cohort["slope"].astype(float)
    cohort["shared_variant_cpg_count"] = cohort.groupby("variant_id")["cpg_id"].transform("size")
    return (
        cohort,
        wt_sequences,
        alt_sequences,
        torch.from_numpy(context_array),
        torch.from_numpy(missing_array),
    )


def matching_balance(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ("maf", "absolute_distance_bp", "true_median_beta"):
        anchor = (
            matched.loc[matched["significant_label"] == 1, ["match_set_id", variable]]
            .set_index("match_set_id")[variable]
            .sort_index()
        )
        comparator = (
            matched.loc[matched["significant_label"] == 0]
            .groupby("match_set_id")[variable]
            .mean()
            .sort_index()
        )
        anchor, comparator = anchor.align(comparator, join="inner")
        pooled_sd = np.sqrt((anchor.var(ddof=1) + comparator.var(ddof=1)) / 2.0)
        rows.append(
            {
                "variable": variable,
                "significant_mean": float(anchor.mean()),
                "matched_comparator_set_mean": float(comparator.mean()),
                "mean_paired_difference": float((anchor - comparator).mean()),
                "standardized_mean_difference": (
                    float((anchor.mean() - comparator.mean()) / pooled_sd)
                    if np.isfinite(pooled_sd) and pooled_sd > 0
                    else np.nan
                ),
                "median_absolute_within_set_difference": float((anchor - comparator).abs().median()),
                "n_match_sets": int(len(anchor)),
            }
        )
    return pd.DataFrame(rows)


def point_metrics(frame: pd.DataFrame, score_column: str) -> dict:
    labels = frame["significant_label"].to_numpy(dtype=int)
    scores = frame[score_column].abs().to_numpy(dtype=float)
    if np.unique(labels).size != 2:
        return {}
    baseline = float(labels.mean())
    order = np.argsort(-scores)
    result = {
        "n_rows": int(len(frame)),
        "n_match_sets": int(frame["match_set_id"].nunique()),
        "n_significant": int(labels.sum()),
        "n_nonsignificant": int((labels == 0).sum()),
        "positive_fraction": baseline,
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
    set_wins = []
    for _, group in frame.groupby("match_set_id", sort=False):
        positive_score = group.loc[group["significant_label"] == 1, score_column].abs()
        negative_scores = group.loc[group["significant_label"] == 0, score_column].abs()
        if len(positive_score) == 1 and len(negative_scores):
            set_wins.append(float(positive_score.iloc[0] > negative_scores.median()))
    result["matched_set_positive_above_negative_median_fraction"] = float(np.mean(set_wins))
    for fraction, label in ((0.10, "top_10pct"), (0.20, "top_20pct")):
        count = max(1, int(np.ceil(fraction * len(frame))))
        top_fraction = float(labels[order[:count]].mean())
        result[f"{label}_positive_fraction"] = top_fraction
        result[f"{label}_enrichment_over_baseline"] = top_fraction / baseline
    return result


def fast_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return {"auroc": np.nan, "average_precision": np.nan}

    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    auc = (
        ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)

    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    cumulative_positive = np.cumsum(ordered_labels)
    positive_positions = np.flatnonzero(ordered_labels == 1)
    ap = float(
        np.mean(cumulative_positive[positive_positions] / (positive_positions + 1))
    )
    baseline = n_positive / len(labels)
    output = {"auroc": float(auc), "average_precision": ap}
    for fraction, label in ((0.10, "top_10pct"), (0.20, "top_20pct")):
        count = max(1, int(np.ceil(fraction * len(labels))))
        top_fraction = float(ordered_labels[:count].mean())
        output[f"{label}_positive_fraction"] = top_fraction
        output[f"{label}_enrichment_over_baseline"] = top_fraction / baseline
    return output


def bootstrap_metrics(frame: pd.DataFrame, score_column: str, replicates: int, seed: int) -> dict:
    if replicates == 0:
        return {}
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {}
    groups = []
    for _, group in frame.groupby("match_set_id", sort=False):
        groups.append(
            (
                group["significant_label"].to_numpy(dtype=np.int8),
                group[score_column].abs().to_numpy(dtype=float),
            )
        )
    for _ in tqdm(range(replicates), desc=f"Bootstrap {score_column}", leave=False):
        sampled = rng.integers(0, len(groups), size=len(groups))
        labels = np.concatenate([groups[int(index)][0] for index in sampled])
        scores = np.concatenate([groups[int(index)][1] for index in sampled])
        metrics = fast_binary_metrics(labels, scores)
        for key, value in metrics.items():
            if isinstance(value, float) and np.isfinite(value):
                distributions.setdefault(key, []).append(value)
    intervals = {}
    for key, values in distributions.items():
        low, high = np.quantile(values, [0.025, 0.975])
        intervals[f"{key}_match_set_bootstrap_ci_low"] = float(low)
        intervals[f"{key}_match_set_bootstrap_ci_high"] = float(high)
    return intervals


def within_set_permutation_p(frame: pd.DataFrame, score_column: str, replicates: int, seed: int) -> dict:
    if replicates == 0:
        return {"auroc_within_set_permutation_p": np.nan, "average_precision_within_set_permutation_p": np.nan}
    labels = frame["significant_label"].to_numpy(dtype=int)
    scores = frame[score_column].abs().to_numpy(dtype=float)
    observed_auc = roc_auc_score(labels, scores)
    observed_ap = average_precision_score(labels, scores)
    group_indices = [group.index.to_numpy(dtype=int) for _, group in frame.groupby("match_set_id", sort=False)]
    rng = np.random.default_rng(seed)
    permuted = np.zeros((replicates, len(frame)), dtype=np.int8)
    replicate_indices = np.arange(replicates)
    for indices in group_indices:
        choices = indices[rng.integers(0, len(indices), size=replicates)]
        permuted[replicate_indices, choices] = 1

    n_positive = len(group_indices)
    n_negative = len(frame) - n_positive
    score_ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    positive_rank_sums = permuted @ score_ranks
    null_auc = (
        positive_rank_sums - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)

    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = permuted[:, order]
    cumulative_positive = np.cumsum(ordered_labels, axis=1)
    precisions = cumulative_positive / np.arange(1, len(frame) + 1)[None, :]
    null_ap = (precisions * ordered_labels).sum(axis=1) / n_positive
    exceed_auc = int(np.count_nonzero(null_auc >= observed_auc))
    exceed_ap = int(np.count_nonzero(null_ap >= observed_ap))
    return {
        "auroc_within_set_permutation_p": float((exceed_auc + 1) / (replicates + 1)),
        "average_precision_within_set_permutation_p": float((exceed_ap + 1) / (replicates + 1)),
    }


def compute_all_metrics(aggregate: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    score_columns = ("predicted_delta_m", "predicted_delta_beta")
    for model_index, (model, model_frame) in enumerate(aggregate.groupby("model", sort=False)):
        model_frame = model_frame.reset_index(drop=True)
        for score_index, score_column in enumerate(score_columns):
            seed = args.statistics_seed + model_index * 10000 + score_index * 1000
            metrics = point_metrics(model_frame, score_column)
            metrics.update(bootstrap_metrics(model_frame, score_column, args.bootstrap_replicates, seed))
            metrics.update(within_set_permutation_p(model_frame, score_column, args.permutation_replicates, seed + 101))
            metrics.update(
                model=model,
                score=score_column,
                seed_count=int(model_frame["seed_count"].iloc[0]),
                seeds=str(model_frame["seeds"].iloc[0]),
                comparator_definition=(
                    f"eGTEx lead associations with q>={args.negative_q_threshold:g}, matched without replacement"
                ),
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def save_plots(aggregate: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for model, frame in aggregate.groupby("model", sort=False):
        labels = frame["significant_label"].to_numpy(dtype=int)
        scores = frame["predicted_delta_m"].abs().to_numpy(dtype=float)
        metric = metrics[(metrics["model"] == model) & (metrics["score"] == "predicted_delta_m")].iloc[0]

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
        axes[0].boxplot(
            [scores[labels == 0], scores[labels == 1]],
            tick_labels=["Nonsignificant\ncomparators", "Significant\nleads"],
            showfliers=False,
        )
        axes[0].set_ylabel(r"Absolute predicted $\Delta\hat{M}$")
        axes[0].set_title("Matched score distributions")

        fpr, tpr, _ = roc_curve(labels, scores)
        axes[1].plot(fpr, tpr, color="#1b7837", lw=2)
        axes[1].plot([0, 1], [0, 1], "--", color="0.5", lw=1)
        axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC (AUROC={metric.auroc:.3f})")

        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[2].plot(recall, precision, color="#2166ac", lw=2)
        axes[2].axhline(labels.mean(), ls="--", color="0.5", lw=1)
        axes[2].set(xlabel="Recall", ylabel="Precision", title=f"PR (AP={metric.average_precision:.3f})")
        fig.suptitle(f"eGTEx significant-vs-nonsignificant lead benchmark: {model}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{model}_matched_negative_discrimination.png", dpi=300)
        plt.close(fig)


def write_summary(metrics: pd.DataFrame, matched: pd.DataFrame, path: Path, smoke_test: bool) -> None:
    lines = []
    if smoke_test:
        lines.extend(["SMOKE TEST ONLY -- NOT REPORTABLE", ""])
    lines.extend(
        [
            "SilentMethyl eGTEx matched significant-versus-nonsignificant lead benchmark",
            "Primary score: absolute RC-averaged, cross-seed predicted delta M",
            "Interpretation: discrimination against matched nonsignificant leads, not confirmed causal nulls",
            f"Match sets: {matched['match_set_id'].nunique()}",
            f"Rows: {len(matched)} ({int(matched['significant_label'].sum())} significant; "
            f"{int((matched['significant_label'] == 0).sum())} nonsignificant)",
            "",
        ]
    )
    for row in metrics.itertuples(index=False):
        lines.extend(
            [
                f"Model={row.model}, score={row.score}",
                f"  AUROC={row.auroc:.6f} "
                f"[{getattr(row, 'auroc_match_set_bootstrap_ci_low'):.6f}, "
                f"{getattr(row, 'auroc_match_set_bootstrap_ci_high'):.6f}]",
                f"  AUROC within-set permutation p={row.auroc_within_set_permutation_p:.6g}",
                f"  Average precision={row.average_precision:.6f} "
                f"[{getattr(row, 'average_precision_match_set_bootstrap_ci_low'):.6f}, "
                f"{getattr(row, 'average_precision_match_set_bootstrap_ci_high'):.6f}]",
                f"  AP within-set permutation p={row.average_precision_within_set_permutation_p:.6g}",
                "  Significant above matched-negative median fraction="
                f"{row.matched_set_positive_above_negative_median_fraction:.6f}",
                f"  Top-10% enrichment={row.top_10pct_enrichment_over_baseline:.6f}",
                f"  Top-20% enrichment={row.top_20pct_enrichment_over_baseline:.6f}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines).rstrip() + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    validate_args(args)
    pc = import_positive_control(args.positive_control_script)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible, eligibility_audit = prepare_eligible_leads(args, pc)
    atomic_csv(eligible, output_dir / "eligible_significant_and_nonsignificant_leads.csv")
    matched, match_sets = match_leads(eligible, args)
    atomic_csv(matched, output_dir / "matched_lead_cohort.csv")
    atomic_csv(match_sets, output_dir / "match_sets.csv")
    balance = matching_balance(matched)
    atomic_csv(balance, output_dir / "matching_balance.csv")
    cohort, wt_sequences, alt_sequences, context, missing = build_sequences_and_context(matched, pc)

    device = pc.resolve_device(args.device)
    LOGGER.info("Device=%s models=%s seeds=%s precision=%s", device, args.models, args.seeds, "AMP" if args.amp else "FP32")
    tokenizer = pc.get_tokenizer(args.model_path)
    score_frames, checkpoint_manifest = [], []
    for model in args.models:
        for seed in args.seeds:
            scores, manifest = pc.score_model_seed(
                model,
                seed,
                cohort,
                wt_sequences,
                alt_sequences,
                context,
                missing,
                tokenizer,
                args,
                device,
            )
            score_frames.append(scores)
            checkpoint_manifest.append(manifest)
            atomic_csv(scores, output_dir / model / f"seed{seed}" / "matched_lead_predictions.csv")

    long_scores = pd.concat(score_frames, ignore_index=True)
    atomic_csv(long_scores, output_dir / "matched_lead_predictions_all_models_seeds.csv")
    aggregate = pc.aggregate_seed_predictions(long_scores, cohort)
    atomic_csv(aggregate, output_dir / "matched_lead_predictions_seed_aggregate.csv")
    metrics = compute_all_metrics(aggregate, args)
    atomic_csv(metrics, output_dir / "matched_negative_metrics.csv")
    save_plots(aggregate, metrics, output_dir / "plots")
    write_summary(metrics, matched, output_dir / "matched_negative_summary.txt", args.max_positive > 0)

    tier_counts = matched.loc[matched["significant_label"] == 0, "match_tier"].value_counts().to_dict()
    run_summary = {
        "analysis": "matched significant-versus-nonsignificant eGTEx breast lead associations",
        "reportable_primary_run": args.max_positive == 0,
        "interpretation": (
            "Tests whether absolute SilentMethyl allelic scores discriminate significant from matched "
            "nonsignificant lead associations; nonsignificant leads are comparators, not confirmed nulls."
        ),
        "lead_mqtl_file": str(Path(args.lead_mqtl_file)),
        "lead_mqtl_sha256": pc.sha256_file(args.lead_mqtl_file),
        "test_csv": str(Path(args.test_csv)),
        "test_csv_sha256": pc.sha256_file(args.test_csv),
        "hm450_manifest": str(Path(args.hm450_manifest)),
        "hm450_manifest_sha256": pc.sha256_file(args.hm450_manifest),
        "positive_control_implementation": str(Path(args.positive_control_script)),
        "positive_control_implementation_sha256": pc.sha256_file(args.positive_control_script),
        "eligibility_audit": eligibility_audit,
        "match_set_count": int(matched["match_set_id"].nunique()),
        "matched_row_count": int(len(matched)),
        "comparator_matching_tier_counts": tier_counts,
        "matching_balance": balance.to_dict(orient="records"),
        "matching_without_replacement": True,
        "matching_algorithm": "global linear-sum assignment with unmatched-positive dummy columns",
        "hard_matching_constraints": {
            "same_chromosome": True,
            "maximum_absolute_maf_difference": float(args.maf_caliper),
        },
        "requested_comparators_per_positive": int(args.comparators_per_positive),
        "minimum_comparators_per_positive": int(args.minimum_comparators_per_positive),
        "models": args.models,
        "seeds": args.seeds,
        "precision": "cuda_amp" if args.amp and device.type == "cuda" else "fp32",
        "bootstrap_unit": "matched set",
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "permutation_null": "one significant label randomly reassigned within each matched set",
        "permutation_replicates": int(args.permutation_replicates),
        "checkpoints": checkpoint_manifest,
        "metrics": metrics.to_dict(orient="records"),
    }
    atomic_json(run_summary, output_dir / "run_summary.json")
    LOGGER.info("Finished matched negative-control analysis: %s", output_dir)


if __name__ == "__main__":
    main()
