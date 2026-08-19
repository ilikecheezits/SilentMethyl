#!/usr/bin/env python3
"""Stratify frozen held-out results by genomic and epigenomic context."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


LOGGER = logging.getLogger("biological_context")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_MODELS = ("epi", "sequence", "fusion")
ISLAND_ORDER = ("Island", "Shore", "Shelf", "Open sea")
ATAC_ORDER = ("Q1 low", "Q2", "Q3", "Q4 high", "Missing")
H3K27AC_ORDER = ATAC_ORDER
GENOMIC_REGION_ORDER = ("Promoter/TSS", "UTR", "Gene body", "Intergenic")
DISTANCE_ORDER = ("0--50", "51--100", "101--250", "251--500")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=Path("data/datafiles/test.csv"))
    parser.add_argument(
        "--prediction-template",
        default="results/journal/seed{seed}/{model}/predictions.csv",
    )
    parser.add_argument(
        "--cpg-island-annotation",
        type=Path,
        default=Path("data/HM450.hg38.manifest.CpGIsland.tsv.gz"),
    )
    parser.add_argument(
        "--gencode-gtf",
        type=Path,
        default=Path("data/reference/gencode.v44.annotation.gtf.gz"),
        help=(
            "GENCODE v44 GTF used to assign each target CpG to an exclusive "
            "promoter/TSS, UTR, gene-body, or intergenic category."
        ),
    )
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=Path("results/journal/candidates/candidate_matched_background_statistics.csv"),
    )
    parser.add_argument(
        "--mqtl-path",
        type=Path,
        default=Path("results/journal/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/journal/biological_context"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--block-size-bp", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=20260814)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def normalize_chr(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    return text.where(text.str.startswith("chr"), "chr" + text)


def load_test_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    required = [
        "probeID",
        "chr",
        "pos",
        "Ref_ATAC_Signal",
        "Ref_H3K27ac_Signal",
    ]
    require_columns(header, required, path)
    usecols = required + [
        column
        for column in (
            "Ref_ATAC_Signal_Missing",
            "Ref_H3K27ac_Signal_Missing",
        )
        if column in header.columns
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame["probeID"] = frame["probeID"].astype(str)
    frame["chr"] = normalize_chr(frame["chr"])
    frame["pos"] = pd.to_numeric(frame["pos"], errors="raise").astype(np.int64)
    if frame["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probeID values in {path}")

    for signal, missing_column, stratum_column in (
        ("Ref_ATAC_Signal", "Ref_ATAC_Signal_Missing", "ATAC_Stratum"),
        (
            "Ref_H3K27ac_Signal",
            "Ref_H3K27ac_Signal_Missing",
            "H3K27ac_Stratum",
        ),
    ):
        if missing_column not in frame:
            frame[missing_column] = frame[signal].isna().astype(int)
        missing = frame[missing_column].astype(bool)
        observed = pd.to_numeric(frame.loc[~missing, signal], errors="coerce")
        if observed.isna().any():
            raise ValueError(f"Observed {signal} values contain nonnumeric or missing entries")

        frame[stratum_column] = "Missing"
        if len(observed):
            percentile = observed.rank(method="average", pct=True)
            labels = pd.cut(
                percentile,
                bins=[0.0, 0.25, 0.50, 0.75, 1.0],
                labels=list(ATAC_ORDER[:4]),
                include_lowest=True,
            )
            frame.loc[observed.index, stratum_column] = labels.astype(str)
    return frame


def _merge_intervals(intervals: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    if not intervals:
        return [], []
    intervals.sort()
    merged: list[list[int]] = [[int(intervals[0][0]), int(intervals[0][1])]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], int(end))
        else:
            merged.append([int(start), int(end)])
    return [item[0] for item in merged], [item[1] for item in merged]


def _contains(interval_index: tuple[list[int], list[int]], position_1based: int) -> bool:
    starts, ends = interval_index
    location = bisect.bisect_right(starts, int(position_1based)) - 1
    return location >= 0 and int(position_1based) <= ends[location]


def load_gencode_regions(
    path: Path,
    chromosomes: set[str],
) -> dict[str, dict[str, tuple[list[int], list[int]]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw: dict[str, dict[str, list[tuple[int, int]]]] = {
        chrom: {"promoter": [], "utr": [], "transcript": []}
        for chrom in chromosomes
    }
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom = fields[0] if fields[0].startswith("chr") else f"chr{fields[0]}"
            if chrom not in raw:
                continue
            feature = fields[2]
            if feature not in {"transcript", "UTR"}:
                continue
            start, end = int(fields[3]), int(fields[4])
            strand = fields[6]
            if feature == "transcript":
                # Use the union of annotated transcript spans for the gene-body
                # category.  A GTF gene span can bridge bases that are outside
                # every transcript when a gene has separated isoforms, which
                # incorrectly inflates the gene-body class.
                raw[chrom]["transcript"].append((start, end))
                tss = start if strand == "+" else end
                if strand == "+":
                    promoter = (max(1, tss - 1500), tss + 200)
                else:
                    promoter = (max(1, tss - 200), tss + 1500)
                raw[chrom]["promoter"].append(promoter)
            elif feature == "UTR":
                raw[chrom]["utr"].append((start, end))

    return {
        chrom: {name: _merge_intervals(values) for name, values in groups.items()}
        for chrom, groups in raw.items()
    }


def annotate_genomic_region(test: pd.DataFrame, gtf_path: Path) -> pd.DataFrame:
    intervals = load_gencode_regions(gtf_path, set(test["chr"].astype(str)))
    labels: list[str] = []
    overlap_sets: list[str] = []
    promoter_hits: list[bool] = []
    utr_hits: list[bool] = []
    transcript_hits: list[bool] = []
    for row in test[["chr", "pos"]].itertuples(index=False):
        # Model/HM450 positions are 0-based; GTF coordinates are 1-based,
        # closed intervals.
        position_1based = int(row.pos) + 1
        chrom = str(row.chr)
        groups = intervals.get(chrom, {})
        promoter_hit = _contains(groups.get("promoter", ([], [])), position_1based)
        utr_hit = _contains(groups.get("utr", ([], [])), position_1based)
        transcript_hit = _contains(groups.get("transcript", ([], [])), position_1based)
        promoter_hits.append(promoter_hit)
        utr_hits.append(utr_hit)
        transcript_hits.append(transcript_hit)

        # The categories are deliberately exclusive. A base can overlap the
        # promoter of one transcript and the UTR/body of another, so priority
        # must be applied after recording all overlaps.
        if promoter_hit:
            label = "Promoter/TSS"
        elif utr_hit:
            label = "UTR"
        elif transcript_hit:
            label = "Gene body"
        else:
            label = "Intergenic"
        labels.append(label)
        names = []
        if promoter_hit:
            names.append("Promoter/TSS")
        if utr_hit:
            names.append("UTR")
        if transcript_hit:
            names.append("Transcript span")
        overlap_sets.append("; ".join(names) if names else "None")
    annotated = test.copy()
    annotated["Genomic_Region"] = labels
    annotated["Promoter_TSS_Overlap"] = promoter_hits
    annotated["UTR_Overlap"] = utr_hits
    annotated["Transcript_Span_Overlap"] = transcript_hits
    annotated["Genomic_Region_Overlap_Set"] = overlap_sets
    return annotated


def load_island_annotation(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CpG-island annotation: {path}. Download the file listed in instructions.md."
        )
    header = pd.read_csv(path, sep="\t", nrows=0)
    probe_column = next(
        (name for name in ("probeID", "Probe_ID", "IlmnID", "Name") if name in header.columns),
        None,
    )
    relation_column = next(
        (
            name
            for name in (
                "CGIposition",
                "Relation_to_UCSC_CpG_Island",
                "Relation_to_Island",
            )
            if name in header.columns
        ),
        None,
    )
    if probe_column is None or relation_column is None:
        raise ValueError(
            f"Could not identify probe/relation columns in {path}; columns={list(header.columns)}"
        )
    frame = pd.read_csv(path, sep="\t", usecols=[probe_column, relation_column])
    frame = frame.rename(columns={probe_column: "probeID", relation_column: "CGIposition"})
    frame["probeID"] = frame["probeID"].astype(str)
    if frame["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probeID values in {path}")

    raw = frame["CGIposition"].fillna("NA").astype(str).str.strip()

    def collapse(value: str) -> str:
        lower = value.lower().replace("-", "_").replace(" ", "_")
        if lower in {"na", "nan", "none", "", "opensea", "open_sea"}:
            return "Open sea"
        if "shore" in lower:
            return "Shore"
        if "shelf" in lower:
            return "Shelf"
        if "island" in lower or lower == "cgi":
            return "Island"
        return "Unclassified"

    frame["CpG_Island_Context"] = raw.map(collapse)
    return frame[["probeID", "CpG_Island_Context"]]


def load_predictions(
    template: str,
    seeds: list[int],
    models: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    required = [
        "probeID",
        "chr",
        "pos",
        "true_beta",
        "true_m",
        "binary_true",
        "pred_beta_rc_avg",
        "pred_m_rc_avg",
        "class_prob_rc_avg",
    ]
    rows: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for seed in seeds:
        for model in models:
            path = Path(template.format(seed=seed, model=model))
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            require_columns(frame, required, path)
            frame = frame[required].copy()
            frame["probeID"] = frame["probeID"].astype(str)
            frame["chr"] = normalize_chr(frame["chr"])
            frame["pos"] = pd.to_numeric(frame["pos"], errors="raise").astype(np.int64)
            if frame["probeID"].duplicated().any():
                raise ValueError(f"Duplicate probeID values in {path}")
            frame["Seed"] = int(seed)
            frame["Model"] = str(model)
            rows.append(frame)
            hashes[path.as_posix()] = sha256_file(path)

    long = pd.concat(rows, ignore_index=True)
    reference = long[(long["Seed"] == seeds[0]) & (long["Model"] == models[0])]
    reference = reference.sort_values("probeID").reset_index(drop=True)
    for (seed, model), current in long.groupby(["Seed", "Model"], sort=False):
        current = current.sort_values("probeID").reset_index(drop=True)
        if current["probeID"].tolist() != reference["probeID"].tolist():
            raise ValueError(f"Held-out probe set differs for seed={seed}, model={model}")
        for column in ("true_beta", "true_m", "binary_true"):
            if not np.allclose(current[column], reference[column], rtol=0, atol=1e-10):
                raise ValueError(f"Truth differs for seed={seed}, model={model}, column={column}")
    return long, hashes


def ensemble_predictions(long: pd.DataFrame) -> pd.DataFrame:
    truth = long.sort_values(["Seed", "Model"]).drop_duplicates("probeID")[[
        "probeID", "chr", "pos", "true_beta", "true_m", "binary_true"
    ]]
    means = (
        long.groupby(["Model", "probeID"], as_index=False)[
            ["pred_beta_rc_avg", "pred_m_rc_avg", "class_prob_rc_avg"]
        ]
        .mean()
    )
    return means.merge(truth, on="probeID", validate="many_to_one")


def safe_auc(y: pd.Series, scores: pd.Series) -> float:
    labels = y.to_numpy(dtype=int)
    if len(labels) < 2 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores.to_numpy(dtype=float)))


def metric_rows(frame: pd.DataFrame, grouping: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (model, stratum), group in frame.groupby(["Model", grouping], sort=False):
        beta_error = group["pred_beta_rc_avg"] - group["true_beta"]
        m_error = group["pred_m_rc_avg"] - group["true_m"]
        rows.append({
            "Grouping": grouping,
            "Stratum": str(stratum),
            "Model": str(model),
            "N_CpGs": int(len(group)),
            "N_Positive": int(group["binary_true"].sum()),
            "Beta_MAE": float(beta_error.abs().mean()),
            "Beta_RMSE": float(np.sqrt(np.mean(np.square(beta_error)))),
            "M_MAE": float(m_error.abs().mean()),
            "M_RMSE": float(np.sqrt(np.mean(np.square(m_error)))),
            "ROC_AUC": safe_auc(group["binary_true"], group["class_prob_rc_avg"]),
        })
    return rows


def paired_gain_frame(annotated: pd.DataFrame) -> pd.DataFrame:
    required_models = {"sequence", "fusion"}
    available = set(annotated["Model"].unique())
    if not required_models.issubset(available):
        raise ValueError(f"Need sequence and fusion predictions; available={sorted(available)}")
    truth = annotated.drop_duplicates("probeID")[[
        "probeID",
        "chr",
        "pos",
        "true_beta",
        "CpG_Island_Context",
        "ATAC_Stratum",
        "H3K27ac_Stratum",
        "Genomic_Region",
    ]]
    wide = annotated.pivot(
        index="probeID", columns="Model", values="pred_beta_rc_avg"
    ).reset_index()
    frame = truth.merge(wide, on="probeID", validate="one_to_one")
    frame["Sequence_Absolute_Error"] = (frame["sequence"] - frame["true_beta"]).abs()
    frame["Fusion_Absolute_Error"] = (frame["fusion"] - frame["true_beta"]).abs()
    frame["Fusion_Minus_Sequence_Absolute_Error"] = (
        frame["Fusion_Absolute_Error"] - frame["Sequence_Absolute_Error"]
    )
    return frame


def block_interval(
    group: pd.DataFrame,
    block_size_bp: int,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    values = group["Fusion_Minus_Sequence_Absolute_Error"].to_numpy(dtype=float)
    block_labels = (
        group["chr"].astype(str)
        + ":"
        + (group["pos"].astype(np.int64) // block_size_bp).astype(str)
    )
    codes, unique = pd.factorize(block_labels, sort=True)
    n_blocks = len(unique)
    observed = float(np.mean(values))
    if n_blocks < 2 or replicates <= 0:
        return {
            "Fusion_Minus_Sequence_Beta_MAE": observed,
            "Bootstrap_CI_Low": float("nan"),
            "Bootstrap_CI_High": float("nan"),
            "Bootstrap_Probability_Difference_Greater_Equal_Zero": float("nan"),
            "N_Genomic_Blocks": int(n_blocks),
        }

    counts_by_block = np.bincount(codes, minlength=n_blocks).astype(float)
    sums_by_block = np.bincount(codes, weights=values, minlength=n_blocks).astype(float)
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(
        n_blocks,
        np.full(n_blocks, 1.0 / n_blocks),
        size=replicates,
    ).astype(float)
    denominator = multiplicity @ counts_by_block
    distribution = (multiplicity @ sums_by_block) / denominator
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "Fusion_Minus_Sequence_Beta_MAE": observed,
        "Bootstrap_CI_Low": float(low),
        "Bootstrap_CI_High": float(high),
        "Bootstrap_Probability_Difference_Greater_Equal_Zero": float(
            np.mean(distribution >= 0)
        ),
        "N_Genomic_Blocks": int(n_blocks),
    }


def gain_rows(
    frame: pd.DataFrame,
    grouping: str,
    block_size_bp: int,
    replicates: int,
    random_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (stratum, group) in enumerate(frame.groupby(grouping, sort=False)):
        sequence_mae = float(group["Sequence_Absolute_Error"].mean())
        fusion_mae = float(group["Fusion_Absolute_Error"].mean())
        row: dict[str, object] = {
            "Grouping": grouping,
            "Stratum": str(stratum),
            "N_CpGs": int(len(group)),
            "Sequence_Beta_MAE": sequence_mae,
            "Fusion_Beta_MAE": fusion_mae,
            "Sequence_Minus_Fusion_Beta_MAE": sequence_mae - fusion_mae,
        }
        row.update(block_interval(
            group,
            block_size_bp=block_size_bp,
            replicates=replicates,
            seed=random_seed + 1000 * (index + 1),
        ))
        rows.append(row)
    return rows


def distance_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-0.001, 50, 100, 250, 500],
        labels=list(DISTANCE_ORDER),
        include_lowest=True,
    )


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3 or np.unique(a[finite]).size < 2 or np.unique(b[finite]).size < 2:
        return float("nan")
    return float(spearmanr(a[finite], b[finite]).statistic)


def summarize_variant_distance(
    candidate_path: Path,
    mqtl_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    candidates = pd.read_csv(candidate_path)
    candidate_distance = next(
        (
            column
            for column in (
                "Absolute_Distance_From_Target_CpG",
                "Absolute_Distance_To_CpG",
                "absolute_distance_bp",
            )
            if column in candidates.columns
        ),
        None,
    )
    require_columns(candidates, ["Predicted_Delta_Beta"], candidate_path)
    if candidate_distance is None:
        raise ValueError(f"Could not identify candidate distance column in {candidate_path}")
    candidates["Distance_Bin"] = distance_bin(candidates[candidate_distance])

    rows: list[dict[str, object]] = []
    for label in DISTANCE_ORDER:
        group = candidates[candidates["Distance_Bin"].astype(str) == label]
        if group.empty:
            continue
        absolute = group["Predicted_Delta_Beta"].abs()
        rows.append({
            "Dataset": "TCGA synonymous candidates",
            "Model": "fusion",
            "Distance_Bin": label,
            "N_Associations": int(len(group)),
            "Median_Absolute_Predicted_Response": float(absolute.median()),
            "Mean_Absolute_Predicted_Response": float(absolute.mean()),
            "Signed_Spearman_With_Observed_Effect": float("nan"),
            "Absolute_Spearman_With_Observed_Effect": float("nan"),
            "Direction_Concordance": float("nan"),
        })

    if not mqtl_path.is_file():
        raise FileNotFoundError(mqtl_path)
    mqtl = pd.read_csv(mqtl_path)
    require_columns(
        mqtl,
        ["model", "absolute_distance_bp", "predicted_delta_m", "slope_alt_aligned"],
        mqtl_path,
    )
    mqtl = mqtl[mqtl["model"].astype(str) == "fusion"].copy()
    mqtl["Distance_Bin"] = distance_bin(mqtl["absolute_distance_bp"])
    for label in DISTANCE_ORDER:
        group = mqtl[mqtl["Distance_Bin"].astype(str) == label]
        if group.empty:
            continue
        predicted = group["predicted_delta_m"]
        observed = group["slope_alt_aligned"]
        rows.append({
            "Dataset": "eGTEx positive control",
            "Model": "fusion",
            "Distance_Bin": label,
            "N_Associations": int(len(group)),
            "Median_Absolute_Predicted_Response": float(predicted.abs().median()),
            "Mean_Absolute_Predicted_Response": float(predicted.abs().mean()),
            "Signed_Spearman_With_Observed_Effect": safe_spearman(predicted, observed),
            "Absolute_Spearman_With_Observed_Effect": safe_spearman(
                predicted.abs(), observed.abs()
            ),
            "Direction_Concordance": float(
                np.mean(np.sign(predicted.to_numpy(float)) == np.sign(observed.to_numpy(float)))
            ),
        })
    summary = pd.DataFrame(rows)
    return candidates, mqtl, summary


def summarize_variant_context(
    candidates: pd.DataFrame,
    mqtl: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    context_columns = [
        "Genomic_Region",
        "CpG_Island_Context",
        "ATAC_Stratum",
        "H3K27ac_Stratum",
    ]
    candidate_annotated = candidates.merge(
        assignments[["probeID", *context_columns]],
        on="probeID",
        how="left",
        validate="many_to_one",
    )
    cpg_column = "cpg_id" if "cpg_id" in mqtl.columns else "probeID"
    if cpg_column not in mqtl.columns:
        raise ValueError("Could not identify the mQTL target-CpG column")
    mqtl_annotated = mqtl.merge(
        assignments.rename(columns={"probeID": cpg_column})[
            [cpg_column, *context_columns]
        ],
        on=cpg_column,
        how="left",
        validate="many_to_one",
    )

    rows: list[dict[str, object]] = []
    candidate_groupings = ["Variant_Genomic_Region", *context_columns]
    for grouping in candidate_groupings:
        for stratum, group in candidate_annotated.groupby(grouping, dropna=False, sort=False):
            effect = pd.to_numeric(group["Predicted_Delta_Beta"], errors="coerce")
            effect = effect[np.isfinite(effect)]
            if effect.empty:
                continue
            rows.append({
                "Dataset": "TCGA synonymous candidates",
                "Grouping": grouping,
                "Stratum": "Unclassified" if pd.isna(stratum) else str(stratum),
                "N_Associations": int(len(effect)),
                "N_Unique_Variants": int(
                    group["Variant_UID"].nunique()
                    if "Variant_UID" in group.columns
                    else len(group)
                ),
                "Median_Absolute_Predicted_Response": float(effect.abs().median()),
                "Mean_Absolute_Predicted_Response": float(effect.abs().mean()),
                "Signed_Spearman_With_Observed_Effect": float("nan"),
                "Absolute_Spearman_With_Observed_Effect": float("nan"),
                "Direction_Concordance": float("nan"),
            })

        if grouping == "Variant_Genomic_Region":
            continue
        for stratum, group in mqtl_annotated.groupby(grouping, dropna=False, sort=False):
            predicted = pd.to_numeric(group["predicted_delta_m"], errors="coerce")
            observed = pd.to_numeric(group["slope_alt_aligned"], errors="coerce")
            finite = np.isfinite(predicted) & np.isfinite(observed)
            predicted, observed = predicted[finite], observed[finite]
            if predicted.empty:
                continue
            rows.append({
                "Dataset": "eGTEx positive control",
                "Grouping": grouping,
                "Stratum": "Unclassified" if pd.isna(stratum) else str(stratum),
                "N_Associations": int(len(predicted)),
                "N_Unique_Variants": int(
                    group.loc[finite, "variant_id"].nunique()
                    if "variant_id" in group.columns
                    else len(predicted)
                ),
                "Median_Absolute_Predicted_Response": float(predicted.abs().median()),
                "Mean_Absolute_Predicted_Response": float(predicted.abs().mean()),
                "Signed_Spearman_With_Observed_Effect": safe_spearman(predicted, observed),
                "Absolute_Spearman_With_Observed_Effect": safe_spearman(
                    predicted.abs(), observed.abs()
                ),
                "Direction_Concordance": float(
                    np.mean(np.sign(predicted.to_numpy()) == np.sign(observed.to_numpy()))
                ),
            })
    return pd.DataFrame(rows)


def annotate_candidate_variant_regions(
    candidates: pd.DataFrame,
    gtf_path: Path,
) -> pd.DataFrame:
    require_columns(candidates, ["chr"], "candidate table")
    if "Variant_Position_0based" in candidates.columns:
        position = pd.to_numeric(
            candidates["Variant_Position_0based"], errors="raise"
        ).astype(np.int64)
    elif "Variant_Position_1based" in candidates.columns:
        position = (
            pd.to_numeric(candidates["Variant_Position_1based"], errors="raise")
            .astype(np.int64)
            - 1
        )
    else:
        raise ValueError(
            "Candidate table requires Variant_Position_0based or "
            "Variant_Position_1based for variant-region annotation"
        )
    positions = pd.DataFrame(
        {
            "chr": normalize_chr(candidates["chr"]),
            "pos": position,
        },
        index=candidates.index,
    )
    annotated_positions = annotate_genomic_region(positions, gtf_path)
    annotated = candidates.copy()
    annotated["Variant_Genomic_Region"] = annotated_positions[
        "Genomic_Region"
    ].to_numpy()
    for source, target in (
        ("Promoter_TSS_Overlap", "Variant_Promoter_TSS_Overlap"),
        ("UTR_Overlap", "Variant_UTR_Overlap"),
        ("Transcript_Span_Overlap", "Variant_Transcript_Span_Overlap"),
        ("Genomic_Region_Overlap_Set", "Variant_Genomic_Region_Overlap_Set"),
    ):
        annotated[target] = annotated_positions[source].to_numpy()
    return annotated


def genomic_region_audit(
    loci: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Return final-category and raw-overlap counts for manual verification."""
    rows: list[dict[str, object]] = []
    specifications = (
        (
            "Held-out target CpGs",
            loci,
            "Genomic_Region",
            "Genomic_Region_Overlap_Set",
        ),
        (
            "Candidate variant positions",
            candidates,
            "Variant_Genomic_Region",
            "Variant_Genomic_Region_Overlap_Set",
        ),
    )
    for dataset, frame, category_column, overlap_column in specifications:
        total = int(len(frame))
        for value, count in frame[category_column].value_counts(dropna=False).items():
            rows.append({
                "Dataset": dataset,
                "Audit_Type": "Exclusive final category",
                "Label": "Missing" if pd.isna(value) else str(value),
                "N": int(count),
                "Percent": 100.0 * int(count) / total if total else float("nan"),
            })
        for value, count in frame[overlap_column].value_counts(dropna=False).items():
            rows.append({
                "Dataset": dataset,
                "Audit_Type": "Raw annotation overlap set",
                "Label": "Missing" if pd.isna(value) else str(value),
                "N": int(count),
                "Percent": 100.0 * int(count) / total if total else float("nan"),
            })
    return pd.DataFrame(rows)


def plot_context_gain(gain: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(3.45, 8.4), sharey=False)
    specifications = [
        ("Genomic_Region", list(GENOMIC_REGION_ORDER), "Genomic region"),
        ("CpG_Island_Context", list(ISLAND_ORDER), "CpG-island context"),
        ("ATAC_Stratum", list(ATAC_ORDER), "MCF-10A ATAC signal"),
        ("H3K27ac_Stratum", list(H3K27AC_ORDER), "MCF-10A H3K27ac signal"),
    ]
    for axis, (grouping, order, title) in zip(axes, specifications):
        current = gain[gain["Grouping"] == grouping].copy()
        current["Stratum"] = pd.Categorical(current["Stratum"], categories=order, ordered=True)
        current = current.dropna(subset=["Stratum"]).sort_values("Stratum")
        x = np.arange(len(current))
        improvement = current["Sequence_Minus_Fusion_Beta_MAE"].to_numpy(float)
        ci_low = -current["Bootstrap_CI_High"].to_numpy(float)
        ci_high = -current["Bootstrap_CI_Low"].to_numpy(float)
        yerr = np.vstack([improvement - ci_low, ci_high - improvement])
        yerr = np.maximum(yerr, 0)
        axis.errorbar(
            x,
            improvement,
            yerr=yerr,
            fmt="o",
            color="#2f6f9f",
            ecolor="#7aa6c2",
            capsize=3,
            linewidth=1.2,
        )
        axis.axhline(0, color="0.35", linewidth=0.9, linestyle="--")
        axis.set_xticks(x)
        axis.set_xticklabels(current["Stratum"].astype(str), rotation=22, ha="right")
        axis.set_title(title, loc="left", fontsize=9)
        axis.set_ylabel(r"MAE gain")
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Target-CpG context")
    fig.suptitle("Gain from adding reference context", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.8)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_distance_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    candidate = summary[summary["Dataset"] == "TCGA synonymous candidates"].copy()
    mqtl = summary[summary["Dataset"] == "eGTEx positive control"].copy()
    for current in (candidate, mqtl):
        current["Distance_Bin"] = pd.Categorical(
            current["Distance_Bin"], categories=list(DISTANCE_ORDER), ordered=True
        )
        current.sort_values("Distance_Bin", inplace=True)

    x = np.arange(len(candidate))
    axes[0].plot(
        x,
        candidate["Median_Absolute_Predicted_Response"],
        marker="o",
        color="#7b3294",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(candidate["Distance_Bin"].astype(str), rotation=25, ha="right")
    axes[0].set_ylabel(r"Median absolute predicted $\Delta\hat{\beta}$")
    axes[0].set_title("TCGA synonymous candidates")
    for location, (_, row) in zip(x, candidate.iterrows()):
        axes[0].annotate(
            f"n={int(row['N_Associations'])}",
            (location, row["Median_Absolute_Predicted_Response"]),
            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7,
        )

    x = np.arange(len(mqtl))
    axes[1].plot(
        x,
        mqtl["Direction_Concordance"],
        marker="o",
        color="#20854e",
    )
    axes[1].axhline(0.5, color="0.4", linestyle="--", linewidth=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mqtl["Distance_Bin"].astype(str), rotation=25, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Direction concordance")
    axes[1].set_title("eGTEx positive control")
    for location, (_, row) in zip(x, mqtl.iterrows()):
        axes[1].annotate(
            f"n={int(row['N_Associations'])}",
            (location, row["Direction_Concordance"]),
            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7,
        )
    for axis in axes:
        axis.set_xlabel("Absolute variant--CpG distance (bp)")
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Predicted sequence responses by variant--CpG distance")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args.block_size_bp <= 0 or args.bootstrap_replicates < 0:
        raise ValueError("Block size must be positive and bootstrap replicates nonnegative")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    test = load_test_metadata(args.test_path)
    island = load_island_annotation(args.cpg_island_annotation)
    test = test.merge(island, on="probeID", how="left", validate="one_to_one")
    test["CpG_Island_Context"] = test["CpG_Island_Context"].fillna("Unclassified")
    test = annotate_genomic_region(test, args.gencode_gtf)

    long, prediction_hashes = load_predictions(
        args.prediction_template, args.seeds, args.models
    )
    ensemble = ensemble_predictions(long)
    annotated = ensemble.merge(
        test[[
            "probeID",
            "Ref_ATAC_Signal",
            "Ref_ATAC_Signal_Missing",
            "ATAC_Stratum",
            "Ref_H3K27ac_Signal",
            "Ref_H3K27ac_Signal_Missing",
            "H3K27ac_Stratum",
            "CpG_Island_Context",
            "Genomic_Region",
        ]],
        on="probeID",
        how="inner",
        validate="many_to_one",
    )
    expected = ensemble["probeID"].nunique()
    observed = annotated["probeID"].nunique()
    if observed != expected:
        raise ValueError(f"Annotated {observed} of {expected} held-out probes")

    context_rows = []
    context_rows.extend(metric_rows(annotated, "Genomic_Region"))
    context_rows.extend(metric_rows(annotated, "CpG_Island_Context"))
    context_rows.extend(metric_rows(annotated, "ATAC_Stratum"))
    context_rows.extend(metric_rows(annotated, "H3K27ac_Stratum"))
    context_metrics = pd.DataFrame(context_rows)
    atomic_csv(context_metrics, output / "locus_metrics_by_context.csv")

    paired = paired_gain_frame(annotated)
    gain = pd.DataFrame(
        gain_rows(
            paired,
            "Genomic_Region",
            args.block_size_bp,
            args.bootstrap_replicates,
            args.random_seed,
        )
        + gain_rows(
            paired,
            "CpG_Island_Context",
            args.block_size_bp,
            args.bootstrap_replicates,
            args.random_seed + 100_000,
        )
        + gain_rows(
            paired,
            "ATAC_Stratum",
            args.block_size_bp,
            args.bootstrap_replicates,
            args.random_seed + 200_000,
        )
        + gain_rows(
            paired,
            "H3K27ac_Stratum",
            args.block_size_bp,
            args.bootstrap_replicates,
            args.random_seed + 300_000,
        )
    )
    atomic_csv(gain, output / "fusion_gain_by_context.csv")

    locus_assignments = test[[
        "probeID",
        "chr",
        "pos",
        "Genomic_Region",
        "Promoter_TSS_Overlap",
        "UTR_Overlap",
        "Transcript_Span_Overlap",
        "Genomic_Region_Overlap_Set",
        "CpG_Island_Context",
        "Ref_ATAC_Signal",
        "Ref_ATAC_Signal_Missing",
        "ATAC_Stratum",
        "Ref_H3K27ac_Signal",
        "Ref_H3K27ac_Signal_Missing",
        "H3K27ac_Stratum",
    ]].copy()
    atomic_csv(locus_assignments, output / "heldout_cpg_context_assignments.csv")

    candidates, mqtl, distance_summary = summarize_variant_distance(
        args.candidate_path, args.mqtl_path
    )
    candidates = annotate_candidate_variant_regions(candidates, args.gencode_gtf)
    assignment_audit = genomic_region_audit(locus_assignments, candidates)
    atomic_csv(
        assignment_audit,
        output / "genomic_region_classification_audit.csv",
    )
    atomic_csv(distance_summary, output / "variant_response_by_distance.csv")
    response_context = summarize_variant_context(
        candidates,
        mqtl,
        locus_assignments,
    )
    atomic_csv(response_context, output / "variant_response_by_context.csv")

    plot_context_gain(gain, output / "plots" / "fusion_gain_by_context.png")
    plot_distance_summary(
        distance_summary,
        output / "plots" / "variant_response_by_distance.png",
    )

    payload = {
        "analysis_status": "COMPLETE",
        "interpretation": (
            "Post hoc descriptive stratification of frozen held-out predictions; "
            "not an additional training experiment or causal analysis."
        ),
        "seeds": [int(seed) for seed in args.seeds],
        "models": list(args.models),
        "heldout_cpg_count": int(expected),
        "candidate_count": int(len(candidates)),
        "mqtl_fusion_association_count": int(len(mqtl)),
        "block_size_bp": int(args.block_size_bp),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "random_seed": int(args.random_seed),
        "inputs": {
            args.test_path.as_posix(): sha256_file(args.test_path),
            args.cpg_island_annotation.as_posix(): sha256_file(args.cpg_island_annotation),
            args.gencode_gtf.as_posix(): sha256_file(args.gencode_gtf),
            args.candidate_path.as_posix(): sha256_file(args.candidate_path),
            args.mqtl_path.as_posix(): sha256_file(args.mqtl_path),
            **prediction_hashes,
        },
        "distance_bins_bp": list(DISTANCE_ORDER),
        "atac_strata": list(ATAC_ORDER),
        "h3k27ac_strata": list(H3K27AC_ORDER),
        "genomic_region_strata": list(GENOMIC_REGION_ORDER),
        "genomic_region_definition": (
            "Exclusive hierarchy: GENCODE v44 transcript TSS -1500/+200 bp; "
            "then explicit GENCODE UTR features; then the union of GENCODE "
            "transcript spans; otherwise intergenic. Input positions are "
            "0-based and converted to the GTF's 1-based closed coordinates."
        ),
        "genomic_region_audit_file": (
            output / "genomic_region_classification_audit.csv"
        ).as_posix(),
        "cpg_island_strata": list(ISLAND_ORDER) + ["Unclassified"],
    }
    atomic_json(payload, output / "run_summary.json")

    print("SilentMethyl biological-context analysis")
    print(f"  held-out CpGs: {expected}")
    print(f"  candidate rows: {len(candidates)}")
    print(f"  fusion mQTL associations: {len(mqtl)}")
    print(f"  output: {output}")


if __name__ == "__main__":
    main()