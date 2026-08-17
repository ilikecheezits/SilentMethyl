#!/usr/bin/env python3
"""QC held-out coverage and the HM450 probe mask."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


NORMAL_BARCODE = re.compile(
    r"^TCGA[-.][A-Z0-9]{2}[-.][A-Z0-9]{4}[-.]11[A-Z0-9](?:[-.]|$)",
    flags=re.IGNORECASE,
)
THRESHOLDS = [("all", 1), (">=49/97", 49), (">=78/97", 78),
              (">=88/97", 88), (">=93/97", 93), ("97/97", 97)]
MODELS = ("epi", "sequence", "fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("all", "coverage", "manifest"), default="all",
        help="Run both audits or only one component.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("data/TCGA-BRCA.methylation450.tsv.gz"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/datafiles/test.csv"),
    )
    parser.add_argument(
        "--prediction-template",
        default="results/journal/seed42/{model}/predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/journal/target_qc"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/HM450.hg38.manifest.tsv.gz"),
    )
    parser.add_argument(
        "--split-csvs",
        nargs="*",
        type=Path,
        default=[
            Path("data/datafiles/train.csv"),
            Path("data/datafiles/val.csv"),
            Path("data/datafiles/test.csv"),
        ],
    )
    parser.add_argument("--chunksize", type=int, default=50_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coerce_boolean(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    true_values = {"true", "t", "1", "yes", "y"}
    false_values = {"false", "f", "0", "no", "n", "", "<na>", "nan"}
    unexpected = sorted(set(normalized.dropna()) - true_values - false_values)
    if unexpected:
        raise ValueError(f"Unexpected values in {column_name}: {unexpected[:10]}")
    return normalized.isin(true_values)


def audit_manifest(args: argparse.Namespace) -> dict:
    header = pd.read_csv(args.manifest, sep="\t", nrows=0)
    probe_column = next(
        (column for column in ("probeID", "IlmnID", "Name") if column in header.columns),
        None,
    )
    if probe_column is None or "MASK_general" not in header.columns:
        raise ValueError(f"{args.manifest} requires a probe ID column and MASK_general")

    component_columns = [
        column for column in header.columns
        if column.startswith("MASK_") and column != "MASK_general"
    ]
    usecols = [probe_column, "MASK_general", *component_columns]
    manifest = pd.read_csv(args.manifest, sep="\t", usecols=usecols, low_memory=False)
    manifest[probe_column] = manifest[probe_column].astype(str)
    if manifest[probe_column].duplicated().any():
        raise ValueError("HM450 manifest contains duplicate probe identifiers")

    general = coerce_boolean(manifest["MASK_general"], "MASK_general")
    result = {
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_rows": int(len(manifest)),
        "mask_general_source": "provider-supplied aggregate flag; not reconstructed",
        "mask_general_false": int((~general).sum()),
        "mask_general_true": int(general.sum()),
        "component_true_counts": {
            column: int(coerce_boolean(manifest[column], column).sum())
            for column in component_columns
        },
        "split_audits": {},
    }

    mapping = pd.DataFrame({
        "probeID": manifest[probe_column],
        "MASK_general": general,
    })
    for split_path in args.split_csvs:
        probes = pd.read_csv(split_path, usecols=["probeID"])
        probes["probeID"] = probes["probeID"].astype(str)
        merged = probes.merge(mapping, on="probeID", how="left", validate="one_to_one")
        result["split_audits"][split_path.stem] = {
            "path": str(split_path),
            "rows": int(len(probes)),
            "missing_manifest": int(merged["MASK_general"].isna().sum()),
            "masked": int(merged["MASK_general"].fillna(False).sum()),
        }

    output = args.output_dir / "hm450_manifest_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("HM450 manifest audit:")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved: {output}")
    return result


def matrix_columns(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return handle.readline().rstrip("\n\r").split("\t")


def compute_coverage(args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    test_ids = pd.read_csv(args.test, usecols=["probeID"])["probeID"].astype(str)
    if test_ids.duplicated().any():
        raise ValueError("test.csv contains duplicate probeID values")
    wanted = set(test_ids)

    columns = matrix_columns(args.matrix)
    probe_column = columns[0]
    normal_columns = [column for column in columns[1:] if NORMAL_BARCODE.match(column)]
    if len(normal_columns) != 97:
        raise ValueError(f"Expected 97 normal columns; found {len(normal_columns)}")

    pieces: list[pd.DataFrame] = []
    usecols = [probe_column] + normal_columns
    for chunk in pd.read_csv(
        args.matrix,
        sep="\t",
        usecols=usecols,
        chunksize=args.chunksize,
        low_memory=False,
    ):
        chunk[probe_column] = chunk[probe_column].astype(str)
        selected = chunk.loc[chunk[probe_column].isin(wanted)]
        if selected.empty:
            continue
        pieces.append(pd.DataFrame({
            "probeID": selected[probe_column].to_numpy(),
            "n_observed_normals": selected[normal_columns].notna().sum(axis=1).to_numpy(),
        }))

    if not pieces:
        raise ValueError("No test CpGs were found in the raw methylation matrix")
    coverage = pd.concat(pieces, ignore_index=True)
    if coverage["probeID"].duplicated().any():
        duplicates = coverage.loc[coverage["probeID"].duplicated(), "probeID"].head().tolist()
        raise ValueError(f"Raw matrix contains duplicate test probe IDs: {duplicates}")
    missing = sorted(wanted - set(coverage["probeID"]))
    if missing:
        raise ValueError(f"Missing {len(missing)} test CpGs from raw matrix; examples: {missing[:5]}")
    coverage = test_ids.to_frame().merge(coverage, on="probeID", how="left", validate="one_to_one")
    return coverage, len(normal_columns)


def finite_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["true_beta", "true_m", "binary_true", "pred_beta_rc_avg",
                "pred_m_rc_avg", "class_prob_rc_avg"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Prediction file is missing columns: {missing}")
    mask = np.isfinite(frame[required].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[mask].copy()


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    frame = finite_rows(frame)
    beta_error = frame["pred_beta_rc_avg"] - frame["true_beta"]
    m_error = frame["pred_m_rc_avg"] - frame["true_m"]
    labels = frame["binary_true"].astype(int)
    auc = np.nan
    if labels.nunique() == 2:
        auc = float(roc_auc_score(labels, frame["class_prob_rc_avg"]))
    return {
        "n_loci": int(len(frame)),
        "beta_mae": float(beta_error.abs().mean()),
        "beta_rmse": float(np.sqrt(np.mean(np.square(beta_error)))),
        "m_mae": float(m_error.abs().mean()),
        "m_rmse": float(np.sqrt(np.mean(np.square(m_error)))),
        "roc_auc": auc,
    }


def analyze_model(model: str, coverage: pd.DataFrame, args: argparse.Namespace):
    path = Path(args.prediction_template.format(model=model))
    pred = pd.read_csv(path)
    pred["probeID"] = pred["probeID"].astype(str)
    if pred["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probeID values in {path}")
    merged = pred.merge(coverage, on="probeID", how="inner", validate="one_to_one")
    if len(merged) != len(coverage):
        raise ValueError(
            f"{path}: merged {len(merged)} of {len(coverage)} held-out CpGs"
        )

    threshold_rows = []
    for label, minimum in THRESHOLDS:
        subset = merged.loc[merged["n_observed_normals"] >= minimum]
        row = {"model": model, "coverage_subset": label, "minimum_observed": minimum}
        row.update(metrics(subset))
        row["fraction_of_test"] = len(subset) / len(merged)
        threshold_rows.append(row)

    bins = pd.cut(
        merged["n_observed_normals"],
        bins=[0, 48, 77, 92, 96, 97],
        labels=["1-48", "49-77", "78-92", "93-96", "97"],
        include_lowest=True,
    )
    bin_rows = []
    for label in bins.cat.categories:
        subset = merged.loc[bins == label]
        if subset.empty:
            continue
        row = {"model": model, "coverage_bin": str(label)}
        row.update(metrics(subset))
        bin_rows.append(row)

    abs_beta_error = (merged["pred_beta_rc_avg"] - merged["true_beta"]).abs()
    rho, p_value = spearmanr(merged["n_observed_normals"], abs_beta_error)
    correlation = {
        "model": model,
        "n_loci": int(len(merged)),
        "spearman_coverage_vs_absolute_beta_error": float(rho),
        "spearman_p": float(p_value),
    }
    return merged, threshold_rows, bin_rows, correlation


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"all", "manifest"}:
        audit_manifest(args)
    if args.mode == "manifest":
        return

    coverage, n_normals = compute_coverage(args)
    coverage.to_csv(args.output_dir / "test_coverage_per_probe.csv", index=False)

    threshold_rows = []
    bin_rows = []
    correlations = []
    for model in MODELS:
        _, model_thresholds, model_bins, correlation = analyze_model(model, coverage, args)
        threshold_rows.extend(model_thresholds)
        bin_rows.extend(model_bins)
        correlations.append(correlation)

    thresholds = pd.DataFrame(threshold_rows)
    bins = pd.DataFrame(bin_rows)
    correlations_df = pd.DataFrame(correlations)
    thresholds.to_csv(args.output_dir / "coverage_threshold_metrics.csv", index=False)
    bins.to_csv(args.output_dir / "coverage_bin_metrics.csv", index=False)
    correlations_df.to_csv(args.output_dir / "coverage_error_correlations.csv", index=False)

    lines = [
        f"Normal columns: {n_normals}",
        f"Held-out CpGs: {len(coverage)}",
        "",
        "Threshold sensitivity:",
        thresholds.to_string(index=False),
        "",
        "Non-overlapping coverage bins:",
        bins.to_string(index=False),
        "",
        "Coverage versus absolute beta error:",
        correlations_df.to_string(index=False),
    ]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "coverage_sensitivity_summary.txt").write_text(summary)
    print(summary)
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
