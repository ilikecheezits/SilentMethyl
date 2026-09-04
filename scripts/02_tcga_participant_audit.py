#!/usr/bin/env python3
"""Audit TCGA-BRCA normal-sample selection and matrix provenance."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


NORMAL_SAMPLE_RE = re.compile(
    r"^TCGA[-.][A-Z0-9]{2}[-.][A-Z0-9]{4}[-.]11[A-Z0-9](?:[-.]|$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("data/TCGA-BRCA.methylation450.tsv.gz"),
    )
    parser.add_argument(
        "--selected-samples",
        type=Path,
        default=Path("data/datafiles/tcga_normal_sample_ids.json"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/datafiles/training_data_manifest.json"),
    )
    parser.add_argument(
        "--provenance-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing JSON provenance record. Repeat for multiple files. If omitted, "
            "the script checks the conventional root and reproducibility locations."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reproducibility"))
    parser.add_argument("--expected-selected-count", type=int, default=97)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument(
        "--skip-matrix-sha256",
        action="store_true",
        help="Skip hashing the archived matrix if its digest is already recorded elsewhere.",
    )
    parser.add_argument(
        "--skip-duplicate-sensitivity",
        action="store_true",
        help="Do not stream the matrix when duplicate participants are found.",
    )
    parser.add_argument("--matrix-source", default=None)
    parser.add_argument("--matrix-source-id", default=None)
    parser.add_argument("--matrix-source-url", default=None)
    parser.add_argument("--matrix-data-type", default=None)
    parser.add_argument("--matrix-processing", default=None)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def load_selected_samples(path: Path) -> list[str]:
    payload = load_json(path)
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = None
        for key in ("sample_ids", "normal_sample_ids", "samples", "columns"):
            if isinstance(payload.get(key), list):
                values = payload[key]
                break
        if values is None:
            raise ValueError(f"Could not locate a sample list in {path}")
    else:
        raise TypeError(f"Expected a JSON list or object in {path}")
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"Sample list in {path} contains a non-string or empty value")
    return list(values)


def read_matrix_header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        line = handle.readline()
    if not line:
        raise ValueError(f"Matrix is empty: {path}")
    return line.rstrip("\r\n").split("\t")


def normalized_tcga_barcode(value: str) -> str:
    return "-".join(part for part in re.split(r"[-.]", value.strip()) if part)


def parse_tcga_column(column: str) -> dict[str, Any]:
    if not NORMAL_SAMPLE_RE.search(column):
        raise ValueError(f"Column does not satisfy the training sample-type-11 rule: {column}")
    tokens = normalized_tcga_barcode(column).split("-")
    if len(tokens) < 4 or tokens[0].upper() != "TCGA":
        raise ValueError(f"Could not parse TCGA barcode: {column}")
    sample_token = tokens[3].upper()
    if len(sample_token) < 3 or not sample_token[:2].isdigit():
        raise ValueError(f"Could not parse TCGA sample component: {column}")
    portion_analyte = tokens[4].upper() if len(tokens) >= 5 else ""
    return {
        "matrix_column": column,
        "normalized_barcode": "-".join(token.upper() for token in tokens),
        "participant_id": "-".join(token.upper() for token in tokens[:3]),
        "sample_id": "-".join(token.upper() for token in tokens[:4]),
        "sample_type_code": sample_token[:2],
        "vial": sample_token[2:],
        "portion": portion_analyte[:2] if len(portion_analyte) >= 2 else "",
        "analyte": portion_analyte[2:] if len(portion_analyte) >= 3 else "",
        "plate": tokens[5].upper() if len(tokens) >= 6 else "",
        "center": tokens[6].upper() if len(tokens) >= 7 else "",
        "barcode_component_count": len(tokens),
    }


def file_record(path: Path, *, include_hash: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def discover_provenance_files(explicit: Iterable[Path]) -> list[Path]:
    candidates = list(explicit)
    if not candidates:
        candidates = [
            Path("reproducibility/tcga_methylation_input_audit.json"),
            Path("tcga_methylation_input_audit.json"),
        ]
    seen: set[Path] = set()
    output: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.exists():
            seen.add(resolved)
            output.append(candidate)
    return output


def count_summary(counter: Counter[str]) -> dict[str, Any]:
    values = np.asarray(list(counter.values()), dtype=int)
    if not len(values):
        return {}
    return {
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "entities_with_multiple_columns": int((values > 1).sum()),
    }


def safe_nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def duplicate_participant_sensitivity(
    matrix: Path,
    probe_column: str,
    selected: list[str],
    participant_by_column: dict[str, str],
    chunk_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for column in selected:
        groups[participant_by_column[column]].append(column)

    differences: list[np.ndarray] = []
    top_frames: list[pd.DataFrame] = []
    valid_rows = 0
    all_nan_rows = 0
    reader = pd.read_csv(
        matrix,
        sep="\t",
        usecols=[probe_column, *selected],
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        if chunk_number == 1 or chunk_number % 10 == 0:
            print(
                f"Participant-aggregation sensitivity: processing matrix chunk {chunk_number}",
                flush=True,
            )
        numeric = chunk[selected].apply(pd.to_numeric, errors="coerce")
        original = safe_nanmedian(numeric.to_numpy(dtype=float), axis=1)
        participant_values = np.column_stack(
            [
                safe_nanmedian(numeric[columns].to_numpy(dtype=float), axis=1)
                for columns in groups.values()
            ]
        )
        participant_equal = safe_nanmedian(participant_values, axis=1)
        valid = np.isfinite(original) & np.isfinite(participant_equal)
        all_nan_rows += int((~valid).sum())
        if not valid.any():
            continue
        absolute_difference = np.abs(participant_equal[valid] - original[valid])
        differences.append(absolute_difference)
        valid_rows += int(valid.sum())
        frame = pd.DataFrame(
            {
                "probeID": chunk.loc[valid, probe_column].astype(str).to_numpy(),
                "column_weighted_median_beta": original[valid],
                "participant_equal_median_beta": participant_equal[valid],
                "absolute_difference": absolute_difference,
            }
        )
        top_frames.append(frame.nlargest(min(100, len(frame)), "absolute_difference"))

    if not differences:
        return {
            "performed": True,
            "valid_probe_rows": 0,
            "all_nan_probe_rows": all_nan_rows,
            "error": "No probe had finite values under both aggregation strategies.",
        }

    diff = np.concatenate(differences)
    top = (
        pd.concat(top_frames, ignore_index=True)
        .nlargest(min(100, valid_rows), "absolute_difference")
        .reset_index(drop=True)
    )
    top_path = output_dir / "tcga_participant_aggregation_sensitivity_top100.csv"
    top.to_csv(top_path, index=False)
    quantiles = np.quantile(diff, [0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "performed": True,
        "definition": (
            "absolute difference between the original median across selected columns and "
            "the median across within-participant medians"
        ),
        "valid_probe_rows": int(valid_rows),
        "all_nan_probe_rows": int(all_nan_rows),
        "mean_absolute_difference": float(diff.mean()),
        "median_absolute_difference": float(quantiles[0]),
        "p90_absolute_difference": float(quantiles[1]),
        "p95_absolute_difference": float(quantiles[2]),
        "p99_absolute_difference": float(quantiles[3]),
        "maximum_absolute_difference": float(quantiles[4]),
        "fractions_exceeding": {
            "0.001": float(np.mean(diff > 0.001)),
            "0.005": float(np.mean(diff > 0.005)),
            "0.010": float(np.mean(diff > 0.010)),
            "0.020": float(np.mean(diff > 0.020)),
            "0.050": float(np.mean(diff > 0.050)),
        },
        "top_changed_rows": str(top_path.resolve()),
    }


def main() -> int:
    args = parse_args()
    required = [args.matrix, args.selected_samples]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header = read_matrix_header(args.matrix)
    if len(header) < 2:
        raise ValueError("Methylation matrix header does not contain sample columns")
    probe_column = header[0]
    matrix_columns = header[1:]
    selected = load_selected_samples(args.selected_samples)
    recomputed = [column for column in matrix_columns if NORMAL_SAMPLE_RE.search(column)]

    hard_errors: list[str] = []
    audit_warnings: list[str] = []
    if len(selected) != args.expected_selected_count:
        hard_errors.append(
            f"Selected sample count is {len(selected)}, expected {args.expected_selected_count}."
        )
    if len(selected) != len(set(selected)):
        hard_errors.append("Saved selected-sample list contains duplicate column names.")
    missing_from_matrix = sorted(set(selected) - set(matrix_columns))
    unexpected_selected = sorted(set(selected) - set(recomputed))
    omitted_normal_columns = sorted(set(recomputed) - set(selected))
    if missing_from_matrix:
        hard_errors.append(f"{len(missing_from_matrix)} selected columns are absent from the matrix.")
    if unexpected_selected:
        hard_errors.append(
            f"{len(unexpected_selected)} selected columns do not satisfy the training type-11 rule."
        )
    if omitted_normal_columns:
        hard_errors.append(
            f"{len(omitted_normal_columns)} matrix columns satisfy the type-11 rule but are omitted."
        )
    if selected != recomputed and set(selected) == set(recomputed):
        audit_warnings.append("Saved selection has the same columns as the matrix rule but a different order.")

    parsed_rows: list[dict[str, Any]] = []
    for column in selected:
        try:
            parsed_rows.append(parse_tcga_column(column))
        except ValueError as exc:
            hard_errors.append(str(exc))
    sample_table = pd.DataFrame(parsed_rows)
    if not sample_table.empty and not sample_table["sample_type_code"].eq("11").all():
        hard_errors.append("At least one parsed selected column is not sample type 11.")

    participant_counts = Counter(sample_table.get("participant_id", pd.Series(dtype=str)))
    sample_counts = Counter(sample_table.get("sample_id", pd.Series(dtype=str)))
    duplicated_participants = {
        key: count for key, count in sorted(participant_counts.items()) if count > 1
    }
    duplicated_samples = {key: count for key, count in sorted(sample_counts.items()) if count > 1}
    if duplicated_participants:
        audit_warnings.append(
            f"{len(duplicated_participants)} participants contribute multiple selected columns."
        )

    sample_csv = args.output_dir / "tcga_selected_normal_samples.csv"
    sample_table.to_csv(sample_csv, index=False)

    provenance_files = discover_provenance_files(args.provenance_file)
    provenance_records: list[dict[str, Any]] = []
    for path in provenance_files:
        try:
            provenance_records.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "content": load_json(path),
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            audit_warnings.append(f"Could not read provenance file {path}: {exc}")

    explicit_provenance = {
        "source": args.matrix_source,
        "source_id_or_uuid": args.matrix_source_id,
        "source_url": args.matrix_source_url,
        "data_type": args.matrix_data_type,
        "processing_and_normalization": args.matrix_processing,
    }
    if not provenance_records and not any(explicit_provenance.values()):
        audit_warnings.append(
            "Exact acquisition/processing provenance was not supplied and cannot be inferred "
            "from the matrix filename or contents alone."
        )

    training_manifest_record: dict[str, Any] | None = None
    if args.training_manifest.exists():
        training_manifest_record = {
            "path": str(args.training_manifest.resolve()),
            "sha256": sha256_file(args.training_manifest),
            "content": load_json(args.training_manifest),
        }
    else:
        audit_warnings.append(f"Training manifest not found: {args.training_manifest}")

    sensitivity: dict[str, Any] = {
        "performed": False,
        "reason": "No participant contributes multiple columns.",
    }
    if duplicated_participants:
        if args.skip_duplicate_sensitivity:
            sensitivity = {
                "performed": False,
                "reason": "Skipped by --skip-duplicate-sensitivity.",
            }
        elif not hard_errors:
            participant_by_column = dict(
                zip(sample_table["matrix_column"], sample_table["participant_id"])
            )
            sensitivity = duplicate_participant_sensitivity(
                args.matrix,
                probe_column,
                selected,
                participant_by_column,
                args.chunk_size,
                args.output_dir,
            )
        else:
            sensitivity = {
                "performed": False,
                "reason": "Not run because sample-selection hard errors were detected.",
            }

    if not args.skip_matrix_sha256:
        print(f"Hashing archived matrix: {args.matrix}", flush=True)
    matrix_record = file_record(
        args.matrix,
        include_hash=not args.skip_matrix_sha256,
    )
    selected_record = file_record(args.selected_samples, include_hash=True)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if hard_errors else ("PASS_WITH_WARNINGS" if audit_warnings else "PASS"),
        "hard_error_count": len(hard_errors),
        "warning_count": len(audit_warnings),
        "hard_errors": hard_errors,
        "warnings": audit_warnings,
        "matrix": {
            **matrix_record,
            "probe_identifier_column": probe_column,
            "total_sample_columns": len(matrix_columns),
            "type11_columns_recomputed": len(recomputed),
            "explicit_provenance": explicit_provenance,
            "existing_provenance_records": provenance_records,
        },
        "selection": {
            "rule": NORMAL_SAMPLE_RE.pattern,
            "selected_samples_file": selected_record,
            "selected_column_count": len(selected),
            "expected_selected_column_count": args.expected_selected_count,
            "selection_exactly_matches_matrix_rule_and_order": selected == recomputed,
            "missing_from_matrix": missing_from_matrix,
            "unexpected_selected_columns": unexpected_selected,
            "omitted_type11_columns": omitted_normal_columns,
        },
        "participants": {
            "unique_participant_count": len(participant_counts),
            "unique_sample_count": len(sample_counts),
            "selected_column_count": len(sample_table),
            "columns_per_participant": count_summary(participant_counts),
            "columns_per_sample": count_summary(sample_counts),
            "participants_with_multiple_columns": duplicated_participants,
            "samples_with_multiple_columns": duplicated_samples,
            "participant_equal_aggregation_sensitivity": sensitivity,
        },
        "training_manifest": training_manifest_record,
        "outputs": {
            "selected_sample_table": str(sample_csv.resolve()),
        },
    }
    report_path = args.output_dir / "tcga_participant_matrix_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("TCGA participant/matrix audit")
    print(f"  status: {report['status']}")
    print(f"  matrix sample columns: {len(matrix_columns)}")
    print(f"  selected type-11 columns: {len(selected)}")
    print(f"  unique participants: {len(participant_counts)}")
    print(f"  participants with multiple columns: {len(duplicated_participants)}")
    print(f"  unique sample barcodes: {len(sample_counts)}")
    print(f"  samples with multiple columns: {len(duplicated_samples)}")
    if sensitivity.get("performed"):
        print(
            "  participant aggregation sensitivity: "
            f"median={sensitivity['median_absolute_difference']:.6g}, "
            f"p99={sensitivity['p99_absolute_difference']:.6g}, "
            f"max={sensitivity['maximum_absolute_difference']:.6g}"
        )
    print(f"  report: {report_path}")
    print(f"  selected-sample table: {sample_csv}")
    if hard_errors:
        print("\nHard errors:", file=sys.stderr)
        for error in hard_errors:
            print(f"  - {error}", file=sys.stderr)
    if audit_warnings:
        print("\nWarnings:")
        for message in audit_warnings:
            print(f"  - {message}")
    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
