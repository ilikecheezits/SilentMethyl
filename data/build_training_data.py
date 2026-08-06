#!/usr/bin/env python3
"""Build leakage-resistant training, validation, and test data for SilentMethyl.

Key changes from the original pipeline:
  * chromosome-held-out splits are assigned before any reverse-complement data are created;
  * forward and reverse-complement sequence representations are both emitted;
  * ordered central-base PhyloP features are swapped under reverse complement;
  * missing BigWig values are tracked explicitly and imputed from the training split only;
  * exact split, sample, input, and output provenance is written to JSON manifests.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR

WINDOW_SIZE = 5000
CENTER_C_INDEX = 2499
CENTER_G_INDEX = 2500
FASTA_SLICE = slice(2450, 2550)
VALID_CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])

# These sets are fixed before observing model performance. They place roughly
# chromosome-sized 10% partitions into validation and test data.
DEFAULT_VAL_CHROMS = ("chr10", "chr11")
DEFAULT_TEST_CHROMS = ("chr8", "chr9")

NORMAL_SAMPLE_RE = re.compile(
    r"^TCGA[-.][A-Z0-9]{2}[-.][A-Z0-9]{4}[-.]11[A-Z0-9](?:[-.]|$)",
    re.IGNORECASE,
)

RC_TABLE = str.maketrans(
    "ACGTRYMKBDHVNacgtrymkbdhvn",
    "TGCAYRKMVHDBNtgcayrkmvhdbn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-chroms", nargs="+", default=list(DEFAULT_VAL_CHROMS))
    parser.add_argument("--test-chroms", nargs="+", default=list(DEFAULT_TEST_CHROMS))
    parser.add_argument(
        "--hash-large-inputs",
        action="store_true",
        help="Also SHA-256 hash the large FASTA and methylation matrix.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, hash_file: bool = True) -> dict:
    record = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
    }
    if hash_file:
        record["sha256"] = sha256_file(path)
    return record


def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC_TABLE)[::-1]


def natural_chrom_rank(chrom: str) -> int:
    chrom = str(chrom)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    if chrom.isdigit():
        return int(chrom)
    return {"X": 23, "Y": 24}.get(chrom, 10_000)


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    out = df.copy()
    out["_chrom_rank"] = out["chr"].map(natural_chrom_rank)
    out["pos"] = pd.to_numeric(out["pos"], errors="raise").astype(int)
    out = out.sort_values(["_chrom_rank", "pos", "probeID"], kind="mergesort")
    return out.drop(columns="_chrom_rank").reset_index(drop=True)


def is_normal_sample_column(column: str) -> bool:
    return bool(NORMAL_SAMPLE_RE.search(str(column)))


def validate_split_chromosomes(val_chroms: Iterable[str], test_chroms: Iterable[str]) -> tuple[set[str], set[str], set[str]]:
    val_set = set(val_chroms)
    test_set = set(test_chroms)
    valid_set = set(VALID_CHROMS)

    unknown = (val_set | test_set) - valid_set
    if unknown:
        raise ValueError(f"Unknown chromosomes in split definition: {sorted(unknown)}")
    overlap = val_set & test_set
    if overlap:
        raise ValueError(f"Validation and test chromosome sets overlap: {sorted(overlap)}")

    train_set = valid_set - val_set - test_set
    if not train_set or not val_set or not test_set:
        raise ValueError("Train, validation, and test chromosome sets must all be nonempty.")
    return train_set, val_set, test_set


def assign_split(chrom: str, val_chroms: set[str], test_chroms: set[str]) -> str:
    if chrom in test_chroms:
        return "test"
    if chrom in val_chroms:
        return "val"
    return "train"


def open_bigwig_handles(paths: dict[str, Path]) -> dict[str, pyBigWig.pyBigWig]:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required BigWig files:\n" + "\n".join(missing))
    return {name: pyBigWig.open(str(path)) for name, path in paths.items()}


def get_bw_signal(bw_obj: pyBigWig.pyBigWig, chrom: str, start: int, end: int) -> float:
    """Return a finite mean signal or NaN. Missingness is never encoded as 0."""
    try:
        bw_chroms = bw_obj.chroms()
        query_chrom = chrom
        if query_chrom not in bw_chroms:
            alternate = chrom.replace("chr", "", 1) if chrom.startswith("chr") else f"chr{chrom}"
            if alternate not in bw_chroms:
                return np.nan
            query_chrom = alternate

        chrom_length = int(bw_chroms[query_chrom])
        bounded_start = max(0, int(start))
        bounded_end = min(chrom_length, int(end))
        if bounded_end <= bounded_start:
            return np.nan

        stat = bw_obj.stats(query_chrom, bounded_start, bounded_end, type="mean")
        if not stat or stat[0] is None:
            return np.nan
        value = float(stat[0])
        return value if math.isfinite(value) else np.nan
    except Exception:
        return np.nan


def write_fasta_file(df: pd.DataFrame, output_path: Path, sequence_column: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in df.itertuples(index=False):
            sequence = str(getattr(row, sequence_column))
            if len(sequence) != 100:
                raise ValueError(f"Unexpected 100-bp sequence length for {row.probeID}: {len(sequence)}")
            handle.write(f">{row.probeID}\n{sequence}\n")


def summarize_split(df: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}
    for split_name, split_df in df.groupby("Split", sort=False):
        summary[split_name] = {
            "rows": int(len(split_df)),
            "unique_probes": int(split_df["probeID"].nunique()),
            "chromosomes": sorted(split_df["chr"].unique().tolist(), key=natural_chrom_rank),
            "mean_beta": float(split_df["Median_Beta"].mean()),
            "positive_fraction": float(split_df["Binary_State_Target"].mean()),
        }
    return summary


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    datafiles_dir = data_dir / "datafiles"
    datafiles_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = data_dir / "hg38.fa"
    manifest_path = data_dir / "HM450.hg38.manifest.tsv.gz"
    meth_path = data_dir / "TCGA-BRCA.methylation450.tsv.gz"
    base_ref = data_dir / "reference"
    bw_paths = {
        "Ref_ATAC_Signal": base_ref / "ATAC_seq.bw",
        "Ref_H3K4me3_Signal": base_ref / "H3K4me3.bw",
        "Ref_H3K27ac_Signal": base_ref / "H3K27ac.bw",
        "Ref_H3K27me3_Signal": base_ref / "H3K27me3.bw",
        "Ref_H3K9me3_Signal": base_ref / "H3K9me3.bw",
        "Ref_H3K36me3_Signal": base_ref / "H3K36me3.bw",
        "Ref_H3K4me1_Signal": base_ref / "H3K4me1.bw",
        "Target_Base_PhyloP_100way_1": base_ref / "hg38.phyloP100way.bw",
        "Target_Base_PhyloP_100way_2": base_ref / "hg38.phyloP100way.bw",
    }

    required_paths = [fasta_path, manifest_path, meth_path, *bw_paths.values()]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing_paths))

    train_chroms, val_chroms, test_chroms = validate_split_chromosomes(args.val_chroms, args.test_chroms)

    print("==========================================")
    print("STEP 1: SEQUENCE AND TARGET CONSTRUCTION")
    print("==========================================")

    genome = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    df_manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        usecols=["probeID", "CpG_chrm", "CpG_beg"],
        dtype={"probeID": "string", "CpG_chrm": "string"},
    )
    df_manifest = df_manifest.rename(columns={"CpG_chrm": "chr", "CpG_beg": "pos"})
    df_manifest = df_manifest[df_manifest["chr"].isin(VALID_CHROMS)].dropna(subset=["probeID", "chr", "pos"])
    df_manifest["pos"] = pd.to_numeric(df_manifest["pos"], errors="raise").astype(int)
    df_manifest = df_manifest.sort_values(["chr", "pos", "probeID"], kind="mergesort")
    df_manifest = df_manifest.drop_duplicates(subset="probeID", keep="first").reset_index(drop=True)

    sequences: list[str | float] = []
    valid_rows: list[bool] = []
    off_center = 0
    out_of_bounds = 0

    for row in tqdm(df_manifest.itertuples(index=False), total=len(df_manifest), desc="Extracting 5-kb windows"):
        start = int(row.pos) - CENTER_C_INDEX
        end = start + WINDOW_SIZE
        if start < 0 or end > len(genome[row.chr]):
            sequences.append(np.nan)
            valid_rows.append(False)
            out_of_bounds += 1
            continue

        sequence = str(genome[row.chr][start:end]).upper()
        if len(sequence) != WINDOW_SIZE or sequence[CENTER_C_INDEX : CENTER_G_INDEX + 1] != "CG":
            sequences.append(np.nan)
            valid_rows.append(False)
            off_center += 1
            continue

        sequences.append(sequence)
        valid_rows.append(True)

    df_manifest["Healthy_5000bp_DNA"] = sequences
    df_seq = df_manifest.loc[valid_rows].copy().reset_index(drop=True)
    relevant_probes = set(df_seq["probeID"].astype(str))
    del df_manifest, sequences, valid_rows
    gc.collect()

    with gzip.open(meth_path, "rt") as handle:
        methylation_header = handle.readline().rstrip("\n").split("\t")

    probe_column = methylation_header[0]
    normal_columns = [column for column in methylation_header[1:] if is_normal_sample_column(column)]
    if not normal_columns:
        raise RuntimeError("No TCGA sample-type-11 columns were detected in the methylation matrix.")

    print(f"Selected {len(normal_columns)} solid-tissue-normal columns from {len(methylation_header) - 1} sample columns.")

    target_chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        meth_path,
        sep="\t",
        usecols=[probe_column, *normal_columns],
        chunksize=50_000,
        dtype={column: "float32" for column in normal_columns},
    )
    for chunk in tqdm(reader, desc="Calculating normal-tissue targets"):
        chunk = chunk.rename(columns={probe_column: "probeID"})
        chunk = chunk[chunk["probeID"].astype(str).isin(relevant_probes)].copy()
        if chunk.empty:
            continue
        chunk = chunk.loc[~chunk[normal_columns].isna().all(axis=1)].copy()
        if chunk.empty:
            continue

        values = chunk[normal_columns].to_numpy(dtype=np.float32)
        chunk["Median_Beta"] = np.nanmedian(values, axis=1)
        beta_clipped = np.clip(chunk["Median_Beta"].to_numpy(dtype=float), 0.0001, 0.9999)
        chunk["M_Value_Target"] = np.log2(beta_clipped / (1.0 - beta_clipped))
        chunk["Binary_State_Target"] = (chunk["Median_Beta"] > 0.5).astype(np.int8)
        target_chunks.append(chunk[["probeID", "Median_Beta", "M_Value_Target", "Binary_State_Target"]])
        del chunk, values

    if not target_chunks:
        raise RuntimeError("No methylation targets matched the sequence-valid HM450 probes.")

    target_map = pd.concat(target_chunks, ignore_index=True)
    if target_map["probeID"].duplicated().any():
        duplicate_count = int(target_map["probeID"].duplicated(keep=False).sum())
        raise RuntimeError(f"Methylation matrix contains duplicated probe rows ({duplicate_count} duplicated records).")

    df_master = df_seq.merge(target_map, on="probeID", how="inner", validate="one_to_one")
    df_master["Split"] = df_master["chr"].map(lambda chrom: assign_split(chrom, val_chroms, test_chroms))
    del df_seq, target_map, target_chunks
    gc.collect()

    print("==========================================")
    print("STEP 2: REFERENCE FEATURE EXTRACTION")
    print("==========================================")

    bw_handles = open_bigwig_handles(bw_paths)
    feature_values = {name: [] for name in bw_paths}

    try:
        for row in tqdm(df_master.itertuples(index=False), total=len(df_master), desc="Reading BigWigs"):
            chrom = str(row.chr)
            pos = int(row.pos)
            region_start = pos - 49
            region_end = pos + 51

            for feature_name, bw_obj in bw_handles.items():
                if feature_name == "Target_Base_PhyloP_100way_1":
                    value = get_bw_signal(bw_obj, chrom, pos, pos + 1)
                elif feature_name == "Target_Base_PhyloP_100way_2":
                    value = get_bw_signal(bw_obj, chrom, pos + 1, pos + 2)
                else:
                    value = get_bw_signal(bw_obj, chrom, region_start, region_end)
                feature_values[feature_name].append(value)
    finally:
        for handle in bw_handles.values():
            handle.close()

    for feature_name, values in feature_values.items():
        df_master[feature_name] = np.asarray(values, dtype=np.float32)
        df_master[f"{feature_name}_Missing"] = df_master[feature_name].isna().astype(np.int8)

    print("==========================================")
    print("STEP 3: TRAIN-ONLY IMPUTATION AND RC VIEWS")
    print("==========================================")

    train_mask = df_master["Split"].eq("train")
    imputation_values: dict[str, float] = {}
    phylo_features = ["Target_Base_PhyloP_100way_1", "Target_Base_PhyloP_100way_2"]
    pooled_phylo = pd.concat([df_master.loc[train_mask, feature] for feature in phylo_features], ignore_index=True)
    pooled_phylo_median = float(pooled_phylo.median(skipna=True))
    if not math.isfinite(pooled_phylo_median):
        pooled_phylo_median = 0.0

    for feature_name in bw_paths:
        if feature_name in phylo_features:
            median_value = pooled_phylo_median
        else:
            median_value = float(df_master.loc[train_mask, feature_name].median(skipna=True))
            if not math.isfinite(median_value):
                median_value = 0.0
        imputation_values[feature_name] = median_value
        df_master[feature_name] = df_master[feature_name].fillna(median_value).astype(np.float32)

    # Store compact 100-bp forward/RC model views. The full 5-kb RC window is
    # reconstructed on demand from Healthy_5000bp_DNA, avoiding a multi-gigabyte
    # duplicated CSV column. Splitting has already occurred at this point.
    df_master["Healthy_100bp_DNA"] = df_master["Healthy_5000bp_DNA"].str[FASTA_SLICE]
    df_master["Healthy_100bp_DNA_RC"] = df_master["Healthy_100bp_DNA"].map(reverse_complement)
    invalid_forward = df_master["Healthy_100bp_DNA"].str[49:51].ne("CG")
    invalid_rc = df_master["Healthy_100bp_DNA_RC"].str[49:51].ne("CG")
    if invalid_forward.any() or invalid_rc.any():
        raise RuntimeError(
            "Forward/RC 100-bp centering failed for "
            f"{int(invalid_forward.sum())}/{int(invalid_rc.sum())} probes."
        )

    df_master["Target_Base_PhyloP_100way_1_RC"] = df_master["Target_Base_PhyloP_100way_2"]
    df_master["Target_Base_PhyloP_100way_2_RC"] = df_master["Target_Base_PhyloP_100way_1"]
    df_master["Target_Base_PhyloP_100way_1_RC_Missing"] = df_master["Target_Base_PhyloP_100way_2_Missing"]
    df_master["Target_Base_PhyloP_100way_2_RC_Missing"] = df_master["Target_Base_PhyloP_100way_1_Missing"]

    split_frames = {
        split_name: sort_dataframe(df_master[df_master["Split"].eq(split_name)].copy())
        for split_name in ("train", "val", "test")
    }
    if any(frame.empty for frame in split_frames.values()):
        raise RuntimeError("At least one chromosome-held-out split is empty.")

    print("==========================================")
    print("STEP 4: WRITE DATA AND MANIFESTS")
    print("==========================================")

    output_paths: dict[str, Path] = {}
    for split_name, split_df in split_frames.items():
        csv_path = datafiles_dir / f"{split_name}.csv"
        split_df.to_csv(csv_path, index=False)
        output_paths[f"{split_name}_csv"] = csv_path

        forward_fasta = datafiles_dir / f"{split_name}_100bp.fasta"
        rc_fasta = datafiles_dir / f"{split_name}_100bp_rc.fasta"
        write_fasta_file(split_df, forward_fasta, "Healthy_100bp_DNA")
        write_fasta_file(split_df, rc_fasta, "Healthy_100bp_DNA_RC")
        output_paths[f"{split_name}_forward_fasta"] = forward_fasta
        output_paths[f"{split_name}_rc_fasta"] = rc_fasta

    normal_samples_path = datafiles_dir / "tcga_normal_sample_ids.json"
    normal_samples_path.write_text(json.dumps(normal_columns, indent=2) + "\n")
    output_paths["normal_sample_ids"] = normal_samples_path

    imputation_path = datafiles_dir / "feature_imputation.json"
    imputation_payload = {
        "schema_version": 1,
        "fit_split": "train",
        "strategy": "feature-wise median; pooled median shared by the two ordered central-base PhyloP features",
        "values": imputation_values,
        "missing_indicator_suffix": "_Missing",
    }
    imputation_path.write_text(json.dumps(imputation_payload, indent=2, sort_keys=True) + "\n")
    output_paths["feature_imputation"] = imputation_path

    split_manifest_path = datafiles_dir / "split_manifest.json"
    split_manifest = {
        "schema_version": 1,
        "strategy": "chromosome-held-out",
        "assigned_before_reverse_complement_augmentation": True,
        "train_chromosomes": sorted(train_chroms, key=natural_chrom_rank),
        "validation_chromosomes": sorted(val_chroms, key=natural_chrom_rank),
        "test_chromosomes": sorted(test_chroms, key=natural_chrom_rank),
        "window_size_bp": WINDOW_SIZE,
        "central_c_index_zero_based": CENTER_C_INDEX,
        "central_g_index_zero_based": CENTER_G_INDEX,
        "split_summary": summarize_split(df_master),
    }
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    output_paths["split_manifest"] = split_manifest_path

    input_records = {
        "manifest": file_record(manifest_path),
        "methylation_matrix": file_record(meth_path, hash_file=args.hash_large_inputs),
        "reference_fasta": file_record(fasta_path, hash_file=args.hash_large_inputs),
        "reference_fasta_index": file_record(Path(str(fasta_path) + ".fai")) if Path(str(fasta_path) + ".fai").exists() else None,
        "bigwigs": {name: file_record(path, hash_file=args.hash_large_inputs) for name, path in bw_paths.items()},
    }

    build_manifest_path = datafiles_dir / "training_data_manifest.json"
    build_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": file_record(Path(__file__).resolve()),
        "inputs": input_records,
        "normal_sample_selection": {
            "rule": NORMAL_SAMPLE_RE.pattern,
            "selected_count": len(normal_columns),
            "total_sample_columns": len(methylation_header) - 1,
            "sample_ids_file": normal_samples_path.name,
        },
        "sequence_filtering": {
            "valid_sequence_rows_before_target_join": int(len(relevant_probes)),
            "out_of_bounds_windows": out_of_bounds,
            "off_center_or_non_cg_windows": off_center,
            "rows_after_target_join": int(len(df_master)),
        },
        "split": split_manifest,
        "reverse_complement": {
            "forward_sequence_column": "Healthy_100bp_DNA",
            "reverse_complement_sequence_column": "Healthy_100bp_DNA_RC",
            "full_reverse_complement_rule": "reverse_complement(Healthy_5000bp_DNA)",
            "ordered_feature_transform": {
                "Target_Base_PhyloP_100way_1_RC": "Target_Base_PhyloP_100way_2",
                "Target_Base_PhyloP_100way_2_RC": "Target_Base_PhyloP_100way_1",
            },
            "note": "Epigenomic region-average features are unchanged under orientation reversal.",
        },
        "missingness_and_imputation": imputation_payload,
        "outputs": {name: file_record(path) for name, path in output_paths.items()},
    }
    build_manifest_path.write_text(json.dumps(build_manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(split_manifest["split_summary"], indent=2))
    print(f"Training data build complete: {datafiles_dir}")
    print(f"Manifest: {build_manifest_path}")


if __name__ == "__main__":
    main()
