#!/usr/bin/env python3
"""External eGTEx breast mQTL positive-control validation for SilentMethyl.

This script evaluates whether frozen SilentMethyl journal checkpoints recover
the direction and relative magnitude of experimentally observed local breast
mQTL effects.  The strict primary benchmark is the 81 q<0.05 lead mQTLs whose
CpGs and sequence perturbations fall in the chromosome-8/9 test split and the
trained 1,000-bp model window.

For every CpG/variant pair the script:

1. reconstructs the REF (WT) and ALT sequence with coordinate/allele audits;
2. keeps the reference epigenomic context fixed between REF and ALT;
3. evaluates forward and reverse-complement orientations in FP32 by default;
4. computes ALT-minus-REF changes in predicted M-value and beta; and
5. compares those changes with the eGTEx QTLtools slope; and
6. repeats the statistics after conservative HM450 probe-footprint exclusions.

QTLtools regresses the phenotype on VCF genotype dosage.  Under the standard
VCF definition, dosage counts the ALT allele; therefore the default alignment
is positive slope == higher methylation with more ALT alleles.  The alignment
is explicit and recorded in every output.  Use ``--slope-effect-allele ref``
only if independent provenance shows that a transformed input reversed it.

These eGTEx variants are inherited/germline mQTL alleles, whereas the downstream
TCGA application scores somatic synonymous variants.  The benchmark therefore
validates that the learned local sequence response recovers an external,
tissue-matched allelic methylation signal; it does not prove identical effect
distributions or mechanisms in the somatic candidate domain.

The script never loads the train/validation mQTL rows and contains no effect
threshold selection.  Thus the held-out benchmark cannot be used silently for
model selection. Probe-footprint exclusions are labelled post hoc sensitivity
analyses and do not redefine the frozen primary cohort.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# This file lives in SilentMethyl/scripts/experiments. Resolve all project paths
# from project markers rather than the launch directory.
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

PROJECT_ROOT = find_project_root(SCRIPT_DIR)
PROJECT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if PROJECT_SCRIPTS_DIR.is_dir() and str(PROJECT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPTS_DIR))

from training_common import (
    FusionModel,
    MISSING_FEATURES,
    SequenceOnlyModel,
    TABULAR_FEATURES,
    autocast_context,
    centered_crop,
    get_tokenizer,
    load_model_state,
    m_to_beta_tensor,
    reverse_complement,
    set_seed,
    validate_split_dataframe,
)


LOGGER = logging.getLogger("silentmethyl.egtex_mqtl")

FULL_SEQUENCE_LENGTH = 5000
FULL_TARGET_C_INDEX = 2499
MODEL_WINDOW_SIZE = 1000
MODEL_TARGET_C_INDEX = 499
MODEL_TARGET_G_INDEX = 500
PROTECTED_OFFSETS = frozenset({0, 1})
DNA_BASES = frozenset("ACGT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the held-out eGTEx breast mQTL positive-control benchmark"
    )
    parser.add_argument(
        "--mqtl-csv",
        default=str(PROJECT_ROOT / "data" / "egtex_breast_mqtl_heldout_qc.csv"),
        help="QC-checked 81-row chr8/9 benchmark produced from the eGTEx lead-mQTL file.",
    )
    parser.add_argument(
        "--test-csv",
        default=str(PROJECT_ROOT / "data" / "datafiles" / "test.csv"),
    )
    parser.add_argument(
        "--hm450-manifest",
        default=str(PROJECT_ROOT / "data" / "HM450.hg38.manifest.tsv.gz"),
        help=(
            "Zhou-lab hg38 HM450 annotation containing probeBeg/probeEnd. "
            "Used only for post hoc probe-footprint sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("fusion", "sequence"),
        default=("fusion", "sequence"),
        help="Frozen journal models to evaluate. The fusion model is the paper's primary model.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--fusion-weights-template",
        default=str(
            PROJECT_ROOT
            / "checkpoints_journal"
            / "seed{seed}"
            / "fusion"
            / "best_weights.pth"
        ),
    )
    parser.add_argument(
        "--sequence-weights-template",
        default=str(
            PROJECT_ROOT
            / "checkpoints_journal"
            / "seed{seed}"
            / "sequence"
            / "best_weights.pth"
        ),
    )
    parser.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local-model-dir", default=str(PROJECT_ROOT / "dnabert2_local"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "journal" / "egtex_mqtl_positive_control"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=MODEL_WINDOW_SIZE)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Enable CUDA mixed precision. Disabled by default because the benchmark subtracts "
            "nearly identical REF/ALT predictions and primary scoring should remain FP32."
        ),
    )
    parser.add_argument(
        "--slope-effect-allele",
        choices=("alt", "ref"),
        default="alt",
        help="Allele whose increasing dosage corresponds to the reported eGTEx slope.",
    )
    parser.add_argument("--expected-heldout-rows", type=int, default=81)
    parser.add_argument(
        "--heldout-chromosomes",
        nargs="+",
        default=("chr8", "chr9"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    parser.add_argument("--statistics-seed", type=int, default=20260810)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Positive values enable an explicitly labelled smoke test; 0 runs the locked benchmark.",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def boolean_series(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        examples = values[parsed.isna()].head(5).tolist()
        raise ValueError(f"{name} contains non-boolean values, examples={examples}")
    return parsed.astype(bool)


def unique_in_order(values: Iterable[int | str]) -> list:
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def validate_cli(args: argparse.Namespace) -> None:
    args.seeds = [int(seed) for seed in unique_in_order(args.seeds)]
    args.models = [str(model) for model in unique_in_order(args.models)]
    args.heldout_chromosomes = [normalize_chromosome(chrom) for chrom in args.heldout_chromosomes]
    if not args.seeds:
        raise ValueError("At least one trained seed is required")
    if args.window_size != MODEL_WINDOW_SIZE:
        raise ValueError("The journal positive control is locked to the trained 1000-bp window")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.bootstrap_replicates < 0 or args.permutation_replicates < 0:
        raise ValueError("Bootstrap/permutation replicate counts cannot be negative")
    if args.max_rows < 0:
        raise ValueError("--max-rows cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def annotate_hm450_probe_geometry(
    cohort: pd.DataFrame,
    manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Attach probe coordinates and conservative hybridization-artifact flags.

    ``probeBeg``/``probeEnd`` come from the provider's mapped hg38 probe
    annotation.  We record both the usual half-open overlap and a conservative
    closed interval expanded by one nucleotide on each side.  The latter
    protects the sensitivity analysis against endpoint convention and
    strand-dependent extension-boundary ambiguity; it is deliberately not used
    to redefine the frozen 81-row primary benchmark.
    """
    manifest_path = Path(manifest_path)
    columns = [
        "CpG_chrm",
        "CpG_end",
        "probe_strand",
        "probeID",
        "nextBase",
        "nextBaseRef",
        "probeType",
        "probeBeg",
        "probeEnd",
        "MASK_snp5_common",
        "MASK_snp5_GMAF1p",
        "MASK_extBase",
        "MASK_general",
    ]
    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        compression="infer",
        usecols=columns,
        low_memory=False,
    )
    require_columns(manifest, columns, str(manifest_path))
    if manifest["probeID"].duplicated().any():
        examples = manifest.loc[
            manifest["probeID"].duplicated(False), "probeID"
        ].head(10).tolist()
        raise ValueError(f"HM450 manifest has duplicate probeID rows: {examples}")

    for column in ("CpG_end", "probeBeg", "probeEnd"):
        manifest[column] = pd.to_numeric(manifest[column], errors="coerce")
    selected = manifest[manifest["probeID"].isin(cohort["probeID"])].copy()
    if selected[["CpG_end", "probeBeg", "probeEnd"]].isna().any().any():
        bad = selected.loc[
            selected[["CpG_end", "probeBeg", "probeEnd"]].isna().any(axis=1),
            "probeID",
        ].head(10).tolist()
        raise ValueError(f"Selected HM450 probes have missing coordinates: {bad}")
    for column in ("CpG_end", "probeBeg", "probeEnd"):
        selected[column] = selected[column].astype(np.int64)

    renamed = selected.rename(
        columns={
            "CpG_chrm": "hm450_cpg_chr",
            "CpG_end": "hm450_cpg_end",
            "probe_strand": "hm450_probe_strand",
            "nextBase": "hm450_next_base",
            "nextBaseRef": "hm450_next_base_ref",
            "probeType": "hm450_probe_type",
            "probeBeg": "hm450_probe_beg",
            "probeEnd": "hm450_probe_end",
            "MASK_snp5_common": "hm450_mask_snp5_common",
            "MASK_snp5_GMAF1p": "hm450_mask_snp5_gmaf1p",
            "MASK_extBase": "hm450_mask_extbase",
            "MASK_general": "hm450_mask_general",
        }
    )
    annotated = cohort.merge(
        renamed,
        on="probeID",
        how="left",
        validate="one_to_one",
        indicator="hm450_probe_merge",
    )
    if not annotated["hm450_probe_merge"].eq("both").all():
        missing = annotated.loc[
            annotated["hm450_probe_merge"] != "both", "probeID"
        ].head(10).tolist()
        raise ValueError(f"Benchmark probes missing from HM450 manifest: {missing}")
    annotated = annotated.drop(columns="hm450_probe_merge")

    manifest_chr = annotated["hm450_cpg_chr"].map(normalize_chromosome)
    if not manifest_chr.eq(annotated["cpg_chr"]).all():
        raise ValueError("HM450 manifest chromosome disagrees with benchmark CpG chromosome")
    annotated["hm450_cpg_chr"] = manifest_chr

    coordinate_offsets = (
        annotated["hm450_cpg_end"].to_numpy(dtype=np.int64)
        - annotated["cpg_pos0"].to_numpy(dtype=np.int64)
    )
    unique_offsets = sorted(np.unique(coordinate_offsets).tolist())
    # The annotation represents a CpG as a two-base, zero-based half-open
    # interval: CpG_end is therefore target-C position + 2.  Requiring this
    # relation also verifies that benchmark and manifest coordinates share the
    # same hg38 convention before probe intervals are used.
    if unique_offsets != [2]:
        raise ValueError(
            "Unexpected HM450 CpG coordinate relation: "
            f"expected [2], observed unique(CpG_end - cpg_pos0)={unique_offsets}"
        )

    probe_lo = np.minimum(
        annotated["hm450_probe_beg"].to_numpy(dtype=np.int64),
        annotated["hm450_probe_end"].to_numpy(dtype=np.int64),
    )
    probe_hi = np.maximum(
        annotated["hm450_probe_beg"].to_numpy(dtype=np.int64),
        annotated["hm450_probe_end"].to_numpy(dtype=np.int64),
    )
    variant_pos0 = annotated["var_pos0"].to_numpy(dtype=np.int64)
    annotated["hm450_probe_span_lo0"] = probe_lo
    annotated["hm450_probe_span_hi0"] = probe_hi
    annotated["variant_overlaps_probe_span_half_open"] = (
        (variant_pos0 >= probe_lo) & (variant_pos0 < probe_hi)
    )
    annotated["variant_overlaps_probe_span_closed"] = (
        (variant_pos0 >= probe_lo) & (variant_pos0 <= probe_hi)
    )
    annotated["variant_overlaps_probe_or_extension_conservative"] = (
        (variant_pos0 >= probe_lo - 1) & (variant_pos0 <= probe_hi + 1)
    )

    for column in (
        "hm450_mask_snp5_common",
        "hm450_mask_snp5_gmaf1p",
        "hm450_mask_extbase",
        "hm450_mask_general",
    ):
        annotated[column] = boolean_series(annotated[column], column)

    audit = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "coordinate_check_unique_CpG_end_minus_cpg_pos0": unique_offsets,
        "probe_span_definition": "min(probeBeg,probeEnd) through max(probeBeg,probeEnd)",
        "primary_sensitivity_exclusion": (
            "variant position within the closed annotated probe span expanded by one bp "
            "on both sides"
        ),
        "n_loci": int(len(annotated)),
        "n_probe_span_half_open_overlap": int(
            annotated["variant_overlaps_probe_span_half_open"].sum()
        ),
        "n_probe_span_closed_overlap": int(
            annotated["variant_overlaps_probe_span_closed"].sum()
        ),
        "n_probe_or_extension_conservative_overlap": int(
            annotated["variant_overlaps_probe_or_extension_conservative"].sum()
        ),
        "n_manifest_mask_general_true": int(annotated["hm450_mask_general"].sum()),
        "n_manifest_mask_extbase_true": int(annotated["hm450_mask_extbase"].sum()),
    }
    return annotated, audit


def validate_mqtl_table(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = {
        "cpg_id",
        "variant_id",
        "qval",
        "slope",
        "var_chr",
        "var_pos1",
        "ref",
        "alt",
        "cpg_chr",
        "cpg_pos0",
        "split",
        "var_pos0",
        "offset_from_cpg_C",
        "model_visible",
        "target_cpg_variant",
        "hg38_ref_match",
    }
    require_columns(frame, required, args.mqtl_csv)
    result = frame.copy()

    if args.max_rows > 0:
        result = result.head(args.max_rows).copy()
        LOGGER.warning(
            "SMOKE TEST ONLY: using %d/%d benchmark rows; these statistics are not reportable",
            len(result),
            len(frame),
        )
    elif len(result) != args.expected_heldout_rows:
        raise ValueError(
            f"Locked benchmark expected {args.expected_heldout_rows} rows, found {len(result)}"
        )

    if result.empty:
        raise ValueError("mQTL benchmark is empty")
    if result["cpg_id"].duplicated().any():
        duplicated = result.loc[result["cpg_id"].duplicated(False), "cpg_id"].tolist()
        raise ValueError(f"Lead-mQTL benchmark contains duplicate CpG IDs: {duplicated[:10]}")
    if result[["cpg_id", "variant_id"]].duplicated().any():
        raise ValueError("Benchmark contains duplicate CpG/variant pairs")

    for column in ("qval", "slope", "var_pos1", "var_pos0", "cpg_pos0", "offset_from_cpg_C"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains missing or non-finite values")

    if not (result["qval"] < 0.05).all():
        raise ValueError("All primary benchmark rows must have eGTEx qval < 0.05")
    if (result["slope"] == 0).any():
        raise ValueError("A benchmark slope is exactly zero, so its direction is undefined")

    for column in ("var_pos1", "var_pos0", "cpg_pos0", "offset_from_cpg_C"):
        if not np.equal(result[column], np.floor(result[column])).all():
            raise ValueError(f"{column} must contain integer coordinates")
        result[column] = result[column].astype(np.int64)

    if not np.array_equal(result["var_pos0"].to_numpy(), result["var_pos1"].to_numpy() - 1):
        raise ValueError("eGTEx 1-based to 0-based variant conversion is inconsistent")
    computed_offset = result["var_pos0"] - result["cpg_pos0"]
    if not np.array_equal(computed_offset.to_numpy(), result["offset_from_cpg_C"].to_numpy()):
        raise ValueError("offset_from_cpg_C is inconsistent with the genomic coordinates")

    result["var_chr"] = result["var_chr"].map(normalize_chromosome)
    result["cpg_chr"] = result["cpg_chr"].map(normalize_chromosome)
    if not result["var_chr"].eq(result["cpg_chr"]).all():
        raise ValueError("At least one mQTL variant and CpG are on different chromosomes")
    if not set(result["cpg_chr"]).issubset(set(args.heldout_chromosomes)):
        counts = result["cpg_chr"].value_counts().to_dict()
        raise ValueError(f"Non-held-out chromosome found in strict benchmark: {counts}")
    if not result["split"].astype(str).str.lower().eq("test").all():
        raise ValueError(f"mQTL split column is not test-only: {result['split'].value_counts().to_dict()}")

    model_visible = boolean_series(result["model_visible"], "model_visible")
    target_variant = boolean_series(result["target_cpg_variant"], "target_cpg_variant")
    ref_match = boolean_series(result["hg38_ref_match"], "hg38_ref_match")
    if not model_visible.all():
        raise ValueError("Benchmark contains an mQTL outside the 1000-bp model window")
    if target_variant.any():
        raise ValueError("Benchmark contains a variant that alters the target CpG")
    if not ref_match.all():
        raise ValueError("Benchmark contains an hg38 REF mismatch")

    offsets = result["offset_from_cpg_C"]
    if not offsets.between(-MODEL_TARGET_C_INDEX, MODEL_WINDOW_SIZE - MODEL_TARGET_C_INDEX - 1).all():
        raise ValueError("An mQTL offset is outside the trained 1000-bp crop")
    if offsets.isin(PROTECTED_OFFSETS).any():
        raise ValueError("An mQTL alters the protected target CpG C or G")

    result["ref"] = result["ref"].astype(str).str.upper()
    result["alt"] = result["alt"].astype(str).str.upper()
    valid_alleles = result["ref"].isin(DNA_BASES) & result["alt"].isin(DNA_BASES)
    if not valid_alleles.all() or result["ref"].eq(result["alt"]).any():
        raise ValueError("Benchmark contains an invalid or non-SNV REF/ALT allele")
    return result.reset_index(drop=True)


def load_and_build_cohort(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str], list[str], torch.Tensor, torch.Tensor]:
    mqtl_raw = pd.read_csv(args.mqtl_csv)
    mqtl = validate_mqtl_table(mqtl_raw, args)

    test = pd.read_csv(args.test_csv)
    validate_split_dataframe(test, "test", args.test_csv)
    merged = mqtl.merge(
        test,
        left_on="cpg_id",
        right_on="probeID",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_silentmethyl"),
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"] != "both", "cpg_id"].tolist()
        raise ValueError(f"Held-out mQTL CpGs missing from SilentMethyl test.csv: {missing[:10]}")

    test_chr = merged["chr"].map(normalize_chromosome)
    if not test_chr.eq(merged["cpg_chr"]).all():
        raise ValueError("CpG chromosome disagrees between the mQTL file and test.csv")
    test_pos = pd.to_numeric(merged["pos"], errors="coerce")
    if test_pos.isna().any() or not np.array_equal(
        test_pos.astype(np.int64).to_numpy(), merged["cpg_pos0"].to_numpy()
    ):
        raise ValueError("CpG zero-based position disagrees between the mQTL file and test.csv")

    wt_sequences: list[str] = []
    alt_sequences: list[str] = []
    mutation_indices_5kb: list[int] = []
    mutation_indices_1kb: list[int] = []
    wt_hashes: list[str] = []
    alt_hashes: list[str] = []

    for row in merged.itertuples(index=False):
        full_wt = str(row.Healthy_5000bp_DNA).upper()
        if len(full_wt) != FULL_SEQUENCE_LENGTH:
            raise ValueError(
                f"{row.cpg_id}: expected a 5000-bp WT sequence, found {len(full_wt)}"
            )
        if set(full_wt) - set("ACGTN"):
            raise ValueError(f"{row.cpg_id}: WT sequence contains an invalid DNA character")
        if full_wt[FULL_TARGET_C_INDEX : FULL_TARGET_C_INDEX + 2] != "CG":
            raise ValueError(f"{row.cpg_id}: stored 5-kb sequence is not centered on a CpG")

        offset = int(row.offset_from_cpg_C)
        full_index = FULL_TARGET_C_INDEX + offset
        crop_index = MODEL_TARGET_C_INDEX + offset
        if full_wt[full_index] != str(row.ref):
            raise ValueError(
                f"{row.cpg_id}/{row.variant_id}: stored sequence has {full_wt[full_index]} "
                f"at mutation index, expected REF={row.ref}"
            )
        full_alt = full_wt[:full_index] + str(row.alt) + full_wt[full_index + 1 :]
        wt = centered_crop(full_wt, MODEL_WINDOW_SIZE)
        alt = centered_crop(full_alt, MODEL_WINDOW_SIZE)
        differences = [index for index, pair in enumerate(zip(wt, alt)) if pair[0] != pair[1]]
        if differences != [crop_index]:
            raise ValueError(
                f"{row.cpg_id}: REF/ALT crops differ at {differences}, expected only {crop_index}"
            )
        if wt[crop_index] != str(row.ref) or alt[crop_index] != str(row.alt):
            raise ValueError(f"{row.cpg_id}: model-crop REF/ALT reconstruction failed")
        if wt[MODEL_TARGET_C_INDEX : MODEL_TARGET_G_INDEX + 1] != "CG":
            raise ValueError(f"{row.cpg_id}: REF crop does not preserve the target CpG")
        if alt[MODEL_TARGET_C_INDEX : MODEL_TARGET_G_INDEX + 1] != "CG":
            raise ValueError(f"{row.cpg_id}: ALT crop alters the protected target CpG")

        wt_sequences.append(wt)
        alt_sequences.append(alt)
        mutation_indices_5kb.append(full_index)
        mutation_indices_1kb.append(crop_index)
        wt_hashes.append(sha256_text(wt))
        alt_hashes.append(sha256_text(alt))

    context_values = merged[TABULAR_FEATURES].to_numpy(dtype=np.float32, copy=True)
    context_missing = merged[MISSING_FEATURES].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(context_values).all():
        raise ValueError("Non-finite SilentMethyl context value after train-derived preprocessing")
    if not np.isin(context_missing, [0.0, 1.0]).all():
        raise ValueError("Context missingness indicators must be binary")

    cohort = mqtl.copy()
    cohort["probeID"] = merged["probeID"].astype(str).to_numpy()
    cohort["test_chr"] = test_chr.to_numpy()
    cohort["test_cpg_pos0"] = test_pos.astype(np.int64).to_numpy()
    cohort["true_median_beta"] = merged["Median_Beta"].to_numpy(dtype=float)
    cohort["true_m_value"] = merged["M_Value_Target"].to_numpy(dtype=float)
    cohort["mutation_index_5000_0based"] = mutation_indices_5kb
    cohort["mutation_index_1000_0based"] = mutation_indices_1kb
    cohort["wt_1000bp_sha256"] = wt_hashes
    cohort["alt_1000bp_sha256"] = alt_hashes
    cohort["slope_effect_allele"] = args.slope_effect_allele.upper()
    cohort["slope_alt_aligned"] = (
        cohort["slope"].to_numpy(dtype=float)
        if args.slope_effect_allele == "alt"
        else -cohort["slope"].to_numpy(dtype=float)
    )
    cohort["absolute_distance_bp"] = cohort["offset_from_cpg_C"].abs()
    cohort["shared_variant_cpg_count"] = cohort.groupby("variant_id")["cpg_id"].transform("size")

    LOGGER.info(
        "Validated held-out positive-control cohort: n=%d CpGs, %d unique variants, chromosomes=%s",
        len(cohort),
        cohort["variant_id"].nunique(),
        sorted(cohort["cpg_chr"].unique()),
    )
    return (
        cohort,
        wt_sequences,
        alt_sequences,
        torch.from_numpy(context_values),
        torch.from_numpy(context_missing),
    )


def make_rc_context(
    context: torch.Tensor,
    missing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rc_context = context.clone()
    rc_missing = missing.clone()
    rc_context[:, -2], rc_context[:, -1] = context[:, -1].clone(), context[:, -2].clone()
    rc_missing[:, -2], rc_missing[:, -1] = missing[:, -1].clone(), missing[:, -2].clone()
    return rc_context, rc_missing


@torch.inference_mode()
def infer_sequences(
    model: torch.nn.Module,
    model_name: str,
    tokenizer,
    sequences: list[str],
    context: torch.Tensor,
    missing: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    description: str,
) -> dict[str, np.ndarray]:
    m_values: list[np.ndarray] = []
    beta_values: list[np.ndarray] = []
    gate_values: list[np.ndarray] = []

    for start in tqdm(range(0, len(sequences), batch_size), desc=description, leave=False):
        stop = min(start + batch_size, len(sequences))
        encoded = tokenizer(
            sequences[start:stop],
            truncation=True,
            max_length=MODEL_WINDOW_SIZE,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with autocast_context(device, bool(use_amp and device.type == "cuda")):
            if model_name == "fusion":
                _, m_pred, gates = model(
                    context[start:stop].to(device),
                    missing[start:stop].to(device),
                    input_ids,
                    attention_mask,
                )
            elif model_name == "sequence":
                _, m_pred = model(input_ids, attention_mask)
                gates = None
            else:
                raise ValueError(f"Unknown model: {model_name}")

        if not use_amp and m_pred.dtype != torch.float32:
            raise RuntimeError(
                f"FP32 inference requested, but {model_name} regression output is {m_pred.dtype}"
            )
        m_float = m_pred.float()
        beta = m_to_beta_tensor(m_float)
        m_values.append(m_float.detach().cpu().numpy().reshape(-1))
        beta_values.append(beta.detach().cpu().numpy().reshape(-1))
        if gates is not None:
            gate_values.append(gates.detach().cpu().float().numpy())

    output = {
        "m": np.concatenate(m_values).astype(np.float64),
        "beta": np.concatenate(beta_values).astype(np.float64),
    }
    if gate_values:
        output["gates"] = np.concatenate(gate_values, axis=0).astype(np.float64)
    return output


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def instantiate_model(
    model_name: str,
    args: argparse.Namespace,
    weights_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    if model_name == "fusion":
        model = FusionModel(
            args.model_path,
            fusion_mode="gated",
            tabular_dim=len(TABULAR_FEATURES),
            local_dir=args.local_model_dir,
        )
    elif model_name == "sequence":
        model = SequenceOnlyModel(args.model_path, args.local_model_dir)
    else:
        raise ValueError(model_name)

    state = load_model_state(str(weights_path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    if not args.amp:
        model = model.float()
    model = model.to(device)
    model.eval()
    dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    )
    LOGGER.info("%s checkpoint model parameter dtypes: %s", model_name, dtypes)
    if not args.amp and dtypes != ["torch.float32"]:
        raise RuntimeError(f"FP32 requested, but {model_name} parameters have dtypes={dtypes}")
    return model


def gate_share(gates: np.ndarray) -> np.ndarray:
    return gates[:, 0] / np.clip(gates[:, 0] + gates[:, 1], 1e-12, None)


def score_model_seed(
    model_name: str,
    seed: int,
    cohort: pd.DataFrame,
    wt_sequences: list[str],
    alt_sequences: list[str],
    context: torch.Tensor,
    missing: torch.Tensor,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    template = (
        args.fusion_weights_template if model_name == "fusion" else args.sequence_weights_template
    )
    weights_path = Path(template.format(seed=seed))
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {model_name} seed-{seed} checkpoint: {weights_path}")

    set_seed(seed)
    LOGGER.info("Loading %s seed %d from %s", model_name, seed, weights_path)
    model = instantiate_model(model_name, args, weights_path, device)
    rc_context, rc_missing = make_rc_context(context, missing)
    wt_rc_sequences = [reverse_complement(sequence) for sequence in wt_sequences]
    alt_rc_sequences = [reverse_complement(sequence) for sequence in alt_sequences]

    outputs = {}
    for label, sequences, tab, tab_missing in (
        ("wt_fwd", wt_sequences, context, missing),
        ("wt_rc", wt_rc_sequences, rc_context, rc_missing),
        ("alt_fwd", alt_sequences, context, missing),
        ("alt_rc", alt_rc_sequences, rc_context, rc_missing),
    ):
        outputs[label] = infer_sequences(
            model,
            model_name,
            tokenizer,
            sequences,
            tab,
            tab_missing,
            args.batch_size,
            device,
            args.amp,
            f"{model_name} seed{seed} {label}",
        )

    wt_m_avg = (outputs["wt_fwd"]["m"] + outputs["wt_rc"]["m"]) / 2.0
    alt_m_avg = (outputs["alt_fwd"]["m"] + outputs["alt_rc"]["m"]) / 2.0
    wt_beta_avg = (outputs["wt_fwd"]["beta"] + outputs["wt_rc"]["beta"]) / 2.0
    alt_beta_avg = (outputs["alt_fwd"]["beta"] + outputs["alt_rc"]["beta"]) / 2.0

    scored = cohort.copy()
    scored["model"] = model_name
    scored["seed"] = int(seed)
    scored["weights_path"] = str(weights_path)
    scored["weights_sha256"] = sha256_file(weights_path)
    for prefix in ("wt_fwd", "wt_rc", "alt_fwd", "alt_rc"):
        scored[f"{prefix}_m"] = outputs[prefix]["m"]
        scored[f"{prefix}_beta"] = outputs[prefix]["beta"]
    scored["wt_m_rc_avg"] = wt_m_avg
    scored["alt_m_rc_avg"] = alt_m_avg
    scored["wt_beta_rc_avg"] = wt_beta_avg
    scored["alt_beta_rc_avg"] = alt_beta_avg
    scored["delta_m_fwd"] = outputs["alt_fwd"]["m"] - outputs["wt_fwd"]["m"]
    scored["delta_m_rc"] = outputs["alt_rc"]["m"] - outputs["wt_rc"]["m"]
    scored["predicted_delta_m"] = alt_m_avg - wt_m_avg
    scored["delta_beta_fwd"] = outputs["alt_fwd"]["beta"] - outputs["wt_fwd"]["beta"]
    scored["delta_beta_rc"] = outputs["alt_rc"]["beta"] - outputs["wt_rc"]["beta"]
    scored["predicted_delta_beta"] = alt_beta_avg - wt_beta_avg
    scored["delta_m_rc_abs_difference"] = np.abs(
        scored["delta_m_fwd"] - scored["delta_m_rc"]
    )
    scored["delta_beta_rc_abs_difference"] = np.abs(
        scored["delta_beta_fwd"] - scored["delta_beta_rc"]
    )
    scored["delta_m_rc_sign_agree"] = (
        np.sign(scored["delta_m_fwd"]) == np.sign(scored["delta_m_rc"])
    ).astype(int)
    scored["delta_beta_rc_sign_agree"] = (
        np.sign(scored["delta_beta_fwd"]) == np.sign(scored["delta_beta_rc"])
    ).astype(int)

    if model_name == "fusion":
        for prefix in ("wt_fwd", "wt_rc", "alt_fwd", "alt_rc"):
            gates = outputs[prefix]["gates"]
            scored[f"{prefix}_gate_dna"] = gates[:, 0]
            scored[f"{prefix}_gate_epi"] = gates[:, 1]
            scored[f"{prefix}_gate_dna_share"] = gate_share(gates)

    manifest = {
        "model": model_name,
        "seed": int(seed),
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scored, manifest


def aggregate_seed_predictions(long_scores: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    prediction_columns = [
        "wt_m_rc_avg",
        "alt_m_rc_avg",
        "wt_beta_rc_avg",
        "alt_beta_rc_avg",
        "delta_m_fwd",
        "delta_m_rc",
        "predicted_delta_m",
        "delta_beta_fwd",
        "delta_beta_rc",
        "predicted_delta_beta",
        "delta_m_rc_abs_difference",
        "delta_beta_rc_abs_difference",
        "delta_m_rc_sign_agree",
        "delta_beta_rc_sign_agree",
    ]
    gate_columns = [
        column
        for column in long_scores.columns
        if "_gate_" in column and pd.api.types.is_numeric_dtype(long_scores[column])
    ]

    for model_name, model_frame in long_scores.groupby("model", sort=False):
        grouped = model_frame.groupby(["cpg_id", "variant_id"], sort=False)
        means = grouped[prediction_columns + gate_columns].mean().reset_index()
        means = means.rename(
            columns={
                column: f"{column}_seed_mean"
                for column in prediction_columns + gate_columns
            }
        )
        standard_deviations = grouped[["predicted_delta_m", "predicted_delta_beta"]].std(
            ddof=0
        ).reset_index()
        standard_deviations = standard_deviations.rename(
            columns={
                "predicted_delta_m": "predicted_delta_m_seed_sd",
                "predicted_delta_beta": "predicted_delta_beta_seed_sd",
            }
        )
        base = cohort.merge(means, on=["cpg_id", "variant_id"], validate="one_to_one")
        base = base.merge(
            standard_deviations,
            on=["cpg_id", "variant_id"],
            validate="one_to_one",
        )
        base["model"] = model_name
        base["seed_count"] = int(model_frame["seed"].nunique())
        base["seeds"] = ",".join(str(value) for value in sorted(model_frame["seed"].unique()))
        base["predicted_delta_m"] = base["predicted_delta_m_seed_mean"]
        base["predicted_delta_beta"] = base["predicted_delta_beta_seed_mean"]
        rows.append(base)
    return pd.concat(rows, ignore_index=True)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    result = pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def point_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    slope = frame["slope_alt_aligned"].to_numpy(dtype=float)
    delta_m = frame["predicted_delta_m"].to_numpy(dtype=float)
    delta_beta = frame["predicted_delta_beta"].to_numpy(dtype=float)
    if not (np.isfinite(slope).all() and np.isfinite(delta_m).all() and np.isfinite(delta_beta).all()):
        raise ValueError("Non-finite effect found during positive-control statistics")

    rho_m, rho_m_p = safe_spearman(delta_m, slope)
    rho_beta, rho_beta_p = safe_spearman(delta_beta, slope)
    rho_abs_m, rho_abs_m_p = safe_spearman(np.abs(delta_m), np.abs(slope))
    rho_abs_beta, rho_abs_beta_p = safe_spearman(np.abs(delta_beta), np.abs(slope))
    pearson_m, pearson_m_p = safe_pearson(delta_m, slope)
    pearson_beta, pearson_beta_p = safe_pearson(delta_beta, slope)

    observed_positive = (slope > 0).astype(int)
    auc_m = float(roc_auc_score(observed_positive, delta_m)) if len(np.unique(observed_positive)) == 2 else np.nan
    auc_beta = (
        float(roc_auc_score(observed_positive, delta_beta))
        if len(np.unique(observed_positive)) == 2
        else np.nan
    )
    predicted_sign = np.sign(delta_m)
    observed_sign = np.sign(slope)
    direction_agreement = predicted_sign == observed_sign
    nonzero = predicted_sign != 0
    successes_all = int(direction_agreement.sum())
    successes_nonzero = int(direction_agreement[nonzero].sum())
    n_all = int(len(frame))
    n_nonzero = int(nonzero.sum())

    binomial_greater = binomtest(successes_all, n_all, p=0.5, alternative="greater")
    binomial_two_sided = binomtest(successes_all, n_all, p=0.5, alternative="two-sided")
    return {
        "n_loci": n_all,
        "n_unique_variants": int(frame["variant_id"].nunique()),
        "n_positive_slopes": int(observed_positive.sum()),
        "n_negative_slopes": int((observed_positive == 0).sum()),
        "n_exact_zero_predicted_delta_m": int((~nonzero).sum()),
        "spearman_delta_m": rho_m,
        "spearman_delta_m_asymptotic_p": rho_m_p,
        "spearman_delta_beta": rho_beta,
        "spearman_delta_beta_asymptotic_p": rho_beta_p,
        "spearman_absolute_delta_m_vs_absolute_slope": rho_abs_m,
        "spearman_absolute_delta_m_vs_absolute_slope_asymptotic_p": rho_abs_m_p,
        "spearman_absolute_delta_beta_vs_absolute_slope": rho_abs_beta,
        "spearman_absolute_delta_beta_vs_absolute_slope_asymptotic_p": rho_abs_beta_p,
        "pearson_delta_m": pearson_m,
        "pearson_delta_m_p": pearson_m_p,
        "pearson_delta_beta": pearson_beta,
        "pearson_delta_beta_p": pearson_beta_p,
        "direction_successes_all": successes_all,
        "direction_concordance_all": successes_all / n_all,
        "direction_successes_nonzero": successes_nonzero,
        "direction_n_nonzero": n_nonzero,
        "direction_concordance_nonzero": successes_nonzero / n_nonzero if n_nonzero else np.nan,
        "direction_binomial_greater_p": float(binomial_greater.pvalue),
        "direction_binomial_two_sided_p": float(binomial_two_sided.pvalue),
        "auc_positive_slope_delta_m": auc_m,
        "auc_positive_slope_delta_beta": auc_beta,
        "median_absolute_predicted_delta_m": float(np.median(np.abs(delta_m))),
        "median_absolute_predicted_delta_beta": float(np.median(np.abs(delta_beta))),
        "mean_delta_m_rc_abs_difference": float(frame["delta_m_rc_abs_difference_seed_mean"].mean()),
        "mean_delta_beta_rc_abs_difference": float(frame["delta_beta_rc_abs_difference_seed_mean"].mean()),
        "delta_m_rc_sign_agreement_fraction": float(frame["delta_m_rc_sign_agree_seed_mean"].mean()),
        "delta_beta_rc_sign_agreement_fraction": float(frame["delta_beta_rc_sign_agree_seed_mean"].mean()),
    }


def cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if replicates == 0:
        return {}
    rng = np.random.default_rng(seed)
    cluster_to_indices = {
        variant: indices.to_numpy(dtype=int)
        for variant, indices in frame.groupby("variant_id", sort=False).groups.items()
    }
    clusters = np.asarray(list(cluster_to_indices), dtype=object)
    distributions = {
        "spearman_delta_m": [],
        "spearman_delta_beta": [],
        "spearman_absolute_delta_m_vs_absolute_slope": [],
        "spearman_absolute_delta_beta_vs_absolute_slope": [],
        "direction_concordance_all": [],
        "auc_positive_slope_delta_m": [],
        "auc_positive_slope_delta_beta": [],
    }

    for _ in tqdm(range(replicates), desc="Cluster bootstrap", leave=False):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([cluster_to_indices[cluster] for cluster in sampled])
        sample = frame.iloc[indices]
        slope = sample["slope_alt_aligned"].to_numpy(dtype=float)
        delta_m = sample["predicted_delta_m"].to_numpy(dtype=float)
        delta_beta = sample["predicted_delta_beta"].to_numpy(dtype=float)
        distributions["spearman_delta_m"].append(safe_spearman(delta_m, slope)[0])
        distributions["spearman_delta_beta"].append(safe_spearman(delta_beta, slope)[0])
        distributions["spearman_absolute_delta_m_vs_absolute_slope"].append(
            safe_spearman(np.abs(delta_m), np.abs(slope))[0]
        )
        distributions["spearman_absolute_delta_beta_vs_absolute_slope"].append(
            safe_spearman(np.abs(delta_beta), np.abs(slope))[0]
        )
        distributions["direction_concordance_all"].append(
            float(np.mean(np.sign(delta_m) == np.sign(slope)))
        )
        labels = (slope > 0).astype(int)
        if len(np.unique(labels)) == 2:
            distributions["auc_positive_slope_delta_m"].append(
                float(roc_auc_score(labels, delta_m))
            )
            distributions["auc_positive_slope_delta_beta"].append(
                float(roc_auc_score(labels, delta_beta))
            )

    intervals: dict[str, float] = {}
    for name, values in distributions.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if len(array) == 0:
            intervals[f"{name}_cluster_bootstrap_ci_low"] = np.nan
            intervals[f"{name}_cluster_bootstrap_ci_high"] = np.nan
        else:
            low, high = np.quantile(array, [0.025, 0.975])
            intervals[f"{name}_cluster_bootstrap_ci_low"] = float(low)
            intervals[f"{name}_cluster_bootstrap_ci_high"] = float(high)
    return intervals


def cluster_sign_flip_permutation_p(
    frame: pd.DataFrame,
    effect_column: str,
    replicates: int,
    seed: int,
) -> float:
    if replicates == 0:
        return np.nan
    slope = frame["slope_alt_aligned"].to_numpy(dtype=float)
    effect = frame[effect_column].to_numpy(dtype=float)
    observed = safe_spearman(effect, slope)[0]
    if not np.isfinite(observed):
        return np.nan

    cluster_codes, unique_clusters = pd.factorize(frame["variant_id"], sort=False)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in tqdm(range(replicates), desc=f"Sign-flip {effect_column}", leave=False):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(unique_clusters))
        statistic = safe_spearman(effect * signs[cluster_codes], slope)[0]
        if np.isfinite(statistic) and abs(statistic) >= abs(observed):
            exceedances += 1
    return float((exceedances + 1) / (replicates + 1))


def compute_metrics(
    aggregate: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    nested = {}
    for model_index, (model_name, frame) in enumerate(aggregate.groupby("model", sort=False)):
        frame = frame.reset_index(drop=True)
        metrics = point_statistics(frame)
        metrics.update(
            cluster_bootstrap_intervals(
                frame,
                args.bootstrap_replicates,
                args.statistics_seed + 1000 * model_index,
            )
        )
        metrics["spearman_delta_m_cluster_sign_flip_p"] = cluster_sign_flip_permutation_p(
            frame,
            "predicted_delta_m",
            args.permutation_replicates,
            args.statistics_seed + 1000 * model_index + 101,
        )
        metrics["spearman_delta_beta_cluster_sign_flip_p"] = cluster_sign_flip_permutation_p(
            frame,
            "predicted_delta_beta",
            args.permutation_replicates,
            args.statistics_seed + 1000 * model_index + 202,
        )
        metrics.update(
            {
                "model": model_name,
                "seed_count": int(frame["seed_count"].iloc[0]),
                "seeds": str(frame["seeds"].iloc[0]),
                "slope_effect_allele": str(frame["slope_effect_allele"].iloc[0]),
                "bootstrap_unit": "eGTEx variant_id cluster",
                "permutation_null": "independent random sign flip per variant_id cluster",
            }
        )
        nested[model_name] = metrics
        rows.append(metrics)
    return pd.DataFrame(rows), nested


def compute_probe_overlap_sensitivity(
    aggregate: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    """Recompute all statistics after two post hoc probe-footprint exclusions."""
    definitions = (
        (
            "exclude_closed_probe_span",
            "variant_overlaps_probe_span_closed",
            "exclude variants inside the inclusive annotated probe span",
        ),
        (
            "exclude_probe_plus_extension_conservative",
            "variant_overlaps_probe_or_extension_conservative",
            "exclude variants inside or within one bp of the annotated probe span",
        ),
    )
    frames = []
    nested: dict[str, dict] = {}
    for subset_index, (label, exclusion_column, description) in enumerate(definitions):
        if exclusion_column not in aggregate.columns:
            raise ValueError(f"Aggregate predictions lack {exclusion_column}")
        retained = aggregate.loc[~aggregate[exclusion_column].astype(bool)].copy()
        if retained.empty:
            raise ValueError(f"Probe sensitivity subset {label} is empty")

        # Use a distinct deterministic resampling stream for each sensitivity
        # definition without changing the frozen point estimates.
        sensitivity_args = argparse.Namespace(**vars(args))
        sensitivity_args.statistics_seed = args.statistics_seed + 100_000 * (subset_index + 1)
        metrics_frame, metrics_nested = compute_metrics(retained, sensitivity_args)
        metrics_frame.insert(0, "analysis_subset", label)
        metrics_frame.insert(1, "exclusion_rule", description)
        metrics_frame.insert(
            2,
            "n_excluded_loci",
            metrics_frame["model"].map(
                aggregate.groupby("model").size() - retained.groupby("model").size()
            ).astype(int),
        )
        frames.append(metrics_frame)
        nested[label] = {
            "exclusion_rule": description,
            "metrics": metrics_nested,
        }
    return pd.concat(frames, ignore_index=True), nested


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    if len(values) == 1:
        return np.array([50.0])
    return 100.0 * (ranks - 1.0) / (len(values) - 1.0)


def _zero_rank_boundary(values: np.ndarray, ranks: np.ndarray) -> float:
    negative = ranks[values < 0]
    positive = ranks[values > 0]
    if len(negative) and len(positive):
        return float((negative.max() + positive.min()) / 2.0)
    return 50.0


def save_model_plots(frame: pd.DataFrame, metrics: dict, output_dir: Path) -> None:
    """Write the single primary rank plot for one model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(frame["model"].iloc[0])
    observed = frame["slope_alt_aligned"].to_numpy(dtype=float)
    predicted = frame["predicted_delta_m"].to_numpy(dtype=float)
    observed_rank = _percentile_ranks(observed)
    predicted_rank = _percentile_ranks(predicted)
    concordant = np.sign(observed) == np.sign(predicted)

    fig, axis = plt.subplots(figsize=(6.4, 6.0))
    for mask, label, color, marker in (
        (concordant, "Concordant direction", "#20854e", "o"),
        (~concordant, "Discordant direction", "#d95f02", "X"),
    ):
        axis.scatter(
            predicted_rank[mask],
            observed_rank[mask],
            s=58,
            alpha=0.84,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.55,
            label=f"{label} (n={int(mask.sum())})",
            zorder=3,
        )

    axis.plot(
        [0, 100], [0, 100], color="0.35", linewidth=1.2,
        linestyle="--", label="Identical rank",
    )
    axis.axvline(
        _zero_rank_boundary(predicted, predicted_rank),
        color="0.65", linewidth=0.9, linestyle=":",
    )
    axis.axhline(
        _zero_rank_boundary(observed, observed_rank),
        color="0.65", linewidth=0.9, linestyle=":",
    )
    axis.set(
        xlim=(-3, 103),
        ylim=(-3, 103),
        aspect="equal",
        xlabel=r"Predicted $\Delta\hat{M}$ signed percentile rank",
        ylabel="eGTEx ALT-dosage slope signed percentile rank",
        title=f"eGTEx breast mQTL positive control: {model_name}",
    )
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_yticks(np.arange(0, 101, 20))
    axis.text(
        0.04,
        0.96,
        f"Spearman $\\rho$ = {metrics['spearman_delta_m']:.3f}\n"
        f"Direction concordance = {int(concordant.sum())}/{len(frame)} "
        f"({concordant.mean():.1%})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.90,
              "edgecolor": "0.8"},
    )
    axis.legend(frameon=False, loc="lower right")
    axis.grid(alpha=0.15, zorder=0)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_effect_rank_scatter.png", dpi=300)
    plt.close(fig)


def save_cross_model_plot(aggregate: pd.DataFrame, output_dir: Path) -> None:
    if set(aggregate["model"].unique()) != {"fusion", "sequence"}:
        return
    wide = aggregate.pivot(
        index=["cpg_id", "variant_id"],
        columns="model",
        values="predicted_delta_m",
    ).dropna()
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(wide["sequence"], wide["fusion"], s=35, alpha=0.8)
    low = float(min(wide.min()))
    high = float(max(wide.max()))
    axis.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="black")
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.axvline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("Sequence-only predicted delta M")
    axis.set_ylabel("Fusion predicted delta M")
    axis.set_title("Held-out eGTEx mQTL effects across model variants")
    fig.tight_layout()
    fig.savefig(output_dir / "fusion_vs_sequence_delta_m.png", dpi=300)
    plt.close(fig)


def _loo_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame["slope_alt_aligned"].to_numpy(dtype=float)
    predicted = frame["predicted_delta_m"].to_numpy(dtype=float)
    labels = (observed > 0).astype(int)
    auc = (
        float(roc_auc_score(labels, predicted))
        if np.unique(labels).size == 2 else float("nan")
    )
    return {
        "n_loci": int(len(frame)),
        "n_variants": int(frame["variant_id"].nunique()),
        "spearman_delta_m": safe_spearman(predicted, observed)[0],
        "direction_concordance": float(
            np.mean(np.sign(predicted) == np.sign(observed))
        ),
        "auc_positive_slope_delta_m": auc,
    }


def save_leave_one_variant_out(aggregate: pd.DataFrame, output_dir: Path) -> None:
    """Run variant-cluster influence analysis on each model's seed aggregate."""
    rows: list[dict[str, object]] = []
    summaries: list[str] = []
    for model_name, group in aggregate.groupby("model", sort=True):
        group = group.reset_index(drop=True)
        full = _loo_metrics(group)
        model_rows: list[dict[str, object]] = []
        for omitted_variant in sorted(group["variant_id"].astype(str).unique()):
            retained = group[group["variant_id"].astype(str) != omitted_variant]
            loo = _loo_metrics(retained)
            row: dict[str, object] = {
                "model": model_name,
                "seeds": str(group["seeds"].iloc[0]),
                "omitted_variant_id": omitted_variant,
                "n_removed_loci": int(len(group) - len(retained)),
                **loo,
            }
            for metric in (
                "spearman_delta_m",
                "direction_concordance",
                "auc_positive_slope_delta_m",
            ):
                row[f"full_{metric}"] = full[metric]
                row[f"change_{metric}"] = float(loo[metric]) - float(full[metric])
            rows.append(row)
            model_rows.append(row)

        current = pd.DataFrame(model_rows)
        summaries.extend([
            f"model={model_name}, seeds={group['seeds'].iloc[0]}",
            (
                f"  Full: n={full['n_loci']}, variants={full['n_variants']}, "
                f"rho={full['spearman_delta_m']:.6f}, "
                f"direction={full['direction_concordance']:.6f}, "
                f"AUROC={full['auc_positive_slope_delta_m']:.6f}"
            ),
        ])
        for metric, short_name in (
            ("spearman_delta_m", "rho"),
            ("direction_concordance", "direction"),
            ("auc_positive_slope_delta_m", "AUROC"),
        ):
            change_column = f"change_{metric}"
            worst = current.loc[current[change_column].abs().idxmax()]
            summaries.append(
                f"  LOO {short_name}: range "
                f"[{current[metric].min():.6f}, {current[metric].max():.6f}], "
                f"largest absolute change={abs(worst[change_column]):.6f} "
                f"after omitting {worst['omitted_variant_id']} "
                f"({int(worst['n_removed_loci'])} locus/loci)"
            )
        summaries.append("")

    atomic_csv(pd.DataFrame(rows), output_dir / "mqtl_leave_one_variant_out.csv")
    summary_path = output_dir / "mqtl_leave_one_variant_out_summary.txt"
    temporary = summary_path.with_name(summary_path.name + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(summaries).rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)


def write_text_summary(
    metrics_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
    probe_audit: dict,
    path: Path,
    smoke_test: bool,
) -> None:
    lines = []
    if smoke_test:
        lines.append("SMOKE TEST ONLY -- NOT REPORTABLE")
        lines.append("")
    lines.append("SilentMethyl held-out eGTEx breast mQTL positive control")
    lines.append("Slope orientation: increasing VCF ALT-allele dosage")
    lines.append("Primary interpretation: direction and rank agreement, not slope-scale calibration")
    lines.append("")
    for row in metrics_frame.itertuples(index=False):
        lines.extend(
            [
                f"Model: {row.model}",
                f"  loci: {row.n_loci}",
                f"  unique variants: {row.n_unique_variants}",
                f"  Spearman(delta M, slope): {row.spearman_delta_m:.6f}",
                "  Spearman(abs(delta M), abs(slope)): "
                f"{row.spearman_absolute_delta_m_vs_absolute_slope:.6f}",
                f"  cluster sign-flip p: {row.spearman_delta_m_cluster_sign_flip_p:.6g}",
                f"  direction concordance: {row.direction_concordance_all:.6f}",
                f"  exact binomial greater p: {row.direction_binomial_greater_p:.6g}",
                f"  AUROC for positive slope: {row.auc_positive_slope_delta_m:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "HM450 probe-footprint audit:",
            f"  annotated probe-span overlaps: {probe_audit['n_probe_span_closed_overlap']}",
            "  conservative probe/extension-adjacent overlaps: "
            f"{probe_audit['n_probe_or_extension_conservative_overlap']}",
            "",
            "Post hoc probe-footprint sensitivity:",
        ]
    )
    for row in sensitivity_frame.itertuples(index=False):
        lines.extend(
            [
                f"  subset={row.analysis_subset}, model={row.model}",
                f"    retained={row.n_loci}, excluded={row.n_excluded_loci}",
                f"    Spearman(delta M, slope)={row.spearman_delta_m:.6f}",
                f"    direction={row.direction_concordance_all:.6f}",
                f"    AUROC={row.auc_positive_slope_delta_m:.6f}",
                "    Spearman(abs(delta M), abs(slope))="
                f"{row.spearman_absolute_delta_m_vs_absolute_slope:.6f}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines).rstrip() + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    validate_cli(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort, wt_sequences, alt_sequences, context, missing = load_and_build_cohort(args)
    cohort, probe_audit = annotate_hm450_probe_geometry(cohort, args.hm450_manifest)
    atomic_csv(cohort, output_dir / "validated_heldout_cohort.csv")
    atomic_json(probe_audit, output_dir / "hm450_probe_overlap_audit.json")

    device = resolve_device(args.device)
    LOGGER.info(
        "Device=%s | models=%s | seeds=%s | precision=%s",
        device,
        args.models,
        args.seeds,
        "AMP" if args.amp and device.type == "cuda" else "FP32",
    )
    tokenizer = get_tokenizer(args.model_path)
    score_frames = []
    checkpoint_manifest = []
    for model_name in args.models:
        for seed in args.seeds:
            scores, manifest = score_model_seed(
                model_name,
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
            atomic_csv(
                scores,
                output_dir / model_name / f"seed{seed}" / "mqtl_predictions.csv",
            )

    long_scores = pd.concat(score_frames, ignore_index=True)
    atomic_csv(long_scores, output_dir / "mqtl_predictions_all_models_seeds.csv")
    aggregate = aggregate_seed_predictions(long_scores, cohort)
    atomic_csv(aggregate, output_dir / "mqtl_predictions_seed_aggregate.csv")

    metrics_frame, metrics_nested = compute_metrics(aggregate, args)
    atomic_csv(metrics_frame, output_dir / "mqtl_positive_control_metrics.csv")
    sensitivity_frame, sensitivity_nested = compute_probe_overlap_sensitivity(aggregate, args)
    atomic_csv(
        sensitivity_frame,
        output_dir / "mqtl_probe_overlap_sensitivity_metrics.csv",
    )

    plot_dir = output_dir / "plots"
    for model_name, model_frame in aggregate.groupby("model", sort=False):
        save_model_plots(model_frame, metrics_nested[model_name], plot_dir)
    save_cross_model_plot(aggregate, plot_dir)
    save_leave_one_variant_out(aggregate, output_dir)
    write_text_summary(
        metrics_frame,
        sensitivity_frame,
        probe_audit,
        output_dir / "mqtl_positive_control_summary.txt",
        smoke_test=args.max_rows > 0,
    )

    run_summary = {
        "analysis": "held-out eGTEx Breast-Mammary Tissue lead-mQTL positive control",
        "reportable_primary_run": args.max_rows == 0,
        "mqtl_csv": str(Path(args.mqtl_csv)),
        "mqtl_csv_sha256": sha256_file(args.mqtl_csv),
        "test_csv": str(Path(args.test_csv)),
        "test_csv_sha256": sha256_file(args.test_csv),
        "hm450_manifest": str(Path(args.hm450_manifest)),
        "hm450_manifest_sha256": sha256_file(args.hm450_manifest),
        "n_loci": int(len(cohort)),
        "n_unique_variants": int(cohort["variant_id"].nunique()),
        "heldout_chromosomes": args.heldout_chromosomes,
        "window_size_bp": MODEL_WINDOW_SIZE,
        "target_c_index_1000_0based": MODEL_TARGET_C_INDEX,
        "target_c_index_5000_0based": FULL_TARGET_C_INDEX,
        "models": args.models,
        "seeds": args.seeds,
        "precision": "cuda_amp" if args.amp and device.type == "cuda" else "fp32_forced_and_audited",
        "inference": "REF/ALT forward and reverse-complement; average orientations before ALT-minus-REF subtraction",
        "context_counterfactual_policy": (
            "All regional epigenomic features and target-CpG PhyloP values are held fixed between "
            "REF and ALT; ordered target-base PhyloP features swap only under reverse complement."
        ),
        "slope_effect_allele_input": args.slope_effect_allele.upper(),
        "slope_alignment_used": "ALT minus REF",
        "primary_statistics": "Spearman rank correlation and direction concordance",
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_unit": "variant_id cluster",
        "permutation_replicates": int(args.permutation_replicates),
        "permutation_null": "independent random sign flip per variant_id cluster",
        "statistics_seed": int(args.statistics_seed),
        "checkpoints": checkpoint_manifest,
        "metrics": metrics_nested,
        "probe_overlap_audit": probe_audit,
        "probe_overlap_sensitivity": sensitivity_nested,
    }
    atomic_json(run_summary, output_dir / "run_summary.json")
    LOGGER.info("Finished eGTEx positive control. Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
