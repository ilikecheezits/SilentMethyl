#!/usr/bin/env python3
"""Audit processed SilentMethyl data files for integrity errors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

CSV_FIELD_LIMIT = min(sys.maxsize, 2_147_483_647)
csv.field_size_limit(CSV_FIELD_LIMIT)

DNA_RE = re.compile(r"^[ACGTN]+$")
SBS96_RE = re.compile(r"^[ACGT]\[[CT]>[ACGT]\][ACGT]$")
CENTER_C = 2499
CENTER_G = 2500
SLICE_START = 2450
SLICE_END = 2550
MAX_EXAMPLES = 5

EPIGENETIC_FEATURES = [
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

TRAIN_REQUIRED = [
    "chr", "pos", "probeID", "Healthy_5000bp_DNA",
    "Median_Beta", "M_Value_Target", "Binary_State_Target", "Split",
    "Healthy_100bp_DNA", "Healthy_100bp_DNA_RC",
    "Target_Base_PhyloP_100way_1_RC",
    "Target_Base_PhyloP_100way_2_RC",
    "Target_Base_PhyloP_100way_1_RC_Missing",
    "Target_Base_PhyloP_100way_2_RC_Missing",
] + EPIGENETIC_FEATURES + [f"{x}_Missing" for x in EPIGENETIC_FEATURES]

CAND_REQUIRED = [
    "Candidate_ID", "chr", "Variant_Position_1based",
    "Variant_Position_0based", "Reference_Allele", "Alternate_Allele",
    "Selected_Gene_Name", "Selected_Transcript_ID",
    "Selected_Transcript_Strand", "Reference_Codon", "Alternate_Codon",
    "Amino_Acid", "Reference_Trinucleotide", "SBS96_Class",
    "probeID", "pos", "Mutation_Offset_From_CpG",
    "Absolute_Distance_To_CpG", "Mutation_Index_5000_ZeroBased",
    "Model_Split", "Healthy_5000bp_DNA", "Mutated_5000bp_DNA",
    "Healthy_5000bp_DNA_RC", "Mutated_5000bp_DNA_RC",
    "Healthy_100bp_DNA", "Mutated_100bp_DNA",
    "Healthy_100bp_DNA_RC", "Mutated_100bp_DNA_RC",
    "GDC_SSM_IDs", "GDC_Case_IDs", "GDC_Case_Submitter_IDs",
    "GDC_Occurrence_Count", "All_Synonymous_Transcript_Annotations",
] + EPIGENETIC_FEATURES + [f"{x}_Missing" for x in EPIGENETIC_FEATURES] + [
    "Target_Base_PhyloP_100way_1_RC",
    "Target_Base_PhyloP_100way_2_RC",
    "Target_Base_PhyloP_100way_1_RC_Missing",
    "Target_Base_PhyloP_100way_2_RC_Missing",
]


def rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def finite_float(value: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"nonfinite value {value!r}")
    return x


def add_example(store: dict[str, list[str]], key: str, identifier: str) -> None:
    if len(store[key]) < MAX_EXAMPLES:
        store[key].append(identifier)


@dataclass
class NumericStats:
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    zero_count: int = 0
    negative_count: int = 0

    def add(self, x: float) -> None:
        self.count += 1
        self.total += x
        self.minimum = min(self.minimum, x)
        self.maximum = max(self.maximum, x)
        self.zero_count += int(x == 0.0)
        self.negative_count += int(x < 0.0)

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0, "mean": None, "min": None, "max": None,
                "zero_count": 0, "negative_count": 0,
            }
        return {
            "count": self.count,
            "mean": self.total / self.count,
            "min": self.minimum,
            "max": self.maximum,
            "zero_count": self.zero_count,
            "negative_count": self.negative_count,
        }


class Audit:
    def __init__(self) -> None:
        self.errors: Counter[str] = Counter()
        self.warnings: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = defaultdict(list)

    def error(self, key: str, identifier: str) -> None:
        self.errors[key] += 1
        add_example(self.examples, f"ERROR::{key}", identifier)

    def warning(self, key: str, identifier: str) -> None:
        self.warnings[key] += 1
        add_example(self.examples, f"WARNING::{key}", identifier)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def check_columns(path: Path, required: Iterable[str], audit: Audit) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            audit.error("empty_csv", str(path))
            return []
    missing = sorted(set(required) - set(header))
    for column in missing:
        audit.error("missing_required_column", f"{path.name}:{column}")
    return header


def expected_split(chrom: str, split_manifest: dict[str, Any]) -> str | None:
    if chrom in set(split_manifest["train_chromosomes"]):
        return "train"
    if chrom in set(split_manifest["validation_chromosomes"]):
        return "val"
    if chrom in set(split_manifest["test_chromosomes"]):
        return "test"
    return None


def audit_training_file(
    path: Path,
    split_name: str,
    split_manifest: dict[str, Any],
    imputation_values: dict[str, float],
    audit: Audit,
    global_probe_split: dict[str, str],
    progress_every: int,
) -> dict[str, Any]:
    check_columns(path, TRAIN_REQUIRED, audit)

    rows = 0
    chromosome_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    feature_stats = {name: NumericStats() for name in EPIGENETIC_FEATURES}
    missing_counts: Counter[str] = Counter()
    n_sequence_rows = 0
    local_probe_ids: set[str] = set()
    beta_stats = NumericStats()
    m_stats = NumericStats()
    max_m_reconstruction_error = 0.0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            ident = row.get("probeID", f"row-{rows}")
            if progress_every and rows % progress_every == 0:
                print(f"[{path.name}] audited {rows:,} rows", flush=True)

            chrom = row["chr"]
            chromosome_counts[chrom] += 1
            class_counts[row["Binary_State_Target"]] += 1

            if row["Split"] != split_name:
                audit.error("training_split_label_mismatch", ident)
            if expected_split(chrom, split_manifest) != split_name:
                audit.error("training_chromosome_split_mismatch", ident)

            if ident in local_probe_ids:
                audit.error("duplicate_probe_within_split", ident)
            local_probe_ids.add(ident)
            previous_split = global_probe_split.get(ident)
            if previous_split is not None and previous_split != split_name:
                audit.error("probe_present_across_splits", ident)
            else:
                global_probe_split[ident] = split_name

            try:
                pos = int(row["pos"])
                if pos < 0:
                    audit.error("negative_cpg_position", ident)
            except Exception:
                audit.error("invalid_cpg_position", ident)

            seq5 = row["Healthy_5000bp_DNA"].upper()
            seq100 = row["Healthy_100bp_DNA"].upper()
            seq100_rc = row["Healthy_100bp_DNA_RC"].upper()

            if len(seq5) != 5000:
                audit.error("training_5kb_length", ident)
            if len(seq100) != 100:
                audit.error("training_100bp_length", ident)
            if len(seq100_rc) != 100:
                audit.error("training_100bp_rc_length", ident)

            if not DNA_RE.fullmatch(seq5):
                audit.error("training_5kb_invalid_alphabet", ident)
            if not DNA_RE.fullmatch(seq100):
                audit.error("training_100bp_invalid_alphabet", ident)
            if "N" in seq5:
                n_sequence_rows += 1

            if len(seq5) == 5000 and seq5[CENTER_C:CENTER_G + 1] != "CG":
                audit.error("training_5kb_not_cpg_centered", ident)
            if len(seq100) == 100 and seq100[49:51] != "CG":
                audit.error("training_100bp_not_cpg_centered", ident)
            if len(seq100_rc) == 100 and seq100_rc[49:51] != "CG":
                audit.error("training_rc_100bp_not_cpg_centered", ident)
            if len(seq5) == 5000 and seq100 != seq5[SLICE_START:SLICE_END]:
                audit.error("training_100bp_not_exact_5kb_slice", ident)
            if seq100_rc != rc(seq100):
                audit.error("training_100bp_rc_incorrect", ident)

            try:
                beta = finite_float(row["Median_Beta"])
                m_value = finite_float(row["M_Value_Target"])
                binary = int(row["Binary_State_Target"])
                beta_stats.add(beta)
                m_stats.add(m_value)

                if not (0.0 <= beta <= 1.0):
                    audit.error("beta_out_of_range", ident)
                if binary not in (0, 1):
                    audit.error("binary_target_not_0_or_1", ident)
                if binary != int(beta > 0.5):
                    audit.error("binary_target_inconsistent_with_beta", ident)

                clipped = min(max(beta, 0.0001), 0.9999)
                expected_m = math.log2(clipped / (1.0 - clipped))
                err = abs(m_value - expected_m)
                max_m_reconstruction_error = max(max_m_reconstruction_error, err)
                if err > 1e-5:
                    audit.error("m_value_inconsistent_with_beta", ident)
            except Exception:
                audit.error("invalid_target_value", ident)

            for feature in EPIGENETIC_FEATURES:
                try:
                    x = finite_float(row[feature])
                    feature_stats[feature].add(x)
                    flag = int(row[f"{feature}_Missing"])
                    if flag not in (0, 1):
                        audit.error("missing_indicator_not_binary", f"{ident}:{feature}")
                    if flag == 1:
                        missing_counts[feature] += 1
                        expected = float(imputation_values[feature])
                        if not math.isclose(x, expected, rel_tol=0.0, abs_tol=2e-6):
                            audit.error("flagged_missing_value_not_training_imputation", f"{ident}:{feature}")
                except Exception:
                    audit.error("invalid_feature_or_missing_indicator", f"{ident}:{feature}")

            try:
                p1 = finite_float(row["Target_Base_PhyloP_100way_1"])
                p2 = finite_float(row["Target_Base_PhyloP_100way_2"])
                p1rc = finite_float(row["Target_Base_PhyloP_100way_1_RC"])
                p2rc = finite_float(row["Target_Base_PhyloP_100way_2_RC"])
                m1 = int(row["Target_Base_PhyloP_100way_1_Missing"])
                m2 = int(row["Target_Base_PhyloP_100way_2_Missing"])
                m1rc = int(row["Target_Base_PhyloP_100way_1_RC_Missing"])
                m2rc = int(row["Target_Base_PhyloP_100way_2_RC_Missing"])
                if not math.isclose(p1rc, p2, abs_tol=2e-6, rel_tol=0.0):
                    audit.error("training_phylo_rc_position1_not_swapped", ident)
                if not math.isclose(p2rc, p1, abs_tol=2e-6, rel_tol=0.0):
                    audit.error("training_phylo_rc_position2_not_swapped", ident)
                if m1rc != m2 or m2rc != m1:
                    audit.error("training_phylo_rc_missingness_not_swapped", ident)
            except Exception:
                audit.error("invalid_training_rc_phylo_fields", ident)

    for feature, stats in feature_stats.items():
        result = stats.result()
        if feature.startswith("Ref_") and result["min"] is not None and result["min"] < 0:
            audit.warning("negative_reference_signal_observed", f"{path.name}:{feature}:{result['min']}")
        if feature.startswith("Target_Base_PhyloP") and result["min"] is not None:
            if result["min"] < -25 or result["max"] > 25:
                audit.warning("extreme_phyloP_value", f"{path.name}:{feature}:{result['min']}..{result['max']}")

    return {
        "rows": rows,
        "unique_probe_ids": len(local_probe_ids),
        "chromosome_counts": dict(chromosome_counts),
        "binary_class_counts": dict(class_counts),
        "beta": beta_stats.result(),
        "m_value": m_stats.result(),
        "max_m_value_reconstruction_error": max_m_reconstruction_error,
        "rows_containing_N_in_5kb": n_sequence_rows,
        "missing_counts": dict(missing_counts),
        "feature_stats": {k: v.result() for k, v in feature_stats.items()},
    }


def parse_json_list(value: str, identifier: str, field: str, audit: Audit) -> list[Any]:
    try:
        obj = json.loads(value)
        if not isinstance(obj, list):
            raise ValueError("not a list")
        return obj
    except Exception:
        audit.error("invalid_json_list_field", f"{identifier}:{field}")
        return []


def audit_candidate_file(
    path: Path,
    split_manifest: dict[str, Any],
    imputation_values: dict[str, float],
    audit: Audit,
    progress_every: int,
) -> tuple[dict[str, Any], set[str], set[str]]:
    check_columns(path, CAND_REQUIRED, audit)

    rows = 0
    candidate_ids: set[str] = set()
    test_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    chromosome_counts: Counter[str] = Counter()
    feature_stats = {name: NumericStats() for name in EPIGENETIC_FEATURES}
    missing_counts: Counter[str] = Counter()
    central_cpg_destroyed = 0
    mutation_inside_100bp = 0
    mutation_outside_100bp = 0
    n_sequence_rows = 0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            ident = row.get("Candidate_ID", f"row-{rows}")
            if progress_every and rows % progress_every == 0:
                print(f"[{path.name}] audited {rows:,} rows", flush=True)

            if ident in candidate_ids:
                audit.error("duplicate_candidate_id", ident)
            candidate_ids.add(ident)

            chrom = row["chr"]
            model_split = row["Model_Split"]
            split_counts[model_split] += 1
            chromosome_counts[chrom] += 1
            if model_split == "test":
                test_ids.add(ident)
            if expected_split(chrom, split_manifest) != model_split:
                audit.error("candidate_chromosome_split_mismatch", ident)

            if not row["Selected_Transcript_ID"] or not row["Selected_Gene_Name"]:
                audit.error("candidate_missing_selected_annotation", ident)
            if row["Selected_Transcript_Strand"] not in ("+", "-"):
                audit.error("candidate_invalid_transcript_strand", ident)
            if len(row["Reference_Codon"]) != 3 or len(row["Alternate_Codon"]) != 3:
                audit.error("candidate_invalid_codon_length", ident)

            annotations = parse_json_list(
                row["All_Synonymous_Transcript_Annotations"],
                ident, "All_Synonymous_Transcript_Annotations", audit
            )
            if not annotations:
                audit.error("candidate_no_synonymous_annotation", ident)

            ref = row["Reference_Allele"].upper()
            alt = row["Alternate_Allele"].upper()
            if ref not in "ACGT" or alt not in "ACGT" or ref == alt or len(ref) != 1 or len(alt) != 1:
                audit.error("candidate_not_valid_snv", ident)

            try:
                pos1 = int(row["Variant_Position_1based"])
                pos0 = int(row["Variant_Position_0based"])
                cpg_pos = int(row["pos"])
                offset = int(row["Mutation_Offset_From_CpG"])
                abs_distance = int(row["Absolute_Distance_To_CpG"])
                mut_idx = int(row["Mutation_Index_5000_ZeroBased"])

                if pos0 != pos1 - 1:
                    audit.error("candidate_1based_0based_mismatch", ident)
                if offset != pos0 - cpg_pos:
                    audit.error("candidate_offset_coordinate_mismatch", ident)
                if abs_distance != abs(offset):
                    audit.error("candidate_absolute_distance_mismatch", ident)
                if mut_idx != CENTER_C + offset:
                    audit.error("candidate_mutation_index_offset_mismatch", ident)
                if not (0 <= mut_idx < 5000):
                    audit.error("candidate_mutation_index_out_of_bounds", ident)
            except Exception:
                audit.error("candidate_invalid_coordinate_field", ident)
                continue

            h5 = row["Healthy_5000bp_DNA"].upper()
            m5 = row["Mutated_5000bp_DNA"].upper()
            h5rc = row["Healthy_5000bp_DNA_RC"].upper()
            m5rc = row["Mutated_5000bp_DNA_RC"].upper()
            h100 = row["Healthy_100bp_DNA"].upper()
            m100 = row["Mutated_100bp_DNA"].upper()
            h100rc = row["Healthy_100bp_DNA_RC"].upper()
            m100rc = row["Mutated_100bp_DNA_RC"].upper()

            for name, seq, expected_len in [
                ("healthy_5kb", h5, 5000), ("mutated_5kb", m5, 5000),
                ("healthy_5kb_rc", h5rc, 5000), ("mutated_5kb_rc", m5rc, 5000),
                ("healthy_100bp", h100, 100), ("mutated_100bp", m100, 100),
                ("healthy_100bp_rc", h100rc, 100), ("mutated_100bp_rc", m100rc, 100),
            ]:
                if len(seq) != expected_len:
                    audit.error(f"candidate_{name}_length", ident)
                if not DNA_RE.fullmatch(seq):
                    audit.error(f"candidate_{name}_invalid_alphabet", ident)

            if "N" in h5 or "N" in m5:
                n_sequence_rows += 1

            if len(h5) == 5000 and h5[CENTER_C:CENTER_G + 1] != "CG":
                audit.error("candidate_healthy_5kb_not_cpg_centered", ident)
            if len(h100) == 100 and h100[49:51] != "CG":
                audit.error("candidate_healthy_100bp_not_cpg_centered", ident)
            if len(h5rc) == 5000 and h5rc[CENTER_C:CENTER_G + 1] != "CG":
                audit.error("candidate_healthy_rc_not_cpg_centered", ident)
            if len(h100rc) == 100 and h100rc[49:51] != "CG":
                audit.error("candidate_healthy_100bp_rc_not_cpg_centered", ident)

            if len(m5) == 5000 and m5[CENTER_C:CENTER_G + 1] != "CG":
                central_cpg_destroyed += 1
                audit.error("candidate_mutation_changes_target_cpg", ident)

            if h5rc != rc(h5):
                audit.error("candidate_healthy_5kb_rc_incorrect", ident)
            if m5rc != rc(m5):
                audit.error("candidate_mutated_5kb_rc_incorrect", ident)
            if h100 != h5[SLICE_START:SLICE_END]:
                audit.error("candidate_healthy_100bp_not_exact_5kb_slice", ident)
            if m100 != m5[SLICE_START:SLICE_END]:
                audit.error("candidate_mutated_100bp_not_exact_5kb_slice", ident)
            if h100rc != rc(h100):
                audit.error("candidate_healthy_100bp_rc_incorrect", ident)
            if m100rc != rc(m100):
                audit.error("candidate_mutated_100bp_rc_incorrect", ident)

            if len(h5) == 5000 and len(m5) == 5000 and 0 <= mut_idx < 5000:
                differences = [i for i, (a, b) in enumerate(zip(h5, m5)) if a != b]
                if differences != [mut_idx]:
                    audit.error("candidate_not_exactly_one_change_at_mutation_index", ident)
                if h5[mut_idx] != ref or m5[mut_idx] != alt:
                    audit.error("candidate_reference_or_alternate_not_at_mutation_index", ident)

                relative_100 = mut_idx - SLICE_START
                if 0 <= relative_100 < 100:
                    mutation_inside_100bp += 1
                    differences_100 = [i for i, (a, b) in enumerate(zip(h100, m100)) if a != b]
                    if differences_100 != [relative_100]:
                        audit.error("candidate_100bp_change_not_at_expected_index", ident)
                else:
                    mutation_outside_100bp += 1
                    if h100 != m100:
                        audit.error("candidate_100bp_changed_for_outside_window_mutation", ident)

            trinuc = row["Reference_Trinucleotide"].upper()
            sbs = row["SBS96_Class"].upper()
            if len(trinuc) != 3 or not DNA_RE.fullmatch(trinuc):
                audit.error("candidate_invalid_reference_trinucleotide", ident)
            if not SBS96_RE.fullmatch(sbs):
                audit.error("candidate_invalid_sbs96_class", ident)

            ssm_ids = parse_json_list(row["GDC_SSM_IDs"], ident, "GDC_SSM_IDs", audit)
            case_ids = parse_json_list(row["GDC_Case_IDs"], ident, "GDC_Case_IDs", audit)
            submitter_ids = parse_json_list(
                row["GDC_Case_Submitter_IDs"], ident, "GDC_Case_Submitter_IDs", audit
            )
            try:
                occurrence = int(row["GDC_Occurrence_Count"])
                if occurrence < 1:
                    audit.error("candidate_nonpositive_occurrence_count", ident)
                if occurrence != max(len(case_ids), len(submitter_ids)):
                    audit.error("candidate_occurrence_count_mismatch", ident)
            except Exception:
                audit.error("candidate_invalid_occurrence_count", ident)
            if not ssm_ids:
                audit.error("candidate_missing_gdc_ssm_id", ident)

            for feature in EPIGENETIC_FEATURES:
                try:
                    x = finite_float(row[feature])
                    feature_stats[feature].add(x)
                    flag = int(row[f"{feature}_Missing"])
                    if flag not in (0, 1):
                        audit.error("candidate_missing_indicator_not_binary", f"{ident}:{feature}")
                    if flag == 1:
                        missing_counts[feature] += 1
                        expected = float(imputation_values[feature])
                        if not math.isclose(x, expected, rel_tol=0.0, abs_tol=2e-6):
                            audit.error("candidate_flagged_missing_not_training_imputation", f"{ident}:{feature}")
                except Exception:
                    audit.error("candidate_invalid_feature_or_missing_indicator", f"{ident}:{feature}")

            try:
                p1 = finite_float(row["Target_Base_PhyloP_100way_1"])
                p2 = finite_float(row["Target_Base_PhyloP_100way_2"])
                p1rc = finite_float(row["Target_Base_PhyloP_100way_1_RC"])
                p2rc = finite_float(row["Target_Base_PhyloP_100way_2_RC"])
                m1 = int(row["Target_Base_PhyloP_100way_1_Missing"])
                m2 = int(row["Target_Base_PhyloP_100way_2_Missing"])
                m1rc = int(row["Target_Base_PhyloP_100way_1_RC_Missing"])
                m2rc = int(row["Target_Base_PhyloP_100way_2_RC_Missing"])
                if not math.isclose(p1rc, p2, abs_tol=2e-6, rel_tol=0.0):
                    audit.error("candidate_phylo_rc_position1_not_swapped", ident)
                if not math.isclose(p2rc, p1, abs_tol=2e-6, rel_tol=0.0):
                    audit.error("candidate_phylo_rc_position2_not_swapped", ident)
                if m1rc != m2 or m2rc != m1:
                    audit.error("candidate_phylo_rc_missingness_not_swapped", ident)
            except Exception:
                audit.error("candidate_invalid_rc_phylo_fields", ident)

    for feature, stats in feature_stats.items():
        result = stats.result()
        if feature.startswith("Ref_") and result["min"] is not None and result["min"] < 0:
            audit.warning("candidate_negative_reference_signal_observed", f"{path.name}:{feature}:{result['min']}")
        if feature.startswith("Target_Base_PhyloP") and result["min"] is not None:
            if result["min"] < -25 or result["max"] > 25:
                audit.warning("candidate_extreme_phyloP_value", f"{path.name}:{feature}:{result['min']}..{result['max']}")

    return ({
        "rows": rows,
        "unique_candidate_ids": len(candidate_ids),
        "split_counts": dict(split_counts),
        "chromosome_counts": dict(chromosome_counts),
        "central_target_cpg_changed_by_mutation": central_cpg_destroyed,
        "mutations_inside_100bp_sequence_shape_window": mutation_inside_100bp,
        "mutations_outside_100bp_sequence_shape_window": mutation_outside_100bp,
        "rows_containing_N": n_sequence_rows,
        "missing_counts": dict(missing_counts),
        "feature_stats": {k: v.result() for k, v in feature_stats.items()},
    }, candidate_ids, test_ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_path = args.output or (data_dir / "data_purity_audit.json")

    required_files = [
        data_dir / "train.csv",
        data_dir / "val.csv",
        data_dir / "test.csv",
        data_dir / "testing_data.csv",
        data_dir / "testing_data_test_only.csv",
        data_dir / "split_manifest.json",
        data_dir / "feature_imputation.json",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    split_manifest = load_json(data_dir / "split_manifest.json")
    imputation_payload = load_json(data_dir / "feature_imputation.json")
    imputation_values = imputation_payload["values"]

    audit = Audit()
    training_summary: dict[str, Any] = {}
    global_probe_split: dict[str, str] = {}

    for split_name in ("train", "val", "test"):
        training_summary[split_name] = audit_training_file(
            data_dir / f"{split_name}.csv",
            split_name,
            split_manifest,
            imputation_values,
            audit,
            global_probe_split,
            args.progress_every,
        )

    full_summary, full_ids, full_test_ids = audit_candidate_file(
        data_dir / "testing_data.csv",
        split_manifest,
        imputation_values,
        audit,
        args.progress_every,
    )
    test_only_summary, test_only_ids, test_only_test_ids = audit_candidate_file(
        data_dir / "testing_data_test_only.csv",
        split_manifest,
        imputation_values,
        audit,
        args.progress_every,
    )

    if test_only_ids != full_test_ids:
        for ident in sorted(full_test_ids - test_only_ids)[:MAX_EXAMPLES]:
            audit.error("test_only_file_missing_expected_candidate", ident)
        for ident in sorted(test_only_ids - full_test_ids)[:MAX_EXAMPLES]:
            audit.error("test_only_file_contains_unexpected_candidate", ident)
        missing_n = len(full_test_ids - test_only_ids)
        unexpected_n = len(test_only_ids - full_test_ids)
        if missing_n > MAX_EXAMPLES:
            audit.errors["test_only_file_missing_expected_candidate"] += missing_n - MAX_EXAMPLES
        if unexpected_n > MAX_EXAMPLES:
            audit.errors["test_only_file_contains_unexpected_candidate"] += unexpected_n - MAX_EXAMPLES

    if test_only_test_ids != test_only_ids:
        non_test = test_only_ids - test_only_test_ids
        for ident in sorted(non_test)[:MAX_EXAMPLES]:
            audit.error("test_only_file_contains_non_test_split", ident)
        if len(non_test) > MAX_EXAMPLES:
            audit.errors["test_only_file_contains_non_test_split"] += len(non_test) - MAX_EXAMPLES

    report = {
        "status": "PASS" if not audit.errors else "FAIL",
        "hard_error_count": int(sum(audit.errors.values())),
        "warning_count": int(sum(audit.warnings.values())),
        "errors": dict(audit.errors),
        "warnings": dict(audit.warnings),
        "examples": dict(audit.examples),
        "training": training_summary,
        "candidates_full": full_summary,
        "candidates_test_only": test_only_summary,
        "test_only_exactly_matches_full_test_subset": test_only_ids == full_test_ids,
        "split_manifest": split_manifest,
        "imputation": imputation_payload,
    }

    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\n==========================================")
    print(f"DATA PURITY AUDIT: {report['status']}")
    print("==========================================")
    print(f"Hard errors: {report['hard_error_count']}")
    print(f"Warnings:    {report['warning_count']}")
    print(f"Report:      {output_path}")
    if audit.errors:
        print("\nHard error categories:")
        for key, value in audit.errors.most_common():
            print(f"  {key}: {value}")
    if audit.warnings:
        print("\nWarning categories:")
        for key, value in audit.warnings.most_common():
            print(f"  {key}: {value}")

    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
