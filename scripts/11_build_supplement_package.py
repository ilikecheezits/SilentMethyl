#!/usr/bin/env python3
"""Build the SilentMethyl supplementary data package from frozen result tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SEEDS = (42, 43, 44)
MODELS = ("epi", "sequence", "fusion")


@dataclass(frozen=True)
class Item:
    supplement_id: str
    source: str
    destination: str
    description: str
    required: bool = True


STATIC_ITEMS = (
    Item("S1", "results/journal/candidates/candidate_matched_background_statistics.csv",
         "Supplementary_Data_S1_Candidates/S1_complete_candidate_ranking.csv",
         "Complete 440-candidate fusion-ensemble ranking with matched-background and orientation diagnostics."),
    Item("S1", "results/journal/candidates/candidate_seed_scores_long.csv",
         "Supplementary_Data_S1_Candidates/S1_candidate_scores_by_seed.csv",
         "Per-seed forward, reverse-complement, and averaged fusion candidate scores."),
    Item("S1", "results/journal/candidates/eligible_heldout_model_visible_cohort.csv",
         "Supplementary_Data_S1_Candidates/S1_eligible_candidate_cohort.csv",
         "Probe-QC-passing, chromosome-held-out, model-visible candidate cohort."),
    Item("S1", "results/journal/candidates/stability/candidate_run_consensus.csv",
         "Supplementary_Data_S1_Candidates/S1_candidate_cross_seed_consensus.csv",
         "Cross-seed consensus ranking and sign consistency."),
    Item("S1", "results/journal/candidates/stability/pairwise_run_stability.csv",
         "Supplementary_Data_S1_Candidates/S1_pairwise_seed_stability.csv",
         "Pairwise cross-seed candidate stability."),
    Item("S1", "results/journal/candidates/stability/per_run_orientation_stability.csv",
         "Supplementary_Data_S1_Candidates/S1_orientation_stability_by_seed.csv",
         "Forward/reverse-complement stability for each seed."),
    Item("S1", "results/journal/candidates/model_comparison/candidate_sequence_vs_fusion.csv",
         "Supplementary_Data_S1_Candidates/S1_sequence_vs_fusion_candidates.csv",
         "Candidate-level sequence-only versus fusion comparison."),
    Item("S1", "results/journal/candidates/model_comparison/pairwise_seed_stability_by_model.csv",
         "Supplementary_Data_S1_Candidates/S1_seed_stability_by_model.csv",
         "Fusion and sequence-only seed-stability comparison."),
    Item("S1", "results/journal/candidates/top_candidate_matched_comparators_long.csv",
         "Supplementary_Data_S1_Candidates/S1_top_candidate_comparators.csv",
         "Comparator records for the rank-1 matched-background figure."),
    Item("S1", "results/journal/candidates/candidate_analysis_summary.json",
         "Supplementary_Data_S1_Candidates/S1_candidate_analysis_summary.json",
         "Candidate-analysis configuration and checkpoint hashes."),
    Item("S1", "results/journal/candidates/top_candidate_case_study.csv",
         "Supplementary_Data_S1_Candidates/S1_top_candidate_case_study.csv",
         "Data underlying the rank-1 candidate case-study figure."),

    Item("S2", "results/journal/egtex_mqtl_positive_control/validated_heldout_cohort.csv",
         "Supplementary_Data_S2_mQTL_Positive_Control/S2_all_81_associations.csv",
         "Complete held-out 81-association eGTEx breast mQTL cohort with probe geometry."),
    Item("S2", "results/journal/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv",
         "Supplementary_Data_S2_mQTL_Positive_Control/S2_cross_seed_predictions.csv",
         "Cross-seed mQTL prediction aggregates for both models."),
    Item("S2", "results/journal/egtex_mqtl_positive_control/mqtl_positive_control_metrics.csv",
         "Supplementary_Data_S2_mQTL_Positive_Control/S2_primary_metrics.csv",
         "Cluster-aware signed-rank, direction, and AUROC results."),
    Item("S2", "results/journal/egtex_mqtl_positive_control/mqtl_probe_overlap_sensitivity_metrics.csv",
         "Supplementary_Data_S2_mQTL_Positive_Control/S2_probe_overlap_sensitivity_metrics.csv",
         "Post hoc conservative probe-footprint sensitivity statistics."),
    Item("S2", "results/journal/egtex_mqtl_positive_control/mqtl_leave_one_variant_out.csv",
         "Supplementary_Data_S2_mQTL_Positive_Control/S2_leave_one_variant_out.csv",
         "Leave-one-variant-out robustness results."),
    Item("S2", "results/journal/egtex_mqtl_positive_control/hm450_probe_overlap_audit.json",
         "Supplementary_Data_S2_mQTL_Positive_Control/S2_probe_overlap_audit.json",
         "HM450 probe-footprint coordinate audit."),
    Item("S2", "results/journal/egtex_mqtl_positive_control/run_summary.json",
         "Supplementary_Data_S2_mQTL_Positive_Control/S2_run_summary.json",
         "Positive-control inputs, checkpoint hashes, parameters, and nested results."),

    Item("S3", "results/journal/egtex_mqtl_matched_negative/matched_lead_cohort.csv",
         "Supplementary_Data_S3_mQTL_Matched_Negative/S3_matched_35_pair_cohort.csv",
         "Final 35 significant/nonsignificant matched pairs (70 rows)."),
    Item("S3", "results/journal/egtex_mqtl_matched_negative/match_sets.csv",
         "Supplementary_Data_S3_mQTL_Matched_Negative/S3_match_assignments.csv",
         "Pair assignments, matching tiers, and costs."),
    Item("S3", "results/journal/egtex_mqtl_matched_negative/matching_balance.csv",
         "Supplementary_Data_S3_mQTL_Matched_Negative/S3_matching_balance.csv",
         "Balance diagnostics for matched variables."),
    Item("S3", "results/journal/egtex_mqtl_matched_negative/matched_lead_predictions_seed_aggregate.csv",
         "Supplementary_Data_S3_mQTL_Matched_Negative/S3_cross_seed_predictions.csv",
         "Cross-seed matched-lead prediction aggregates."),
    Item("S3", "results/journal/egtex_mqtl_matched_negative/matched_negative_metrics.csv",
         "Supplementary_Data_S3_mQTL_Matched_Negative/S3_discrimination_metrics.csv",
         "AUROC, average precision, confidence intervals, and within-set permutation results."),
    Item("S3", "results/journal/egtex_mqtl_matched_negative/run_summary.json",
         "Supplementary_Data_S3_mQTL_Matched_Negative/S3_run_summary.json",
         "Matching specification, eligibility audit, checkpoint hashes, and metrics."),

    Item("S4", "results/journal/paired_model_bootstrap/model_metrics_recomputed.csv",
         "Supplementary_Data_S4_Model_Performance/S4_model_metrics_recomputed.csv",
         "Per-seed and ensemble held-out metrics recomputed from aligned predictions."),
    Item("S4", "results/journal/paired_model_bootstrap/paired_model_difference_bootstrap.csv",
         "Supplementary_Data_S4_Model_Performance/S4_paired_genomic_block_bootstrap.csv",
         "Paired 1-Mb genomic-block bootstrap differences and intervals."),
    Item("S4", "results/journal/paired_model_bootstrap/run_summary.json",
         "Supplementary_Data_S4_Model_Performance/S4_bootstrap_run_summary.json",
         "Bootstrap parameters and comparison manifest."),
    Item("S4", "results/journal/biological_context/locus_metrics_by_context.csv",
         "Supplementary_Data_S4_Model_Performance/S4_locus_metrics_by_context.csv",
         "Held-out performance stratified by genomic region, CpG-island context, ATAC, and H3K27ac."),
    Item("S4", "results/journal/biological_context/fusion_gain_by_context.csv",
         "Supplementary_Data_S4_Model_Performance/S4_fusion_gain_by_context.csv",
         "Fusion-versus-sequence Beta-MAE gain by genomic and epigenomic context."),
    Item("S4", "results/journal/biological_context/variant_response_by_distance.csv",
         "Supplementary_Data_S4_Model_Performance/S4_variant_response_by_distance.csv",
         "Candidate response and mQTL agreement summarized by variant-to-CpG distance."),
    Item("S4", "results/journal/biological_context/variant_response_by_context.csv",
         "Supplementary_Data_S4_Model_Performance/S4_variant_response_by_context.csv",
         "Candidate response summarized by variant and target-CpG context, with mQTL agreement by target-CpG context."),
    Item("S4", "results/journal/manuscript_figures/run_summary.json",
         "Supplementary_Data_S4_Model_Performance/S4_manuscript_figure_run_summary.json",
         "Input hashes and numerical summaries for the generated manuscript figures."),

    Item("S5", "results/journal/target_qc/hm450_manifest_audit.json",
         "Supplementary_Data_S5_Target_QC/S5_hm450_manifest_audit.json",
         "Manifest identity and split-level MASK_general audit."),
    Item("S5", "results/journal/target_qc/coverage_threshold_metrics.csv",
         "Supplementary_Data_S5_Target_QC/S5_coverage_threshold_metrics.csv",
         "Held-out metrics across minimum normal-sample coverage thresholds."),
    Item("S5", "results/journal/target_qc/coverage_bin_metrics.csv",
         "Supplementary_Data_S5_Target_QC/S5_coverage_bin_metrics.csv",
         "Held-out metrics in nonoverlapping normal-sample coverage bins."),
    Item("S5", "results/journal/target_qc/coverage_error_correlations.csv",
         "Supplementary_Data_S5_Target_QC/S5_coverage_error_correlations.csv",
         "Coverage versus absolute prediction-error correlations."),

    Item("S6", "reproducibility/data_purity_audit.json",
         "Supplementary_Data_S6_Reproducibility/S6_data_purity_audit.json",
         "Processed-data leakage, sequence, split, and feature audit."),
    Item("S6", "reproducibility/tcga_methylation_input_audit.json",
         "Supplementary_Data_S6_Reproducibility/S6_tcga_matrix_input_audit.json",
         "Archived TCGA methylation-matrix identity and selected columns."),
    Item("S6", "reproducibility/tcga_participant_matrix_audit.json",
         "Supplementary_Data_S6_Reproducibility/S6_tcga_participant_audit.json",
         "Participant uniqueness, sample selection, and matrix provenance audit."),
    Item("S6", "reproducibility/tcga_selected_normal_samples.csv",
         "Supplementary_Data_S6_Reproducibility/S6_tcga_selected_normal_samples.csv",
         "The 97 selected sample-type-11 columns and parsed participant identifiers."),
    Item("S6", "reproducibility/gdc_query_audit_sample.json",
         "Supplementary_Data_S6_Reproducibility/S6_gdc_query_audit_sample.json",
         "Frozen GDC candidate-query audit record."),
    Item("S6", "reproducibility/environment_snapshot.txt",
         "Supplementary_Data_S6_Reproducibility/S6_environment_snapshot.txt",
         "Software and compute-environment snapshot."),
    Item("S6", "reproducibility/analysis_audit.txt",
         "Supplementary_Data_S6_Reproducibility/S6_active_code_sha256.txt",
         "SHA-256 manifest for active analysis code."),
    Item("S6", "reproducibility/processed_data_sha256.txt",
         "Supplementary_Data_S6_Reproducibility/S6_processed_data_sha256.txt",
         "SHA-256 manifest for processed modeling inputs."),
    Item("S6", "reproducibility/reference_audit.txt",
         "Supplementary_Data_S6_Reproducibility/S6_reference_sha256.txt",
         "SHA-256 manifest for reference resources."),
    Item("S6", "reproducibility/external_input_sha256.txt",
         "Supplementary_Data_S6_Reproducibility/S6_external_input_sha256.txt",
         "SHA-256 manifest for public and frozen external inputs.", required=False),
    Item("S6", "data/datafiles/split_manifest.json",
         "Supplementary_Data_S6_Reproducibility/S6_split_manifest.json",
         "Chromosome split definition and counts."),
    Item("S6", "data/datafiles/feature_imputation.json",
         "Supplementary_Data_S6_Reproducibility/S6_feature_imputation.json",
         "Training-derived feature-imputation values."),
    Item("S6", "data/datafiles/training_data_manifest.json",
         "Supplementary_Data_S6_Reproducibility/S6_training_data_manifest.json",
         "Training-target construction manifest."),
    Item("S6", "data/datafiles/candidate_cohort_manifest.json",
         "Supplementary_Data_S6_Reproducibility/S6_candidate_cohort_manifest.json",
         "Somatic candidate-cohort construction manifest."),
    Item("S6", "instructions.md",
         "Supplementary_Data_S6_Reproducibility/S6_reproduction_instructions.md",
         "End-to-end command guide."),
    Item("S6", "requirements.txt",
         "Supplementary_Data_S6_Reproducibility/S6_python_requirements.txt",
         "Recorded Python dependencies."),

    Item("SF", "results/journal/biological_context/plots/information_source_gains.png",
         "Supplementary_Figures/SF1_information_source_gains.png",
         "Residual gains from adding sequence or context to the other modality.", False),
    Item("SF", "results/journal/biological_context/plots/fusion_gain_by_genomic_region.png",
         "Supplementary_Figures/SF2_fusion_gain_by_genomic_region.png",
         "Fusion improvement by GENCODE genomic region.", False),
)

MANUSCRIPT_FIGURE_NAMES = {
    "candidate_response_by_context.png",
    "fusion_gain_by_epigenomic_context.png",
    "model_incremental_performance.png",
    "mqtl_magnitude_rank.png",
    "mqtl_signed_rank.png",
    "top_candidate_matched_background.png",
}
ALLOWED_SUPPLEMENTARY_FIGURES = {
    "Supplementary_Figures/SF1_information_source_gains.png",
    "Supplementary_Figures/SF2_fusion_gain_by_genomic_region.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("supplementary_package"))
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing package after moving it to a timestamped backup.",
    )
    parser.add_argument(
        "--allow-missing-required",
        action="store_true",
        help="Create an explicitly incomplete development package instead of failing.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def false_like(value: str) -> bool:
    return str(value).strip().lower() in {"false", "0", "0.0", "no", "n"}


def derive_probe_filtered_tables(project: Path, package: Path) -> list[tuple[str, Path, str]]:
    cohort_path = project / "results/journal/egtex_mqtl_positive_control/validated_heldout_cohort.csv"
    aggregate_path = project / "results/journal/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv"
    if not cohort_path.is_file() or not aggregate_path.is_file():
        return []

    cohort = csv_rows(cohort_path)
    flag = "variant_overlaps_probe_or_extension_conservative"
    if not cohort or flag not in cohort[0]:
        raise ValueError(f"Positive-control cohort lacks required column: {flag}")
    retained = [row for row in cohort if false_like(row[flag])]
    if len(cohort) != 81 or len(retained) != 53:
        raise ValueError(
            f"Expected 81 total and 53 conservatively filtered associations; "
            f"observed {len(cohort)} and {len(retained)}"
        )
    cohort_out = package / "Supplementary_Data_S2_mQTL_Positive_Control/S2_probe_filtered_53_associations.csv"
    write_csv(cohort_out, retained, list(cohort[0]))

    aggregate = csv_rows(aggregate_path)
    if not aggregate:
        raise ValueError(f"Empty aggregate prediction table: {aggregate_path}")
    if flag in aggregate[0]:
        aggregate_retained = [row for row in aggregate if false_like(row[flag])]
    else:
        keys = {(row["variant_id"], row["cpg_id"]) for row in retained}
        aggregate_retained = [
            row for row in aggregate if (row.get("variant_id"), row.get("cpg_id")) in keys
        ]
    models = {row.get("model", "") for row in aggregate_retained}
    expected = 53 * len(models)
    if len(aggregate_retained) != expected:
        raise ValueError(
            f"Expected 53 filtered rows per model ({expected} total); "
            f"observed {len(aggregate_retained)}"
        )
    aggregate_out = package / "Supplementary_Data_S2_mQTL_Positive_Control/S2_probe_filtered_cross_seed_predictions.csv"
    write_csv(aggregate_out, aggregate_retained, list(aggregate[0]))
    return [
        ("S2", cohort_out, "Explicit conservative 53-association probe-filtered cohort."),
        ("S2", aggregate_out, "Cross-seed predictions for the conservative 53-association subset."),
    ]


def dynamic_items() -> list[Item]:
    items: list[Item] = []
    for seed in SEEDS:
        for model in MODELS:
            items.append(
                Item(
                    "S4",
                    f"results/journal/seed{seed}/{model}/metrics.json",
                    f"Supplementary_Data_S4_Model_Performance/per_seed/seed{seed}_{model}_metrics.json",
                    f"Held-out metrics for {model}, seed {seed}.",
                )
            )
    return items


def validate_primary_tables(project: Path) -> None:
    checks = {
        "results/journal/candidates/candidate_matched_background_statistics.csv": 440,
        "results/journal/egtex_mqtl_positive_control/validated_heldout_cohort.csv": 81,
        "results/journal/egtex_mqtl_matched_negative/matched_lead_cohort.csv": 70,
        "results/journal/biological_context/heldout_cpg_context_assignments.csv": 26_570,
        "results/journal/candidates/top_candidate_case_study.csv": 1,
    }
    for relative, expected in checks.items():
        path = project / relative
        if not path.is_file():
            continue
        observed = len(csv_rows(path))
        if observed != expected:
            raise ValueError(f"{relative}: expected {expected} rows, observed {observed}")

    participant_path = project / "reproducibility/tcga_participant_matrix_audit.json"
    if participant_path.is_file():
        audit = json.loads(participant_path.read_text())
        if audit.get("status") != "PASS":
            raise ValueError("TCGA participant audit does not have status PASS")


def readme_text(missing: list[Item]) -> str:
    completeness = "COMPLETE" if not missing else "INCOMPLETE DEVELOPMENT BUILD"
    return f"""# SilentMethyl supplementary data package

Package status: **{completeness}**

This package contains the complete tabular evidence and provenance records
needed to inspect the SilentMethyl manuscript. Genomic coordinates are hg38
unless a file explicitly states otherwise.

## Contents

- **Supplementary Data S1 — Candidates:** the 440-candidate ranking, per-seed
  scores, stability tables, sequence-versus-fusion comparison, and the rank-1
  case-study source table.
- **Supplementary Data S2 — mQTL positive control:** the 81-association cohort,
  cross-seed aggregates, clustered metrics, probe-footprint sensitivity, and
  leave-one-variant-out results.
- **Supplementary Data S3 — mQTL matched negative:** the final 35 matched pairs,
  matching balance, cross-seed aggregates, and discrimination statistics.
- **Supplementary Data S4 — model performance:** per-seed held-out metrics,
  paired genomic-block bootstrap differences, and biological-context source
  tables for the manuscript and supplementary figures.
- **Supplementary Data S5 — target QC:** HM450 mask audit and coverage
  sensitivity summaries.
- **Supplementary Data S6 — reproducibility:** construction manifests, audits,
  environment information, and SHA-256 manifests.
- **Supplementary Figures:** only the two manuscript-aligned panels that are
  not main-text figures (`information_source_gains` and
  `fusion_gain_by_genomic_region`). Exploratory plot directories and known-variant
  application figures are excluded.

## Interpretation boundaries

- eGTEx slopes are aligned to increasing VCF ALT dosage. The positive control
  supports signed allelic direction and rank, not direct slope-scale calibration.
- The conservative 53-association subset excludes variants within the annotated
  HM450 probe footprint expanded by one base on either side.
- Matched-background candidate quantities are descriptive comparisons among
  observed synonymous candidates, not calibrated null p-values.
- Nonsignificant mQTL leads are matched comparators, not confirmed causal nulls.
- Candidate outputs are variant-hypothesis rankings and are not experimental or
  causal effect estimates.

## Rebuilding

From the project root, after scripts 02--10 and 12--14 have completed:

```bash
python -u scripts/11_build_supplement_package.py
```

Use `--replace` to rebuild while preserving the prior package in a timestamped
backup. `supplement_manifest.csv` maps every packaged file to its source and
description. `SHA256SUMS.txt` verifies packaged bytes.

Raw/controlled-access data, large reference tracks, model checkpoints,
TensorBoard events, smoke-test outputs, and transient logs are intentionally
excluded. They belong in their governed source repositories or code release,
not in a journal data supplement.
"""


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = project / output

    validate_primary_tables(project)
    items = list(STATIC_ITEMS) + dynamic_items()
    figure_destinations = [item.destination for item in items if item.supplement_id == "SF"]
    unexpected = [path for path in figure_destinations if path not in ALLOWED_SUPPLEMENTARY_FIGURES]
    if unexpected:
        raise ValueError(f"Unexpected supplementary figures: {unexpected}")
    overlap = [
        item.destination
        for item in items
        if Path(item.destination).name in MANUSCRIPT_FIGURE_NAMES
        or Path(item.source).name in MANUSCRIPT_FIGURE_NAMES
    ]
    if overlap:
        raise ValueError(f"Manuscript figures must not be packaged as supplementary files: {overlap}")
    missing = [item for item in items if item.required and not (project / item.source).is_file()]
    if missing and not args.allow_missing_required:
        print("Missing required supplement sources:", file=sys.stderr)
        for item in missing:
            print(f"  {item.source}", file=sys.stderr)
        raise SystemExit(2)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    manifest_rows: list[dict[str, str | int | bool]] = []
    for item in items:
        source = project / item.source
        if not source.is_file():
            if item.required:
                manifest_rows.append({
                    "supplement_id": item.supplement_id,
                    "packaged_path": item.destination,
                    "source_path": item.source,
                    "description": item.description,
                    "required": True,
                    "status": "missing",
                    "bytes": "",
                    "sha256": "",
                })
            continue
        destination = temporary / item.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest_rows.append({
            "supplement_id": item.supplement_id,
            "packaged_path": item.destination,
            "source_path": item.source,
            "description": item.description,
            "required": item.required,
            "status": "included",
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })

    for supplement_id, derived_path, description in derive_probe_filtered_tables(project, temporary):
        relative = derived_path.relative_to(temporary).as_posix()
        manifest_rows.append({
            "supplement_id": supplement_id,
            "packaged_path": relative,
            "source_path": "derived from validated_heldout_cohort.csv and mqtl_predictions_seed_aggregate.csv",
            "description": description,
            "required": True,
            "status": "included",
            "bytes": derived_path.stat().st_size,
            "sha256": sha256_file(derived_path),
        })

    manifest_path = temporary / "supplement_manifest.csv"
    fields = ["supplement_id", "packaged_path", "source_path", "description", "required", "status", "bytes", "sha256"]
    write_csv(manifest_path, manifest_rows, fields)
    (temporary / "README.md").write_text(readme_text(missing), encoding="utf-8")

    checksum_paths = sorted(
        path for path in temporary.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}"
        for path in checksum_paths
    ]
    (temporary / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    if output.exists():
        if not args.replace:
            shutil.rmtree(temporary)
            raise SystemExit(f"Output already exists: {output}. Re-run with --replace.")
        backup = output.with_name(f"{output.name}.backup.{stamp}")
        output.rename(backup)
        print(f"Previous package moved to: {backup}")
    temporary.rename(output)

    included = sum(row["status"] == "included" for row in manifest_rows)
    included_bytes = sum(int(row["bytes"]) for row in manifest_rows if row["status"] == "included")
    print("SilentMethyl supplementary package")
    print(f"  status: {'COMPLETE' if not missing else 'INCOMPLETE'}")
    print(f"  included files: {included}")
    print(f"  included bytes: {included_bytes}")
    print(f"  package: {output}")
    print(f"  manifest: {output / 'supplement_manifest.csv'}")
    print(f"  checksums: {output / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    main()
