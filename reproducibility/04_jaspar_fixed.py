#!/usr/bin/env python3
"""
Corrected JASPAR motif-disruption scan.

Key differences from the original script:
1. Keeps each JASPAR matrix ID separately instead of overwriting matrices
   that share the same transcription-factor name.
2. Scores only motif placements whose footprint overlaps the mutated base.
3. Compares WT and mutant scores at the same motif matrix, position, and strand.
4. Records matrix ID, strand, position, threshold crossing, and gain/loss status.
5. Can run on the exact 757-row eligible cohort.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from pyjaspar import jaspardb
from tqdm import tqdm

REQUIRED_COLUMNS = {
    "pos",
    "Gene",
    "GDC_Genomic_DNA_Change",
    "Healthy_5000bp_DNA",
    "Mutated_5000bp_DNA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="results/seq_epi_stability/stability_eligible_cohort.csv",
        help="Eligible cohort CSV containing WT and mutant 5000-bp sequences.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/jaspar_motif_disruptions_fixed.csv",
    )
    parser.add_argument("--release", default="JASPAR2022")
    parser.add_argument("--collection", default="CORE")
    parser.add_argument("--window-size", type=int, default=41)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=5.0,
        help="Raw log-odds threshold. Gain/loss requires crossing this threshold.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help=(
            "Optional genomic changes to scan. Omit to scan the full input cohort."
        ),
    )
    return parser.parse_args()


def parse_mutation_id(mut_id: object) -> tuple[int | None, str | None, str | None]:
    text = str(mut_id).upper()
    match = re.search(r"(\d+)\s*([ACGT])\s*>\s*([ACGT])", text)
    if not match:
        return None, None, None
    return int(match.group(1)), match.group(2), match.group(3)


def as_score_array(values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.atleast_1d(arr)


def classify_crossing(wt_score: float, mut_score: float, threshold: float) -> str:
    wt_pass = wt_score > threshold
    mut_pass = mut_score > threshold
    if not wt_pass and mut_pass:
        return "gain"
    if wt_pass and not mut_pass:
        return "loss"
    if wt_pass and mut_pass:
        return "strengthened" if mut_score > wt_score else "weakened"
    return "subthreshold_change"


def extract_pairs(df: pd.DataFrame, window_size: int) -> list[dict]:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if window_size % 2 != 1:
        raise ValueError("--window-size must be odd so the mutation is centered.")

    flank = window_size // 2
    pairs: list[dict] = []

    for _, row in df.iterrows():
        mut_pos, ref_base, alt_base = parse_mutation_id(
            row["GDC_Genomic_DNA_Change"]
        )
        if mut_pos is None:
            continue

        wt_full = str(row["Healthy_5000bp_DNA"]).upper()
        mut_full = str(row["Mutated_5000bp_DNA"]).upper()

        # The eligible-cohort file may store either the original 5,000-bp
        # strings or the cropped 1,000-bp model inputs. Locate the mutation
        # directly from the unique WT--mutant sequence difference instead of
        # assuming a 5,000-bp coordinate system.
        if len(wt_full) != len(mut_full):
            continue

        full_diff_positions = [
            i for i, (a, b) in enumerate(zip(wt_full, mut_full)) if a != b
        ]
        if len(full_diff_positions) != 1:
            continue

        mut_idx = full_diff_positions[0]
        if wt_full[mut_idx] != ref_base or mut_full[mut_idx] != alt_base:
            continue

        start = mut_idx - flank
        end = mut_idx + flank + 1
        if start < 0 or end > len(wt_full):
            continue

        wt_seq = wt_full[start:end]
        mut_seq = mut_full[start:end]

        if len(wt_seq) != window_size or len(mut_seq) != window_size:
            continue
        if wt_seq[flank] != ref_base or mut_seq[flank] != alt_base:
            continue
        if any(base not in "ACGT" for base in wt_seq + mut_seq):
            continue

        diff_positions = [
            i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b
        ]
        if diff_positions != [flank]:
            continue

        uid = row.get("Variant_UID")
        if uid is None or pd.isna(uid):
            probe = str(row.get("probeID", ""))
            uid = f"{row['GDC_Genomic_DNA_Change']}|{probe}"

        pairs.append(
            {
                "Variant_UID": str(uid),
                "Gene": str(row["Gene"]),
                "probeID": str(row.get("probeID", "")),
                "GDC_Genomic_DNA_Change": str(
                    row["GDC_Genomic_DNA_Change"]
                ),
                "WT_Seq": wt_seq,
                "MUT_Seq": mut_seq,
            }
        )

    return pairs


def overlapping_placements(
    pssm,
    wt_seq: str,
    mut_seq: str,
    mutation_index: int,
) -> list[dict]:
    """Return same-position WT/MUT score comparisons overlapping mutation."""
    motif_len = len(pssm)
    sequence_len = len(wt_seq)
    placements: list[dict] = []

    wt_fwd = as_score_array(pssm.calculate(Seq(wt_seq)))
    mut_fwd = as_score_array(pssm.calculate(Seq(mut_seq)))

    for start, (wt_score, mut_score) in enumerate(zip(wt_fwd, mut_fwd)):
        if start <= mutation_index < start + motif_len:
            if np.isfinite(wt_score) and np.isfinite(mut_score):
                placements.append(
                    {
                        "strand": "+",
                        "start_0based": start,
                        "end_0based_exclusive": start + motif_len,
                        "WT_Motif_Score": float(wt_score),
                        "MUT_Motif_Score": float(mut_score),
                    }
                )

    wt_rc = str(Seq(wt_seq).reverse_complement())
    mut_rc = str(Seq(mut_seq).reverse_complement())
    wt_rev = as_score_array(pssm.calculate(Seq(wt_rc)))
    mut_rev = as_score_array(pssm.calculate(Seq(mut_rc)))

    for rc_start, (wt_score, mut_score) in enumerate(zip(wt_rev, mut_rev)):
        original_start = sequence_len - (rc_start + motif_len)
        original_end = sequence_len - rc_start

        if original_start <= mutation_index < original_end:
            if np.isfinite(wt_score) and np.isfinite(mut_score):
                placements.append(
                    {
                        "strand": "-",
                        "start_0based": original_start,
                        "end_0based_exclusive": original_end,
                        "WT_Motif_Score": float(wt_score),
                        "MUT_Motif_Score": float(mut_score),
                    }
                )

    return placements


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.targets:
        requested = set(args.targets)
        df = df[
            df["GDC_Genomic_DNA_Change"].astype(str).isin(requested)
        ].copy()

    pairs = extract_pairs(df, args.window_size)
    logging.info("Prepared %d valid sequence pairs.", len(pairs))

    if not pairs:
        raise RuntimeError("No valid sequence pairs were extracted.")

    logging.info(
        "Loading %s %s vertebrate motifs...",
        args.release,
        args.collection,
    )
    jdb = jaspardb(release=args.release)
    all_motifs = jdb.fetch_motifs(collection=args.collection)
    motifs = [
        motif
        for motif in all_motifs
        if getattr(motif, "tax_group", None)
        and "vertebrate" in str(motif.tax_group).lower()
    ]
    if not motifs:
        raise RuntimeError("No vertebrate motifs were returned.")

    matrix_records = []
    for motif in motifs:
        pwm = motif.counts.normalize(pseudocounts=0.5)
        pssm = pwm.log_odds()
        matrix_records.append(
            {
                "matrix_id": str(getattr(motif, "matrix_id", "")),
                "motif_name": str(getattr(motif, "name", "")),
                "pssm": pssm,
            }
        )

    logging.info("Loaded %d distinct motif matrices.", len(matrix_records))

    mutation_index = args.window_size // 2
    results: list[dict] = []

    for pair in tqdm(pairs, desc="Scanning variants"):
        best = None

        for matrix in matrix_records:
            placements = overlapping_placements(
                matrix["pssm"],
                pair["WT_Seq"],
                pair["MUT_Seq"],
                mutation_index,
            )

            for placement in placements:
                wt_score = placement["WT_Motif_Score"]
                mut_score = placement["MUT_Motif_Score"]

                if max(wt_score, mut_score) <= args.score_threshold:
                    continue

                delta = mut_score - wt_score
                candidate = {
                    **pair,
                    "JASPAR_Release": args.release,
                    "Collection": args.collection,
                    "Matrix_ID": matrix["matrix_id"],
                    "TF_Name": matrix["motif_name"],
                    "Strand": placement["strand"],
                    "Hit_Start_0based": placement["start_0based"],
                    "Hit_End_0based_Exclusive": placement[
                        "end_0based_exclusive"
                    ],
                    "WT_Motif_Score": wt_score,
                    "MUT_Motif_Score": mut_score,
                    "Motif_Delta_Score": delta,
                    "Absolute_Disruption": abs(delta),
                    "Score_Threshold": args.score_threshold,
                    "Threshold_Classification": classify_crossing(
                        wt_score,
                        mut_score,
                        args.score_threshold,
                    ),
                }

                if (
                    best is None
                    or candidate["Absolute_Disruption"]
                    > best["Absolute_Disruption"]
                ):
                    best = candidate

        if best is not None:
            results.append(best)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        numeric_cols = [
            "WT_Motif_Score",
            "MUT_Motif_Score",
            "Motif_Delta_Score",
            "Absolute_Disruption",
        ]
        result_df[numeric_cols] = result_df[numeric_cols].round(4)
        result_df = result_df.sort_values(
            "Absolute_Disruption",
            ascending=False,
        )

    result_df.to_csv(output_path, index=False)
    logging.info("Wrote %d rows to %s", len(result_df), output_path)


if __name__ == "__main__":
    main()
