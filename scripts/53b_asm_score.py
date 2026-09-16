#!/usr/bin/env python3
"""
Score the ASM SNV-CpG pairs with the frozen checkpoints (Task E1, stage 2).

Pure inference. No gradient steps, no new weights, nothing on disk is modified
except this script's own output directory.

What makes this different from 20_variant_scoring.py
-----------------------------------------------------
Script 20 reads its target CpGs from the prebuilt split CSVs and then hard-fails
any probeID absent from the HM450 manifest. ASM CpGs are arbitrary genomic CpGs,
so both checks would reject every row here. This script takes the CpG records
built by `53_asm_build.py` instead -- same 5,000-bp sequence convention, same
seven context tracks, same phyloP pair, same train-split imputation -- and
reuses script 20's model construction and inference by path-loading it, the same
idiom `23_context_permutation.py` uses. Script 20 itself is untouched, so every
frozen analysis that depends on it is unaffected.

Numerics are therefore identical to the GENOA/eGTEx runs: FP32 by default,
forward/reverse-complement averaging, the phyloP swap when building the RC
context vector, and the M-scale delta as the primary quantity.

Arms
----
fusion    checkpoints_ablation/breast_epithelium/seed42/fusion/best_weights.pth
sequence  checkpoints_journal/seed42/sequence/best_weights.pth

The sequence arm is tissue-agnostic and shares the journal checkpoint, which is
what `results/journal/ablation_breast_epithelium/seed42/sequence/metrics.json`
records. Seed 42 only, per the standing single-seed directive.

Usage
-----
    python -u scripts/53b_asm_score.py --limit 500        # smoke
    python -u scripts/53b_asm_score.py                    # full run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SCORER_PATH = HERE / "20_variant_scoring.py"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from training_common import (  # noqa: E402
    MISSING_FEATURES,
    TABULAR_FEATURES,
    centered_crop,
    get_tokenizer,
)

LOGGER = logging.getLogger("silentmethyl.asm")

FULL_SEQUENCE_LENGTH = 5000
FULL_TARGET_C_INDEX = 2499
MODEL_WINDOW_SIZE = 1000
CENTER_C_INDEX = 499
CENTER_G_INDEX = 500
PROTECTED_WINDOW_INDICES = frozenset({CENTER_C_INDEX, CENTER_G_INDEX})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_scorer():
    """Import scripts/20 by path; its numeric stem makes it unimportable normally."""
    if not SCORER_PATH.is_file():
        raise SystemExit(f"STOP: {SCORER_PATH} not found -- run from the repo root")
    spec = importlib.util.spec_from_file_location("variant_scoring", SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["variant_scoring"] = module
    spec.loader.exec_module(module)
    return module


def build_inputs(pairs: pd.DataFrame, records: pd.DataFrame) -> tuple:
    """WT/MUT 1,000-bp windows plus aligned context, with loud rejection counters.

    Every check here mirrors 20_variant_scoring.build_chunk. The reference-base
    check in particular is a loud drop rather than a silent skip: an off-by-one
    between the build's `cpg_pos0` and the variant coordinate would send nearly
    every row into that bucket, which is exactly the signal wanted.
    """
    index = {
        str(r.cpg_id): (
            str(r.Healthy_5000bp_DNA).upper(),
            np.asarray([getattr(r, f) for f in TABULAR_FEATURES], dtype=np.float32),
            np.asarray([getattr(r, f) for f in MISSING_FEATURES], dtype=np.float32),
        )
        for r in records.itertuples(index=False)
    }

    counters = {
        "cpg_record_missing": 0, "bad_sequence_length": 0,
        "sequence_not_cpg_centred": 0, "outside_model_window": 0,
        "alters_target_cpg": 0, "reference_base_mismatch": 0,
        "unexpected_window_difference": 0, "window_not_cpg_centred": 0,
        "scoreable": 0,
    }
    rows, wt_windows, mut_windows, tabs, missings = [], [], [], [], []

    for row in pairs.itertuples(index=False):
        record = index.get(str(row.cpg_id))
        if record is None:
            counters["cpg_record_missing"] += 1
            continue
        sequence, tabular, missing = record
        if len(sequence) != FULL_SEQUENCE_LENGTH:
            counters["bad_sequence_length"] += 1
            continue
        if sequence[FULL_TARGET_C_INDEX:FULL_TARGET_C_INDEX + 2] != "CG":
            counters["sequence_not_cpg_centred"] += 1
            continue

        offset = int(row.distance_bp)
        full_index = FULL_TARGET_C_INDEX + offset
        window_index = CENTER_C_INDEX + offset
        if not 0 <= window_index < MODEL_WINDOW_SIZE:
            counters["outside_model_window"] += 1
            continue
        if window_index in PROTECTED_WINDOW_INDICES:
            counters["alters_target_cpg"] += 1
            continue
        if sequence[full_index] != str(row.Ref).upper():
            counters["reference_base_mismatch"] += 1
            continue

        mutated = sequence[:full_index] + str(row.Alt).upper() + sequence[full_index + 1:]
        wt_window = centered_crop(sequence, MODEL_WINDOW_SIZE)
        mut_window = centered_crop(mutated, MODEL_WINDOW_SIZE)
        differences = [i for i, (a, b) in enumerate(zip(wt_window, mut_window)) if a != b]
        if len(differences) != 1 or differences[0] != window_index:
            counters["unexpected_window_difference"] += 1
            continue
        if wt_window[CENTER_C_INDEX:CENTER_G_INDEX + 1] != "CG":
            counters["window_not_cpg_centred"] += 1
            continue

        record_row = row._asdict()
        record_row["Pair_UID"] = f"{row.Variant_ID}|{row.cpg_id}"
        record_row["Mutation_Window_Index"] = int(window_index)
        rows.append(record_row)
        wt_windows.append(wt_window)
        mut_windows.append(mut_window)
        tabs.append(tabular)
        missings.append(missing)
        counters["scoreable"] += 1

    if not rows:
        raise RuntimeError(f"no scoreable pairs; counters={counters}")
    cohort = pd.DataFrame(rows).reset_index(drop=True)
    tab = torch.from_numpy(np.stack(tabs))
    missing = torch.from_numpy(np.stack(missings))
    return cohort, wt_windows, mut_windows, tab, missing, counters


def run(args: argparse.Namespace) -> int:
    scorer = load_scorer()

    pairs = pd.read_csv(args.pairs_csv)
    records = pd.read_csv(args.records_csv)
    if args.limit:
        pairs = pairs.head(args.limit).copy()
    LOGGER.info("pairs=%d  cpg records=%d", len(pairs), len(records))

    cohort, wt, mut, tab, missing, counters = build_inputs(pairs, records)
    LOGGER.info("input construction: %s", counters)
    if counters["reference_base_mismatch"] > 0.01 * len(pairs):
        raise RuntimeError(
            f"{counters['reference_base_mismatch']} reference-base mismatches out of "
            f"{len(pairs)} pairs -- coordinates disagree; fix the build before scoring")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    LOGGER.info("device: %s", device)
    tokenizer = get_tokenizer(args.model_path)

    weights = {
        "fusion": args.fusion_weights,
        "sequence": args.sequence_weights,
    }
    frames = []
    for model_type in args.models:
        path = Path(weights[model_type])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        LOGGER.info("scoring %s from %s", model_type, path)
        model = scorer.build_model(model_type, args, args.seed, path, device)
        frames.append(scorer.score_chunk(
            model, model_type, tokenizer, cohort, wt, mut, tab, missing,
            args, device, args.seed, path, digest))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "asm_pair_scores.csv"
    tmp = out_path.with_suffix(f".csv.tmp.{os.getpid()}")
    scores.to_csv(tmp, index=False)
    tmp.replace(out_path)

    summary = {
        "analysis": "ASM validation (Task E1) scoring",
        "analysis_status": "COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "seed": int(args.seed),
        "models": list(args.models),
        "single_seed_note": "Seed 42 only, per the standing directive.",
        "input_counters": counters,
        "pairs_in": int(len(pairs)),
        "pairs_scored": int(counters["scoreable"]),
        "rows_written": int(len(scores)),
        "weights": {m: {"path": str(weights[m]), "sha256": sha256_file(Path(weights[m]))}
                    for m in args.models},
        "inputs": {
            args.pairs_csv.as_posix(): sha256_file(args.pairs_csv),
            args.records_csv.as_posix(): sha256_file(args.records_csv),
        },
    }
    (args.output_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("=" * 78)
    print(f"ASM scoring complete: {counters['scoreable']:,} pairs x {len(args.models)} arms")
    print("=" * 78)
    for key, value in counters.items():
        if value:
            print(f"  {key:32} {value:>8,}")
    print(f"\nwrote {out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-csv", type=Path,
                   default=Path("data/external/asm_atlas/scoring/asm_scoring_pairs.csv"))
    p.add_argument("--records-csv", type=Path,
                   default=Path("data/external/asm_atlas/scoring/asm_cpg_records.csv"))
    p.add_argument("--models", nargs="+", default=["fusion", "sequence"],
                   choices=["fusion", "sequence"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fusion-weights",
                   default="checkpoints_ablation/breast_epithelium/seed42/fusion/best_weights.pth")
    p.add_argument("--sequence-weights",
                   default="checkpoints_journal/seed42/sequence/best_weights.pth")
    p.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    p.add_argument("--local-model-dir", default="./dnabert2_local")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/asm_validation"))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
