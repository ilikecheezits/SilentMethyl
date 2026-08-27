#!/usr/bin/env python3
"""
Score GENOA meQTL variant-CpG pairs with the frozen SilentMethyl checkpoints.

What this is
------------
Stage B.1. Pure inference on the nine existing checkpoints -- no gradient steps,
no new weights, nothing on disk is modified. It takes the harmonized GENOA pairs
emitted by data/build_genoa_scoring_input.py, reconstructs the WT and MUT 1,000-bp
model windows, and writes the RC-averaged MUT-minus-WT delta for every pair.

This is mentor requirement 6, independent variant evaluation, going from n=81 to
n=39,692 unique variants over 66,557 held-out pairs.

Why this is not scripts/14_known_variant_application.py
-------------------------------------------------------
Script 14 is the right engine for a handful of published variants and the wrong
one here, for three reasons that would each have surfaced as a failure:

  1. It requires Variant_ID to be unique and raises otherwise. GENOA has 66,557
     pairs over 39,692 variants -- most variants sit near more than one probe --
     so script 14 aborts on the first line of validation.
  2. It re-derives pairs, scoring every model-visible CpG within the window of
     each variant. GENOA tested specific variant-probe pairs and reports an effect
     size for each; re-deriving them breaks the join back to beta_genoa.
  3. It loads all 418k rows of train/val/test including the 5,000-bp sequence
     column. Here only the probes named in the input are needed -- 19,084 for the
     held-out stratum -- so the read is filtered on arrival.

Numerics are deliberately identical to scripts/05_matched_background.py: FP32 by
default, the same forward/RC averaging, and the same phyloP swap when building the
reverse-complement context vector.

Two conventions the analysis must respect
------------------------------------------
**Effect direction.** Compare the model delta against `beta_genoa_ref_to_alt`, not
raw `beta_genoa`. GENOA is GEMMA output whose beta is keyed to the minor allele,
while the model delta is keyed to hg38 REF->ALT. build_genoa_scoring_input.py
re-signs it; this script carries the column through and refuses to run without it.

**Effect scale.** GENOA effect sizes are on a normalized-phenotype scale, not on
the beta or M scale the model predicts. Rank and sign comparisons -- Spearman rho,
direction agreement, AUROC -- are meaningful. A Pearson correlation of raw
magnitudes, or any claim of magnitude calibration against GENOA, is not.

Tissue caveat to carry into the write-up
----------------------------------------
GENOA measured blood. The fusion model's context features are MCF-10A breast
tracks. Scoring GENOA with the fusion model is therefore cross-tissue transfer,
not tissue-matched validation. The sequence-only model is tissue-agnostic and is
the honest comparator -- which is why --models defaults to running both, and why
the fusion-minus-sequence difference is the quantity worth reporting.

Usage (run from the repository root)
------------------------------------
    # smoke test on CPU or one GPU, 200 pairs, one seed, both models
    python -u scripts/19_genoa_variant_scoring.py --limit 200 --seeds 42

    # the real held-out run
    python -u scripts/19_genoa_variant_scoring.py \
        --input-csv data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv \
        --stratum heldout --seeds 42 43 44 --models fusion sequence

    # one shard of a Slurm array
    python -u scripts/19_genoa_variant_scoring.py --shard "${SLURM_ARRAY_TASK_ID}" \
        --num-shards 6 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from training_common import (  # noqa: E402
    MISSING_FEATURES,
    PHYLOP_1,
    PHYLOP_2,
    TABULAR_FEATURES,
    FusionModel,
    SequenceOnlyModel,
    autocast_context,
    centered_crop,
    get_tokenizer,
    load_model_state,
    m_to_beta_tensor,
    reverse_complement,
    set_seed,
)
from matched_background_utils import (  # noqa: E402
    CENTER_C_INDEX,
    CENTER_G_INDEX,
    PROTECTED_CPG_INDICES,
)

LOGGER = logging.getLogger("silentmethyl.genoa")

# Index of the target CpG's C inside the 5,000-bp stored sequence. centered_crop()
# takes seq[2000:3000] for a 1,000-bp window, mapping index 2499 -> 499, which is
# CENTER_C_INDEX. Both constants are asserted at startup rather than trusted.
FULL_TARGET_C_INDEX = 2499
FULL_SEQUENCE_LENGTH = 5000
MODEL_WINDOW_SIZE = 1000

REQUIRED_INPUT_COLUMNS = [
    "Variant_ID", "chr", "Position_1based", "Ref", "Alt",
    "probeID", "probe_split", "distance_bp", "beta_genoa_ref_to_alt",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input-csv", type=Path,
        default=Path("data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv"),
    )
    p.add_argument(
        "--stratum", default="heldout", choices=("heldout", "model_visible"),
        help="Labels the output directory. The two are never pooled.",
    )
    p.add_argument("--models", nargs="+", default=["fusion", "sequence"],
                   choices=("fusion", "sequence"))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--weights-template",
                   default="checkpoints_journal/seed{seed}/{model}/best_weights.pth")
    p.add_argument("--split-template", default="data/datafiles/{split}.csv")
    p.add_argument("--hm450-manifest", type=Path,
                   default=Path("data/HM450.hg38.manifest.tsv.gz"))
    p.add_argument("--include-masked-probes", action="store_true")
    p.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    p.add_argument("--local-model-dir", default="./dnabert2_local")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=20_000,
                   help="Pairs held in memory at once. Bounds RAM, not results.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument(
        "--amp", action="store_true",
        help="CUDA mixed precision. OFF by default: a variant effect is the "
             "difference between two nearly identical predictions and must be "
             "scored in FP32, matching scripts/05_matched_background.py.",
    )
    p.add_argument("--shard", type=int, default=0,
                   help="0-based shard index for a Slurm array. Shards split pairs.")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Smoke-test only: score just the first N pairs.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/genoa_variant_scoring"))
    return p.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def coerce_boolean(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "t": True, "1": True, "yes": True,
               "false": False, "f": False, "0": False, "no": False}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        bad = series[parsed.isna()].astype(str).unique()[:5].tolist()
        raise ValueError(f"Cannot parse {name} as boolean; examples={bad}")
    return parsed.astype(bool)


def apply_probe_qc(pairs: pd.DataFrame, manifest_path: Path,
                   include_masked: bool) -> tuple[pd.DataFrame, int]:
    """Drop pairs whose target probe fails HM450 MASK_general, as script 05 does."""
    manifest = pd.read_csv(manifest_path, sep="\t", compression="infer",
                           low_memory=False)
    probe_col = next((c for c in ("probeID", "IlmnID", "Name")
                      if c in manifest.columns), None)
    if probe_col is None or "MASK_general" not in manifest.columns:
        raise ValueError(f"{manifest_path} needs a probe ID column and MASK_general")
    mapping = manifest[[probe_col, "MASK_general"]].rename(
        columns={probe_col: "probeID"}).copy()
    mapping["probeID"] = mapping["probeID"].astype(str)
    mapping["HM450_MASK_general"] = coerce_boolean(
        mapping.pop("MASK_general"), "MASK_general")
    if mapping["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probe IDs in {manifest_path}")

    out = pairs.copy()
    out["probeID"] = out["probeID"].astype(str)
    out = out.merge(mapping, on="probeID", how="left", validate="many_to_one")
    unknown = int(out["HM450_MASK_general"].isna().sum())
    if unknown:
        raise ValueError(f"{unknown} GENOA pairs name a probe absent from the "
                         f"HM450 manifest -- the harmonizer and the manifest "
                         f"disagree, which must be resolved before scoring")
    if include_masked:
        return out, 0
    before = len(out)
    out = out[~out["HM450_MASK_general"].astype(bool)].reset_index(drop=True)
    return out, before - len(out)


def load_probe_records(split_template: str, probe_ids: set[str]) -> dict:
    """probeID -> (5000-bp sequence, tabular vector, missingness vector).

    Read in chunks and filtered on arrival: only probes named by the GENOA input
    are materialised, so this costs ~100 MB for the held-out stratum instead of
    the ~6 GB it would take to hold all three splits.
    """
    needed = ["probeID", "Healthy_5000bp_DNA", *TABULAR_FEATURES, *MISSING_FEATURES]
    records: dict[str, tuple] = {}
    for split in ("train", "val", "test"):
        path = Path(split_template.format(split=split))
        if not path.is_file():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0)
        absent = [c for c in needed if c not in header.columns]
        if absent:
            raise ValueError(f"{path} is missing required columns: {absent}")
        kept = 0
        for chunk in pd.read_csv(path, usecols=needed, chunksize=20_000):
            chunk = chunk[chunk["probeID"].astype(str).isin(probe_ids)]
            if chunk.empty:
                continue
            tabular = chunk[TABULAR_FEATURES].to_numpy(dtype=np.float32)
            missing = chunk[MISSING_FEATURES].to_numpy(dtype=np.float32)
            if not np.isfinite(tabular).all():
                raise ValueError(
                    f"{path} contains non-finite context features for a GENOA "
                    f"target probe; inputs must already use train-derived imputation")
            for i, probe in enumerate(chunk["probeID"].astype(str).tolist()):
                records[probe] = (
                    str(chunk["Healthy_5000bp_DNA"].iloc[i]).upper(),
                    tabular[i],
                    missing[i],
                )
            kept += len(chunk)
        LOGGER.info("%-5s split contributed %d target probes", split, kept)
    return records


def build_chunk(pairs: pd.DataFrame, records: dict, counters: dict):
    """Turn a slice of GENOA pairs into model inputs, dropping what cannot be scored."""
    rows, wt_windows, mut_windows = [], [], []
    tabs, missings = [], []

    for row in pairs.itertuples(index=False):
        probe = str(row.probeID)
        record = records.get(probe)
        if record is None:
            counters["probe_not_in_splits"] += 1
            continue
        sequence, tabular, missing = record
        if len(sequence) != FULL_SEQUENCE_LENGTH:
            counters["bad_stored_sequence_length"] += 1
            continue
        if sequence[FULL_TARGET_C_INDEX:FULL_TARGET_C_INDEX + 2] != "CG":
            counters["stored_sequence_not_cpg_centred"] += 1
            continue

        offset = int(row.distance_bp)
        full_index = FULL_TARGET_C_INDEX + offset
        window_index = CENTER_C_INDEX + offset
        if not 0 <= window_index < MODEL_WINDOW_SIZE:
            counters["outside_model_window"] += 1
            continue
        if window_index in PROTECTED_CPG_INDICES:
            counters["alters_target_cpg"] += 1
            continue

        # Loud check, not a silent skip. If `pos` in the split CSVs and
        # `cpg_pos_hg38` in the harmonized pairs were off by one, essentially every
        # pair would land here -- which is exactly the signal we want.
        if sequence[full_index] != str(row.Ref).upper():
            counters["reference_base_mismatch"] += 1
            continue

        mutated = (sequence[:full_index] + str(row.Alt).upper()
                   + sequence[full_index + 1:])
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
        record_row["Pair_UID"] = f"{row.Variant_ID}|{probe}"
        record_row["Mutation_Window_Index"] = int(window_index)
        rows.append(record_row)
        wt_windows.append(wt_window)
        mut_windows.append(mut_window)
        tabs.append(tabular)
        missings.append(missing)
        counters["scoreable"] += 1

    if not rows:
        return None
    return (
        pd.DataFrame(rows).reset_index(drop=True),
        wt_windows,
        mut_windows,
        torch.tensor(np.stack(tabs), dtype=torch.float32),
        torch.tensor(np.stack(missings), dtype=torch.float32),
    )


def _phylop_column_indices() -> tuple[int, int]:
    """Positions of the two target-base phyloP features, resolved by NAME.

    These two are the C and the G of the target CpG and exchange under reverse
    complementation; every other context feature is strand-symmetric and must stay
    put. Resolving by name rather than by position (`[:, -2], [:, -1]`) is what
    makes TABULAR_FEATURES safe to extend -- a positional swap would silently
    exchange the wrong columns the moment a feature is appended, with no error and
    no way to notice afterwards. Mirrors scripts/05_matched_background.py.
    """
    try:
        return TABULAR_FEATURES.index(PHYLOP_1), TABULAR_FEATURES.index(PHYLOP_2)
    except ValueError as exc:
        raise RuntimeError(
            f"{PHYLOP_1} and {PHYLOP_2} must both appear in TABULAR_FEATURES; the "
            f"reverse-complement context transform is defined in terms of them."
        ) from exc


PHYLOP_COLUMN_INDICES = _phylop_column_indices()


def make_rc_context(tab: torch.Tensor, missing: torch.Tensor):
    """Exchange the two target-base phyloP features for the RC pass."""
    first, second = PHYLOP_COLUMN_INDICES
    rc_tab, rc_missing = tab.clone(), missing.clone()
    rc_tab[:, first], rc_tab[:, second] = tab[:, second].clone(), tab[:, first].clone()
    rc_missing[:, first], rc_missing[:, second] = (
        missing[:, second].clone(), missing[:, first].clone())
    return rc_tab, rc_missing


def build_model(model_type: str, args: argparse.Namespace, seed: int,
                weights_path: Path, device: torch.device):
    set_seed(seed)
    if model_type == "fusion":
        model = FusionModel(args.model_path, fusion_mode="gated",
                            tabular_dim=len(TABULAR_FEATURES),
                            local_dir=args.local_model_dir)
    else:
        model = SequenceOnlyModel(args.model_path, local_dir=args.local_model_dir)

    state = load_model_state(str(weights_path), map_location="cpu")
    missing_keys, unexpected = model.load_state_dict(state, strict=True)
    if missing_keys or unexpected:
        raise RuntimeError(f"Strict checkpoint mismatch for {model_type} seed {seed}: "
                           f"missing={missing_keys}, unexpected={unexpected}")
    if not args.amp:
        model = model.float()
        dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
        if dtypes != ["torch.float32"]:
            raise RuntimeError(f"FP32 inference requested but model holds {dtypes}")
    return model.to(device).eval()


@torch.inference_mode()
def infer(model, model_type: str, tokenizer, sequences: list[str],
          tab: torch.Tensor, missing: torch.Tensor, args: argparse.Namespace,
          device: torch.device, desc: str) -> dict:
    m_values, gate_values = [], []
    use_amp = bool(args.amp and device.type == "cuda")
    for start in tqdm(range(0, len(sequences), args.batch_size), desc=desc,
                      leave=False):
        stop = min(len(sequences), start + args.batch_size)
        encoded = tokenizer(sequences[start:stop], truncation=True,
                            max_length=len(sequences[start]),
                            padding="max_length", return_tensors="pt")
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with autocast_context(device, use_amp):
            if model_type == "fusion":
                _, m_pred, gates = model(tab[start:stop].to(device),
                                         missing[start:stop].to(device), ids, mask)
            else:
                _, m_pred = model(ids, mask)
                gates = None
        if not use_amp and m_pred.dtype != torch.float32:
            raise RuntimeError(
                f"FP32 inference requested but regression output is {m_pred.dtype}")
        m_values.append(m_pred.float().detach().cpu().numpy().reshape(-1))
        if gates is not None:
            gate_values.append(gates.float().detach().cpu().numpy())

    m_array = np.concatenate(m_values)
    out = {"m": m_array, "beta": m_to_beta_tensor(torch.tensor(m_array)).numpy()}
    if gate_values:
        out["gates"] = np.concatenate(gate_values, axis=0)
    return out


def score_chunk(model, model_type: str, tokenizer, cohort: pd.DataFrame,
                wt: list[str], mut: list[str], tab: torch.Tensor,
                missing: torch.Tensor, args: argparse.Namespace,
                device: torch.device, seed: int, weights_path: Path,
                weights_sha: str) -> pd.DataFrame:
    rc_tab, rc_missing = make_rc_context(tab, missing)
    wt_rc = [reverse_complement(s) for s in wt]
    mut_rc = [reverse_complement(s) for s in mut]

    wt_f = infer(model, model_type, tokenizer, wt, tab, missing, args, device, "WT fwd")
    wt_r = infer(model, model_type, tokenizer, wt_rc, rc_tab, rc_missing, args, device, "WT rc")
    mu_f = infer(model, model_type, tokenizer, mut, tab, missing, args, device, "MUT fwd")
    mu_r = infer(model, model_type, tokenizer, mut_rc, rc_tab, rc_missing, args, device, "MUT rc")

    wt_beta = (wt_f["beta"] + wt_r["beta"]) / 2.0
    mut_beta = (mu_f["beta"] + mu_r["beta"]) / 2.0
    wt_m = (wt_f["m"] + wt_r["m"]) / 2.0
    mut_m = (mu_f["m"] + mu_r["m"]) / 2.0
    delta_fwd = mu_f["beta"] - wt_f["beta"]
    delta_rc = mu_r["beta"] - wt_r["beta"]

    out = cohort.copy()
    out["Model"] = model_type
    out["Seed"] = int(seed)
    out["Weights_Path"] = str(weights_path)
    out["Weights_SHA256"] = weights_sha
    out["WT_M_RC_Avg"] = wt_m
    out["MUT_M_RC_Avg"] = mut_m
    out["WT_Beta_RC_Avg"] = wt_beta
    out["MUT_Beta_RC_Avg"] = mut_beta
    out["Predicted_Delta_Beta"] = mut_beta - wt_beta
    # The M-scale delta is the primary quantity. Absolute beta error and beta
    # deltas are compressed near 0 and 1, which is what results/journal/
    # rc_uncertainty_conditional_s50/ showed is enough to invert an estimator
    # ranking. Report on M and say why.
    out["Predicted_Delta_M"] = mut_m - wt_m
    out["Absolute_Delta_Beta"] = np.abs(out["Predicted_Delta_Beta"])
    out["Absolute_Delta_M"] = np.abs(out["Predicted_Delta_M"])
    out["Delta_Beta_FWD"] = delta_fwd
    out["Delta_Beta_RC"] = delta_rc
    # Free per-locus uncertainty: the forward/RC disagreement the averaging throws
    # away. Retains ~75-83% of the 3-seed ensemble's incremental signal.
    out["Delta_Beta_RC_Absolute_Difference"] = np.abs(delta_fwd - delta_rc)
    out["Delta_Beta_RC_Sign_Agree"] = (np.sign(delta_fwd) == np.sign(delta_rc)).astype(int)

    if "gates" in wt_f:
        for prefix, array in (("WT_Gate_Avg", (wt_f["gates"] + wt_r["gates"]) / 2.0),
                              ("MUT_Gate_Avg", (mu_f["gates"] + mu_r["gates"]) / 2.0)):
            out[f"{prefix}_DNA"] = array[:, 0]
            out[f"{prefix}_EPI"] = array[:, 1]
            out[f"{prefix}_DNA_Share"] = array[:, 0] / np.clip(
                array[:, 0] + array[:, 1], 1e-12, None)
    return out


def choose_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # Assert the window geometry instead of trusting the constants above.
    probe_sequence = "N" * FULL_SEQUENCE_LENGTH
    if centered_crop(probe_sequence, MODEL_WINDOW_SIZE) != "N" * MODEL_WINDOW_SIZE:
        raise RuntimeError("centered_crop did not return a 1000-bp window")
    if FULL_TARGET_C_INDEX - (FULL_SEQUENCE_LENGTH // 2 - MODEL_WINDOW_SIZE // 2) != CENTER_C_INDEX:
        raise RuntimeError(
            "FULL_TARGET_C_INDEX and CENTER_C_INDEX disagree about where the target "
            "CpG lands after cropping; scoring would silently mutate the wrong base")

    if not args.input_csv.is_file():
        raise SystemExit(f"scoring input not found: {args.input_csv}")
    pairs = pd.read_csv(args.input_csv)
    absent = [c for c in REQUIRED_INPUT_COLUMNS if c not in pairs.columns]
    if absent:
        raise SystemExit(
            f"{args.input_csv} is missing {absent}. Regenerate it with "
            f"`python -u data/build_genoa_scoring_input.py` -- in particular "
            f"beta_genoa_ref_to_alt is required, because comparing the model delta "
            f"against raw beta_genoa mixes two effect-allele conventions.")
    LOGGER.info("%d pairs read from %s", len(pairs), args.input_csv)

    expected_split = {"heldout": {"test"}, "model_visible": {"train", "val"}}[args.stratum]
    actual = set(pairs["probe_split"].astype(str).unique())
    if not actual <= expected_split:
        raise SystemExit(
            f"--stratum {args.stratum} expects probe_split in {sorted(expected_split)} "
            f"but the file contains {sorted(actual)}. The two strata are never pooled.")

    pairs, masked_dropped = apply_probe_qc(pairs, args.hm450_manifest,
                                           args.include_masked_probes)
    LOGGER.info("HM450 probe QC dropped %d pairs; %d remain", masked_dropped, len(pairs))

    if args.num_shards > 1:
        pairs = pairs.iloc[args.shard::args.num_shards].reset_index(drop=True)
        LOGGER.info("shard %d/%d -> %d pairs", args.shard, args.num_shards, len(pairs))
    if args.limit > 0:
        pairs = pairs.head(args.limit).copy()
        LOGGER.warning("SMOKE TEST: limited to %d pairs", len(pairs))

    records = load_probe_records(args.split_template,
                                 set(pairs["probeID"].astype(str)))
    LOGGER.info("materialised %d target probes", len(records))

    device = choose_device(args.device)
    LOGGER.info("device=%s precision=%s", device,
                "CUDA AMP" if args.amp and device.type == "cuda" else "FP32")
    tokenizer = get_tokenizer(args.model_path)

    counters = {k: 0 for k in (
        "scoreable", "probe_not_in_splits", "bad_stored_sequence_length",
        "stored_sequence_not_cpg_centred", "outside_model_window",
        "alters_target_cpg", "reference_base_mismatch",
        "unexpected_window_difference", "window_not_cpg_centred")}

    chunks = [pairs.iloc[i:i + args.chunk_size]
              for i in range(0, len(pairs), args.chunk_size)]
    prepared = []
    for chunk in chunks:
        built = build_chunk(chunk, records, counters)
        if built is not None:
            prepared.append(built)
    if not prepared:
        raise SystemExit("no GENOA pair survived construction; see the counters above")

    if counters["reference_base_mismatch"] > 0.05 * max(1, counters["scoreable"]):
        raise SystemExit(
            f"{counters['reference_base_mismatch']} pairs disagree with the stored "
            f"reference base. That rate indicates a coordinate mismatch between "
            f"`pos` in data/datafiles/*.csv and `cpg_pos_hg38` in the harmonized "
            f"pairs, not bad data. Resolve it before scoring.")
    LOGGER.info("construction counters: %s", counters)

    # Per-run tag. Pair scores are already separated by model/seed directory, but
    # the summary is not -- and this script is meant to be run as a Slurm array with
    # one task per model-seed. Without the tag, six concurrent tasks would each
    # overwrite the same run_summary.json and only the last one to finish would be
    # recorded.
    suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
    run_tag = "_".join(
        ["-".join(args.models), "seed" + "-".join(str(s) for s in args.seeds)]
        + ([f"shard{args.shard}of{args.num_shards}"] if args.num_shards > 1 else [])
    )
    written = []
    for model_type in args.models:
        for seed in dict.fromkeys(int(s) for s in args.seeds):
            weights = Path(args.weights_template.format(seed=seed, model=model_type))
            if not weights.is_file():
                raise SystemExit(f"checkpoint not found: {weights}")
            weights_sha = sha256_file(weights)
            LOGGER.info("scoring %s seed %d", model_type, seed)
            model = build_model(model_type, args, seed, weights, device)
            scored = pd.concat(
                [score_chunk(model, model_type, tokenizer, cohort, wt, mut, tab,
                             miss, args, device, seed, weights, weights_sha)
                 for cohort, wt, mut, tab, miss in prepared],
                ignore_index=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            target = (args.output_dir / args.stratum / model_type /
                      f"seed{seed}" / f"pair_scores{suffix}.csv")
            atomic_csv(scored, target)
            written.append({"model": model_type, "seed": seed, "rows": int(len(scored)),
                            "file": str(target), "weights_sha256": weights_sha})
            LOGGER.info("  -> %s (%d rows)", target, len(scored))

    atomic_json(
        {
            "analysis": "GENOA meQTL variant scoring on frozen SilentMethyl checkpoints",
            "purpose": "mentor requirement 6: independent variant evaluation",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stratum": args.stratum,
            "input_csv": str(args.input_csv),
            "input_sha256": sha256_file(args.input_csv),
            "pairs_after_probe_qc": int(len(pairs)),
            "masked_probe_pairs_dropped": int(masked_dropped),
            "construction_counters": counters,
            "shard": args.shard, "num_shards": args.num_shards,
            "smoke_test_limit": args.limit,
            "precision": "cuda_amp" if args.amp and device.type == "cuda" else "fp32",
            "outputs": written,
            "effect_allele_convention": (
                "Compare Predicted_Delta_M / Predicted_Delta_Beta against "
                "beta_genoa_ref_to_alt. Raw beta_genoa is keyed to the minor allele; "
                "the model delta is keyed to hg38 REF->ALT."),
            "effect_scale_caveat": (
                "GENOA effect sizes are on a normalized-phenotype scale, not the beta "
                "or M scale the model predicts. Spearman rho, direction agreement and "
                "AUROC are meaningful; magnitude calibration against GENOA is not."),
            "tissue_caveat": (
                "GENOA measured blood; the fusion model's context features are MCF-10A "
                "breast tracks. Fusion results here are cross-tissue transfer, not "
                "tissue-matched validation. The sequence-only model is tissue-agnostic "
                "and is the comparator that makes the fusion result interpretable."),
            "reporting_guidance": (
                "Report the heldout stratum as primary. Never pool the strata. Report "
                "CpG-creating/destroying variants separately from the rest."),
        },
        args.output_dir / args.stratum / f"run_summary_{run_tag}.json",
    )

    print()
    print("=" * 70)
    print(f"stratum          : {args.stratum}")
    print(f"pairs scored     : {counters['scoreable']:,}")
    for key, value in counters.items():
        if key != "scoreable" and value:
            print(f"  dropped, {key:<32} {value:>8,}")
    for item in written:
        print(f"  {item['model']:<9} seed {item['seed']}  {item['rows']:>8,} rows")
    print(f"output           : {args.output_dir / args.stratum}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
