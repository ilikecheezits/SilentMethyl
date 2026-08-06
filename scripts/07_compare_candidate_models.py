#!/usr/bin/env python3
"""Compare three-seed sequence-only and gated-fusion candidate effects.

This is a standalone follow-up analysis. It does not modify the existing
05_matched_background.py outputs. It reuses that script's cohort/QC helpers,
scores the same eligible candidates with sequence-only checkpoints, applies
the same descriptive matched-background procedure, and compares the resulting
sequence ensemble with the existing fusion ensemble.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "scripts").is_dir() else Path.cwd()
PROJECT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(PROJECT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPTS))

from training_common import (  # noqa: E402
    SequenceOnlyModel,
    autocast_context,
    get_tokenizer,
    load_model_state,
    m_to_beta_tensor,
    reverse_complement,
    set_seed,
)
from matched_background_utils import compute_matched_background_statistics  # noqa: E402


LOGGER = logging.getLogger("silentmethyl.candidate_model_comparison")


def load_candidate_module():
    path = PROJECT_SCRIPTS / "05_matched_background.py"
    if not path.is_file():
        raise FileNotFoundError(f"Required existing analysis script not found: {path}")
    spec = importlib.util.spec_from_file_location("candidate_background_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_candidate_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score held-out candidates with sequence-only models and compare with fusion"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--sequence-weights-template",
        default="checkpoints_journal/seed{seed}/sequence/best_weights.pth",
    )
    parser.add_argument(
        "--fusion-aggregate",
        default="results/journal/candidates/candidate_matched_background_statistics.csv",
    )
    parser.add_argument(
        "--fusion-seed-scores",
        default="results/journal/candidates/candidate_seed_scores_long.csv",
    )
    parser.add_argument(
        "--input-csv", default="data/datafiles/testing_data_test_only.csv"
    )
    parser.add_argument(
        "--hm450-manifest", default="data/HM450.hg38.manifest.tsv.gz"
    )
    parser.add_argument(
        "--output-dir", default="results/journal/candidates/model_comparison"
    )
    parser.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local-model-dir", default="./dnabert2_local")
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-comparators", type=int, default=20)
    parser.add_argument("--max-comparators", type=int, default=1000)
    parser.add_argument("--background-random-seed", type=int, default=42)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision. FP32 is the default for small counterfactual differences.",
    )
    return parser.parse_args()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


@torch.inference_mode()
def infer_sequence_model(
    model,
    tokenizer,
    sequences: list[str],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, np.ndarray]:
    m_values: list[np.ndarray] = []
    beta_values: list[np.ndarray] = []
    for start in tqdm(range(0, len(sequences), batch_size), desc="Sequence inference", leave=False):
        stop = min(len(sequences), start + batch_size)
        encoded = tokenizer(
            sequences[start:stop],
            truncation=True,
            max_length=len(sequences[start]),
            padding="max_length",
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with autocast_context(device, bool(use_amp and device.type == "cuda")):
            _, m_pred = model(ids, mask)
        if not use_amp and m_pred.dtype != torch.float32:
            raise RuntimeError(f"Expected FP32 regression output; observed {m_pred.dtype}")
        m_pred = m_pred.float()
        beta = m_to_beta_tensor(m_pred)
        m_values.append(m_pred.cpu().numpy().reshape(-1))
        beta_values.append(beta.cpu().float().numpy().reshape(-1))
    return {"m": np.concatenate(m_values), "beta": np.concatenate(beta_values)}


def score_sequence_seed(
    seed: int,
    weights_path: Path,
    cohort: pd.DataFrame,
    wt_sequences: list[str],
    mut_sequences: list[str],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    set_seed(seed)
    model = SequenceOnlyModel(args.model_path, local_dir=args.local_model_dir)
    state = load_model_state(str(weights_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Sequence checkpoint mismatch for seed {seed}: missing={missing}, unexpected={unexpected}"
        )
    if not args.amp:
        model = model.float()
    model = model.to(device).eval()

    wt_rc = [reverse_complement(x) for x in wt_sequences]
    mut_rc = [reverse_complement(x) for x in mut_sequences]
    LOGGER.info("Sequence seed %d: WT forward", seed)
    wt_fwd = infer_sequence_model(model, tokenizer, wt_sequences, args.batch_size, device, args.amp)
    LOGGER.info("Sequence seed %d: WT reverse complement", seed)
    wt_rc_out = infer_sequence_model(model, tokenizer, wt_rc, args.batch_size, device, args.amp)
    LOGGER.info("Sequence seed %d: MUT forward", seed)
    mut_fwd = infer_sequence_model(model, tokenizer, mut_sequences, args.batch_size, device, args.amp)
    LOGGER.info("Sequence seed %d: MUT reverse complement", seed)
    mut_rc_out = infer_sequence_model(model, tokenizer, mut_rc, args.batch_size, device, args.amp)

    wt_beta_avg = (wt_fwd["beta"] + wt_rc_out["beta"]) / 2.0
    mut_beta_avg = (mut_fwd["beta"] + mut_rc_out["beta"]) / 2.0
    delta_fwd = mut_fwd["beta"] - wt_fwd["beta"]
    delta_rc = mut_rc_out["beta"] - wt_rc_out["beta"]
    delta_avg = mut_beta_avg - wt_beta_avg

    out = cohort.copy()
    out["Model"] = "sequence"
    out["Seed"] = int(seed)
    out["Weights_Path"] = str(weights_path)
    out["Weights_SHA256"] = BASE.sha256_file(weights_path)
    out["WT_M_FWD"] = wt_fwd["m"]
    out["WT_M_RC"] = wt_rc_out["m"]
    out["WT_M_RC_Avg"] = (wt_fwd["m"] + wt_rc_out["m"]) / 2.0
    out["WT_Beta_FWD"] = wt_fwd["beta"]
    out["WT_Beta_RC"] = wt_rc_out["beta"]
    out["WT_Beta_RC_Avg"] = wt_beta_avg
    out["MUT_M_FWD"] = mut_fwd["m"]
    out["MUT_M_RC"] = mut_rc_out["m"]
    out["MUT_M_RC_Avg"] = (mut_fwd["m"] + mut_rc_out["m"]) / 2.0
    out["MUT_Beta_FWD"] = mut_fwd["beta"]
    out["MUT_Beta_RC"] = mut_rc_out["beta"]
    out["MUT_Beta_RC_Avg"] = mut_beta_avg
    out["Delta_Beta_FWD"] = delta_fwd
    out["Delta_Beta_RC"] = delta_rc
    out["Predicted_Delta_Beta"] = delta_avg
    out["Absolute_Delta_Beta"] = np.abs(delta_avg)
    out["Delta_Beta_RC_Absolute_Difference"] = np.abs(delta_fwd - delta_rc)
    out["Delta_Beta_RC_Sign_Agree"] = (np.sign(delta_fwd) == np.sign(delta_rc)).astype(int)
    out["Absolute_Delta_Beta_Rank_Within_Seed"] = (
        out["Absolute_Delta_Beta"].rank(ascending=False, method="min").astype(int)
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def aggregate_sequence(long_df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    seeds = sorted(long_df["Seed"].unique().tolist())
    rows = []
    for uid, group in long_df.groupby("Variant_UID", sort=False):
        group = group.sort_values("Seed")
        if group["Seed"].nunique() != len(seeds):
            raise RuntimeError(f"Missing sequence seed score for {uid}")
        delta = group["Predicted_Delta_Beta"].to_numpy(float)
        ranks = group["Absolute_Delta_Beta_Rank_Within_Seed"].to_numpy(float)
        signs = np.sign(delta)
        nz = signs[signs != 0]
        sign_consistency = 1.0
        if len(nz):
            _, counts = np.unique(nz, return_counts=True)
            sign_consistency = float(counts.max() / len(nz))
        rows.append({
            "Variant_UID": uid,
            "Model": "sequence",
            "Seed_Count": len(seeds),
            "Seeds": ",".join(map(str, seeds)),
            "Predicted_Delta_Beta": float(delta.mean()),
            "Predicted_Delta_Beta_SD": float(delta.std(ddof=0)),
            "Delta_Beta_Sign_Consistency": sign_consistency,
            "Mean_Within_Seed_Rank": float(ranks.mean()),
            "SD_Within_Seed_Rank": float(ranks.std(ddof=0)),
            "Top10_Seed_Frequency": float(np.mean(ranks <= 10)),
            "Top20_Seed_Frequency": float(np.mean(ranks <= 20)),
            "Mean_Delta_RC_Absolute_Difference": float(group["Delta_Beta_RC_Absolute_Difference"].mean()),
            "Delta_RC_Sign_Agreement_Fraction": float(group["Delta_Beta_RC_Sign_Agree"].mean()),
        })
    agg = cohort.drop_duplicates("Variant_UID").merge(
        pd.DataFrame(rows), on="Variant_UID", how="inner", validate="one_to_one"
    )
    agg["Absolute_Delta_Beta"] = agg["Predicted_Delta_Beta"].abs()
    agg["Absolute_Delta_Beta_Rank"] = agg["Absolute_Delta_Beta"].rank(
        ascending=False, method="min"
    ).astype(int)
    return agg.sort_values(["Absolute_Delta_Beta", "Variant_UID"], ascending=[False, True]).reset_index(drop=True)


def pairwise_seed_stability(long_df: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    seeds = sorted(long_df["Seed"].unique())
    for i, seed_a in enumerate(seeds):
        a = long_df[long_df["Seed"] == seed_a][["Variant_UID", "Predicted_Delta_Beta"]]
        for seed_b in seeds[i + 1:]:
            b = long_df[long_df["Seed"] == seed_b][["Variant_UID", "Predicted_Delta_Beta"]]
            x = a.merge(b, on="Variant_UID", suffixes=("_a", "_b"), validate="one_to_one")
            da = x["Predicted_Delta_Beta_a"].to_numpy(float)
            db = x["Predicted_Delta_Beta_b"].to_numpy(float)
            row = {
                "Model": model,
                "Seed_A": int(seed_a),
                "Seed_B": int(seed_b),
                "N": len(x),
                "Signed_Spearman": spearmanr(da, db).statistic,
                "Absolute_Spearman": spearmanr(np.abs(da), np.abs(db)).statistic,
                "Sign_Agreement": float(np.mean(np.sign(da) == np.sign(db))),
            }
            for k in (10, 20):
                top_a = set(x.iloc[np.argsort(-np.abs(da))[:k]]["Variant_UID"])
                top_b = set(x.iloc[np.argsort(-np.abs(db))[:k]]["Variant_UID"])
                row[f"Top{k}_Overlap"] = len(top_a & top_b)
                row[f"Top{k}_Jaccard"] = len(top_a & top_b) / len(top_a | top_b)
            rows.append(row)
    return pd.DataFrame(rows)


def compare_ensembles(fusion: pd.DataFrame, sequence: pd.DataFrame, top_k: list[int]):
    required = {"Variant_UID", "Predicted_Delta_Beta", "Absolute_Delta_Beta_Rank"}
    for name, frame in (("fusion", fusion), ("sequence", sequence)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} aggregate missing columns: {sorted(missing)}")

    fcols = [
        "Variant_UID", "Predicted_Delta_Beta", "Absolute_Delta_Beta",
        "Absolute_Delta_Beta_Rank", "Delta_Beta_Sign_Consistency",
        "Delta_RC_Sign_Agreement_Fraction", "Mean_Delta_RC_Absolute_Difference",
    ]
    fcols = [c for c in fcols if c in fusion.columns]
    scols = [
        "Variant_UID", "Predicted_Delta_Beta", "Absolute_Delta_Beta",
        "Absolute_Delta_Beta_Rank", "Delta_Beta_Sign_Consistency",
        "Delta_RC_Sign_Agreement_Fraction", "Mean_Delta_RC_Absolute_Difference",
        "Matched_Background_Absolute_Effect_Percentile", "Matched_Background_Tier",
    ]
    scols = [c for c in scols if c in sequence.columns]
    joined = fusion[fcols].merge(
        sequence[scols], on="Variant_UID", suffixes=("_Fusion", "_Sequence"), validate="one_to_one"
    )
    df = joined
    fd = df["Predicted_Delta_Beta_Fusion"].to_numpy(float)
    sd = df["Predicted_Delta_Beta_Sequence"].to_numpy(float)
    summary = {
        "n_shared_candidates": int(len(df)),
        "signed_effect_spearman": finite_or_none(spearmanr(fd, sd).statistic),
        "absolute_effect_spearman": finite_or_none(spearmanr(np.abs(fd), np.abs(sd)).statistic),
        "effect_sign_agreement": float(np.mean(np.sign(fd) == np.sign(sd))),
        "mean_absolute_effect_difference": float(np.mean(np.abs(fd - sd))),
    }
    for k in sorted(set(top_k)):
        ftop = set(fusion.nsmallest(k, "Absolute_Delta_Beta_Rank")["Variant_UID"])
        stop = set(sequence.nsmallest(k, "Absolute_Delta_Beta_Rank")["Variant_UID"])
        summary[f"top{k}_overlap"] = len(ftop & stop)
        summary[f"top{k}_jaccard"] = len(ftop & stop) / len(ftop | stop)
    df["Absolute_Rank_Change_Sequence_Minus_Fusion"] = (
        df["Absolute_Delta_Beta_Rank_Sequence"] - df["Absolute_Delta_Beta_Rank_Fusion"]
    )
    return df.sort_values("Absolute_Delta_Beta_Rank_Fusion"), summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args.seeds = BASE.validate_seed_list(args.seeds)
    if args.window_size != 1000:
        raise ValueError("This analysis must use the trained 1000-bp crop")

    fusion_path = Path(args.fusion_aggregate)
    fusion_long_path = Path(args.fusion_seed_scores)
    if not fusion_path.is_file() or not fusion_long_path.is_file():
        raise FileNotFoundError(
            "Run scripts/05_matched_background.py --seeds 42 43 44 first; "
            f"missing {fusion_path if not fusion_path.is_file() else fusion_long_path}"
        )

    raw = pd.read_csv(args.input_csv)
    BASE.validate_candidate_table(raw, args.input_csv)
    raw = BASE.apply_probe_qc(raw, args.hm450_manifest, include_masked=False)
    if args.max_candidates > 0:
        raw = raw.head(args.max_candidates).copy()
        LOGGER.warning("Smoke test: %d input rows", len(raw))
    cohort, wt, mut, _, _ = BASE.build_model_visible_cohort(raw, args.window_size)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_csv(cohort, out / "eligible_candidate_cohort.csv")
    tokenizer = get_tokenizer(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device=%s seeds=%s precision=%s", device, args.seeds, "AMP" if args.amp else "FP32")

    seed_frames = []
    checkpoints = []
    for seed in args.seeds:
        weights = Path(args.sequence_weights_template.format(seed=seed))
        frame = score_sequence_seed(seed, weights, cohort, wt, mut, tokenizer, args, device)
        atomic_csv(frame, out / "sequence" / f"seed{seed}" / "candidate_scores.csv")
        seed_frames.append(frame)
        checkpoints.append({"seed": seed, "path": str(weights), "sha256": BASE.sha256_file(weights)})
    sequence_long = pd.concat(seed_frames, ignore_index=True)
    atomic_csv(sequence_long, out / "sequence" / "candidate_seed_scores_long.csv")

    sequence_agg = aggregate_sequence(sequence_long, cohort)
    sequence_scored, _ = compute_matched_background_statistics(
        sequence_agg,
        delta_column="Predicted_Delta_Beta",
        min_comparators=args.min_comparators,
        max_comparators=args.max_comparators,
        random_seed=args.background_random_seed,
    )
    sequence_scored = sequence_scored.sort_values(
        ["Absolute_Delta_Beta", "Variant_UID"], ascending=[False, True]
    ).reset_index(drop=True)
    sequence_scored["Absolute_Delta_Beta_Rank"] = np.arange(1, len(sequence_scored) + 1)
    atomic_csv(sequence_scored, out / "sequence" / "candidate_matched_background_statistics.csv")
    atomic_csv(sequence_scored.head(max(args.top_k)), out / "sequence" / "top_candidates.csv")

    fusion = pd.read_csv(fusion_path)
    fusion_long = pd.read_csv(fusion_long_path)
    if args.max_candidates > 0:
        keep = set(sequence_scored["Variant_UID"])
        fusion = fusion[fusion["Variant_UID"].isin(keep)].copy()
        fusion_long = fusion_long[fusion_long["Variant_UID"].isin(keep)].copy()
    comparison, summary = compare_ensembles(fusion, sequence_scored, args.top_k)
    atomic_csv(comparison, out / "candidate_sequence_vs_fusion.csv")

    stability = pd.concat([
        pairwise_seed_stability(fusion_long, "fusion"),
        pairwise_seed_stability(sequence_long, "sequence"),
    ], ignore_index=True)
    atomic_csv(stability, out / "pairwise_seed_stability_by_model.csv")

    summary.update({
        "seeds": args.seeds,
        "sequence_checkpoints": checkpoints,
        "sequence_mean_orientation_sign_agreement": float(sequence_scored["Delta_RC_Sign_Agreement_Fraction"].mean()),
        "fusion_mean_orientation_sign_agreement": float(fusion["Delta_RC_Sign_Agreement_Fraction"].mean()),
        "interpretation": (
            "Compares model-derived candidate sensitivity across architectures; "
            "it does not constitute biological validation."
        ),
    })
    atomic_json(summary, out / "sequence_vs_fusion_summary.json")
    LOGGER.info("Saved sequence-versus-fusion candidate comparison to %s", out)


if __name__ == "__main__":
    main()
