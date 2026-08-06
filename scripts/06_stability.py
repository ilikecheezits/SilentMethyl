#!/usr/bin/env python3
"""Measure candidate-score stability across one or more trained runs.

A run can be a conventional training seed (loaded from ``--scores-template``)
or any explicitly labelled model tweak supplied as ``--run LABEL=CSV``.  A
single run still produces orientation diagnostics and a candidate table;
cross-run comparisons are added automatically when two or more runs exist.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def unique_seeds(values: Iterable[int]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        seed = int(value)
        if seed not in seen:
            seen.add(seed)
            output.append(seed)
    if not output:
        raise ValueError("At least one seed is required")
    return output


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def parse_labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Expected LABEL=PATH, got {specification!r}")
    label, value = specification.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Missing label in {specification!r}")
    return label, Path(value).expanduser()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SilentMethyl orientation, seed, and model-tweak stability")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Checkpoint seeds. If no seeds or --run values are given, seed 42 is used.")
    p.add_argument("--scores-template", default="results/journal/candidates/seed{seed}/candidate_scores.csv")
    p.add_argument("--run", action="append", default=[], metavar="LABEL=CSV",
                   help="Additional named score table; repeat for multiple model/configuration tweaks.")
    p.add_argument("--output-dir", default="results/journal/candidates/stability")
    p.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    return p.parse_args()


def load_scores(label: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Run {label!r} not found: {path}")
    df = pd.read_csv(path)
    if "Variant_UID" not in df or "Predicted_Delta_Beta" not in df:
        raise ValueError(f"{path} requires Variant_UID and Predicted_Delta_Beta")
    if df["Variant_UID"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate Variant_UID values")
    df = df.copy()
    df["Run_Label"] = label
    df["Variant_UID"] = df["Variant_UID"].astype(str)
    df["Predicted_Delta_Beta"] = pd.to_numeric(df["Predicted_Delta_Beta"], errors="raise")
    df["Absolute_Delta_Beta"] = df["Predicted_Delta_Beta"].abs()
    df["Absolute_Delta_Beta_Rank_Within_Run"] = df["Absolute_Delta_Beta"].rank(
        method="first", ascending=False
    ).astype(int)
    return df.sort_values("Variant_UID").reset_index(drop=True)


def top_uids(df: pd.DataFrame, k: int) -> set[str]:
    return set(df.nlargest(min(k, len(df)), "Absolute_Delta_Beta")["Variant_UID"])


def orientation_row(label: str, df: pd.DataFrame, top_ks: list[int]) -> dict:
    row = {"Run_Label": label, "N": len(df), "Orientation_Columns_Available": False}
    if not {"Delta_Beta_FWD", "Delta_Beta_RC"}.issubset(df.columns):
        return row
    fwd = pd.to_numeric(df["Delta_Beta_FWD"], errors="coerce").to_numpy(float)
    rc = pd.to_numeric(df["Delta_Beta_RC"], errors="coerce").to_numpy(float)
    valid = np.isfinite(fwd) & np.isfinite(rc)
    row.update({
        "Orientation_Columns_Available": True,
        "N_Orientation_Valid": int(valid.sum()),
        "FWD_RC_Spearman": safe_spearman(fwd[valid], rc[valid])[0],
        "Absolute_FWD_RC_Spearman": safe_spearman(np.abs(fwd[valid]), np.abs(rc[valid]))[0],
        "FWD_RC_MAE": float(np.mean(np.abs(fwd[valid] - rc[valid]))) if valid.any() else np.nan,
        "FWD_RC_Sign_Agreement": float(np.mean(np.sign(fwd[valid]) == np.sign(rc[valid]))) if valid.any() else np.nan,
    })
    temp = df.loc[valid].copy()
    temp["_f"] = np.abs(fwd[valid]); temp["_r"] = np.abs(rc[valid])
    for k in top_ks:
        a = set(temp.nlargest(min(k, len(temp)), "_f")["Variant_UID"])
        b = set(temp.nlargest(min(k, len(temp)), "_r")["Variant_UID"])
        row[f"Top{k}_FWD_RC_Overlap"] = len(a & b)
        row[f"Top{k}_FWD_RC_Jaccard"] = len(a & b) / len(a | b) if a | b else np.nan
    return row


def pairwise_row(a_label: str, a: pd.DataFrame, b_label: str, b: pd.DataFrame, top_ks: list[int]) -> dict:
    m = a[["Variant_UID", "Predicted_Delta_Beta"]].merge(
        b[["Variant_UID", "Predicted_Delta_Beta"]], on="Variant_UID", suffixes=("_A", "_B")
    )
    x, y = m["Predicted_Delta_Beta_A"].to_numpy(float), m["Predicted_Delta_Beta_B"].to_numpy(float)
    row = {
        "Run_A": a_label, "Run_B": b_label, "N_Shared": len(m),
        "N_Only_A": len(a) - len(m), "N_Only_B": len(b) - len(m),
        "Delta_Spearman": safe_spearman(x, y)[0],
        "Absolute_Delta_Spearman": safe_spearman(np.abs(x), np.abs(y))[0],
        "Delta_MAE": float(np.mean(np.abs(x-y))) if len(m) else np.nan,
        "Sign_Agreement": float(np.mean(np.sign(x) == np.sign(y))) if len(m) else np.nan,
    }
    for k in top_ks:
        left, right = top_uids(a, k), top_uids(b, k)
        row[f"Top{k}_Overlap"] = len(left & right)
        row[f"Top{k}_Jaccard"] = len(left & right) / len(left | right) if left | right else np.nan
    return row


def candidate_consensus(frames: dict[str, pd.DataFrame], top_ks: list[int]) -> pd.DataFrame:
    long = pd.concat(frames.values(), ignore_index=True)
    rows = []
    for uid, g in long.groupby("Variant_UID", sort=False):
        d = g["Predicted_Delta_Beta"].to_numpy(float)
        signs = np.sign(d); signs = signs[signs != 0]
        sign_consistency = (np.unique(signs, return_counts=True)[1].max() / len(signs)) if len(signs) else 1.0
        first = g.iloc[0]
        row = {
            "Variant_UID": uid,
            "Gene": first.get("Gene", first.get("Selected_Gene_Name", "Unknown")),
            "probeID": first.get("probeID", ""),
            "Run_Count": len(g), "Delta_Beta_Mean": np.mean(d),
            "Delta_Beta_SD": np.std(d, ddof=0), "Absolute_Mean_Delta_Beta": abs(np.mean(d)),
            "Sign_Consistency": sign_consistency,
            "Mean_Absolute_Rank": g["Absolute_Delta_Beta_Rank_Within_Run"].mean(),
        }
        for k in top_ks:
            row[f"Top{k}_Run_Fraction"] = float((g["Absolute_Delta_Beta_Rank_Within_Run"] <= k).mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["Absolute_Mean_Delta_Beta", "Sign_Consistency"], ascending=[False, False]
    ).reset_index(drop=True)
    out.insert(0, "Consensus_Rank", np.arange(1, len(out)+1))
    return out


def main() -> None:
    args = parse_args()
    top_ks = sorted(set(int(k) for k in args.top_k if k > 0))
    frames: dict[str, pd.DataFrame] = {}
    seeds = unique_seeds(args.seeds) if args.seeds else ([] if args.run else [42])
    for seed in seeds:
        label = f"seed{seed}"
        frames[label] = load_scores(label, Path(args.scores_template.format(seed=seed)))
    for spec in args.run:
        label, path = parse_labelled_path(spec)
        if label in frames:
            raise ValueError(f"Duplicate run label: {label}")
        frames[label] = load_scores(label, path)
    if not frames:
        raise ValueError("Supply at least one --seed or --run LABEL=CSV")

    output = Path(args.output_dir)
    orientation = pd.DataFrame([orientation_row(k, v, top_ks) for k, v in frames.items()])
    pairwise_rows = [pairwise_row(ka, a, kb, b, top_ks)
                     for (ka, a), (kb, b) in combinations(frames.items(), 2)]
    pairwise_columns = [
        "Run_A", "Run_B", "N_Shared", "N_Only_A", "N_Only_B", "Delta_Spearman",
        "Absolute_Delta_Spearman", "Delta_MAE", "Sign_Agreement",
        *[name for k in top_ks for name in
          (f"Top{k}_Overlap", f"Top{k}_Jaccard")],
    ]
    pairwise = pd.DataFrame(pairwise_rows, columns=pairwise_columns)
    consensus = candidate_consensus(frames, top_ks)
    atomic_csv(orientation, output / "per_run_orientation_stability.csv")
    atomic_csv(pairwise, output / "pairwise_run_stability.csv")
    atomic_csv(consensus, output / "candidate_run_consensus.csv")
    atomic_json({
        "run_count": len(frames), "run_labels": list(frames),
        "cross_run_available": len(frames) > 1, "top_k": top_ks,
        "interpretation": "Stability describes reproducibility across supplied runs; it is not biological validation.",
    }, output / "stability_summary.json")
    print(f"Saved stability outputs to {output} ({len(frames)} run(s))")


if __name__ == "__main__":
    main()
