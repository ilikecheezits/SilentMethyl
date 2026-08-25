from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_metrics(path: str | Path, metrics: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, sort_keys=True)
        handle.write("\n")


def enrich_common_metrics(metrics: Dict[str, float], predictions: pd.DataFrame) -> Dict[str, float]:
    out = dict(metrics)
    signed = predictions["pred_beta_rc_avg"].to_numpy(dtype=float) - predictions["true_beta"].to_numpy(dtype=float)
    out["beta_mean_signed_error"] = float(np.mean(signed))
    out["beta_median_absolute_error"] = float(np.median(np.abs(signed)))
    out["n_test_loci"] = int(len(predictions))
    return out


def save_standard_figures(
    predictions: pd.DataFrame,
    metrics: Dict[str, float],
    output_dir: str | Path,
    model_label: str,
) -> None:
    out = ensure_output_dir(output_dir)

    beta_true = predictions["true_beta"].to_numpy(dtype=float)
    beta_pred = predictions["pred_beta_rc_avg"].to_numpy(dtype=float)
    binary_true = predictions["binary_true"].to_numpy(dtype=int)
    class_prob = predictions["class_prob_rc_avg"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(beta_true, beta_pred, gridsize=45, bins="log", mincnt=1)
    ax.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Ideal fit")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="True methylation fraction (beta)", ylabel="RC-averaged predicted beta")
    ax.set_title(f"{model_label}: predicted vs. true beta")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_1_density_scatter.png", dpi=300)
    plt.close(fig)

    signed = beta_pred - beta_true
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(signed, bins=60)
    ax.axvline(0.0, linestyle="--", linewidth=1.5)
    ax.set(xlabel="Signed beta error (predicted - true)", ylabel="Probe count", title=f"{model_label}: signed beta error")
    ax.text(
        0.03,
        0.97,
        f"Beta MAE: {metrics['beta_mae']:.4f}\nMean signed error: {np.mean(signed):.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )
    fig.tight_layout()
    fig.savefig(out / "fig_2a_signed_error.png", dpi=300)
    plt.close(fig)

    absolute = np.abs(signed)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(absolute, bins=60)
    ax.axvline(metrics["beta_mae"], linestyle="--", linewidth=1.5, label=f"MAE = {metrics['beta_mae']:.4f}")
    ax.set(xlabel="Absolute beta error", ylabel="Probe count", title=f"{model_label}: absolute beta error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_2b_absolute_error.png", dpi=300)
    plt.close(fig)

    bins = np.linspace(0.0, 1.0, 61)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(beta_true, bins=bins, density=True, histtype="step", linewidth=1.8, label="True")
    ax.hist(beta_pred, bins=bins, density=True, histtype="step", linewidth=1.8, label="Predicted")
    ax.set(xlabel="DNA methylation fraction (beta)", ylabel="Density", title=f"{model_label}: beta distribution recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_3_beta_distribution.png", dpi=300)
    plt.close(fig)

    if len(np.unique(binary_true)) == 2 and np.isfinite(metrics.get("auc", np.nan)):
        fpr, tpr, _ = roc_curve(binary_true, class_prob)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, linewidth=1.8, label=f"AUC = {metrics['auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", linewidth=1.2)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{model_label}: ROC")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "fig_4_roc.png", dpi=300)
        plt.close(fig)

        fraction_positive, mean_predicted = calibration_curve(binary_true, class_prob, n_bins=10, strategy="uniform")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(mean_predicted, fraction_positive, marker="o", linewidth=1.8, label=model_label)
        ax.plot([0, 1], [0, 1], "--", linewidth=1.2, label="Perfect calibration")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive fraction", title=f"{model_label}: calibration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_5_calibration.png", dpi=300)
        plt.close(fig)


def save_gate_figures(predictions: pd.DataFrame, output_dir: str | Path) -> None:
    out = ensure_output_dir(output_dir)
    share = predictions["gate_dna_share_avg"].to_numpy(dtype=float)
    share_fwd = predictions["gate_dna_share_fwd"].to_numpy(dtype=float)
    share_rc = predictions["gate_dna_share_rc"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(share, bins=np.linspace(0, 1, 51))
    ax.axvline(0.5, linestyle="--", linewidth=1.3)
    ax.set(xlim=(0, 1), xlabel="Relative DNA gate share", ylabel="Probe count", title="Gated fusion: modality-utilization distribution")
    fig.tight_layout()
    fig.savefig(out / "fig_6_gate_share_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(share_fwd, share_rc, gridsize=45, bins="log", mincnt=1)
    ax.plot([0, 1], [0, 1], "--", linewidth=1.2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Forward DNA gate share", ylabel="RC DNA gate share", title="Gated fusion: orientation consistency")
    fig.tight_layout()
    fig.savefig(out / "fig_7_gate_rc_consistency.png", dpi=300)
    plt.close(fig)
