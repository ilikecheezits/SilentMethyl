#!/usr/bin/env python3
"""
Evaluate a frozen journal checkpoint on the untouched chromosome-held-out test
split. Merges what were scripts 02_test_epi / 02_test_sequence / 02_test_fusion.

    python -u scripts/13_test_model.py --model fusion
    python -u scripts/13_test_model.py --model sequence --seed 43
    python -u scripts/13_test_model.py --model epi

The three arms differ only in which towers the model has, and therefore in what
the forward pass takes and returns. Everything after the forward pass -- the RC
average, the metrics, the predictions frame, the figures -- was already
identical across the three files and is now written once.

Defaults for weights path, output directory and batch size follow `--model`;
pass the flag explicitly to override. Nothing about the arms' behaviour changed
in the merge: the gate block runs only for fusion, the phyloP-swap note stays on
the context-only arm, and each arm keeps its own `model_type` string, figure
title and log line.

Results section R1 (absolute prediction) and R2 (gate telemetry).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training_common import (
    EpigeneticOnlyModel,
    FusionModel,
    OrientationAwareDataset,
    SequenceOnlyModel,
    autocast_context,
    get_dnabert_hidden_size,
    get_tokenizer,
    load_model_state,
    m_to_beta_tensor,
    orientation_agreement_metrics,
    regression_and_classification_metrics,
    set_seed,
    validate_split_dataframe,
)
from testing_common import (
    enrich_common_metrics,
    ensure_output_dir,
    save_gate_figures,
    save_metrics,
    save_standard_figures,
)


# --------------------------------------------------------------------------
# per-arm specification -- the only place the three models differ
# --------------------------------------------------------------------------

ARMS = {
    "epi": {
        "desc": "[TEST CONTEXT RC-AVERAGED]",
        "model_type": "context_only",
        "figure_title": "Context-only",
        "log_name": "context",
        "batch_size": 512,
        "include_sequence": False,
        "include_context": True,
        "has_gates": False,
        "records_window": False,
        "inference": "forward_reverse_complement_average_with_ordered_phylop_swap",
    },
    "sequence": {
        "desc": "[TEST SEQUENCE RC-AVERAGED]",
        "model_type": "sequence_only",
        "figure_title": "Sequence-only",
        "log_name": "sequence",
        "batch_size": 16,
        "include_sequence": True,
        "include_context": False,
        "has_gates": False,
        "records_window": True,
        "inference": "forward_reverse_complement_average",
    },
    "fusion": {
        "desc": "[TEST GATED FUSION RC-AVERAGED]",
        "model_type": "gated_sequence_context_fusion",
        "figure_title": "Gated sequence + context",
        "log_name": "fusion",
        "batch_size": 16,
        "include_sequence": True,
        "include_context": True,
        "has_gates": True,
        "records_window": True,
        "inference": "forward_reverse_complement_average",
    },
}


def _quantiles(x: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_q10": float(np.quantile(x, 0.10)),
        f"{prefix}_q25": float(np.quantile(x, 0.25)),
        f"{prefix}_median": float(np.quantile(x, 0.50)),
        f"{prefix}_q75": float(np.quantile(x, 0.75)),
        f"{prefix}_q90": float(np.quantile(x, 0.90)),
    }


def forward_pass(model, batch, device, arm: str, orientation: str):
    """One orientation through the model. The three arms take different inputs
    and the fusion arm additionally returns its gate activations."""
    if arm == "epi":
        logits, m = model(
            batch[f"tab_{orientation}"].to(device),
            batch[f"tab_missing_{orientation}"].to(device),
        )
        return logits, m, None
    if arm == "sequence":
        logits, m = model(
            batch[f"input_ids_{orientation}"].to(device),
            batch[f"attention_mask_{orientation}"].to(device),
        )
        return logits, m, None
    logits, m, gates = model(
        batch[f"tab_{orientation}"].to(device),
        batch[f"tab_missing_{orientation}"].to(device),
        batch[f"input_ids_{orientation}"].to(device),
        batch[f"attention_mask_{orientation}"].to(device),
    )
    return logits, m, gates


def evaluate(model, loader, device, amp_enabled, arm: str):
    spec = ARMS[arm]
    model.eval()
    row_indices = []
    beta_true, m_true, binary_true = [], [], []
    m_fwd_all, m_rc_all = [], []
    beta_fwd_all, beta_rc_all = [], []
    prob_fwd_all, prob_rc_all = [], []
    gate_fwd_all, gate_rc_all = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=spec["desc"]):
            with autocast_context(device, amp_enabled):
                logits_fwd, m_fwd, gates_fwd = forward_pass(model, batch, device, arm, "fwd")
                logits_rc, m_rc, gates_rc = forward_pass(model, batch, device, arm, "rc")
                beta_fwd = m_to_beta_tensor(m_fwd)
                beta_rc = m_to_beta_tensor(m_rc)
                prob_fwd = torch.sigmoid(logits_fwd)
                prob_rc = torch.sigmoid(logits_rc)

            row_indices.extend(batch["index"].cpu().numpy().astype(int).tolist())
            beta_true.extend(batch["beta_value"].cpu().float().numpy().ravel())
            m_true.extend(batch["m_value"].cpu().float().numpy().ravel())
            binary_true.extend(batch["binary_state"].cpu().float().numpy().ravel())
            m_fwd_all.extend(m_fwd.cpu().float().numpy().ravel())
            m_rc_all.extend(m_rc.cpu().float().numpy().ravel())
            beta_fwd_all.extend(beta_fwd.cpu().float().numpy().ravel())
            beta_rc_all.extend(beta_rc.cpu().float().numpy().ravel())
            prob_fwd_all.extend(prob_fwd.cpu().float().numpy().ravel())
            prob_rc_all.extend(prob_rc.cpu().float().numpy().ravel())
            if spec["has_gates"]:
                gate_fwd_all.append(gates_fwd.cpu().float().numpy())
                gate_rc_all.append(gates_rc.cpu().float().numpy())

    arrays = [np.asarray(x, dtype=np.float64) for x in (
        m_fwd_all, m_rc_all, beta_fwd_all, beta_rc_all, prob_fwd_all, prob_rc_all
    )]
    m_fwd, m_rc, beta_fwd, beta_rc, prob_fwd, prob_rc = arrays
    m_avg = (m_fwd + m_rc) / 2.0
    beta_avg = (beta_fwd + beta_rc) / 2.0
    prob_avg = (prob_fwd + prob_rc) / 2.0

    metrics = regression_and_classification_metrics(
        beta_true, m_true, m_avg, prob_avg, binary_true, beta_pred_avg=beta_avg
    )
    metrics.update(orientation_agreement_metrics(beta_fwd, beta_rc, prob_fwd, prob_rc))

    source = loader.dataset.df.iloc[row_indices].reset_index(drop=True)
    columns = {
        "probeID": source["probeID"].astype(str).to_numpy(),
        "chr": source["chr"].astype(str).to_numpy() if "chr" in source.columns else "",
        "pos": source["pos"].to_numpy() if "pos" in source.columns else np.nan,
        "true_beta": np.asarray(beta_true, dtype=np.float64),
        "true_m": np.asarray(m_true, dtype=np.float64),
        "binary_true": np.asarray(binary_true, dtype=np.int64),
        "pred_m_fwd": m_fwd,
        "pred_m_rc": m_rc,
        "pred_m_rc_avg": m_avg,
        "pred_beta_fwd": beta_fwd,
        "pred_beta_rc": beta_rc,
        "pred_beta_rc_avg": beta_avg,
        "class_prob_fwd": prob_fwd,
        "class_prob_rc": prob_rc,
        "class_prob_rc_avg": prob_avg,
    }

    if spec["has_gates"]:
        gf = np.concatenate(gate_fwd_all, axis=0).astype(np.float64)
        gr = np.concatenate(gate_rc_all, axis=0).astype(np.float64)
        gavg = (gf + gr) / 2.0
        dna_gate = gavg[:, 0]
        epi_gate = gavg[:, 1]
        dna_share = dna_gate / np.maximum(dna_gate + epi_gate, 1e-12)
        dna_share_fwd = gf[:, 0] / np.maximum(gf[:, 0] + gf[:, 1], 1e-12)
        dna_share_rc = gr[:, 0] / np.maximum(gr[:, 0] + gr[:, 1], 1e-12)

        metrics.update({
            "gate_dna_mean": float(dna_gate.mean()),
            "gate_epi_mean": float(epi_gate.mean()),
            "gate_dna_mean_fwd": float(gf[:, 0].mean()),
            "gate_epi_mean_fwd": float(gf[:, 1].mean()),
            "gate_dna_mean_rc": float(gr[:, 0].mean()),
            "gate_epi_mean_rc": float(gr[:, 1].mean()),
            "gate_fwd_rc_mae": float(np.mean(np.abs(gf - gr))),
            "gate_dna_share_mean": float(dna_share.mean()),
            "gate_dna_share_fwd_rc_mae": float(np.mean(np.abs(dna_share_fwd - dna_share_rc))),
            "gate_dna_dominant_fraction": float(np.mean(dna_share > 0.60)),
            "gate_epi_dominant_fraction": float(np.mean(dna_share < 0.40)),
            "gate_balanced_fraction": float(np.mean((dna_share >= 0.40) & (dna_share <= 0.60))),
        })
        metrics.update(_quantiles(dna_gate, "gate_dna"))
        metrics.update(_quantiles(epi_gate, "gate_epi"))
        metrics.update(_quantiles(dna_share, "gate_dna_share"))

        columns.update({
            "gate_dna_fwd": gf[:, 0],
            "gate_epi_fwd": gf[:, 1],
            "gate_dna_rc": gr[:, 0],
            "gate_epi_rc": gr[:, 1],
            "gate_dna_avg": dna_gate,
            "gate_epi_avg": epi_gate,
            "gate_dna_share_avg": dna_share,
            "gate_dna_share_fwd": dna_share_fwd,
            "gate_dna_share_rc": dna_share_rc,
        })

    predictions = pd.DataFrame(columns)
    predictions["beta_signed_error"] = predictions["pred_beta_rc_avg"] - predictions["true_beta"]
    predictions["beta_absolute_error"] = predictions["beta_signed_error"].abs()
    return metrics, predictions


def build_model(arm: str, args):
    if arm == "epi":
        hidden_size = get_dnabert_hidden_size(args.model_path, args.local_model_dir)
        return EpigeneticOnlyModel(hidden_size=hidden_size, tabular_dim=9)
    if arm == "sequence":
        return SequenceOnlyModel(args.model_path, args.local_model_dir)
    return FusionModel(
        args.model_path,
        fusion_mode="gated",
        tabular_dim=9,
        local_dir=args.local_model_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_path", default="data/datafiles/test.csv")
    parser.add_argument("--weights_path", default=None,
                        help="default: checkpoints_journal/seed<SEED>/<MODEL>/best_weights.pth")
    parser.add_argument("--output_dir", default=None,
                        help="default: results/journal/seed<SEED>/<MODEL>")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local_model_dir", default="./dnabert2_local")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="default: 512 for epi, 16 for sequence and fusion")
    parser.add_argument("--window_size", type=int, default=1000,
                        help="ignored by the context-only arm")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=0,
                        help="0 uses the full test split; positive values are smoke-test only")
    args = parser.parse_args()

    arm = args.model
    spec = ARMS[arm]
    if args.weights_path is None:
        args.weights_path = f"checkpoints_journal/seed{args.seed}/{arm}/best_weights.pth"
    if args.output_dir is None:
        args.output_dir = f"results/journal/seed{args.seed}/{arm}"
    if args.batch_size is None:
        args.batch_size = spec["batch_size"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    out = ensure_output_dir(args.output_dir)

    df = pd.read_csv(args.test_path)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    validate_split_dataframe(df, "test", args.test_path)

    dataset_kwargs = dict(
        training=False,
        seed=args.seed,
        include_sequence=spec["include_sequence"],
        include_context=spec["include_context"],
    )
    if spec["include_sequence"]:
        dataset_kwargs["tokenizer"] = get_tokenizer(args.model_path)
        dataset_kwargs["seq_window_size"] = args.window_size
    dataset = OrientationAwareDataset(df, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    model = build_model(arm, args)
    model.load_state_dict(load_model_state(args.weights_path, map_location="cpu"),
                          strict=True)
    model = model.to(device)
    model.eval()

    metrics, predictions = evaluate(model, loader, device, amp_enabled, arm)
    metrics = enrich_common_metrics(metrics, predictions)
    metrics.update({
        "model_type": spec["model_type"],
        "split": "test",
        "weights_path": str(Path(args.weights_path)),
        "test_path": str(Path(args.test_path)),
        "inference": spec["inference"],
    })
    if spec["records_window"]:
        metrics["window_size_bp"] = int(args.window_size)
    if spec["has_gates"]:
        metrics["gate_semantics"] = (
            "descriptive sample-specific modality utilization/scaling; not causal "
            "attribution; independent sigmoid gates do not sum to one")

    predictions.to_csv(out / "predictions.csv", index=False)
    save_metrics(out / "metrics.json", metrics)
    save_standard_figures(predictions, metrics, out, spec["figure_title"])
    if spec["has_gates"]:
        save_gate_figures(predictions, out)

    logging.info(
        "TEST %s | n=%d | beta MAE %.5f RMSE %.5f | M MAE %.5f RMSE %.5f | "
        "AUC %.5f | RC beta MAE %.5f corr %.5f",
        spec["log_name"], len(predictions), metrics["beta_mae"], metrics["beta_rmse"],
        metrics["m_mae"], metrics["m_rmse"], metrics["auc"],
        metrics["beta_fwd_rc_mae"], metrics["beta_fwd_rc_pearson"])
    if spec["has_gates"]:
        logging.info(
            "Gates | DNA mean %.3f | EPI mean %.3f | DNA share mean %.3f | "
            "DNA-dom %.1f%% balanced %.1f%% EPI-dom %.1f%% | gate RC MAE %.5f",
            metrics["gate_dna_mean"], metrics["gate_epi_mean"],
            metrics["gate_dna_share_mean"],
            100.0 * metrics["gate_dna_dominant_fraction"],
            100.0 * metrics["gate_balanced_fraction"],
            100.0 * metrics["gate_epi_dominant_fraction"],
            metrics["gate_fwd_rc_mae"])
    logging.info("Saved journal test outputs to %s", out)


if __name__ == "__main__":
    main()
