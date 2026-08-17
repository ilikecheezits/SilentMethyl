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
    FusionModel,
    OrientationAwareDataset,
    autocast_context,
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


def _quantiles(x: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_q10": float(np.quantile(x, 0.10)),
        f"{prefix}_q25": float(np.quantile(x, 0.25)),
        f"{prefix}_median": float(np.quantile(x, 0.50)),
        f"{prefix}_q75": float(np.quantile(x, 0.75)),
        f"{prefix}_q90": float(np.quantile(x, 0.90)),
    }


def evaluate(model, loader, device, amp_enabled):
    model.eval()
    row_indices = []
    beta_true, m_true, binary_true = [], [], []
    m_fwd_all, m_rc_all = [], []
    beta_fwd_all, beta_rc_all = [], []
    prob_fwd_all, prob_rc_all = [], []
    gate_fwd_all, gate_rc_all = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[TEST GATED FUSION RC-AVERAGED]"):
            with autocast_context(device, amp_enabled):
                logits_fwd, m_fwd, gates_fwd = model(
                    batch["tab_fwd"].to(device),
                    batch["tab_missing_fwd"].to(device),
                    batch["input_ids_fwd"].to(device),
                    batch["attention_mask_fwd"].to(device),
                )
                logits_rc, m_rc, gates_rc = model(
                    batch["tab_rc"].to(device),
                    batch["tab_missing_rc"].to(device),
                    batch["input_ids_rc"].to(device),
                    batch["attention_mask_rc"].to(device),
                )
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

    source = loader.dataset.df.iloc[row_indices].reset_index(drop=True)
    predictions = pd.DataFrame({
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
    predictions["beta_signed_error"] = predictions["pred_beta_rc_avg"] - predictions["true_beta"]
    predictions["beta_absolute_error"] = predictions["beta_signed_error"].abs()
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate journal gated sequence/context model on the untouched chromosome-held-out test split")
    parser.add_argument("--test_path", default="data/datafiles/test.csv")
    parser.add_argument("--weights_path", default="checkpoints_journal/seed42/fusion/best_weights.pth")
    parser.add_argument("--output_dir", default="results/journal/seed42/fusion")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local_model_dir", default="./dnabert2_local")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=0, help="0 uses the full test split; positive values are smoke-test only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    out = ensure_output_dir(args.output_dir)

    df = pd.read_csv(args.test_path)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    validate_split_dataframe(df, "test", args.test_path)

    tokenizer = get_tokenizer(args.model_path)
    dataset = OrientationAwareDataset(
        df,
        tokenizer=tokenizer,
        seq_window_size=args.window_size,
        training=False,
        seed=args.seed,
        include_sequence=True,
        include_context=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = FusionModel(
        args.model_path,
        fusion_mode="gated",
        tabular_dim=9,
        local_dir=args.local_model_dir,
    )
    model.load_state_dict(load_model_state(args.weights_path, map_location="cpu"), strict=True)
    model = model.to(device)
    model.eval()

    metrics, predictions = evaluate(model, loader, device, amp_enabled)
    metrics = enrich_common_metrics(metrics, predictions)
    metrics.update({
        "model_type": "gated_sequence_context_fusion",
        "split": "test",
        "weights_path": str(Path(args.weights_path)),
        "test_path": str(Path(args.test_path)),
        "window_size_bp": int(args.window_size),
        "inference": "forward_reverse_complement_average",
        "gate_semantics": "descriptive sample-specific modality utilization/scaling; not causal attribution; independent sigmoid gates do not sum to one",
    })

    predictions.to_csv(out / "predictions.csv", index=False)
    save_metrics(out / "metrics.json", metrics)
    save_standard_figures(predictions, metrics, out, "Gated sequence + context")
    save_gate_figures(predictions, out)

    logging.info("TEST fusion | n=%d | beta MAE %.5f RMSE %.5f | M MAE %.5f RMSE %.5f | AUC %.5f | RC beta MAE %.5f corr %.5f",
                 len(predictions), metrics["beta_mae"], metrics["beta_rmse"], metrics["m_mae"], metrics["m_rmse"], metrics["auc"], metrics["beta_fwd_rc_mae"], metrics["beta_fwd_rc_pearson"])
    logging.info("Gates | DNA mean %.3f | EPI mean %.3f | DNA share mean %.3f | DNA-dom %.1f%% balanced %.1f%% EPI-dom %.1f%% | gate RC MAE %.5f",
                 metrics["gate_dna_mean"], metrics["gate_epi_mean"], metrics["gate_dna_share_mean"], 100.0 * metrics["gate_dna_dominant_fraction"], 100.0 * metrics["gate_balanced_fraction"], 100.0 * metrics["gate_epi_dominant_fraction"], metrics["gate_fwd_rc_mae"])
    logging.info("Saved journal test outputs to %s", out)


if __name__ == "__main__":
    main()
