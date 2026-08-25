from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from training_common import (
    FusionModel,
    OrientationAwareDataset,
    autocast_context,
    freeze_module,
    get_tokenizer,
    load_submodule_from_checkpoint,
    m_to_beta_tensor,
    make_amp,
    orientation_agreement_metrics,
    regression_and_classification_metrics,
    save_run_config,
    set_seed,
    validate_split_dataframe,
)


class ResilientSummaryWriter:

    def __init__(self, log_dir: str, logger: logging.Logger) -> None:
        self.log_dir = log_dir
        self.logger = logger
        self._writer = None
        self._disabled = False
        try:
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir, max_queue=20, flush_secs=120)
        except OSError as exc:
            self._disable(exc, during="initialization")

    @property
    def enabled(self) -> bool:
        return (not self._disabled) and (self._writer is not None)

    def _disable(self, exc: BaseException, during: str) -> None:
        if not self._disabled:
            self.logger.error(
                "TensorBoard %s failed with %s: %s. Disabling TensorBoard for the rest "
                "of this process; training and checkpointing will continue.",
                during,
                type(exc).__name__,
                exc,
            )
        self._disabled = True
        writer, self._writer = self._writer, None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def add_scalar(self, *args, **kwargs) -> None:
        if not self.enabled:
            return
        try:
            self._writer.add_scalar(*args, **kwargs)
        except OSError as exc:
            self._disable(exc, during="write")

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self._writer.flush()
        except OSError as exc:
            self._disable(exc, during="flush")

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
        except OSError as exc:
            self.logger.warning("TensorBoard close failed with %s: %s", type(exc).__name__, exc)
        finally:
            self._writer = None
            self._disabled = True


def atomic_torch_save(obj, path: str, logger: logging.Logger, retries: int = 3) -> None:
    import time

    tmp_path = f"{path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except (OSError, RuntimeError) as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt == retries:
                logger.exception("Checkpoint save failed after %d attempts: %s", retries, path)
                raise
            delay = 5 * attempt
            logger.warning(
                "Checkpoint save attempt %d/%d failed for %s (%s: %s); retrying in %ds",
                attempt,
                retries,
                path,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)


def atomic_dataframe_to_csv(df: pd.DataFrame, path: str, logger: logging.Logger, retries: int = 3) -> None:
    import time

    tmp_path = f"{path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt == retries:
                logger.exception("CSV save failed after %d attempts: %s", retries, path)
                raise
            delay = 5 * attempt
            logger.warning(
                "CSV save attempt %d/%d failed for %s (%s: %s); retrying in %ds",
                attempt,
                retries,
                path,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)


def make_train_loader(dataset, args, epoch):
    generator = torch.Generator()
    generator.manual_seed(args.seed + 1009 * epoch)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=False,
    )


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    gate_fwd, gate_rc = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[VAL RC-AVERAGED]", leave=False):
            target_m = batch["m_value"].to(device).view(-1, 1)
            target_beta = batch["beta_value"].to(device).view(-1, 1)
            target_binary = batch["binary_state"].to(device).view(-1, 1)

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
            beta_true.extend(target_beta.cpu().float().numpy().ravel())
            m_true.extend(target_m.cpu().float().numpy().ravel())
            binary_true.extend(target_binary.cpu().float().numpy().ravel())
            m_fwd_all.extend(m_fwd.cpu().float().numpy().ravel())
            m_rc_all.extend(m_rc.cpu().float().numpy().ravel())
            beta_fwd_all.extend(beta_fwd.cpu().float().numpy().ravel())
            beta_rc_all.extend(beta_rc.cpu().float().numpy().ravel())
            prob_fwd_all.extend(prob_fwd.cpu().float().numpy().ravel())
            prob_rc_all.extend(prob_rc.cpu().float().numpy().ravel())
            gate_fwd.append(gates_fwd.cpu().float().numpy())
            gate_rc.append(gates_rc.cpu().float().numpy())

    m_fwd = np.asarray(m_fwd_all, dtype=np.float64)
    m_rc = np.asarray(m_rc_all, dtype=np.float64)
    beta_fwd = np.asarray(beta_fwd_all, dtype=np.float64)
    beta_rc = np.asarray(beta_rc_all, dtype=np.float64)
    prob_fwd = np.asarray(prob_fwd_all, dtype=np.float64)
    prob_rc = np.asarray(prob_rc_all, dtype=np.float64)

    m_avg = (m_fwd + m_rc) / 2.0
    beta_avg = (beta_fwd + beta_rc) / 2.0
    prob_avg = (prob_fwd + prob_rc) / 2.0

    metrics = regression_and_classification_metrics(
        beta_true,
        m_true,
        m_avg,
        prob_avg,
        binary_true,
        beta_pred_avg=beta_avg,
    )
    metrics.update(orientation_agreement_metrics(beta_fwd, beta_rc, prob_fwd, prob_rc))

    gf = np.concatenate(gate_fwd, axis=0).astype(np.float64)
    gr = np.concatenate(gate_rc, axis=0).astype(np.float64)
    gavg = (gf + gr) / 2.0

    dna_gate = gavg[:, 0]
    epi_gate = gavg[:, 1]
    denom = np.maximum(dna_gate + epi_gate, 1e-12)
    dna_share = dna_gate / denom

    share_fwd = gf[:, 0] / np.maximum(gf[:, 0] + gf[:, 1], 1e-12)
    share_rc = gr[:, 0] / np.maximum(gr[:, 0] + gr[:, 1], 1e-12)

    metrics.update(
        {
            "gate_dna_mean": float(dna_gate.mean()),
            "gate_epi_mean": float(epi_gate.mean()),
            "gate_dna_mean_fwd": float(gf[:, 0].mean()),
            "gate_epi_mean_fwd": float(gf[:, 1].mean()),
            "gate_dna_mean_rc": float(gr[:, 0].mean()),
            "gate_epi_mean_rc": float(gr[:, 1].mean()),
            "gate_fwd_rc_mae": float(np.mean(np.abs(gf - gr))),
            "gate_dna_share_mean": float(dna_share.mean()),
            "gate_dna_share_fwd_rc_mae": float(np.mean(np.abs(share_fwd - share_rc))),
            "gate_dna_dominant_fraction": float(np.mean(dna_share > 0.60)),
            "gate_epi_dominant_fraction": float(np.mean(dna_share < 0.40)),
            "gate_balanced_fraction": float(np.mean((dna_share >= 0.40) & (dna_share <= 0.60))),
        }
    )
    metrics.update(_quantiles(dna_gate, "gate_dna"))
    metrics.update(_quantiles(epi_gate, "gate_epi"))
    metrics.update(_quantiles(dna_share, "gate_dna_share"))

    source_df = loader.dataset.df.iloc[row_indices].reset_index(drop=True)
    gate_df = pd.DataFrame(
        {
            "probeID": source_df["probeID"].astype(str).to_numpy(),
            "chr": source_df["chr"].astype(str).to_numpy() if "chr" in source_df.columns else "",
            "pos": source_df["pos"].to_numpy() if "pos" in source_df.columns else np.nan,
            "true_beta": np.asarray(beta_true, dtype=np.float64),
            "pred_beta_rc_avg": beta_avg,
            "true_m": np.asarray(m_true, dtype=np.float64),
            "pred_m_rc_avg": m_avg,
            "class_prob_rc_avg": prob_avg,
            "gate_dna_fwd": gf[:, 0],
            "gate_epi_fwd": gf[:, 1],
            "gate_dna_rc": gr[:, 0],
            "gate_epi_rc": gr[:, 1],
            "gate_dna_avg": dna_gate,
            "gate_epi_avg": epi_gate,
            "gate_dna_share_avg": dna_share,
            "gate_dna_share_fwd": share_fwd,
            "gate_dna_share_rc": share_rc,
        }
    )

    return metrics, gate_df


def main():
    parser = argparse.ArgumentParser(
        description="SilentMethyl journal gated sequence/context fusion training"
    )
    parser.add_argument("--train_path", default="data/datafiles/train.csv")
    parser.add_argument("--val_path", default="data/datafiles/val.csv")
    parser.add_argument("--sequence_weights", required=True)
    parser.add_argument("--epi_weights", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local_model_dir", default="./dnabert2_local")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--warmup_fraction", type=float, default=0.10)
    parser.add_argument("--rc_probability", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--tensorboard_log_every",
        type=int,
        default=50,
        help="Log per-step TensorBoard scalars every N optimizer updates; epoch metrics are always logged.",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=0,
        help="0 uses all rows; positive values are for smoke testing only",
    )
    parser.add_argument(
        "--max_val_rows",
        type=int,
        default=0,
        help="0 uses all rows; positive values are for smoke testing only",
    )
    parser.add_argument(
        "--unfreeze_towers",
        action="store_true",
        help=(
            "Fine-tune the already task-trained sequence/context encoders during fusion. "
            "Default keeps them frozen so the learned gates describe weighting of fixed, "
            "independently trained modality representations."
        ),
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=2e-5,
        help="Used only with --unfreeze_towers.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    writer = ResilientSummaryWriter(
        log_dir=os.path.join(args.save_dir, "tensorboard"),
        logger=logger,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    if args.max_train_rows > 0:
        train_df = train_df.head(args.max_train_rows).copy()
    if args.max_val_rows > 0:
        val_df = val_df.head(args.max_val_rows).copy()

    validate_split_dataframe(train_df, "train", args.train_path)
    validate_split_dataframe(val_df, "val", args.val_path)
    if set(train_df.probeID).intersection(set(val_df.probeID)):
        raise ValueError("Train/validation probe overlap detected")

    tokenizer = get_tokenizer(args.model_path)
    train_dataset = OrientationAwareDataset(
        train_df,
        tokenizer=tokenizer,
        seq_window_size=args.window_size,
        training=True,
        rc_probability=args.rc_probability,
        seed=args.seed,
        include_sequence=True,
        include_context=True,
    )
    val_dataset = OrientationAwareDataset(
        val_df,
        tokenizer=tokenizer,
        seq_window_size=args.window_size,
        training=False,
        seed=args.seed,
        include_sequence=True,
        include_context=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    model = FusionModel(
        args.model_path,
        fusion_mode="gated",
        tabular_dim=9,
        local_dir=args.local_model_dir,
    )

    load_submodule_from_checkpoint(
        model.sequence_encoder, args.sequence_weights, "sequence_encoder", map_location="cpu"
    )
    load_submodule_from_checkpoint(
        model.epi_encoder, args.epi_weights, "epi_encoder", map_location="cpu"
    )
    sequence_sha256 = _sha256(args.sequence_weights)
    epi_sha256 = _sha256(args.epi_weights)
    logger.info("Loaded sequence ancestor strictly: %s", args.sequence_weights)
    logger.info("Sequence ancestor SHA256: %s", sequence_sha256)
    logger.info("Loaded context ancestor strictly: %s", args.epi_weights)
    logger.info("Context ancestor SHA256: %s", epi_sha256)

    if not args.unfreeze_towers:
        freeze_module(model.sequence_encoder)
        freeze_module(model.epi_encoder)

    model = model.to(device)

    if args.unfreeze_towers:
        encoder_params = list(model.sequence_encoder.parameters()) + list(model.epi_encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        fusion_params = [p for p in model.parameters() if id(p) not in encoder_ids]
        optimizer = optim.AdamW(
            [
                {"params": encoder_params, "lr": args.encoder_lr},
                {"params": fusion_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    micro_batches = math.ceil(len(train_dataset) / args.batch_size)
    updates_per_epoch = math.ceil(micro_batches / args.grad_accum_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_fraction * total_updates),
        num_training_steps=total_updates,
    )

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_huber = nn.HuberLoss(delta=1.345)
    scaler, amp_enabled = make_amp(device)

    run_config = vars(args).copy()
    run_config.update(
        {
            "model_type": "gated_sequence_context_fusion",
            "dataset_rows_train": len(train_dataset),
            "dataset_rows_val": len(val_dataset),
            "training_row_multiplier": 1,
            "fusion": "independent sigmoid scalar gates applied after LayerNorm to sequence and context embeddings",
            "gate_semantics": (
                "descriptive sample-specific modality utilization/scaling; "
                "not causal attribution and not constrained to sum to one"
            ),
            "ancestor_loading": "encoder submodules only; fresh gate network and prediction heads",
            "sequence_weights_path": os.path.abspath(args.sequence_weights),
            "sequence_weights_sha256": sequence_sha256,
            "epi_weights_path": os.path.abspath(args.epi_weights),
            "epi_weights_sha256": epi_sha256,
            "towers_frozen": not args.unfreeze_towers,
            "validation_inference": "forward/RC prediction average",
            "gate_validation": "forward and RC gates both recorded; per-locus averages saved for best checkpoint",
            "checkpoint_metric": "regression_head_beta_mae_after_RC_averaging",
        }
    )
    save_run_config(args.save_dir, run_config)

    latest_path = os.path.join(args.save_dir, "latest_checkpoint.pt")
    best_path = os.path.join(args.save_dir, "best_weights.pth")
    best_gates_path = os.path.join(args.save_dir, "best_validation_gates.csv")
    start_epoch = 1
    best_val_beta_mae = float("inf")
    global_step = 0

    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_val_beta_mae = float(ckpt["best_val_beta_mae"])
        global_step = int(ckpt.get("global_step", 0))
        logger.info("Resuming at epoch %d", start_epoch)

    logger.info(
        "Gated fusion training: %d train loci, %d val loci, window=%dbp, RC p=%.2f, seed=%d, towers_frozen=%s",
        len(train_dataset),
        len(val_dataset),
        args.window_size,
        args.rc_probability,
        args.seed,
        not args.unfreeze_towers,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loader = make_train_loader(train_dataset, args, epoch)
        model.train()

        if not args.unfreeze_towers:
            model.sequence_encoder.eval()
            model.epi_encoder.eval()

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        rc_count = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN GATED]")
        for step, batch in enumerate(pbar):
            group_start = (step // args.grad_accum_steps) * args.grad_accum_steps
            group_end = min(group_start + args.grad_accum_steps, len(train_loader))
            group_size = group_end - group_start

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tab = batch["tab"].to(device)
            tab_missing = batch["tab_missing"].to(device)
            m_target = batch["m_value"].to(device).view(-1, 1)
            binary_target = batch["binary_state"].to(device).view(-1, 1)
            rc_count += float(batch["use_rc"].sum().item())
            seen += int(batch["use_rc"].numel())

            with autocast_context(device, amp_enabled):
                logits, m_pred, _gates = model(tab, tab_missing, input_ids, attention_mask)
                loss_bce = criterion_bce(logits, binary_target)
                loss_huber = criterion_huber(m_pred, m_target)
                raw_loss = loss_bce + loss_huber
                loss = raw_loss / group_size

            scaler.scale(loss).backward()
            running_loss += raw_loss.item()

            if step + 1 == group_end:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scale_before <= scaler.get_scale():
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or global_step % max(1, args.tensorboard_log_every) == 0:
                    writer.add_scalar("Train/Loss", raw_loss.item(), global_step)
                    writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix(loss=f"{raw_loss.item():.4f}")

        metrics, gate_df = evaluate(model, val_loader, device, amp_enabled)
        rc_fraction = rc_count / max(seen, 1)
        avg_train_loss = running_loss / len(train_loader)

        logger.info(
            "Epoch %d | train loss %.4f | train RC %.3f | M MAE %.4f RMSE %.4f | "
            "beta MAE %.4f RMSE %.4f | AUC %.4f | FWD/RC beta MAE %.5f corr %.5f",
            epoch,
            avg_train_loss,
            rc_fraction,
            metrics["m_mae"],
            metrics["m_rmse"],
            metrics["beta_mae"],
            metrics["beta_rmse"],
            metrics["auc"],
            metrics["beta_fwd_rc_mae"],
            metrics["beta_fwd_rc_pearson"],
        )
        logger.info(
            "Gates | DNA raw mean %.3f (median %.3f) | EPI raw mean %.3f (median %.3f) | "
            "relative DNA share mean %.3f (median %.3f) | DNA-dom %.1f%% balanced %.1f%% EPI-dom %.1f%% | "
            "gate RC MAE %.4f",
            metrics["gate_dna_mean"],
            metrics["gate_dna_median"],
            metrics["gate_epi_mean"],
            metrics["gate_epi_median"],
            metrics["gate_dna_share_mean"],
            metrics["gate_dna_share_median"],
            100.0 * metrics["gate_dna_dominant_fraction"],
            100.0 * metrics["gate_balanced_fraction"],
            100.0 * metrics["gate_epi_dominant_fraction"],
            metrics["gate_fwd_rc_mae"],
        )

        for key, value in metrics.items():
            writer.add_scalar(f"Val/{key}", value, epoch)
        writer.add_scalar("Train/Epoch_Loss", avg_train_loss, epoch)
        writer.add_scalar("Train/RC_Fraction", rc_fraction, epoch)
        writer.flush()

        if metrics["beta_mae"] < best_val_beta_mae:
            best_val_beta_mae = metrics["beta_mae"]
            atomic_torch_save(model.state_dict(), best_path, logger)
            atomic_dataframe_to_csv(gate_df, best_gates_path, logger)
            logger.info(
                "[★] New best gated fusion: regression beta MAE %.5f; saved weights + per-locus validation gates",
                best_val_beta_mae,
            )

        atomic_torch_save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_beta_mae": best_val_beta_mae,
                "global_step": global_step,
                "seed": args.seed,
                "rc_probability": args.rc_probability,
                "fusion_mode": "gated",
                "last_val_metrics": metrics,
            },
            latest_path,
            logger,
        )

    writer.close()


if __name__ == "__main__":
    main()
