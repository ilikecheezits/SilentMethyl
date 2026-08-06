from __future__ import annotations

import argparse
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
    OrientationAwareDataset,
    SequenceOnlyModel,
    autocast_context,
    get_tokenizer,
    m_to_beta_tensor,
    make_amp,
    orientation_agreement_metrics,
    regression_and_classification_metrics,
    save_run_config,
    set_seed,
    validate_split_dataframe,
)


class ResilientSummaryWriter:
    """TensorBoard wrapper that never lets event-file I/O kill training.

    TensorBoard is diagnostic only. If its event file becomes unwritable (for
    example a transient shared-filesystem EIO), the writer is disabled for the
    remainder of the process while model training/checkpointing continues.
    """

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
    """Write a PyTorch checkpoint atomically, retrying transient filesystem errors."""
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
    """Atomically write a CSV, retrying transient filesystem errors."""
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


def evaluate(model, loader, device, amp_enabled):
    model.eval()
    beta_true, m_true, binary_true = [], [], []
    m_fwd_all, m_rc_all = [], []
    beta_fwd_all, beta_rc_all = [], []
    prob_fwd_all, prob_rc_all = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[VAL RC-AVERAGED]", leave=False):
            targets_m = batch["m_value"].to(device).view(-1, 1)
            targets_beta = batch["beta_value"].to(device).view(-1, 1)
            targets_binary = batch["binary_state"].to(device).view(-1, 1)

            with autocast_context(device, amp_enabled):
                logits_fwd, m_fwd = model(
                    batch["input_ids_fwd"].to(device),
                    batch["attention_mask_fwd"].to(device),
                )
                logits_rc, m_rc = model(
                    batch["input_ids_rc"].to(device),
                    batch["attention_mask_rc"].to(device),
                )
                beta_fwd = m_to_beta_tensor(m_fwd)
                beta_rc = m_to_beta_tensor(m_rc)
                prob_fwd = torch.sigmoid(logits_fwd)
                prob_rc = torch.sigmoid(logits_rc)

            beta_true.extend(targets_beta.cpu().float().numpy().ravel())
            m_true.extend(targets_m.cpu().float().numpy().ravel())
            binary_true.extend(targets_binary.cpu().float().numpy().ravel())
            m_fwd_all.extend(m_fwd.cpu().float().numpy().ravel())
            m_rc_all.extend(m_rc.cpu().float().numpy().ravel())
            beta_fwd_all.extend(beta_fwd.cpu().float().numpy().ravel())
            beta_rc_all.extend(beta_rc.cpu().float().numpy().ravel())
            prob_fwd_all.extend(prob_fwd.cpu().float().numpy().ravel())
            prob_rc_all.extend(prob_rc.cpu().float().numpy().ravel())

    m_fwd = np.asarray(m_fwd_all)
    m_rc = np.asarray(m_rc_all)
    beta_fwd = np.asarray(beta_fwd_all)
    beta_rc = np.asarray(beta_rc_all)
    prob_fwd = np.asarray(prob_fwd_all)
    prob_rc = np.asarray(prob_rc_all)

    m_avg = (m_fwd + m_rc) / 2.0
    beta_avg = (beta_fwd + beta_rc) / 2.0
    prob_avg = (prob_fwd + prob_rc) / 2.0

    metrics = regression_and_classification_metrics(
        beta_true=beta_true,
        m_true=m_true,
        m_pred_avg=m_avg,
        beta_pred_avg=beta_avg,
        class_prob_avg=prob_avg,
        binary_true=binary_true,
    )
    metrics.update(orientation_agreement_metrics(beta_fwd, beta_rc, prob_fwd, prob_rc))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Journal sequence-only DNABERT-2 training")
    parser.add_argument("--train_path", default="data/datafiles/train.csv")
    parser.add_argument("--val_path", default="data/datafiles/val.csv")
    parser.add_argument("--save_dir", default="checkpoints_journal_sequence_seed42")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local_model_dir", default="./dnabert2_local")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
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
    parser.add_argument("--max_train_rows", type=int, default=0, help="0 uses all rows; positive values are for smoke testing only")
    parser.add_argument("--max_val_rows", type=int, default=0, help="0 uses all rows; positive values are for smoke testing only")
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
    overlap = set(train_df.probeID).intersection(set(val_df.probeID))
    if overlap:
        raise ValueError(f"Train/validation probe overlap detected: {len(overlap)}")

    tokenizer = get_tokenizer(args.model_path)
    train_dataset = OrientationAwareDataset(
        train_df,
        tokenizer=tokenizer,
        seq_window_size=args.window_size,
        training=True,
        rc_probability=args.rc_probability,
        seed=args.seed,
        include_sequence=True,
        include_context=False,
    )
    val_dataset = OrientationAwareDataset(
        val_df,
        tokenizer=tokenizer,
        seq_window_size=args.window_size,
        training=False,
        seed=args.seed,
        include_sequence=True,
        include_context=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    model = SequenceOnlyModel(args.model_path, args.local_model_dir).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    run_config.update({
        "model_type": "sequence_only",
        "dataset_rows_train": len(train_dataset),
        "dataset_rows_val": len(val_dataset),
        "training_row_multiplier": 1,
        "validation_inference": "forward/RC average",
        "checkpoint_metric": "regression_head_beta_mae_after_RC_averaging",
    })
    save_run_config(args.save_dir, run_config)

    latest_path = os.path.join(args.save_dir, "latest_checkpoint.pt")
    best_path = os.path.join(args.save_dir, "best_weights.pth")
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
        "Sequence-only training: %d train loci, %d val loci, window=%dbp, RC p=%.2f, seed=%d",
        len(train_dataset), len(val_dataset), args.window_size, args.rc_probability, args.seed,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loader = make_train_loader(train_dataset, args, epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        rc_count = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        for step, batch in enumerate(pbar):
            group_start = (step // args.grad_accum_steps) * args.grad_accum_steps
            group_end = min(group_start + args.grad_accum_steps, len(train_loader))
            group_size = group_end - group_start

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            m_target = batch["m_value"].to(device).view(-1, 1)
            binary_target = batch["binary_state"].to(device).view(-1, 1)
            rc_count += float(batch["use_rc"].sum().item())
            seen += int(batch["use_rc"].numel())

            with autocast_context(device, amp_enabled):
                logits, m_pred = model(input_ids, attention_mask)
                loss_bce = criterion_bce(logits, binary_target)
                loss_huber = criterion_huber(m_pred, m_target)
                raw_loss = loss_bce + loss_huber
                loss = raw_loss / group_size

            scaler.scale(loss).backward()
            running_loss += raw_loss.item()

            if step + 1 == group_end:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        metrics = evaluate(model, val_loader, device, amp_enabled)
        rc_fraction = rc_count / max(seen, 1)
        avg_train_loss = running_loss / len(train_loader)

        logger.info(
            "Epoch %d | train loss %.4f | train RC %.3f | M MAE %.4f RMSE %.4f | "
            "beta MAE %.4f RMSE %.4f | AUC %.4f | FWD/RC beta MAE %.5f corr %.5f",
            epoch, avg_train_loss, rc_fraction,
            metrics["m_mae"], metrics["m_rmse"], metrics["beta_mae"], metrics["beta_rmse"],
            metrics["auc"], metrics["beta_fwd_rc_mae"], metrics["beta_fwd_rc_pearson"],
        )

        for key, value in metrics.items():
            writer.add_scalar(f"Val/{key}", value, epoch)
        writer.add_scalar("Train/Epoch_Loss", avg_train_loss, epoch)
        writer.add_scalar("Train/RC_Fraction", rc_fraction, epoch)
        writer.flush()

        is_best = metrics["beta_mae"] < best_val_beta_mae
        if is_best:
            best_val_beta_mae = metrics["beta_mae"]
            atomic_torch_save(model.state_dict(), best_path, logger)
            logger.info("[★] New best sequence model: regression beta MAE %.5f", best_val_beta_mae)

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
                "last_val_metrics": metrics,
            },
            latest_path,
            logger,
        )

    writer.close()


if __name__ == "__main__":
    main()
