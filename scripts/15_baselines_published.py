#!/usr/bin/env python
"""CpGenie and DeepCpG architectures, reimplemented and trained on our splits.

WHY REIMPLEMENT RATHER THAN RUN THE ORIGINAL CODE

Both are 2017-era Keras/Theano/TF1 and do not install on current hardware. More
importantly, the published CpGenie weights were trained on GM12878
lymphoblastoid cells, not breast, so scoring them against our targets would hand
them a cross-tissue handicap and produce a strawman comparison. Training their
architectures on our data, our splits and our tissue is the fair test and the
one the reviewer question actually asks.

ARCHITECTURES, TAKEN FROM SOURCE, NOT FROM MEMORY

CpGenie -- data/external/baselines/CpGenie/cnn/seq_128x3_5_5_2f_simple.template
    The template's Convolution2D(128, 1, 5) over a (4, 1, L) input is a width-5
    1-D convolution with the four bases as channels.
        conv[128@5,same] -> maxpool[5,stride3]
        conv[256@5,same] -> maxpool[5,stride3]
        conv[512@5,same] -> maxpool[5,stride3]
        flatten -> dense[64] -> dropout -> dense[64] -> dropout -> head
    Max-norm 3 on the convolutional weights; RMSprop(rho=0.9, eps=1e-6).
    Dropout and learning rate were tuned by the authors via hyperas over
    dropout in {0.3, 0.5, 0.7} and lr in {0.01, 0.001, 0.0001}; we tune over the
    same grid on our validation split rather than fixing an arbitrary value.

DeepCpG -- data/external/baselines/deepcpg/deepcpg/models/dna.py, CnnL2h128
        conv[128@11] -> maxpool[4] -> conv[256@3] -> maxpool[2]
        -> flatten -> dense[128] -> dropout -> head
    Base-class defaults: dropout 0.0, l1_decay 0.0, l2_decay 0.0,
    glorot_uniform (models/utils.py:441). We additionally tune dropout on our
    validation split, which can only help the baseline.
    Faithfulness check: their docstring states 4,100,000 parameters; at our
    1,000 bp input the trunk computes to 3,997,824 plus heads. FAITHFULNESS_CHECK
    below asserts this, so a misreading of the layer spec fails loudly.

DELIBERATE DEVIATIONS, ALL STATED IN THE MANUSCRIPT

  * Output head. Both originals predict a binary methylation state. Our metrics
    include beta MAE and M-value MAE, so each published trunk carries the same
    dual head our models use (M-value regression + binary logit) and the same
    losses (Huber delta=1.345 + BCEWithLogits). This isolates the architecture
    rather than the output parameterisation, which is the comparison we want.
  * Window. Trained at our 1,000 bp rather than their defaults, so the input is
    identical across all models in the table.
  * Reverse-complement handling. Same deterministic RC augmentation during
    training and RC-averaged prediction at test time as our models, imported
    from training_common rather than reimplemented.

    python -u scripts/15_baselines_published.py \
        --arch cpgenie --dropout 0.5 --lr 0.001 --seed 42
    python -u scripts/15_baselines_published.py --arch cpgenie --grid
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# training_common imports transformers at module level, which walks the whole
# transformers package tree on the shared filesystem and stalls for minutes on a
# cold node. A CNN baseline has no business paying that cost for four small pure
# functions, so they are duplicated here VERBATIM from training_common and
# checked against it by --verify (the pattern scripts/23 already uses).

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def centered_crop(seq: str, window_size: int) -> str:
    seq = seq.upper()
    if window_size <= 0 or window_size > len(seq):
        raise ValueError(f"window_size={window_size} is invalid for sequence length {len(seq)}")
    center_right = len(seq) // 2
    start = center_right - (window_size // 2)
    end = start + window_size
    return seq[start:end]


def m_to_beta_numpy(m_value: np.ndarray) -> np.ndarray:
    x = np.asarray(m_value, dtype=np.float64) * math.log(2.0)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def deterministic_rc_choice(seed: int, epoch: int, idx: int, probability: float) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    local_seed = (
        (int(seed) * 0x9E3779B185EBCA87)
        ^ (int(epoch) * 0xC2B2AE3D27D4EB4F)
        ^ (int(idx) * 0x165667B19E3779F9)
    ) & ((1 << 64) - 1)
    return random.Random(local_seed).random() < probability


def verify_against_training_common() -> dict:
    """Prove the duplicated helpers match the originals. Pays the slow import."""
    try:
        import training_common as tc
    except Exception as exc:                                   # noqa: BLE001
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"}
    rng = random.Random(0)
    seqs = ["".join(rng.choice("ACGTN") for _ in range(5000)) for _ in range(20)]
    for s in seqs:
        assert centered_crop(s, 1000) == tc.centered_crop(s, 1000), "centered_crop"
        assert reverse_complement(s) == tc.reverse_complement(s), "reverse_complement"
    m = np.linspace(-8, 8, 401)
    assert np.allclose(m_to_beta_numpy(m), tc.m_to_beta_numpy(m)), "m_to_beta_numpy"
    for seed in (42, 43):
        for epoch in range(3):
            for idx in range(200):
                assert (deterministic_rc_choice(seed, epoch, idx, 0.5)
                        == tc.deterministic_rc_choice(seed, epoch, idx, 0.5)), "rc_choice"
    return {"checked": True, "sequences": len(seqs), "rc_draws": 2 * 3 * 200}

BASES = {"A": 0, "C": 1, "G": 2, "T": 3}
M_COL, BIN_COL, BETA_COL = "M_Value_Target", "Binary_State_Target", "Median_Beta"
FAITHFULNESS_CHECK = {"deepcpg_trunk_params": 3_997_824}


# ----------------------------------------------------------------------------- data
class OneHotDataset(Dataset):
    """One-hot sequence + targets, using training_common's crop and RC logic."""

    def __init__(self, df: pd.DataFrame, window: int = 1000, training: bool = False,
                 rc_probability: float = 0.5, seed: int = 42) -> None:
        for c in ("Healthy_5000bp_DNA", M_COL, BIN_COL, BETA_COL):
            if c not in df.columns:
                raise ValueError(f"input lacks {c}")
        self.df = df.reset_index(drop=True)
        self.window, self.training = int(window), bool(training)
        self.rc_probability, self.seed, self.epoch = float(rc_probability), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def encode(seq: str) -> np.ndarray:
        x = np.zeros((4, len(seq)), dtype=np.float32)
        for i, b in enumerate(seq):
            j = BASES.get(b)
            if j is not None:          # N and other ambiguity codes stay all-zero
                x[j, i] = 1.0
        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = centered_crop(str(row["Healthy_5000bp_DNA"]), self.window)
        if self.training and deterministic_rc_choice(
                self.seed, self.epoch, idx, self.rc_probability):
            seq = reverse_complement(seq)
        return (torch.from_numpy(self.encode(seq)),
                torch.tensor(float(row[M_COL]), dtype=torch.float32),
                torch.tensor(float(row[BIN_COL]), dtype=torch.float32))


# ----------------------------------------------------------------- architectures
class MaxNorm:
    """Renormalise conv weights to a maximum norm, as CpGenie's W_constraint does."""

    def __init__(self, model: nn.Module, max_value: float = 3.0) -> None:
        self.convs = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
        self.max_value = float(max_value)

    @torch.no_grad()
    def apply(self) -> None:
        for conv in self.convs:
            w = conv.weight
            norm = w.flatten(1).norm(2, dim=1).clamp(min=1e-12)
            factor = (norm.clamp(max=self.max_value) / norm).view(-1, 1, 1)
            w.mul_(factor)


class DualHead(nn.Module):
    """Same head as our models: M-value regression plus a binary logit."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.m = nn.Linear(in_dim, 1)
        self.logit = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.m(x).squeeze(-1), self.logit(x).squeeze(-1)


class CpGenieCNN(nn.Module):
    def __init__(self, window: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 128, 5, padding="same"), nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.Conv1d(128, 256, 5, padding="same"), nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.Conv1d(256, 512, 5, padding="same"), nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 4, window)).flatten(1).shape[1]
        self.dense = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = DualHead(64)

    def forward(self, x):
        return self.head(self.dense(self.features(x)))


class DeepCpGDnaCNN(nn.Module):
    """CnnL2h128 from deepcpg/models/dna.py. No padding, matching Keras 'valid'."""

    def __init__(self, window: int = 1000, dropout: float = 0.0,
                 hidden: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 128, 11), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 3), nn.ReLU(), nn.MaxPool1d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 4, window)).flatten(1).shape[1]
        self.dense = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = DualHead(hidden)
        self.trunk_params = flat * hidden + hidden

    def forward(self, x):
        return self.head(self.dense(self.features(x)))


def build(arch: str, window: int, dropout: float) -> nn.Module:
    if arch == "cpgenie":
        return CpGenieCNN(window, dropout)
    if arch == "deepcpg":
        model = DeepCpGDnaCNN(window, dropout)
        expect = FAITHFULNESS_CHECK["deepcpg_trunk_params"]
        if window == 1000 and model.trunk_params != expect:
            raise SystemExit(
                f"DeepCpG dense-layer parameter count is {model.trunk_params}, "
                f"expected {expect}. The layer spec has been misread; refusing "
                f"to train a model that is not the published architecture.")
        return model
    raise SystemExit(f"unknown arch {arch}")


# ------------------------------------------------------------------------ metrics
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> tuple[dict, pd.DataFrame]:
    """RC-averaged prediction, matching our own test protocol."""
    model.eval()
    m_pred, m_true, logit, binary = [], [], [], []
    for x, m, b in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        mf, lf = model(x)
        mr, lr = model(torch.flip(x, dims=[1, 2]))   # reverse complement
        m_pred.append(((mf + mr) / 2).cpu().numpy())
        logit.append(((lf + lr) / 2).cpu().numpy())
        m_true.append(m.numpy())
        binary.append(b.numpy())
    m_pred = np.concatenate(m_pred); m_true = np.concatenate(m_true)
    logit = np.concatenate(logit); binary = np.concatenate(binary)
    beta_pred = m_to_beta_numpy(m_pred); beta_true = m_to_beta_numpy(m_true)
    prob = 1.0 / (1.0 + np.exp(-logit))
    metrics = {
        "n": int(len(m_true)),
        "m_mae": float(np.mean(np.abs(m_pred - m_true))),
        "m_rmse": float(np.sqrt(np.mean((m_pred - m_true) ** 2))),
        "beta_mae": float(np.mean(np.abs(beta_pred - beta_true))),
        "beta_rmse": float(np.sqrt(np.mean((beta_pred - beta_true) ** 2))),
        "auc": float(roc_auc_score(binary, prob)) if len(np.unique(binary)) == 2 else None,
    }
    preds = pd.DataFrame({"true_m": m_true, "pred_m_rc_avg": m_pred,
                          "true_beta": beta_true, "pred_beta_rc_avg": beta_pred,
                          "binary_true": binary.astype(int),
                          "class_prob_rc_avg": prob})
    return metrics, preds


def train_one(args, dropout: float, lr: float, seed: int,
              train_df, val_df, device) -> tuple[dict, nn.Module]:
    set_seed(seed)
    model = build(args.arch, args.window, dropout).to(device)
    maxnorm = MaxNorm(model, 3.0) if args.arch == "cpgenie" else None
    opt = (torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-6)
           if args.arch == "cpgenie"
           else torch.optim.Adam(model.parameters(), lr=lr))
    bce, huber = nn.BCEWithLogitsLoss(), nn.HuberLoss(delta=1.345)

    tr = OneHotDataset(train_df, args.window, True, args.rc_probability, seed)
    va = OneHotDataset(val_df, args.window, False, 0.0, seed)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val = DataLoader(va, batch_size=args.batch_size * 2, shuffle=False,
                     num_workers=args.num_workers, pin_memory=True)

    best, best_state = math.inf, None
    for epoch in range(1, args.epochs + 1):
        tr.set_epoch(epoch)
        model.train()
        total = 0.0
        for x, m, b in tqdm(trl, desc=f"{args.arch} d={dropout} lr={lr} e{epoch}"):
            x, m, b = x.to(device), m.to(device), b.to(device)
            opt.zero_grad(set_to_none=True)
            mp, lg = model(x)
            loss = huber(mp, m) + bce(lg, b)
            loss.backward()
            opt.step()
            if maxnorm:
                maxnorm.apply()
            total += float(loss)
        vm, _ = evaluate(model, val, device)
        print(f"  epoch {epoch}: train_loss {total/max(1,len(trl)):.4f}  "
              f"val beta_mae {vm['beta_mae']:.4f}  val auc {vm['auc']:.4f}")
        if vm["beta_mae"] < best:
            best = vm["beta_mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"dropout": dropout, "lr": lr, "seed": seed, "val_beta_mae": best}, model



# --------------------------------------------------------- variant-effect mode
# Mirrors scripts/23 score_variants() exactly -- same constants, same guards,
# same emitted schema -- so scripts/20 evaluates these baselines through the
# identical code path as the neural models: identical distance matching,
# identical 1 Mb block bootstrap. CpGenie in particular was designed for this
# task ("Predicting the impact of non-coding variants on DNA methylation"), so
# this is the comparison on its home ground rather than on absolute prediction.
FULL_TARGET_C_INDEX = 2499
FULL_SEQUENCE_LENGTH = 5000
MODEL_WINDOW_SIZE = 1000
CENTER_C_INDEX, CENTER_G_INDEX = 499, 500
PROTECTED_CPG_INDICES = frozenset({CENTER_C_INDEX, CENTER_G_INDEX})


@torch.no_grad()
def _predict_m(model: nn.Module, windows: list[str], device, batch: int) -> np.ndarray:
    """RC-averaged M-value for each window, batched."""
    out = []
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        x = torch.from_numpy(np.stack([OneHotDataset.encode(w) for w in chunk]))
        x = x.to(device)
        mf, _ = model(x)
        mr, _ = model(torch.flip(x, dims=[1, 2]))
        out.append((((mf + mr) / 2).cpu().numpy()))
    return np.concatenate(out) if out else np.zeros(0)


def score_pairs(args, device) -> int:
    pairs = pd.read_csv(args.pairs, low_memory=False)
    probes = pd.read_csv(args.test, low_memory=False)
    records = dict(zip(probes["probeID"].astype(str),
                       probes["Healthy_5000bp_DNA"].astype(str)))
    print(f"pairs {len(pairs)}   probe sequences {len(records)}")

    need = ["Variant_ID", "probeID", "distance_bp", "Ref", "Alt"]
    missing = [c for c in need if c not in pairs.columns]
    if missing:
        print(f"STOP: pairs file lacks {missing}")
        return 1

    counters = {k: 0 for k in ("probe_not_in_splits", "bad_stored_sequence_length",
                               "stored_sequence_not_cpg_centred",
                               "outside_model_window", "alters_target_cpg",
                               "reference_base_mismatch", "window_not_cpg_centred",
                               "scoreable")}
    rows, wt_windows, mut_windows = [], [], []
    for row in pairs.itertuples(index=False):
        probe = str(row.probeID)
        sequence = records.get(probe)
        if sequence is None:
            counters["probe_not_in_splits"] += 1; continue
        sequence = sequence.upper()
        if len(sequence) != FULL_SEQUENCE_LENGTH:
            counters["bad_stored_sequence_length"] += 1; continue
        if sequence[FULL_TARGET_C_INDEX:FULL_TARGET_C_INDEX + 2] != "CG":
            counters["stored_sequence_not_cpg_centred"] += 1; continue
        offset = int(row.distance_bp)
        full_index = FULL_TARGET_C_INDEX + offset
        window_index = CENTER_C_INDEX + offset
        if not 0 <= window_index < MODEL_WINDOW_SIZE:
            counters["outside_model_window"] += 1; continue
        if window_index in PROTECTED_CPG_INDICES:
            counters["alters_target_cpg"] += 1; continue
        if sequence[full_index] != str(row.Ref).upper():
            counters["reference_base_mismatch"] += 1; continue
        mutated = (sequence[:full_index] + str(row.Alt).upper()
                   + sequence[full_index + 1:])
        wt_window = centered_crop(sequence, MODEL_WINDOW_SIZE)
        mut_window = centered_crop(mutated, MODEL_WINDOW_SIZE)
        if wt_window[CENTER_C_INDEX:CENTER_G_INDEX + 1] != "CG":
            counters["window_not_cpg_centred"] += 1; continue
        rec = row._asdict()
        rec["Pair_UID"] = f"{row.Variant_ID}|{probe}"
        rec["Mutation_Window_Index"] = int(window_index)
        rows.append(rec); wt_windows.append(wt_window); mut_windows.append(mut_window)
        counters["scoreable"] += 1

    print(f"counters: {counters}")
    if not rows:
        print("STOP: nothing scoreable")
        return 1
    base = pd.DataFrame(rows).reset_index(drop=True)

    out_root = Path(args.output_dir) / "variant_scoring" / args.stratum / args.arch
    for seed in args.seeds:
        w = Path(args.output_dir) / args.arch / f"seed{seed}" / "weights.pth"
        if not w.exists():
            print(f"STOP: missing {w}; train the seeds first")
            return 1
        model = build(args.arch, args.window, args.dropout).to(device)
        model.load_state_dict(torch.load(w, map_location=device))
        model.eval()
        wt_m = _predict_m(model, wt_windows, device, args.batch_size * 2)
        mut_m = _predict_m(model, mut_windows, device, args.batch_size * 2)
        out = base.copy()
        out["Model"] = args.arch
        out["Seed"] = int(seed)
        out["WT_M_RC_Avg"] = wt_m
        out["MUT_M_RC_Avg"] = mut_m
        out["WT_Beta_RC_Avg"] = m_to_beta_numpy(wt_m)
        out["MUT_Beta_RC_Avg"] = m_to_beta_numpy(mut_m)
        out["Predicted_Delta_M"] = mut_m - wt_m
        out["Predicted_Delta_Beta"] = out["MUT_Beta_RC_Avg"] - out["WT_Beta_RC_Avg"]
        out["Absolute_Delta_M"] = np.abs(out["Predicted_Delta_M"])
        out["Absolute_Delta_Beta"] = np.abs(out["Predicted_Delta_Beta"])
        target = out_root / f"seed{seed}" / "pair_scores.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        print(f"  seed {seed}: {len(out)} pairs -> {target}")
        print(f"    median |delta beta| {out['Absolute_Delta_Beta'].median():.5f}")
    (out_root / "scoring_summary.json").write_text(json.dumps(
        {"arch": args.arch, "pairs_file": args.pairs, "stratum": args.stratum,
         "seeds": list(args.seeds), "counters": counters,
         "scoreable": counters["scoreable"]}, indent=2) + "\n")
    print(f"\nwrote {out_root}")
    return 0

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=("cpgenie", "deepcpg"), required=True)
    p.add_argument("--train", default="data/datafiles/train.csv")
    p.add_argument("--val", default="data/datafiles/val.csv")
    p.add_argument("--test", default="data/datafiles/test.csv")
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--rc-probability", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", action="store_true",
                   help="Tune on the validation split over the authors' own grid.")
    p.add_argument("--verify", action="store_true",
                   help="Check the duplicated helpers against training_common. "
                        "Pays the slow transformers import; run once, not per job.")
    p.add_argument("--max-rows", type=int, default=0, help="smoke testing only")
    p.add_argument("--score-pairs", action="store_true",
                   help="Score variant pairs with trained checkpoints instead of training.")
    p.add_argument("--pairs", default="data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv")
    p.add_argument("--stratum", default="heldout")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--output-dir", default="results/journal/published_baselines")
    args = p.parse_args()

    if args.verify:
        report = verify_against_training_common()
        print(f"training_common verification: {report}")
        if not report.get("checked"):
            print("  could not import training_common; helpers UNVERIFIED")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if args.score_pairs:
        sel = Path(args.output_dir) / args.arch / "selected_hyperparameters.json"
        if sel.exists():
            cfg = json.loads(sel.read_text())
            args.dropout = float(cfg["dropout"])
            print(f"using selected dropout {args.dropout} from {sel}")
        return score_pairs(args, device)

    train_df, val_df = pd.read_csv(args.train), pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)
    if args.max_rows > 0:
        train_df = train_df.head(args.max_rows)
        val_df = val_df.head(max(64, args.max_rows // 4))
        test_df = test_df.head(max(64, args.max_rows // 4))
    print(f"train {len(train_df)}  val {len(val_df)}  test {len(test_df)}")
    model = build(args.arch, args.window, args.dropout)
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()):,} parameters")

    out = Path(args.output_dir) / args.arch
    out.mkdir(parents=True, exist_ok=True)

    if args.grid:
        grid = ([(d, lr) for d in (0.3, 0.5, 0.7) for lr in (0.01, 0.001, 0.0001)]
                if args.arch == "cpgenie"
                else [(d, 0.001) for d in (0.0, 0.3, 0.5)])
        print(f"tuning over {len(grid)} configurations on the validation split")
        rows = []
        for d, lr in grid:
            r, _ = train_one(args, d, lr, args.seed, train_df, val_df, device)
            rows.append(r)
            print(f"  -> dropout {d} lr {lr}: val beta MAE {r['val_beta_mae']:.4f}")
        tab = pd.DataFrame(rows).sort_values("val_beta_mae")
        tab.to_csv(out / "grid_search.csv", index=False)
        best = tab.iloc[0]
        print(f"\nselected: dropout {best['dropout']} lr {best['lr']} "
              f"(val beta MAE {best['val_beta_mae']:.4f})")
        (out / "selected_hyperparameters.json").write_text(json.dumps(
            {"dropout": float(best["dropout"]), "lr": float(best["lr"]),
             "selected_on": "validation beta MAE", "grid": [list(g) for g in grid]},
            indent=2) + "\n")
        return 0

    res, model = train_one(args, args.dropout, args.lr, args.seed,
                           train_df, val_df, device)
    te = OneHotDataset(test_df, args.window, False, 0.0, args.seed)
    tel = DataLoader(te, batch_size=args.batch_size * 2, shuffle=False,
                     num_workers=args.num_workers, pin_memory=True)
    metrics, preds = evaluate(model, tel, device)
    metrics.update({"arch": args.arch, "seed": args.seed, "dropout": args.dropout,
                    "lr": args.lr, "window": args.window, "epochs": args.epochs,
                    "val_beta_mae": res["val_beta_mae"],
                    "parameters": int(sum(p.numel() for p in model.parameters()))})
    seed_dir = out / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(seed_dir / "predictions.csv", index=False)
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(model.state_dict(), seed_dir / "weights.pth")
    print(f"\nTEST  beta MAE {metrics['beta_mae']:.4f}   M MAE {metrics['m_mae']:.4f}"
          f"   AUC {metrics['auc']:.4f}   n={metrics['n']}")
    print(f"wrote {seed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
