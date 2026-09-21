#!/usr/bin/env python3
"""Classical sequence baselines on the same splits and the same evaluator as the neural
models, so their variant-effect numbers have a referent. Two models: composition (GC
fraction, CpG count, CpG observed/expected) and a ridge regression on reverse-
complement-collapsed k-mer counts for k = 1..K. Both are deterministic single fits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matched_background_utils import (  # noqa: E402
    CENTER_C_INDEX,
    CENTER_G_INDEX,
    PROTECTED_CPG_INDICES,
    reverse_complement,
)


def centered_crop(seq: str, window_size: int) -> str:
    seq = seq.upper()
    if window_size <= 0 or window_size > len(seq):
        raise ValueError(f"window_size={window_size} is invalid for "
                         f"sequence length {len(seq)}")
    center_right = len(seq) // 2
    start = center_right - (window_size // 2)
    return seq[start:start + window_size]


def m_to_beta_numpy(m_value: np.ndarray) -> np.ndarray:
    x = np.asarray(m_value, dtype=np.float64) * np.log(2.0)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def verify_against_training_common() -> dict:
    """Assert the local helpers match scripts/training_common.py, if importable."""
    try:
        import training_common as tc
    except Exception as exc:  # noqa: BLE001 - any import failure is the same case
        return {"checked": False, "reason": type(exc).__name__}
    rng = np.random.default_rng(3)
    seq = "".join(rng.choice(list("ACGT"), FULL_SEQUENCE_LENGTH))
    for window in (100, 400, MODEL_WINDOW_SIZE, 2000):
        if centered_crop(seq, window) != tc.centered_crop(seq, window):
            raise RuntimeError(
                f"centered_crop in scripts/23 disagrees with training_common at "
                f"window={window}. The baseline would be scoring a different "
                f"sequence than the models -- fix before trusting any number.")
    m = rng.normal(0, 4, 5000)
    delta = float(np.max(np.abs(m_to_beta_numpy(m) - tc.m_to_beta_numpy(m))))
    if delta > 1e-12:
        raise RuntimeError(f"m_to_beta_numpy disagrees with training_common "
                           f"(max {delta:g})")
    return {"checked": True, "m_to_beta_max_abs_diff": delta}

LOGGER = logging.getLogger("silentmethyl.baselines")

FULL_TARGET_C_INDEX = 2499
FULL_SEQUENCE_LENGTH = 5000
MODEL_WINDOW_SIZE = 1000

BASE_CODE = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    BASE_CODE[ord(_b)] = _i
    BASE_CODE[ord(_b.lower())] = _i

ALPHA_GRID = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]


class KmerEncoder:
    """RC-collapsed k-mer counts for k = 1..k_max.

    A k-mer and its reverse complement share a column. That makes the whole
    feature map exactly strand-invariant, so RC-averaging -- which the neural
    models need -- is provably a no-op here rather than an approximation. The
    invariance is asserted at startup, not assumed.
    """

    def __init__(self, k_max: int):
        if not 1 <= k_max <= 8:
            raise ValueError("k_max must be in 1..8")
        self.k_max = k_max
        self.canonical: dict[int, np.ndarray] = {}
        self.offset: dict[int, int] = {}
        self.n_features = 0
        for k in range(1, k_max + 1):
            size = 4 ** k
            codes = np.arange(size, dtype=np.int64)
            rc = np.zeros(size, dtype=np.int64)
            tmp = codes.copy()
            for _ in range(k):
                rc = rc * 4 + (3 - (tmp % 4))
                tmp //= 4
            canon = np.minimum(codes, rc)
            uniq, inverse = np.unique(canon, return_inverse=True)
            self.canonical[k] = inverse.astype(np.int64)
            self.offset[k] = self.n_features
            self.n_features += len(uniq)

    def names(self) -> list[str]:
        return [f"k{k}_{j}" for k in range(1, self.k_max + 1)
                for j in range(self._width(k))]

    def _width(self, k: int) -> int:
        return int(self.canonical[k].max()) + 1

    @staticmethod
    def encode_bases(seq: str) -> np.ndarray:
        return BASE_CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]

    def counts(self, seq: str) -> np.ndarray:
        """Dense count vector. k-mers containing a non-ACGT base are dropped."""
        bases = self.encode_bases(seq)
        out = np.zeros(self.n_features, dtype=np.float64)
        valid_base = bases >= 0
        for k in range(1, self.k_max + 1):
            n = len(bases) - k + 1
            if n <= 0:
                continue
            code = np.zeros(n, dtype=np.int64)
            ok = np.ones(n, dtype=bool)
            for j in range(k):
                code = code * 4 + np.maximum(bases[j:j + n], 0)
                ok &= valid_base[j:j + n]
            code = code[ok]
            if code.size == 0:
                continue
            cols = self.canonical[k][code] + self.offset[k]
            out += np.bincount(cols, minlength=self.n_features)
        return out


def composition_features(seq: str) -> np.ndarray:
    """GC fraction, CpG count, CpG observed/expected. The floor any model must clear."""
    bases = KmerEncoder.encode_bases(seq)
    valid = bases >= 0
    n = int(valid.sum())
    if n == 0:
        return np.zeros(3, dtype=np.float64)
    c = int((bases == 1).sum())
    g = int((bases == 2).sum())
    cpg = int(((bases[:-1] == 1) & (bases[1:] == 2)).sum())
    expected = (c * g) / n if c and g else 0.0
    return np.array([(c + g) / n, float(cpg),
                     (cpg / expected) if expected > 0 else 0.0], dtype=np.float64)


class FeatureMap:
    """Uniform interface over the two baselines."""

    def __init__(self, name: str, k_max: int):
        self.name = name
        if name == "kmer_ridge":
            self.encoder = KmerEncoder(k_max)
            self.n_features = self.encoder.n_features
        elif name == "composition":
            self.encoder = None
            self.n_features = 3
        else:
            raise ValueError(f"unknown baseline {name!r}")

    def __call__(self, seq: str) -> np.ndarray:
        if self.encoder is not None:
            return self.encoder.counts(seq)
        return composition_features(seq)


class SufficientStats:
    """X'X, X'y, sum(X), sum(y), y'y, n -- everything ridge needs, in one pass."""

    def __init__(self, d: int):
        self.d = d
        self.xtx = np.zeros((d, d), dtype=np.float64)
        self.xty = np.zeros(d, dtype=np.float64)
        self.xsum = np.zeros(d, dtype=np.float64)
        self.ysum = 0.0
        self.yty = 0.0
        self.n = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.xsum += x.sum(axis=0)
        self.ysum += float(y.sum())
        self.yty += float(y @ y)
        self.n += len(y)

    def standardised(self, mean: np.ndarray, scale: np.ndarray, ybar: float):
        """Gram and cross-product for centred/scaled X and centred y."""
        n = self.n
        gram = self.xtx - np.outer(self.xsum, mean) - np.outer(mean, self.xsum) \
            + n * np.outer(mean, mean)
        gram /= np.outer(scale, scale)
        cross = (self.xty - mean * self.ysum - ybar * self.xsum + n * mean * ybar) / scale
        ss = self.yty - 2 * ybar * self.ysum + n * ybar ** 2
        return gram, cross, ss


def stream_split(path: Path, feature_map: FeatureMap, chunk_size: int,
                 limit: int = 0):
    """Yield (features, M-target, frame) chunk by chunk. Sequences never accumulate."""
    needed = ["probeID", "Healthy_5000bp_DNA", "M_Value_Target",
              "Median_Beta", "Binary_State_Target"]
    header = pd.read_csv(path, nrows=0)
    absent = [c for c in needed if c not in header.columns]
    if absent:
        raise SystemExit(f"{path} is missing {absent}")
    seen = 0
    for chunk in pd.read_csv(path, usecols=needed, chunksize=chunk_size):
        if limit and seen >= limit:
            return
        if limit:
            chunk = chunk.head(limit - seen)
        seqs = chunk["Healthy_5000bp_DNA"].astype(str)
        bad = (seqs.str.len() != FULL_SEQUENCE_LENGTH)
        if bad.any():
            raise SystemExit(f"{path}: {int(bad.sum())} sequences are not "
                             f"{FULL_SEQUENCE_LENGTH} bp")
        windows = [centered_crop(s, MODEL_WINDOW_SIZE) for s in seqs]
        x = np.vstack([feature_map(w) for w in windows])
        y = chunk["M_Value_Target"].to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise SystemExit(f"{path} contains non-finite M_Value_Target")
        seen += len(chunk)
        yield x, y, chunk.reset_index(drop=True)


def fit_ridge(gram: np.ndarray, cross: np.ndarray, alpha: float) -> np.ndarray:
    d = gram.shape[0]
    return np.linalg.solve(gram + alpha * np.eye(d), cross)


def mse_from_stats(w: np.ndarray, gram: np.ndarray, cross: np.ndarray,
                   ss: float, n: int) -> float:
    """Exact MSE of a linear fit from sufficient statistics -- no re-reading."""
    return float((w @ gram @ w - 2 * w @ cross + ss) / n)


def train_baseline(name: str, args) -> dict:
    feature_map = FeatureMap(name, args.k_max)
    LOGGER.info("[%s] %d features", name, feature_map.n_features)

    stats = {}
    for split in ("train", "val"):
        path = Path(args.split_template.format(split=split))
        acc = SufficientStats(feature_map.n_features)
        rows = 0
        for x, y, _ in stream_split(path, feature_map, args.chunk_size, args.limit):
            acc.update(x, y)
            rows += len(y)
            if rows % (20 * args.chunk_size) == 0:
                LOGGER.info("[%s] %s: %d rows", name, split, rows)
        LOGGER.info("[%s] %s: %d rows accumulated", name, split, acc.n)
        stats[split] = acc

    train = stats["train"]
    mean = train.xsum / train.n
    var = np.diag(train.xtx) / train.n - mean ** 2
    var = np.maximum(var, 0.0)
    scale = np.sqrt(var)
    constant = scale <= 0
    if constant.any():
        LOGGER.warning("[%s] %d constant features held out of the fit",
                       name, int(constant.sum()))
        scale = np.where(constant, 1.0, scale)
    ybar = train.ysum / train.n

    g_tr, c_tr, ss_tr = train.standardised(mean, scale, ybar)
    g_va, c_va, ss_va = stats["val"].standardised(mean, scale, ybar)

    search = []
    for alpha in ALPHA_GRID:
        w = fit_ridge(g_tr, c_tr, alpha)
        w[constant] = 0.0
        search.append({
            "alpha": alpha,
            "train_mse": mse_from_stats(w, g_tr, c_tr, ss_tr, train.n),
            "val_mse": mse_from_stats(w, g_va, c_va, ss_va, stats["val"].n),
        })
        LOGGER.info("[%s] alpha=%-8g train_mse=%.5f val_mse=%.5f", name, alpha,
                    search[-1]["train_mse"], search[-1]["val_mse"])
    best = min(search, key=lambda r: r["val_mse"])
    LOGGER.info("[%s] selected alpha=%g on validation MSE", name, best["alpha"])
    w = fit_ridge(g_tr, c_tr, best["alpha"])
    w[constant] = 0.0

    return {"name": name, "feature_map": feature_map, "w": w, "mean": mean,
            "scale": scale, "ybar": ybar, "alpha": float(best["alpha"]),
            "alpha_search": search, "n_train": int(train.n),
            "n_val": int(stats["val"].n)}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - model["mean"]) / model["scale"]) @ model["w"] + model["ybar"]


def evaluate_absolute(model: dict, args) -> dict:
    path = Path(args.split_template.format(split="test"))
    preds, truth, beta_true, binary = [], [], [], []
    for x, y, frame in stream_split(path, model["feature_map"],
                                    args.chunk_size, args.limit):
        preds.append(predict(model, x))
        truth.append(y)
        beta_true.append(frame["Median_Beta"].to_numpy(dtype=float))
        binary.append(frame["Binary_State_Target"].to_numpy(dtype=float))
    m_pred = np.concatenate(preds)
    m_true = np.concatenate(truth)
    b_true = np.concatenate(beta_true)
    b_pred = m_to_beta_numpy(m_pred)
    y_bin = np.concatenate(binary).astype(int)

    metrics = {
        "n_test_loci": int(len(m_true)),
        "m_rmse": float(np.sqrt(np.mean((m_true - m_pred) ** 2))),
        "m_mae": float(np.mean(np.abs(m_true - m_pred))),
        "m_pearson": float(pearsonr(m_true, m_pred).statistic),
        "m_spearman": float(spearmanr(m_true, m_pred).statistic),
        "beta_rmse": float(np.sqrt(np.mean((b_true - b_pred) ** 2))),
        "beta_mae": float(np.mean(np.abs(b_true - b_pred))),
        "auc": (float(roc_auc_score(y_bin, m_pred))
                if len(np.unique(y_bin)) == 2 else float("nan")),
    }
    return metrics


def load_probe_windows(split_template: str, probe_ids: set) -> dict:
    """probeID -> 5,000-bp sequence, filtered on arrival."""
    records = {}
    for split in ("train", "val", "test"):
        path = Path(split_template.format(split=split))
        if not path.is_file():
            raise FileNotFoundError(path)
        for chunk in pd.read_csv(path, usecols=["probeID", "Healthy_5000bp_DNA"],
                                 chunksize=20_000):
            chunk = chunk[chunk["probeID"].astype(str).isin(probe_ids)]
            for probe, seq in zip(chunk["probeID"].astype(str),
                                  chunk["Healthy_5000bp_DNA"].astype(str)):
                records[probe] = seq.upper()
    return records


def score_variants(model: dict, pairs: pd.DataFrame, records: dict,
                   counters: dict) -> pd.DataFrame:
    fmap = model["feature_map"]
    rows, wt_m, mut_m = [], [], []

    for row in pairs.itertuples(index=False):
        probe = str(row.probeID)
        sequence = records.get(probe)
        if sequence is None:
            counters["probe_not_in_splits"] += 1
            continue
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
        if sequence[full_index] != str(row.Ref).upper():
            counters["reference_base_mismatch"] += 1
            continue

        mutated = (sequence[:full_index] + str(row.Alt).upper()
                   + sequence[full_index + 1:])
        wt_window = centered_crop(sequence, MODEL_WINDOW_SIZE)
        mut_window = centered_crop(mutated, MODEL_WINDOW_SIZE)
        if wt_window[CENTER_C_INDEX:CENTER_G_INDEX + 1] != "CG":
            counters["window_not_cpg_centred"] += 1
            continue

        record = row._asdict()
        record["Pair_UID"] = f"{row.Variant_ID}|{probe}"
        record["Mutation_Window_Index"] = int(window_index)
        rows.append(record)
        wt_m.append(predict(model, fmap(wt_window)[None, :])[0])
        mut_m.append(predict(model, fmap(mut_window)[None, :])[0])
        counters["scoreable"] += 1

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).reset_index(drop=True)
    wt_m = np.asarray(wt_m)
    mut_m = np.asarray(mut_m)
    out["Model"] = model["name"]
    out["Seed"] = -1
    out["WT_M_RC_Avg"] = wt_m
    out["MUT_M_RC_Avg"] = mut_m
    out["WT_Beta_RC_Avg"] = m_to_beta_numpy(wt_m)
    out["MUT_Beta_RC_Avg"] = m_to_beta_numpy(mut_m)
    out["Predicted_Delta_M"] = mut_m - wt_m
    out["Predicted_Delta_Beta"] = out["MUT_Beta_RC_Avg"] - out["WT_Beta_RC_Avg"]
    out["Absolute_Delta_M"] = np.abs(out["Predicted_Delta_M"])
    out["Absolute_Delta_Beta"] = np.abs(out["Predicted_Delta_Beta"])
    return out


def run_self_checks(k_max: int) -> dict:
    """Three properties this script's conclusions depend on. Verified, not assumed."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 4000))
    enc = KmerEncoder(k_max)

    if centered_crop("N" * FULL_SEQUENCE_LENGTH, MODEL_WINDOW_SIZE) != "N" * MODEL_WINDOW_SIZE:
        raise RuntimeError("centered_crop did not return a 1000-bp window")
    if FULL_TARGET_C_INDEX - (FULL_SEQUENCE_LENGTH // 2 - MODEL_WINDOW_SIZE // 2) != CENTER_C_INDEX:
        raise RuntimeError("window geometry disagrees with scripts/19")

    fwd = enc.counts(seq)
    rev = enc.counts(reverse_complement(seq))
    rc_max_abs = float(np.max(np.abs(fwd - rev)))
    if rc_max_abs > 1e-9:
        raise RuntimeError(f"k-mer features are not RC-invariant (max {rc_max_abs})")

    comp_fwd = composition_features(seq)
    comp_rev = composition_features(reverse_complement(seq))
    comp_max_abs = float(np.max(np.abs(comp_fwd - comp_rev)))
    if comp_max_abs > 1e-9:
        raise RuntimeError(f"composition features are not RC-invariant ({comp_max_abs})")

    return {"rc_invariance_max_abs_diff_kmer": rc_max_abs,
            "rc_invariance_max_abs_diff_composition": comp_max_abs,
            "n_kmer_features": int(enc.n_features),
            "training_common_agreement": verify_against_training_common()}


def atomic_write(frame_or_payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if isinstance(frame_or_payload, pd.DataFrame):
        frame_or_payload.to_csv(tmp, index=False)
    else:
        with tmp.open("w") as fh:
            json.dump(frame_or_payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("absolute", "variant", "both"), default="both")
    p.add_argument("--models", nargs="+", default=["composition", "kmer_ridge"],
                   choices=("composition", "kmer_ridge"))
    p.add_argument("--k-max", type=int, default=6)
    p.add_argument("--split-template", default="data/datafiles/{split}.csv")
    p.add_argument("--input-csv", type=Path,
                   default=Path("data/external/genoa_meqtl/scoring/"
                                "genoa_scoring_input_heldout.csv"))
    p.add_argument("--stratum", default="heldout",
                   choices=("heldout", "model_visible"))
    p.add_argument("--chunk-size", type=int, default=5_000)
    p.add_argument("--limit", type=int, default=0,
                   help="Smoke test: use only the first N rows of each split.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/sequence_baselines"))
    p.add_argument("--variant-output-dir", type=Path, default=None,
                   help="Defaults to <output-dir>/variant_scoring, laid out like "
                        "scripts/19 so scripts/20 reads it unchanged.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    checks = run_self_checks(args.k_max)
    LOGGER.info("self-checks passed: %s", checks)

    variant_dir = args.variant_output_dir or (args.output_dir / "variant_scoring")
    absolute_rows, variant_written = [], []

    for name in args.models:
        model = train_baseline(name, args)

        if args.task in ("absolute", "both"):
            metrics = evaluate_absolute(model, args)
            LOGGER.info("[%s] test: m_rmse=%.4f m_pearson=%.4f auc=%.4f",
                        name, metrics["m_rmse"], metrics["m_pearson"], metrics["auc"])
            absolute_rows.append({"model": name, "alpha": model["alpha"],
                                  "n_features": model["feature_map"].n_features,
                                  **metrics})

        if args.task in ("variant", "both"):
            if not args.input_csv.is_file():
                raise SystemExit(f"scoring input not found: {args.input_csv}")
            pairs = pd.read_csv(args.input_csv)
            need = ["Variant_ID", "chr", "Position_1based", "Ref", "Alt",
                    "probeID", "probe_split", "distance_bp"]
            absent = [c for c in need if c not in pairs.columns]
            if absent:
                raise SystemExit(f"{args.input_csv} is missing {absent}")
            expected = {"heldout": {"test"},
                        "model_visible": {"train", "val"}}[args.stratum]
            actual = set(pairs["probe_split"].astype(str).unique())
            if not actual <= expected:
                raise SystemExit(f"--stratum {args.stratum} expects probe_split in "
                                 f"{sorted(expected)}, file has {sorted(actual)}")
            if args.limit:
                pairs = pairs.head(args.limit)

            records = load_probe_windows(args.split_template,
                                         set(pairs["probeID"].astype(str)))
            LOGGER.info("[%s] materialised %d target probes", name, len(records))
            counters = {k: 0 for k in (
                "scoreable", "probe_not_in_splits", "bad_stored_sequence_length",
                "stored_sequence_not_cpg_centred", "outside_model_window",
                "alters_target_cpg", "reference_base_mismatch",
                "window_not_cpg_centred")}
            scored = score_variants(model, pairs, records, counters)
            LOGGER.info("[%s] variant counters: %s", name, counters)
            if scored.empty:
                raise SystemExit(f"[{name}] no pair survived construction")
            target = variant_dir / args.stratum / name / "seed-1" / "pair_scores.csv"
            atomic_write(scored, target)
            variant_written.append({"model": name, "rows": int(len(scored)),
                                    "file": str(target), "counters": counters})
            LOGGER.info("[%s] -> %s (%d rows)", name, target, len(scored))

    if absolute_rows:
        frame = pd.DataFrame(absolute_rows)
        atomic_write(frame, args.output_dir / "absolute_prediction_metrics.csv")
        print()
        print("=" * 74)
        print("absolute methylation prediction -- test split, same splits as the "
              "neural models")
        print(frame[["model", "n_features", "alpha", "m_rmse", "m_pearson",
                     "m_spearman", "beta_mae", "auc"]].to_string(index=False))
        print("=" * 74)

    atomic_write(
        {
            "analysis": "classical sequence baselines on SilentMethyl splits",
            "purpose": "classical baselines: the referent for the variant-effect numbers",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": args.models,
            "k_max": args.k_max,
            "alpha_grid": ALPHA_GRID,
            "self_checks": checks,
            "splits": args.split_template,
            "absolute": absolute_rows,
            "variant_scoring": variant_written,
            "interpretation": (
                "These baselines are the referent for signed rho and direction "
                "agreement. If kmer_ridge matches the neural models on variant "
                "effects, the finding is that sequence-only allelic methylation "
                "prediction is much harder than reported, and DNABERT-2 "
                "pretraining is not what closes the gap. If it does not, the "
                "architecture is earning its keep. Report whichever is true."),
            "fairness_note": (
                "Trained on the same splits, same target column, same 1,000-bp "
                "window, same target-CpG exclusions, and scored through "
                "scripts/20. No published cross-tissue weights are involved, so "
                "no tissue handicap favours either side."),
        },
        args.output_dir / "run_summary.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
