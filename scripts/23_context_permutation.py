#!/usr/bin/env python3
"""Measure whether the epigenomic context vector carries allele information. The context
tower receives the same vector for both alleles, so structurally it can rescale a
sequence-driven effect but not create one; this scores the same pairs under substituted
context (shuffled, per-tissue, cross-tissue mean) and compares how much methylation
levels move against how much variant effects move. The verdict is decided on normalised
MAE, with Pearson reported alongside because levels are bimodal and a high Pearson on
them is cheap.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
SCORER_PATH = HERE / "20_variant_scoring.py"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from training_common import (  # noqa: E402
    MISSING_FEATURES,
    PHYLOP_1,
    PHYLOP_2,
    TABULAR_FEATURES,
)

LOGGER = logging.getLogger("silentmethyl.ctxperm")

SCHEMES = ("identity", "shuffle", "median", "xtissue_mean",
           "tissue_ColonTransverse", "tissue_KidneyCortex", "tissue_Lung")

TISSUE_CONTEXT = {
    "BreastEpithelium": "data/datafiles_breast_epithelium/test.csv",
    "ColonTransverse": "data/datafiles_multitissue/ColonTransverse/test.csv",
    "KidneyCortex": "data/datafiles_multitissue/KidneyCortex/test.csv",
    "Lung": "data/datafiles_multitissue/Lung/test.csv",
}
LEVEL_COL = "WT_M_RC_Avg"
DELTA_COL = "Predicted_Delta_M"


def load_scorer():
    """Import scripts/20 by path; its numeric stem makes it unimportable normally."""
    if not SCORER_PATH.is_file():
        raise SystemExit(f"STOP: {SCORER_PATH} not found -- run from the repo root")
    spec = importlib.util.spec_from_file_location("variant_scoring", SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["variant_scoring"] = module
    spec.loader.exec_module(module)
    return module


def load_tissue_contexts(probe_ids: list[str]) -> dict:
    """probeID -> (features, missing) for each tissue, aligned to TABULAR_FEATURES.

    Only probes present in ALL four builds are usable; a locus missing from one
    tissue cannot have a cross-tissue mean and would silently bias the rung if
    filled some other way, so it is reported and excluded from the swap.
    """
    wanted = set(probe_ids)
    out = {}
    for tissue, path in TISSUE_CONTEXT.items():
        d = pd.read_csv(path, usecols=["probeID"] + TABULAR_FEATURES + MISSING_FEATURES)
        d = d[d["probeID"].astype(str).isin(wanted)].set_index("probeID")
        out[tissue] = d
        LOGGER.info("context source %-18s %6d of %d requested probes",
                    tissue, len(d), len(wanted))
    shared = set.intersection(*[set(d.index.astype(str)) for d in out.values()])
    LOGGER.info("probes with context in ALL four tissues: %d of %d",
                len(shared), len(wanted))
    return {"by_tissue": out, "shared": shared}


def _tissue_tensor(ctx: dict, tissue: str, probe_ids: list[str],
                   tab: torch.Tensor, missing: torch.Tensor):
    """Replace each row's context with the same probe's context in `tissue`.

    Rows whose probe lacks a context in every tissue keep their native vector;
    they are counted so the caller can assert the perturbation actually applied.
    """
    src = ctx["by_tissue"][tissue]
    shared = ctx["shared"]
    new_tab, new_missing = tab.clone(), missing.clone()
    hits = 0
    feat = src[TABULAR_FEATURES].to_numpy(dtype="float32")
    miss = src[MISSING_FEATURES].to_numpy(dtype="float32")
    index = {pid: i for i, pid in enumerate(src.index.astype(str))}
    for row, pid in enumerate(probe_ids):
        if pid in shared and pid in index:
            j = index[pid]
            new_tab[row] = torch.from_numpy(feat[j])
            new_missing[row] = torch.from_numpy(miss[j])
            hits += 1
    return new_tab, new_missing, hits


def _xtissue_mean_tensor(ctx: dict, probe_ids: list[str],
                         tab: torch.Tensor, missing: torch.Tensor):
    """Per-locus mean context across the four tissues.

    The mean is over tissues at the SAME locus, not over loci -- locus identity
    is preserved and only tissue identity is averaged away. Missing flags are
    OR-ed: a feature absent in any tissue cannot contribute to an honest mean.
    """
    tissues = list(TISSUE_CONTEXT)
    shared = ctx["shared"]
    frames = {t: ctx["by_tissue"][t] for t in tissues}
    index = {t: {pid: i for i, pid in enumerate(frames[t].index.astype(str))}
             for t in tissues}
    feats = {t: frames[t][TABULAR_FEATURES].to_numpy(dtype="float32") for t in tissues}
    miss = {t: frames[t][MISSING_FEATURES].to_numpy(dtype="float32") for t in tissues}
    new_tab, new_missing = tab.clone(), missing.clone()
    hits = 0
    for row, pid in enumerate(probe_ids):
        if pid not in shared:
            continue
        stack, mstack = [], []
        for t in tissues:
            j = index[t].get(pid)
            if j is None:
                break
            stack.append(feats[t][j]); mstack.append(miss[t][j])
        else:
            new_tab[row] = torch.from_numpy(np.mean(stack, axis=0))
            new_missing[row] = torch.from_numpy(
                (np.max(mstack, axis=0) > 0.5).astype("float32"))
            hits += 1
    return new_tab, new_missing, hits


def permute(tab: torch.Tensor, missing: torch.Tensor, scheme: str,
            rng: np.random.Generator, probe_ids=None, ctx=None):
    """Return (tab, missing) under one scheme. Never modifies the inputs.

    `missing` is permuted WITH `tab`, not independently: the missingness flags
    describe the values they accompany, and separating them would hand the model
    a combination it never saw in training, which is a different experiment.
    """
    if scheme == "identity":
        return tab.clone(), missing.clone()
    if scheme == "shuffle":
        n = tab.shape[0]
        if n < 2:
            raise SystemExit("STOP: shuffle needs at least 2 rows")
        order = rng.permutation(n)
        fixed = np.flatnonzero(order == np.arange(n))
        if len(fixed) > 1:
            order[fixed] = order[rng.permutation(fixed)]
        idx = torch.as_tensor(order, dtype=torch.long)
        return tab[idx].clone(), missing[idx].clone()
    if scheme == "median":
        med = tab.median(dim=0, keepdim=True).values
        med_missing = (missing.float().mean(dim=0, keepdim=True) > 0.5).to(missing.dtype)
        return (med.expand_as(tab).clone(), med_missing.expand_as(missing).clone())
    if scheme.startswith("tissue_") or scheme == "xtissue_mean":
        if ctx is None or probe_ids is None:
            raise SystemExit(f"STOP: {scheme} needs the tissue context tables")
        if scheme == "xtissue_mean":
            t, m, hits = _xtissue_mean_tensor(ctx, probe_ids, tab, missing)
        else:
            tissue = scheme.removeprefix("tissue_")
            if tissue not in TISSUE_CONTEXT:
                raise SystemExit(f"STOP: unknown tissue {tissue!r}")
            t, m, hits = _tissue_tensor(ctx, tissue, probe_ids, tab, missing)
        LOGGER.info("  %s: context replaced for %d of %d rows (%.1f%%)",
                    scheme, hits, len(probe_ids), 100 * hits / max(1, len(probe_ids)))
        if hits < 0.9 * len(probe_ids):
            raise SystemExit(
                f"STOP: {scheme} only replaced {hits}/{len(probe_ids)} contexts; "
                "the tissue builds do not cover this cohort well enough to make "
                "the rung interpretable")
        return t, m
    raise SystemExit(f"STOP: unknown scheme {scheme!r}")


def agreement(reference: np.ndarray, other: np.ndarray) -> dict:
    keep = np.isfinite(reference) & np.isfinite(other)
    if keep.sum() < 10:
        return {"n": int(keep.sum())}
    a, b = reference[keep], other[keep]
    out = {
        "n": int(keep.sum()),
        "pearson": float(stats.pearsonr(a, b).statistic),
        "spearman": float(stats.spearmanr(a, b).statistic),
        "mae": float(np.abs(a - b).mean()),
        "sd_reference": float(a.std()),
    }
    nz = (np.sign(a) != 0) & (np.sign(b) != 0)
    if nz.sum() >= 10:
        out["sign_agreement"] = float((np.sign(a[nz]) == np.sign(b[nz])).mean())
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-csv", type=Path,
                   default=Path("data/external/egtex_breast/scoring/"
                                "egtex_scoring_input_heldout.csv"))
    p.add_argument("--stratum", default="heldout", choices=("heldout", "model_visible"))
    p.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    p.add_argument("--seed", type=int, default=42,
                   help="model seed; context permutation is a mechanism check, "
                        "so one seed is enough unless the result is marginal")
    p.add_argument("--weights-template",
                   default="checkpoints_journal/seed{seed}/{model}/best_weights.pth")
    p.add_argument("--split-template", default="data/datafiles/{split}.csv")
    p.add_argument("--hm450-manifest", type=Path,
                   default=Path("data/HM450.hg38.manifest.tsv.gz"))
    p.add_argument("--include-masked-probes", action="store_true")
    p.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    p.add_argument("--local-model-dir", default="./dnabert2_local")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=20_000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="smoke test only")
    p.add_argument("--permutation-seed", type=int, default=1234)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/context_permutation"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    ev = load_scorer()

    if "identity" not in args.schemes:
        raise SystemExit("STOP: 'identity' is the reference every other scheme is "
                         "compared against; it cannot be omitted")

    if not args.input_csv.is_file():
        raise SystemExit(f"STOP: scoring input not found: {args.input_csv}")
    pairs = pd.read_csv(args.input_csv)
    absent = [c for c in ev.REQUIRED_INPUT_COLUMNS if c not in pairs.columns]
    if absent:
        raise SystemExit(f"STOP: {args.input_csv} is missing {absent}")

    effect = ev.resolve_column(pairs, "auto", ev.EFFECT_COLUMN_CANDIDATES, "effect")
    pvalue = ev.resolve_column(pairs, "auto", ev.PVALUE_COLUMN_CANDIDATES, "pvalue")
    if effect != ev.CANONICAL_EFFECT:
        pairs[ev.CANONICAL_EFFECT] = pairs[effect]
    if pvalue != ev.CANONICAL_PVALUE:
        pairs[ev.CANONICAL_PVALUE] = pairs[pvalue]

    pairs, dropped = ev.apply_probe_qc(pairs, args.hm450_manifest,
                                       args.include_masked_probes)
    LOGGER.info("HM450 QC dropped %d; %d pairs remain", dropped, len(pairs))
    if args.limit > 0:
        pairs = pairs.head(args.limit).copy()
        LOGGER.warning("SMOKE TEST: limited to %d pairs", len(pairs))

    records = ev.load_probe_records(args.split_template,
                                    set(pairs["probeID"].astype(str)))
    device = ev.choose_device(args.device)
    tokenizer = ev.get_tokenizer(args.model_path)

    counters = {k: 0 for k in (
        "scoreable", "probe_not_in_splits", "bad_stored_sequence_length",
        "stored_sequence_not_cpg_centred", "outside_model_window",
        "alters_target_cpg", "reference_base_mismatch",
        "unexpected_window_difference", "window_not_cpg_centred")}
    prepared = []
    for i in range(0, len(pairs), args.chunk_size):
        built = ev.build_chunk(pairs.iloc[i:i + args.chunk_size], records, counters)
        if built is not None:
            prepared.append(built)
    if not prepared:
        raise SystemExit("STOP: no pair survived construction")
    LOGGER.info("construction counters: %s", counters)

    sizes = [t.shape[0] for _, _, _, t, _ in prepared]
    all_tab = torch.cat([t for _, _, _, t, _ in prepared], dim=0)
    all_missing = torch.cat([m for _, _, _, _, m in prepared], dim=0)
    LOGGER.info("context matrix: %s", tuple(all_tab.shape))

    probe_ids = pd.concat([c for c, _, _, _, _ in prepared],
                          ignore_index=True)["probeID"].astype(str).tolist()
    if len(probe_ids) != all_tab.shape[0]:
        raise SystemExit(f"STOP: {len(probe_ids)} probe ids for "
                         f"{all_tab.shape[0]} context rows -- row alignment is broken")
    needs_tissue = any(sc.startswith("tissue_") or sc == "xtissue_mean"
                       for sc in args.schemes)
    ctx = load_tissue_contexts(probe_ids) if needs_tissue else None

    weights = Path(args.weights_template.format(seed=args.seed, model="fusion"))
    if not weights.is_file():
        raise SystemExit(f"STOP: checkpoint not found: {weights}")
    weights_sha = ev.sha256_file(weights)
    model = ev.build_model("fusion", args, args.seed, weights, device)

    rng = np.random.default_rng(args.permutation_seed)
    scored: dict[str, pd.DataFrame] = {}
    for scheme in args.schemes:
        LOGGER.info("scoring under scheme=%s", scheme)
        p_tab, p_missing = permute(all_tab, all_missing, scheme, rng,
                                   probe_ids=probe_ids, ctx=ctx)
        if scheme.startswith("tissue_") or scheme == "xtissue_mean":
            pidx = [TABULAR_FEATURES.index(PHYLOP_1), TABULAR_FEATURES.index(PHYLOP_2)]
            shifted = float((p_tab[:, pidx] - all_tab[:, pidx]).abs().max())
            LOGGER.info("  PhyloP max shift under %s: %.3e (expected 0)", scheme, shifted)
            if shifted > 1e-5:
                raise SystemExit(
                    f"STOP: {scheme} moved PhyloP by {shifted:.3e}; conservation is "
                    "tissue-invariant, so the context swap is reading wrong columns")
        if scheme != "identity":
            moved = float((p_tab != all_tab).any(dim=1).float().mean())
            LOGGER.info("  %.1f%% of loci received a different context vector",
                        100 * moved)
            if moved < 0.5:
                raise SystemExit(f"STOP: scheme {scheme} changed only {moved:.1%} of "
                                 f"rows; the perturbation did not apply")
        offset, frames = 0, []
        for (cohort, wt, mut, _, _), n in zip(prepared, sizes):
            frames.append(ev.score_chunk(
                model, "fusion", tokenizer, cohort, wt, mut,
                p_tab[offset:offset + n], p_missing[offset:offset + n],
                args, device, args.seed, weights, weights_sha))
            offset += n
        scored[scheme] = pd.concat(frames, ignore_index=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ref = scored["identity"]
    rows = []
    for scheme in args.schemes:
        if scheme == "identity":
            continue
        other = scored[scheme]
        if len(other) != len(ref):
            raise SystemExit(f"STOP: {scheme} produced {len(other)} rows, "
                             f"identity produced {len(ref)}")
        for label, col in (("absolute methylation (WT_M)", LEVEL_COL),
                           ("variant effect (Delta_M)", DELTA_COL)):
            stat = agreement(ref[col].to_numpy(float), other[col].to_numpy(float))
            stat.update({"scheme": scheme, "quantity": label, "column": col})
            rows.append(stat)

    truth = {}
    for split in ("test",):
        tp = Path(args.split_template.format(split=split))
        if tp.is_file():
            t = pd.read_csv(tp, usecols=["probeID", "Median_Beta"])
            truth.update(dict(zip(t["probeID"].astype(str), t["Median_Beta"])))
    level_rows = []
    if truth:
        for scheme in args.schemes:
            f = scored[scheme][["probeID", "WT_Beta_RC_Avg"]].copy()
            f["probeID"] = f["probeID"].astype(str)
            f = f.drop_duplicates("probeID")
            f["true_beta"] = f["probeID"].map(truth)
            f = f[f["true_beta"].notna()]
            err = (f["WT_Beta_RC_Avg"] - f["true_beta"]).abs()
            level_rows.append({
                "scheme": scheme, "n_probes": int(len(f)),
                "beta_mae": float(err.mean()),
                "beta_rmse": float(np.sqrt((err ** 2).mean())),
                "beta_pearson": float(np.corrcoef(f["WT_Beta_RC_Avg"], f["true_beta"])[0, 1]),
            })
            LOGGER.info("level  %-24s beta MAE %.4f  (n=%d)",
                        scheme, level_rows[-1]["beta_mae"], len(f))
    else:
        LOGGER.warning("no test split found at %s; level MAE skipped",
                       args.split_template)

    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "agreement_with_identity.csv", index=False)
    if level_rows:
        pd.DataFrame(level_rows).to_csv(
            args.output_dir / "level_accuracy_by_scheme.csv", index=False)
    for scheme, frame in scored.items():
        frame.to_csv(args.output_dir / f"pair_scores_{scheme}.csv", index=False)

    print()
    print("=" * 84)
    print("CONTEXT PERMUTATION -- agreement with the true-context run")
    print("=" * 84)
    print(f"{'scheme':<14}{'quantity':<30}{'normMAE':>9}{'pearson':>9}"
          f"{'spearman':>10}{'MAE':>9}{'sign':>8}")
    for _, r in table.iterrows():
        sd = float(r.get("sd_reference", float("nan")))
        mae = float(r.get("mae", float("nan")))
        norm = mae / sd if sd else float("nan")
        print(f"{r['scheme']:<14}{r['quantity']:<30}{norm:>9.4f}"
              f"{r.get('pearson', float('nan')):>9.4f}"
              f"{r.get('spearman', float('nan')):>10.4f}{mae:>9.4f}"
              f"{r.get('sign_agreement', float('nan')):>8.3f}")
    print("-" * 84)

    verdict = {}
    for scheme in args.schemes:
        if scheme == "identity":
            continue
        lvl = table[(table.scheme == scheme) & (table.column == LEVEL_COL)]
        dlt = table[(table.scheme == scheme) & (table.column == DELTA_COL)]
        if lvl.empty or dlt.empty:
            continue
        lp, dp = float(lvl.iloc[0]["pearson"]), float(dlt.iloc[0]["pearson"])
        ln = float(lvl.iloc[0]["mae"]) / float(lvl.iloc[0]["sd_reference"])
        dn = float(dlt.iloc[0]["mae"]) / float(dlt.iloc[0]["sd_reference"])
        verdict[scheme] = {
            "levels_normalised_mae": ln,
            "deltas_normalised_mae": dn,
            "levels_over_deltas_ratio": (ln / dn) if dn else float("nan"),
            "deltas_preserved_more_normalised_mae": bool(dn < ln),
            "levels_pearson": lp,
            "deltas_pearson": dp,
            "deltas_preserved_more_pearson": bool(dp > lp),
        }
        print(f"{scheme:<14} normMAE levels {ln:.4f}  deltas {dn:.4f}"
              f"  ({ln / dn:.2f}x)   [pearson {lp:.4f} / {dp:.4f}]")

    primary = [v["deltas_preserved_more_normalised_mae"] for v in verdict.values()]
    secondary = [v["deltas_preserved_more_pearson"] for v in verdict.values()]
    if primary and all(primary):
        print("\nOn normalised error, deltas survive the perturbation better than")
        print("levels in EVERY scheme. Context sets the methylation LEVEL and")
        print("contributes comparatively little to the predicted variant EFFECT.")
        if not all(secondary):
            print("\nNOTE: Pearson disagrees for at least one scheme. That is expected")
            print("and is not a counter-result: levels are bimodal (SD ~3.18 M-units)")
            print("so their Pearson is inflated and not comparable against the deltas'.")
            print("Report the disagreement explicitly rather than omitting it.")
    elif primary:
        print("\nOn normalised error, at least one scheme moved the deltas as much as")
        print("the levels. Do not claim allele invariance from this run; report it.")
    print("=" * 84)

    with (args.output_dir / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "context permutation: does the context vector carry allele "
                        "information?",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verdict_metric": (
                "normalised MAE (mae / sd_reference), computed per quantity so "
                "M-unit levels and delta-scale effects are on one scale. Pearson "
                "is reported alongside but does NOT decide the verdict: levels are "
                "bimodal with SD ~3.18 M-units, so their Pearson is inflated and "
                "not comparable against the deltas'. Under realistic context "
                "substitution both quantities barely move, so correlation-based "
                "statistics saturate and flip on noise. Changed 15 Sep 2026; "
                "before that the verdict was decided on Pearson and was wrong."),
            "prediction_made_before_running": (
                "absolute methylation degrades under permutation; predicted variant "
                "effects do not, because the context vector is identical for "
                "reference and alternate alleles and can enter a paired contrast "
                "only through the gate"),
            "why_this_substitutes_for_a_tissue_swap": (
                "a random locus's chromatin is further from the truth than another "
                "tissue's chromatin at the same locus, so a null here implies a null "
                "for the tissue swap"),
            "input_csv": str(args.input_csv),
            "stratum": args.stratum,
            "model": "fusion", "seed": args.seed,
            "weights_sha256": weights_sha,
            "pairs": int(len(ref)),
            "schemes": list(args.schemes),
            "permutation_seed": args.permutation_seed,
            "construction_counters": counters,
            "agreement": table.to_dict("records"),
            "verdict": verdict,
        }, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
