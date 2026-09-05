#!/usr/bin/env python3
"""Does the epigenomic context vector carry allele information?

The claim this tests
--------------------
SilentMethyl's context tower receives the SAME nine-feature vector for the
reference and the alternate allele. Structurally, therefore, context cannot
create a variant effect -- it can only rescale a sequence-driven one through the
gate. That is an argument about the architecture. This script turns it into a
measurement.

The experiment
--------------
Score the same variant-CpG pairs three times, changing only the context:

    identity   the real context for each locus (the reference run)
    shuffle    every locus is given some OTHER locus's context vector
    median     every locus is given the cohort-median context vector

Both perturbations are applied identically to the reference and alternate
alleles, because that is what the architecture does -- so the perturbation
cannot introduce an allele asymmetry that was not already there.

The prediction, stated before running:

    ABSOLUTE methylation (WT_M_RC_Avg) should degrade badly. Context is most of
    what determines the level, so giving a locus someone else's chromatin should
    move the prediction a lot.

    DELTAS (Predicted_Delta_M) should barely move. If the correlation with the
    identity run stays near 1.0, the context contributes essentially nothing to
    the predicted variant effect, and a tissue-aware or jointly trained model --
    which changes the context, not its allele-invariance -- could not change the
    variant-effect result either.

If instead the deltas move substantially, the argument is wrong and a
tissue-aware model might well help. That is the point of running it.

Why this substitutes for a tissue-context swap
----------------------------------------------
Swapping in another tissue's ATAC/histone tracks needs a cell line profiled for
all nine features, which we do not have. Permutation is a STRICTER perturbation
than a tissue swap -- a random locus's chromatin is further from the truth than
another tissue's chromatin at the same locus -- so a null result here implies a
null result for the tissue swap. It also needs no new data.

Only the fusion model has a context tower; the sequence-only arm is rejected.

Usage (run from the repository root)
------------------------------------
    python -u scripts/23_context_permutation.py --limit 500        # smoke test
    python -u scripts/23_context_permutation.py                    # full cohort
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

LOGGER = logging.getLogger("silentmethyl.ctxperm")

SCHEMES = ("identity", "shuffle", "median")
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


def permute(tab: torch.Tensor, missing: torch.Tensor, scheme: str,
            rng: np.random.Generator):
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
        # A derangement is not required, but a locus keeping its own context
        # weakens the perturbation, so resample the fixed points once.
        fixed = np.flatnonzero(order == np.arange(n))
        if len(fixed) > 1:
            order[fixed] = order[rng.permutation(fixed)]
        idx = torch.as_tensor(order, dtype=torch.long)
        return tab[idx].clone(), missing[idx].clone()
    if scheme == "median":
        med = tab.median(dim=0, keepdim=True).values
        med_missing = (missing.float().mean(dim=0, keepdim=True) > 0.5).to(missing.dtype)
        return (med.expand_as(tab).clone(), med_missing.expand_as(missing).clone())
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

    # Permute across the WHOLE cohort, not within chunks: a within-chunk shuffle
    # would keep each locus near its genomic neighbours, whose chromatin is
    # correlated, and would understate the perturbation.
    sizes = [t.shape[0] for _, _, _, t, _ in prepared]
    all_tab = torch.cat([t for _, _, _, t, _ in prepared], dim=0)
    all_missing = torch.cat([m for _, _, _, _, m in prepared], dim=0)
    LOGGER.info("context matrix: %s", tuple(all_tab.shape))

    weights = Path(args.weights_template.format(seed=args.seed, model="fusion"))
    if not weights.is_file():
        raise SystemExit(f"STOP: checkpoint not found: {weights}")
    weights_sha = ev.sha256_file(weights)
    model = ev.build_model("fusion", args, args.seed, weights, device)

    rng = np.random.default_rng(args.permutation_seed)
    scored: dict[str, pd.DataFrame] = {}
    for scheme in args.schemes:
        LOGGER.info("scoring under scheme=%s", scheme)
        p_tab, p_missing = permute(all_tab, all_missing, scheme, rng)
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

    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "agreement_with_identity.csv", index=False)
    for scheme, frame in scored.items():
        frame.to_csv(args.output_dir / f"pair_scores_{scheme}.csv", index=False)

    print()
    print("=" * 84)
    print("CONTEXT PERMUTATION -- agreement with the true-context run")
    print("=" * 84)
    print(f"{'scheme':<10}{'quantity':<30}{'pearson':>9}{'spearman':>10}"
          f"{'MAE':>9}{'sign':>8}")
    for _, r in table.iterrows():
        print(f"{r['scheme']:<10}{r['quantity']:<30}{r.get('pearson', float('nan')):>9.4f}"
              f"{r.get('spearman', float('nan')):>10.4f}{r.get('mae', float('nan')):>9.4f}"
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
        verdict[scheme] = {"levels_pearson": lp, "deltas_pearson": dp,
                           "deltas_preserved_more": bool(dp > lp)}
        print(f"{scheme}: levels r={lp:.4f}   deltas r={dp:.4f}")
    if all(v["deltas_preserved_more"] for v in verdict.values()):
        print("\nDeltas survive the perturbation better than levels in every scheme.")
        print("Context determines the methylation LEVEL and contributes little to")
        print("the predicted variant EFFECT -- the allele-invariance argument,")
        print("measured rather than asserted.")
    else:
        print("\nAt least one scheme moved the deltas as much as the levels.")
        print("The allele-invariance argument does NOT hold empirically here.")
        print("Do not claim it; report this instead.")
    print("=" * 84)

    with (args.output_dir / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "context permutation: does the context vector carry allele "
                        "information?",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
