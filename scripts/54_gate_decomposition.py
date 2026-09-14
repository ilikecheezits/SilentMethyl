#!/usr/bin/env python3
"""
Gate decomposition of the fusion model's variant effect.

What this measures
------------------
The allele-invariance claim in this paper has so far been an *argument*: the
context vector is identical for REF and ALT, so it cannot carry an allele
effect. This script turns it into a measured decomposition.

The gated fusion forward pass is

    dna  = LayerNorm(SequenceEncoder(window))          <- depends on the allele
    epi  = LayerNorm(EpigeneticEncoder(tab, missing))  <- does NOT depend on it
    g    = GateNet([dna, epi])                          <- depends on the allele,
                                                           through `dna` only
    m    = RegressionHead(g_dna * dna + g_epi * epi)

so there are exactly two channels by which an allele can move the prediction:

  1. the DNA channel   -- `dna` itself changes, gate held at its REF value;
  2. the gate channel  -- `g` changes, `dna` held at its REF value.

`epi` is bit-identical across alleles by construction. Its *contribution to the
fused vector*, `g_epi * epi`, is NOT, because `g_epi` reads `dna`. That is the
only route by which context can influence a variant effect at all, and its size
is the number this script exists to report.

Two subcommands
---------------
`instrument`  GPU. Re-scores a cohort's variant pairs and saves, per pair and
              per strand, the per-allele gates and four counterfactual
              predictions that make the channel decomposition exact:

                  m_ref        = head(g(R)_d * dna_R + g(R)_e * epi)
                  m_alt        = head(g(A)_d * dna_A + g(A)_e * epi)
                  m_alt_gateR  = head(g(R)_d * dna_A + g(R)_e * epi)   DNA only
                  m_ref_gateA  = head(g(A)_d * dna_R + g(A)_e * epi)   gate only

              m_ref and m_alt reproduce the ordinary scoring path exactly; they
              are asserted against scripts/20_variant_scoring.py output when
              that file is available.

`analyse`     CPU. The regression and the distributions. Runs on instrumented
              output when present, and falls back to the WT_/MUT_Gate_Avg
              columns already in scripts/20 pair_scores.csv, which carry the
              per-allele gates (fwd/RC averaged) but not the counterfactuals.

Usage
-----
    python -u scripts/54_gate_decomposition.py instrument \
        --input-csv data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv \
        --cohort egtex --seed 42 --device cuda \
        --weights-template 'checkpoints_ablation/breast_epithelium/seed{seed}/fusion/best_weights.pth' \
        --split-template 'data/datafiles_breast_epithelium/{split}.csv' \
        --output-dir results/journal/ablation_breast_epithelium/gate_decomposition

    python -u scripts/54_gate_decomposition.py analyse \
        --output-dir results/journal/ablation_breast_epithelium/gate_decomposition
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LOGGER = logging.getLogger("silentmethyl.gatedecomp")

# Cohort -> (default scoring input, default scripts/20 pair_scores location under
# the ablation tree). Both are overridable; these are what the R6/R7 chain used.
COHORTS = {
    "genoa": (
        "data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv",
        "results/journal/ablation_breast_epithelium/genoa_variant_scoring/heldout",
    ),
    "egtex": (
        "data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv",
        "results/journal/ablation_breast_epithelium/egtex_variant_scoring/heldout",
    ),
}


def _load_scoring_module():
    """Import scripts/20_variant_scoring.py -- the name is not a valid identifier."""
    path = SCRIPT_DIR / "20_variant_scoring.py"
    spec = importlib.util.spec_from_file_location("variant_scoring_20", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["variant_scoring_20"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# instrument
# ---------------------------------------------------------------------------

def run_instrument(args: argparse.Namespace) -> int:
    import torch
    from training_common import (MISSING_FEATURES, TABULAR_FEATURES, FusionModel,
                                 autocast_context, centered_crop, get_tokenizer,
                                 load_model_state, m_to_beta_tensor, reverse_complement,
                                 set_seed)
    vs = _load_scoring_module()

    if args.amp:
        raise SystemExit(
            "--amp is refused here. A variant effect is the difference of two "
            "nearly identical predictions and the gate channel is smaller still; "
            "it must be measured in FP32.")

    pairs = pd.read_csv(args.input_csv)
    absent = [c for c in vs.REQUIRED_INPUT_COLUMNS if c not in pairs.columns]
    if absent:
        raise SystemExit(f"{args.input_csv} is missing {absent}")
    actual = set(pairs["probe_split"].astype(str).unique())
    if not actual <= {"test"}:
        raise SystemExit(f"held-out stratum only; file contains splits {sorted(actual)}")

    pairs, masked_dropped = vs.apply_probe_qc(pairs, args.hm450_manifest, False)
    LOGGER.info("HM450 probe QC dropped %d pairs; %d remain", masked_dropped, len(pairs))
    if args.limit > 0:
        pairs = pairs.head(args.limit).copy()
        LOGGER.warning("SMOKE TEST: limited to %d pairs", len(pairs))

    records = vs.load_probe_records(args.split_template,
                                    set(pairs["probeID"].astype(str)))
    LOGGER.info("materialised %d target probes", len(records))

    device = vs.choose_device(args.device)
    tokenizer = get_tokenizer(args.model_path)
    counters = {k: 0 for k in (
        "scoreable", "probe_not_in_splits", "bad_stored_sequence_length",
        "stored_sequence_not_cpg_centred", "outside_model_window",
        "alters_target_cpg", "reference_base_mismatch",
        "unexpected_window_difference", "window_not_cpg_centred")}
    prepared = []
    for i in range(0, len(pairs), args.chunk_size):
        built = vs.build_chunk(pairs.iloc[i:i + args.chunk_size], records, counters)
        if built is not None:
            prepared.append(built)
    if not prepared:
        raise SystemExit("no pair survived construction; see counters above")
    LOGGER.info("construction counters: %s", counters)

    weights = Path(args.weights_template.format(seed=args.seed, model="fusion"))
    if not weights.is_file():
        raise SystemExit(f"checkpoint not found: {weights}")
    weights_sha = vs.sha256_file(weights)

    set_seed(args.seed)
    model = FusionModel(args.model_path, fusion_mode="gated",
                        tabular_dim=len(TABULAR_FEATURES),
                        local_dir=args.local_model_dir)
    state = load_model_state(str(weights), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.float().to(device).eval()
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    if dtypes != ["torch.float32"]:
        raise RuntimeError(f"FP32 required but model holds {dtypes}")

    @torch.inference_mode()
    def branch_pass(seqs, tab, missing, desc):
        """Per-allele branch internals for one strand.

        Returns the fused-vector pieces rather than a prediction, so the caller
        can recombine REF gates with ALT DNA and vice versa.
        """
        dna_all, epi_all = [], []
        for start in range(0, len(seqs), args.batch_size):
            stop = min(len(seqs), start + args.batch_size)
            enc = tokenizer(seqs[start:stop], truncation=True,
                            max_length=len(seqs[start]), padding="max_length",
                            return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            with autocast_context(device, False):
                dna = model.norm_dna(model.sequence_encoder(ids, mask))
                epi = model.norm_epi(model.epi_encoder(
                    tab[start:stop].to(device), missing[start:stop].to(device)))
            dna_all.append(dna.float().cpu())
            epi_all.append(epi.float().cpu())
        return torch.cat(dna_all), torch.cat(epi_all)

    @torch.inference_mode()
    def gate_of(dna, epi):
        out = []
        for start in range(0, len(dna), 1024):
            stop = min(len(dna), start + 1024)
            joined = torch.cat([dna[start:stop].to(device),
                                epi[start:stop].to(device)], dim=1)
            out.append(model.gate_network(joined).float().cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def head_of(fused):
        out = []
        for start in range(0, len(fused), 1024):
            stop = min(len(fused), start + 1024)
            _, m_pred = model.heads(fused[start:stop].to(device))
            out.append(m_pred.float().cpu().reshape(-1))
        return torch.cat(out)

    frames = []
    for cohort, wt, mut, tab, missing in prepared:
        rc_tab, rc_missing = vs.make_rc_context(tab, missing)
        out = cohort.copy()
        out["Model"] = "fusion_instrumented"
        out["Seed"] = int(args.seed)
        out["Weights_Path"] = str(weights)
        out["Weights_SHA256"] = weights_sha

        strand_m = {}
        for strand, (wt_s, mut_s, t, mss) in {
            "FWD": (wt, mut, tab, missing),
            "RC": ([reverse_complement(s) for s in wt],
                   [reverse_complement(s) for s in mut], rc_tab, rc_missing),
        }.items():
            dna_r, epi_r = branch_pass(wt_s, t, mss, f"REF {strand}")
            dna_a, epi_a = branch_pass(mut_s, t, mss, f"ALT {strand}")

            # The epigenetic branch takes only (tab, missing), which are identical
            # for the two alleles, so `epi` must be bit-identical. Measured, not
            # assumed: a future refactor that let the context see the sequence
            # would invalidate the entire allele-invariance argument, and this is
            # where that would surface.
            epi_bit_identical = torch.equal(epi_r, epi_a)
            epi_max_abs_diff = float((epi_r - epi_a).abs().max())

            g_r = gate_of(dna_r, epi_r)
            g_a = gate_of(dna_a, epi_a)

            fused_ref = dna_r * g_r[:, 0:1] + epi_r * g_r[:, 1:2]
            fused_alt = dna_a * g_a[:, 0:1] + epi_a * g_a[:, 1:2]
            # DNA channel only: the allele changes `dna`, the gate is pinned at REF.
            fused_alt_gateR = dna_a * g_r[:, 0:1] + epi_r * g_r[:, 1:2]
            # Gate channel only: `dna` is pinned at REF, the gate takes its ALT value.
            fused_ref_gateA = dna_r * g_a[:, 0:1] + epi_r * g_a[:, 1:2]

            m_ref = head_of(fused_ref)
            m_alt = head_of(fused_alt)
            m_alt_gateR = head_of(fused_alt_gateR)
            m_ref_gateA = head_of(fused_ref_gateA)
            strand_m[strand] = (m_ref, m_alt)

            # The epi branch's contribution to the fused vector. Allele-invariant
            # only to the extent that g_epi is.
            contrib_r = epi_r * g_r[:, 1:2]
            contrib_a = epi_a * g_a[:, 1:2]
            contrib_identical = (contrib_r == contrib_a).all(dim=1).numpy()
            contrib_l2 = (contrib_r - contrib_a).norm(dim=1).numpy()
            contrib_ref_l2 = contrib_r.norm(dim=1).numpy()

            p = strand
            out[f"Epi_Vector_Bit_Identical_{p}"] = bool(epi_bit_identical)
            out[f"Epi_Vector_Max_Abs_Diff_{p}"] = epi_max_abs_diff
            out[f"Epi_Contribution_Bit_Identical_{p}"] = contrib_identical.astype(int)
            out[f"Epi_Contribution_L2_Diff_{p}"] = contrib_l2
            out[f"Epi_Contribution_L2_Ref_{p}"] = contrib_ref_l2
            out[f"Gate_DNA_REF_{p}"] = g_r[:, 0].numpy()
            out[f"Gate_EPI_REF_{p}"] = g_r[:, 1].numpy()
            out[f"Gate_DNA_ALT_{p}"] = g_a[:, 0].numpy()
            out[f"Gate_EPI_ALT_{p}"] = g_a[:, 1].numpy()
            out[f"M_REF_{p}"] = m_ref.numpy()
            out[f"M_ALT_{p}"] = m_alt.numpy()
            out[f"M_ALT_GateFrozenREF_{p}"] = m_alt_gateR.numpy()
            out[f"M_REF_GateALT_{p}"] = m_ref_gateA.numpy()
            out[f"Delta_M_Total_{p}"] = (m_alt - m_ref).numpy()
            out[f"Delta_M_DNA_Channel_{p}"] = (m_alt_gateR - m_ref).numpy()
            out[f"Delta_M_Gate_Channel_{p}"] = (m_alt - m_alt_gateR).numpy()
            out[f"Delta_M_Gate_Channel_AtREF_{p}"] = (m_ref_gateA - m_ref).numpy()
            out[f"DNA_L2_Diff_{p}"] = (dna_a - dna_r).norm(dim=1).numpy()

        # RC-averaged quantities, matching the convention scripts/20 reports on.
        m_ref_avg = (strand_m["FWD"][0] + strand_m["RC"][0]) / 2.0
        m_alt_avg = (strand_m["FWD"][1] + strand_m["RC"][1]) / 2.0
        out["WT_M_RC_Avg"] = m_ref_avg.numpy()
        out["MUT_M_RC_Avg"] = m_alt_avg.numpy()
        out["WT_Beta_RC_Avg"] = m_to_beta_tensor(m_ref_avg).numpy()
        out["MUT_Beta_RC_Avg"] = m_to_beta_tensor(m_alt_avg).numpy()
        out["Predicted_Delta_M"] = (m_alt_avg - m_ref_avg).numpy()
        out["Predicted_Delta_Beta"] = out["MUT_Beta_RC_Avg"] - out["WT_Beta_RC_Avg"]
        for name in ("Delta_M_Total", "Delta_M_DNA_Channel",
                     "Delta_M_Gate_Channel", "Delta_M_Gate_Channel_AtREF"):
            out[f"{name}_RC_Avg"] = (out[f"{name}_FWD"] + out[f"{name}_RC"]) / 2.0
        for name in ("Gate_DNA_REF", "Gate_EPI_REF", "Gate_DNA_ALT", "Gate_EPI_ALT"):
            out[f"{name}_RC_Avg"] = (out[f"{name}_FWD"] + out[f"{name}_RC"]) / 2.0
        frames.append(out)

    scored = pd.concat(frames, ignore_index=True)

    # Reproduction check against the ordinary scorer. The instrumented path
    # rebuilds the forward pass by hand; if it has drifted from FusionModel.forward
    # every number below is measuring the wrong model.
    reproduction = None
    if args.reference_pair_scores:
        ref_path = Path(args.reference_pair_scores)
        if ref_path.is_file():
            ref = pd.read_csv(ref_path, usecols=["Pair_UID", "Predicted_Delta_M",
                                                 "WT_M_RC_Avg", "MUT_M_RC_Avg"])
            merged = scored[["Pair_UID", "Predicted_Delta_M", "WT_M_RC_Avg",
                             "MUT_M_RC_Avg"]].merge(
                ref, on="Pair_UID", how="inner", suffixes=("_new", "_ref"))
            reproduction = {
                "reference_file": str(ref_path),
                "pairs_matched": int(len(merged)),
                "max_abs_diff_delta_m": float(
                    (merged["Predicted_Delta_M_new"] - merged["Predicted_Delta_M_ref"]).abs().max()),
                "max_abs_diff_wt_m": float(
                    (merged["WT_M_RC_Avg_new"] - merged["WT_M_RC_Avg_ref"]).abs().max()),
            }
            LOGGER.info("reproduction check: %s", reproduction)
            if reproduction["max_abs_diff_delta_m"] > 1e-4:
                raise SystemExit(
                    "instrumented forward pass does not reproduce "
                    f"{ref_path}: max |delta_M| difference "
                    f"{reproduction['max_abs_diff_delta_m']:.3e}")
        else:
            LOGGER.warning("reference pair_scores not found: %s", ref_path)

    target = args.output_dir / args.cohort / f"seed{args.seed}" / "instrumented_pairs.csv"
    vs.atomic_csv(scored, target)
    vs.atomic_json(
        {
            "analysis": "gate decomposition of the fusion variant effect (instrumented forward pass)",
            "cohort": args.cohort,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_csv": str(args.input_csv),
            "input_sha256": vs.sha256_file(args.input_csv),
            "split_template": args.split_template,
            "weights": str(weights),
            "weights_sha256": weights_sha,
            "seed": int(args.seed),
            "precision": "fp32",
            "pairs_scored": int(len(scored)),
            "masked_probe_pairs_dropped": int(masked_dropped),
            "construction_counters": counters,
            "reproduction_check": reproduction,
            "channel_definition": (
                "Delta_M_DNA_Channel = head(g(REF)_d*dna_ALT + g(REF)_e*epi) - m_REF; "
                "Delta_M_Gate_Channel = m_ALT - head(g(REF)_d*dna_ALT + g(REF)_e*epi). "
                "They sum exactly to Delta_M_Total by construction."),
            "output": str(target),
        },
        args.output_dir / args.cohort / f"seed{args.seed}" / "run_summary.json",
    )
    LOGGER.info("wrote %s (%d rows)", target, len(scored))
    return 0


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------

def _ols_r2(x: np.ndarray, y: np.ndarray, fit_intercept: bool = True) -> dict:
    """Univariate OLS of y on x. R^2 is 1 - SSres/SStot about the mean of y."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 10:
        return {"n": int(len(x))}
    design = np.column_stack([x, np.ones_like(x)]) if fit_intercept else x[:, None]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(x)),
        "slope": float(coef[0]),
        "intercept": float(coef[1]) if fit_intercept else 0.0,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "residual_sd": float(np.sqrt(ss_res / max(1, len(x) - design.shape[1]))),
    }


def _describe(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    q = np.quantile(values, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "n": int(len(values)), "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
        "min": float(q[0]), "p1": float(q[1]), "p5": float(q[2]), "p25": float(q[3]),
        "median": float(q[4]), "p75": float(q[5]), "p95": float(q[6]),
        "p99": float(q[7]), "max": float(q[8]),
    }


def _block_bootstrap_r2(x, y, blocks, rng, draws=1000):
    """Percentile CI for R^2, resampling 1 Mb blocks rather than pairs.

    Pairs sharing a locus are not independent; resampling them individually
    would give an interval that is far too tight.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, blocks = x[finite], y[finite], blocks[finite]
    uniq = np.unique(blocks)
    index = {b: np.flatnonzero(blocks == b) for b in uniq}
    out = []
    for _ in range(draws):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[b] for b in picked])
        stat = _ols_r2(x[idx], y[idx])
        if "r2" in stat:
            out.append(stat["r2"])
    if not out:
        return None
    lo, hi = np.quantile(out, [0.025, 0.975])
    return [float(lo), float(hi)]


def analyse_cohort(cohort: str, args: argparse.Namespace) -> dict:
    _, default_scores = COHORTS[cohort]
    score_root = Path(args.pair_scores_root.format(cohort=cohort)
                      if args.pair_scores_root else default_scores)
    fusion_path = score_root / "fusion" / f"seed{args.seed}" / "pair_scores.csv"
    sequence_path = score_root / "sequence" / f"seed{args.seed}" / "pair_scores.csv"
    for path in (fusion_path, sequence_path):
        if not path.is_file():
            raise SystemExit(f"required pair_scores not found: {path}")

    keep = ["Pair_UID", "probeID", "chr", "Position_1based", "distance_bp",
            "Predicted_Delta_M", "Predicted_Delta_Beta", "WT_M_RC_Avg", "MUT_M_RC_Avg"]
    gate_cols = ["WT_Gate_Avg_DNA", "WT_Gate_Avg_EPI", "WT_Gate_Avg_DNA_Share",
                 "MUT_Gate_Avg_DNA", "MUT_Gate_Avg_EPI", "MUT_Gate_Avg_DNA_Share"]
    fusion = pd.read_csv(fusion_path, usecols=keep + gate_cols)
    sequence = pd.read_csv(sequence_path, usecols=keep)
    merged = fusion.merge(sequence, on="Pair_UID", how="inner",
                          suffixes=("_fusion", "_sequence"), validate="one_to_one")
    LOGGER.info("%s: %d fusion, %d sequence -> %d matched pairs",
                cohort, len(fusion), len(sequence), len(merged))

    delta_fusion = merged["Predicted_Delta_M_fusion"].to_numpy(float)
    delta_seq = merged["Predicted_Delta_M_sequence"].to_numpy(float)
    gate_ref = merged["WT_Gate_Avg_DNA"].to_numpy(float)
    gate_alt = merged["MUT_Gate_Avg_DNA"].to_numpy(float)
    gate_mid = 0.5 * (gate_ref + gate_alt)
    blocks = (merged["chr_fusion"].astype(str) + ":"
              + (merged["Position_1based_fusion"].to_numpy() // 1_000_000).astype(str)
              ).to_numpy()
    rng = np.random.default_rng(args.seed)

    result = {
        "cohort": cohort,
        "pairs": int(len(merged)),
        "fusion_pair_scores": str(fusion_path),
        "sequence_pair_scores": str(sequence_path),
        "scale": "M-value (the scale scripts/20 reports as primary)",
        "decompositions": {},
        "gate_allele_invariance": {},
    }

    models = {
        # The claim: the fusion variant effect is the sequence variant effect
        # rescaled by the learned DNA gate.
        "delta_fusion ~ delta_sequence * gate_dna(REF)": delta_seq * gate_ref,
        "delta_fusion ~ delta_sequence * gate_dna(mean of REF,ALT)": delta_seq * gate_mid,
        # Reference points. The first says how much the gate rescaling adds over
        # the raw sequence delta; the second is the null of no sequence signal.
        "delta_fusion ~ delta_sequence (no gate)": delta_seq,
    }
    for name, predictor in models.items():
        stat = _ols_r2(predictor, delta_fusion)
        stat["r2_block_bootstrap_95ci"] = _block_bootstrap_r2(
            predictor, delta_fusion, blocks, rng, args.bootstrap_draws)
        result["decompositions"][name] = stat

    gate_shift = np.abs(gate_alt - gate_ref)
    share_shift = np.abs(merged["MUT_Gate_Avg_DNA_Share"].to_numpy(float)
                         - merged["WT_Gate_Avg_DNA_Share"].to_numpy(float))
    epi_shift = np.abs(merged["MUT_Gate_Avg_EPI"].to_numpy(float)
                       - merged["WT_Gate_Avg_EPI"].to_numpy(float))
    result["gate_allele_invariance"] = {
        "abs_gate_dna_alt_minus_ref": _describe(gate_shift),
        "abs_gate_epi_alt_minus_ref": _describe(epi_shift),
        "abs_gate_dna_share_alt_minus_ref": _describe(share_shift),
        "gate_dna_ref": _describe(gate_ref),
        "gate_dna_share_ref": _describe(merged["WT_Gate_Avg_DNA_Share"].to_numpy(float)),
        "relative_gate_dna_shift_median": float(
            np.median(gate_shift / np.clip(gate_ref, 1e-12, None))),
        "note": ("gates here are the forward/RC average scripts/20 saves. The "
                 "instrumented run reports them per strand as well."),
    }

    # Instrumented output, when it exists: the exact channel decomposition.
    inst_path = args.output_dir / cohort / f"seed{args.seed}" / "instrumented_pairs.csv"
    if inst_path.is_file():
        inst = pd.read_csv(inst_path)
        total = inst["Delta_M_Total_RC_Avg"].to_numpy(float)
        dna_ch = inst["Delta_M_DNA_Channel_RC_Avg"].to_numpy(float)
        gate_ch = inst["Delta_M_Gate_Channel_RC_Avg"].to_numpy(float)
        denom = np.clip(np.abs(total), 1e-12, None)
        bit_cols = [c for c in inst.columns if c.startswith("Epi_Contribution_Bit_Identical_")]
        vec_cols = [c for c in inst.columns if c.startswith("Epi_Vector_Bit_Identical_")]
        result["instrumented"] = {
            "file": str(inst_path),
            "pairs": int(len(inst)),
            "identity_check_max_abs_residual": float(
                np.abs(total - (dna_ch + gate_ch)).max()),
            "epi_encoder_output_bit_identical_across_alleles": {
                c: bool(inst[c].all()) for c in vec_cols},
            "epi_encoder_output_max_abs_diff": {
                c: float(inst[c].max()) for c in inst.columns
                if c.startswith("Epi_Vector_Max_Abs_Diff_")},
            "epi_gated_contribution_bit_identical_fraction": {
                c: float(inst[c].mean()) for c in bit_cols},
            "delta_m_total": _describe(total),
            "delta_m_dna_channel": _describe(dna_ch),
            "delta_m_gate_channel": _describe(gate_ch),
            "gate_channel_share_of_abs_effect": _describe(np.abs(gate_ch) / denom),
            "variance_share_dna_channel": float(np.var(dna_ch) / np.var(total)),
            "variance_share_gate_channel": float(np.var(gate_ch) / np.var(total)),
            "corr_dna_channel_with_total": float(np.corrcoef(dna_ch, total)[0, 1]),
            "corr_gate_channel_with_total": float(np.corrcoef(gate_ch, total)[0, 1]),
        }
        for strand in ("FWD", "RC"):
            g_ref = inst[f"Gate_DNA_REF_{strand}"].to_numpy(float)
            g_alt = inst[f"Gate_DNA_ALT_{strand}"].to_numpy(float)
            result["gate_allele_invariance"][f"abs_gate_dna_alt_minus_ref_{strand}"] = \
                _describe(np.abs(g_alt - g_ref))
    return result


def run_analyse(args: argparse.Namespace) -> int:
    vs = _load_scoring_module()
    payload = {
        "analysis": "gate decomposition of the fusion model's variant effect",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "single_seed_note": ("Seed 42 only, by design. This analysis characterises "
                             "one trained model's internal arithmetic; it is not a "
                             "run-to-run variability estimate."),
        "bootstrap": f"{args.bootstrap_draws} draws over 1 Mb genomic blocks",
        "cohorts": {},
    }
    for cohort in args.cohorts:
        payload["cohorts"][cohort] = analyse_cohort(cohort, args)
    vs.atomic_json(payload, args.output_dir / "gate_decomposition_summary.json")

    print()
    print("=" * 78)
    for cohort, res in payload["cohorts"].items():
        print(f"\n### {cohort}  ({res['pairs']:,} pairs, M scale)")
        for name, stat in res["decompositions"].items():
            ci = stat.get("r2_block_bootstrap_95ci")
            ci_s = f"  [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
            print(f"  R2 {stat['r2']:.5f}{ci_s}   slope {stat['slope']:+.4f}   {name}")
        inv = res["gate_allele_invariance"]["abs_gate_dna_alt_minus_ref"]
        base = res["gate_allele_invariance"]["gate_dna_ref"]
        print(f"  |gate_dna(ALT) - gate_dna(REF)|: median {inv['median']:.3e}  "
              f"p95 {inv['p95']:.3e}  max {inv['max']:.3e}")
        print(f"  gate_dna(REF) itself:            median {base['median']:.4f}  "
              f"p5 {base['p5']:.4f}  p95 {base['p95']:.4f}")
        if "instrumented" in res:
            ins = res["instrumented"]
            print(f"  channel variance share: DNA {ins['variance_share_dna_channel']:.4f}"
                  f"  gate {ins['variance_share_gate_channel']:.4f}")
            print(f"  epi encoder output bit-identical across alleles: "
                  f"{ins['epi_encoder_output_bit_identical_across_alleles']}")
            print(f"  epi *gated contribution* bit-identical fraction: "
                  f"{ins['epi_gated_contribution_bit_identical_fraction']}")
    print("=" * 78)
    print(f"wrote {args.output_dir / 'gate_decomposition_summary.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common_out = dict(
        type=Path,
        default=Path("results/journal/ablation_breast_epithelium/gate_decomposition"))

    i = sub.add_parser("instrument", help="GPU: per-allele gates and channel counterfactuals")
    i.add_argument("--cohort", choices=sorted(COHORTS), required=True)
    i.add_argument("--input-csv", type=Path, default=None)
    i.add_argument("--seed", type=int, default=42)
    i.add_argument("--weights-template",
                   default="checkpoints_ablation/breast_epithelium/seed{seed}/fusion/best_weights.pth")
    i.add_argument("--split-template",
                   default="data/datafiles_breast_epithelium/{split}.csv")
    i.add_argument("--hm450-manifest", type=Path,
                   default=Path("data/HM450.hg38.manifest.tsv.gz"))
    i.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    i.add_argument("--local-model-dir", default="./dnabert2_local")
    i.add_argument("--batch-size", type=int, default=32)
    i.add_argument("--chunk-size", type=int, default=20_000)
    i.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    i.add_argument("--amp", action="store_true", help="refused; see the error text")
    i.add_argument("--limit", type=int, default=0)
    i.add_argument("--reference-pair-scores", default=None,
                   help="scripts/20 pair_scores.csv to reproduce as a correctness check")
    i.add_argument("--output-dir", **common_out)

    a = sub.add_parser("analyse", help="CPU: the decomposition regression and distributions")
    a.add_argument("--cohorts", nargs="+", default=["genoa", "egtex"],
                   choices=sorted(COHORTS))
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--pair-scores-root", default=None,
                   help="override, may contain {cohort}")
    a.add_argument("--bootstrap-draws", type=int, default=1000)
    a.add_argument("--output-dir", **common_out)

    args = p.parse_args()
    if args.command == "instrument" and args.input_csv is None:
        args.input_csv = Path(COHORTS[args.cohort][0])
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = parse_args()
    return run_instrument(args) if args.command == "instrument" else run_analyse(args)


if __name__ == "__main__":
    sys.exit(main())
