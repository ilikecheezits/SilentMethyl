#!/usr/bin/env python3
"""
Where does zero-shot cross-tissue transfer fail?

The question
------------
The held-out-BreastEpithelium joint model never saw a breast probe in training.
Its sequence arm still transfers almost perfectly (+0.0013 beta MAE). The
hypothesis is that the sequence->methylation mapping is tissue-invariant EXCEPT
where methylation is itself tissue-plastic, and that those loci have identifiable
regulatory character.

This localises the error against (a) measured cross-tissue methylation variance
and (b) genomic and chromatin annotation.

Split discipline -- read this before comparing to Task C
--------------------------------------------------------
Task C (`55_gate_plasticity.py`) computes cross-tissue variance on the joint
**validation** probes, chr10 + chr11. This script works on the held-out
**test** probes, chr8 + chr9. The two probe sets are DISJOINT -- zero overlap --
so the variance here is recomputed from the four tissues' own `test.csv`, by the
same method. Both are legitimate; they are not the same analysis and must not be
described as one.

Coverage: 26,558 of the 26,570 held-out breast test probes are measured in all
four tissues (99.95%).

Annotation
----------
CpG-island class comes from the HM450 hg38 CGI manifest (`CGIposition`:
Island / N_Shore / S_Shore / N_Shelf / S_Shelf, with missing = OpenSea).

"Promoter" is operationalised as high H3K4me3 rather than as gene overlap. The
manifest's `gene` column marks any probe inside a gene body, which is far too
coarse to mean promoter, whereas H3K4me3 is the canonical promoter mark and is
already one of the seven reference tracks the model consumes. Using the model's
own input tracks also keeps the annotation on the same footing as the features.

Usage
-----
    python -u scripts/56_transfer_failure.py \
        --output-dir results/journal/joint/transfer_failure
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger("silentmethyl.transferfail")

TISSUE_TEST = {
    "BreastEpithelium": "data/datafiles_breast_epithelium/test.csv",
    "ColonTransverse": "data/datafiles_multitissue/ColonTransverse/test.csv",
    "KidneyCortex": "data/datafiles_multitissue/KidneyCortex/test.csv",
    "Lung": "data/datafiles_multitissue/Lung/test.csv",
}

CHROMATIN = ["Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal",
             "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal",
             "Ref_H3K4me1_Signal"]


def _block_ci(values, blocks, rng, draws, stat=np.mean):
    """Percentile CI for a statistic, resampling 1 Mb blocks."""
    uniq = np.unique(blocks)
    index = {b: np.flatnonzero(blocks == b) for b in uniq}
    out = []
    for _ in range(draws):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[b] for b in pick])
        out.append(float(stat(values[idx])))
    lo, hi = np.quantile(out, [0.025, 0.975])
    return [float(lo), float(hi)]


def load(args) -> pd.DataFrame:
    frames = {}
    for arm in ("fusion", "sequence", "epi"):
        p = args.pred_root / arm / "predictions.csv"
        if not p.is_file():
            LOGGER.warning("missing %s arm at %s", arm, p)
            continue
        d = pd.read_csv(p, usecols=["probeID", "chr", "pos", "true_beta",
                                    "pred_beta_rc_avg", "beta_absolute_error"])
        frames[arm] = d.rename(columns={
            "pred_beta_rc_avg": f"pred_{arm}", "beta_absolute_error": f"abserr_{arm}"})
        LOGGER.info("%s: %d probes", arm, len(d))

    base = frames["fusion"][["probeID", "chr", "pos", "true_beta"]].copy()
    for arm, d in frames.items():
        base = base.merge(d[["probeID", f"pred_{arm}", f"abserr_{arm}"]],
                          on="probeID", how="left")

    betas = []
    for tissue, path in TISSUE_TEST.items():
        d = pd.read_csv(path, usecols=["probeID", "Median_Beta"])
        betas.append(d.rename(columns={"Median_Beta": tissue}).set_index("probeID"))
    beta = pd.concat(betas, axis=1, join="inner").reset_index()
    LOGGER.info("probes measured in all four tissues (chr8/9): %d", len(beta))

    df = base.merge(beta, on="probeID", how="inner", validate="one_to_one")
    mat = df[list(TISSUE_TEST)].to_numpy(float)
    df["cross_tissue_var"] = mat.var(axis=1, ddof=0)
    df["cross_tissue_sd"] = np.sqrt(df["cross_tissue_var"])
    df["cross_tissue_range"] = mat.max(axis=1) - mat.min(axis=1)
    df["cross_tissue_mean_beta"] = mat.mean(axis=1)

    cgi = pd.read_csv(args.cgi_manifest, sep="\t",
                      usecols=["probeID", "CGI", "CGIposition"])
    df = df.merge(cgi, on="probeID", how="left")
    df["cgi_class"] = df["CGIposition"].fillna("OpenSea").replace(
        {"N_Shore": "Shore", "S_Shore": "Shore",
         "N_Shelf": "Shelf", "S_Shelf": "Shelf"})

    chrom = pd.read_csv(args.breast_test_csv,
                        usecols=["probeID"] + CHROMATIN)
    df = df.merge(chrom, on="probeID", how="left")
    df["block"] = (df["chr"].astype(str) + ":"
                   + (df["pos"].to_numpy() // 1_000_000).astype(str))
    LOGGER.info("final analysis frame: %d probes", len(df))
    return df


def run(args) -> int:
    df = load(args)
    rng = np.random.default_rng(args.seed)
    arms = [a for a in ("sequence", "fusion", "epi") if f"abserr_{a}" in df.columns]
    primary = "sequence" if "sequence" in arms else arms[0]

    df["var_decile"] = pd.qcut(df["cross_tissue_var"], 10, labels=False, duplicates="drop")
    by_var = []
    for d, sub in df.groupby("var_decile"):
        row = {"decile": int(d) + 1, "n": int(len(sub)),
               "cross_tissue_var_median": float(sub["cross_tissue_var"].median()),
               "cross_tissue_sd_median": float(sub["cross_tissue_sd"].median()),
               "mean_beta_median": float(sub["cross_tissue_mean_beta"].median())}
        for a in arms:
            v = sub[f"abserr_{a}"].to_numpy(float)
            row[f"mae_{a}"] = float(np.mean(v))
            row[f"mae_{a}_95ci"] = _block_ci(v, sub["block"].to_numpy(), rng,
                                             args.bootstrap_draws)
        by_var.append(row)

    rho_var = stats.spearmanr(df[f"abserr_{primary}"], df["cross_tissue_var"])
    def _rank(a): return stats.rankdata(np.asarray(a, float))
    def _resid(y, x):
        A = np.column_stack([x, np.ones_like(x)])
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rl = _rank(df["cross_tissue_mean_beta"])
    partial_var = float(np.corrcoef(_resid(_rank(df[f"abserr_{primary}"]), rl),
                                    _resid(_rank(df["cross_tissue_var"]), rl))[0, 1])

    thresh = float(df[f"abserr_{primary}"].quantile(0.9))
    df["top_decile_error"] = df[f"abserr_{primary}"] >= thresh
    top, rest = df[df.top_decile_error], df[~df.top_decile_error]

    cgi_enrich = {}
    for cls in sorted(df["cgi_class"].dropna().unique()):
        p_top = float((top["cgi_class"] == cls).mean())
        p_rest = float((rest["cgi_class"] == cls).mean())
        cgi_enrich[str(cls)] = {
            "fraction_top_decile": p_top, "fraction_rest": p_rest,
            "log2_enrichment": float(np.log2(p_top / p_rest)) if p_rest > 0 and p_top > 0 else None,
            "n_top": int((top["cgi_class"] == cls).sum()),
        }

    chrom_contrast = {}
    for c in CHROMATIN:
        if c not in df.columns or df[c].isna().all():
            continue
        t, r = top[c].dropna().to_numpy(float), rest[c].dropna().to_numpy(float)
        if len(t) < 10 or len(r) < 10:
            continue
        chrom_contrast[c] = {
            "median_top_decile": float(np.median(t)),
            "median_rest": float(np.median(r)),
            "mann_whitney_p": float(stats.mannwhitneyu(t, r, alternative="two-sided").pvalue),
            "rank_biserial": float(2 * stats.mannwhitneyu(t, r).statistic / (len(t) * len(r)) - 1),
        }

    top_var = float(top["cross_tissue_var"].median())
    rest_var = float(rest["cross_tissue_var"].median())

    payload = {
        "analysis": "where zero-shot cross-tissue transfer fails",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "single_seed_note": "Seed 42 only, by design; this localises one model's errors.",
        "model": str(args.pred_root),
        "probes": int(len(df)),
        "arms": arms,
        "primary_arm": primary,
        "split_discipline": (
            "Held-out TEST probes, chr8+chr9. Task C's plasticity null is on joint "
            "VALIDATION probes, chr10+chr11. The two probe sets are disjoint; "
            "cross-tissue variance here is recomputed from the four tissues' own "
            "test.csv by the same method. Both valid, not the same analysis."),
        "bootstrap": f"{args.bootstrap_draws} draws over 1 Mb blocks",
        "error_vs_variance": {
            "spearman_rho": float(rho_var.statistic),
            "spearman_p": float(rho_var.pvalue),
            "partial_spearman_given_mean_beta": partial_var,
            "decile_table": by_var,
        },
        "top_decile_error": {
            "threshold_abs_beta_error": thresh,
            "n": int(len(top)),
            "cross_tissue_var_median_top": top_var,
            "cross_tissue_var_median_rest": rest_var,
            "fold_difference": float(top_var / rest_var) if rest_var > 0 else None,
            "cgi_class_enrichment": cgi_enrich,
            "chromatin_contrast": chrom_contrast,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "transfer_failure_summary.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(out)
    keep = ["probeID", "chr", "pos", "true_beta", "cgi_class", "cross_tissue_var",
            "cross_tissue_sd", "cross_tissue_mean_beta", "top_decile_error"] + \
           [f"abserr_{a}" for a in arms] + CHROMATIN
    csv = args.output_dir / "transfer_failure_per_probe.csv"
    tmpc = csv.with_suffix(".csv.tmp")
    df[[c for c in keep if c in df.columns]].to_csv(tmpc, index=False)
    tmpc.replace(csv)

    print("\n" + "=" * 78)
    print(f"transfer failure, {len(df):,} held-out breast test probes (chr8+chr9)")
    print(f"primary arm: {primary}")
    print("-" * 78)
    print(f"  abs error vs cross-tissue variance:  rho {rho_var.statistic:+.4f}"
          f"   partial(|mean beta) {partial_var:+.4f}")
    print(f"\n  decile of cross-tissue variance -> MAE:")
    for r in by_var:
        cis = "  ".join(f"{a[:3]} {r[f'mae_{a}']:.4f}" for a in arms)
        print(f"    d{r['decile']:<2d} var {r['cross_tissue_var_median']:.5f}  {cis}  n={r['n']}")
    print(f"\n  top-decile error (>= {thresh:.4f} beta), n={len(top):,}:")
    print(f"    cross-tissue var median  {top_var:.5f} vs {rest_var:.5f} elsewhere"
          f"  ({top_var/rest_var:.2f}x)" if rest_var > 0 else "")
    print("    CpG-island class enrichment (log2 top vs rest):")
    for k, v in sorted(cgi_enrich.items(), key=lambda kv: -(kv[1]["log2_enrichment"] or -9)):
        l = v["log2_enrichment"]
        print(f"      {k:10s} {v['fraction_top_decile']*100:5.1f}% vs {v['fraction_rest']*100:5.1f}%"
              f"   log2 {l:+.3f}" if l is not None else f"      {k}: n/a")
    print("    chromatin (median top vs rest, rank-biserial):")
    for k, v in sorted(chrom_contrast.items(), key=lambda kv: -abs(kv[1]["rank_biserial"])):
        print(f"      {k:24s} {v['median_top_decile']:7.4f} vs {v['median_rest']:7.4f}"
              f"   rb {v['rank_biserial']:+.4f}  p={v['mann_whitney_p']:.2e}")
    print("=" * 78)
    print(f"wrote {out}\nwrote {csv}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-root", type=Path,
                   default=Path("results/journal/joint/holdout_BreastEpithelium/seed42"))
    p.add_argument("--cgi-manifest", type=Path,
                   default=Path("data/HM450.hg38.manifest.CpGIsland.tsv.gz"))
    p.add_argument("--breast-test-csv", type=Path,
                   default=Path("data/datafiles_breast_epithelium/test.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-draws", type=int, default=400)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/joint/transfer_failure"))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
