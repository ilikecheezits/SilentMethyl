#!/usr/bin/env python3
"""Check the per-tissue builds for consistency before joint training. Catches failures that
are silent downstream: a build pointed at the wrong reference or target matrix, detected
as two tissues that should differ coming out identical; per-tissue scale or missingness
offsets a joint model could use as a tissue label; and tissues that do not hold out the
same chromosomes. The scale check matters because the context tower applies no per-
feature standardisation, so raw bigWig magnitude reaches the first linear layer
directly. Exit code is 0 only if no hard failure fired; warnings do not fail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_TISSUES = {
    "BreastEpithelium": SCRIPT_DIR / "datafiles_breast_epithelium",
    "KidneyCortex": SCRIPT_DIR / "datafiles_multitissue" / "KidneyCortex",
    "Lung": SCRIPT_DIR / "datafiles_multitissue" / "Lung",
    "ColonTransverse": SCRIPT_DIR / "datafiles_multitissue" / "ColonTransverse",
}

CONTEXT_FEATURES = [
    "Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal",
]
PHYLOP_FEATURES = ["Target_Base_PhyloP_100way_1", "Target_Base_PhyloP_100way_2"]

SCALE_RATIO_WARN = 3.0
SCALE_RATIO_FAIL = 10.0

IDENTITY_CORR = 0.999


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  FAIL  {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  WARN  {msg}")

    def ok(self, msg: str) -> None:
        print(f"  ok    {msg}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissue", action="append", default=None, metavar="NAME=DIR")
    ap.add_argument("--sample", type=int, default=50_000,
                    help="Probes sampled for the pairwise comparisons "
                         "(default 50000). Full files are used for schema, "
                         "split and missingness checks.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None,
                    help="Also write the full report here.")
    return ap.parse_args()


def resolve(overrides) -> dict[str, Path]:
    tissues = dict(DEFAULT_TISSUES)
    for item in overrides or []:
        name, _, path = item.partition("=")
        tissues[name.strip()] = Path(path.strip()).resolve()
    return tissues


def main() -> None:
    args = parse_args()
    tissues = resolve(args.tissue)
    rep = Report()
    summary: dict = {"tissues": {}, "checks": {}}

    print("\n[1] builds present and complete")
    for name, d in tissues.items():
        missing = [f for f in ("train.csv", "val.csv", "test.csv",
                               "split_manifest.json", "feature_imputation.json",
                               "training_data_manifest.json")
                   if not (d / f).is_file()]
        if missing:
            rep.fail(f"{name}: missing {', '.join(missing)} in {d}")
        else:
            rep.ok(f"{name}: all files present")
    if rep.failures:
        raise SystemExit("\nSTOP: finish the builds before auditing.")

    print("\n[2] each tissue read its own reference directory")
    for name, d in tissues.items():
        man = json.loads((d / "training_data_manifest.json").read_text())
        paths = {k: v["path"] for k, v in man["inputs"]["bigwigs"].items()}
        ctx_dirs = {Path(p).parent.name for k, p in paths.items()
                    if k in CONTEXT_FEATURES}
        matrices = [Path(m["path"]).name for m in
                    man["inputs"].get("methylation_matrices",
                                      [man["inputs"]["methylation_matrix"]])]
        summary["tissues"][name] = {"context_dirs": sorted(ctx_dirs),
                                    "matrices": matrices}
        if len(ctx_dirs) != 1:
            rep.fail(f"{name}: context tracks come from several directories "
                     f"{sorted(ctx_dirs)}")
        else:
            got = ctx_dirs.pop()
            note = "" if got.lower().startswith(name.lower()[:5]) else \
                   "   <- does this match the tissue name?"
            rep.ok(f"{name:18s} context={got:<20s} targets={', '.join(matrices)}{note}")
        phylo = {Path(paths[f]).name for f in PHYLOP_FEATURES if f in paths}
        if phylo != {"hg38.phyloP100way.bw"}:
            rep.fail(f"{name}: phyloP is not the shared genome track: {phylo}")

    print("\n[3] held-out chromosomes agree across tissues")
    blocks = {}
    for name, d in tissues.items():
        m = json.loads((d / "split_manifest.json").read_text())
        blocks[name] = (tuple(sorted(m["validation_chromosomes"])),
                        tuple(sorted(m["test_chromosomes"])))
    if len(set(blocks.values())) == 1:
        val, test = next(iter(blocks.values()))
        rep.ok(f"val={list(val)}  test={list(test)} in all {len(tissues)} tissues")
        summary["checks"]["held_out"] = {"val": list(val), "test": list(test)}
    else:
        for name, b in blocks.items():
            print(f"        {name}: val={list(b[0])} test={list(b[1])}")
        rep.fail("tissues do not share held-out chromosomes -- a probe held out "
                 "in one is trained on in another, contaminating the joint test set")

    print("\n[4] loading train splits")
    frames: dict[str, pd.DataFrame] = {}
    for name, d in tissues.items():
        frames[name] = pd.read_csv(d / "train.csv")
        print(f"        {name:18s} {len(frames[name]):>8,} rows  "
              f"{len(frames[name].columns)} columns")

    print("\n[5] schema is identical across tissues (required to concatenate)")
    schemas = {n: tuple(f.columns) for n, f in frames.items()}
    if len(set(schemas.values())) == 1:
        rep.ok(f"all tissues share {len(next(iter(schemas.values())))} columns")
    else:
        ref_name, ref_cols = next(iter(schemas.items()))
        for n, cols in schemas.items():
            if cols != ref_cols:
                extra = set(cols) - set(ref_cols)
                lack = set(ref_cols) - set(cols)
                rep.fail(f"{n} schema differs from {ref_name}: "
                         f"extra={sorted(extra)} missing={sorted(lack)}")

    present = [f for f in CONTEXT_FEATURES if f in next(iter(frames.values())).columns]
    if len(present) != len(CONTEXT_FEATURES):
        rep.fail(f"expected 7 context features, found {len(present)}: {present}")

    print("\n[6] no probe appears in two splits of the same tissue")
    for name, d in tissues.items():
        ids = {}
        for split in ("train", "val", "test"):
            f = frames[name] if split == "train" else pd.read_csv(
                d / f"{split}.csv", usecols=["probeID"])
            ids[split] = set(f["probeID"].astype(str))
        bad = (ids["train"] & ids["val"]) | (ids["train"] & ids["test"]) | \
              (ids["val"] & ids["test"])
        if bad:
            rep.fail(f"{name}: {len(bad)} probes appear in more than one split")
        else:
            rep.ok(f"{name}: splits are disjoint "
                   f"({len(ids['train']):,}/{len(ids['val']):,}/{len(ids['test']):,})")

    print("\n[7] shared probe universe")
    universe = None
    for f in frames.values():
        s = set(f["probeID"].astype(str))
        universe = s if universe is None else (universe & s)
    shared = sorted(universe)
    print(f"        {len(shared):,} probes covered in every tissue")
    for name, f in frames.items():
        drop = len(f) - len(shared)
        print(f"        {name:18s} drops {drop:>6,} "
              f"({drop / len(f) * 100:.2f}%) not covered everywhere")
    summary["checks"]["shared_universe"] = len(shared)
    if len(shared) < 0.95 * min(len(f) for f in frames.values()):
        rep.warn("shared universe is under 95% of the smallest tissue -- "
                 "the joint model will train on noticeably fewer probes")
    else:
        rep.ok("shared universe is within 5% of every tissue")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(shared), size=min(args.sample, len(shared)),
                      replace=False)
    probes = pd.Index([shared[i] for i in sorted(pick)])
    aligned = {n: f.set_index(f["probeID"].astype(str)).reindex(probes)
               for n, f in frames.items()}

    print("\n[8] the same probe has the same DNA in every tissue")
    seq_col = "Healthy_100bp_DNA"
    if seq_col in next(iter(aligned.values())).columns:
        ref_name, ref = next(iter(aligned.items()))
        for n, f in aligned.items():
            if n == ref_name:
                continue
            diff = int((f[seq_col].values != ref[seq_col].values).sum())
            if diff:
                rep.fail(f"{n} vs {ref_name}: {diff} probes have different "
                         "sequence -- the builds are not on the same genome "
                         "or the probe IDs do not mean the same thing")
        if not rep.failures or all("sequence" not in x for x in rep.failures):
            rep.ok(f"sequence identical across tissues on {len(probes):,} probes")
    else:
        rep.warn(f"{seq_col} not in the CSVs; skipped")

    print("\n[9] targets differ between tissues (not the same matrix twice)")
    names = list(aligned)
    tgt_corr = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = aligned[a]["Median_Beta"].to_numpy(float)
            y = aligned[b]["Median_Beta"].to_numpy(float)
            r = float(np.corrcoef(x, y)[0, 1])
            tgt_corr[f"{a}|{b}"] = r
            same = np.allclose(x, y, atol=1e-6)
            flag = "  <- IDENTICAL" if same or r > IDENTITY_CORR else ""
            print(f"        {a[:16]:16s} vs {b[:16]:16s}  r={r:.4f}{flag}")
            if same or r > IDENTITY_CORR:
                rep.fail(f"{a} and {b} have effectively identical targets "
                         "-- one build almost certainly read the wrong matrix")
    summary["checks"]["target_correlation"] = tgt_corr
    if all(r <= IDENTITY_CORR for r in tgt_corr.values()):
        rep.ok(f"every pair differs; r ranges "
               f"{min(tgt_corr.values()):.3f}-{max(tgt_corr.values()):.3f}")

    print("\n[10] context differs between tissues (not the same tracks twice)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            identical = [f for f in present
                         if np.allclose(aligned[a][f].fillna(-999).to_numpy(float),
                                        aligned[b][f].fillna(-999).to_numpy(float),
                                        atol=1e-9)]
            if identical:
                rep.fail(f"{a} and {b} share identical values for "
                         f"{', '.join(identical)} -- same bigWig read twice")
    if not any("identical values" in x for x in rep.failures):
        rep.ok("no feature is bit-identical between any two tissues")

    print("\n[11] feature SCALE is comparable across tissues")
    print("     The context tower has no per-feature standardization, so a "
          "feature on a\n     different scale in one tissue is a label the "
          "model can route on.\n")
    scale: dict[str, dict] = {}
    header = f"     {'feature':<24}" + "".join(f"{n[:12]:>13s}" for n in names) + f"{'ratio':>9s}"
    print(header)
    for feat in present + [f for f in PHYLOP_FEATURES if f in aligned[names[0]].columns]:
        means = {}
        for n in names:
            v = aligned[n][feat].to_numpy(float)
            means[n] = float(np.nanmean(v))
        finite = [m for m in means.values() if np.isfinite(m) and abs(m) > 1e-9]
        ratio = (max(finite) / min(finite)) if len(finite) == len(names) else float("inf")
        scale[feat] = {"means": means, "ratio": ratio}
        row = f"     {feat:<24}" + "".join(f"{means[n]:>13.4f}" for n in names)
        print(f"{row}{ratio:>9.1f}x")
    summary["checks"]["feature_scale"] = scale
    print()
    for feat, rec in scale.items():
        if feat in PHYLOP_FEATURES:
            continue
        if rec["ratio"] >= SCALE_RATIO_FAIL:
            rep.fail(f"{feat}: {rec['ratio']:.1f}x between tissues -- large "
                     "enough that magnitude alone identifies the tissue")
        elif rec["ratio"] >= SCALE_RATIO_WARN:
            rep.warn(f"{feat}: {rec['ratio']:.1f}x between tissues -- check "
                     "these are the same normalisation before training")
    if not any(r["ratio"] >= SCALE_RATIO_WARN for f, r in scale.items()
               if f not in PHYLOP_FEATURES):
        rep.ok(f"every context feature is within {SCALE_RATIO_WARN:g}x across tissues")

    print("\n[11b] phyloP is the SAME value per probe in every tissue")
    print("     Conservation is a property of the genome, not the tissue, so "
          "every build\n     reads one shared track. Comparing means would be "
          "too weak -- the values\n     must be equal probe by probe. Positions "
          "that were imputed are excluded,\n     since imputation uses each "
          "build's own train-set median.\n")
    ref_name = names[0]
    for feat in [f for f in PHYLOP_FEATURES if f in aligned[ref_name].columns]:
        miss_col = f"{feat}_Missing"
        usable = np.ones(len(probes), dtype=bool)
        for n in names:
            if miss_col in aligned[n].columns:
                usable &= aligned[n][miss_col].to_numpy() == 0
        if usable.sum() == 0:
            rep.warn(f"{feat}: every sampled probe was imputed somewhere; skipped")
            continue
        worst = 0.0
        for n in names[1:]:
            d = np.abs(aligned[n][feat].to_numpy(float)[usable]
                       - aligned[ref_name][feat].to_numpy(float)[usable])
            worst = max(worst, float(np.nanmax(d)))
        if worst > 1e-4:
            rep.fail(f"{feat}: differs by up to {worst:.4f} between tissues on "
                     "non-imputed probes -- a build read a different "
                     "conservation file, or the probe coordinates disagree")
        else:
            rep.ok(f"{feat}: identical across tissues on "
                   f"{int(usable.sum()):,} non-imputed probes")

    print("\n[12] missingness is comparable across tissues")
    print("     _Missing indicators are model inputs. A tissue with "
          "systematically more\n     uncovered positions carries its identity "
          "in those columns.\n")
    miss: dict[str, dict] = {}
    print(f"     {'feature':<24}" + "".join(f"{n[:12]:>13s}" for n in names))
    for feat in present:
        col = f"{feat}_Missing"
        if col not in frames[names[0]].columns:
            continue
        rates = {n: float(frames[n][col].mean()) for n in names}
        miss[feat] = rates
        print(f"     {feat:<24}" + "".join(f"{rates[n] * 100:>12.3f}%" for n in names))
    summary["checks"]["missing_rate"] = miss
    print()
    for feat, rates in miss.items():
        hi, lo = max(rates.values()), min(rates.values())
        if hi > 0.02 and hi - lo > 0.05:
            rep.fail(f"{feat}_Missing spans {lo*100:.1f}%-{hi*100:.1f}% "
                     "across tissues -- that gap identifies the tissue")
        elif hi - lo > 0.01:
            rep.warn(f"{feat}_Missing spans {lo*100:.2f}%-{hi*100:.2f}% "
                     "across tissues")
    if not miss:
        rep.warn("no _Missing columns found")
    elif not any(max(r.values()) - min(r.values()) > 0.01 for r in miss.values()):
        rep.ok("every feature's missing rate is within 1 point across tissues")

    print("\n[13] no NaN or infinity in model inputs")
    for n in names:
        cols = present + [f for f in PHYLOP_FEATURES if f in frames[n].columns]
        arr = frames[n][cols].to_numpy(float)
        bad = int((~np.isfinite(arr)).sum())
        if bad:
            rep.fail(f"{n}: {bad:,} non-finite values in context features "
                     "after imputation")
        else:
            rep.ok(f"{n}: all context features finite")

    print("\n" + "=" * 74)
    if rep.failures:
        print(f"{len(rep.failures)} FAILURE(S) -- do not train on this data")
        for f in rep.failures:
            print(f"  - {f}")
    else:
        print("No failures.")
    if rep.warnings:
        print(f"\n{len(rep.warnings)} warning(s) -- judgement calls, not blockers:")
        for w in rep.warnings:
            print(f"  - {w}")
    print("=" * 74)

    summary["failures"] = rep.failures
    summary["warnings"] = rep.warnings
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"report written to {args.json}")

    raise SystemExit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
