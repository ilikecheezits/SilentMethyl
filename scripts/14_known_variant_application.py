#!/usr/bin/env python3
"""Score a published SNV against every model-visible SilentMethyl CpG.

The script is an illustrative application, not an external validation test. It
starts from a minimal variant table (or a built-in MLH1 example), locates every
unmasked HM450 CpG for which the SNV lies inside the trained 1,000-bp window,
constructs WT and MUT sequences, reuses the frozen fusion scoring code, and
reports the split membership of every target CpG.  The nearest eligible CpG is
chosen before model scores are inspected and is marked as the primary display
target.  If no CpG is model-visible, the visibility audit is still written and
the script exits successfully without inventing an application result.

Required custom-variant columns are:
Variant_ID, Gene, chr, Position_1based, Ref, Alt. Optional descriptive columns
are Citation_Key, Source_URL, ClinVar_URL, Transcript_Annotation,
Gene_Function, Disease_Context, and Reported_Biological_Evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


LOGGER = logging.getLogger("silentmethyl.known_variant")
FULL_TARGET_C_INDEX = 2499
MODEL_TARGET_C_INDEX = 499
MODEL_WINDOW_SIZE = 1000

DEFAULT_VARIANTS = pd.DataFrame(
    [
        {
            "Variant_ID": "MLH1_c.27G>A",
            "Gene": "MLH1",
            "chr": "chr3",
            "Position_1based": 36993574,
            "Ref": "G",
            "Alt": "A",
            "Citation_Key": "alvarez2025mlh1",
            "Source_URL": "https://pubmed.ncbi.nlm.nih.gov/40715574/",
            "ClinVar_URL": "https://www.ncbi.nlm.nih.gov/clinvar/RCV000166655/",
            "Transcript_Annotation": "NM_000249.4:c.27G>A (p.Arg9=)",
            "Gene_Function": "MLH1 DNA mismatch repair",
            "Disease_Context": "Lynch syndrome and hereditary cancer predisposition",
            "Reported_Biological_Evidence": (
                "The c.27A allele was reported in cis with variably mosaic "
                "constitutional MLH1 methylation."
            ),
        }
    ]
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant-csv",
        type=Path,
        default=None,
        help="Optional custom variant list. If omitted, the built-in MLH1 example is used.",
    )
    parser.add_argument(
        "--split-template",
        default="data/datafiles/{split}.csv",
        help="Template for train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--candidate-scorer",
        type=Path,
        default=Path("scripts/05_matched_background.py"),
        help="Active candidate script whose frozen inference functions are reused.",
    )
    parser.add_argument(
        "--context-script",
        type=Path,
        default=Path("scripts/12_biological_context_analysis.py"),
        help="Active context-analysis script whose annotation functions are reused.",
    )
    parser.add_argument(
        "--hm450-manifest",
        type=Path,
        default=Path("data/HM450.hg38.manifest.tsv.gz"),
    )
    parser.add_argument(
        "--cpg-island-annotation",
        type=Path,
        default=Path("data/HM450.hg38.manifest.CpGIsland.tsv.gz"),
    )
    parser.add_argument(
        "--gencode-gtf",
        type=Path,
        default=Path("data/reference/gencode.v44.annotation.gtf.gz"),
    )
    parser.add_argument(
        "--weights-template",
        default="checkpoints_journal/seed{seed}/fusion/best_weights.pth",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--model-path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--local-model-dir", default="./dnabert2_local")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/journal/known_variant_application"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_module(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def annotate_target_cpgs(
    cohort: pd.DataFrame,
    context_module,
    cpg_island_path: Path,
    gencode_gtf: Path,
) -> pd.DataFrame:
    """Attach target-CpG gene-region and CpG-island annotations."""
    target = cohort[["probeID", "chr", "pos"]].drop_duplicates("probeID").copy()
    island = context_module.load_island_annotation(cpg_island_path)
    target = target.merge(island, on="probeID", how="left", validate="one_to_one")
    target["CpG_Island_Context"] = target["CpG_Island_Context"].fillna(
        "Unclassified"
    )
    target = context_module.annotate_genomic_region(target, gencode_gtf)
    annotated = cohort.copy()
    region_map = target.set_index("probeID")["Genomic_Region"]
    island_map = target.set_index("probeID")["CpG_Island_Context"]
    annotated["Genomic_Region"] = annotated["probeID"].map(region_map)
    annotated["CpG_Island_Context"] = annotated["probeID"].map(island_map)
    if annotated[["Genomic_Region", "CpG_Island_Context"]].isna().any().any():
        raise ValueError("Could not annotate every model-visible target CpG")
    return annotated


def normalize_chr(values: pd.Series) -> pd.Series:
    values = values.astype(str).str.strip()
    return values.where(values.str.startswith("chr"), "chr" + values)


def load_variants(path: Path | None) -> pd.DataFrame:
    frame = DEFAULT_VARIANTS.copy() if path is None else pd.read_csv(path)
    required = ["Variant_ID", "Gene", "chr", "Position_1based", "Ref", "Alt"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Variant table is missing columns: {missing}")
    frame = frame.copy()
    frame["chr"] = normalize_chr(frame["chr"])
    frame["Position_1based"] = pd.to_numeric(
        frame["Position_1based"], errors="raise"
    ).astype(np.int64)
    for column in ("Ref", "Alt"):
        frame[column] = frame[column].astype(str).str.upper().str.strip()
        if (~frame[column].str.fullmatch("[ACGT]")).any():
            raise ValueError(f"{column} must contain one A/C/G/T base per row")
    if (frame["Ref"] == frame["Alt"]).any():
        raise ValueError("REF and ALT must differ")
    if frame["Variant_ID"].astype(str).duplicated().any():
        raise ValueError("Variant_ID values must be unique")
    for column in (
        "Citation_Key",
        "Source_URL",
        "ClinVar_URL",
        "Transcript_Annotation",
        "Gene_Function",
        "Disease_Context",
        "Reported_Biological_Evidence",
    ):
        if column not in frame:
            frame[column] = ""
    return frame


def load_loci(split_template: str, scorer) -> tuple[pd.DataFrame, dict[str, str]]:
    required = [
        "probeID",
        "chr",
        "pos",
        "Healthy_5000bp_DNA",
        *scorer.TABULAR_FEATURES,
        *scorer.MISSING_FEATURES,
    ]
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = Path(split_template.format(split=split))
        if not path.is_file():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0)
        missing = [column for column in required if column not in header.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = pd.read_csv(path, usecols=required)
        frame["Model_Split"] = split
        frames.append(frame)
        hashes[str(path)] = sha256(path)
    loci = pd.concat(frames, ignore_index=True)
    loci["probeID"] = loci["probeID"].astype(str)
    loci["chr"] = normalize_chr(loci["chr"])
    loci["pos"] = pd.to_numeric(loci["pos"], errors="raise").astype(np.int64)
    if loci["probeID"].duplicated().any():
        raise ValueError("A probeID appears in more than one model split")
    return loci, hashes


def build_pairs(
    variants: pd.DataFrame,
    loci: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    audit_rows: list[dict] = []
    for variant in variants.itertuples(index=False):
        chromosome_loci = loci[loci["chr"] == variant.chr].copy()
        variant_pos0 = int(variant.Position_1based) - 1
        chromosome_loci["Offset_From_Target_CpG_C"] = variant_pos0 - chromosome_loci["pos"]
        nearby = chromosome_loci[
            chromosome_loci["Offset_From_Target_CpG_C"].between(-499, 500)
        ].copy()
        counters = {
            "Variant_ID": str(variant.Variant_ID),
            "Gene": str(variant.Gene),
            "Chromosome": str(variant.chr),
            "Position_1based": int(variant.Position_1based),
            "Nearby_HM450_CpGs_Before_Allele_Audit": int(len(nearby)),
            "Reference_Allele_Mismatches": 0,
            "Protected_Target_CpG_Changes": 0,
            "Constructed_Pairs_Before_Probe_QC": 0,
        }
        for locus in nearby.itertuples(index=False):
            sequence = str(locus.Healthy_5000bp_DNA).upper()
            if len(sequence) != 5000 or sequence[FULL_TARGET_C_INDEX:FULL_TARGET_C_INDEX + 2] != "CG":
                continue
            offset = int(locus.Offset_From_Target_CpG_C)
            mutation_index = FULL_TARGET_C_INDEX + offset
            model_index = MODEL_TARGET_C_INDEX + offset
            if not (0 <= mutation_index < len(sequence)):
                continue
            if sequence[mutation_index] != variant.Ref:
                counters["Reference_Allele_Mismatches"] += 1
                continue
            if model_index in {MODEL_TARGET_C_INDEX, MODEL_TARGET_C_INDEX + 1}:
                counters["Protected_Target_CpG_Changes"] += 1
                continue
            mutated = sequence[:mutation_index] + variant.Alt + sequence[mutation_index + 1:]
            record = locus._asdict()
            record.update(
                {
                    "Candidate_ID": f"{variant.Variant_ID}|{locus.probeID}",
                    "Published_Variant_ID": str(variant.Variant_ID),
                    "Selected_Gene_Name": str(variant.Gene),
                    "GDC_Genomic_DNA_Change": (
                        f"{variant.chr}:g.{int(variant.Position_1based)}"
                        f"{variant.Ref}>{variant.Alt}"
                    ),
                    "Variant_Position_1based": int(variant.Position_1based),
                    "Reference_Allele": str(variant.Ref),
                    "Alternate_Allele": str(variant.Alt),
                    "Mutated_5000bp_DNA": mutated,
                    "Citation_Key": str(variant.Citation_Key),
                    "Source_URL": str(variant.Source_URL),
                    "ClinVar_URL": str(variant.ClinVar_URL),
                    "Transcript_Annotation": str(variant.Transcript_Annotation),
                    "Gene_Function": str(variant.Gene_Function),
                    "Disease_Context": str(variant.Disease_Context),
                    "Reported_Biological_Evidence": str(
                        variant.Reported_Biological_Evidence
                    ),
                }
            )
            records.append(record)
        counters["Constructed_Pairs_Before_Probe_QC"] = int(
            sum(item["Published_Variant_ID"] == str(variant.Variant_ID) for item in records)
        )
        audit_rows.append(counters)
    return pd.DataFrame(records), pd.DataFrame(audit_rows)


def choose_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mark_primary_target(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Is_Primary_Nearest_Target"] = False
    for _, group in frame.groupby("Published_Variant_ID", sort=False):
        ordered = group.sort_values(
            ["Absolute_Distance_From_Target_CpG", "probeID"],
            kind="mergesort",
        )
        frame.loc[ordered.index[0], "Is_Primary_Nearest_Target"] = True
    return frame


def plot_primary(frame: pd.DataFrame, output: Path) -> None:
    primary = frame[frame["Is_Primary_Nearest_Target"].astype(bool)].copy()
    if primary.empty:
        return
    primary.sort_values(["Published_Variant_ID", "Seed"], inplace=True)
    variants = list(dict.fromkeys(primary["Published_Variant_ID"].astype(str)))
    fig, axis = plt.subplots(figsize=(3.45, max(2.35, 0.7 * len(variants) + 1.5)))
    for position, variant_id in enumerate(variants):
        group = primary[primary["Published_Variant_ID"].astype(str) == variant_id]
        values = pd.to_numeric(group["Predicted_Delta_Beta"], errors="raise").to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        axis.scatter(values, position + jitter, color="0.45", s=22, zorder=2)
        axis.errorbar(
            float(np.mean(values)),
            position,
            xerr=float(np.std(values, ddof=0)),
            fmt="D",
            color="#D95F02",
            capsize=2.5,
            markersize=5,
            zorder=3,
        )
    axis.axvline(0, color="0.45", linestyle="--", linewidth=0.8)
    axis.set_yticks(np.arange(len(variants)), variants)
    axis.invert_yaxis()
    axis.set_xlabel(r"Predicted MUT $-$ WT $\Delta\hat{\beta}$")
    axis.set_title("Published-variant application")
    axis.grid(axis="x", alpha=0.16)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = load_module(
        args.candidate_scorer.resolve(), "silentmethyl_candidate_scorer"
    )
    context_module = load_module(
        args.context_script.resolve(), "silentmethyl_context_analysis"
    )
    variants = load_variants(args.variant_csv)
    loci, split_hashes = load_loci(args.split_template, scorer)
    raw_pairs, visibility = build_pairs(variants, loci)
    atomic_csv(visibility, args.output_dir / "known_variant_visibility_audit.csv")
    atomic_csv(variants, args.output_dir / "known_variant_input.csv")

    if raw_pairs.empty:
        atomic_json(
            {
                "analysis_status": "NO_MODEL_VISIBLE_CPG",
                "interpretation": (
                    "No eligible HM450 CpG placed the prespecified variant inside "
                    "the trained sequence window; no prediction was manufactured."
                ),
                "input_sha256": split_hashes,
            },
            args.output_dir / "run_summary.json",
        )
        print("No model-visible CpG was found; wrote the visibility audit.")
        return

    raw_pairs = scorer.apply_probe_qc(
        raw_pairs,
        str(args.hm450_manifest),
        include_masked=False,
    )
    if raw_pairs.empty:
        visibility["Status_After_Probe_QC"] = "all nearby probes masked"
        atomic_csv(visibility, args.output_dir / "known_variant_visibility_audit.csv")
        atomic_json(
            {
                "analysis_status": "NO_UNMASKED_MODEL_VISIBLE_CPG",
                "interpretation": "All model-visible target probes failed HM450 probe QC.",
            },
            args.output_dir / "run_summary.json",
        )
        print("All model-visible CpGs were masked; wrote the visibility audit.")
        return

    cohort, wt, mut, tab, missing = scorer.build_model_visible_cohort(
        raw_pairs,
        MODEL_WINDOW_SIZE,
    )
    cohort = annotate_target_cpgs(
        cohort,
        context_module,
        args.cpg_island_annotation,
        args.gencode_gtf,
    )
    cohort = mark_primary_target(cohort)
    atomic_csv(cohort, args.output_dir / "known_variant_model_visible_cpgs.csv")

    device = choose_device(args.device)
    tokenizer = scorer.get_tokenizer(args.model_path)
    scorer_args = argparse.Namespace(
        model_path=args.model_path,
        local_model_dir=args.local_model_dir,
        batch_size=args.batch_size,
        amp=args.amp,
    )
    seed_outputs: list[pd.DataFrame] = []
    checkpoint_hashes: dict[str, str] = {}
    for seed in dict.fromkeys(int(value) for value in args.seeds):
        checkpoint = Path(args.weights_template.format(seed=seed))
        scored = scorer.score_seed(
            seed,
            str(checkpoint),
            cohort,
            wt,
            mut,
            tab,
            missing,
            scorer_args,
            device,
            tokenizer,
        )
        seed_outputs.append(scored)
        checkpoint_hashes[str(checkpoint)] = sha256(checkpoint)
    long_scores = pd.concat(seed_outputs, ignore_index=True)
    atomic_csv(long_scores, args.output_dir / "known_variant_predictions_all_seeds.csv")

    ensemble = scorer.aggregate_seeds(long_scores, cohort)
    ensemble = mark_primary_target(ensemble)
    ensemble["Application_Interpretation"] = np.where(
        ensemble["Model_Split"].astype(str) == "test",
        "Illustrative application at a held-out target CpG; not a causal experiment.",
        "Illustrative application; the target CpG was available during model development.",
    )
    atomic_csv(ensemble, args.output_dir / "known_variant_predictions_ensemble.csv")
    primary = ensemble[ensemble["Is_Primary_Nearest_Target"].astype(bool)].copy()
    atomic_csv(primary, args.output_dir / "known_variant_primary_case.csv")
    plot_primary(
        long_scores,
        args.output_dir / "known_variant_primary_case.png",
    )

    atomic_json(
        {
            "analysis_status": "COMPLETE",
            "variant_count": int(variants["Variant_ID"].nunique()),
            "model_visible_cpg_pairs": int(len(cohort)),
            "primary_case_rows": int(len(primary)),
            "selection_rule": (
                "Nearest unmasked model-visible HM450 CpG, with probeID used only "
                "as a deterministic tie breaker; model scores were not used."
            ),
            "interpretation": (
                "Illustrative application of frozen checkpoints. Split membership "
                "is retained so training-locus examples are not presented as validation."
            ),
            "input_sha256": {
                **split_hashes,
                str(args.hm450_manifest): sha256(args.hm450_manifest),
                str(args.candidate_scorer): sha256(args.candidate_scorer),
                str(args.context_script): sha256(args.context_script),
                str(args.cpg_island_annotation): sha256(
                    args.cpg_island_annotation
                ),
                str(args.gencode_gtf): sha256(args.gencode_gtf),
                **checkpoint_hashes,
            },
        },
        args.output_dir / "run_summary.json",
    )
    print("SilentMethyl known-variant application")
    print(f"  model-visible CpG pairs: {len(cohort)}")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
