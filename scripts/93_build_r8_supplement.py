#!/usr/bin/env python3
"""
Assemble the R8 supplementary data sections (S7-S11). CPU only, seconds.

Why this is separate from 90_build_supplement_package.py
---------------------------------------------------------
Script 90 carries a hard-coded manifest of the published S1-S6 sections and
validates them on the way through. This adds the R8 analyses as S7-S11 into their
own package directory, leaving script 90 and the published sections untouched.
The two directories are merged at submission time; both follow the same layout
(numbered section folders, a README naming every file, and SHA256SUMS.txt).

Sections
--------
    S7   Gate decomposition and the sub-channel split        (Task A)
    S8   Context dose-response ladder                        (Task B)
    S9   Where zero-shot transfer fails                      (Task D)
    S10  Where the fusion gain concentrates                  (Task F)
    S11  Allele-specific methylation validation              (Task E1)

Missing inputs are reported, not fatal: S11 does not exist until the ASM job has
landed, and a partial package is more useful than a failed build. The README
states plainly which sections were omitted so a partial package cannot be
mistaken for a complete one.

Usage
-----
    python -u scripts/93_build_r8_supplement.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

JOURNAL = Path("results/journal")
ABL = JOURNAL / "ablation_breast_epithelium"


@dataclass(frozen=True)
class Item:
    section: str
    source: Path
    destination: str
    description: str


SECTION_TITLES = {
    "S7": ("Supplementary_Data_S7_Gate_Decomposition",
           "Gate decomposition of the fusion model's variant effect (Task A)"),
    "S8": ("Supplementary_Data_S8_Context_Ladder",
           "Context dose-response ladder and effect-size stratification (Task B)"),
    "S9": ("Supplementary_Data_S9_Transfer_Failure",
           "Where zero-shot cross-tissue transfer fails (Task D)"),
    "S10": ("Supplementary_Data_S10_Fusion_Gain_By_Region",
            "Where the fusion gain concentrates, by genomic region (Task F)"),
    "S11": ("Supplementary_Data_S11_ASM_Validation",
            "Allele-specific methylation validation (Task E1)"),
    "S12": ("Supplementary_Data_S12_ASM_Signed_Validation",
            "Allele-specific methylation, signed validation and replication "
            "(Tasks E1 + E2, Do & Tycko 2020)"),
}

ITEMS = (
    Item("S7", ABL / "gate_decomposition/gate_decomposition_summary.json",
         "S7_gate_decomposition_summary.json",
         "Variance decomposition of the fusion variant effect into DNA and gate "
         "channels, both cohorts, with the allele-invariance diagnostics."),

    Item("S8", ABL / "context_ladder/agreement_with_identity.csv",
         "S8_ladder_agreement_with_native_context.csv",
         "Agreement of each perturbation rung with the native-context run, for "
         "absolute methylation and for variant effects."),
    Item("S8", ABL / "context_ladder/level_accuracy_by_scheme.csv",
         "S8_ladder_level_accuracy.csv",
         "Held-out methylation accuracy under each context substitution."),
    Item("S8", ABL / "context_ladder/run_summary.json",
         "S8_ladder_run_summary.json",
         "Provenance and guard counters for the four-rung ladder."),
    Item("S8", ABL / "context_ladder_stratified/ladder_effect_size_stratification.csv",
         "S8_ladder_by_observed_effect_size.csv",
         "Ladder dissociation stratified by OBSERVED effect size (beta_ref_to_alt), "
         "the bias-free stratifier."),
    Item("S8", ABL / "context_ladder_stratified/ladder_effect_size_stratification_predicted.csv",
         "S8_ladder_by_predicted_effect_size_ARTEFACT.csv",
         "The same stratification binned on the model's OWN predicted effect. "
         "Included only to document the regression-to-the-mean artefact it "
         "produces; it is NOT a result."),
    Item("S8", ABL / "context_ladder_stratified/run_summary.json",
         "S8_ladder_stratification_summary.json",
         "Provenance for the effect-size stratification."),

    Item("S9", JOURNAL / "joint/transfer_failure/transfer_failure_summary.json",
         "S9_transfer_failure_summary.json",
         "Error-versus-plasticity deciles, CpG-island enrichment and chromatin "
         "contrasts for the top error decile."),
    Item("S9", JOURNAL / "joint/transfer_failure/transfer_failure_per_probe.csv",
         "S9_transfer_failure_per_probe.csv",
         "Per-probe held-out errors and cross-tissue variance."),

    Item("S10", ABL / "fusion_gain_stratified/fusion_gain_stratified.csv",
         "S10_fusion_gain_stratified.csv",
         "Paired fusion-minus-sequence beta MAE and AUROC by CpG-island context "
         "and by all seven chromatin tracks, with 1 Mb block-bootstrap intervals. "
         "Compare strata on Relative_Beta_MAE_Reduction, not the absolute column."),
    Item("S10", ABL / "fusion_gain_stratified/run_summary.json",
         "S10_fusion_gain_summary.json",
         "Provenance and the absolute-versus-relative caveat."),
    Item("S10", ABL / "biological_context/fusion_gain_by_context.csv",
         "S10_fusion_gain_by_context_prior.csv",
         "The earlier two-track stratification, retained so the newer all-track "
         "table can be checked against it."),

    Item("S11", JOURNAL / "asm_validation/asm_discrimination.csv",
         "S11_asm_discrimination.csv",
         "AUROC separating ASM SNV-CpG pairs from distance-matched non-ASM pairs, "
         "per contrast, stratum and model arm, with block-bootstrap intervals."),
    Item("S11", JOURNAL / "asm_validation/asm_matching_balance.csv",
         "S11_asm_matching_balance.csv",
         "Variant-to-CpG distance balance between matched positives and negatives."),
    Item("S11", JOURNAL / "asm_validation/evaluation_summary.json",
         "S11_asm_evaluation_summary.json",
         "Provenance and the distance-baseline sanity check. This catalogue "
         "publishes ASM significance but no signed allelic difference, so "
         "direction concordance and signed Spearman are not computable here; "
         "they are in S12, from Do & Tycko 2020."),
    Item("S11", Path("data/external/asm_atlas/scoring/build_summary.json"),
         "S11_asm_build_summary.json",
         "How the ASM pairs were constructed: liftOver counters, window "
         "arithmetic, positive/negative tier definitions and counts."),
    Item("S11", Path("data/external/asm_atlas_natcommun2025/SOURCE.txt"),
         "S11_asm_source_provenance.txt",
         "Citation, download URLs and contents of the source ASM catalogue."),

    Item("S12", JOURNAL / "asm_validation_tycko/tycko_e2_signed_agreement.csv",
         "S12_asm_signed_agreement.csv",
         "Direction concordance, signed Spearman and positive-versus-negative "
         "AUROC per ASM index SNP (n=722), by stratum and model arm, with 1 Mb "
         "block-bootstrap intervals. The unit is one SNP: the prediction is the "
         "mean predicted delta over the CpGs scored inside that SNP's ASM DMR, "
         "matching how the published effect was averaged. The mammary stratum "
         "(n=53) is underpowered and is not a headline. The |effect| >= 20 pp "
         "stratum is post-hoc and was not pre-registered."),
    Item("S12", JOURNAL / "asm_validation_tycko/tycko_e1_discrimination.csv",
         "S12_asm_discrimination_replication.csv",
         "Discrimination of ASM CpGs from distance-matched non-DMR CpGs in the "
         "second, independent catalogue, with the distance-only null baseline. "
         "Compare against S11: the two catalogues' estimates lie inside each "
         "other's confidence intervals."),
    Item("S12", JOURNAL / "asm_validation_tycko/tycko_evaluation_summary.json",
         "S12_asm_signed_evaluation_summary.json",
         "Provenance, bootstrap settings, and the scale caveat: observed effects "
         "are percentage points of methylation and predicted deltas are on the "
         "model's M scale, so only sign, rank and AUROC are compared and no "
         "magnitude calibration is claimed."),
    Item("S12", JOURNAL / "asm_validation_tycko/score_summary.json",
         "S12_asm_score_summary.json",
         "Scoring provenance: checkpoints and their hashes, seed, and the "
         "input-validation counters (reference-base mismatches, off-window and "
         "non-CpG-centred pairs, all zero)."),
    Item("S12", Path("data/external/asm_tycko_gb2020/scoring/build_summary.json"),
         "S12_asm_build_summary.json",
         "How the SNP-CpG pairs were built: the hg19 DMR liftOver, the GRCh38 "
         "SNP positions from Ensembl (deliberately NOT lifted), the multi-allelic "
         "exclusion, and the REF-versus-hg38 sign-convention check."),
    Item("S12", Path("data/external/asm_tycko_gb2020/SOURCE.txt"),
         "S12_asm_source_provenance.txt",
         "Citation, download URL and contents of the Do & Tycko 2020 catalogue."),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def readme_text(present: list[Item], missing: list[Item]) -> str:
    lines = [
        "# SilentMethyl supplementary data, sections S7-S11",
        "",
        "Companion to the published S1-S6 package built by",
        "`scripts/90_build_supplement_package.py`. These sections cover the R8",
        "analyses. Merge the two directories at submission.",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}.",
        "",
    ]
    if missing:
        lines += [
            "## INCOMPLETE PACKAGE",
            "",
            "The following files were not present when this package was built, so",
            "their sections are absent or partial. Rebuild once they exist:",
            "",
        ]
        lines += [f"- `{item.source}` (section {item.section})" for item in missing]
        lines.append("")

    by_section: dict[str, list[Item]] = {}
    for item in present:
        by_section.setdefault(item.section, []).append(item)

    for section in sorted(by_section, key=lambda s: int(s[1:])):
        folder, title = SECTION_TITLES[section]
        lines += [f"## {section} — {title}", "", f"Directory: `{folder}/`", ""]
        for item in by_section[section]:
            lines += [f"- **{item.destination}** — {item.description}"]
        lines.append("")

    lines += [
        "## Two cautions carried from the analyses",
        "",
        "1. **S10**: absolute fusion gain is bounded by the sequence-only error in",
        "   each stratum, so it ranks strata by how badly they start. Compare",
        "   strata on the relative reduction.",
        "2. **S11**: the source catalogue publishes significance but no signed",
        "   allelic methylation difference, so direction concordance and signed",
        "   Spearman are not available and are not reported.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    present = [i for i in ITEMS if i.source.is_file()]
    missing = [i for i in ITEMS if not i.source.is_file()]

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    copied: list[tuple[str, str]] = []
    for item in present:
        folder = SECTION_TITLES[item.section][0]
        destination = args.output_dir / folder / item.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, destination)
        copied.append((str(destination.relative_to(args.output_dir)),
                       sha256_file(destination)))

    (args.output_dir / "README.md").write_text(readme_text(present, missing),
                                               encoding="utf-8")
    lines = [f"{digest}  {name}" for name, digest in sorted(copied)]
    readme_digest = sha256_file(args.output_dir / "README.md")
    lines.append(f"{readme_digest}  README.md")
    (args.output_dir / "SHA256SUMS.txt").write_text("\n".join(sorted(lines)) + "\n",
                                                    encoding="utf-8")

    summary = {
        "analysis": "R8 supplementary data package (S7-S11)",
        "analysis_status": "COMPLETE" if not missing else "PARTIAL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sections": sorted({i.section for i in present}, key=lambda s: int(s[1:])),
        "files_copied": len(copied),
        "missing_inputs": [
            {"section": i.section, "source": str(i.source)} for i in missing
        ],
        "note": (
            "Separate from the published S1-S6 package; script 90 is not modified. "
            "Merge the two directories at submission time."
        ),
    }
    (args.output_dir / "package_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 74)
    print(f"R8 supplement package -> {args.output_dir}")
    print("=" * 74)
    for section in sorted({i.section for i in present}, key=lambda s: int(s[1:])):
        folder, title = SECTION_TITLES[section]
        n = sum(1 for i in present if i.section == section)
        print(f"  {section:<4} {n} file(s)  {title}")
    if missing:
        print("\n  MISSING (section absent or partial):")
        for item in missing:
            print(f"    {item.section}  {item.source}")
    print(f"\n  {len(copied)} files, checksums in SHA256SUMS.txt")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/supplementary_package_r8"))
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
