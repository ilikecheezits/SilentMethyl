#!/usr/bin/env python3
"""Discover, audit, and score published non-synonymous breast-cancer SNVs.

This script builds a broad candidate panel from two reproducible sources:

1. a small curated list of experimentally characterized breast-cancer variants;
2. live ClinVar searches for protein-altering SNVs in breast-cancer genes.

Every candidate is normalized to a GRCh38 chromosome allele, compared against
the project's existing TCGA candidate and eGTEx benchmark variants, screened
for an HM450 CpG within SilentMethyl's trained 1,000-bp window, and then passed
to scripts/14_known_variant_application.py.  It does not silently discard
failures: the output audit records each exclusion reason.

The model chooses neither the literature panel nor the target CpG.  The latter
is determined by the existing application script's model-visibility and probe
QC rules.  Model scores are used only after these prespecified filters.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DBSNP_BASE = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp"

GRCH38_ACCESSION_TO_CHROM = {
    "NC_000001.11": "chr1", "NC_000002.12": "chr2",
    "NC_000003.12": "chr3", "NC_000004.12": "chr4",
    "NC_000005.10": "chr5", "NC_000006.12": "chr6",
    "NC_000007.14": "chr7", "NC_000008.11": "chr8",
    "NC_000009.12": "chr9", "NC_000010.11": "chr10",
    "NC_000011.10": "chr11", "NC_000012.12": "chr12",
    "NC_000013.11": "chr13", "NC_000014.9": "chr14",
    "NC_000015.10": "chr15", "NC_000016.10": "chr16",
    "NC_000017.11": "chr17", "NC_000018.10": "chr18",
    "NC_000019.10": "chr19", "NC_000020.11": "chr20",
    "NC_000021.9": "chr21", "NC_000022.11": "chr22",
    "NC_000023.11": "chrX", "NC_000024.10": "chrY",
}

PROTEIN_ALTERING_TERMS = {
    "missense variant", "missense_variant", "nonsense", "stop gained",
    "stop_gained", "stop lost", "stop_lost", "start lost", "start_lost",
    "protein altering variant", "protein_altering_variant",
}

DEFAULT_GENES = [
    "AKT1", "ATM", "BARD1", "BRCA1", "BRCA2", "CASP8", "CDH1",
    "CDKN2A", "CHEK2", "ERBB2", "ESR1", "FGFR1", "FOXA1", "GATA3",
    "MAP2K4", "MAP3K1", "MYC", "NF1", "NOTCH1", "PALB2", "PIK3CA",
    "PTEN", "RAD51C", "RAD51D", "RB1", "SF3B1", "STK11", "TBX3",
    "TP53", "TSC1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-csv", type=Path,
        default=Path("scripts/literature_breast_variant_seeds.csv"),
    )
    parser.add_argument("--genes", nargs="+", default=DEFAULT_GENES)
    parser.add_argument("--clinvar-retmax-per-query", type=int, default=250)
    parser.add_argument(
        "--clinvar-search-mode", choices=("pathogenic", "breast", "both"),
        default="both",
        help="Search pathogenic/likely-pathogenic records, breast records, or both.",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL", ""))
    parser.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY", ""))
    parser.add_argument("--request-delay", type=float, default=0.36)
    parser.add_argument("--split-template", default="data/datafiles/{split}.csv")
    parser.add_argument(
        "--existing-candidate-csv", type=Path,
        default=Path("data/datafiles/testing_data.csv"),
    )
    parser.add_argument(
        "--existing-mqtl-csv", type=Path,
        default=Path("data/egtex_breast_mqtl_heldout_qc.csv"),
    )
    parser.add_argument(
        "--raw-breast-mqtl", type=Path,
        default=Path("data/BreastMammaryTissue.regular.perm.fdr.txt"),
    )
    parser.add_argument(
        "--allow-existing-mqtl-variants", action="store_true",
        help="By default any variant already in the frozen mQTL benchmark is excluded.",
    )
    parser.add_argument(
        "--known-variant-script", type=Path,
        default=Path("scripts/14_known_variant_application.py"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/journal/literature_variant_screen"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/literature_variant_screen"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize_chr(value: Any) -> str:
    text = str(value).strip()
    if not text.startswith("chr"):
        text = "chr" + text
    return text


def locus_key(chrom: Any, pos1: Any, ref: Any, alt: Any) -> str:
    alleles = sorted([str(ref).upper(), str(alt).upper()])
    return f"{normalize_chr(chrom)}:{int(pos1)}:{alleles[0]}:{alleles[1]}"


class CachedHTTP:
    def __init__(self, cache_dir: Path, offline: bool, refresh: bool, delay: float):
        self.cache_dir = cache_dir
        self.offline = offline
        self.refresh = refresh
        self.delay = delay
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str, suffix: str) -> Path:
        key = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / f"{key}.{suffix}"

    def bytes(self, url: str, suffix: str = "bin") -> bytes:
        path = self._path(url, suffix)
        if path.is_file() and not self.refresh:
            return path.read_bytes()
        if self.offline:
            raise FileNotFoundError(f"Offline cache miss for {url}")
        request = Request(url, headers={"User-Agent": "SilentMethyl-literature-screen/1.0"})
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urlopen(request, timeout=90) as response:
                    payload = response.read()
                path.write_bytes(payload)
                time.sleep(self.delay)
                return payload
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to retrieve {url}: {last_error}")

    def json(self, url: str) -> Any:
        return json.loads(self.bytes(url, "json").decode("utf-8"))


def ncbi_params(args: argparse.Namespace, **kwargs: Any) -> str:
    values = {key: value for key, value in kwargs.items() if value not in (None, "")}
    values["tool"] = "SilentMethyl_literature_variant_screen"
    if args.email:
        values["email"] = args.email
    if args.api_key:
        values["api_key"] = args.api_key
    return urlencode(values, doseq=True)


def clinvar_search(http: CachedHTTP, args: argparse.Namespace, term: str) -> list[str]:
    url = f"{NCBI_BASE}/esearch.fcgi?" + ncbi_params(
        args, db="clinvar", term=term, retmode="json",
        retmax=args.clinvar_retmax_per_query,
    )
    payload = http.json(url)
    return [str(value) for value in payload.get("esearchresult", {}).get("idlist", [])]


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def clinvar_summaries(
    http: CachedHTTP, args: argparse.Namespace, ids: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch in chunks(list(dict.fromkeys(ids)), 100):
        url = f"{NCBI_BASE}/esummary.fcgi?" + ncbi_params(
            args, db="clinvar", id=",".join(batch), retmode="json",
        )
        result = http.json(url).get("result", {})
        for clinvar_id in batch:
            if clinvar_id in result and isinstance(result[clinvar_id], dict):
                records.append(result[clinvar_id])
    return records


def extract_genes(summary: dict[str, Any]) -> list[str]:
    genes: list[str] = []
    for gene in summary.get("genes", []) or []:
        symbol = gene.get("symbol") or gene.get("gene_symbol")
        if symbol:
            genes.append(str(symbol))
    for variant in summary.get("variation_set", []) or []:
        for gene in variant.get("variation_loc", []) or []:
            symbol = gene.get("gene_symbol") or gene.get("symbol")
            if symbol:
                genes.append(str(symbol))
    title = str(summary.get("title", ""))
    match = re.search(r"\(([A-Za-z0-9-]+)\)", title)
    if match:
        genes.append(match.group(1))
    return list(dict.fromkeys(genes))


def parse_spdi(spdi: str) -> tuple[str, int, str, str] | None:
    fields = str(spdi).split(":")
    if len(fields) != 4:
        return None
    accession, position0, ref, alt = fields
    chrom = GRCH38_ACCESSION_TO_CHROM.get(accession)
    if chrom is None or not str(position0).isdigit():
        return None
    ref, alt = ref.upper(), alt.upper()
    if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT" or ref == alt:
        return None
    return chrom, int(position0) + 1, ref, alt


def is_protein_altering(summary: dict[str, Any]) -> bool:
    text_parts = [
        str(summary.get("title", "")),
        str(summary.get("protein_change", "")),
        *[str(x) for x in summary.get("molecular_consequence_list", []) or []],
    ]
    for variant in summary.get("variation_set", []) or []:
        text_parts.extend(str(x) for x in variant.get("molecular_consequence_list", []) or [])
    text = " ".join(text_parts).lower()
    return any(term in text for term in PROTEIN_ALTERING_TERMS)


def clinvar_to_candidates(
    summaries: list[dict[str, Any]], query_gene: str, search_mode: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        if not is_protein_altering(summary):
            continue
        clinvar_id = str(summary.get("uid", "")).strip()
        title = str(summary.get("title", "")).strip()
        genes = extract_genes(summary)
        gene = query_gene if query_gene in genes or not genes else genes[0]
        classification = summary.get("germline_classification", {}) or {}
        significance = str(classification.get("description", ""))
                                                                             
                                                                               
                                                          
        if search_mode == "pathogenic" and "pathogenic" not in significance.lower():
            continue
        if search_mode == "breast" and any(
            label in significance.lower()
            for label in ("benign", "uncertain", "conflicting")
        ):
            continue
        for variant in summary.get("variation_set", []) or []:
            parsed = parse_spdi(str(variant.get("canonical_spdi", "")))
            if parsed is None:
                continue
            chrom, pos1, ref, alt = parsed
            xrefs = variant.get("variation_xrefs", []) or []
            rsids = [
                str(x.get("db_id") or x.get("id"))
                for x in xrefs
                if str(x.get("db_source", "")).lower() == "dbsnp"
                and (x.get("db_id") or x.get("id"))
            ]
            rsid = f"rs{rsids[0]}" if rsids and not rsids[0].startswith("rs") else (rsids[0] if rsids else "")
            label = rsid or f"ClinVar{clinvar_id}"
            rows.append({
                "Variant_ID": f"{label}:{ref}>{alt}",
                "Rsid": rsid,
                "Gene": gene,
                "chr": chrom,
                "Position_1based": pos1,
                "Ref": ref,
                "Alt": alt,
                "Transcript_Annotation": title,
                "Variant_Origin": "ClinVar record; germline/somatic status must be checked for the selected case",
                "Clinical_Significance": significance,
                "Discovery_Source": f"ClinVar-{search_mode}",
                "Evidence_Grade": (
                    "B_ClinVar_pathogenic_breast_record"
                    if "pathogenic" in significance.lower()
                    else "C_published_breast_record"
                ),
                "Citation_Key": "",
                "Source_URL": "",
                "ClinVar_URL": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{clinvar_id}/",
                "ClinVar_ID": clinvar_id,
                "Gene_Function": "",
                "Disease_Context": "breast cancer or hereditary breast-cancer gene panel",
                "Reported_Biological_Evidence": title,
                "Expected_mQTL_CpG": "",
                "mQTL_Tissue_Qualifier": "",
            })
    return rows


def resolve_rsid(http: CachedHTTP, rsid: str) -> list[tuple[str, int, str, str]]:
    number = re.sub(r"^rs", "", str(rsid).strip(), flags=re.IGNORECASE)
    if not number.isdigit():
        return []
    payload = http.json(f"{DBSNP_BASE}/{number}")
    placements = payload.get("primary_snapshot_data", {}).get("placements_with_allele", [])
    resolved: list[tuple[str, int, str, str]] = []
    for placement in placements:
        seq_id = str(placement.get("seq_id", ""))
        chrom = GRCH38_ACCESSION_TO_CHROM.get(seq_id)
        if chrom is None:
            continue
        annot = placement.get("placement_annot", {}) or {}
        if not (placement.get("is_ptlp") or annot.get("seq_type") == "refseq_chromosome"):
            continue
        alleles = placement.get("alleles", []) or []
        spdis = [item.get("allele", {}).get("spdi", {}) for item in alleles]
        ref_candidates = [
            str(spdi.get("deleted_sequence", "")).upper()
            for spdi in spdis
            if spdi.get("deleted_sequence") == spdi.get("inserted_sequence")
        ]
        if not ref_candidates:
            continue
        ref = ref_candidates[0]
        if len(ref) != 1 or ref not in "ACGT":
            continue
        for spdi in spdis:
            deleted = str(spdi.get("deleted_sequence", "")).upper()
            inserted = str(spdi.get("inserted_sequence", "")).upper()
            if deleted != ref or len(inserted) != 1 or inserted not in "ACGT" or inserted == ref:
                continue
            resolved.append((chrom, int(spdi["position"]) + 1, ref, inserted))
    return list(dict.fromkeys(resolved))


def load_seed_candidates(path: Path, http: CachedHTTP) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    seeds = pd.read_csv(path, dtype=str).fillna("")
    if "Rsid" not in seeds.columns:
        raise ValueError(f"{path} must contain an Rsid column")
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for seed in seeds.to_dict("records"):
        resolved = resolve_rsid(http, seed["Rsid"])
        if not resolved:
            audit.append({"Input": seed["Rsid"], "Status": "rsid_not_resolved_to_GRCh38_SNV"})
            continue
        requested_alt = seed.get("Expected_Alt", "").upper()
        if not requested_alt and len(resolved) > 1:
            audit.append({
                "Input": seed["Rsid"],
                "Status": "ambiguous_multiallelic_rsid_requires_Expected_Alt",
                "Resolved_Alleles": "|".join(
                    f"{chrom}:{pos1}:{ref}>{alt}" for chrom, pos1, ref, alt in resolved
                ),
            })
            continue
        for chrom, pos1, ref, alt in resolved:
            if requested_alt and alt != requested_alt:
                continue
            row = dict(seed)
            row.update({
                "Variant_ID": f"{seed['Rsid']}:{ref}>{alt}",
                "chr": chrom,
                "Position_1based": pos1,
                "Ref": ref,
                "Alt": alt,
                "Discovery_Source": "curated-primary-literature",
                "Evidence_Grade": seed.get(
                    "Evidence_Grade", "A_curated_primary_breast_evidence"
                ),
                "ClinVar_ID": seed.get("ClinVar_ID", ""),
                "ClinVar_URL": seed.get("ClinVar_URL", ""),
                "Clinical_Significance": seed.get("Clinical_Significance", ""),
                "Variant_Origin": seed.get("Variant_Origin", "somatic or germline; see cited study"),
                "Expected_mQTL_CpG": seed.get("Expected_mQTL_CpG", ""),
                "mQTL_Tissue_Qualifier": seed.get("mQTL_Tissue_Qualifier", ""),
            })
            rows.append(row)
            audit.append({"Input": seed["Rsid"], "Status": "resolved", "Resolved": row["Variant_ID"]})
    return rows, audit


def discover_clinvar(http: CachedHTTP, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for gene in list(dict.fromkeys(args.genes)):
        terms: list[tuple[str, str]] = []
        if args.clinvar_search_mode in {"pathogenic", "both"}:
            terms.append((
                "pathogenic",
                f'{gene}[gene] AND '
                '(pathogenic[Clinical Significance] OR "likely pathogenic"[Clinical Significance]) AND '
                '("breast cancer"[phenotype] OR "breast-ovarian cancer"[phenotype] OR '
                '"hereditary cancer-predisposing syndrome"[phenotype]) AND '
                '"single nucleotide variant"[Type of variation]',
            ))
        if args.clinvar_search_mode in {"breast", "both"}:
            terms.append((
                "breast",
                f'{gene}[gene] AND ("breast cancer"[phenotype] OR "breast-ovarian cancer"[phenotype]) AND "single nucleotide variant"[Type of variation]',
            ))
        for mode, term in terms:
            ids = clinvar_search(http, args, term)
            summaries = clinvar_summaries(http, args, ids)
            rows = clinvar_to_candidates(summaries, gene, mode)
            candidates.extend(rows)
            audit.append({
                "Input": f"ClinVar:{gene}:{mode}", "Status": "queried",
                "Record_Count": len(ids), "Protein_Altering_GRCh38_SNV_Count": len(rows),
            })
            if args.smoke_test:
                break
        if args.smoke_test and len(audit) >= 2:
            break
    return candidates, audit


def collapse_candidates(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).fillna("")
    frame["Locus_Key"] = [
        locus_key(c, p, r, a)
        for c, p, r, a in zip(frame["chr"], frame["Position_1based"], frame["Ref"], frame["Alt"])
    ]
    text_columns = [
        "Rsid", "Gene", "Transcript_Annotation", "Variant_Origin",
        "Clinical_Significance", "Discovery_Source", "Citation_Key",
        "Source_URL", "ClinVar_URL", "ClinVar_ID", "Gene_Function",
        "Disease_Context", "Reported_Biological_Evidence", "Expected_mQTL_CpG",
        "mQTL_Tissue_Qualifier", "Protein_Change",
        "Evidence_Grade",
    ]
    for column in text_columns:
        if column not in frame:
            frame[column] = ""

    collapsed: list[dict[str, Any]] = []
    for _, group in frame.groupby("Locus_Key", sort=False):
        first = group.iloc[0].to_dict()
        for column in text_columns:
            values = [str(v).strip() for v in group[column] if str(v).strip()]
            first[column] = " | ".join(dict.fromkeys(values))
        first["Variant_ID"] = (
            (first.get("Rsid") or f"chr{str(first['chr']).replace('chr', '')}_{int(first['Position_1based'])}")
            + f":{first['Ref']}>{first['Alt']}"
        )
        collapsed.append(first)
    result = pd.DataFrame(collapsed)
    result.sort_values(["chr", "Position_1based", "Ref", "Alt"], inplace=True, kind="mergesort")
    return result.reset_index(drop=True)


def load_existing_variant_keys(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required current-data exclusion file is missing: {path}. "
            "Pass the correct path; do not bypass this check for manuscript screening."
        )
    frame = pd.read_csv(path)
    choices = [
        ("chr", "Variant_Position_1based", "Reference_Allele", "Alternate_Allele"),
        ("chr", "Position_1based", "Ref", "Alt"),
    ]
    for chrom, pos, ref, alt in choices:
        if {chrom, pos, ref, alt}.issubset(frame.columns):
            return {
                locus_key(c, p, r, a)
                for c, p, r, a in zip(frame[chrom], frame[pos], frame[ref], frame[alt])
            }
    raise ValueError(f"Could not identify variant columns in {path}")


def parse_mqtl_variant_id(value: Any) -> tuple[str, int, str, str] | None:
    match = re.fullmatch(r"(chr[^_]+)_(\d+)_([ACGT])_([ACGT])_b38", str(value))
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3), match.group(4)


def load_existing_mqtl_keys(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required current mQTL exclusion file is missing: {path}. "
            "Pass the frozen benchmark path or explicitly use --allow-existing-mqtl-variants."
        )
    frame = pd.read_csv(path)
    variant_keys: set[str] = set()
    pair_keys: set[str] = set()
    if "variant_id" in frame.columns:
        for row in frame.itertuples(index=False):
            parsed = parse_mqtl_variant_id(row.variant_id)
            if parsed is None:
                continue
            key = locus_key(*parsed)
            variant_keys.add(key)
            cpg = getattr(row, "cpg_id", getattr(row, "probeID", ""))
            if cpg:
                pair_keys.add(f"{key}|{cpg}")
    else:
        variant_keys = load_existing_variant_keys(path)
    return variant_keys, pair_keys


def load_probe_index(split_template: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = Path(split_template.format(split=split))
        if not path.is_file():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0)
        required = {"probeID", "chr", "pos"}
        missing = required - set(header.columns)
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        frame = pd.read_csv(path, usecols=["probeID", "chr", "pos"])
        frame["Model_Split"] = split
        frames.append(frame)
    probes = pd.concat(frames, ignore_index=True)
    probes["chr"] = probes["chr"].map(normalize_chr)
    probes["pos"] = pd.to_numeric(probes["pos"], errors="raise").astype(np.int64)
    return probes


def audit_and_prescreen(
    candidates: pd.DataFrame,
    current_tcga: set[str],
    current_mqtl: set[str],
    probes: pd.DataFrame,
    allow_existing_mqtl: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    probes_by_chr = {chrom: group for chrom, group in probes.groupby("chr")}
    for row in candidates.to_dict("records"):
        key = row["Locus_Key"]
        status = "eligible_for_allele_and_probe_QC"
        nearby = pd.DataFrame()
        if key in current_tcga:
            status = "excluded_variant_already_in_TCGA_candidate_data"
        elif key in current_mqtl and not allow_existing_mqtl:
            status = "excluded_variant_already_in_frozen_mQTL_benchmark"
        else:
            chrom_probes = probes_by_chr.get(row["chr"], pd.DataFrame())
            if not chrom_probes.empty:
                variant_pos0 = int(row["Position_1based"]) - 1
                offsets = variant_pos0 - chrom_probes["pos"]
                nearby = chrom_probes[offsets.between(-499, 500)]
            if nearby.empty:
                status = "excluded_no_HM450_CpG_within_model_window"
        audit_rows.append({
            "Variant_ID": row["Variant_ID"], "Locus_Key": key,
            "Gene": row.get("Gene", ""), "chr": row["chr"],
            "Position_1based": int(row["Position_1based"]),
            "Ref": row["Ref"], "Alt": row["Alt"], "Status": status,
            "Nearby_CpG_Count_Pre_QC": int(len(nearby)),
            "Nearby_Splits_Pre_QC": "|".join(sorted(set(nearby.get("Model_Split", [])))),
            "Discovery_Source": row.get("Discovery_Source", ""),
        })
        if status == "eligible_for_allele_and_probe_QC":
            eligible_rows.append(row)
    return pd.DataFrame(eligible_rows), pd.DataFrame(audit_rows)


def pubmed_links_for_clinvar(
    http: CachedHTTP, args: argparse.Namespace, clinvar_id: str
) -> list[str]:
    if not clinvar_id:
        return []
    url = f"{NCBI_BASE}/elink.fcgi?" + ncbi_params(
        args, dbfrom="clinvar", db="pubmed", id=clinvar_id,
    )
    root = ET.fromstring(http.bytes(url, "xml"))
    return list(dict.fromkeys(node.text for node in root.findall(".//LinkSetDb/Link/Id") if node.text))


def attach_publications(
    frame: pd.DataFrame, http: CachedHTTP, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    if frame.empty:
        if "PMID" not in frame:
            frame["PMID"] = pd.Series(dtype=str)
        if "Source_URL" not in frame:
            frame["Source_URL"] = pd.Series(dtype=str)
        return frame, pd.DataFrame(columns=[
            "Variant_ID", "Published_Source_Count", "PMIDs", "Status",
        ])
    audits: list[dict[str, Any]] = []
    pmid_values: list[str] = []
    source_values: list[str] = []
    for row in frame.itertuples(index=False):
        existing_pmids = [
            value for value in re.split(r"[|,; ]+", str(getattr(row, "PMID", "")))
            if value.isdigit()
        ]
        clinvar_ids = [
            value for value in re.split(r"[|,; ]+", str(getattr(row, "ClinVar_ID", "")))
            if value.isdigit()
        ]
        linked: list[str] = []
        for clinvar_id in clinvar_ids:
            linked.extend(pubmed_links_for_clinvar(http, args, clinvar_id))
        pmids = list(dict.fromkeys(existing_pmids + linked))
        urls = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]
        existing_urls = [
            value.strip() for value in str(getattr(row, "Source_URL", "")).split("|")
            if value.strip()
        ]
        pmid_values.append("|".join(pmids))
        source_values.append(" | ".join(dict.fromkeys(existing_urls + urls)))
        audits.append({
            "Variant_ID": row.Variant_ID,
            "Published_Source_Count": len(pmids) + len(existing_urls),
            "PMIDs": "|".join(pmids),
            "Status": "published_source_found" if pmids or existing_urls else "excluded_no_linked_publication",
        })
    frame["PMID"] = pmid_values
    frame["Source_URL"] = source_values
    pub_audit = pd.DataFrame(audits)
    keep = pub_audit["Status"].eq("published_source_found").to_numpy()
    return frame.loc[keep].reset_index(drop=True), pub_audit


def load_external_mqtl_hits(path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Locus_Key", "probeID", "eGTEx_variant_id", "eGTEx_qval",
        "eGTEx_slope", "eGTEx_maf", "External_Tissue",
    ]
    if not path.is_file() or candidates.empty:
        return pd.DataFrame(columns=columns)
    wanted = set(candidates["Locus_Key"])
    hits: list[pd.DataFrame] = []
    usecols = ["cpg_id", "variant_id", "maf", "slope", "qval"]
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=300_000):
        parsed = chunk["variant_id"].astype(str).str.extract(
            r"^(chr[^_]+)_(\d+)_([ACGT])_([ACGT])_b38$"
        )
        valid = parsed.notna().all(axis=1)
        if not valid.any():
            continue
        parsed = parsed.loc[valid]
        keys = [
            locus_key(c, p, r, a)
            for c, p, r, a in zip(parsed[0], parsed[1], parsed[2], parsed[3])
        ]
        sub = chunk.loc[valid].copy()
        sub["Locus_Key"] = keys
        sub = sub[sub["Locus_Key"].isin(wanted)]
        if not sub.empty:
            hits.append(sub)
    if not hits:
        return pd.DataFrame(columns=columns)
    result = pd.concat(hits, ignore_index=True).rename(columns={
        "cpg_id": "probeID", "variant_id": "eGTEx_variant_id",
        "qval": "eGTEx_qval", "slope": "eGTEx_slope", "maf": "eGTEx_maf",
    })
    result["eGTEx_qval"] = pd.to_numeric(result["eGTEx_qval"], errors="coerce")
    result = result[result["eGTEx_qval"] < 0.05].copy()
    result["External_Tissue"] = "eGTEx Breast Mammary Tissue"
    return result[columns]


def write_scorer_input(frame: pd.DataFrame, path: Path) -> None:
    required = [
        "Variant_ID", "Gene", "chr", "Position_1based", "Ref", "Alt",
        "Citation_Key", "Source_URL", "ClinVar_URL", "Transcript_Annotation",
        "Gene_Function", "Disease_Context", "Reported_Biological_Evidence",
    ]
    output = frame.copy()
    for column in required:
        if column not in output:
            output[column] = ""
    atomic_csv(output[required], path)


def run_scorer(args: argparse.Namespace, variant_csv: Path, output_dir: Path) -> None:
    command = [
        sys.executable, "-u", str(args.known_variant_script),
        "--variant-csv", str(variant_csv),
        "--split-template", args.split_template,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--output-dir", str(output_dir),
    ]
    if args.amp:
        command.append("--amp")
    subprocess.run(command, check=True)


def rank_predictions(
    scorer_output: Path,
    candidate_metadata: pd.DataFrame,
    mqtl_hits: pd.DataFrame,
    existing_pair_keys: set[str],
    output_dir: Path,
) -> pd.DataFrame:
    path = scorer_output / "known_variant_predictions_ensemble.csv"
    if not path.is_file():
        return pd.DataFrame()
    predictions = pd.read_csv(path)
    if "Variant_ID" not in predictions and "Published_Variant_ID" in predictions:
        predictions = predictions.rename(columns={"Published_Variant_ID": "Variant_ID"})
    metadata = candidate_metadata.drop_duplicates("Variant_ID")
    predictions = predictions.merge(metadata, on="Variant_ID", how="left", suffixes=("", "_screen"))
    delta_col = next(
        (column for column in ["Predicted_Delta_Beta_Mean", "Predicted_Delta_Beta"] if column in predictions),
        None,
    )
    if delta_col is None:
        raise ValueError(f"Cannot identify delta-beta column in {path}")
    predictions["Absolute_Delta_Beta"] = pd.to_numeric(predictions[delta_col], errors="raise").abs()
    if not mqtl_hits.empty:
        predictions = predictions.merge(mqtl_hits, on=["Locus_Key", "probeID"], how="left")
    else:
        predictions["eGTEx_qval"] = np.nan
        predictions["eGTEx_slope"] = np.nan
        predictions["External_Tissue"] = ""
    predictions["Exact_External_Breast_mQTL_Target"] = predictions["eGTEx_qval"].notna()
    predictions["External_Pair_Previously_In_Benchmark"] = [
        f"{key}|{probe}" in existing_pair_keys
        for key, probe in zip(predictions["Locus_Key"], predictions["probeID"])
    ]
    predictions["Heldout_Test_Target"] = predictions["Model_Split"].astype(str).eq("test")
    predictions["Nearest_Model_Visible_Target"] = predictions[
        "Is_Primary_Nearest_Target"
    ].astype(bool)
    has_external_target = predictions.groupby("Variant_ID")[
        "Exact_External_Breast_mQTL_Target"
    ].transform("any")
    predictions["Prespecified_Application_Target"] = np.where(
        has_external_target,
        predictions["Exact_External_Breast_mQTL_Target"],
        predictions["Nearest_Model_Visible_Target"],
    )
    predictions["Application_Target_Rule"] = np.where(
        has_external_target,
        "exact significant eGTEx breast-mQTL CpG",
        "nearest eligible model-visible HM450 CpG",
    )
    predictions["Selection_Tier"] = np.select(
        [
            predictions["Heldout_Test_Target"] & predictions["Exact_External_Breast_mQTL_Target"],
            predictions["Heldout_Test_Target"],
            predictions["Model_Split"].astype(str).eq("val"),
        ],
        [
            "1_heldout_test_plus_external_breast_mQTL",
            "2_heldout_test_exploratory",
            "3_validation_split_exploratory",
        ],
        default="4_training_split_illustration_only",
    )
    predictions.sort_values(
        ["Selection_Tier", "Absolute_Delta_Beta", "Variant_ID", "probeID"],
        ascending=[True, False, True, True], inplace=True, kind="mergesort",
    )
    atomic_csv(predictions, output_dir / "literature_variant_predictions_all_pairs.csv")
    ranked = predictions[predictions["Prespecified_Application_Target"].astype(bool)].copy()
    ranked["Rank_Within_Selection_Tier"] = (
        ranked.groupby("Selection_Tier")["Absolute_Delta_Beta"]
        .rank(method="first", ascending=False).astype(int)
    )
    atomic_csv(ranked, output_dir / "literature_variant_predictions_ranked.csv")
    top = ranked.groupby("Selection_Tier", sort=False).head(1)
    atomic_csv(top, output_dir / "top_case_per_evidence_tier.csv")
    return ranked


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    http = CachedHTTP(args.cache_dir, args.offline, args.refresh_cache, args.request_delay)

    seed_rows, seed_audit = load_seed_candidates(args.seed_csv, http)
    clinvar_rows, clinvar_audit = discover_clinvar(http, args)
    all_candidates = collapse_candidates(seed_rows + clinvar_rows)
    if all_candidates.empty:
        raise RuntimeError("No GRCh38 protein-altering SNV candidates were assembled")

    current_tcga = load_existing_variant_keys(args.existing_candidate_csv)
    if args.allow_existing_mqtl_variants:
        current_mqtl, current_pairs = set(), set()
    else:
        current_mqtl, current_pairs = load_existing_mqtl_keys(args.existing_mqtl_csv)
    probes = load_probe_index(args.split_template)
    eligible, screening_audit = audit_and_prescreen(
        all_candidates, current_tcga, current_mqtl, probes,
        args.allow_existing_mqtl_variants,
    )
    eligible_with_papers, publication_audit = attach_publications(eligible, http, args)

    atomic_csv(all_candidates, args.output_dir / "all_resolved_literature_candidates.csv")
    atomic_csv(screening_audit, args.output_dir / "candidate_exclusion_audit.csv")
    atomic_csv(publication_audit, args.output_dir / "publication_link_audit.csv")
    atomic_csv(pd.DataFrame(seed_audit + clinvar_audit), args.output_dir / "source_query_audit.csv")
    atomic_csv(eligible_with_papers, args.output_dir / "eligible_published_candidates.csv")

    mqtl_hits = load_external_mqtl_hits(args.raw_breast_mqtl, eligible_with_papers)
    atomic_csv(mqtl_hits, args.output_dir / "candidate_breast_mqtl_hits.csv")
    scorer_input = args.output_dir / "silentmethyl_variant_input.csv"
    write_scorer_input(eligible_with_papers, scorer_input)

    hashes = {
        str(path): sha256(path)
        for path in [args.seed_csv, args.existing_candidate_csv, args.existing_mqtl_csv]
        if path.is_file()
    }
    summary: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "resolved_unique_GRCh38_SNVs": int(len(all_candidates)),
        "excluded_as_existing_TCGA_variant": int(
            screening_audit["Status"].eq("excluded_variant_already_in_TCGA_candidate_data").sum()
        ),
        "excluded_as_existing_mQTL_variant": int(
            screening_audit["Status"].eq("excluded_variant_already_in_frozen_mQTL_benchmark").sum()
        ),
        "eligible_before_publication_check": int(len(eligible)),
        "eligible_published_candidates_for_scoring": int(len(eligible_with_papers)),
        "exact_external_breast_mQTL_rows_found": int(len(mqtl_hits)),
        "input_sha256": hashes,
        "selection_policy": [
            "Prefer held-out chr8/9 target CpGs with an exact independent breast-mQTL pair.",
            "Otherwise use the largest held-out chr8/9 response and label it exploratory.",
            "Validation and training split results are illustrations, not independent validation.",
            "A selected maximum is discovery-biased and requires experimental follow-up.",
        ],
    }

    if args.prepare_only:
        summary["analysis_status"] = "PREPARED_NOT_SCORED"
        atomic_json(summary, args.output_dir / "run_summary.json")
        print(f"Prepared {len(eligible_with_papers)} published candidates: {scorer_input}")
        return
    if eligible_with_papers.empty:
        summary["analysis_status"] = "NO_ELIGIBLE_PUBLISHED_MODEL_VISIBLE_VARIANTS"
        atomic_json(summary, args.output_dir / "run_summary.json")
        print("No eligible published variants survived the prespecified filters.")
        return

    scorer_output = args.output_dir / "silentmethyl_scoring"
    run_scorer(args, scorer_input, scorer_output)
    ranked = rank_predictions(
        scorer_output, eligible_with_papers, mqtl_hits, current_pairs, args.output_dir,
    )
    summary["analysis_status"] = "COMPLETE" if not ranked.empty else "NO_MODEL_VISIBLE_UNMASKED_CPG"
    summary["scored_variant_cpg_pairs"] = int(len(ranked))
    summary["selection_tier_counts"] = (
        ranked["Selection_Tier"].value_counts().to_dict() if not ranked.empty else {}
    )
    atomic_json(summary, args.output_dir / "run_summary.json")
    print("SilentMethyl literature-variant screen")
    print(f"  resolved GRCh38 SNVs: {len(all_candidates)}")
    print(f"  candidates sent to scoring: {len(eligible_with_papers)}")
    print(f"  scored variant-CpG pairs: {len(ranked)}")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
