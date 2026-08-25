#!/usr/bin/env python3
"""Build the somatic synonymous-variant application cohort."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyBigWig
import requests
from pyfaidx import Fasta
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR
GDC_URL = "https://api.gdc.cancer.gov/ssms"
WINDOW_SIZE = 5000
CENTER_C_INDEX = 2499
CENTER_G_INDEX = 2500
FASTA_SLICE = slice(2450, 2550)
VALID_CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])

MUTATION_RE = re.compile(r"^(chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)):g\.(\d+)([ACGT])>([ACGT])$", re.IGNORECASE)
RC_TABLE = str.maketrans(
    "ACGTRYMKBDHVNacgtrymkbdhvn",
    "TGCAYRKMVHDBNtgcayrkmvhdbn",
)
COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--refresh-gdc",
        action="store_true",
        help="Ignore an existing raw GDC cache and issue a fresh query.",
    )
    parser.add_argument("--page-size", type=int, default=10_000)
    parser.add_argument("--max-threads", type=int, default=1, help="Reserved for future parallel feature extraction.")
    parser.add_argument(
        "--hash-large-inputs",
        action="store_true",
        help="Also SHA-256 hash the large reference FASTA.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: Path, hash_file: bool = True) -> dict:
    record = {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
    if hash_file:
        record["sha256"] = sha256_file(path)
    return record


def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC_TABLE)[::-1]


def complement_base(base: str) -> str:
    return base.translate(COMPLEMENT).upper()


def natural_chrom_rank(chrom: str) -> int:
    token = str(chrom).replace("chr", "", 1)
    if token.isdigit():
        return int(token)
    return {"X": 23, "Y": 24}.get(token, 10_000)


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    out = df.copy()
    out["_chrom_rank"] = out["chr"].map(natural_chrom_rank)
    out = out.sort_values(
        ["_chrom_rank", "Variant_Position_1based", "probeID", "Selected_Transcript_ID", "Candidate_ID"],
        kind="mergesort",
    )
    return out.drop(columns="_chrom_rank").reset_index(drop=True)


def parse_gtf_attributes(text: str) -> dict[str, list[str]]:
    attributes: dict[str, list[str]] = defaultdict(list)
    for field in text.strip().strip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        if " " not in field:
            continue
        key, value = field.split(" ", 1)
        attributes[key].append(value.strip().strip('"'))
    return dict(attributes)


def first_attribute(attributes: dict[str, list[str]], key: str, default: str = "") -> str:
    values = attributes.get(key, [])
    return values[0] if values else default


def interval_contains_any(sorted_positions: list[int], start: int, end: int) -> bool:
    index = bisect.bisect_left(sorted_positions, start)
    return index < len(sorted_positions) and sorted_positions[index] <= end


def transcript_priority(annotation: dict) -> tuple:
    tags = {tag.lower() for tag in annotation.get("tags", [])}
    tsl_raw = str(annotation.get("transcript_support_level", "NA")).split(" ", 1)[0]
    try:
        tsl = int(tsl_raw)
    except ValueError:
        tsl = 99

    return (
        0 if "mane_select" in tags else 1,
        0 if "ensembl_canonical" in tags else 1,
        0 if any(tag.startswith("appris_principal") for tag in tags) else 1,
        0 if annotation.get("transcript_type") == "protein_coding" else 1,
        0 if "basic" in tags else 1,
        tsl,
        -int(annotation.get("cds_length", 0)),
        str(annotation.get("transcript_id", "")),
    )


def discover_candidate_transcripts(
    gtf_path: Path,
    positions_by_chrom: dict[str, list[int]],
) -> set[str]:
    selected: set[str] = set()
    with gzip.open(gtf_path, "rt") as handle:
        for line in tqdm(handle, desc="GENCODE pass 1: locating candidate CDS transcripts"):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "CDS":
                continue
            chrom = fields[0]
            positions = positions_by_chrom.get(chrom)
            if not positions:
                continue
            start, end = int(fields[3]), int(fields[4])
            if not interval_contains_any(positions, start, end):
                continue
            attrs = parse_gtf_attributes(fields[8])
            transcript_id = first_attribute(attrs, "transcript_id")
            if transcript_id:
                selected.add(transcript_id)
    return selected


def load_transcript_models(gtf_path: Path, transcript_ids: set[str]) -> dict[str, dict]:
    models: dict[str, dict] = {
        transcript_id: {"transcript_id": transcript_id, "segments": [], "tags": []}
        for transcript_id in transcript_ids
    }

    with gzip.open(gtf_path, "rt") as handle:
        for line in tqdm(handle, desc="GENCODE pass 2: loading fixed transcript models"):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {"transcript", "CDS"}:
                continue
            attrs = parse_gtf_attributes(fields[8])
            transcript_id = first_attribute(attrs, "transcript_id")
            if transcript_id not in models:
                continue

            model = models[transcript_id]
            model.update(
                {
                    "chrom": fields[0],
                    "strand": fields[6],
                    "gene_id": first_attribute(attrs, "gene_id", model.get("gene_id", "")),
                    "gene_name": first_attribute(attrs, "gene_name", model.get("gene_name", "")),
                    "transcript_name": first_attribute(attrs, "transcript_name", model.get("transcript_name", "")),
                    "transcript_type": first_attribute(
                        attrs,
                        "transcript_type",
                        first_attribute(attrs, "transcript_biotype", model.get("transcript_type", "")),
                    ),
                    "transcript_support_level": first_attribute(
                        attrs,
                        "transcript_support_level",
                        model.get("transcript_support_level", "NA"),
                    ),
                }
            )
            if attrs.get("tag"):
                model["tags"] = sorted(set(model.get("tags", [])) | set(attrs["tag"]))
            if fields[2] == "CDS":
                model["segments"].append(
                    {
                        "start": int(fields[3]),
                        "end": int(fields[4]),
                        "phase": fields[7],
                    }
                )

    return {key: value for key, value in models.items() if value.get("segments")}


def prepare_transcript_models(models: dict[str, dict], genome: Fasta) -> dict[str, dict]:
    prepared: dict[str, dict] = {}
    for transcript_id, model in tqdm(models.items(), desc="Constructing transcript CDS sequences"):
        tags_lower = {tag.lower() for tag in model.get("tags", [])}
        if model.get("transcript_type") != "protein_coding" or "cds_start_nf" in tags_lower:
            continue
        strand = model["strand"]
        segments = sorted(model["segments"], key=lambda segment: segment["start"], reverse=(strand == "-"))
        if not segments or str(segments[0].get("phase", "0")) not in {"0", "."}:
            continue
        cds_parts: list[str] = []
        segment_offsets: list[dict] = []
        cumulative = 0

        for segment in segments:
            genomic = str(genome[model["chrom"]][segment["start"] - 1 : segment["end"]]).upper()
            transcript_sequence = reverse_complement(genomic) if strand == "-" else genomic
            segment_offsets.append({**segment, "offset": cumulative})
            cds_parts.append(transcript_sequence)
            cumulative += len(transcript_sequence)

        cds_sequence = "".join(cds_parts)
        if len(cds_sequence) < 3:
            continue

        prepared_model = dict(model)
        prepared_model["segments"] = segment_offsets
        prepared_model["cds_sequence"] = cds_sequence
        prepared_model["cds_length"] = len(cds_sequence)
        prepared[transcript_id] = prepared_model
    return prepared


def annotate_variant_against_transcript(
    chrom: str,
    position: int,
    ref: str,
    alt: str,
    model: dict,
) -> dict | None:
    if model.get("chrom") != chrom:
        return None

    containing_segment = None
    for segment in model["segments"]:
        if segment["start"] <= position <= segment["end"]:
            containing_segment = segment
            break
    if containing_segment is None:
        return None

    if model["strand"] == "+":
        transcript_offset = containing_segment["offset"] + (position - containing_segment["start"])
        transcript_ref = ref
        transcript_alt = alt
    else:
        transcript_offset = containing_segment["offset"] + (containing_segment["end"] - position)
        transcript_ref = complement_base(ref)
        transcript_alt = complement_base(alt)

    cds_sequence = model["cds_sequence"]
    if transcript_offset < 0 or transcript_offset >= len(cds_sequence):
        return None
    if cds_sequence[transcript_offset] != transcript_ref:
        return None

    codon_start = transcript_offset - (transcript_offset % 3)
    codon_ref = cds_sequence[codon_start : codon_start + 3]
    if len(codon_ref) != 3 or any(base not in "ACGT" for base in codon_ref):
        return None

    codon_alt_list = list(codon_ref)
    codon_alt_list[transcript_offset % 3] = transcript_alt
    codon_alt = "".join(codon_alt_list)
    aa_ref = CODON_TABLE.get(codon_ref)
    aa_alt = CODON_TABLE.get(codon_alt)
    if aa_ref is None or aa_alt is None:
        return None

    return {
        "transcript_id": model["transcript_id"],
        "transcript_name": model.get("transcript_name", ""),
        "gene_id": model.get("gene_id", ""),
        "gene_name": model.get("gene_name", ""),
        "transcript_type": model.get("transcript_type", ""),
        "transcript_support_level": model.get("transcript_support_level", "NA"),
        "tags": model.get("tags", []),
        "strand": model["strand"],
        "cds_length": model["cds_length"],
        "cds_position_1based": transcript_offset + 1,
        "codon_ref": codon_ref,
        "codon_alt": codon_alt,
        "amino_acid_ref": aa_ref,
        "amino_acid_alt": aa_alt,
        "is_synonymous": aa_ref == aa_alt,
    }


def fetch_gdc_hits(cache_path: Path, refresh: bool, page_size: int) -> tuple[list[dict], dict]:
    if cache_path.exists() and not refresh:
        with gzip.open(cache_path, "rt") as handle:
            cached = json.load(handle)
        return cached["hits"], cached["query_metadata"]

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {
                "op": "in",
                "content": {
                    "field": "consequence.transcript.consequence_type",
                    "value": ["synonymous_variant"],
                },
            },
        ],
    }

    session = requests.Session()
    all_hits: list[dict] = []
    offset = 0
    total = None

    while total is None or offset < total:
        params = {
            "filters": json.dumps(filters, separators=(",", ":")),
            "expand": "occurrence.case,consequence.transcript.gene",
            "fields": (
                "ssm_id,genomic_dna_change,chromosome,start_position,end_position,"
                "reference_allele,tumor_allele,mutation_type,mutation_subtype,"
                "occurrence.case.case_id,occurrence.case.submitter_id"
            ),
            "format": "JSON",
            "from": str(offset),
            "size": str(page_size),
        }
        response = session.get(GDC_URL, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {})
        hits = data.get("hits", [])
        pagination = data.get("pagination", {})
        if total is None:
            total = int(pagination.get("total", len(hits)))
        all_hits.extend(hits)
        if not hits:
            break
        offset += len(hits)
        print(f"Retrieved {len(all_hits):,}/{total:,} GDC SSM records")

    query_metadata = {
        "endpoint": GDC_URL,
        "filters": filters,
        "expand": "occurrence.case,consequence.transcript.gene",
        "fields": (
            "ssm_id,genomic_dna_change,chromosome,start_position,end_position,"
            "reference_allele,tumor_allele,mutation_type,mutation_subtype,"
            "occurrence.case.case_id,occurrence.case.submitter_id"
        ),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "page_size": page_size,
        "reported_total": total,
        "retrieved_hits": len(all_hits),
    }
    cache_payload = {"query_metadata": query_metadata, "hits": all_hits}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt") as handle:
        json.dump(cache_payload, handle, sort_keys=True)
    return all_hits, query_metadata


def extract_case_identifiers(hit: dict) -> tuple[set[str], set[str]]:
    case_ids: set[str] = set()
    submitter_ids: set[str] = set()
    for occurrence in hit.get("occurrence", []) or []:
        case = occurrence.get("case", {}) or {}
        case_id = case.get("case_id") or case.get("id")
        submitter = case.get("submitter_id")
        if case_id:
            case_ids.add(str(case_id))
        if submitter:
            submitter_ids.add(str(submitter))
    return case_ids, submitter_ids


def parse_gdc_variant(hit: dict) -> dict | None:
    genomic_change = str(hit.get("genomic_dna_change", "")).strip()
    match = MUTATION_RE.match(genomic_change)
    if match:
        chrom, position, ref, alt = match.groups()
        chrom = chrom.replace("CHR", "chr").replace("Chr", "chr")
        return {
            "chrom": chrom,
            "position": int(position),
            "ref": ref.upper(),
            "alt": alt.upper(),
            "genomic_change": f"{chrom}:g.{int(position)}{ref.upper()}>{alt.upper()}",
        }

    chrom = str(hit.get("chromosome", ""))
    if chrom and not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    position = hit.get("start_position")
    ref = str(hit.get("reference_allele", "")).upper()
    alt = str(hit.get("tumor_allele", "")).upper()
    if chrom in VALID_CHROMS and position is not None and len(ref) == len(alt) == 1 and ref in "ACGT" and alt in "ACGT":
        return {
            "chrom": chrom,
            "position": int(position),
            "ref": ref,
            "alt": alt,
            "genomic_change": f"{chrom}:g.{int(position)}{ref}>{alt}",
        }
    return None


def aggregate_gdc_variants(hits: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, int, str, str], dict] = {}
    dropped_unparsed = 0

    for hit in hits:
        parsed = parse_gdc_variant(hit)
        if parsed is None or parsed["ref"] == parsed["alt"]:
            dropped_unparsed += 1
            continue

        key = (parsed["chrom"], parsed["position"], parsed["ref"], parsed["alt"])
        record = grouped.setdefault(
            key,
            {
                **parsed,
                "gdc_ssm_ids": set(),
                "gdc_case_ids": set(),
                "gdc_case_submitter_ids": set(),
            },
        )
        ssm_id = hit.get("ssm_id") or hit.get("id")
        if ssm_id:
            record["gdc_ssm_ids"].add(str(ssm_id))
        case_ids, submitter_ids = extract_case_identifiers(hit)
        record["gdc_case_ids"].update(case_ids)
        record["gdc_case_submitter_ids"].update(submitter_ids)

    records: list[dict] = []
    for record in grouped.values():
        records.append(
            {
                **{key: value for key, value in record.items() if not isinstance(value, set)},
                "gdc_ssm_ids": sorted(record["gdc_ssm_ids"]),
                "gdc_case_ids": sorted(record["gdc_case_ids"]),
                "gdc_case_submitter_ids": sorted(record["gdc_case_submitter_ids"]),
            }
        )

    records.sort(key=lambda row: (natural_chrom_rank(row["chrom"]), row["position"], row["ref"], row["alt"]))
    return records, {"dropped_unparsed_or_non_snv": dropped_unparsed, "unique_genomic_snvs": len(records)}


def build_probe_index(manifest_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        usecols=["probeID", "CpG_chrm", "CpG_beg"],
        dtype={"probeID": "string", "CpG_chrm": "string"},
    ).rename(columns={"CpG_chrm": "chr", "CpG_beg": "pos"})
    manifest = manifest[manifest["chr"].isin(VALID_CHROMS)].dropna(subset=["probeID", "chr", "pos"])
    manifest["pos"] = pd.to_numeric(manifest["pos"], errors="raise").astype(int)
    manifest = manifest.sort_values(["chr", "pos", "probeID"], kind="mergesort")
    manifest = manifest.drop_duplicates(subset="probeID", keep="first")

    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, chrom_df in manifest.groupby("chr", sort=False):
        index[str(chrom)] = (
            chrom_df["pos"].to_numpy(dtype=np.int64),
            chrom_df["probeID"].astype(str).to_numpy(),
        )
    return index


def nearest_probe(probe_index: dict[str, tuple[np.ndarray, np.ndarray]], chrom: str, position_zero_based: int) -> tuple[str, int] | None:
    if chrom not in probe_index:
        return None
    positions, probe_ids = probe_index[chrom]
    insertion = int(np.searchsorted(positions, position_zero_based, side="left"))
    candidates: list[int] = []
    if insertion < len(positions):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    if not candidates:
        return None

    best = min(candidates, key=lambda idx: (abs(int(positions[idx]) - position_zero_based), int(positions[idx]), str(probe_ids[idx])))
    return str(probe_ids[best]), int(positions[best])


def open_bigwig_handles(paths: dict[str, Path]) -> dict[str, pyBigWig.pyBigWig]:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required BigWig files:\n" + "\n".join(missing))
    return {name: pyBigWig.open(str(path)) for name, path in paths.items()}


def get_bw_signal(bw_obj: pyBigWig.pyBigWig, chrom: str, start: int, end: int) -> float:
    try:
        chroms = bw_obj.chroms()
        query_chrom = chrom
        if query_chrom not in chroms:
            alternate = chrom.replace("chr", "", 1) if chrom.startswith("chr") else f"chr{chrom}"
            if alternate not in chroms:
                return np.nan
            query_chrom = alternate

        bounded_start = max(0, int(start))
        bounded_end = min(int(chroms[query_chrom]), int(end))
        if bounded_end <= bounded_start:
            return np.nan
        stat = bw_obj.stats(query_chrom, bounded_start, bounded_end, type="mean")
        if not stat or stat[0] is None:
            return np.nan
        value = float(stat[0])
        return value if math.isfinite(value) else np.nan
    except Exception:
        return np.nan


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def model_split_for_chrom(chrom: str, split_manifest: dict) -> str:
    if chrom in set(split_manifest["test_chromosomes"]):
        return "test"
    if chrom in set(split_manifest["validation_chromosomes"]):
        return "val"
    if chrom in set(split_manifest["train_chromosomes"]):
        return "train"
    raise ValueError(f"Chromosome {chrom} is absent from split_manifest.json")


def sbs96_class(genome: Fasta, chrom: str, position_1based: int, ref: str, alt: str) -> tuple[str, str]:
    position_zero = position_1based - 1
    if position_zero <= 0 or position_zero + 1 >= len(genome[chrom]):
        return "", ""
    trinucleotide = str(genome[chrom][position_zero - 1 : position_zero + 2]).upper()
    if len(trinucleotide) != 3 or trinucleotide[1] != ref:
        return trinucleotide, ""

    oriented_ref, oriented_alt, oriented_tri = ref, alt, trinucleotide
    if ref in {"A", "G"}:
        oriented_ref = complement_base(ref)
        oriented_alt = complement_base(alt)
        oriented_tri = reverse_complement(trinucleotide)
    return trinucleotide, f"{oriented_tri[0]}[{oriented_ref}>{oriented_alt}]{oriented_tri[2]}"


def write_candidate_fastas(df: pd.DataFrame, paths: dict[str, tuple[Path, str]]) -> None:
    handles = {name: path.open("w") for name, (path, _) in paths.items()}
    try:
        for row in df.itertuples(index=False):
            for name, (_, sequence_column) in paths.items():
                sequence = str(getattr(row, sequence_column))
                if len(sequence) != WINDOW_SIZE:
                    raise ValueError(f"Unexpected sequence length for {row.Candidate_ID}: {len(sequence)}")
                handles[name].write(f">{row.Candidate_ID}\n{sequence[FASTA_SLICE]}\n")
    finally:
        for handle in handles.values():
            handle.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    datafiles_dir = data_dir / "datafiles"
    datafiles_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = data_dir / "hg38.fa"
    manifest_path = data_dir / "HM450.hg38.manifest.tsv.gz"
    gtf_path = data_dir / "reference" / "gencode.v44.annotation.gtf.gz"
    split_manifest_path = datafiles_dir / "split_manifest.json"
    imputation_path = datafiles_dir / "feature_imputation.json"
    raw_gdc_path = datafiles_dir / "gdc_tcga_brca_synonymous_raw.json.gz"

    base_ref = data_dir / "reference"
    bw_paths = {
        "Ref_ATAC_Signal": base_ref / "ATAC_seq.bw",
        "Ref_H3K4me3_Signal": base_ref / "H3K4me3.bw",
        "Ref_H3K27ac_Signal": base_ref / "H3K27ac.bw",
        "Ref_H3K27me3_Signal": base_ref / "H3K27me3.bw",
        "Ref_H3K9me3_Signal": base_ref / "H3K9me3.bw",
        "Ref_H3K36me3_Signal": base_ref / "H3K36me3.bw",
        "Ref_H3K4me1_Signal": base_ref / "H3K4me1.bw",
        "Target_Base_PhyloP_100way_1": base_ref / "hg38.phyloP100way.bw",
        "Target_Base_PhyloP_100way_2": base_ref / "hg38.phyloP100way.bw",
    }

    required = [fasta_path, manifest_path, gtf_path, split_manifest_path, imputation_path, *bw_paths.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    split_manifest = load_json(split_manifest_path)
    imputation_payload = load_json(imputation_path)
    imputation_values = imputation_payload["values"]
    absent_imputation = set(bw_paths) - set(imputation_values)
    if absent_imputation:
        raise RuntimeError(f"Missing imputation values for: {sorted(absent_imputation)}")

    genome = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    print("==========================================")
    print("STEP 1: GDC COHORT ASCERTAINMENT")
    print("==========================================")
    hits, query_metadata = fetch_gdc_hits(raw_gdc_path, args.refresh_gdc, args.page_size)
    variants, aggregation_stats = aggregate_gdc_variants(hits)
    print(f"Parsed {len(variants):,} unique single-nucleotide substitutions from {len(hits):,} GDC records.")

    reference_valid_variants: list[dict] = []
    reference_mismatch = 0
    for variant in variants:
        pos0 = variant["position"] - 1
        observed = str(genome[variant["chrom"]][pos0 : pos0 + 1]).upper()
        if observed != variant["ref"]:
            reference_mismatch += 1
            continue
        reference_valid_variants.append(variant)

    print("==========================================")
    print("STEP 2: FIXED GENCODE v44 REANNOTATION")
    print("==========================================")
    positions_by_chrom: dict[str, list[int]] = defaultdict(list)
    for variant in reference_valid_variants:
        positions_by_chrom[variant["chrom"]].append(variant["position"])
    positions_by_chrom = {chrom: sorted(set(positions)) for chrom, positions in positions_by_chrom.items()}

    candidate_transcript_ids = discover_candidate_transcripts(gtf_path, positions_by_chrom)
    raw_models = load_transcript_models(gtf_path, candidate_transcript_ids)
    transcript_models = prepare_transcript_models(raw_models, genome)

    position_to_transcripts: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for model in transcript_models.values():
        for segment in model["segments"]:
            positions = positions_by_chrom.get(model["chrom"], [])
            start_index = bisect.bisect_left(positions, segment["start"])
            end_index = bisect.bisect_right(positions, segment["end"])
            for position in positions[start_index:end_index]:
                position_to_transcripts[(model["chrom"], position)].append(model)

    annotated_variants: list[dict] = []
    no_local_synonymous = 0
    for variant in tqdm(reference_valid_variants, desc="Verifying synonymous consequences"):
        annotations: list[dict] = []
        for model in position_to_transcripts.get((variant["chrom"], variant["position"]), []):
            annotation = annotate_variant_against_transcript(
                variant["chrom"], variant["position"], variant["ref"], variant["alt"], model
            )
            if annotation and annotation["is_synonymous"]:
                annotations.append(annotation)

        if not annotations:
            no_local_synonymous += 1
            continue
        annotations.sort(key=transcript_priority)
        selected = annotations[0]
        annotated_variants.append({**variant, "selected_annotation": selected, "all_synonymous_annotations": annotations})

    print(f"Retained {len(annotated_variants):,} variants synonymous in at least one fixed GENCODE v44 transcript.")

    print("==========================================")
    print("STEP 3: CpG PAIRING AND MULTIMODAL INPUTS")
    print("==========================================")
    probe_index = build_probe_index(manifest_path)
    bw_handles = open_bigwig_handles(bw_paths)
    output_rows: list[dict] = []
    counters = defaultdict(int)

    try:
        for variant in tqdm(annotated_variants, desc="Building candidate inputs"):
            chrom = variant["chrom"]
            position_1based = variant["position"]
            position_zero = position_1based - 1
            nearest = nearest_probe(probe_index, chrom, position_zero)
            if nearest is None:
                counters["no_probe_on_chromosome"] += 1
                continue
            probe_id, cpg_pos = nearest
            offset = position_zero - cpg_pos
            if abs(offset) > CENTER_C_INDEX:
                counters["variant_outside_5kb_window"] += 1
                continue

            if offset in (0, 1):
                counters["variant_overlaps_target_cpg"] += 1
                continue

            sequence_start = cpg_pos - CENTER_C_INDEX
            sequence_end = sequence_start + WINDOW_SIZE
            if sequence_start < 0 or sequence_end > len(genome[chrom]):
                counters["window_out_of_bounds"] += 1
                continue

            healthy_sequence = str(genome[chrom][sequence_start:sequence_end]).upper()
            if len(healthy_sequence) != WINDOW_SIZE or healthy_sequence[CENTER_C_INDEX : CENTER_G_INDEX + 1] != "CG":
                counters["centered_cpg_failure"] += 1
                continue

            mutation_index = CENTER_C_INDEX + offset
            if healthy_sequence[mutation_index] != variant["ref"]:
                counters["window_reference_mismatch"] += 1
                continue
            mutated_sequence = (
                healthy_sequence[:mutation_index] + variant["alt"] + healthy_sequence[mutation_index + 1 :]
            )

            selected = variant["selected_annotation"]
            candidate_id = (
                f"{chrom}_{position_1based}_{variant['ref']}_{variant['alt']}__"
                f"{probe_id}__{selected['transcript_id']}"
            ).replace(".", "_")
            trinucleotide, sbs_class = sbs96_class(
                genome, chrom, position_1based, variant["ref"], variant["alt"]
            )

            row = {
                "Candidate_ID": candidate_id,
                "chr": chrom,
                "Variant_Position_1based": position_1based,
                "Variant_Position_0based": position_zero,
                "Reference_Allele": variant["ref"],
                "Alternate_Allele": variant["alt"],
                "GDC_Genomic_DNA_Change": variant["genomic_change"],
                "GDC_SSM_IDs": json.dumps(variant["gdc_ssm_ids"], separators=(",", ":")),
                "GDC_Case_IDs": json.dumps(variant["gdc_case_ids"], separators=(",", ":")),
                "GDC_Case_Submitter_IDs": json.dumps(
                    variant["gdc_case_submitter_ids"], separators=(",", ":")
                ),
                "GDC_Occurrence_Count": max(
                    len(variant["gdc_case_ids"]), len(variant["gdc_case_submitter_ids"])
                ),
                "Selected_Gene_ID": selected["gene_id"],
                "Selected_Gene_Name": selected["gene_name"],
                "Selected_Transcript_ID": selected["transcript_id"],
                "Selected_Transcript_Name": selected["transcript_name"],
                "Selected_Transcript_Type": selected["transcript_type"],
                "Selected_Transcript_Strand": selected["strand"],
                "Selected_Transcript_Tags": json.dumps(selected["tags"], separators=(",", ":")),
                "Selected_Transcript_Support_Level": selected["transcript_support_level"],
                "CDS_Position_1based": selected["cds_position_1based"],
                "Reference_Codon": selected["codon_ref"],
                "Alternate_Codon": selected["codon_alt"],
                "Amino_Acid": selected["amino_acid_ref"],
                "All_Synonymous_Transcript_Annotations": json.dumps(
                    variant["all_synonymous_annotations"], sort_keys=True, separators=(",", ":")
                ),
                "Reference_Trinucleotide": trinucleotide,
                "SBS96_Class": sbs_class,
                "probeID": probe_id,
                "pos": cpg_pos,
                "Mutation_Offset_From_CpG": offset,
                "Absolute_Distance_To_CpG": abs(offset),
                "Mutation_Index_5000_ZeroBased": mutation_index,
                "Model_Split": model_split_for_chrom(chrom, split_manifest),
                "Healthy_5000bp_DNA": healthy_sequence,
                "Mutated_5000bp_DNA": mutated_sequence,
                "Healthy_5000bp_DNA_RC": reverse_complement(healthy_sequence),
                "Mutated_5000bp_DNA_RC": reverse_complement(mutated_sequence),
                "Healthy_100bp_DNA": healthy_sequence[FASTA_SLICE],
                "Mutated_100bp_DNA": mutated_sequence[FASTA_SLICE],
                "Healthy_100bp_DNA_RC": reverse_complement(healthy_sequence[FASTA_SLICE]),
                "Mutated_100bp_DNA_RC": reverse_complement(mutated_sequence[FASTA_SLICE]),
            }

            region_start = cpg_pos - 49
            region_end = cpg_pos + 51
            raw_features: dict[str, float] = {}
            for feature_name, bw_obj in bw_handles.items():
                if feature_name == "Target_Base_PhyloP_100way_1":
                    value = get_bw_signal(bw_obj, chrom, cpg_pos, cpg_pos + 1)
                elif feature_name == "Target_Base_PhyloP_100way_2":
                    value = get_bw_signal(bw_obj, chrom, cpg_pos + 1, cpg_pos + 2)
                else:
                    value = get_bw_signal(bw_obj, chrom, region_start, region_end)
                raw_features[feature_name] = value

            for feature_name, value in raw_features.items():
                missing = not math.isfinite(float(value)) if not pd.isna(value) else True
                row[f"{feature_name}_Missing"] = int(missing)
                row[feature_name] = float(imputation_values[feature_name]) if missing else float(value)

            row["Target_Base_PhyloP_100way_1_RC"] = row["Target_Base_PhyloP_100way_2"]
            row["Target_Base_PhyloP_100way_2_RC"] = row["Target_Base_PhyloP_100way_1"]
            row["Target_Base_PhyloP_100way_1_RC_Missing"] = row["Target_Base_PhyloP_100way_2_Missing"]
            row["Target_Base_PhyloP_100way_2_RC_Missing"] = row["Target_Base_PhyloP_100way_1_Missing"]

            if row["Healthy_5000bp_DNA_RC"][CENTER_C_INDEX : CENTER_G_INDEX + 1] != "CG":
                counters["rc_centering_failure"] += 1
                continue
            output_rows.append(row)
    finally:
        for handle in bw_handles.values():
            handle.close()

    if not output_rows:
        raise RuntimeError("No candidate rows survived fixed annotation and sequence construction.")

    df_final = sort_dataframe(pd.DataFrame(output_rows))
    duplicate_mask = df_final.duplicated(
        subset=["chr", "Variant_Position_1based", "Reference_Allele", "Alternate_Allele", "probeID", "Selected_Transcript_ID"],
        keep="first",
    )
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        df_final = df_final.loc[~duplicate_mask].reset_index(drop=True)

    all_output_path = datafiles_dir / "testing_data.csv"
    test_only_path = datafiles_dir / "testing_data_test_only.csv"
    df_final.to_csv(all_output_path, index=False)
    df_final[df_final["Model_Split"].eq("test")].to_csv(test_only_path, index=False)

    fasta_paths = {
        "healthy_forward": (datafiles_dir / "test_healthy_100bp.fasta", "Healthy_5000bp_DNA"),
        "healthy_rc": (datafiles_dir / "test_healthy_100bp_rc.fasta", "Healthy_5000bp_DNA_RC"),
        "mutated_forward": (datafiles_dir / "test_mutated_100bp.fasta", "Mutated_5000bp_DNA"),
        "mutated_rc": (datafiles_dir / "test_mutated_100bp_rc.fasta", "Mutated_5000bp_DNA_RC"),
    }
    write_candidate_fastas(df_final, fasta_paths)

    gdc_query_hash = canonical_json_sha256({"query_metadata": query_metadata, "hits": hits})
    input_records = {
        "reference_fasta": file_record(fasta_path, hash_file=args.hash_large_inputs),
        "reference_fasta_index": file_record(Path(str(fasta_path) + ".fai")) if Path(str(fasta_path) + ".fai").exists() else None,
        "hm450_manifest": file_record(manifest_path),
        "gencode_v44_gtf": file_record(gtf_path),
        "split_manifest": file_record(split_manifest_path),
        "feature_imputation": file_record(imputation_path),
        "bigwigs": {name: file_record(path, hash_file=args.hash_large_inputs) for name, path in bw_paths.items()},
        "raw_gdc_cache": file_record(raw_gdc_path),
    }

    output_records = {
        "testing_data": file_record(all_output_path),
        "testing_data_test_only": file_record(test_only_path),
        **{name: file_record(path) for name, (path, _) in fasta_paths.items()},
    }

    cohort_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": file_record(Path(__file__).resolve()),
        "cohort_definition": {
            "project": "TCGA-BRCA",
            "discovery_filter": "GDC consequence.transcript.consequence_type contains synonymous_variant",
            "local_eligibility": "single-nucleotide substitution verified synonymous against at least one GENCODE v44 CDS transcript; variant must not alter either base of the centered target CpG",
            "selected_transcript_rule": [
                "MANE_Select",
                "Ensembl_canonical",
                "APPRIS principal",
                "protein_coding transcript type",
                "GENCODE basic",
                "lowest transcript support level",
                "longest CDS",
                "lexicographic transcript ID",
            ],
            "cbioportal_filter_used": False,
            "methylation_availability_filter_used": False,
            "candidate_labels_used": False,
            "primary_candidate_subset": "Model_Split == test",
        },
        "gdc_query": {**query_metadata, "canonical_query_and_hits_sha256": gdc_query_hash},
        "counts": {
            "raw_gdc_hits": len(hits),
            **aggregation_stats,
            "reference_allele_mismatches": reference_mismatch,
            "reference_valid_snvs": len(reference_valid_variants),
            "candidate_transcripts_examined": len(transcript_models),
            "not_synonymous_in_fixed_gencode_v44": no_local_synonymous,
            "fixed_release_synonymous_variants": len(annotated_variants),
            "sequence_and_probe_valid_rows_before_deduplication": len(output_rows),
            "duplicate_candidate_rows_removed": duplicate_count,
            "final_candidate_rows": len(df_final),
            "final_test_chromosome_candidates": int(df_final["Model_Split"].eq("test").sum()),
            "final_validation_chromosome_candidates": int(df_final["Model_Split"].eq("val").sum()),
            "final_training_chromosome_candidates": int(df_final["Model_Split"].eq("train").sum()),
            **{key: int(value) for key, value in counters.items()},
        },
        "reverse_complement": {
            "forward_and_rc_sequences_emitted": True,
            "inference_requirement": "average forward and reverse-complement predictions",
            "ordered_feature_transform": {
                "Target_Base_PhyloP_100way_1_RC": "Target_Base_PhyloP_100way_2",
                "Target_Base_PhyloP_100way_2_RC": "Target_Base_PhyloP_100way_1",
            },
        },
        "inputs": input_records,
        "outputs": output_records,
    }
    cohort_manifest_path = datafiles_dir / "candidate_cohort_manifest.json"
    cohort_manifest_path.write_text(json.dumps(cohort_manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(cohort_manifest["counts"], indent=2, sort_keys=True))
    print(f"Clean candidate cohort written to: {all_output_path}")
    print(f"Held-out candidate subset written to: {test_only_path}")
    print(f"Cohort manifest written to: {cohort_manifest_path}")


if __name__ == "__main__":
    main()
