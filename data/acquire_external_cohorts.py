#!/usr/bin/env python3
"""Download and checksum the external cohorts used for variant-effect validation. Every
file is recorded in data/external/external_manifest.json with its URL, byte count and
sha256, and source assemblies are declared per source, since several mQTL resources are
hg19 while this project is hg38. Sources without a stable public URL are listed as
MANUAL: the script says what to place where, then validates it like any other source.
Use --check for reachability, --only NAME or --all to fetch, --verify-only to re-
checksum what is already on disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional


DEFAULT_ROOT = Path("data/external")
MANIFEST_NAME = "external_manifest.json"
USER_AGENT = "SilentMethyl-StageA/1.0 (academic use)"
CHUNK = 1 << 20


@dataclasses.dataclass(frozen=True)
class Source:
    key: str
    title: str
    purpose: str
    build: str
    files: tuple
    manual_note: Optional[str] = None
    approx_size: str = "unknown"
    citation: str = ""

    @property
    def is_manual(self) -> bool:
        return any(url is None for _, url in self.files)

    @property
    def needs_liftover(self) -> bool:
        return self.build == "hg19"


def _zenodo(record: str, filename: str) -> str:
    return f"https://zenodo.org/records/{record}/files/{filename}?download=1"


GENOA_RECORD = "7697509"

SOURCES: tuple = (
    Source(
        key="genoa_meqtl",
        title="GENOA cis-meQTL summary statistics (African American, EPIC)",
        purpose="independent variant evaluation; cross-ancestry replication",
        build="hg19",
        approx_size="5.3 GB (22 files)",
        citation="Shang et al., Nat Commun 2023; Zenodo 10.5281/zenodo.7697509",
        files=(
            ("GENOA_meQTL_README.txt", _zenodo(GENOA_RECORD, "GENOA_meQTL_README.txt")),
            *(
                (f"meQTL_summarystat_chr{c}.txt.gz",
                 _zenodo(GENOA_RECORD, f"meQTL_summarystat_chr{c}.txt.gz"))
                for c in range(1, 23)
            ),
        ),
    ),
    Source(
        key="godmc_mqtl",
        title="GoDMC mQTL meta-analysis (blood, n=32,851)",
        purpose="independent variant evaluation; European-ancestry arm",
        build="hg19",
        approx_size="large; depends on release tier",
        citation="Min et al., Nat Genet 2021; http://mqtldb.godmc.org.uk",
        files=(("godmc_mqtl_results.tsv.gz", None),),
        manual_note=textwrap.dedent("""\
            GoDMC serves results through http://mqtldb.godmc.org.uk (browser +
            RESTful API) rather than a stable bulk URL. Fetch the full
            meta-analysis tier, place it at the path above, then re-run with
            --verify-only.

            CONFIRM BEFORE USE: the genome build of the SNP and CpG coordinate
            columns. The 2021 release is ARIES/HM450-era and is expected to be
            hg19, but this script will not assume it -- set build in the
            registry once you have read the column spec, or liftover will be
            skipped or applied wrongly."""),
    ),
    Source(
        key="egtex_mqtl",
        title="eGTEx mQTL summary statistics, 9 tissues",
        purpose="cross-tissue validation; independent variant evaluation",
        build="hg38",
        approx_size="varies by tissue",
        citation="Oliva et al., Nat Genet 2023; GTEx Portal; GEO GSE213478",
        files=(("egtex_mqtl_by_tissue.tar", None),),
        manual_note=textwrap.dedent("""\
            Download the methylation QTL bundle from the GTEx Portal datasets
            page (https://gtexportal.org/home/datasets). Normalized methylation
            matrices are additionally at GEO GSE213478. Individual genotypes are
            dbGaP-protected and are NOT needed -- summary statistics only.

            GTEx v8+ is hg38, so no liftover. Verify that the mQTL files you
            pull are the hg38 release and not a legacy v7 export."""),
    ),
    Source(
        key="clinvar",
        title="ClinVar variant summary (weekly VCF, GRCh38)",
        purpose="clinical-relevance figure; BEND ClinVar task cross-check",
        build="hg38",
        approx_size="~250 MB",
        citation="NCBI ClinVar",
        files=(
            ("clinvar_GRCh38.vcf.gz",
             "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"),
            ("clinvar_GRCh38.vcf.gz.tbi",
             "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi"),
        ),
    ),
    Source(
        key="jaspar",
        title="JASPAR CORE vertebrates, non-redundant PFMs",
        purpose="regulatory enrichment; motif recovery",
        build="n/a",
        approx_size="~5 MB",
        citation="JASPAR",
        files=(
            ("JASPAR_CORE_vertebrates_non-redundant_pfms_jaspar.txt", None),
        ),
        manual_note=textwrap.dedent("""\
            Download the current CORE vertebrates non-redundant PFM set from
            https://jaspar.elixir.no/downloads/ . Pin the release year in the
            manifest note -- motif set version materially changes enrichment
            results and reviewers ask which release was used."""),
    ),
    Source(
        key="hocomoco",
        title="HOCOMOCO core motif collection",
        purpose="regulatory enrichment; second motif reference",
        build="n/a",
        approx_size="~10 MB",
        citation="HOCOMOCO",
        files=(("hocomoco_core_pwm.tar.gz", None),),
        manual_note="Download the current human CORE collection from https://hocomoco.autosome.org/downloads .",
    ),
    Source(
        key="gnomad_af",
        title="gnomAD population allele frequencies",
        purpose="SNP-under-probe audit, stratified by population",
        build="hg38",
        approx_size="very large; consider per-chromosome sites-only VCF",
        citation="gnomAD",
        files=(("gnomad_sites_af.tsv.gz", None),),
        manual_note=textwrap.dedent("""\
            You do not need full gnomAD. Extract a sites-only table of
            (chrom, pos, ref, alt, AF_afr, AF_amr, AF_eas, AF_nfe, AF_sas)
            restricted to +/- 60 bp around HM450/EPIC probe coordinates.
            That reduces this from terabytes to a few hundred MB and is the
            only part the probe audit uses."""),
    ),
    Source(
        key="bend",
        title="BEND benchmark task data",
        purpose="multi-task validation; the NMI 'several genomic tasks' criterion",
        build="hg38",
        approx_size="varies",
        citation="Marin et al., ICLR 2024; github.com/frederikkemarin/BEND",
        files=(("BEND_repo", None),),
        manual_note=textwrap.dedent("""\
            git clone https://github.com/frederikkemarin/BEND into this path and
            follow its data download instructions.

            NOTE: BEND uses a frozen-embedding protocol -- precompute
            embeddings once, then train lightweight downstream heads. Entering
            BEND does NOT require fine-tuning SilentMethyl per task. You will
            need to add an embedder class in bend/utils/embedders.py that
            inherits BaseEmbedder and implements load_model() and embed()."""),
    ),
)

SOURCES_BY_KEY = {s.key: s for s in SOURCES}


def sha256_of(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def _request(url: str, method: str = "GET", extra_headers: Optional[dict] = None):
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    return urllib.request.Request(url, method=method, headers=headers)


def check_url(url: str, timeout: int = 30) -> tuple:
    """Return (ok, status_or_error, content_length)."""
    for method in ("HEAD", "GET"):
        try:
            req = _request(url, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                length = resp.headers.get("Content-Length")
                return True, resp.status, int(length) if length else None
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in (403, 405, 501):
                continue
            return False, f"HTTP {exc.code}", None
        except Exception as exc:  # noqa: BLE001 - report anything, keep going
            if method == "HEAD":
                continue
            return False, f"{type(exc).__name__}: {exc}", None
    return False, "unreachable", None


def download(url: str, dest: Path, timeout: int = 60, retries: int = 3) -> None:
    """Download with resume support and simple retry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else None
        try:
            req = _request(url, extra_headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total else None
                got = have
                last = time.time()
                with tmp.open(mode) as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        got += len(block)
                        if time.time() - last > 5:
                            pct = f" ({100 * got / total:.1f}%)" if total else ""
                            print(f"      {human(got)}{pct}", flush=True)
                            last = time.time()
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"      attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def sniff_header(path: Path, n: int = 3) -> list:
    """Return the first n lines of a text or gzip file, for schema recording."""
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", errors="replace") as fh:  # type: ignore[operator]
            return [next(fh).rstrip("\n") for _ in range(n)]
    except StopIteration:
        return []
    except Exception as exc:  # noqa: BLE001
        return [f"<unreadable: {exc}>"]


CHAIN_DEFAULT = Path("data/reference/hg19ToHg38.over.chain.gz")


def liftover_available() -> tuple:
    """Report which liftover backend is usable, if any."""
    if shutil.which("CrossMap.py") or shutil.which("CrossMap"):
        return True, "CrossMap"
    try:
        import pyliftover  # noqa: F401
        return True, "pyliftover"
    except ImportError:
        pass
    if shutil.which("liftOver"):
        return True, "ucsc-liftOver"
    return False, "none"


def report_liftover_plan(sources: Iterable[Source], chain: Path) -> list:
    """Do not silently lift. Report exactly what will need lifting and how."""
    notes = []
    ok, backend = liftover_available()
    needing = [s for s in sources if s.needs_liftover]
    if not needing:
        return notes

    print("\n[liftover] Sources declared hg19 -- these MUST be lifted to hg38:")
    for s in needing:
        print(f"    - {s.key:<16} {s.title}")
        notes.append({"source": s.key, "from": "hg19", "to": "hg38"})
    print(f"[liftover] chain file: {chain} ({'present' if chain.exists() else 'MISSING'})")
    print(f"[liftover] backend:    {backend}")
    if not ok:
        print("[liftover] No backend found. Install one before the harmonization step:")
        print("             pip install CrossMap        # or")
        print("             pip install pyliftover")
    print("[liftover] Lifting is intentionally NOT performed by this script. It")
    print("           belongs in the harmonization step so that the raw download")
    print("           stays byte-identical to the published release and remains")
    print("           checksum-verifiable against the source.")
    return notes


def build_manifest(root: Path, sources: Iterable[Source], chain: Path) -> dict:
    entries = {}
    missing = []
    for src in sources:
        for rel, url in src.files:
            path = root / src.key / rel
            record = {
                "source": src.key,
                "title": src.title,
                "purpose": src.purpose,
                "genome_build": src.build,
                "needs_liftover_to_hg38": src.needs_liftover,
                "citation": src.citation,
                "url": url,
                "acquisition": "manual" if url is None else "automatic",
            }
            if path.exists() and path.is_file():
                record["bytes"] = path.stat().st_size
                record["sha256"] = sha256_of(path)
                record["head"] = sniff_header(path)
                record["status"] = "present"
            elif path.exists() and path.is_dir():
                record["status"] = "present_directory"
            else:
                record["status"] = "MISSING"
                missing.append(str(path))
            entries[str(path)] = record

    _, backend = liftover_available()
    return {
        "stage": "A -- external data acquisition",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "entry_count": len(entries),
        "missing_count": len(missing),
        "missing": missing,
        "liftover": {
            "chain_file": str(chain),
            "chain_present": chain.exists(),
            "backend_available": backend,
            "sources_requiring_liftover": [s.key for s in sources if s.needs_liftover],
            "policy": (
                "Raw downloads are stored byte-identical to the published release. "
                "Coordinate conversion happens in the harmonization step and writes "
                "new files; it never modifies a checksummed download in place."
            ),
        },
        "entries": entries,
    }


def write_manifest(root: Path, manifest: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    with path.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def cmd_check(sources: Iterable[Source]) -> int:
    print("Probing endpoints. Manual sources are listed but not probed.\n")
    problems = 0
    for src in sources:
        flag = " [MANUAL]" if src.is_manual else ""
        lift = "  (hg19 -> needs liftover)" if src.needs_liftover else ""
        print(f"{src.key}{flag}{lift}")
        print(f"    {src.title}")
        print(f"    serves: {src.purpose}")
        print(f"    size:   {src.approx_size}")
        if src.manual_note:
            for line in src.manual_note.splitlines():
                print(f"    | {line}")
        for rel, url in src.files:
            if url is None:
                print(f"    - {rel:<52} place manually")
                continue
            ok, status, length = check_url(url)
            size = f"  {human(length)}" if length else ""
            mark = "ok " if ok else "ERR"
            print(f"    - {rel:<52} {mark} {status}{size}")
            if not ok:
                problems += 1
        print()
    if problems:
        print(f"{problems} endpoint(s) failed to respond. "
              "Re-check from the cluster: compute nodes are often egress-restricted, "
              "so run downloads from a login or transfer node.")
    return 0 if problems == 0 else 1


def cmd_fetch(sources: Iterable[Source], root: Path, force: bool) -> int:
    failures = 0
    for src in sources:
        print(f"\n=== {src.key}: {src.title}")
        if src.is_manual and src.manual_note:
            print("--- MANUAL SOURCE ---")
            for line in src.manual_note.splitlines():
                print(f"    {line}")
        for rel, url in src.files:
            dest = root / src.key / rel
            if url is None:
                state = "present" if dest.exists() else "AWAITING MANUAL PLACEMENT"
                print(f"  {rel:<52} {state}")
                continue
            if dest.exists() and not force:
                print(f"  {rel:<52} already present ({human(dest.stat().st_size)})")
                continue
            print(f"  {rel:<52} downloading...")
            try:
                download(url, dest)
                print(f"      done, {human(dest.stat().st_size)}")
            except Exception as exc:  # noqa: BLE001
                print(f"      FAILED: {exc}")
                failures += 1
    return failures


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage A external data acquisition for SilentMethyl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"destination root (default: {DEFAULT_ROOT})")
    ap.add_argument("--chain", type=Path, default=CHAIN_DEFAULT,
                    help="hg19->hg38 chain file used by the later harmonization step")
    ap.add_argument("--check", action="store_true",
                    help="probe endpoints and print the plan; download nothing. "
                         "Combine with --only to check a subset.")
    ap.add_argument("--only", action="append", metavar="KEY",
                    help="restrict to one source (repeatable); keys: "
                         + ", ".join(SOURCES_BY_KEY))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="fetch every automatic source")
    mode.add_argument("--verify-only", action="store_true",
                      help="checksum what is on disk and rewrite the manifest")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--list", action="store_true", help="list source keys and exit")
    args = ap.parse_args(argv)

    if args.list:
        for s in SOURCES:
            print(f"{s.key:<16} {s.build:<5} {s.title}")
        return 0

    if args.only:
        unknown = [k for k in args.only if k not in SOURCES_BY_KEY]
        if unknown:
            ap.error(f"unknown source key(s): {', '.join(unknown)}")
        selected = tuple(SOURCES_BY_KEY[k] for k in args.only)
    else:
        selected = SOURCES

    if args.check:
        return cmd_check(selected)

    failures = 0
    if not args.verify_only:
        if not (args.all or args.only):
            ap.error("choose one of --check, --all, --only KEY, or --verify-only")
        failures = cmd_fetch(selected, args.root, args.force)

    report_liftover_plan(selected, args.chain)

    print("\n[manifest] checksumming...")
    manifest = build_manifest(args.root, SOURCES, args.chain)
    path = write_manifest(args.root, manifest)
    print(f"[manifest] wrote {path}")
    print(f"[manifest] {manifest['entry_count'] - manifest['missing_count']}"
          f"/{manifest['entry_count']} artifacts present")
    if manifest["missing"]:
        print("[manifest] still missing:")
        for m in manifest["missing"]:
            print(f"    - {m}")

    if failures:
        print(f"\n{failures} download(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
