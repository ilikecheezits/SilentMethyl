#!/usr/bin/env python3
"""Resolve and fetch the targets and context tracks for the multi-tissue build.

What this gets, per tissue
--------------------------
targets   TCGA solid-tissue-normal HM450 beta values, as the pre-assembled
          per-project matrix from the UCSC Xena GDC hub. This is the same
          source the published breast model used --
          TCGA-BRCA.methylation450.tsv.gz, recorded in README.md -- so the four
          new tissues arrive in exactly the format build_training_data.py
          already consumes. Pulling the 471 per-aliquot files from the GDC
          instead would be ~60 GB and would need assembling into a matrix that
          Xena has already assembled.

context   Seven ENCODE bigWigs: ATAC plus six histone marks, all as
          'fold change over control', GRCh38, released.

The consistency problem this fixes
----------------------------------
The published breast context is NOT internally consistent, which
data/audit_reference_tracks.py established on 10 Sep 2026: the six histone
marks are Mint-ChIP-seq fold-change bigWigs downloaded from ENCODE, but
accessibility is a coverage track generated locally from an unreplicated
single-nucleus ATAC BAM (ENCFF021PIS, ENCSR037XNN) produced by a lab-custom
pipeline. Its conversion command was never recorded.

That recipe cannot be repeated for kidney, lung, prostate and colon, because
snATAC in those primary tissues largely does not exist -- what they have is
bulk ATAC-seq. So this script resolves BULK ATAC for every tissue INCLUDING
breast, giving one assay and one processing type across all five. The cost is
that breast's accessibility feature changes, which is why the plan calls for one
extra single-tissue breast run as the matched control. The published MCF-10A
model is untouched and remains the paper's primary result.

Everything is recorded before anything is downloaded. Each ENCODE file carries
an md5 from the portal, so downloads are verified rather than trusted -- which
is the provenance gap that made the breast tracks unreconstructable in the first
place.

Usage (run from the repository root)
------------------------------------
    python -u data/acquire_multitissue_inputs.py                    # resolve + plan only
    python -u data/acquire_multitissue_inputs.py --tissues Lung KidneyCortex
    python -u data/acquire_multitissue_inputs.py --apply            # download too
    python -u data/acquire_multitissue_inputs.py --apply --context-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ENCODE = "https://www.encodeproject.org"
XENA = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/{project}.methylation450.tsv.gz"
UA = {"Accept": "application/json", "User-Agent": "SilentMethyl-acquire"}

# Seven tissue-specific features, matching TABULAR_FEATURES in
# scripts/training_common.py. PhyloP is genomic conservation, identical in every
# tissue, and is already present as data/reference/hg38.phyloP100way.bw.
MARKS = [
    ("Ref_ATAC_Signal",     "ATAC-seq",         None,       "ATAC_seq.bw"),
    ("Ref_H3K4me3_Signal",  "Histone ChIP-seq", "H3K4me3",  "H3K4me3.bw"),
    ("Ref_H3K27ac_Signal",  "Histone ChIP-seq", "H3K27ac",  "H3K27ac.bw"),
    ("Ref_H3K27me3_Signal", "Histone ChIP-seq", "H3K27me3", "H3K27me3.bw"),
    ("Ref_H3K9me3_Signal",  "Histone ChIP-seq", "H3K9me3",  "H3K9me3.bw"),
    ("Ref_H3K36me3_Signal", "Histone ChIP-seq", "H3K36me3", "H3K36me3.bw"),
    ("Ref_H3K4me1_Signal",  "Histone ChIP-seq", "H3K4me1",  "H3K4me1.bw"),
]

# Mint-ChIP-seq is a distinct assay_title on the portal from Histone ChIP-seq,
# and the breast tracks are Mint. Both are bulk histone ChIP producing the same
# fold-change output, so either is acceptable -- but which one was used has to
# be recorded, and mixing them within a tissue would not be.
HISTONE_ASSAYS = ["Histone ChIP-seq", "Mint-ChIP-seq"]

OUTPUT_TYPE = "fold change over control"

# ---------------------------------------------------------------------------
# snATAC -> bigWig conversion
# ---------------------------------------------------------------------------
# ENCODE publishes no fold-change bigWig for single-nucleus ATAC, so keeping
# MCF-10A as the breast context means converting a BAM per tissue. The command
# below is applied IDENTICALLY to every tissue, MCF-10A included, and is written
# into the manifest with each track. That last part is the whole point: the
# existing data/reference/ATAC_seq.bw was converted with parameters nobody
# recorded, which is why it cannot be matched and has to be redone.
#
# Why each flag:
#   --normalizeUsing CPM   depth differs between experiments, and without
#                          normalisation a deeper-sequenced tissue simply gets
#                          larger numbers -- a scale offset perfectly correlated
#                          with tissue, which is the artifact this route exists
#                          to avoid. CPM is the standard choice and needs no
#                          input control, which snATAC does not have.
#   --binSize 10           near the resolution of the ENCODE histone signal the
#                          other six features come from. Larger bins blur the
#                          accessibility peaks that matter at a 1 kb window.
#   --ignoreDuplicates     10x libraries carry PCR duplicates.
#   --minMappingQuality 30 drops multi-mapping reads, standard for ATAC.
#   --blacklist            ENCODE GRCh38 exclusion list; ATAC is especially
#                          prone to artifact pileups in these regions.
SNATAC_BAMCOVERAGE = [
    "bamCoverage",
    "--normalizeUsing", "CPM",
    "--binSize", "10",
    "--ignoreDuplicates",
    "--minMappingQuality", "30",
    "--extendReads",
    "--numberOfProcessors", "max",
]

# deeptools renames and drops options between releases -- --ignoreDuplicates is
# absent from recent builds, for instance. Silently dropping a filter would
# change the output while the recorded command still claimed it, so any flag the
# installed binary does not accept is swapped for a documented equivalent, or
# the run stops. SAM flag 1024 is the PCR/optical duplicate bit, so excluding it
# is exactly what --ignoreDuplicates did.
FLAG_FALLBACKS = {
    "--ignoreDuplicates": ["--samFlagExclude", "1024"],
}
# ENCODE GRCh38 exclusion list (Amemiya et al. 2019), fetched if absent.
BLACKLIST_ACCESSION = "ENCFF356LFX"

# TCGA projects supply the targets; ENCODE biosample terms supply the context.
# Donor counts from data/survey_tcga_normal_cohorts.py, 10 Sep 2026.
#
# Kidney is KIRC only. Pooling KIRP and KICH reaches 205 donors, but KICH is
# chromophobe, which arises from the distal nephron and collecting duct rather
# than the cortex, and the eGTEx tissue being matched is Kidney Cortex.
# Anatomical match beats sample count here.
TISSUES = {
    "BreastMammaryTissue": {
        "projects": ["TCGA-BRCA"],
        "encode": ["breast epithelium", "MCF 10A",
                   "luminal epithelial cell of mammary gland"],
        "note": "already published on MCF-10A context; rebuilt here only so all "
                "five tissues share one accessibility assay",
    },
    "KidneyCortex": {
        "projects": ["TCGA-KIRC"],
        "encode": ["kidney", "kidney epithelial cell", "renal cortex interstitium"],
        "note": "KIRC only; KICH is not cortex-derived",
    },
    "Lung": {
        "projects": ["TCGA-LUAD", "TCGA-LUSC"],
        "encode": ["lung", "upper lobe of left lung"],
        "note": "adenocarcinoma and squamous normals pooled",
    },
    "Prostate": {
        "projects": ["TCGA-PRAD"],
        "encode": ["prostate gland", "prostate", "epithelial cell of prostate"],
        "note": "",
    },
    "ColonTransverse": {
        "projects": ["TCGA-COAD", "TCGA-READ"],
        "encode": ["transverse colon", "sigmoid colon"],
        "note": "COAD and READ normals pooled; transverse colon preferred to "
                "match the eGTEx tissue",
    },
}


# Friendly names accepted in a picks file, mapped to the feature the model uses.
# Anything not listed here is not part of the context vector: TABULAR_FEATURES in
# scripts/training_common.py fixes it at seven tissue-specific inputs, so an
# eighth mark would change the epigenomic tower's input dimension and stop being
# the published architecture. H3K9ac is the usual near-miss -- ENCODE has it for
# many tissues and it looks like it belongs, but the model was never built with
# it.
MARK_ALIASES = {
    "atac": "Ref_ATAC_Signal", "atac-seq": "Ref_ATAC_Signal",
    "atac_seq": "Ref_ATAC_Signal",
    # snATAC accessibility, supplied as a BAM to convert locally. ENCODE
    # publishes no fold-change bigWig for single-nucleus experiments, so every
    # tissue on this route goes through the same conversion below.
    "atac_bam": "Ref_ATAC_Signal", "snatac": "Ref_ATAC_Signal",
    "snatac_bam": "Ref_ATAC_Signal",
    "dnase": None,
    "h3k4me3": "Ref_H3K4me3_Signal",
    "h3k27ac": "Ref_H3K27ac_Signal",
    "h3k27me3": "Ref_H3K27me3_Signal",
    "h3k9me3": "Ref_H3K9me3_Signal",
    "h3k36me3": "Ref_H3K36me3_Signal",
    "h3k4me1": "Ref_H3K4me1_Signal",
}
MARK_TARGET = {feature: target for feature, _, target, _ in MARKS}
MARK_FILENAME = {feature: filename for feature, _, _, filename in MARKS}


class Unreachable(RuntimeError):
    pass


def _json(url: str, timeout: int = 90):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.load(fh)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:      # ENCODE says "no matches" with a 404
            return None
        raise Unreachable(f"HTTP {exc.code} from {url.split('?')[0]}") from exc
    except urllib.error.URLError as exc:
        raise Unreachable(f"cannot reach {url.split('?')[0]} ({exc.reason})") from exc


# ---------------------------------------------------------------------------
# ENCODE resolution
# ---------------------------------------------------------------------------

def _file_search(term: str, assay: str, target: str | None) -> list[dict]:
    params = [
        ("type", "File"), ("status", "released"),
        ("file_format", "bigWig"), ("assembly", "GRCh38"),
        ("output_type", OUTPUT_TYPE),
        ("biosample_ontology.term_name", term),
        ("assay_title", assay),
        ("format", "json"), ("limit", "all"),
    ]
    if target:
        params.append(("target.label", target))
    hit = _json(f"{ENCODE}/search/?{urllib.parse.urlencode(params)}")
    return [] if not hit else hit.get("@graph", [])


def _pick(files: list[dict]) -> dict | None:
    """Deterministic choice among equivalent files.

    ENCODE flags one file per experiment as preferred_default; take it when
    present. Otherwise prefer a file covering more biological replicates -- a
    two-replicate pooled signal is a better feature than one replicate -- and
    break ties on accession so repeated runs resolve identically.
    """
    if not files:
        return None
    def key(f):
        return (0 if f.get("preferred_default") else 1,
                -len(f.get("biological_replicates") or []),
                f.get("accession", ""))
    return sorted(files, key=key)[0]


def resolve_context(tissue: str, terms: list[str], verbose: bool = True) -> dict:
    """Best biosample term for this tissue, and one file accession per mark."""
    best = None
    for term in terms:
        chosen, assay_used = {}, {}
        for feature, assay, target, filename in MARKS:
            assays = HISTONE_ASSAYS if assay == "Histone ChIP-seq" else [assay]
            hit = None
            for a in assays:
                found = _file_search(term, a, target)
                time.sleep(0.1)
                if found:
                    hit = _pick(found)
                    assay_used[feature] = a
                    break
            if hit:
                chosen[feature] = {
                    "accession": hit.get("accession"),
                    "filename": filename,
                    "assay_title": assay_used[feature],
                    "target": target,
                    "output_type": hit.get("output_type"),
                    "assembly": hit.get("assembly"),
                    "md5": hit.get("md5sum"),
                    "bytes": hit.get("file_size"),
                    "biological_replicates": hit.get("biological_replicates"),
                    "url": f"{ENCODE}{hit['href']}" if hit.get("href") else None,
                    "dataset": hit.get("dataset"),
                }
        n = len(chosen)
        if verbose:
            print(f"    {term:<38} {n}/{len(MARKS)}")
        if best is None or n > len(best["files"]):
            best = {"term": term, "files": chosen}
        if n == len(MARKS):
            break
    best["complete"] = len(best["files"]) == len(MARKS)
    best["missing"] = [f for f, _, _, _ in MARKS if f not in best["files"]]
    assays = {v["assay_title"] for v in best["files"].values()}
    best["histone_assays"] = sorted(a for a in assays if "ChIP" in a)
    return best


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def read_picks(path: Path) -> dict[str, dict[str, str]]:
    """Parse a hand-written picks file into {tissue: {feature: accession}}.

    Format is deliberately forgiving, because these get pasted together from
    portal searches. One record per line, blank lines and # comments ignored:

        # tissue        mark        accession-or-url
        Lung            atac        ENCFF260QGF
        Lung            h3k4me1     https://www.encodeproject.org/files/ENCFF325EMH/@@download/ENCFF325EMH.bigWig

    A bare `Tissue:` line also opens a block, so a list can be written per
    tissue without repeating the name on every row.
    """
    picks: dict[str, dict[str, str]] = {}
    current = None
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":") and " " not in line.rstrip(":"):
            current = line.rstrip(":")
            picks.setdefault(current, {})
            continue
        parts = line.replace(",", " ").replace("\t", " ").split()
        parts = [p for p in parts if p not in ("-", "=")]
        if len(parts) >= 3:
            tissue, mark, token = parts[0], parts[1], parts[2]
        elif len(parts) == 2 and current:
            tissue, mark, token = current, parts[0], parts[1]
        else:
            raise SystemExit(f"STOP: {path}:{lineno}: cannot parse {raw!r}")

        key = mark.strip().lower().rstrip(":-")
        if key not in MARK_ALIASES:
            raise SystemExit(
                f"STOP: {path}:{lineno}: {mark!r} is not one of the model's "
                f"context features.\n"
                f"  the line reads: {raw.rstrip()!r}\n"
                f"The seven are: "
                f"{', '.join(sorted(k for k, v in MARK_ALIASES.items() if v))}.\n"
                f"H3K9ac and DNase are NOT used -- adding one would change the "
                f"epigenomic tower's input dimension.\n"
                f"If this line is a note rather than a pick, start it with '#'.")
        feature = MARK_ALIASES[key]
        if feature is None:
            raise SystemExit(f"STOP: {path}:{lineno}: {mark!r} is not a model input")

        acc = token.strip()
        as_bam = key in ("atac_bam", "snatac", "snatac_bam") or acc.endswith(".bam")
        if "/files/" in acc:                     # a full download URL
            acc = acc.split("/files/")[1].split("/")[0]
        if not acc.startswith("ENCFF"):
            raise SystemExit(f"STOP: {path}:{lineno}: {token!r} is not an ENCFF accession")
        picks.setdefault(tissue, {})[feature] = {"accession": acc, "as_bam": as_bam}
    return picks


def verify_pick(feature: str, accession: str) -> dict:
    """Fetch a hand-picked file's metadata and check it is what it claims."""
    f = _json(f"{ENCODE}/files/{accession}/?format=json")
    if not f:
        return {"accession": accession, "feature": feature, "ok": False,
                "problems": ["no such file on ENCODE"]}

    rec = {
        "accession": accession, "filename": MARK_FILENAME[feature],
        "file_format": f.get("file_format"), "output_type": f.get("output_type"),
        "assembly": f.get("assembly"), "encode_status": f.get("status"),
        "md5": f.get("md5sum"), "bytes": f.get("file_size"),
        "biological_replicates": f.get("biological_replicates"),
        "url": f"{ENCODE}{f['href']}" if f.get("href") else None,
        "dataset": f.get("dataset"),
    }
    exp = _json(f"{ENCODE}{rec['dataset']}?format=json") if rec["dataset"] else None
    if exp:
        rec.update({
            "experiment": exp.get("accession"),
            "assay_title": exp.get("assay_title"),
            "target": (exp.get("target") or {}).get("label"),
            "biosample": (exp.get("biosample_ontology") or {}).get("term_name"),
            "replication_type": exp.get("replication_type"),
            "lab": (exp.get("lab") or {}).get("title"),
        })

    problems = []
    if rec["file_format"] != "bigWig":
        problems.append(f"file_format is {rec['file_format']}, not bigWig")
    if rec["output_type"] != OUTPUT_TYPE:
        problems.append(f"output_type is {rec['output_type']!r}, not {OUTPUT_TYPE!r}")
    if rec["assembly"] != "GRCh38":
        problems.append(f"assembly is {rec['assembly']}, not GRCh38")
    if rec["encode_status"] != "released":
        problems.append(f"status is {rec['encode_status']}, not released")
    want = MARK_TARGET[feature]
    if want and rec.get("target") != want:
        problems.append(f"target is {rec.get('target')!r}, expected {want!r}")
    if not want and "ATAC" not in (rec.get("assay_title") or "").upper():
        problems.append(f"assay is {rec.get('assay_title')!r}, expected ATAC")
    rec["problems"] = problems
    rec["ok"] = not problems
    return rec


def verify_bam_pick(feature: str, accession: str) -> dict:
    """Same as verify_pick, but for an alignment file we will convert ourselves.

    A BAM is judged by different rules than a signal track: there is no output
    type to match, and 'alignments' rather than 'unfiltered alignments' is the
    one to take -- the unfiltered file still contains the reads the ENCODE
    pipeline rejected.
    """
    f = _json(f"{ENCODE}/files/{accession}/?format=json")
    if not f:
        return {"accession": accession, "feature": feature, "ok": False,
                "problems": ["no such file on ENCODE"]}
    rec = {
        "accession": accession, "filename": MARK_FILENAME[feature],
        "file_format": f.get("file_format"), "output_type": f.get("output_type"),
        "assembly": f.get("assembly"), "encode_status": f.get("status"),
        "md5": f.get("md5sum"), "bytes": f.get("file_size"),
        "url": f"{ENCODE}{f['href']}" if f.get("href") else None,
        "dataset": f.get("dataset"), "convert_from_bam": True,
    }
    exp = _json(f"{ENCODE}{rec['dataset']}?format=json") if rec["dataset"] else None
    if exp:
        rec.update({
            "experiment": exp.get("accession"),
            "assay_title": exp.get("assay_title"),
            "biosample": (exp.get("biosample_ontology") or {}).get("term_name"),
            "replication_type": exp.get("replication_type"),
            "lab": (exp.get("lab") or {}).get("title"),
        })
    problems = []
    if rec["file_format"] != "bam":
        problems.append(f"file_format is {rec['file_format']}, not bam")
    if rec["assembly"] != "GRCh38":
        problems.append(f"assembly is {rec['assembly']}, not GRCh38")
    if rec["encode_status"] != "released":
        problems.append(f"status is {rec['encode_status']}, not released")
    if rec.get("output_type") == "unfiltered alignments":
        problems.append("this is the UNFILTERED alignment; take the filtered "
                        "'alignments' file from the same experiment")
    if "ATAC" not in (rec.get("assay_title") or "").upper():
        problems.append(f"assay is {rec.get('assay_title')!r}, expected an ATAC assay")
    rec["problems"] = problems
    rec["ok"] = not problems
    return rec


_RESOLVED_FLAGS: list[str] | None = None


def resolve_bamcoverage_flags() -> list[str]:
    """Adapt the fixed command to the installed bamCoverage, or refuse.

    Resolved once and cached, so every tissue in a run gets the identical
    command -- and because the result is written into the manifest and the
    sidecar, a later tissue converted under a different deeptools build is
    visible rather than silently different.
    """
    global _RESOLVED_FLAGS
    if _RESOLVED_FLAGS is not None:
        return _RESOLVED_FLAGS

    import subprocess
    try:
        helptext = subprocess.run(["bamCoverage", "--help"], capture_output=True,
                                  text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"STOP: could not run 'bamCoverage --help' ({exc})") from exc

    out, notes = [], []
    i = 0
    while i < len(SNATAC_BAMCOVERAGE):
        tok = SNATAC_BAMCOVERAGE[i]
        if not tok.startswith("--"):
            out.append(tok)
            i += 1
            continue
        takes_value = (i + 1 < len(SNATAC_BAMCOVERAGE)
                       and not SNATAC_BAMCOVERAGE[i + 1].startswith("--"))
        if tok in helptext:
            out.append(tok)
            if takes_value:
                out.append(SNATAC_BAMCOVERAGE[i + 1])
        elif tok in FLAG_FALLBACKS and FLAG_FALLBACKS[tok][0] in helptext:
            sub = FLAG_FALLBACKS[tok]
            out.extend(sub)
            notes.append(f"{tok} -> {' '.join(sub)}")
        else:
            raise SystemExit(
                f"STOP: this bamCoverage does not accept {tok}, and no "
                f"documented equivalent is available.\n"
                f"Dropping it would change the output while the recorded "
                f"command still claimed the filter was applied.\n"
                f"Pin a deeptools version that supports it, or add an "
                f"equivalent to FLAG_FALLBACKS with a comment saying why it "
                f"is equivalent.")
        i += 2 if takes_value else 1

    if notes:
        print(f"      adapted to the installed deeptools: {'; '.join(notes)}")
    _RESOLVED_FLAGS = out
    return out


def convert_bam(bam: Path, dest: Path, blacklist: Path | None) -> dict:
    """BAM -> bigWig with the one fixed command, recorded alongside the output."""
    import subprocess
    cmd = list(resolve_bamcoverage_flags()) + ["--bam", str(bam),
                                               "--outFileName", str(dest)]
    if blacklist and blacklist.is_file():
        cmd += ["--blackListFileName", str(blacklist)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"      {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return {"command": " ".join(cmd), "path": str(dest),
            "bytes": dest.stat().st_size, "sha256": None}


def check_xena(project: str) -> dict:
    """Confirm the matrix exists and how big it is, without downloading it."""
    url = XENA.format(project=project)
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
    req.get_method = lambda: "HEAD"
    try:
        with urllib.request.urlopen(req, timeout=60) as fh:
            return {"project": project, "url": url, "available": True,
                    "bytes": int(fh.headers.get("Content-Length") or 0),
                    "etag": (fh.headers.get("ETag") or "").strip('"')}
    except urllib.error.HTTPError as exc:
        return {"project": project, "url": url, "available": False,
                "error": f"HTTP {exc.code}"}
    except urllib.error.URLError as exc:
        raise Unreachable(f"cannot reach the Xena GDC hub ({exc.reason})") from exc


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(url: str, dest: Path, expect_md5: str | None = None,
             retries: int = 4) -> dict:
    """Fetch one file, resuming a partial transfer rather than restarting it.

    This pulls tens of GB in one pass. A login node can kill a long-running
    process, a compute job can hit its walltime, and a network hiccup should not
    cost the whole file -- so an interrupted transfer leaves a .partial and the
    next run continues from that byte offset with an HTTP Range request. The md5
    is streamed over the resumed bytes as well as the new ones, so verification
    still covers the whole file.

    A server that ignores Range and replies 200 restarts cleanly rather than
    appending to a partial file and producing a corrupt one that happens to be
    the right length.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")

    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": UA["User-Agent"]}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as src:
                resuming = src.status == 206
                if have and not resuming:
                    tmp.unlink(missing_ok=True)   # server ignored Range
                    have = 0
                if have:
                    print(f"      resuming at {have / 1e6:.0f} MB")
                # Content-Length is what remains, not the whole file, on a 206.
                remaining = int(src.headers.get("Content-Length") or 0)
                total = have + remaining if remaining else 0
                mode = "ab" if have else "wb"
                done, started, last = have, time.time(), time.time()
                with tmp.open(mode) as out:
                    while True:
                        block = src.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        # A 20 GB BAM with no output is indistinguishable from a
                        # hang. Report every 15 s -- often enough to see life,
                        # rare enough not to flood a nohup log.
                        now = time.time()
                        if now - last >= 15:
                            rate = (done - have) / max(now - started, 1e-9) / 1e6
                            if total:
                                pct = 100 * done / total
                                eta = (total - done) / max(rate * 1e6, 1e-9)
                                print(f"      {done/1e9:.2f}/{total/1e9:.2f} GB "
                                      f"({pct:.1f}%)  {rate:.0f} MB/s  "
                                      f"eta {eta/60:.0f} min", flush=True)
                            else:
                                print(f"      {done/1e9:.2f} GB  {rate:.0f} MB/s",
                                      flush=True)
                            last = now
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise SystemExit(
                    f"STOP: {dest.name} failed after {retries} attempts ({exc}).\n"
                    f"The partial file was kept; re-run to resume from it.")
            wait = 5 * attempt
            print(f"      attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)

    h = hashlib.md5()
    with tmp.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    got = h.hexdigest()
    if expect_md5 and got != expect_md5:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"STOP: md5 mismatch for {dest.name}\n"
            f"  expected {expect_md5}\n  got      {got}\n"
            f"The partial file was removed. Do not use an unverified track.")
    tmp.replace(dest)
    return {"path": str(dest), "bytes": dest.stat().st_size, "md5": got,
            "verified": bool(expect_md5)}


def network_check() -> bool:
    """Can this host reach the two hosts we need? Prints what it finds."""
    ok = True
    for name, url in (("ENCODE", f"{ENCODE}/search/?type=File&limit=1&format=json"),
                      ("Xena GDC hub", XENA.format(project="TCGA-BRCA"))):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
            if "amazonaws" in url:
                req.get_method = lambda: "HEAD"
            with urllib.request.urlopen(req, timeout=30):
                print(f"  {name:<16} reachable")
        except Exception as exc:  # noqa: BLE001 - any failure means unusable
            print(f"  {name:<16} UNREACHABLE ({exc})")
            ok = False
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissues", nargs="+", default=list(TISSUES),
                    choices=list(TISSUES))
    ap.add_argument("--reference-root", type=Path, default=Path("data/reference"),
                    help="Per-tissue context goes to <root>/<Tissue>/. The "
                         "existing breast files at the root are never touched.")
    ap.add_argument("--targets-root", type=Path, default=Path("data/targets"))
    ap.add_argument("--plan", type=Path,
                    default=Path("data/multitissue_acquisition_plan.json"))
    ap.add_argument("--apply", action="store_true",
                    help="Download. Without this, only resolve and write the plan.")
    ap.add_argument("--context-only", action="store_true")
    ap.add_argument("--targets-only", action="store_true")
    ap.add_argument("--picks", type=Path, default=None,
                    help="A file of hand-chosen ENCODE accessions to verify and "
                         "use INSTEAD of automatic resolution. Lines are "
                         "'<tissue> <mark> <accession-or-url>'; see read_picks(). "
                         "Every file is checked for format, output type, "
                         "assembly, released status and target before use.")
    ap.add_argument("--keep-bams", action="store_true",
                    help="Keep the downloaded BAMs after conversion. They are "
                         "13-30 GB each and the bigWig is what the model reads, "
                         "so they are deleted by default.")
    ap.add_argument("--force", action="store_true",
                    help="Download even when the resolved set is inconsistent "
                         "across tissues. Only with a recorded reason.")
    ap.add_argument("--network-check", action="store_true",
                    help="Just test whether this host can reach ENCODE and the "
                         "Xena hub, then exit. Run this inside a compute job to "
                         "find out whether the download can be submitted there.")
    args = ap.parse_args(argv)

    if args.network_check:
        import socket
        print(f"host {socket.gethostname()}")
        return 0 if network_check() else 1

    picks = read_picks(args.picks) if args.picks else None
    if picks is not None:
        unknown = [t for t in picks if t not in TISSUES]
        if unknown:
            raise SystemExit(f"STOP: unknown tissue(s) in {args.picks}: {unknown}\n"
                             f"Known: {', '.join(TISSUES)}")
        args.tissues = [t for t in args.tissues if t in picks] or list(picks)

    plan: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection": "hand-picked accessions" if picks else "resolved automatically",
        "output_type": OUTPUT_TYPE,
        "assembly": "GRCh38",
        "note": ("bulk ATAC-seq for every tissue including breast, so that "
                 "accessibility is one assay across the joint model. The "
                 "published breast model uses a locally converted snATAC BAM "
                 "and is deliberately left alone."),
        "tissues": {},
    }

    try:
        for tissue in args.tissues:
            spec = TISSUES[tissue]
            print(f"\n{tissue}")
            entry: dict = {"note": spec["note"]}

            if not args.targets_only and picks is not None:
                chosen = picks.get(tissue, {})
                print(f"  context (your picks, {len(chosen)} file(s)):")
                files, bad = {}, []
                for feature, pick in chosen.items():
                    acc, as_bam = pick["accession"], pick["as_bam"]
                    rec = (verify_bam_pick(feature, acc) if as_bam
                           else verify_pick(feature, acc))
                    time.sleep(0.15)
                    mark = MARK_TARGET[feature] or "ATAC"
                    if rec["ok"]:
                        tag = " (BAM->bigWig)" if as_bam else ""
                        print(f"    {mark:<10}{acc:<14}{rec.get('biosample',''):<24}"
                              f"{(rec.get('bytes') or 0)/1e6:>7.0f} MB{tag}")
                        files[feature] = rec
                    else:
                        bad.append((mark, acc, rec["problems"], rec.get("experiment")))
                        print(f"    {mark:<10}{acc:<14}REJECTED")
                        for p in rec["problems"]:
                            print(f"      - {p}")
                        # A rejected file usually has an acceptable sibling in
                        # the same experiment -- an hg19 file almost always does,
                        # because ENCODE reprocessed against GRCh38 later. Say
                        # where to look rather than leaving a portal search.
                        if rec.get("experiment"):
                            print(f"      the same experiment {rec['experiment']} "
                                  f"({rec.get('biosample')}) may have a usable file:")
                            print(f"      {ENCODE}/experiments/{rec['experiment']}/"
                                  f" -> filter GRCh38, bigWig, "
                                  f"'{OUTPUT_TYPE}'")
                missing = [MARK_TARGET[f] or "ATAC" for f, _, _, _ in MARKS
                           if f not in files]
                if missing:
                    print(f"    !! missing {', '.join(missing)}")
                entry["context"] = {
                    "term": sorted({r.get("biosample") for r in files.values()
                                    if r.get("biosample")}) or None,
                    "files": files, "complete": not missing and not bad,
                    "missing": missing,
                    "histone_assays": sorted({r.get("assay_title") for r in files.values()
                                              if "ChIP" in (r.get("assay_title") or "")}),
                    "rejected": [{"mark": m, "accession": a, "problems": p,
                                  "experiment": e} for m, a, p, e in bad],
                }
                bios = entry["context"]["term"] or []
                if len(bios) > 1:
                    print(f"    !! more than one biosample in this tissue: {bios}")

            elif not args.targets_only:
                print("  context (ENCODE):")
                ctx = resolve_context(tissue, spec["encode"])
                entry["context"] = ctx
                if not ctx["complete"]:
                    print(f"    !! missing {', '.join(ctx['missing'])} "
                          f"for every candidate biosample")

            ctx = entry.get("context")
            if ctx and len(ctx.get("histone_assays") or []) > 1:
                print(f"    !! histone marks mix assays within this tissue: "
                      f"{ctx['histone_assays']} -- pick one before building")

            if not args.context_only:
                print("  targets (Xena GDC hub):")
                mats = [check_xena(p) for p in spec["projects"]]
                entry["targets"] = mats
                for m in mats:
                    size = f"{m['bytes'] / 1e9:.2f} GB" if m.get("bytes") else "?"
                    print(f"    {m['project']:<14}"
                          f"{'available' if m['available'] else m.get('error'):<14}{size}")

            plan["tissues"][tissue] = entry
    except Unreachable as exc:
        raise SystemExit(
            f"STOP: {exc}.\nCompute nodes have no outbound HTTPS -- run this on "
            f"a login node or your laptop.") from exc

    # Cross-tissue consistency. Checking each tissue on its own is not enough:
    # if breast resolves to Mint-ChIP and lung to Histone ChIP-seq, the context
    # differs between tissues by ASSAY rather than by biology, and a joint model
    # can learn that difference as tissue identity. Same for output type. This
    # is the failure the whole rebuild exists to avoid, so it is checked before
    # anything is downloaded.
    per_feature: dict[str, dict[str, set]] = {}
    for tissue, entry in plan["tissues"].items():
        for feature, f in entry.get("context", {}).get("files", {}).items():
            slot = per_feature.setdefault(feature, {"assay": set(), "output": set()})
            slot["assay"].add(f.get("assay_title"))
            slot["output"].add(f.get("output_type"))
    inconsistent = {f: s for f, s in per_feature.items()
                    if len(s["assay"]) > 1 or len(s["output"]) > 1}
    plan["cross_tissue_consistent"] = not inconsistent
    if inconsistent:
        print()
        print("!! CROSS-TISSUE INCONSISTENCY -- do not download yet")
        for feature, s in inconsistent.items():
            if len(s["assay"]) > 1:
                print(f"   {feature}: assays differ between tissues "
                      f"{sorted(a for a in s['assay'] if a)}")
            if len(s["output"]) > 1:
                print(f"   {feature}: output types differ "
                      f"{sorted(o for o in s['output'] if o)}")
        print("   A feature measured by a different assay in different tissues")
        print("   encodes the assay as well as the biology, and a joint model")
        print("   can use that to identify the tissue. Pin one assay per feature")
        print("   (edit HISTONE_ASSAYS, or drop a tissue) and re-resolve.")
    elif per_feature:
        assays = sorted({a for s in per_feature.values() for a in s["assay"] if a})
        print()
        print(f"Cross-tissue consistent: every feature uses the same assay and "
              f"output type in all tissues.")
        print(f"  assays in use: {assays}")

    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open("w") as fh:
        json.dump(plan, fh, indent=2, sort_keys=True)
        fh.write("\n")

    ctx_files = sum(len(t.get("context", {}).get("files", {}))
                    for t in plan["tissues"].values())
    ctx_bytes = sum(f.get("bytes") or 0 for t in plan["tissues"].values()
                    for f in t.get("context", {}).get("files", {}).values())
    tgt_bytes = sum(m.get("bytes") or 0 for t in plan["tissues"].values()
                    for m in t.get("targets", []))
    print()
    print("=" * 78)
    print(f"plan written to {args.plan}")
    print(f"  context  {ctx_files} bigWigs, {ctx_bytes / 1e9:.1f} GB")
    print(f"  targets  {sum(len(t.get('targets', [])) for t in plan['tissues'].values())} "
          f"matrices, {tgt_bytes / 1e9:.1f} GB")
    print("=" * 78)

    if args.apply and inconsistent and not args.force:
        raise SystemExit(
            "STOP: refusing to download a cross-tissue inconsistent set.\n"
            "Fix it, or pass --force if you have decided the difference is "
            "acceptable and recorded why.")

    if not args.apply:
        print("\nResolve-only. Read the plan, then re-run with --apply to download.")
        print("Every ENCODE file carries an md5 and is verified on arrival; a")
        print("mismatch aborts rather than leaving an unverified track in place.")
        return 0

    # The joint model's breast tracks and the published model's breast tracks
    # are both MCF-10A, carry the same filenames, and share six of seven
    # accessions -- only the locally converted ATAC differs. Writing either into
    # the other's directory would silently swap one model's context for the
    # other's, and nothing downstream would notice.
    for tissue in plan["tissues"]:
        target_dir = (args.reference_root / tissue).resolve()
        if target_dir == args.reference_root.resolve():
            raise SystemExit(
                f"STOP: {tissue} would be written into {args.reference_root} "
                f"itself.\nThat directory holds the PUBLISHED single-tissue "
                f"context and must not be overwritten.")

    # Check the conversion tool BEFORE transferring anything. bamCoverage was
    # previously checked at conversion time, which meant discovering it was
    # missing after a 19.6 GB download had already finished.
    needs_conversion = any(
        f.get("convert_from_bam")
        for t in plan["tissues"].values()
        for f in t.get("context", {}).get("files", {}).values())
    if needs_conversion:
        import shutil as _shutil
        if _shutil.which("bamCoverage") is None:
            raise SystemExit(
                "STOP: bamCoverage not found, and this plan needs it to convert "
                "snATAC BAMs.\nNothing has been downloaded.\n\n"
                "Install it in its OWN environment -- deeptools pins numpy, "
                "scipy and pysam,\nand installing it alongside the training "
                "environment can downgrade them:\n\n"
                "    conda create -n deeptools -c bioconda -c conda-forge "
                "deeptools -y\n    conda activate deeptools\n\n"
                "Then re-run this. Use the same environment for every tissue, "
                "or the\naccessibility feature stops being comparable between "
                "them.")

    print("\ndownloading ...")

    # One shared exclusion list, fetched once, used by every conversion.
    blacklist = None
    if needs_conversion:
        blacklist = args.reference_root / f"{BLACKLIST_ACCESSION}.bed.gz"
        if not blacklist.is_file():
            meta = _json(f"{ENCODE}/files/{BLACKLIST_ACCESSION}/?format=json")
            if meta and meta.get("href"):
                print(f"  exclusion list {BLACKLIST_ACCESSION} -> {blacklist}")
                download(f"{ENCODE}{meta['href']}", blacklist, meta.get("md5sum"))
            else:
                print(f"  WARNING: could not fetch {BLACKLIST_ACCESSION}; "
                      f"converting without an exclusion list")
                blacklist = None
        plan["snatac_conversion"] = {
            "command_template": " ".join(SNATAC_BAMCOVERAGE),
            "blacklist": str(blacklist) if blacklist else None,
            "note": ("applied identically to every tissue on this route, "
                     "including MCF-10A, so the accessibility feature is "
                     "comparable across tissues by construction"),
        }

    for tissue, entry in plan["tissues"].items():
        for feature, f in entry.get("context", {}).get("files", {}).items():
            dest = args.reference_root / tissue / f["filename"]

            # "It exists" is not "it is the right file". Lung/ATAC_seq.bw once
            # survived a switch from bulk ATAC to snATAC purely because the name
            # was unchanged, leaving the plan and the marker file claiming a
            # source the bytes did not come from. Check identity, not presence.
            if dest.exists():
                if f.get("convert_from_bam"):
                    side = dest.with_suffix(dest.suffix + ".source.json")
                    have = (json.loads(side.read_text()).get("accession")
                            if side.is_file() else None)
                    if have == f["accession"]:
                        print(f"  skip {dest} (converted from {have})")
                        continue
                    print(f"  REPLACING {dest}: converted from "
                          f"{have or 'an unrecorded source'}, want {f['accession']}")
                    dest.unlink()
                else:
                    got = hashlib.md5(dest.read_bytes()).hexdigest()
                    if f.get("md5") and got == f["md5"]:
                        print(f"  skip {dest} (md5 matches {f['accession']})")
                        continue
                    print(f"  REPLACING {dest}: md5 {got[:10]} does not match "
                          f"{f['accession']} ({(f.get('md5') or '?')[:10]})")
                    dest.unlink()

            if f.get("convert_from_bam"):
                bam = args.reference_root / tissue / f"{f['accession']}.bam"
                if not bam.exists():
                    print(f"  {f['accession']}  ->  {bam}  "
                          f"({(f.get('bytes') or 0)/1e9:.1f} GB)")
                    f["downloaded"] = download(f["url"], bam, f.get("md5"))
                print(f"  converting {bam.name} -> {dest.name}")
                f["conversion"] = convert_bam(bam, dest, blacklist)
                # A converted track has no upstream md5 to check it against, so
                # record what it came from next to it. This sidecar is what the
                # skip check above reads on a later run.
                dest.with_suffix(dest.suffix + ".source.json").write_text(
                    json.dumps({"accession": f["accession"],
                                "experiment": f.get("experiment"),
                                "assay_title": f.get("assay_title"),
                                "biosample": f.get("biosample"),
                                "command": f["conversion"]["command"],
                                "converted_utc": datetime.now(timezone.utc)
                                .isoformat(timespec="seconds")},
                               indent=2, sort_keys=True) + "\n")
                if not args.keep_bams:
                    bam.unlink(missing_ok=True)
                    f["conversion"]["bam_removed"] = True
                continue

            print(f"  {f['accession']}  ->  {dest}")
            f["downloaded"] = download(f["url"], dest, f.get("md5"))
        for m in entry.get("targets", []):
            if not m.get("available"):
                continue
            dest = args.targets_root / f"{m['project']}.methylation450.tsv.gz"
            if dest.exists():
                print(f"  skip {dest} (exists)")
                continue
            print(f"  {m['project']}  ->  {dest}")
            m["downloaded"] = download(m["url"], dest, None)

    # A marker in each directory, so the two breast track sets can be told
    # apart by looking rather than by remembering.
    for tissue, entry in plan["tissues"].items():
        files = entry.get("context", {}).get("files", {})
        if not files:
            continue
        d = args.reference_root / tissue
        lines = [
            f"# {tissue} context tracks -- JOINT MULTI-TISSUE MODEL",
            "",
            "These belong to the multi-tissue model ONLY.",
            "",
            f"The published single-tissue model reads {args.reference_root}/*.bw",
            "directly -- the files one level up from here. Those are a different",
            "track set and must not be swapped with these, even for breast,",
            "where both are MCF-10A and the filenames are identical.",
            "",
            f"generated {plan['generated_utc']}",
            "",
            "| feature | accession | biosample | source |",
            "|---|---|---|---|",
        ]
        for feature, f in files.items():
            how = ("BAM converted locally" if f.get("convert_from_bam")
                   else f.get("output_type", "?"))
            lines.append(f"| {MARK_TARGET[feature] or 'ATAC'} | "
                         f"{f.get('accession')} | {f.get('biosample')} | {how} |")
        if plan.get("snatac_conversion"):
            lines += ["", "Accessibility conversion, identical for every tissue:",
                      "", "```", plan["snatac_conversion"]["command_template"], "```"]
        lines += ["", "PhyloP is genome conservation, identical in every tissue,",
                  f"and is read from {args.reference_root}/hg38.phyloP100way.bw.", ""]
        (d / "TRACK_SET.md").write_text("\n".join(lines))

    with args.plan.open("w") as fh:
        json.dump(plan, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\ndone; {args.plan} updated with what was written")
    print("each tissue directory carries a TRACK_SET.md naming which model it "
          "belongs to")
    print("Next: data/audit_reference_tracks.py --write-manifest on each new")
    print("tissue directory, so the manifest records these the same way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
