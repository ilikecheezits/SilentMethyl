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

def download(url: str, dest: Path, expect_md5: str | None = None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    h = hashlib.md5()
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=300) as src, tmp.open("wb") as out:
        while True:
            block = src.read(1 << 20)
            if not block:
                break
            out.write(block)
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
    args = ap.parse_args(argv)

    plan: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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

            if not args.targets_only:
                print("  context (ENCODE):")
                ctx = resolve_context(tissue, spec["encode"])
                entry["context"] = ctx
                if not ctx["complete"]:
                    print(f"    !! missing {', '.join(ctx['missing'])} "
                          f"for every candidate biosample")
                if len(ctx["histone_assays"]) > 1:
                    print(f"    !! histone marks mix assays: "
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

    if not args.apply:
        print("\nResolve-only. Read the plan, then re-run with --apply to download.")
        print("Every ENCODE file carries an md5 and is verified on arrival; a")
        print("mismatch aborts rather than leaving an unverified track in place.")
        return 0

    print("\ndownloading ...")
    for tissue, entry in plan["tissues"].items():
        for feature, f in entry.get("context", {}).get("files", {}).items():
            dest = args.reference_root / tissue / f["filename"]
            if dest.exists():
                print(f"  skip {dest} (exists)")
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

    with args.plan.open("w") as fh:
        json.dump(plan, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\ndone; {args.plan} updated with what was written")
    print("Next: data/audit_reference_tracks.py --write-manifest on each new")
    print("tissue directory, so the manifest records these the same way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
