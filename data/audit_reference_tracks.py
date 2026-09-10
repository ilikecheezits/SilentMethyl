#!/usr/bin/env python3
"""Fingerprint the seven context bigWigs, because nothing records where they came from.

The problem
-----------
data/reference/ holds the seven tracks the entire epigenomic tower consumes:

    ATAC_seq.bw  H3K4me3.bw  H3K27ac.bw  H3K27me3.bw
    H3K9me3.bw   H3K36me3.bw H3K4me1.bw

Nothing in the repository says where any of them came from. build_data.sh does
not fetch them and does not list them among its required inputs;
build_training_data.py opens them by hard-coded filename; and
data/external/external_manifest.json -- which records a URL, byte count, sha256
and file head for something as peripheral as the ClinVar VCF -- has no entry for
them at all. There is also a hg19ToHg38.over.chain.gz sitting in the same
directory, which suggests at least some of these were lifted rather than
downloaded against GRCh38, but that is an inference from a neighbouring file
rather than a record.

Three things follow, in increasing order of cost:

  * A reviewer asking "which ENCODE experiment is your ATAC track?" has no
    answer in the repository.
  * The multi-tissue plan is "repeat the breast recipe for four more tissues".
    That is not repeatable while the breast recipe is undocumented.
  * A survey of ENCODE for MCF-10A came back short of all seven marks as
    released GRCh38 experiments, so these files are probably NOT a clean
    ENCODE/GRCh38 set -- which is a methods detail, not a filing detail.

What this does
--------------
Records everything recoverable from the files themselves, and states plainly
what only a human can supply.

The assembly test is the useful part: a bigWig header carries its chromosome
sizes, and chr1 is 249,250,621 bases in hg19 and 248,956,422 in hg38. That
distinguishes a natively-hg38 track from a lifted one without any external
lookup, and a lifted track has coverage gaps that a native one does not -- which
is worth knowing before four more tissues are built the same way.

This writes into the existing manifest rather than a new file, under a
"reference_tracks" key, so there is one provenance record and not two.

Usage (run from the repository root)
------------------------------------
    python -u data/audit_reference_tracks.py
    python -u data/audit_reference_tracks.py --write-manifest
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
UA = {"Accept": "application/json", "User-Agent": "SilentMethyl-track-audit"}

try:
    import numpy as np
    import pyBigWig
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"STOP: {exc}. Run inside the silentmethyl environment, which already "
        f"has pyBigWig -- data/build_training_data.py imports it."
    ) from exc

# Mirrors the mapping in data/build_training_data.py. Kept as a literal so this
# script can be run without importing the builder and its heavy dependencies.
TRACKS = {
    "Ref_ATAC_Signal":      "ATAC_seq.bw",
    "Ref_H3K4me3_Signal":   "H3K4me3.bw",
    "Ref_H3K27ac_Signal":   "H3K27ac.bw",
    "Ref_H3K27me3_Signal":  "H3K27me3.bw",
    "Ref_H3K9me3_Signal":   "H3K9me3.bw",
    "Ref_H3K36me3_Signal":  "H3K36me3.bw",
    "Ref_H3K4me1_Signal":   "H3K4me1.bw",
    "Target_Base_PhyloP_100way": "hg38.phyloP100way.bw",
}

# chr1 length is the cheapest unambiguous assembly discriminator.
CHR1 = {249_250_621: "hg19/GRCh37", 248_956_422: "hg38/GRCh38",
        247_249_719: "hg18/NCBI36"}

# A gene-dense hg38 window used to sanity-check that a track carries signal at
# all. Chosen inside a housekeeping locus so every mark should be non-empty,
# though the repressive marks will be low.
PROBE_REGION = ("chr1", 1_000_000, 1_100_000)


def digests(path: Path, chunk: int = 1 << 20) -> tuple[str, str]:
    """sha256 for our own records, md5 because that is ENCODE's file key."""
    s, m = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            s.update(block)
            m.update(block)
    return s.hexdigest(), m.hexdigest()


def _encode_json(path_or_query: str, timeout: int = 60):
    """GET JSON from the ENCODE portal. None when the search matched nothing.

    The portal answers an empty search with HTTP 404 rather than a 200 carrying
    zero results, so a 404 is an answer and not a transport failure.
    """
    req = urllib.request.Request(f"{ENCODE}{path_or_query}", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.load(fh)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"ENCODE returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach ENCODE ({exc.reason})") from exc


def encode_lookup(md5: str, timeout: int = 60) -> dict | None:
    """Recover a track's provenance from its checksum.

    These files were pulled by hand from the ENCODE portal and renamed
    (ENCFF*.bigWig -> H3K27ac.bw), which loses the accession from the filename
    but not from the bytes. ENCODE indexes every file it serves by md5, so the
    accession, experiment, biosample, assay and processing type can all be
    recovered exactly -- no guessing from filenames, no trusting memory.

    A checksum that does not match anything is itself informative: it means the
    file is not the pristine download, so it was lifted, re-sorted, subset or
    otherwise processed after download, and that processing is undocumented.
    """
    query = urllib.parse.urlencode({"type": "File", "md5sum": md5,
                                    "format": "json", "limit": "1"})
    hit = _encode_json(f"/search/?{query}", timeout)
    if not hit or not hit.get("@graph"):
        return None
    f = hit["@graph"][0]

    out = {
        "accession": f.get("accession"),
        "file_format": f.get("file_format"),
        "output_type": f.get("output_type"),
        "assembly": f.get("assembly"),
        "encode_status": f.get("status"),
        "dataset": f.get("dataset"),
        "file_url": f"{ENCODE}{f['href']}" if f.get("href") else None,
    }

    # The File object names its experiment but not the biology; fetch that too,
    # because "which biosample and which assay" is the part the multi-tissue
    # build has to reproduce for four more tissues.
    if out["dataset"]:
        exp = _encode_json(f"{out['dataset']}?format=json", timeout)
        if exp:
            out["experiment"] = exp.get("accession")
            out["assay_title"] = exp.get("assay_title")
            out["biosample"] = (exp.get("biosample_ontology") or {}).get("term_name")
            target = exp.get("target") or {}
            out["target"] = target.get("label")
            out["lab"] = (exp.get("lab") or {}).get("title")
    return out


def fingerprint(path: Path) -> dict:
    out: dict = {
        "path": str(path),
        "present": path.is_file(),
    }
    if not out["present"]:
        out["note"] = "MISSING"
        return out

    stat = path.stat()
    out["bytes"] = stat.st_size
    out["mtime_utc"] = datetime.fromtimestamp(
        stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    out["sha256"], out["md5"] = digests(path)

    bw = pyBigWig.open(str(path))
    try:
        chroms = bw.chroms()
        out["n_chroms"] = len(chroms)
        chr1 = chroms.get("chr1") or chroms.get("1")
        out["chr1_length"] = chr1
        out["assembly_inferred"] = CHR1.get(chr1, "unrecognised")
        out["chrom_naming"] = ("chr-prefixed" if any(c.startswith("chr")
                                                     for c in chroms)
                               else "ensembl-style (no chr prefix)")
        out["has_alt_contigs"] = any(("_alt" in c or "_random" in c
                                      or c.startswith("chrUn")) for c in chroms)
        hdr = bw.header() or {}
        out["header"] = {k: hdr.get(k) for k in
                         ("version", "nLevels", "nBasesCovered",
                          "minVal", "maxVal", "sumData")}
        if chr1 and hdr.get("nBasesCovered"):
            total = sum(chroms.values())
            out["fraction_of_genome_covered"] = round(
                hdr["nBasesCovered"] / total, 4)

        c, s, e = PROBE_REGION
        if c in chroms and chroms[c] > e:
            vals = np.asarray(bw.values(c, s, e), dtype=float)
            finite = vals[np.isfinite(vals)]
            out["probe_region"] = {
                "region": f"{c}:{s:,}-{e:,}",
                "fraction_defined": round(float(finite.size / vals.size), 4),
                "mean": None if finite.size == 0 else round(float(finite.mean()), 4),
                "max": None if finite.size == 0 else round(float(finite.max()), 4),
            }
    finally:
        bw.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", type=Path, default=Path("data/reference"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/external/external_manifest.json"))
    ap.add_argument("--write-manifest", action="store_true",
                    help="Merge the fingerprints into the manifest under "
                         "'reference_tracks'. Without this the script only reports.")
    ap.add_argument("--no-encode", action="store_true",
                    help="Skip the md5 lookup against the ENCODE portal.")
    args = ap.parse_args(argv)

    records = {feature: fingerprint(args.reference_dir / name)
               for feature, name in TRACKS.items()}

    # Recover provenance from the checksums. These were downloaded by hand from
    # the ENCODE portal and renamed, so the accession is gone from the filename
    # but not from the bytes.
    encode_error = None
    if not args.no_encode:
        print("matching checksums against the ENCODE portal ...")
        for feature, r in records.items():
            if not r.get("md5") or encode_error:
                continue
            try:
                r["encode"] = encode_lookup(r["md5"])
            except RuntimeError as exc:
                encode_error = str(exc)
                print(f"  lookup unavailable ({encode_error}); "
                      f"reporting fingerprints only")
            time.sleep(0.2)

    print()
    print("=" * 96)
    print("REFERENCE TRACK AUDIT   data/reference/")
    print("=" * 96)
    print(f"{'feature':<28}{'file':<24}{'assembly':<16}{'covered':>9}{'sha256':>12}")
    for feature, r in records.items():
        name = Path(r["path"]).name
        if not r["present"]:
            print(f"{feature:<28}{name:<24}{'MISSING':<16}")
            continue
        cov = r.get("fraction_of_genome_covered")
        print(f"{feature:<28}{name:<24}{r['assembly_inferred']:<16}"
              f"{'--' if cov is None else f'{100*cov:.1f}%':>9}"
              f"{r['sha256'][:10]:>12}")

    present = [r for r in records.values() if r["present"]]
    assemblies = {r["assembly_inferred"] for r in present}
    print()
    if len(assemblies) > 1:
        print(f"!! MIXED ASSEMBLIES across the context tracks: {sorted(assemblies)}")
        print("   Every track is read at the same hg38 coordinate in")
        print("   build_training_data.py. If any of these is not hg38, that")
        print("   feature has been sampled at the wrong locus for every probe.")
    elif assemblies == {"hg38/GRCh38"}:
        print("All context tracks report hg38 chromosome sizes.")
        print("That is consistent with either a native GRCh38 download or a")
        print("completed liftOver -- it does not by itself distinguish them.")
        covs = [r.get("fraction_of_genome_covered") for r in present
                if r.get("fraction_of_genome_covered") is not None]
        if covs and min(covs) < 0.5 * max(covs):
            print()
            print("However, genome coverage varies widely between tracks "
                  f"({100*min(covs):.1f}% to {100*max(covs):.1f}%).")
            print("Lifted tracks lose intervals that fail to map, so a track")
            print("covering much less of the genome than its siblings is the")
            print("signature of a liftOver rather than a native download.")
    else:
        print(f"!! Unexpected assemblies: {sorted(assemblies)}")

    if not args.no_encode and not encode_error:
        matched = {f: r for f, r in records.items() if r.get("encode")}
        unmatched = [f for f, r in records.items()
                     if r["present"] and not r.get("encode")]
        print()
        print("-" * 96)
        print("PROVENANCE RECOVERED FROM CHECKSUMS")
        print("-" * 96)
        if matched:
            print(f"{'feature':<24}{'accession':<14}{'biosample':<22}"
                  f"{'output type':<26}{'asm':<8}")
            for feature, r in matched.items():
                e = r["encode"]
                print(f"{feature:<24}{e.get('accession') or '?':<14}"
                      f"{(e.get('biosample') or '?'):<22}"
                      f"{(e.get('output_type') or '?'):<26}"
                      f"{(e.get('assembly') or '?'):<8}")
            biosamples = {r["encode"].get("biosample") for r in matched.values()
                          if r["encode"].get("biosample")}
            outputs = {r["encode"].get("output_type") for r in matched.values()
                       if r["encode"].get("output_type")}
            print()
            if len(biosamples) > 1:
                print(f"!! Tracks come from MORE THAN ONE biosample: "
                      f"{sorted(biosamples)}")
                print("   The context vector is supposed to describe one tissue.")
            if len(outputs) > 1:
                print(f"!! Mixed processing types: {sorted(outputs)}")
                print("   'fold change over control' and 'signal p-value' are on")
                print("   different scales; mixing them makes the feature vector")
                print("   inconsistent between marks.")
            if len(biosamples) == 1 and len(outputs) == 1:
                print(f"Consistent: one biosample ({next(iter(biosamples))}), "
                      f"one processing type ({next(iter(outputs))}).")
                print("That is the recipe to reproduce for the other tissues.")
        if unmatched:
            print()
            print(f"NO ENCODE MATCH for {len(unmatched)} track(s): "
                  f"{', '.join(unmatched)}")
            print("The checksum does not correspond to any file ENCODE serves, so")
            print("these are not pristine downloads -- something was done to them")
            print("after download (liftOver, re-sort, subset) and that step is")
            print("undocumented. phyloP is expected here; it comes from UCSC.")
    else:
        print()
        print("What the fingerprints alone CANNOT recover, and a human must supply:")
        print("  - the ENCODE accession and biosample for each track")
        print("  - whether liftOver was applied, and with which chain file")
        print("  - the ENCODE output type (fold change over control? signal")
        print("    p-value?), which decides whether four more tissues can match")
        print()
        print("Re-run with network to the ENCODE portal and these are recovered")
        print("automatically from the md5 checksums.")

    if args.write_manifest:
        if not args.manifest.is_file():
            raise SystemExit(f"STOP: {args.manifest} not found")
        with args.manifest.open() as fh:
            manifest = json.load(fh)

        block = manifest.setdefault("reference_tracks", {})
        for feature, r in records.items():
            prior = block.get(feature, {})
            entry = dict(r)
            enc = r.get("encode") or {}
            # Prefer what the checksum proved; fall back to anything a human
            # already filled in; never overwrite a real value with a blank.
            for key, from_encode in (("accession", "accession"),
                                     ("biosample", "biosample"),
                                     ("url", "file_url"),
                                     ("encode_file_type", "output_type"),
                                     ("encode_experiment", "experiment"),
                                     ("encode_assembly", "assembly")):
                entry[key] = enc.get(from_encode) or prior.get(key)
            entry["source"] = ("encode" if enc.get("accession")
                               else prior.get("source"))
            entry["liftover_chain"] = prior.get("liftover_chain")
            entry["fingerprinted_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            entry["provenance_status"] = (
                "recovered from md5 against the ENCODE portal"
                if enc.get("accession") else
                "recorded by hand" if entry.get("source") else
                "UNKNOWN -- no ENCODE match; file is not a pristine download")
            block[feature] = entry

        with args.manifest.open("w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        unknown = sum(1 for e in block.values()
                      if e.get("provenance_status", "").startswith("UNKNOWN"))
        print(f"\nwrote {args.manifest}  "
              f"({len(block)} tracks, {unknown} still without a recorded source)")
    else:
        print("\n(report only -- pass --write-manifest to record this in "
              f"{args.manifest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
