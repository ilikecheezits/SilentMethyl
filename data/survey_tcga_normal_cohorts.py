#!/usr/bin/env python3
"""Which tissues can actually support a joint multi-tissue SilentMethyl?

Why this exists
---------------
The multi-tissue plan is to reuse the BRCA recipe -- TCGA solid-tissue-normal
HM450 targets plus a matched ENCODE reference context -- for a handful of other
tissues, each chosen so that an eGTEx mQTL cohort we already score against also
exists for it. A tissue is only usable if BOTH halves are there, and either half
can be missing:

  targets   TCGA must have banked enough adjacent normal tissue AND run HM450
            on it. Counts differ by an order of magnitude between projects --
            some have hundreds of normal donors, TCGA-OV has almost none.

  context   ENCODE must have all seven tissue-specific tracks the epigenomic
            tower consumes. Six histone marks plus ATAC is a demanding set for
            primary tissue; ATAC in particular is patchy outside cell lines,
            because older primary-tissue data used DNase instead.

Guessing either half is expensive: a tissue that turns out to have 12 normal
donors, or five of seven marks, is a wasted track assembly and a wasted training
run. Both numbers also move as GDC and ENCODE release data, so neither belongs
in a comment. This asks both APIs directly.

Only open-access metadata is queried. No controlled-access request is made and
nothing is downloaded; the output is a table of counts.

Two things this script is careful about
---------------------------------------
1. It counts PATIENTS, not files. The GDC facet on /files returns a file count,
   and a donor can contribute several methylation files, so faceting would
   overstate every cohort. This pages through the matching files and counts
   distinct case ids, verifying each record's own sample type rather than
   trusting the filter's case-level join.

2. It requires the ENCODE experiment to have released GRCh38 signal, not merely
   to exist. An experiment in progress, or one only processed against hg19, is
   not a track you can put in a feature vector.

Usage (run from the repository root, on a machine with outbound HTTPS)
----------------------------------------------------------------------
    python -u data/survey_tcga_normal_cohorts.py
    python -u data/survey_tcga_normal_cohorts.py --min-normals 30
    python -u data/survey_tcga_normal_cohorts.py --skip-encode
    python -u data/survey_tcga_normal_cohorts.py --out data/tcga_normal_cohort_survey.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GDC_FILES = "https://api.gdc.cancer.gov/files"
ENCODE_SEARCH = "https://www.encodeproject.org/search/"
UA = {"Accept": "application/json", "User-Agent": "SilentMethyl-cohort-survey"}

# The seven tissue-specific context features. The remaining two inputs to the
# epigenomic tower are PhyloP 100-way conservation, which is a property of the
# genome rather than of the tissue and needs no per-tissue track.
# Mirrors TABULAR_FEATURES in scripts/training_common.py.
REQUIRED_TRACKS = [
    ("ATAC-seq", None),
    ("Histone ChIP-seq", "H3K4me3"),
    ("Histone ChIP-seq", "H3K27ac"),
    ("Histone ChIP-seq", "H3K27me3"),
    ("Histone ChIP-seq", "H3K9me3"),
    ("Histone ChIP-seq", "H3K36me3"),
    ("Histone ChIP-seq", "H3K4me1"),
]

# Candidate TCGA projects and ENCODE biosample terms per eGTEx tissue. A tissue
# with no plausible project is listed with an empty tuple so the report says so
# explicitly rather than quietly omitting it. Several ENCODE terms are tried per
# tissue because primary-tissue naming is inconsistent, and the best-covered one
# is reported.
#
# These are CANDIDATES. Attaching a real number to each is the whole point.
TISSUES = {
    "BreastMammaryTissue": {
        "tcga": ("TCGA-BRCA",),
        "encode": ["MCF-10A", "breast epithelium", "luminal epithelial cell of mammary gland"],
    },
    "ColonTransverse": {
        "tcga": ("TCGA-COAD", "TCGA-READ"),
        "encode": ["transverse colon", "sigmoid colon", "large intestine"],
    },
    "KidneyCortex": {
        "tcga": ("TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"),
        "encode": ["kidney", "kidney epithelial cell", "renal cortex interstitium"],
    },
    "Lung": {
        "tcga": ("TCGA-LUAD", "TCGA-LUSC"),
        "encode": ["lung", "upper lobe of left lung", "IMR-90"],
    },
    "Prostate": {
        "tcga": ("TCGA-PRAD",),
        "encode": ["prostate gland", "prostate", "epithelial cell of prostate"],
    },
    "Ovary": {
        "tcga": ("TCGA-OV",),
        "encode": ["ovary"],
    },
    "Testis": {
        "tcga": ("TCGA-TGCT",),
        "encode": ["testis"],
    },
    "MuscleSkeletal": {
        "tcga": (),
        "encode": ["gastrocnemius medialis", "muscle of leg", "skeletal muscle tissue"],
    },
    "WholeBlood": {
        "tcga": (),
        "encode": ["peripheral blood mononuclear cell", "common myeloid progenitor, CD34-positive"],
    },
}


def _get_json(url: str, timeout: int, what: str):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.load(fh)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"STOP: could not reach {what} ({exc}).\n"
            f"Compute nodes and some login nodes have no outbound HTTPS. Run this\n"
            f"from a machine with network access, or read the same facets by hand\n"
            f"at https://portal.gdc.cancer.gov/ and https://www.encodeproject.org/."
        ) from exc


# ---------------------------------------------------------------------------
# GDC: distinct donors with solid-tissue-normal HM450
# ---------------------------------------------------------------------------

def gdc_normal_donors(platform: str, sample_type: str, timeout: int = 90,
                      page: int = 1000) -> dict[str, set]:
    """Distinct case ids per project, counted from the files themselves.

    Faceting /files by project returns a FILE count. A donor can contribute more
    than one methylation file, so the facet overstates donor counts -- which is
    the number that decides whether a tissue is trainable. Page through instead
    and collect case ids, checking each record's own sample type: the GDC's
    filter join is at case level, so a filter alone can admit a file from a
    tumour sample belonging to a donor who also has a normal sample.
    """
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "data_type",
                                     "value": ["Methylation Beta Value"]}},
            {"op": "in", "content": {"field": "platform", "value": [platform]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type",
                                     "value": [sample_type]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    fields = ",".join([
        "cases.case_id",
        "cases.project.project_id",
        "cases.samples.sample_type",
    ])

    donors: dict[str, set] = {}
    frm, total = 0, None
    while True:
        query = urllib.parse.urlencode({
            "filters": json.dumps(filters), "fields": fields,
            "size": str(page), "from": str(frm), "format": "json",
        })
        payload = _get_json(f"{GDC_FILES}?{query}", timeout, "the GDC API")
        data = payload.get("data", {})
        hits = data.get("hits", [])
        if total is None:
            total = int(data.get("pagination", {}).get("total", len(hits)))
            print(f"  GDC: {total:,} open {platform} '{sample_type}' files to walk")
        for hit in hits:
            for case in hit.get("cases", []):
                types = {s.get("sample_type") for s in case.get("samples", [])}
                if sample_type not in types:
                    continue          # case-level join artefact, not our file
                proj = case.get("project", {}).get("project_id")
                cid = case.get("case_id")
                if proj and cid:
                    donors.setdefault(proj, set()).add(cid)
        frm += len(hits)
        if not hits or frm >= total:
            break
        time.sleep(0.2)
    return donors


# ---------------------------------------------------------------------------
# ENCODE: are all seven tracks available for this biosample?
# ---------------------------------------------------------------------------

def encode_has_track(term: str, assay: str, target: str | None,
                     timeout: int = 60) -> int:
    """Released GRCh38 experiments for one biosample/assay pair."""
    params = [
        ("type", "Experiment"),
        ("status", "released"),
        ("assembly", "GRCh38"),
        ("biosample_ontology.term_name", term),
        ("assay_title", assay),
        ("format", "json"),
        ("limit", "0"),
    ]
    if target:
        params.append(("target.label", target))
    url = f"{ENCODE_SEARCH}?{urllib.parse.urlencode(params)}"
    try:
        payload = _get_json(url, timeout, "the ENCODE portal")
    except SystemExit:
        raise
    return int(payload.get("total", 0))


def encode_coverage(terms: list[str], timeout: int = 60) -> dict:
    """Best-covered biosample term for a tissue, and what it is missing."""
    best = {"term": None, "have": [], "missing": [t for _, t in REQUIRED_TRACKS]}
    for term in terms:
        have, missing = [], []
        for assay, target in REQUIRED_TRACKS:
            label = target or assay
            if encode_has_track(term, assay, target, timeout) > 0:
                have.append(label)
            else:
                missing.append(label)
            time.sleep(0.1)
        if len(have) > len(best["have"]):
            best = {"term": term, "have": have, "missing": missing}
        if not missing:
            break
    best["n_have"] = len(best["have"])
    best["complete"] = best["n_have"] == len(REQUIRED_TRACKS)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="Illumina Human Methylation 450",
                    help="HM450 by default. Mixing in the 27k platform changes "
                         "the probe set and is not recommended.")
    ap.add_argument("--min-normals", type=int, default=30,
                    help="Donors needed to call a tissue usable (default 30). "
                         "BRCA has ~97 and the published model is built on it.")
    ap.add_argument("--skip-encode", action="store_true",
                    help="GDC only. The ENCODE half is ~60 queries and slower.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    print("querying GDC ...")
    donors = gdc_normal_donors(args.platform, "Solid Tissue Normal")

    rows = []
    for tissue, spec in TISSUES.items():
        projects = spec["tcga"]
        ids: set = set()
        for p in projects:
            ids |= donors.get(p, set())
        n = len(ids)

        enc = None
        if not args.skip_encode and spec["encode"]:
            print(f"querying ENCODE for {tissue} ...")
            enc = encode_coverage(spec["encode"])

        targets_ok = bool(projects) and n >= args.min_normals
        context_ok = None if enc is None else enc["complete"]
        reasons = []
        if not projects:
            reasons.append("no TCGA project for this tissue")
        elif n < args.min_normals:
            reasons.append(f"only {n} normal donors (< {args.min_normals})")
        if enc is not None and not enc["complete"]:
            reasons.append("ENCODE missing " + ", ".join(enc["missing"]))

        rows.append({
            "egtex_tissue": tissue,
            "tcga_projects": list(projects),
            "normal_donors": n,
            "targets_ok": targets_ok,
            "encode_term": None if enc is None else enc["term"],
            "encode_tracks_present": None if enc is None else enc["n_have"],
            "encode_missing": None if enc is None else enc["missing"],
            "context_ok": context_ok,
            "usable": targets_ok and (context_ok is not False),
            "reasons": reasons,
        })
    rows.sort(key=lambda r: (-r["normal_donors"], r["egtex_tissue"]))

    ntr = len(REQUIRED_TRACKS)
    print()
    print("=" * 92)
    print(f"JOINT MULTI-TISSUE FEASIBILITY   targets: TCGA {args.platform}")
    print(f"{'':33}context: ENCODE, {ntr} tissue-specific tracks")
    print("=" * 92)
    print(f"{'eGTEx tissue':<22}{'TCGA':<20}{'donors':>8}  {'ENCODE biosample':<32}{'tracks':>7}")
    for r in rows:
        proj = ",".join(p.replace("TCGA-", "") for p in r["tcga_projects"]) or "--"
        term = r["encode_term"] or ("(not queried)" if args.skip_encode else "none found")
        trk = "--" if r["encode_tracks_present"] is None \
              else f"{r['encode_tracks_present']}/{ntr}"
        print(f"{r['egtex_tissue']:<22}{proj:<20}{r['normal_donors']:>8,}"
              f"  {term:<32}{trk:>7}")
        for why in r["reasons"]:
            print(f"{'':<22}  - {why}")

    usable = [r for r in rows if r["usable"]]
    print()
    print(f"usable: {len(usable)}  ({', '.join(r['egtex_tissue'] for r in usable) or 'none'})")
    print(f"dropped: {len(rows) - len(usable)}  "
          f"({', '.join(r['egtex_tissue'] for r in rows if not r['usable'])})")

    # Training rows are probes x tissues, and the stated ceiling is one cohort's
    # worth -- 418,486, the size of the breast probe set -- because that is what
    # keeps a run near the ~46 h the single-tissue model already takes.
    BUDGET = 418_486
    if usable:
        per = BUDGET // len(usable)
        print()
        print(f"At {BUDGET:,} training rows over {len(usable)} tissues: "
              f"{per:,} probes each ({100 * per / BUDGET:.0f}% of the breast set).")
        print("Subsample TRAINING only, by cross-tissue variance. Keep validation")
        print("and test at the full probe set, or the numbers stop being")
        print("comparable to the published single-tissue model.")
        print("Hold out the SAME chromosomes in every tissue, or a probe held out")
        print("in one tissue re-enters training through another.")
        donors_by = sorted(usable, key=lambda r: -r["normal_donors"])
        if donors_by[0]["normal_donors"] > 3 * donors_by[-1]["normal_donors"]:
            print()
            print(f"NOTE: donor counts are unbalanced "
                  f"({donors_by[0]['egtex_tissue']} {donors_by[0]['normal_donors']} vs "
                  f"{donors_by[-1]['egtex_tissue']} {donors_by[-1]['normal_donors']}). "
                  f"Target medians\nfrom a small cohort are noisier, so an equal probe "
                  f"split gives the noisier tissues\nequal weight. Consider capping or "
                  f"weighting per tissue rather than splitting evenly.")

    dropped_with_mqtl = [r["egtex_tissue"] for r in rows if not r["usable"]]
    if dropped_with_mqtl:
        print()
        print("The dropped tissues still have eGTEx mQTLs, so they are a genuine")
        print("unseen-tissue benchmark for the joint model -- held out because the")
        print("data does not exist, not because we chose them:")
        print("  " + ", ".join(dropped_with_mqtl))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            json.dump({
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "platform": args.platform,
                "min_normals": args.min_normals,
                "required_tracks": [t or a for a, t in REQUIRED_TRACKS],
                "sources": {
                    "targets": "GDC /files, open access, distinct case_id per project",
                    "context": "ENCODE /search, released Experiments with GRCh38 assembly",
                },
                "per_tissue": rows,
            }, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
