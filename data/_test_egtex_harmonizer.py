#!/usr/bin/env python3
"""Synthetic end-to-end test for data/harmonize_egtex_mqtl.py.

Builds a fake repository root with a fake hg38, a fake HM450 manifest, fake
split CSVs, and a fake eGTEx all-pairs gzip whose contents exercise every
branch, then checks the harmonizer's output against hand-computed truth.
"""
from __future__ import annotations

import gzip
import json
import random
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

ROOT = Path(tempfile.gettempdir()) / "silentmethyl_egtex_test"
SRC = Path(__file__).resolve().parent

random.seed(7)


def build_fasta(root: Path, chrom_len: int) -> dict:
    """One chromosome of random sequence, with CpGs planted at known offsets."""
    seq = [random.choice("ACGT") for _ in range(chrom_len)]
    cpg_starts = [1000, 5000, 9000, 13000] + [20000 + 500 * i for i in range(40)]
    for s in cpg_starts:
        seq[s] = "C"
        seq[s + 1] = "G"
    text = "".join(seq)
    fa = root / "data" / "hg38.fa"
    fa.parent.mkdir(parents=True, exist_ok=True)
    line = 60
    with fa.open("w") as fh:
        fh.write(">chr1\n")
        for i in range(0, len(text), line):
            fh.write(text[i:i + line] + "\n")
    header_len = len(">chr1\n")
    with (root / "data" / "hg38.fa.fai").open("w") as fh:
        fh.write(f"chr1\t{len(text)}\t{header_len}\t{line}\t{line+1}\n")
    return {"seq": text, "cpg_starts": cpg_starts}


def build_manifest(root: Path, cpg_starts: list[int]) -> dict:
    """Four HM450 probes; the third is MASK_general=True."""
    probes = {}
    rows = ["probeID\tCpG_chrm\tCpG_beg\tCpG_end\tMASK_general"]
    for i, s in enumerate(cpg_starts):
        pid = f"cg{i:08d}"
        masked = "TRUE" if i == 2 else "FALSE"
        rows.append(f"{pid}\tchr1\t{s}\t{s+2}\t{masked}")
        probes[pid] = s
    path = root / "data" / "HM450.hg38.manifest.tsv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(rows) + "\n")
    return probes


def build_splits(root: Path, probes: dict) -> dict:
    """cg00000000 -> train, cg00000001 -> test, cg00000002 -> test, cg00000003 -> val."""
    assign = {"cg00000000": "train", "cg00000001": "test",
              "cg00000002": "test", "cg00000003": "val"}
    for pid in probes:
        assign.setdefault(pid, "train")
    d = root / "data" / "datafiles"
    d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        ids = [p for p, s in assign.items() if s == split]
        (d / f"{split}.csv").write_text("probeID\n" + "\n".join(ids) + "\n")
    return assign


def build_raw(root: Path, fasta: dict, probes: dict) -> list[dict]:
    """The fake all-pairs file. Every row is annotated with what we expect."""
    seq = fasta["seq"]
    expect = []
    lines = []

    def add(probe, pos1, ref, alt, slope, pval, note, build="b38", dist_offset=0):
        cpg0 = probes[probe]
        dist = (pos1 - 1) - cpg0 + dist_offset
        lines.append("\t".join([
            probe, f"chr1_{pos1}_{ref}_{alt}_{build}", str(dist),
            "50", "12", "0.12", f"{pval:g}", f"{slope:g}", "0.05",
        ]))
        expect.append({"probe": probe, "pos1": pos1, "ref": ref, "alt": alt,
                       "note": note})

    p0, p1, p2, p3 = "cg00000000", "cg00000001", "cg00000002", "cg00000003"

    for d in (-499, -100, 2, 250, 500):
        pos1 = probes[p1] + d + 1
        ref = seq[pos1 - 1]
        alt = "A" if ref != "A" else "T"
        add(p1, pos1, ref, alt, 0.3, 1e-9, f"heldout keep d={d}")

    for d in (-500, 501):
        pos1 = probes[p1] + d + 1
        ref = seq[pos1 - 1]
        alt = "A" if ref != "A" else "T"
        add(p1, pos1, ref, alt, 0.3, 1e-9, f"outside window d={d}")

    sign_probes = [p for p in probes if p not in (p0, p2)]
    for probe in sign_probes:
        c0 = probes[probe]
        add(probe, c0 + 1, "C", "A", random.gauss(-0.5, 0.2), 1e-12,
            "destroys target CpG (C>A)")
        add(probe, c0 + 2, "G", "T", random.gauss(-0.5, 0.2), 1e-12,
            "destroys target CpG (G>T)")

    for probe in (p0, p3):
        pos1 = probes[probe] + 60 + 1
        ref = seq[pos1 - 1]
        alt = "A" if ref != "A" else "T"
        add(probe, pos1, ref, alt, 0.2, 1e-6, "model_visible keep")

    pos1 = probes[p2] + 40 + 1
    add(p2, pos1, seq[pos1 - 1], "A" if seq[pos1 - 1] != "A" else "T",
        0.4, 1e-8, "masked probe, dropped")

    lines.append("\t".join(["cg99999999", "chr1_1200_A_G_b38", "100",
                            "50", "12", "0.12", "1e-8", "0.3", "0.05"]))
    expect.append({"note": "EPIC-only probe, dropped"})

    pos1 = probes[p1] + 30 + 1
    lines.append("\t".join([p1, f"chr1_{pos1}_{seq[pos1-1]}_{seq[pos1-1]}AT_b38",
                            str(30), "50", "12", "0.12", "1e-8", "0.3", "0.05"]))
    expect.append({"note": "indel, dropped"})

    pos1 = probes[p1] + 200 + 1
    wrong = "A" if seq[pos1 - 1] != "A" else "T"
    lines.append("\t".join([p1, f"chr1_{pos1}_{wrong}_{seq[pos1-1]}_b38",
                            str(200), "50", "12", "0.12", "1e-8", "0.3", "0.05"]))
    expect.append({"note": "REF mismatch, dropped"})

    lines.append("\t".join([p1, "chr1_400000_A_G_b38", "394999",
                            "50", "12", "0.12", "1e-8", "0.3", "0.05"]))
    expect.append({"note": "beyond +/-600 bp, removed by stage 1"})

    path = root / "data" / "external" / "egtex_breast" / "fake_allpairs.txt.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    return expect


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    (ROOT / "data").mkdir(exist_ok=True)
    shutil.copy(SRC / "harmonize_egtex_mqtl.py", ROOT / "data" / "harmonize_egtex_mqtl.py")
    shutil.copy(SRC / "build_genoa_scoring_input.py",
                ROOT / "data" / "build_genoa_scoring_input.py")

    fasta = build_fasta(ROOT, 500_000)
    probes = build_manifest(ROOT, fasta["cpg_starts"])
    build_splits(ROOT, probes)
    build_raw(ROOT, fasta, probes)

    raw = "data/external/egtex_breast/fake_allpairs.txt.gz"
    pre = "data/external/egtex_breast/fake_within600.tsv.gz"
    cmd = [sys.executable, "-u", "data/harmonize_egtex_mqtl.py",
           "--raw", raw, "--prefiltered", pre,
           "--output-dir", "data/external/egtex_breast/scoring"]

    print("=" * 72)
    print("INSPECT")
    print("=" * 72)
    subprocess.run(cmd[:3] + ["--raw", raw, "--inspect", "--inspect-lines", "50"],
                   cwd=ROOT)

    print()
    print("=" * 72)
    print("RUN")
    print("=" * 72)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"\nFAILED: exit {r.returncode}")
        return 1

    import pandas as pd
    out_dir = ROOT / "data" / "external" / "egtex_breast" / "scoring"
    summary = json.loads((out_dir / "egtex_scoring_summary.json").read_text())

    print()
    print("=" * 72)
    print("ASSERTIONS")
    print("=" * 72)
    failures = []

    held = pd.read_csv(out_dir / "egtex_scoring_input_heldout.csv")
    print(f"heldout rows: {len(held)}")
    print(held[["Variant_ID", "probeID", "distance_bp", "beta_ref_to_alt"]].to_string(index=False))

    got = sorted(held["distance_bp"].tolist())
    want = [-499, -100, 2, 250, 500]
    if got != want:
        failures.append(f"heldout distances {got} != {want}")

    if (held["probe_split"] != "test").any():
        failures.append("heldout contains a non-test probe")
    if held["alters_target_cpg"].any():
        failures.append("target-CpG-altering variant leaked into heldout")
    if "cg00000002" in set(held["probeID"]):
        failures.append("masked probe leaked into heldout")
    if "cg99999999" in set(held["probeID"]):
        failures.append("EPIC-only probe leaked into heldout")

    vis = pd.read_csv(out_dir / "egtex_scoring_input_model_visible.csv")
    if sorted(vis["probeID"].unique()) != ["cg00000000", "cg00000003"]:
        failures.append(f"model_visible probes = {sorted(vis['probeID'].unique())}")

    verdict = summary["effect_allele_convention"]["verification"]["verdict"]
    print(f"\nsign verdict: {verdict}")
    if verdict not in ("confirmed_alt_keyed", "underpowered"):
        failures.append(f"sign verdict {verdict} on data built with negative slopes")

    fc = summary["filter_counts"]
    print(f"filter counts: {json.dumps(fc, indent=2)}")
    for key, expected in (("dropped_non_snv", 1),
                          ("dropped_probe_not_on_hm450", 1),
                          ("dropped_probe_masked", 1),
                          ("reference_base_mismatch", 1)):
        if fc.get(key, 0) != expected:
            failures.append(f"{key} = {fc.get(key)} (expected {expected})")

    if summary["distance_reported_minus_recomputed"].get("0", 0) < 5:
        failures.append("distance agreement diagnostic did not report a modal 0")

    print()
    print("=" * 72)
    print("REVERSED-SIGN SCENARIO (the guard must refuse to write)")
    print("=" * 72)
    import re
    p = ROOT / "data" / "external" / "egtex_breast" / "fake_allpairs.txt.gz"
    with gzip.open(p, "rt") as fh:
        lines = fh.read().splitlines()
    flipped = []
    for ln in lines:
        f = ln.split("\t")
        f[7] = f"{-float(f[7]):g}"
        flipped.append("\t".join(f))
    p2 = ROOT / "data" / "external" / "egtex_breast" / "flipped_allpairs.txt.gz"
    with gzip.open(p2, "wt") as fh:
        fh.write("\n".join(flipped) + "\n")
    r2 = subprocess.run(
        [sys.executable, "-u", "data/harmonize_egtex_mqtl.py",
         "--raw", "data/external/egtex_breast/flipped_allpairs.txt.gz",
         "--prefiltered", "data/external/egtex_breast/flipped_within600.tsv.gz",
         "--output-dir", "data/external/egtex_breast/scoring_flipped"],
        cwd=ROOT, capture_output=True, text=True)
    tail = (r2.stdout + r2.stderr).strip().splitlines()[-6:]
    print("\n".join(tail))
    if r2.returncode == 0:
        failures.append("reversed-sign data did NOT trigger the guard")
    else:
        print(f"\n  guard fired as intended (exit {r2.returncode})")

    print()
    print("=" * 72)
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
