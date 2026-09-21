"""Per-probe count of observed normal (sample-type 11) beta values, summarised per split.

Same definition as scripts/01_target_qc.py (n_observed_normals). Source for main.tex line 323
(median 97/97; 95.1% of training probes with >=78 observed normals). Streams the 2.8 GB TCGA
matrix, ~5 min. Its test-split counts match results/journal/ablation_breast_epithelium/target_qc/
test_coverage_per_probe.csv exactly. Run from the repository root:

    $SILENTMETHYL_PY reproducibility/manuscript_checks/coverage_by_split.py reproducibility/manuscript_checks
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/ocean/projects/med250012p/szhang37/SilentMethyl")
OUT = Path(sys.argv[1])
normals = json.load(open(ROOT / "data/datafiles/tcga_normal_sample_ids.json"))
if isinstance(normals, dict):
    normals = next(v for v in normals.values() if isinstance(v, list))
assert len(normals) == 97, len(normals)

split_of = {}
for split in ("train", "val", "test"):
    for p in pd.read_csv(ROOT / f"data/datafiles_breast_epithelium/{split}.csv", usecols=["probeID"])["probeID"]:
        split_of[p] = split
print("probes in splits:", len(split_of), flush=True)

matrix = ROOT / "data/TCGA-BRCA.methylation450.tsv.gz"
probe_col = pd.read_csv(matrix, sep="\t", nrows=0).columns[0]
parts = []
for i, chunk in enumerate(pd.read_csv(matrix, sep="\t", usecols=[probe_col, *normals], chunksize=50_000)):
    chunk = chunk[chunk[probe_col].isin(split_of)]
    parts.append(pd.DataFrame({"probeID": chunk[probe_col].to_numpy(),
                               "n_observed_normals": chunk[normals].notna().sum(axis=1).to_numpy()}))
    print("chunk", i, flush=True)
cov = pd.concat(parts, ignore_index=True)
cov["split"] = cov["probeID"].map(split_of)
cov.to_csv(OUT / "coverage_per_probe_all_splits.csv.gz", index=False)

rows = []
for name, g in list(cov.groupby("split")) + [("combined", cov)]:
    n = g["n_observed_normals"]
    rows.append({"split": name, "n_probes": len(g), "median": float(n.median()),
                 "pct_ge_78": round(100 * (n >= 78).mean(), 2),
                 "pct_eq_97": round(100 * (n == 97).mean(), 2)})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "coverage_summary.csv", index=False)
print(summary.to_string(index=False))
