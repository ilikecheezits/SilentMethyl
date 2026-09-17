# MCF-10A context tracks -- SUPERSEDED, HISTORICAL RECORD ONLY

**No published number uses these files.** On 11 Sep 2026 the context of the
published single-tissue model changed to primary breast epithelium,
`data/reference/BreastEpithelium/` (see its `TRACK_SET.md`). These seven bigWigs
produced only the pre-swap record: `data/datafiles/`, the epi and fusion
checkpoints under `checkpoints_journal/` and `checkpoints_folds/`, and the
pre-swap results directly under `results/journal/`.

The sequence towers under `checkpoints_journal/seed*/sequence` and
`checkpoints_folds/fold*/sequence_seed42` were also trained from
`data/datafiles/`, but they read no context columns (see REPRODUCE.md,
"Which arms consume context"), so they are not MCF-10A products.

The per-tissue sets in `BreastEpithelium/`, `Lung/`, `ColonTransverse/` and
`KidneyCortex/` use identical filenames and must not be swapped with these.

Accessions were not in the filenames -- these were downloaded by hand from the
ENCODE portal and renamed. They were recovered by matching each file's md5
against the portal, and are re-verifiable at any time with:

    python -u data/audit_reference_tracks.py --derived-from Ref_ATAC_Signal=ENCFF021PIS

which also writes them into `data/external/external_manifest.json` under
`reference_tracks` (add `--write-manifest`). All eight are GRCh38 natively;
nothing was lifted, despite the `hg19ToHg38.over.chain.gz` sitting in this
directory.

Biosample is **MCF 10A** for all seven -- a clonal immortalized cell line, while
the prediction targets are primary tissue. That domain mismatch is why the
context was replaced with primary breast epithelium.

| feature | file | accession | assay | output type |
|---|---|---|---|---|
| H3K4me3  | `H3K4me3.bw`  | ENCFF548SFG | Mint-ChIP-seq | fold change over control |
| H3K27ac  | `H3K27ac.bw`  | ENCFF282YCX | Mint-ChIP-seq | fold change over control |
| H3K27me3 | `H3K27me3.bw` | ENCFF274LWG | Mint-ChIP-seq | fold change over control |
| H3K9me3  | `H3K9me3.bw`  | ENCFF423DKY | Mint-ChIP-seq | fold change over control |
| H3K36me3 | `H3K36me3.bw` | ENCFF634LDP | Mint-ChIP-seq | fold change over control |
| H3K4me1  | `H3K4me1.bw`  | ENCFF714NIL | Mint-ChIP-seq | fold change over control |
| ATAC     | `ATAC_seq.bw` | ENCFF021PIS (exp. ENCSR037XNN) | **snATAC-seq** | **BAM, converted locally** |
| phyloP   | `hg38.phyloP100way.bw` | -- | UCSC hg38.phyloP100way | not ENCODE |

## The ATAC track is not a download, and cannot be reproduced exactly

`ATAC_seq.bw` was produced locally from `ENCFF021PIS`, an **unreplicated
single-nucleus ATAC BAM** from a lab-custom pipeline (`ENCAN638MKH`), not from a
released bigWig. **The conversion tool and its parameters were never recorded.**

Downloading the BAM therefore gets you the input, not the track. Whatever tool
and flags produced the bigWig set its scale, and the epigenomic tower feeds raw
values into its first `nn.Linear` with no per-feature standardisation, so scale
is not cosmetic. A rebuilt ATAC track will differ from the published one by an
unknown factor, and results that depend on it will shift.

This was the one input to the pre-swap model that could not be reproduced
bit-for-bit. The published breast-epithelium ATAC track is a released portal
bigWig, so the limitation no longer applies to any published result.

Two further consequences, both flagged by `audit_reference_tracks.py`:

- the histone marks are **Mint-ChIP-seq**, not conventional ChIP-seq;
- accessibility is a **single-cell** assay mixed with six bulk ones, differing in
  depth, sparsity and noise structure, not only in scale.

Both are fair to use. Neither is what a generic "ChIP-seq and ATAC-seq signal
from ENCODE" methods sentence would describe.
