# BreastEpithelium context tracks -- THE PUBLISHED MODEL'S CONTEXT

Since 11 Sep 2026 these seven tracks are the context of every published
single-tissue SilentMethyl result (`data/datafiles_breast_epithelium/`,
`checkpoints_ablation/breast_epithelium/`,
`results/journal/ablation_breast_epithelium/`) and the breast arm of the joint
multi-tissue model.

The seven bigWigs directly in `data/reference/` are the SUPERSEDED MCF-10A set
(see `data/reference/TRACK_SET.md`). They share these filenames; never swap the
directories.

All seven are released ENCODE GRCh38 bigWigs, biosample **breast epithelium**,
downloaded as-is (no local conversion, no exclusion-list filtering), verified by
the portal md5 below. The md5s were re-checked against the files on disk on
16 Sep 2026: 7/7 match.

| feature | file | accession | experiment | assay | output type | md5 |
|---|---|---|---|---|---|---|
| ATAC | `ATAC_seq.bw` | ENCFF665NGK | ENCSR955JSO | ATAC-seq | fold change over control | `7effdce98e749c91009b74c6eac225cd` |
| H3K27ac | `H3K27ac.bw` | ENCFF085IYD | ENCSR081OTO | Histone ChIP-seq | fold change over control | `5ddb4795f5bcd88dd75ebb874a33199d` |
| H3K27me3 | `H3K27me3.bw` | ENCFF212ZFW | ENCSR134LLK | Histone ChIP-seq | fold change over control | `6c4830851a98900cd213db67ed57a892` |
| H3K36me3 | `H3K36me3.bw` | ENCFF714QJF | ENCSR793QCL | Histone ChIP-seq | fold change over control | `706a5be148d96239ca0ba4cb819af011` |
| H3K4me1 | `H3K4me1.bw` | ENCFF234JZW | ENCSR263XKR | Histone ChIP-seq | fold change over control | `cbfcb57dbc1de468d185392839f8775b` |
| H3K4me3 | `H3K4me3.bw` | ENCFF653CLL | ENCSR224STY | Histone ChIP-seq | fold change over control | `0744ffdccc18c4bbba4f1054eb6cf0d7` |
| H3K9me3 | `H3K9me3.bw` | ENCFF481QEK | ENCSR936LAH | Histone ChIP-seq | fold change over control | `36b8acf0bedfb712692d8117b1f971e0` |

Fetch: `python -u data/acquire_multitissue_inputs.py --tissues BreastEpithelium --context-only --picks data/multitissue_picks.tsv --apply`
(or download each URL `https://www.encodeproject.org/files/<accession>/@@download/<accession>.bigWig`
and rename to the file column).

PhyloP is genome conservation, identical in every tissue, and is read from
`data/reference/hg38.phyloP100way.bw`.

`data/reference/ENCFF356LFX.bed.gz` is the ENCODE GRCh38 exclusion list
(Amemiya et al. 2019). It is **not** a context track and is **not** applied to
these files; it is used only by `data/profile_bigwig.py` (track audits) and by
`acquire_multitissue_inputs.py --blacklist` when converting BAMs.
