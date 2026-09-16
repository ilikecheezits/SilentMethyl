# SilentMethyl

Code for the SilentMethyl manuscript (`main_revised.tex`). SilentMethyl predicts
DNA methylation at a CpG from a 1,000-bp DNA window plus seven chromatin-context
tracks and phyloP. It does this through a gated fusion of a sequence tower
(DNABERT-2) and a context tower, and scores variant effects as MUT − WT.

**The published model's context is primary breast epithelium** (ENCODE,
`data/reference/BreastEpithelium/`). The MCF-10A context used before
11 Sep 2026 is superseded, and no published number reads it.

## Start here

| document | what it is for |
|---|---|
| **[`REPRODUCE.md`](REPRODUCE.md)** | fresh clone → every published result: environment, every input and how to fetch it, build order with explicit commands, runtimes, published-path traps, checksums, known gaps |
| `RESULTS_REVISED.md` | every reported result with its interval and caveat |
| `LAB_NOTES.md` | decision record: what was tried, what failed, why each analysis is specified as it is |
| `reproducibility/MCF10A_AUDIT.md` | where every MCF-10A product sits and why none reaches a published number |
| `data/reference/*/TRACK_SET.md` | accession, assay and md5 of every context track |

## Which arms consume context

| arm | reads the seven context columns? | published weights |
|---|---|---|
| sequence-only (DNABERT-2) | **no** | `checkpoints_journal/seed*/sequence`, `checkpoints_folds/fold*/sequence_seed42` |
| context-only | yes | `checkpoints_ablation/breast_epithelium/*/epi` |
| gated fusion | yes | `checkpoints_ablation/breast_epithelium/*/fusion` |

The sequence towers sit under `checkpoints_journal/` because they were trained
before the context swap. They read no context, and every column they do read is
byte-identical between the two builds, so they were deliberately not retrained.
`REPRODUCE.md` §1 has the evidence.

## Layout

```
scripts/          numbered analyses (0x QC, 1x training/eval, 2x variant effects, 3x transfer,
                  4x tissue specificity, 5x mechanism/ASM, 6x candidates, 7x mQTL controls,
                  9x figures and supplements) plus run_*.sh/.sbatch job wrappers
jobs/             Slurm chains for the breast-epithelium analyses (r6, r7, r8)
data/             acquisition, harmonisation and build scripts; inputs are downloaded, not committed
reproducibility/  environment snapshot, input audits, frozen API responses, output checksums
results/          published summaries (large tables, predictions and figures are gitignored)
build_data.sh     context build -> training data -> splits
run_baseline.sh   sequence-tower training
```

`data/`, `checkpoints_*/`, `dnabert2_local/`, `logs/` and large result files are
gitignored. `REPRODUCE.md` regenerates or downloads every one of them.
`cleanup_workspace.sh` records what was removed from the working tree, and how to
restore each item.
