# SilentMethyl literature-variant screen

This package constructs a broad, auditable panel of published protein-altering
breast-cancer SNVs and runs the frozen SilentMethyl fusion ensemble on every
eligible variant--CpG pair. It keeps the existing NCOA2 analysis unchanged.

## What the script enforces

- GRCh38 normalization through NCBI's RefSNP or ClinVar canonical SPDI record.
- Single-base REF and ALT alleles only.
- Protein-altering ClinVar consequence (missense, stop/start change, or a
  general protein-altering annotation).
- At least one linked PubMed article or curated primary-literature URL.
- Exclusion of any variant already present in `testing_data.csv`.
- Exclusion of every variant already present in the frozen mQTL benchmark.
- An HM450 CpG within the trained sequence window (offset -499 through +500),
  followed by the existing reference-allele, target-CpG, and probe-QC checks.
- Explicit retention of train/validation/test split membership.

## Required project files

Place the supplied script and seed manifest in `scripts/` (keep the README,
source notes, tests, and preview CSV anywhere convenient for reference):

```text
scripts/14_known_variant_application.py
scripts/15_literature_variant_screen.py
scripts/literature_breast_variant_seeds.csv
```

The following existing SilentMethyl files are required:

```text
data/datafiles/train.csv
data/datafiles/val.csv
data/datafiles/test.csv
data/datafiles/testing_data.csv
data/egtex_breast_mqtl_heldout_qc.csv
data/HM450.hg38.manifest.tsv.gz
data/HM450.hg38.manifest.CpGIsland.tsv.gz
data/reference/gencode.v44.annotation.gtf.gz
checkpoints_journal/seed{42,43,44}/fusion/best_weights.pth
scripts/05_matched_background.py
scripts/12_biological_context_analysis.py
```

For exact eGTEx breast-mQTL intersections, also provide the original eGTEx
lead table:

```text
data/BreastMammaryTissue.regular.perm.fdr.txt
```

If it is absent, screening and model scoring still run, but
`candidate_breast_mqtl_hits.csv` will be empty and no candidate may be claimed
as an exact external breast-mQTL pair.

## Environment

No liftover program is required. The code resolves rsIDs through current NCBI
GRCh38 chromosome placements and accepts ClinVar canonical GRCh38 SPDI alleles.
This is safer than lifting an old coordinate without rechecking REF/ALT.

Set an email for NCBI requests; an API key is optional:

```bash
export NCBI_EMAIL="your_email@example.edu"
# export NCBI_API_KEY="..."  # optional
```

## Dry preparation run

This queries/caches the literature databases, applies all pre-scoring filters,
and writes the exact variant table that will be sent to SilentMethyl:

```bash
python -u scripts/15_literature_variant_screen.py \
  --prepare-only \
  2>&1 | tee logs/experiments/15_literature_variant_screen_prepare.log
```

Inspect these files before using GPU time:

```text
results/journal/literature_variant_screen/source_query_audit.csv
results/journal/literature_variant_screen/candidate_exclusion_audit.csv
results/journal/literature_variant_screen/publication_link_audit.csv
results/journal/literature_variant_screen/eligible_published_candidates.csv
results/journal/literature_variant_screen/silentmethyl_variant_input.csv
```

## Full frozen-model run

```bash
python -u scripts/15_literature_variant_screen.py \
  --seeds 42 43 44 \
  --device auto \
  2>&1 | tee logs/experiments/15_literature_variant_screen.log
```

The curated seeds are a guaranteed literature-supported starting set; their
resolved GRCh38 coordinates are provided in `curated_seed_grch38_preview.csv`.
They fall on training chromosomes except ATM D1853N on validation chromosome
11, so none can be presented as a held-out test result. The broader live
ClinVar queries include additional breast-cancer genes on chromosomes 8 and 9
and may produce a defensible held-out application if an eligible published SNV
and unmasked nearby HM450 CpG survive all checks.

The key result is:

```text
results/journal/literature_variant_screen/literature_variant_predictions_ranked.csv
```

Ranking does not choose the most responsive CpG around each variant. If a
candidate has an exact significant eGTEx breast-mQTL target, that published CpG
is used; otherwise the nearest eligible model-visible HM450 CpG is used. All
other scored pairs remain available in
`literature_variant_predictions_all_pairs.csv` for transparent review.

The candidates are separated into four evidence tiers. Use tier 1 if it
exists. Otherwise, tier 2 is the strongest defensible held-out application,
but it must be described as an exploratory, model-prioritized case rather than
external validation. Tier 3 and tier 4 should not be used as held-out evidence.

## Important interpretation rules

1. Selecting the largest response from a screened panel creates selection
   bias. Report the panel size and the prespecified selection rule.
2. A large predicted delta is a prioritization score, not a measured methylation
   effect and not evidence that the variant causes breast cancer.
3. A ClinVar record can describe hereditary risk while the model uses a breast
   reference epigenome. Germline/somatic status and tissue relevance must be
   stated for the final case.
4. An mQTL association supports allele--methylation association, not necessarily
   causal mediation through methylation.
5. The earlier rs10069690--cg03935379 example should not be called a breast
   mQTL: the eGTEx report describes that pair as ovary-specific while noting
   colocalization with a breast-cancer GWAS signal. The script will also reject
   the pair if the target CpG is outside SilentMethyl's 1,000-bp window.
6. Variants already present in either the TCGA application cohort or the frozen
   mQTL benchmark are excluded by default, making the new application panel
   distinct from both analyses already reported in the manuscript.
