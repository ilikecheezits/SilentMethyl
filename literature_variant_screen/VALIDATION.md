# Validation performed in the scratch workspace

- Python syntax compilation passed for the screen and test modules.
- Six offline tests passed, covering GRCh38 SPDI conversion, allele-direction
  invariant duplicate detection, protein-altering ClinVar parsing, live
  significance rechecking, current-data/window exclusions, and prespecified
  target-CpG ranking.
- All 20 curated rsIDs resolved through the live NCBI RefSNP API to the exact
  requested GRCh38 REF/ALT SNV.
- A live CHEK2 ClinVar smoke test returned protein-altering pathogenic records,
  and ELink retrieved linked PubMed identifiers for all three retained records.
- The BRCA1 M1775R seed was corrected during validation from an incorrect rsID
  to rs41293463 (GRCh38 chr17:43051071 A>C).

The full model run could not be executed in this scratch workspace because the
project's split tables, HM450 reference files, DNABERT-2/PyTorch environment,
and frozen fusion checkpoints are not present here. The full-run command in the
README performs those remaining checks and generates the prediction ranking on
the SilentMethyl cluster environment.
