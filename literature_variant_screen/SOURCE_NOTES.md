# Evidence and source notes

The script queries ClinVar live because a hand-written list cannot remain
comprehensive. ClinVar supports ESearch, ESummary, ELink, and EFetch, and its
canonical SPDI is expressed on a GRCh38 chromosome sequence. ELink is used to
require at least one linked PubMed record for dynamically discovered variants.

Official documentation:

- https://www.ncbi.nlm.nih.gov/clinvar/docs/programmatic_access/
- https://ncbiinsights.ncbi.nlm.nih.gov/2020/04/28/canonical-spdi-notation-now-in-clinvar/
- https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/121913273

The curated seeds are not the complete screen. They ensure that several major
experimentally studied classes are present even if a live ClinVar condition
query changes. Examples include:

- PIK3CA H1047R/E545K in mammary epithelial or mammary tumor models:
  https://pubmed.ncbi.nlm.nih.gov/20581867/ and
  https://pubmed.ncbi.nlm.nih.gov/24080956/
- AKT1 E17K in breast tumors:
  https://pubmed.ncbi.nlm.nih.gov/18504432/
- ESR1 activating mutations and endocrine resistance:
  https://pubmed.ncbi.nlm.nih.gov/26836308/ and
  https://pubmed.ncbi.nlm.nih.gov/36198774/
- SF3B1 K700E in mammary tumorigenesis:
  https://pubmed.ncbi.nlm.nih.gov/33031100/
- BRCA1 C61G functional mammary model:
  https://pubmed.ncbi.nlm.nih.gov/22172724/
- CHEK2 I157T case-control and functional evidence:
  https://pubmed.ncbi.nlm.nih.gov/15239132/
- CHEK2 L236P ClinVar evidence and linked publications:
  https://www.ncbi.nlm.nih.gov/clinvar/variation/142448/
- BRCA2 K3326* breast/ovarian risk study:
  https://pubmed.ncbi.nlm.nih.gov/26586665/
- CASP8 D302H breast-cancer association:
  https://doi.org/10.1038/ng1981

## Why rs10069690 is not the default example

The eGTEx study describes rs10069690--cg03935379 as an **ovary-specific** mQTL
whose signal colocalizes with a breast-cancer GWAS association. That is useful
biology, but it is not a breast-tissue mQTL and should not be written as one.
It is also usable by SilentMethyl only if the named CpG lies within the trained
1,000-bp window around the variant. Relevant sources:

- https://www.nature.com/articles/s41588-022-01248-z
- https://pubmed.ncbi.nlm.nih.gov/36510025/

Colocalization and mQTL association do not prove that methylation mediates the
disease association. The final manuscript should reserve causal wording for a
design that directly tests mediation and its assumptions.
