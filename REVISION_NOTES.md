# Revision notes

## Files and length

This version replaces the overly compressed first revision. It restores the older main.tex narrative and fuller scientific explanation while retaining the expanded analyses and scientific corrections.

`SilentMethyl_EXPANDED.tex` is the editable manuscript; `SilentMethyl_EXPANDED_16pages.pdf` is the 16-page reading preview. `FIGURE_GUIDE.md` supplies panel layouts, source paths and construction instructions. `figures/workflow.tikz` remains the completed editable vector workflow. The supplementary TEX/PDF and exact patch from the originally attached revised manuscript are also included.

Compile from this folder with `latexmk -pdf SilentMethyl_EXPANDED.tex`. Add the actual `references.bib` beside it. Final figure PDFs replace the placeholders automatically using the guide's filenames.

| Section | Current word count |
|---|---:|
| Introduction | 631 |
| Results | 3,278 |
| Discussion | 862 |
| **Main text** | **4,771** |
| Methods, excluded from main-text limit | 1,621 |
| Abstract, separately counted | 141 |

The [official article guidance](https://www.nature.com/ncomms/submit/article) identifies a 5,000-word main-text target excluding abstract, Methods, references and figure legends. The [journal's manuscript checklist](https://www.nature.com/documents/ncomms-manuscript-checklist.pdf) specifies Introduction, Results and Discussion as that main text. Counts here include section headings and a token for inline mathematical expressions, using OpenDetex; see WORD_COUNT.md and the extracted counting text. Allow for small differences between counters. The draft leaves 229 words below 5,000 under this convention.

## What was restored and preserved

- Restored the original introduction's progression: biological motivation, existing models, why context matters, evaluation gaps and synonymous variants.
- Reused original sentences and closely retained descriptions of target construction, staged training, paired scoring, and the NCOA2/STK11 case studies. Superseded metrics and overclaims remain corrected.
- Restored the attention-pooling equation and fuller Methods explanations. Methods remains within the manuscript and is not cut to meet the main-text budget.
- Expanded Results into clear subsections with a rationale, observations and interpretation for each analysis, while keeping the five-figure plan.
- Restored a fuller Discussion addressing biological interpretation, evaluation design, applications and limitations. Uncertainty and motif controls now have a main-text subsection as well as supplementary detail.
- Preserved the two-column format, increased body text to 11 points, added 1.12 line spacing and larger paragraph/figure gaps, and used 0.75-inch margins. Page count is allowed to grow rather than being compressed artificially.

The original main(4).tex is unchanged. The presentation remains a visual reference only. Figures 2–5 still require their underlying source tables; placeholder boxes are not final figure dimensions. The complete bibliography also remains unavailable. These facts affect submission readiness, not completion of this wording/layout revision.

## Corrections made

| Original issue | Revision |
|---|---|
| Fixed context said to make tissue-specific allelic effects impossible | Removed: sequence-dependent gates can make context contribute; limited predictive gain is empirical. |
| Gate ratios treated as additive attribution | Defined the ordered nonlinear-head counterfactuals; covariance and order dependence are explicit. |
| Tissue identity said to reside entirely in chromatin | Replaced with substantial transfer under the evaluated design. |
| Error ranking called calibrated uncertainty | Distinguished ranking/scale dependence from verified predictive-interval coverage. |
| Nonsignificant model differences treated as equivalence | Replaced with no clear difference in the particular evaluation. |
| Current candidate screen assigned old three-seed stability | Removed; current-context NCOA2 is −0.1539 at seed 42. |
| Nine-tissue fusion and Melody results mixed across context versions | Kept out of main claims pending the reruns documented in REPRODUCE §10. |
| ASM signed ρ = 0.2425 | Updated to 0.2410 from the dated 16 September lab result. The 0.590 direction-concordance miss remains. |
| Three signed ASM metrics treated as independent confirmations | Identified as correlated summaries of one SNP set. |
| ASM scoring radius described as 1,000 bp | Clarified as inside the actual 1,000-bp centered model window. |
| Candidate extremeness implied discovery/pathogenicity | Kept descriptive, with STK11 training-probe status explicit. |

## Numerical and provenance issues to reconcile

These are source-data issues, not new analyses performed during this edit.

1. **Fold 0:** the manuscript's seed-42 fold table says sequence MAE 0.1099, while REPRODUCE §7.2 says 0.1090. The former also equals the across-seed mean. The old stated fold-gain range does not match the table. This draft retains current fusion fold values and removes the inconsistent range; draw all final paired fold values from one export.
2. **DeepCpG arithmetic:** 0.1253 versus 0.0914 implies 27.1%, not the stated 20.8%. The incorrect percentage is removed from main text.
3. **GENOA exports:** the manuscript reports ρ 0.149/direction 0.556 in direct evaluation and 0.152/0.553 in synthesis. This draft labels the latter as synthesis values. Preserve each export's filters and estimator; do not blend them. Attach the exact cohort/stratum/source identity of the paired fusion-minus-sequence statistics to Figure 4a.
4. **Matching:** prose claims exact distance balance, but commands set `--match-tolerance 10`. This does not prove imbalance. The revised Methods reports configured tolerance; any exact-balance claim needs the realized matched-row distances and distance-only AUROC.
5. **Candidates:** current-context seeds 43/44 and stability exports are missing according to REPRODUCE. Do not substitute old ensemble correlations or NCOA2 dispersion. Verify STK11 numbers against the current case-study export before plotting.
6. **Table S1:** values are retained from the supplied table, not recomputed. M-value rows also occur in the older draft; verify all columns from frozen current outputs. An SD displayed as 0.0000 reflects rounding.
7. **Imputation:** the supplied Methods says feature-global median. That wording is retained rather than silently changing the implementation. Establish whether medians were fit on training loci only and describe the actual procedure accurately.

## Remaining submission work

- Restore and verify `references.bib`. Only citation keys were attached, so the preview retains the existing explicit fallback. Add complete Rosenski/Do–Tycko citations and dataset versions. Recent literature references were not independently audited here.
- Produce Figures 2–5 from actual tables using the guide. Figure 1 already renders. Include panel-level Source Data and complete supplementary-data packages with hashes.
- Resolve the concrete numerical issues above. Existing missing-context reruns remain missing; editing does not substitute for them.
- Add author-approved acknowledgements/funding, contributions, competing interests, ethics/public-data wording, reporting summary and a repository release identifier. These facts were not invented.
- Main-text scope and length were checked against official Nature Communications material in this revision. Figure dimensions and typography in the guide remain proposed design choices; verify final artwork requirements before submission. This is a research draft, not a claim of submission readiness or likely acceptance.
