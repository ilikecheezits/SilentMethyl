# SilentMethyl figure construction guide

This guide implements the five-figure narrative in `SilentMethyl_EXPANDED.tex`. Figure 1 already has an editable vector schematic in `figures/workflow.tikz`; Figures 2–5 have intentional assembly placeholders. Their raw tables and original plotted assets were not attached, so no patient-level points, error bars, or distributions have been fabricated. Source paths below refer to the research repository described in REPRODUCE, not files included in this delivery.

## Visual identity from the reference presentation

The original 40-page presentation was inspected visually. Useful references are slide 6 (compact biological schematic), slide 17 (one clear cohort comparison), slide 21 (consistent matrix labels), slide 29 (aligned paired plots), and slide 33 (grouped biological cards). Use its restrained blue identity, short headings, whitespace, and grouped information. Adapt those choices to a white journal page: small pale-blue panel labels and crisp vector artwork. Keep institutional/conference logos and the slide's full-canvas gradient in presentation material; manuscript affiliation text already supplies institutional attribution. Rebuild original diagrams rather than copying published schematics shown on the slides.

Use this palette consistently across every figure:

| Meaning | Colour | Redundant encoding |
|---|---|---|
| Sequence | Navy `#135E96` | Circle / solid line |
| Reference context | Teal `#239DAD` | Square / dashed line |
| Fusion | Blue `#168BC4` | Diamond / heavier solid line |
| Other learned baselines | Gray `#687582` | Named rows, distinct marker shapes |
| Null/control | Light gray `#AAB4BE` | Open marker / dotted line |
| Highlighted candidate | Orange `#D55E00` | Label and vertical pointer |
| Panel tint | Ice blue `#EDF7FB` | Use only in header strip or input card |
| Text | Charcoal `#263238` | White plot background |

Blue/teal should never be the sole distinction. Direct labels, shapes, and line styles must remain understandable in grayscale.

Proposed production specification (design choices, not a claim of verified current journal limits): full-width canvas 180 mm, heights 85–145 mm depending on figure; sans-serif 8 pt body/axes, 9 pt panel titles, bold 10 pt lowercase panel letters; axes 0.6 pt, data lines 1 pt, arrows 0.8 pt. Allow 5–7 mm gutters. Do not shrink a crowded figure to fit. Use vector PDF with embedded fonts, retain SVG or native source, and rasterize only dense point clouds at 600 dpi while keeping labels vector. No 3D effects, shadows, pictorial bar charts, decorative DNA helices, or significance-star forests. Captions explain methods; panel titles state the question or endpoint in roughly five words.

Nature Communications' [article guidance](https://www.nature.com/ncomms/submit/article) and [official manuscript checklist](https://www.nature.com/documents/ncomms-manuscript-checklist.pdf) support the 5,000-word Introduction–Results–Discussion budget, excluding Methods, abstract, references and figure legends. The expanded manuscript follows this scope. The visual specifications above are proposed design choices; verify final production details against [final artwork guidance](https://www.nature.com/ncomms/submit/guide-to-preparing-final-artwork).

## Source and provenance rules

Set `ABL = results/journal/ablation_breast_epithelium`. Use this tree for current single-tissue fusion/context results. The older `results/journal/manuscript_figures/` contains superseded context-dependent figures; do not load it as a fallback. Current R8 products also occur in `results/journal/manuscript_figures_r8/`, `results/journal/joint/`, and `results/journal/asm_validation*/`.

The figure builder `scripts/91_build_manuscript_figures.py` deletes files outside its whitelist in its output directory. Give it a dedicated staging directory, never the final composite-figure directory. Script 92 makes R8 plots. Use the explicit flags documented in REPRODUCE §5.6; inspect current CSV headers rather than assuming a column schema from this guide.

For each final panel export one source-data table with cohort, model, context version, seed/ensemble, analysis endpoint, filter, sample size and unit, estimate, interval bounds, bootstrap unit/count, and input checksum. Save plotted per-row data when applicable. Keep matched row IDs and pair/group membership. Do not interchange:

- Seed SD and genomic-block confidence intervals.
- 722 SNPs and 15,195 scored SNP–CpG rows in the Do–Tycko source.
- Directional AUROC (positive versus negative observed effect) and discrimination AUROC (association/ASM versus control).
- Three-seed mean metrics and metrics computed from averaged predictions.
- The 418 significant eGTEx pairs and the selected 81-lead positive control.
- Cohort-wide and significant-only correlations.

## Figure 1 — Workflow and model

**Output:** `figures/workflow.pdf`, optional; otherwise the provided TikZ schematic renders automatically. **Canvas:** 180 × 85 mm, two rows, four labeled regions. **Purpose:** explain exactly what goes into the model and distinguish its evaluation tasks.

1. **a, Targets and split:** Draw 97 sample columns converging onto a median-CpG target card. Below it, use three labeled horizontal chromosome blocks: training 345,359; validation chr10–11, 46,557; test chr8–9, 26,570. Do not imply a random participant split. The 418,486 total is the number of retained CpGs, not participants.
2. **b, Sequence and context fusion:** Two parallel inputs. Sequence: 1,000-bp window, target CpG centered at 499–500 → DNABERT-2 → convolution/pooling → LayerNorm → 768-dimensional embedding. Context: seven chromatin tracks + two PhyloP values + nine masks → MLP → LayerNorm → 768-dimensional embedding. Both feed the gate **and** the weighted sum. Connect weighted sum to M regression and continuous classification score. A small training strip says “train towers separately → freeze towers → fit gate and heads.” Add no DNA-shape branch: it is absent from this model version.
3. **c, Allelic contrast:** Use two short aligned REF/ALT sequence strips with one highlighted base. A shared-context bracket supplies both model calls. Show FWD and RC averaging for each allele, followed by ALT minus REF. A note says “same context; gates may differ.” This is more accurate than an arrow showing context canceling completely.
4. **d, Evaluation:** Three compact aligned cards: held-out methylation levels; external mQTL/ASM agreement; hypothesis prioritization. Label the third with NCOA2/STK11 and “experimental follow-up.” Keep it visually separate from external validation.

**Source:** manuscript Methods, README, REPRODUCE §§1, 5.1–5.2. The supplied vector version already encodes these relationships, with abbreviated tower details for legibility. Edit its text/styles directly or replace it with the PDF at the exact path above.

## Figure 2 — Prediction, chromatin context and transfer

**Output:** `figures/fig2_prediction_transfer.pdf`. **Canvas:** 180 × 135 mm, 2 × 2 panels. Use navy and blue paired consistently.

| Panel | Construction | Repository source |
|---|---|---|
| a, Held-out accuracy | Two aligned horizontal dot plots: β MAE and ROC-AUC. Rows: composition, k-mer ridge, context, CpGenie, DeepCpG, sequence, fusion. Show each seed and a larger mean marker; distinguish deterministic ridge fits. Use approximate ranges 0.07–0.21 and 0.85–1.00. Do not connect seeds across unrelated model families. | `ABL/paired_model_bootstrap/model_metrics_recomputed.csv`; baseline exports referenced by scripts 15/16 and Supplementary Table S1 |
| b, Repeated chromosome splits | Paired sequence/fusion dots connected within fold; x = β MAE, y = held-out chromosome set. Every value here is seed 42. Annotate fold-specific n. | `ABL` fold metrics; sequence fold checkpoints/results referenced in REPRODUCE §§1, 5.2, 7.2 |
| c, Relative context benefit | Forest plot of percentage MAE reduction, with separate blocks for island class, ATAC quartiles, H3K27ac quartiles and H3K27me3 quartiles. Show all levels within each displayed block. Use remaining marks/annotations in source data or supplement. | `ABL/fusion_gain_stratified/fusion_gain_stratified.csv`, script 57 |
| d, Transfer error | Point/line plot: measured cross-tissue variance decile 1–10 versus β MAE of breast-held-out model. Show actual available block intervals. Add a compact annotation comparing shore/island proportions in the worst decile and overall. | `results/journal/joint/transfer_failure/transfer_failure_summary.json` and associated per-probe exports, script 56 |

**Checks:** Fusion three-seed mean MAE 0.0914; folds 1–3 fusion 0.0873/0.0934/0.0862. The original fold-0 sequence value 0.1099 is the across-seed mean, whereas REPRODUCE gives seed 42 as 0.1090. Resolve from the prediction export before drawing panel b. Never copy the old claimed fold-gain range. Use relative gains 22.4% for high ATAC and 22.9% for high H3K27ac only with the corresponding current source. Shores are not the largest relative-gain class. The high-H3K27me3 exception must remain visible. Cross-tissue variance is a measured annotation, not a learned gate output.

## Figure 3 — External allelic validation

**Output:** `figures/fig3_external_validation.pdf`. **Canvas:** 180 × 145 mm, 2 × 2 panels; d may contain two compact subplots. Show the paired test and its control together.

**a, Signed cohort agreement.** Two mini forest plots for signed Spearman and direction agreement, rows eGTEx breast and GENOA blood. Use synthesis values consistently: eGTEx ρ 0.246 [0.114, 0.402], GENOA 0.152 [0.118, 0.186]; direction 0.596 [0.547, 0.652] and 0.553 [0.536, 0.571]. Label significant non-CpG-altering n = 418 / 4,037. The original manuscript also reports GENOA ρ = 0.149 in another analysis. Those exports must not be blended into one estimate. Optional signed-effect scatter plots belong in the supplement if the forest panel is crowded; label observed effect in its native study units and predicted ΔM, without a misleading y=x calibration line.

**b, Distance control.** Grouped horizontal forest plot: fusion, sequence, CpGenie, DeepCpG, k-mer, distance-only. Give each model unmatched and matched columns if the actual exports provide both. Use identical matched rows across model comparisons and include the realized distance-only matched AUROC. Reference line 0.5. Fusion unmatched 0.600, matched 0.556 describes a specific reported run; regenerate the entire comparison together rather than mixing per-seed matched exports. Below the plot, a small balance panel can show absolute-distance histograms before and after matching. Check the 10-bp configured tolerance versus the original claim of exact matching.

**c, Association strength.** Two aligned plots (signed ρ, direction agreement) across the seven GENOA p-value strata. Order strongest association left, p > 0.5 right, with n below each tick. Use bootstrap intervals and dashed null lines at 0 and 0.5. The last stratum is “high-p pairs,” not confirmed biological nulls. Do not invent bin edges from the prose; use the evaluation export.

**d, ASM.** Left: forest plot of positive-versus-control discrimination for Rosenski and Do–Tycko, fusion and sequence paired, null 0.5. Right: Do–Tycko signed endpoints, with separate axes for direction agreement and signed ρ; directional AUROC can be in a compact third row with its own axis. Label 722 SNPs / 179 blocks / seed 42. A thin shaded band at 0.60–0.70 on the direction plot shows the recorded expectation and the 0.590 miss. Do not present the three endpoints as three independent confirmations. Keep the n = 53 mammary subset supplementary.

**Sources:** `ABL/variant_effect_synthesis/`, `ABL/genoa_variant_evaluation/`, `ABL/egtex_variant_evaluation/`, `ABL/paired_model_comparison_genoa/`; scripts 21, 30, 31. ASM: `results/journal/asm_validation/asm_discrimination.csv`, `results/journal/asm_validation_tycko/tycko_pair_scores.csv` and script 53e summary outputs. Exact exports and hashes are listed in the reproduction record. Use the 16 September Do–Tycko result: fusion signed ρ = 0.2410, not the stale 0.2425.

## Figure 4 — How context enters an allelic contrast

**Output:** `figures/fig4_context_mechanism.pdf`. **Canvas:** 180 × 125 mm, diagram across the top third; three quantitative panels below. Retain manuscript panel letters a–d even if a sits bottom-left.

**a, Paired model differences.** Three vertically aligned forest rows: signed ρ, direction agreement, distance-matched AUROC. x = fusion minus sequence, dashed zero line. Estimates 0.0026, −0.0007, −0.0021 with their manuscript intervals must come from the same paired-comparison export. Label cohort/stratum/ensemble from that export; if its identity is unresolved, leave this panel unassembled. “No clear added signal” is the appropriate title, not “models are equivalent.”

**b, Counterfactual mechanism.** Draw four cards in order: REF embedding + REF gates; ALT sequence + REF gates; ALT sequence + ALT sequence gate; ALT sequence + both ALT gates. Each card feeds the **same nonlinear regression head H**. Brackets between adjacent outputs label Δsequence, Δsequence-gate, Δcontext-gate. The sum is Δtotal. Keep context embedding the same teal rectangle in all four cards; only its gate changes in the final card. Explicitly note “ordered within-model decomposition; not unique causal attribution.” Include the equations from Methods in the caption or a narrow side block, not full equations in tiny boxes.

**c, Variance versus information.** Two mini plots: unstacked component/total variance ratios (GENOA/eGTEx), and gate-channel partial Spearman after controlling sequence. Gate ratios 0.187/0.195; context-rescaling ratios 0.0617/0.0613. Never stack these into 100%, make a pie chart, or label them percentages of biological variance explained. Covariance is part of Var(total), and different component definitions have different denominators. Partial correlations 0.0152 and 0.0091 get a separate axis and block intervals.

**d, Input perturbations.** Paired dots: normalized prediction deviation for levels versus effects under shuffle and median context. Use 0.3045/0.1123 and 0.1740/0.0692. Label y = deviation / original SD, not methylation-prediction error. A small footer refers to Supplementary Table S3 for Pearson and the failed median-replacement criterion. Do not hide the failed criterion behind the preferred metric.

**Sources:** `ABL/gate_decomposition/gate_decomposition_summary.json` and cohort subdirectories (script 54); `ABL/context_ladder/agreement_with_identity.csv`, `level_accuracy_by_scheme.csv`, and context-permutation exports referenced by script 23. Gate decomposition and perturbation are seed 42; panel a is a separate three-seed comparison. Do not manufacture intervals for deterministic variance ratios from rounded summaries.

## Figure 5 — Candidate prioritization with visible limits

**Output:** `figures/fig5_variant_prioritization.pdf`. **Canvas:** 180 × 110 mm, four panels in 2 × 2 arrangement, visually echoing the reference deck's biological cards.

**a, NCOA2 locus.** Horizontal coordinate schematic centered on cg20699548, with the chr8:g.70148394G>A marker at its actual signed offset. A 9-bp distance bracket is safe before strand/coordinate verification; do not guess left/right orientation. Label gene, probe, genome build, gene-body/OpenSea annotation and current-context Δβ = −0.1539, seed 42. ATAC/H3K27ac Q4 may be annotation badges. Draw real chromatin tracks only if the actual bigWig values are supplied. No invented peaks or arrow implying confirmed NCOA2 repression.

**b, Matched synonymous background.** Empirical CDF or rug-plus-dot plot of |Δβ| for the 66 matched comparators, with NCOA2 highlighted. Use the complete comparator set and display n. Label “background extremeness”; do not show a significance star, FDR badge or discovery threshold. A seed-42 value gets no across-seed error bar.

**c, STK11 illustration.** Histogram of signed Δβ for the complete scored nonsynonymous background, with two labeled vertical lines at +0.0774 (rs2145420809/cg16601904) and −0.0613 (rs148928808/cg08681293). Put “Training-probe examples” in the panel header. Use distinct line styles and offset labels to prevent collisions. Verify current-context provenance of both values from the case-study export.

**d, Clinical annotation control.** Forest plot of top-5% GWAS fraction in eGTEx and GENOA with 95% block intervals. Label the matched cohort's actual null fraction; draw 0.5 only if the final matched design is balanced as intended. Values 0.490 [0.289, 0.673] and 0.596 [0.417, 0.752]. This supports no clear enrichment in this test, not equivalence to chance or absence of biological function.

**Sources:** `ABL/candidates/candidate_matched_background_statistics.csv`, `candidate_seed_scores_long.csv`, `top_candidate_matched_comparators_long.csv`, `top_candidate_case_study.csv`; `ABL/literature_variant_screen/literature_variant_predictions_ranked.csv` and `stk11_case_study_figure_values.csv`; `ABL/gwas_regulatory_enrichment/`. Scripts 60–62, 91 and 52. Current candidate chain is seed 42 only; the three missing candidate-stability exports in REPRODUCE §10 must not be substituted with superseded files.

## Supplementary figures

| Figure | Construction | Sources |
|---|---|---|
| S1, Error-ranking diagnostics | β/M columns; RC disagreement, seed variability, boundary heuristic and random control; risk–coverage and within-stratum correlation. Same retained loci and ranking direction in each comparison. No claim of calibrated interval coverage. | `ABL/rc_uncertainty*`, script 51 |
| S2, Motif controls | Selected motif family correlations side by side for fusion, sequence and k-mer; group redundant ETS matrices; retain composition/substitution controls. | `ABL/motif_disruption/`, context-free k-mer baseline tree, script 50 |
| S3, Target and split QC | Coverage distribution, mask exclusions, train/val/test counts and nearest-training-window similarity. Define denominator for every percentage. | `ABL/target_qc/`, `results/journal/training_data_audit/`, script 01 |
| S4, Detailed external controls | 81-lead sign/magnitude panels, 35 matched-pair discrimination, distance/association-strength strata and n=53 mammary ASM. Keep strong-effect ASM restriction labeled post hoc. | Current positive-control exports and both ASM output trees |

If later restoring the nine-tissue/Melody material, add a supplementary figure only after current-context rescoring and paired source-data verification. Do not use stale multi-tissue fusion results to fill an empty main panel.

## Assembly and acceptance

1. Reconcile the issues in REVISION_NOTES.md, then export source-data CSVs from the existing analyses. No new biological experiment is required to draw the current figures; some provenance gaps require existing-result exports or documented reruns.
2. Generate vector plots from those CSVs using Matplotlib/R; compose in code, Illustrator or Inkscape on the fixed grid. Keep axes editable.
3. Use the exact filenames above. The manuscript loads them automatically from `figures/`. Its small placeholder boxes are assembly markers, not the intended final figure heights.
4. Add per-panel n, uncertainty type and null reference. Verify all printed values against CSVs. Keep captions below 200 words as an editorial target, pending verification of journal rules.
5. Inspect at final 180-mm width, in grayscale, and at 200% zoom. Check font sizes, tick collisions, clipped whiskers, colour consistency and panel-letter order.
6. Compile with `latexmk -pdf SilentMethyl_EXPANDED.tex` after restoring `references.bib`. Compile `supplementary_revision.tex` separately. Recheck page layout after inserting real figures; the current 16-page preview is not the final figure-filled page count.
