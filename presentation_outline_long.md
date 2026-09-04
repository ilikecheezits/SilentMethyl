# SilentMethyl — presentation outline

**Scope.** The whole paper, start to finish: problem → data → model → primary
results → external validation → mechanism → what the model cannot do →
individual variants → limitations → what's next.

**Sizing.** 21 core slides ≈ 20–25 min. Slides marked `[CUT]` drop for a 10-min
version (leaves 12). Slides marked `[EXPAND]` are where a 45-min version grows.
Appendix slides are for questions, not the main arc.

**One-sentence thesis, decide this before anything else.** Everything on every
slide should serve it:

> A gated sequence + epigenomic-context model predicts CpG methylation
> accurately, its variant-effect predictions replicate in two independent
> cohorts with zero heterogeneity, and the ablation tells us *why* the context
> modality helps absolute prediction but cannot help variant effects.

That last clause is the part reviewers and your mentor will find unusual. Do not
bury it — it is the intellectual contribution, not a caveat.

---

## Act I — Problem and setup (slides 1–5)

### 1. Title
- SilentMethyl: multimodal prediction of CpG methylation and variant-associated
  methylation change
- Samuel Zhang, Shaojun Pei, Gil Alterovitz — BWH / Harvard Medical School

### 2. The biological problem
- DNA methylation is central to epigenetic regulation; the same sequence gives
  different methylation in different contexts
- We want: given a nucleotide variant, what happens to methylation at a nearby CpG?
- **Why it's hard:** matched genetic + epigenetic data at scale barely exists.
  Direct measurement means an meQTL study, which needs hundreds of genotyped
  samples with methylation arrays
- *Say:* this is why a predictive model is worth building at all — it substitutes
  computation for a cohort you cannot easily assemble

### 3. What already exists, and the gap
- Sequence-only methylation predictors (CpGenie, DeepCpG lineage)
- Genomic language models (DNABERT-2 and successors)
- **Gap:** none of them combine a pretrained sequence encoder with measured
  epigenomic context *and* test whether that combination helps variant effects
  specifically
- `[CUT]` in the 10-min version — fold one line into slide 2

### 4. Data construction
- **Targets:** HM450 CpG methylation, cohort medians across 97 TCGA
  solid-tissue-normal breast samples
- **Sequence:** 1,000 bp window centred on each CpG, hg38
- **Context (9 features):** MCF-10A ATAC-seq + 6 histone marks (H3K4me1/3,
  H3K27ac, H3K27me3, H3K36me3, H3K9me3) + 2 phyloP conservation tracks
- **Splits — chromosome-blocked, not random:** train 345,359 / val chr10–11
  46,557 / test chr8–9 26,570
- *Say:* chromosome-blocked because random splits leak — neighbouring CpGs are
  correlated, and a random split puts a probe's neighbour in training
- **Figure:** data-construction panel from Fig. 1

### 5. Architecture
- DNABERT-2 (117M) over sequence → sequence representation
- MLP over the 9 context features → context representation
- LayerNorm → **learned sigmoid gates** → weighted sum → dual heads
  (regression on M-value, classification on methylated/unmethylated)
- Every prediction is reverse-complement averaged (forward + RC)
- *Say:* the gate is the whole design. It decides per locus how much to trust
  sequence vs context, and we can read it afterwards — remember this, it comes
  back in Act IV
- **Figure:** architecture panel from Fig. 1

---

## Act II — Does it work? (slides 6–9)

### 6. Primary result — the baseline table
| Metric | Composition (3 feat.) | k-mer ridge (2,772) | Context (9) | Sequence (DNABERT-2) | **Fusion** |
|---|---|---|---|---|---|
| M-value MAE | 1.9600 | 1.6866 | 1.4574 | 1.1941 | **1.0971** |
| β MAE | 0.1954 | 0.1565 | 0.1395 | 0.1099 | **0.0993** |
| ROC-AUC | 0.8748 | 0.9190 | 0.9187 | 0.9569 | **0.9680** |

- Three seeds, 26,570 held-out CpGs, ±SD in the paper
- Fusion cuts β MAE **36.6% below k-mer ridge**; sequence alone 29.8%
- **The line to say out loud:** k-mer ridge AUC 0.9190 ≈ context-only 0.9187.
  A bag of k-mers with no biology matches nine measured epigenomic tracks. That
  is why the classical baselines are in the paper — without them nobody knows
  whether 0.968 is impressive
- **Table:** Table 1

### 7. Where the context modality earns its place
- Fusion beats sequence-only in *every* stratum tested
- Largest gains: **CpG shores +0.0162**, high H3K27ac +0.0147, high ATAC +0.0139
- Smallest: intergenic +0.0088
- *Say:* the gain is largest exactly where regulatory state should carry extra
  information, and smallest where it shouldn't. That pattern is the argument
  that the model is using context rather than just having more parameters
- **Figure:** fusion gain by epigenomic context

### 8. Why you should believe the held-out number
Three independent checks, one slide:
- **Leakage:** MinHash over canonical 31-mers, every test window vs its nearest
  training window — median Jaccard 0.000, 99th percentile 0.094, **4 of 26,570
  (0.02%)** above 0.5
- **Distribution shift:** held-out chromosomes are more methylated (median β
  0.642 vs 0.550). A model that memorised the training distribution would be
  biased low; mean signed error is −0.002 to −0.004, so it tracks the shift
- **Probe QC:** excluding the 3,105 probes (11.7%) flagged `MASK_snp5_common`
  leaves β MAE unchanged at 0.0993 and *raises* AUC 0.9680 → 0.9689
- *Say:* the flagged probes are the **harder** stratum, not the easier one — and
  all three architectures lose discrimination on the same probes. If the model
  were reading genotype artefact, those probes would score better
- `[EXPAND]` — one slide each in the 45-min version

### 9. Uncertainty, and a methodological warning
- Reverse-complement disagreement as an uncertainty signal
- **The confound:** on the bounded β scale a *zero-parameter* heuristic
  (distance from the boundary) outperforms genuine uncertainty estimators —
  and collapses on the unbounded logit scale
- **Recommendation, generalisable beyond this paper:** assess calibration in
  logit space for bounded bimodal targets
- `[CUT]`, or keep as a single line on slide 8 if the audience is methods-minded
- **Figure:** uncertainty scale stability

---

## Act III — Does it generalise? (slides 10–14) — *the core of the paper*

### 10. The validation design
- Absolute methylation prediction is the easy claim. The real question is
  whether **variant-effect** predictions hold up outside our data
- Two independent cohorts, chosen because they fail in *different* directions:
  - **GENOA** — African American, whole blood, 66,495 held-out variant–CpG
    pairs, 39,657 unique variants, covering 19,081 of our 26,570 held-out probes
  - **eGTEx Breast Mammary** — tissue-matched, 418 pairs, much smaller
- *Say:* GENOA is large but the wrong tissue; eGTEx is the right tissue but
  small. Neither alone is sufficient, which is exactly why both are reported

### 11. External validation results
- **eGTEx (tissue-matched):** ρ +0.246, direction 0.596, calibration slope +1.604
- **GENOA (cross-tissue, cross-ancestry):** ρ +0.152 (0.117–0.188), direction
  55.3% (53.6–57.0), calibration +1.171
- **Meta-analysis** (inverse-variance, LD-block-weighted Fisher-z):
  **ρ = 0.178 [0.068, 0.283], p = 1.6×10⁻³, I² = 0%**
- *Say:* **I² = 0%** is the headline. A European-ancestry breast cohort and an
  African American blood cohort disagree by less than sampling noise. That is
  cross-tissue and cross-ancestry portability in one number
- **Table:** two-cohort variant evaluation

### 12. The controls — why the signal is real
- **GENOA null stratum:** ρ −0.004 [−0.021, +0.013]. Tested-but-null pairs give
  nothing, as they must
- **Significance gradient:** agreement rises monotonically with association
  strength and reaches chance in the null stratum — an internal dose-response
- **Distance gradient:** direction agreement 0.605 at 50–100 bp → 0.497 at the
  window edge. The model is strongest where a local mechanism is plausible
- **Distance-matched discrimination:** AUROC fusion **0.570 [0.554, 0.585]** vs
  k-mer ridge 0.503 [0.488, 0.518] vs composition 0.460 — non-overlapping
- **Honest caveat, put it on the slide:** the eGTEx null control is +0.019
  [+0.006, +0.032], which excludes zero. Most likely sub-threshold signal in an
  underpowered cohort; GENOA is the clean control
- *Say:* volunteering the one control that misbehaved is what makes the other
  four credible
- **Figure:** significance gradient + discrimination vs distance

### 13. Data harmonisation — trust nothing `[CUT]`
- GENOA and eGTEx are hg19/hg38 mixed; liftover in-house, raw files untouched
  and checksummed
- Recomputed variant–CpG distance from the HM450 manifest rather than trusting
  theirs
- Verified effect-direction convention with a CpG-destroying positive control —
  95% negative at z = −12.4 — because GEMMA keys β to the minor allele while
  GTEx keys slope to ALT
- REF-base mismatch: **0.0000% across 1,400,543 pairs**
- *Say:* every one of these was a chance to publish a sign error

### 14. Why this is a *variant-effect* result, not a probe-prediction result
- Reframe explicitly: single-probe accuracy is table stakes; the claim is about
  ranking and signing the effect of substitutions the model never saw
- The distance-matched AUROC is the cleanest statement of that: separating true
  meQTLs from tested-but-null pairs, holding distance fixed
- `[EXPAND]` — this is worth its own slide if the audience is genomics-heavy

---

## Act IV — What the model cannot do, and why (slides 15–17)

*This act is what separates the talk from a benchmark report. Do not cut it.*

### 15. Fusion ≡ sequence for variant effects
- Paired equivalence intervals bounded within **±0.007** in both cohorts —
  fusion and sequence-only are indistinguishable on variant effects
- **The mechanism:** the context vector is *identical* for REF and ALT. The
  variant does not change accessibility or histone marks in our feature set.
  So fusion cannot *create* a variant effect — it can only modulate one through
  the gate
- *Say:* this is not a negative result we are apologising for. It is a
  structural fact about allele-invariant features, it explains the number, and
  it predicts the same thing for any model built this way
- **The upside:** the variant pathway is sequence-only, therefore **tissue-portable** —
  which is exactly why GENOA blood works at all

### 16. The motif analysis — and the control that changed its meaning
- Scanned 831 JASPAR motifs across held-out pairs
- ETS-family disruption couples to predicted hypermethylation, ρ up to −0.28
  against a per-factor null centred at −0.004; survives GC content,
  variant–CpG distance, and substitution class
- **Then we ran a k-mer ridge baseline that has never seen a motif — and it
  reproduces the coupling more strongly**
- Conclusion: the relationship tracks local sequence composition, not learned
  regulatory grammar
- Also: top factors share a core motif, mean pairwise Jaccard 0.278 — one
  family, not independent observations
- *Say:* we reported the control that weakened our own result. That is the slide
  a good reviewer remembers

### 17. Three things we tried that did not work `[CUT]`
- **Chromatin partition** of tissue-shared vs tissue-specific meQTLs — null
  (ATAC flat, active marks trending the wrong way, only H3K27me3 nominal)
- **Tissue-sharing accuracy contrast** — real but fragile; loses significance
  under tighter matching
- **ClinVar matched-background enrichment** — pre-registered, then abandoned
  before running: the comparator pool would have been the other pathogenic
  variants, and n = 35 was declared underpowered in advance
- *Say:* pre-registering the analysis is what let us walk away from it cleanly
  instead of tuning it until it worked

---

## Act V — Individual variants and closing (slides 18–21)

### 18. From ranked variants to testable hypotheses
- The point of the framework is to act on a *specific* substitution
- 440 synonymous variant–CpG pairs; cross-seed rank consistency r = 0.681–0.699
- Effect size follows biology without being told to: median |Δβ̂| 0.0036 at
  promoter/TSS, 0.0026 at UTRs, 0.0017 in gene bodies; 0.00512 within 50 bp vs
  0.00111 at 251–500 bp
- **Neither genomic region nor distance was supplied as a feature**
- **Figure:** candidate response by context

### 19. Two worked examples — with split status stated
- **NCOA2**, top-ranked synonymous candidate, 9 bp from its CpG, predicted
  Δβ −0.18, exceeds all 66 matched background variants — **held-out probe**
  *(⚠ verify this before presenting — currently inferred, not confirmed)*
- **STK11**, two variants including a known pathogenic allele — **training-split
  probes**. Say this out loud on the slide
- *Say:* STK11 is hypothesis generation at a locus the model has seen. The
  interesting claim is narrow and honest — a variant already known to be
  pathogenic because it truncates the protein is *additionally* predicted to
  perturb a nearby CpG, which the protein-level annotation does not capture
- **Figure:** matched-background plot + STK11 panel

### 20. Limitations — own them before you're asked
- Targets are cohort medians from **tumour-adjacent** normals — possible field
  effects and cell-composition heterogeneity
- Context tracks are **MCF-10A reference**, not matched samples
- GENOA is **blood**; effects are on a normalised phenotype scale, so rank and
  sign transfer but magnitude calibration does not
- Motif occurrences are **PWM predictions**, not measured binding
- Ancestry stratification in the training cohort is **underpowered**: 84
  European, 4 African, 1 South Asian
- Single chromosome-blocked holdout; repeated splits deferred
- The eGTEx null control excludes zero

### 21. Where this goes
- Head-to-head against CpGenie and DeepCpG
- Repeated chromosome-blocked splits
- Conformal prediction intervals
- HOCOMOCA as a second motif library, to test whether ETS is JASPAR-specific
- Experimental test of a ranked candidate — the actual end of the arc
- Close on the thesis sentence from the top

---

## Appendix slides (for questions, not the arc)

- **A1** Full metric table with per-seed SDs and the M-value scale
- **A2** Gate behaviour — DNA vs EPI share distributions, and how gates shift REF→ALT
- **A3** Two-stage eGTEx significance thresholding (probe-level FDR, then
  calibrated nominal cutoff), and why the choice was made before seeing results
- **A4** 1 Mb block bootstrap and why LD makes naive intervals too narrow
- **A5** TCGA-BRCA tumour as a shifted target domain — inputs fixed, target
  varied. **Unresolved:** context-only does not degrade and its M MAE improves;
  genuine robustness vs bounded-range compression not yet distinguished
- **A6** Journal positioning — for a mentor audience only

---

## Delivery notes

- **Three numbers to memorise:** β MAE 0.0993 / AUC 0.9680 (primary),
  I² = 0% (portability), ±0.007 (the equivalence that explains the mechanism)
- **The strongest slide is 12** (controls), not 6 (headline metrics). Do not
  rush it
- **The most distinctive slide is 15** (allele-invariance). Most talks would
  hide that result; leading with the mechanism turns it into the contribution
- **Every claim about split status must be on the slide, not just in your
  mouth.** STK11 is training-split. Say it before anyone asks
- If asked *"is this better than just knowing the distance?"* — that is slide
  12, distance-matched AUROC 0.570 vs 0.503. Have it ready
- If asked *"why not just use a k-mer model?"* — slide 6 (AUC parity with
  context-only) plus slide 12 (k-mer ridge fails distance-matched
  discrimination). The k-mer model is competitive on absolute prediction and
  useless on variant effects. That contrast is the paper in miniature
