# SilentMethyl — 5–7 minute talk

**8 slides. Setup in 90 seconds, results for 4 minutes, close in 60.**
The 20-minute version is in `presentation_outline_long.md`.

**Thesis — say it on slide 1 and again on slide 8:**
> The model predicts CpG methylation accurately, its variant-effect predictions
> replicate in two independent cohorts with zero heterogeneity, and we can
> explain exactly which part of the model does that work.

---

### 1. Title + the problem — 30 s
- SilentMethyl: predicting CpG methylation and variant-associated methylation change
- Zhang, Pei, Alterovitz — BWH / Harvard Medical School
- **The problem in one sentence:** measuring how a nucleotide variant changes
  nearby methylation requires a genotyped cohort with methylation arrays, which
  mostly doesn't exist — so predict it instead

### 2. What we built — 45 s *(one slide, don't split)*
- DNABERT-2 over 1,000 bp of CpG-centred sequence **+** MLP over 9 breast
  epigenomic features (ATAC, 6 histone marks, 2 conservation)
- **Learned gates** weight the two per locus → methylation prediction
- Variant effect = score REF, score ALT, take the difference
- Chromosome-blocked splits (test chr8–9, 26,570 CpGs), 3 seeds
- *Say:* chromosome-blocked because neighbouring CpGs are correlated and a
  random split leaks
- **Figure:** architecture panel only

### 3. Primary result — 60 s
| | k-mer ridge | Context only | Sequence | **Fusion** |
|---|---|---|---|---|
| β MAE | 0.1565 | 0.1395 | 0.1099 | **0.0993** |
| ROC-AUC | 0.9190 | 0.9187 | 0.9569 | **0.9680** |

- AUC **0.968**, β MAE **0.099** on held-out chromosomes
- **36.6% lower error than a classical k-mer baseline**
- *The line to say:* k-mer ridge 0.9190 ≈ nine measured epigenomic tracks
  0.9187. A bag of k-mers matches real biology — which is why the baselines are
  in the paper. 0.968 only means something against that
- **Figure/table:** Table 1

### 4. It generalises — two independent cohorts — 75 s ★ *biggest slide*
- **GENOA** — African American, whole blood, 66,495 held-out variant–CpG pairs
- **eGTEx** — tissue-matched breast, 418 pairs
- *Say:* one is large but wrong tissue, one is right tissue but small. Neither
  alone is enough, which is why both are here
- **Meta-analysis: ρ = 0.178 [0.068, 0.283], p = 1.6×10⁻³, I² = 0%**
- *Say:* **I² = 0%** — a European breast cohort and an African American blood
  cohort disagree by less than sampling noise. Cross-tissue and cross-ancestry
  portability in one number
- **Table:** two-cohort evaluation

### 5. Why you should believe it — the controls — 60 s
- **Null stratum:** tested-but-null pairs give ρ −0.004 [−0.021, +0.013]
- **Dose–response:** agreement rises with association strength, hits chance in
  the null stratum
- **Distance-matched:** AUROC 0.570 [0.554, 0.585] vs k-mer ridge 0.503
  [0.488, 0.518] — non-overlapping, so it isn't just distance
- **Probe QC:** excluding common-SNP-affected probes leaves error unchanged and
  *raises* AUC to 0.9689
- *Optional, if the room is rigorous:* the eGTEx null control is +0.019 and
  excludes zero — likely sub-threshold signal in a small cohort; GENOA is the
  clean control
- **Figure:** significance gradient

### 6. What the model can and cannot do — 45 s ★ *the distinctive slide*
- Fusion and sequence-only are **indistinguishable** on variant effects
  (equivalence bounded within ±0.007)
- **Why:** the context features are identical for REF and ALT. They're
  allele-invariant, so they *cannot* create a variant effect
- *Say:* this isn't a disappointment, it's the mechanism. Context helps absolute
  methylation prediction; the variant pathway is sequence-only — which is
  exactly why it transfers to blood
- Same discipline elsewhere: our motif result reproduced *more strongly* by a
  k-mer control, so we report it as compositional, not learned grammar

### 7. Individual variants — 30 s
- 440 synonymous variant–CpG pairs ranked; effect size falls with distance
  (0.0051 within 50 bp → 0.0011 at 251–500 bp) and is largest at promoters —
  **neither was given as a feature**
- Top candidate: *NCOA2*, 9 bp from its CpG, predicted Δβ −0.18, exceeds all 66
  matched background variants
- *STK11* pathogenic variant — **training-split probe, hypothesis generation only**
  *(⚠ confirm NCOA2's held-out status before presenting)*

### 8. Limitations + next — 30 s
- Targets from tumour-adjacent normals; context from MCF-10A reference, not
  matched samples; GENOA is blood so sign and rank transfer but magnitude
  doesn't; single chromosome holdout
- Next: head-to-head vs CpGenie/DeepCpG, repeated splits, experimental test of
  a ranked candidate
- Close on the thesis sentence

---

## Delivery

- **Three numbers to land:** 0.968 AUC · I² = 0% · ±0.007
- **Spend your time on slides 4 and 5.** Slide 3 is table stakes; the
  replication and the controls are the talk
- **Slide 6 is what people remember** — most talks bury that result
- **Say STK11 is training-split out loud.** Don't wait to be asked
- Cut first if you're over time: slide 7 down to the NCOA2 line only
- Likely question — *"isn't this just distance?"* → slide 5, 0.570 vs 0.503
