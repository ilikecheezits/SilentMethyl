# Melody vs SilentMethyl — positioning note

Jin, Wang, Qiao et al. "Decoding the sequence determinants of locus-specific DNA
methylation across human tissues." *Nat Commun*, accepted 5 Aug 2026, online
17 Aug 2026. doi:10.1038/s41467-026-76744-5

Read this before writing R3, the Discussion, or the cover letter. Melody was
published in our target journal three weeks before our nine-tissue analysis and
overlaps it directly. That is survivable, but only if we choose the comparison
ground deliberately instead of letting a reviewer choose it for us.

---

## 1. What Melody is

- Fully convolutional encoder–decoder (U-Net-like) over **10-kb** input windows.
- Trained on **39 normal human cell types**; three variants:
  - **Melody-ST** — one track, one cell type.
  - **Melody-MT** — multi-track, shared cross-tissue features. Best on meQTLs.
  - **Melody-G** — adds scRNA-seq foundation-model embeddings via FiLM
    conditioning to predict methylation in **unseen** cell types.
- Multi-task head: per-CpG methylation + 100-bp regional average + CpG density.
- Baselines beaten: INTERACT, CpGenie, iDNA-ABT.
- Headline methylation numbers: Melody-MT Spearman 0.723 (sampling test) and
  0.645 (test chromosome) vs 0.590 / 0.584 for the best competitor;
  AUC 0.921, AP 0.975.

## 2. The direct overlap — Figure 3H

This is the figure to worry about. Melody performs a **cross-cell-type meQTL
validation** on GTEx data across **Breast, Colon, Kidney, Lung, Muscle, Ovary,
Whole Blood** — seven of our nine tissues, from the same source. Their finding:
"related tracks typically achieve the best or second-best performance,"
i.e. the tissue-matched track predicts that tissue's meQTLs best.

So the sentence "we test whether a model trained on one tissue predicts meQTLs
in another" is **not novel as of 17 Aug 2026**. We must not write R3 as though
it were.

## 3. Where we are genuinely not comparable — and stronger

Melody's meQTL evaluation is **Pearson correlation between predicted and
observed effect, computed among meQTLs only** (Fig 3C–H; e.g. GTEx Whole Blood
r = 0.4236, n = 1334; MDSs r = 0.6253, n = 524).

That design has three gaps we already fill:

1. **No negative controls.** Melody never asks whether a variant is an meQTL at
   all — only how well the effect size correlates among variants already known
   to be meQTLs. Our primary metric is discrimination of real meQTLs from
   *tested-but-null* pairs.
2. **No distance control.** Their own Fig 3E shows performance falls with
   variant–CpG distance for every model, which is exactly the confound: distance
   alone reaches AUROC 0.60–0.69 on our cohorts. We match on distance until
   distance-only AUROC is **exactly 0.5000**, then measure what is left. Nothing
   in Melody establishes that its meQTL signal is not distance.
3. **No CpG-altering exclusion.** We drop variants that create or destroy the
   target CpG (~37% of pairs, and enriched for true meQTLs), so our numbers
   exclude the mechanically trivial cases.

**Do not compete on effect-size correlation.** Melody reports Pearson ~0.3–0.5
on GTEx tissues; our signed Spearman among significant pairs is ~0.18 (Lung).
Their 10-kb window and 39-tissue training will win that metric.

> **Superseded in part, 5 Sep 2026.** The original second half of this paragraph
> read "compete on discrimination under a distance-matched null, where they have
> no result at all." The eight-tissue head-to-head (§3c) shows we do not *win*
> there either — we tie. The distance-matched null is still where our
> contribution lives, but the claim it supports is parity-under-asymmetry, not
> superiority. Do not write the superiority sentence.

## 3b. Reanalysis of Melody's own Source Data — the strongest point we have

Computed 5 Sep 2026 from `supplemental_data/supplemental_data.xlsx` (Figure 3H
sheet) and the benchmark CSVs shipped in `meqtl/dataset/processed/GTEX/`. No
compute, no checkpoint, entirely their published numbers.

**Finding 1 — tissue identity of the model track is second-order.**
Two-way decomposition of their 8x8 cross-track matrix:

    variance explained by TARGET dataset : 85.0%
    variance explained by SOURCE track   :  1.4%
    residual (interaction, incl. matching): 13.5%

The matched track's advantage over other tracks on the same target averages
**+0.027 Pearson r**. Their claim that related tracks are "best or second-best"
holds column-wise in 5 of 8 targets, so it is not wrong -- it is just small.
State it that way; do not claim their result fails.

**Finding 2 — what the 85% actually is: distance.**
Median variant-CpG distance per benchmark, against their reported Pearson r:

    Ovary       120 bp   0.407        Kidney      473 bp   0.350
    Prostate    127 bp   0.453        Breast      478 bp   0.371
    Lung        129 bp   0.427        WholeBlood  484 bp   0.324
    Colon       153 bp   0.417        Muscle      684 bp   0.278

    Spearman -0.833 (p = 0.010)      Pearson -0.946 (p = 0.0004)
    cohort size vs r: Pearson +0.323 (p = 0.44)  -- NOT power

Roughly 89% of the variance in Melody's cross-tissue meQTL performance is
explained by how proximal each benchmark's pairs happen to be. Tissue explains
1.4%. Sample size explains nothing.

**Why this matters for our paper.** It is the strongest available motivation for
distance-matched evaluation, and it comes from the leading model rather than
from us. The argument writes itself: performance differences that look like
tissue biology are largely differences in benchmark composition, which is
exactly what a distance-matched null removes.

**Caveats to state honestly, because this is a claim about someone else's work.**
- n = 8 datasets. The correlation is strong but the sample is small, and we
  chose the predictor after seeing the pattern. Present it as a reanalysis that
  motivates our design, not as a definitive decomposition of their method.
- Melody's authors did NOT hide this: their Fig 3E explicitly shows performance
  declining with variant-CpG distance for every model tested. The criticism is
  that no one in this literature CONTROLS for it, not that they concealed it.
  Say so; it is both fairer and more credible.
- A correlation across datasets is not the same as distance driving predictions
  within a dataset. Our own within-cohort result -- distance alone reaching
  AUROC 0.60-0.69, pinned to 0.5000 by matching -- is what closes that gap. The
  two together are the argument; either alone is weaker.

**Consequence for priorities.** This reanalysis makes the head-to-head
confirmatory rather than load-bearing. Still worth doing -- "and when distance
is controlled directly, here is what happens" beats a correlation over 8 points
-- but the paper no longer depends on it.

## 3c. Head-to-head on our distance-matched cohorts — eight tissues

Run 5 Sep 2026, `scripts/31_transfer_discrimination.py`, output under
`results/journal/melody_head_to_head/<Tissue>`. Both models scored on **byte-identical
rows** (SilentMethyl's per-tissue `pair_scores.csv` is the template; only
`Predicted_Delta_M` / `Absolute_Delta_M` are replaced — `scripts/34_melody_by_tissue.py`).
Same matched negatives, same 1-Mb blocks resampled together. Testis excluded: no
testis cell type exists among Melody's 39 tracks.

**The comparison is not symmetric, and the asymmetry is the point.** Melody is
given **tissue-matched Loyfer tracks** in every tissue. SilentMethyl is
**zero-shot**: breast-trained, breast epigenomic context, no target-tissue
information of any kind. Melody has target-tissue data; we have none. Every number
below has to be read through that.

Paired difference, **fusion minus melody** (negative = Melody better):

    tissue                 n_sig   AUROC (distance-matched)          signed rho (significant)
    Lung                    2241   -0.0140 [-0.0343,+0.0052] ns      -0.0606 [-0.1094,-0.0178] MELODY
    ColonTransverse         2118   -0.0080 [-0.0342,+0.0147] ns      -0.0947 [-0.1528,-0.0406] MELODY
    Ovary                   1812   -0.0087 [-0.0368,+0.0144] ns      -0.0019 [-0.0673,+0.0570] ns
    Prostate                 825   -0.0314 [-0.0679,+0.0017] ns      -0.1268 [-0.2244,-0.0458] MELODY
    WholeBlood               183   -0.1085 [-0.1649,-0.0511] MELODY  -0.0033 [-0.1850,+0.1447] ns
    KidneyCortex             157   -0.0301 [-0.0951,+0.0367] ns      +0.2030 [+0.0166,+0.3778] OURS
    BreastMammaryTissue      154   -0.0123 [-0.1016,+0.0663] ns      -0.0592 [-0.2439,+0.0955] ns
    MuscleSkeletal           128   -0.0509 [-0.1432,+0.0369] ns      -0.1356 [-0.3829,+0.1030] ns

**Finding 1 — distance-matched AUROC is a tie in 7 of 8, with a consistent small
tilt toward Melody.** Only Whole Blood separates. But all eight point estimates
are negative (sign test p = 0.0078; median gap -0.013 to -0.030). The tissues are
not independent — overlapping CpGs and shared genome structure — so that p is
anti-conservative and should be reported as a *direction*, not a test. The honest
statement: **a zero-shot breast model is statistically indistinguishable from a
tissue-matched 39-cell-type model on distance-matched meQTL discrimination in
seven of eight tissues, while sitting consistently ~0.02 AUROC behind.**

**Finding 2 — effect-size correlation goes to Melody, as predicted.** Three
tissues favour Melody at interval level (Lung, Colon, Prostate), four tie, and
Kidney Cortex favours us (+0.2030 [+0.0166,+0.3778], n = 157 — small cohort, treat
as an observation not a claim). 7 of 8 signs favour Melody. This is their home
metric; the result is expected and should be stated as expected.

**Finding 3 — the Lung tail result did not replicate. This kills the "our tail
beats theirs" framing.** Counting tissue x threshold cells where a model's tail
enrichment excludes zero against the *distance-only* baseline:

    SilentMethyl (fusion):  7 cells  (Lung 0.1/0.5/1%, Colon 1%, Prostate 0.1/0.5/1%)
                            plus one cell where it is significantly WORSE than
                            distance (MuscleSkeletal top 5%, -0.0030 [-0.0067,-0.0002])
    Melody:                11 cells  (Lung 0.5/1%, Breast 0.5%, Colon 1%,
                                      Muscle 0.5%, Prostate 0.1/0.5/1%,
                                      WholeBlood 0.5/1/5%)

Lung was simply the first tissue run and happened to be the best case for us. The
paired fusion-minus-melody tail differences are indistinguishable in most cells and
favour Melody where they separate (Breast top 1% -0.0146, Kidney top 1% -0.0106,
WholeBlood 0.5/1/5%, Muscle top 5%). **Do not write a tail-superiority claim.**

**Finding 4 — Whole Blood is the one clear loss, and it is interpretable.** Melody
averages *five* Loyfer blood tracks there (B, T-CD3, NK, monocytes, granulocytes),
the deepest and most replicated part of the atlas, and it is the tissue where
tissue-matching plausibly buys the most. It is also the tissue where our breast
context is least applicable. Worth stating rather than hiding: it bounds how far
the parity claim generalises.

**What this does and does not license.**
- Licensed: "tissue-matching buys less than expected" — the head-to-head magnitude
  (~0.02 AUROC) is the same order as the +0.027 matched-track advantage recovered
  from their own Source Data in §3b. Two independent routes to the same number is
  the strongest thing in this document.
- Licensed: our evaluation design is the contribution. Both models were run through
  it; neither had been evaluated this way before.
- **Not licensed:** any claim that SilentMethyl outperforms Melody on meQTLs.
- **Not licensed:** "we trained a better single-nucleotide methylation predictor."
  Nothing here tests methylation *levels* head-to-head. That needs the bidirectional
  levels comparison, not this run.

**Caveat that cuts against us, not for us.** Melody contributes one released
checkpoint; our reference is three seeds averaged. Seed averaging *reduces* noise
in our scores, so it should have helped us. The consistent tilt toward Melody is
therefore not explained by the seed asymmetry.

## 4. The gift — Ovary

Melody, independently: *"Ovary data perform poorly, likely due to either (i)
lower data quality in ovary methylation tracks or (ii) ovary-specific motifs
being underrepresented in available meQTL datasets."*

Our nine-tissue result: **Ovary is the only tissue with a negative tail
difference** (−0.0208 [−0.1633, +0.1857]), despite being third by cohort size
(1,812 significant pairs) and having a perfectly healthy AUROC (0.5864
[0.5653, 0.6100]).

Confirmed directly in the head-to-head (§3c): run on *our* ovary cohort,
**Melody also fails to clear the distance baseline at every threshold**
(top 0.1% +0.0417 [-0.0833,+0.2283]; 0.5% +0.0124; 1% +0.0291; 5% -0.0083 — all
spanning zero), and the paired fusion-minus-melody differences are indistinguishable
throughout. Ovary is the tissue where the two models agree most completely, and
what they agree on is that neither adds anything to distance.

Two different architectures, different training data, different metrics, same
anomaly. That is strong evidence the ovary result is a property of the eGTEx
ovary data rather than of our model — and it lets us cite Melody as
corroboration rather than only as competition. Use this.

It also reinforces the sex argument: ovary underperforms in both papers while
prostate (male-only) is among our best, so donor sex is not the explanation in
either.

## 5. Melody-G and the "unseen tissue" claim

Melody-G predicts methylation in five held-out cell types, achieving AUC 0.663
(G1) and 0.697 (G2) against a 0.633 mean-profile baseline — improvements of
4.7% and 10.1%.

The distinction to draw, precisely: **Melody-G requires scRNA-seq from the
target cell type.** It is conditioned transfer. Our nine-tissue result is
**zero-shot** — no target-tissue data of any kind, no tissue label, no
conditioning. Those answer different questions, and ours is the one relevant to
a tissue with no molecular data available.

Note also that Melody-G's gains are on *methylation prediction*, not meQTL
effect prediction, in unseen types. Nobody has yet shown zero-shot meQTL
discrimination across tissues under a distance-matched null. That is our claim.

## 6. What R3 should say

Frame it as: a breast-trained model, given no information about the target
tissue, discriminates real meQTLs from distance-matched non-meQTLs in 7 of 9
tissues, and does so no better on breast (AUROC 0.6141) than on lung (0.5880),
colon (0.6051), or kidney (0.6062).

Then, per §3c, the head-to-head: a tissue-matched Melody is indistinguishable from
this zero-shot model in 7 of 8 tissues. R3 must not claim we beat Melody.

The "no better on its own tissue" observation is the one that is genuinely ours
and genuinely surprising — and it sits in productive tension with Melody's
Fig 3H, where tissue-matched tracks *do* win. Worth stating that tension
explicitly rather than hiding it: their models are trained per tissue and ours
is not, so the comparison localises where tissue-specificity actually lives.

## 7. Availability to check

- Zenodo record **21386471** — Melody checkpoints. If the weights and the meQTL
  benchmark are public, a head-to-head on our distance-matched cohorts is
  inference-only and cheap. This would be the single strongest addition.
- Melody web server: https://inner.wei-group.net/Melody/
- Their meQTL benchmark sources: Ólafur et al., GTEx, EPIGEN.

## 8. Shared test split

Melody uses chr10 for validation and **chr8 + chr9 for test** — identical to
ours. Say so in Methods. It makes any future head-to-head directly comparable
and pre-empts the "different splits" objection.
