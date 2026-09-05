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
Their 10-kb window and 39-tissue training will win that metric. Compete on
discrimination under a distance-matched null, where they have no result at all.

## 4. The gift — Ovary

Melody, independently: *"Ovary data perform poorly, likely due to either (i)
lower data quality in ovary methylation tracks or (ii) ovary-specific motifs
being underrepresented in available meQTL datasets."*

Our nine-tissue result: **Ovary is the only tissue with a negative tail
difference** (−0.0208 [−0.1633, +0.1857]), despite being third by cohort size
(1,812 significant pairs) and having a perfectly healthy AUROC (0.5864
[0.5653, 0.6100]).

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
