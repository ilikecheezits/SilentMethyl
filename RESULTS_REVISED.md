# SilentMethyl — Results, R1–R7, revised for the breast-epithelium context

Written 12 Sep 2026. Follows the R1–R7 framework in LAB_NOTES §1.2. Values from
`results/journal/ablation_breast_epithelium/` where the rerun is complete.

Markers used throughout:

- **[FILL]** — the value exists on disk and needs reading out. Not guessed.
- **[STALE]** — the number below is from the MCF-10A configuration and has not
  yet been recomputed. Structurally unchanged; the value will move.
Every value below was read from a `metrics.json`, `run_summary.json` or
`decision.json` on disk. Nothing is reconstructed from memory.

The earlier context configuration is an internal record (LAB_NOTES §1.10), not a
result. It appears nowhere below.

---

## The argument, in four steps

1. CpG–variant **distance alone** reaches AUROC ~0.595. A benchmark that does
   not control for it is measuring distance.
2. Chromatin context substantially improves methylation **level** prediction and
   is close to inert for variant **effects** — shown by ablation and causally by
   permutation.
3. The reason is architectural. The context vector is identical for the
   reference and alternate alleles, so it can rescale a predicted effect through
   the gate but cannot create one or set its direction. This also explains why a
   breast-trained model transfers to other tissues: there is little tissue-specific
   information in the variant response to lose.
4. Melody, a different architecture, is subject to the same limit, and its
   published benchmark contains no negatives.

R2 and R3 are one argument, not two. Say so.

---

## R1 — A gated fusion model predicts CpG methylation

*Does combining DNABERT-2 sequence with epigenomic context beat sequence alone,
and beat published architectures?*

Scripts `10/11/12_train_*`, `13_test_model`, `14_baselines_simple`,
`15_baselines_published`, `16_paired_model_bootstrap`.

### Three arms, three seeds, published split (test chr8+chr9)

| arm | β MAE (s42 / s43 / s44) | mean | AUC (s42 / s43 / s44) | mean |
|---|---|---|---|---|
| CpGenie | 0.1296 / 0.1275 / 0.1271 | 0.1281 | 0.9408 / 0.9411 / 0.9400 | 0.9406 |
| DeepCpG | 0.1258 / 0.1249 / 0.1252 | 0.1253 | 0.9435 / 0.9427 / 0.9449 | 0.9437 |
| context-only | 0.1022 / 0.1022 / 0.1023 | 0.1022 | 0.9658 ×3 | 0.9658 |
| sequence-only | 0.1090 / 0.1103 / 0.1104 | 0.1099 | 0.9576 / 0.9568 / 0.9562 | 0.9569 |
| **gated fusion** | 0.0885 / 0.0901 / 0.0956 | **0.0914** | 0.9765 / 0.9751 / 0.9717 | **0.9744** |

Sequence-only, CpGenie and DeepCpG are unchanged by the context swap — they read
DNA only. CpGenie n = 2,006,658 parameters, DeepCpG n = 4,102,402; both tuned by
validation β MAE, 10 epochs, 1 kb window.

**Fusion minus sequence-only, published split:** β MAE −0.0185 (−0.0205 /
−0.0202 / −0.0148), AUC +0.0176 (+0.0189 / +0.0183 / +0.0155).

**Fusion beats both published architectures on every seed**, by 0.034–0.037 β
MAE against CpGenie and 0.031–0.034 against DeepCpG. Note that the *context-only*
arm (0.1022) also beats both, and beats sequence-only — worth stating, because it
is the cleanest evidence that the chromatin features carry real information about
methylation level, which is what R2 then shows does **not** extend to variants.

**Have the answer ready for the context arm's seed invariance.** 0.1022 / 0.1022
/ 0.1023 looks like a frozen run. It is not — a small MLP over nine smooth
features has almost no seed sensitivity, and the same pattern held in the
previous configuration. State it before a reviewer asks.

### Repeated chromosome-blocked folds — the gain reproduces on every split

Fusion retrained per fold; each fold's sequence tower is its own and unchanged.
All values read from `metrics.json`; single seed (42) throughout.

| fold | test chromosomes | n | fusion β MAE | seq β MAE | Δ β MAE | fusion AUC | seq AUC | Δ AUC |
|---|---|---|---|---|---|---|---|---|
| 0 | chr8, chr9 | 26,570 | 0.0885 | 0.1090 | −0.0205 | 0.9765 | 0.9576 | +0.0189 |
| 1 | chr12, chr18 | 26,748 | 0.0873 | 0.1040 | −0.0167 | 0.9773 | 0.9619 | +0.0154 |
| 2 | chr20, chr4 | 26,783 | 0.0934 | 0.1064 | −0.0130 | 0.9724 | 0.9571 | +0.0153 |
| 3 | chr13, chr15, chr21 | 26,806 | 0.0862 | 0.1049 | −0.0187 | 0.9779 | 0.9614 | +0.0165 |

**The fusion-over-sequence gain is ~1.7 percentage points of β MAE**, mean
−0.0173 over all four splits, range −0.0130 to −0.0205, with AUC +0.0153 to
+0.0189. Every split agrees in sign on both metrics.

The published split (−0.0205) sits at the **strong end** of the range rather
than in the middle. The previous wording — "neither lucky nor unlucky" — no
longer holds and must be changed; say that the published split is the most
favourable of the four and that the gain reproduces on all of them.

Absolute performance still varies more than the gain does (0.0862–0.0934), the
expected consequence of chromosomes differing in gene density. The two most
gene-poor folds again give the lowest error, so the model is not carried by
promoter-dense regions. Held-out sizes are matched by construction
(26,570–26,806). Sex chromosomes stay in training for every fold and are never
evaluated — chrY is absent in female donors and chrX carries X-inactivation.

**Fix a known inconsistency while you are here.** LAB_NOTES §1.2's fold table
lists fold 0 as 0.0993 / 0.1099, which is a **three-seed mean**, while folds 1–3
are single-seed. `main_revised.tex` was corrected to the seed-42 values; the
notes were not. Make both single-seed.

### Paired bootstrap, fusion minus sequence-only

Cross-seed ensemble, 26,570 loci, 257 genomic blocks, 5,000 replicates, read
from `paired_model_difference_bootstrap.csv` (**not** `run_summary.json`, which
holds metadata only):

    roc_auc    +0.016643 [+0.014663, +0.018769]   P(diff>=0) = 1.0
    beta_mae   -0.018188 [-0.019616, -0.016756]   P(diff>=0) = 0.0
    beta_rmse  -0.032530 [-0.035247, -0.030062]
    m_mae      -0.156464 [-0.168563, -0.144415]
    m_rmse     -0.254229 [-0.279333, -0.232417]

All three seeds agree in sign with intervals excluding zero (β MAE −0.0204 /
−0.0203 / −0.0148 and AUROC +0.0189 / +0.0183 / +0.0155 for seeds 42 / 43 / 44).

**The gain over sequence-only grew with the better context**, from −0.0104 to
−0.0182 β MAE and from +0.0105 to +0.0166 AUROC.

**The comparison against the context-only arm inverted, and that is the
reframing to carry into the text.** Fusion minus context-only is now roc_auc
+0.009851 [+0.007679, +0.012420] and m_mae −0.159558 [−0.175944, −0.144258] —
about five-fold smaller on AUROC than the previous +0.0507 and 2.4-fold smaller
on m MAE than the previous −0.378, because the context arm itself improved so
much. On absolute β MAE the context-only arm (0.1022) now **beats** sequence-only
(0.1079), and on AUC as well (0.9658 against 0.9590); under the previous context
it did neither. That inversion survives the choice of estimator — mean-of-seeds
gives 0.1022 against 0.1099 — which matters because the absolute values quoted
in the ladder above are mean-of-seeds and the ones in this block are the
cross-seed ensemble. The two differ most for sequence-only (0.1099 against
0.1079) and are not in conflict; per LAB_NOTES §7B, quote the ensemble only as
the ensemble. Fusion now sits closer to its context arm (0.0125 β MAE) than to
its sequence arm (0.0182). The old line — "sequence carries the signal, context
modifies it" — no longer describes the level task and must not be carried over.

This is a statement about methylation **levels** only. It says nothing about
variant effects, where R2 finds the arms indistinguishable.

**How to write it.** The gain is consistent and significant, and now moderate
rather than small: roughly 1.7 points of β MAE. That is larger than the previous
1.0 and worth stating plainly, but it is still not "substantial" — a reviewer
will convert the AUROC delta into plain language whether or not you do, and
overreach here costs credibility on the parts that are strong.

### Classical sequence baselines

| model | features | α | β MAE | AUC | m Pearson | m Spearman |
|---|---|---|---|---|---|---|
| composition | 3 | 0.01 | 0.1954 | 0.8748 | 0.7025 | 0.6309 |
| k-mer ridge (k ≤ 6) | 2,772 | 1e5 | 0.1565 | 0.9190 | 0.7880 | 0.7265 |

Same splits, same target column, same 1 kb window, same target-CpG exclusions,
scored through `20_variant_scoring`. No published cross-tissue weights are
involved, so no tissue handicap favours either side. Both are exactly
RC-invariant by construction (max abs difference 0.0), which is the control for
the neural models' RC disagreement discussed in R5.

**The full ladder on β MAE**: composition 0.1954 → k-mer ridge 0.1565 →
CpGenie 0.1281 → DeepCpG 0.1253 → sequence-only 0.1099 → context-only 0.1022 →
**fusion 0.0914**. Each step is a real gain, and the neural sequence model beats
the k-mer ridge by 0.047 — so DNABERT-2 pretraining is doing work, which is the
question `sequence_baselines`' own interpretation field was written to answer.

**But k-mer ridge is the referent that matters for R2**, not for R1. The
interpretation recorded before the run: *if k-mer ridge matches the neural models
on variant effects, the finding is that sequence-only allelic methylation
prediction is much harder than reported, and DNABERT-2 pretraining is not what
closes the gap.* Its variant scores exist
(`sequence_baselines/variant_scoring/heldout/{composition,kmer_ridge}/seed-1/`,
66,495 pairs, all counters clean).

**The prediction does not fire on the primary metric.** GENOA, non-CpG-altering,
cross-seed ensemble, 1 Mb block bootstrap. The distance-only baseline is
0.5952795746 in every row of both runs, and the cohort sizes match exactly
(42,866 scored / 8,074 matched / 4,037 significant), so the two evaluations are
measuring the same pairs:

| model | AUROC within distance bin | AUROC distance-matched | signed rho | direction agr. |
|---|---|---|---|---|
| composition | 0.4669 [0.4543, 0.4803] | 0.4600 [0.4467, 0.4738] | −0.0377 [−0.0723, −0.0004] | 0.4835 [0.4659, 0.5013] |
| k-mer ridge | 0.5055 [0.4924, 0.5170] | 0.5027 [0.4879, 0.5177] | +0.1000 [+0.0592, +0.1326] | 0.5377 [0.5216, 0.5551] |
| sequence-only | 0.5745 [0.5591, 0.5898] | 0.5629 [0.5473, 0.5784] | +0.1497 [+0.1161, +0.1833] | 0.5541 [0.5362, 0.5728] |
| fusion | 0.5722 [0.5576, 0.5869] | 0.5559 [0.5392, 0.5720] | +0.1487 [+0.1148, +0.1842] | 0.5559 [0.5376, 0.5728] |

**k-mer ridge cannot discriminate meQTLs from distance-matched nulls at all.**
Within distance bin it sits at 0.5055 with the interval spanning 0.5, against
0.572–0.575 for both neural arms with intervals well clear of chance. The gap of
roughly 0.067 is about five times the ~0.012 run-to-run instability of this
metric, so it is readable despite being unpaired. Composition is *below* chance,
as a pure nucleotide-frequency model over a distance-matched cohort should be.
So DNABERT-2 pretraining is what buys distance-controlled discrimination, and the
recorded adverse interpretation is not triggered.

**State the other half honestly: on effect direction it is not clean.** k-mer
ridge reaches signed rho +0.1000 against the neural +0.149, and the intervals
overlap (+0.1326 against +0.1148); direction agreement overlaps likewise. So a
2,772-feature ridge recovers roughly two thirds of the neural rank correlation
among already-significant pairs and cannot be declared different on these
numbers. The claim the data supports is **discrimination, not effect direction** —
write it that way rather than reporting a general baseline win.

**Two provenance points, because both look like problems and are not.** These
two baselines are sequence-only and never read a context feature, so their
numbers are untouched by the context swap and are directly comparable to the
breast-epithelium arms. The `caveats.tissue` field in that `run_summary.json`
nonetheless reads "context features are MCF-10A breast": that is the hardcoded
caveat string recorded in LAB_NOTES §1.10, since fixed in
`21_variant_evaluation.py`, not a statement about how these models were built.

**What would sharpen it.** The comparison above is unpaired — the baselines were
evaluated 28 Aug at 500 replicates, the neural arms 12 Sep at 2,000. The
document's own rule is that only paired differences are reportable below ~0.012,
which is exactly where the signed-rho comparison sits. A paired run is cheap
(the score files exist) but needs a column check first: the baseline scores
predate the `beta_ref_to_alt` / `pvalue` inline-schema change, and mixing schemas
in one scores directory silently yields no per-model metrics for the older arm.

---

## R2 — The gain is in the baseline, not the variant response

*Does the context tower contribute to variant-effect prediction?*

Scripts `20_variant_scoring`, `21_variant_evaluation`,
`22_context_stratification`, `23_context_permutation`.

**Status: done, and causal rather than merely architectural.**

### Ablation — fusion minus sequence, paired, both cohorts

| metric | GENOA (cross-tissue) | eGTEx (tissue-matched) |
|---|---|---|
| signed rho | −0.0010 [−0.0078, +0.0051] | +0.0117 [−0.0085, +0.0320] |
| direction agreement | +0.0017 [−0.0050, +0.0079] | +0.0048 [−0.0169, +0.0234] |
| AUROC marginal | −0.0024 [−0.0060, +0.0009] | +0.0028 [−0.0118, +0.0143] |
| AUROC within distance bin | −0.0023 [−0.0058, +0.0010] | +0.0057 [−0.0094, +0.0176] |

Eight intervals, all spanning zero, 2000 block-bootstrap resamples over 1 Mb
blocks. The arms are **statistically indistinguishable** — never "fusion is
worse"; the intervals overlap heavily.

**Report the paired statistic, not the marginal one.** The distance-matched
AUROC is rebuilt per evaluation and moves ~0.01 between runs of identical data —
the same model on the same score files gave 0.5781 and 0.5900 in two runs, and
the GENOA fusion ensemble gave 0.5645 then 0.5559. The paired differences were
bit-identical across those runs, being computed on fixed pairs rather than a
resampled cohort. **Any marginal difference below ~0.012 is unreadable.**

### Permutation — the causal test

76,893 pairs, identity / shuffle / median, prediction recorded before execution.
Construction counters clean: 76,893 scoreable, 0 reference-base mismatches, 0
CpG-altering, 0 window problems.

| scheme | quantity | MAE/SD | Spearman | sign agreement | Pearson |
|---|---|---|---|---|---|
| shuffle | levels | 0.526 | 0.618 | 0.752 | 0.632 |
| shuffle | deltas | 0.227 | 0.776 | 0.844 | 0.753 |
| median | levels | 0.353 | 0.943 | 0.943 | 0.953 |
| median | deltas | 0.186 | 0.805 | 0.836 | 0.862 |

**Normalised error is 2.3× (shuffle) and 1.9× (median) larger for levels than
for deltas.** The direction reproduces the previous configuration (2.7× and
2.5×) with a somewhat smaller margin, and the absolute normalised errors are
larger throughout — worth one sentence, not a paragraph.

**State the Pearson discrepancy against yourselves.** The pre-specified
`deltas_preserved_more` flag was implemented on **Pearson**, and on that metric
the median scheme still FAILS (deltas 0.862 < levels 0.953) while shuffle passes
(0.753 > 0.632). Report both. Pearson behaves differently because methylation
levels are bimodal with SD ~3.18 M-units, so a high Pearson is cheap on levels
and not comparable across the two quantities — which is why normalised MAE,
Spearman and sign agreement are the fairer reads. Present the discrepancy in the
text rather than selecting the three metrics that agree; a reviewer who
recomputes will find it.

### Why this matters beyond R2

It supplies the **mechanism for R3**. Variant effects are near-invariant to what
chromatin the model is shown, so a breast-context model transferring to lung is
not a surprise — it is the predicted consequence of allele invariance. R2 and R3
become one argument.

It also answers the multi-tissue demand **without retraining**: a random locus's
chromatin is further from the truth than another tissue's chromatin at the same
locus, so a null here implies a null for the tissue swap.

**Hard constraint.** Do not claim anywhere that epigenomic context improves
variant-effect prediction. It improves level prediction (R1) and is close to
inert for deltas (R2). Any sentence implying otherwise contradicts our own data.

---

## R3 — Zero-shot transfer across nine tissues

*Does a breast-trained model prioritise mQTLs in tissues it has never seen,
above a distance-matched null?*

Scripts `30_transfer_synthesis`, `31_transfer_discrimination`,
`32_transfer_summary`.

**[STALE]** — the nine-tissue scoring has not been rerun on the new context.
Given R2, it should barely move; rerun it before submission, but the conclusion
will not change.

Distance-matched AUROC, seed ensemble, distance-only pinned to exactly 0.5000 by
construction, CpG-altering variants excluded (~37% of pairs, enriched for true
mQTLs, so the estimate is conservative):

| tissue | n significant | fusion | sequence-only | excludes 0.5 |
|---|---|---|---|---|
| BreastMammaryTissue *(training tissue)* | 154 | 0.6141 [0.5325, 0.7003] | 0.6129 [0.5219, 0.7013] | yes |
| ColonTransverse | 2,118 | 0.6051 [0.5844, 0.6263] | 0.6056 [0.5839, 0.6283] | yes |
| KidneyCortex | 157 | 0.6062 [0.5404, 0.6701] | 0.6115 [0.5498, 0.6876] | yes |
| Lung | 2,241 | 0.5880 [0.5669, 0.6091] | 0.5869 [0.5658, 0.6094] | yes |
| MuscleSkeletal | 128 | 0.5872 [0.5055, 0.6783] | 0.5891 [0.5093, 0.6672] | yes |
| Ovary | 1,812 | 0.5864 [0.5653, 0.6100] | 0.5878 [0.5679, 0.6105] | yes |
| Prostate | 825 | 0.5789 [0.5520, 0.6071] | 0.5760 [0.5477, 0.6108] | yes |
| Testis | 94 | 0.5645 [0.4850, 0.6510] | 0.5592 [0.4827, 0.6385] | no |
| WholeBlood | 183 | 0.5607 [0.4996, 0.6283] | 0.5547 [0.4862, 0.6206] | no |

**Excludes 0.5 in 7 of 9.** The two that do not are the two smallest cohorts
(n = 94, 183), so this is power, not tissue.

Breast — the training tissue — at 0.6141 is indistinguishable from colon
(0.6051), kidney (0.6062) and lung (0.5880). **The model is not better on the
tissue it was trained on.**

**R2 replicates across all nine tissues.** Fusion and sequence-only are
indistinguishable in every one, with differences of 0.0005–0.0053 and sequence
ahead in four of the nine. Lung's paired difference is +0.0011 [−0.0031,
+0.0057], interval including zero. This is nine more independent confirmations
that the context tower does not contribute to variant effects, and it costs
nothing to say so.

### The Melody-ST control reframes the claim

Melody-ST-Breast — a single-tissue breast model of a completely different
architecture — transfers too, and beats fusion on distance-matched AUROC in four
of nine tissues (7 of 9 by sign). So the claim is **not** "our architecture
transfers". It is **"single-tissue methylation models transfer, and the training
tissue matters far less than assumed"** — a finding about the task,
demonstrated in two architectures sharing nothing but the task. Broader and
harder to attack. **Do not claim SilentMethyl transfers best; it does not.**

Where fusion does win, coherently: Testis (ST-Breast falls below chance, 0.4737
[0.3594, 0.5886], rho −0.0740; paired rho +0.2030 and direction agreement
+0.1170 both DIFFERENT), Kidney (paired rho +0.2364 DIFFERENT), and the Colon
extreme tail (+0.2500 at top 0.1%, +0.0962 at top 0.5%, both DIFFERENT). Fusion
holds up in the low-powered and hardest tissues where the U-Net degrades. Claim
that, not more.

### What tissue matching is worth, measured directly

ST-Lung vs ST-Breast on identical Lung rows — same architecture, same training
scale, same scoring code, only the training tissue differs — gives AUROC −0.0054
[−0.0241, +0.0103], not distinguishable, bounding the effect under ~0.024.
Matching helps only in the top 0.5% tail (+0.1157 [+0.0288, +0.2126]).

**Must include:** the Melody Fig 3H tension (their tissue-matched tracks *do*
win, because they train per tissue and we do not); the prostate-vs-ovary result
killing the donor-sex explanation; Melody's independent ovary anomaly as
corroboration.

---

## R4 — Where transfer fails, and why

*Can the model distinguish shared from tissue-specific mQTLs?*

Script `40_meqtl_tissue_specificity` (`--stage matched,chromatin`),
`41_tissue_specificity_summary`.

**The answer is NO**, and it is unchanged by the context swap — itself
informative.

| discovery → replication | n/arm | direction agreement Δ | signed rho Δ |
|---|---|---|---|
| GENOA → eGTEx | 769 | +0.0461 [+0.0128, +0.0801] | +0.0775 [−0.0018, +0.1578] |
| eGTEx → GENOA | 24 | +0.4130 [+0.2714, +0.5505] | +0.2824 [−0.2943, +0.8916] |

The pre-specified rule requires the same favoured class across metrics.
Direction agreement favours shared in both directions (2 of 2 excluding zero);
signed rho in neither (0 of 2). **The metrics disagree, so this is not a
finding.**

The tissue-specific arm is **negative** in the eGTEx→GENOA direction (−0.0615,
n = 24): the model anti-predicts those pairs. That is the **winner's-curse**
signature — pairs clearing significance in a low-powered cohort are enriched for
false positives — not chromatin mediation. Note the power asymmetry explicitly:
769 pairs per arm in one direction against 24 in the other.

### Melody fails the same test, in the mirrored direction

Melody-MT, 39 tissues, 14 tissue pairs with both ordered directions:

| metric | directions | excluding zero | favouring shared | favouring specific | pairs meeting the rule |
|---|---|---|---|---|---|
| direction agreement | 42 | 8 | 2 | **6** | 1 of 14 |
| signed rho | 42 | 5 | 1 | 4 | **0 of 14** |

**The two models fail in opposite directions** — SilentMethyl's informative
directions favour *shared*, Melody's favour *specific* — which is what locates
the problem in the benchmark rather than in either architecture.

**And Melody's is cleanly explained by power.** Its decision file reports
`cleanly_separated_by_n: true` with `low_power_call: "specific"`: the directions
calling *specific* have 24–304 pairs per arm (median 137) while those calling
*shared* have 593–692 (median 642). The favoured class is entirely determined by
which arm sat in the lower-powered stratum. Melody also has two negative shared
arms under signed rho (MuscleSkeletal→Lung −0.1046 at n = 20;
MuscleSkeletal→Ovary −0.0357 at n = 24).

Our own diagnostic reports `cleanly_separated_by_n: false` — the two directions
span 24 to 769 per arm and do not separate by call — so we cannot dismiss our own
result as power in the same clean way. What kills it for us is metric
disagreement plus the negative specific arm.

**Disclose that our diagnostic was one-sided before this comparison.** It checked
only whether the *specific* arm sat in the lower-powered stratum, so a low-powered
*shared* arm would have passed silently. It is two-sided now, and rerunning it
surfaced a previously invisible negative shared arm in the Melody data
(ColonTransverse→Lung, n = 132). Say so; it shows the diagnostic was fixed rather
than tuned.

*Why it matters:* this was the mentor's headline question and the only remaining
item that would have produced a **biological** rather than methodological
finding. Reporting the null honestly, with the mechanism identified, is the
result.

---

## R5 — Mechanism and limits

*What is the model responding to, and when should it not be trusted?*

Scripts `50_motif_disruption`, `51_rc_uncertainty`, `52_gwas_enrichment`.

### Motif disruption, k-mer-matched null

| class | n | n significant | AUROC |
|---|---|---|---|
| strong disruption | 11,525 | 1,103 | 0.5950 [0.5737, 0.6158] |
| weak disruption | 9,881 | 906 | 0.5954 [0.5739, 0.6185] |

Difference −0.0004, intervals overlapping: **no detected concentration**.

The coupling null, 831 motifs tested:

| | value |
|---|---|
| coupling median | −0.0035 (IQR −0.0448 to +0.0379) |
| coupling median, partial | −0.0032 |
| fraction negative | 0.519 |
| significant at q < 0.05 | 185 negative, 158 positive |
| Spearman(motif GC, coupling) | **−0.2091** |
| top-15 mean pairwise Jaccard | **0.278** (max 0.792) |
| known methylation-sensitive TFs | 18 tested, median rank **154** of 831 |

Three independent reasons this is not a factor-specific finding: the median sits
essentially at zero with 51.9% negative, so there is no global offset *and* no
consistent direction; the GC correlation of −0.21 says the apparent coupling
tracks sequence composition rather than factor identity; and the top-15 motifs
overlap heavily (max pairwise Jaccard 0.79), so they are one motif family rather
than 15 independent factors. The 18 known methylation-sensitive TFs rank at a
median of 154 out of 831 — indistinguishable from the middle of the list.

Consistent with the earlier k-mer control, which reproduced the ETS signal more
strongly than the motif analysis did.

### Reverse-complement disagreement as calibrated uncertainty

Nine model-seed combinations (epi / sequence / fusion × 3 seeds), 2000-replicate
block bootstrap, 19 coverage levels from 10% to 100%, four estimators against a
random baseline.

**The verdict is uniform and negative in the interesting direction.** For all
nine: `beats_random: true`, `beats_cross_seed_sd: false`,
`beats_boundary_distance: false`.

So RC disagreement **is** informative about error — it beats random selective
prediction everywhere — but it does **not** beat the two cheaper estimators: the
three-seed ensemble SD, and simple distance from the decision boundary
|β̂ − 0.5|. Report it that way. The honest claim is that RC disagreement is a
*free* uncertainty signal available from a single model without an ensemble, not
that it is the best one.

The conditional stage exists precisely to separate a genuine difficulty signal
from the mechanically bounded range of β near 0 and 1 — its own motivation field
says so. Detailed values in `partial_correlations.csv`,
`within_stratum_correlations.csv`, `incremental_value.csv`. Calibration must be
assessed in **logit space**.

One number now established: the sequence tower's RC disagreement (β MAE 0.0553,
Pearson 0.9619) is a property of the architecture, not of the context or the
training set — the joint multi-tissue towers reproduce it almost exactly
(0.0558 / 0.9630 and 0.0581 / 0.9594), while the context MLP over nine smooth
features sits at 0.0041 / 0.9996. A transformer reading 100 bp genuinely sees two
different token sequences. Do not present the sequence tower's value as a defect.

One number that is now established and should be stated: the sequence tower's RC
disagreement (β MAE 0.0553, Pearson 0.9619) is a property of the architecture,
not of the context or of the training set — the joint multi-tissue towers
reproduce it almost exactly (0.0558 / 0.9630 and 0.0581 / 0.9594). A transformer
reading 100 bp genuinely sees two different token sequences; the context MLP,
over nine smooth features, does not (0.0041 / 0.9996). Do not present the
sequence tower's value as a defect.

### GWAS regulatory enrichment — null, and pre-registered as such

**Verdict: null.** 1,548 GWAS-overlapping pairs matched 1:1 to a distance- and
allele-frequency-matched background (5 AF strata, 0 unmatched slots), 3,096 pairs
total, 500-replicate block bootstrap.

| | value |
|---|---|
| AUROC, GWAS vs matched background | 0.5158 [0.4926, 0.5439] — **includes 0.5** |
| mean abs Δ M, GWAS | 0.03761 |
| mean abs Δ M, background | 0.03607 |

Tail enrichment, GWAS share of the top-scoring fraction (0.5 = no enrichment):

| fraction | n in tail | GWAS share | 95% CI | excludes 0.5 |
|---|---|---|---|---|
| 5% | 155 | 0.4710 | [0.3692, 0.5563] | no |
| 2% | 62 | 0.5161 | [0.3729, 0.7045] | no |
| 1% | 31 | 0.4839 | [0.2500, 0.7329] | no |

Matching quality is exact on distance (both means 254.02, standardised
difference 0.0; median distance 253 in both arms; distance-only AUROC after
matching = 0.5000) and near-exact on allele frequency (0.2034 vs 0.1952,
standardised difference 0.058). `unchecked_confounds` is empty.

**Write this as a clean pre-registered null.** The preregistration file is in the
same directory and the verdict field was written by the script, not chosen
afterwards. A null with this quality of matching is more informative than a
weakly positive result with none.

---

## R6 — Application: variant prioritisation

Scripts `60`–`64`. STK11 and NCOA2 retained per mentor instruction.

**[STALE], and blocked.** The `60 → 62 → 22 → 91` chain has not been rerun: `22`
and `91` hard-require the candidate CSVs that `60` produces, and `60`/`62` are
GPU jobs of their own.

**`64_literature_variant_screen.py` — fixed 12 Sep 2026.** It shelled out to `63`
without forwarding `--weights-template`, so it would have silently scored the
previous checkpoints and reported a result that looked correct. It now takes a
`--weights-template` of its own and forwards it; the default is unchanged, so
existing published-checkpoint invocations behave exactly as before. Scoring the
ablation requires setting **both** `--weights-template` and `--split-template`.

One blocker remains before `60` can run against the new context, and it is not a
scheduling problem. `60` reads its context features as columns of the candidate
CSV built by `data/build_testing_data.py`, and that script resolves the seven
bigWigs from a hardcoded `<data-dir>/reference` with no `--reference-dir`
override — the separation that `build_training_data.py` already has. So there is
currently no breast-epithelium candidate cohort to score, and pointing `60` at
the existing one would pair new fusion weights with old context columns and
produce a hybrid that is neither model. Give `build_testing_data.py` the same
`--reference-dir` / `--out-dir` pair, rebuild the cohort against
`data/reference/BreastEpithelium`, and reuse the existing GDC response cache so
the candidate variant set stays byte-identical.

On NCOA2, keep the existing framing: its active chromatin context is an
independent annotation of that locus and **not a contributor to the predicted
effect** — required by R2's hard constraint.

---

## R7 — Cross-cohort, cross-ancestry, cross-platform replication

*Does the variant-effect signal survive a genuinely independent cohort?*

Scripts `data/harmonize_genoa_meqtl.py`, `20_variant_scoring` (GENOA),
`21_variant_evaluation`, `30_transfer_synthesis`.

Same design as the nine-tissue work: distance-matched negatives at 10 bp
tolerance, 1 Mb block bootstrap, distance-only baseline reported alongside
(~0.595), a null stratum that reaches chance.

### GENOA, current values

| metric | fusion | sequence |
|---|---|---|
| signed rho (p < 5e-8) | +0.1487 [+0.1148, +0.1842] | +0.1497 [+0.1161, +0.1833] |
| direction agreement | +0.5559 [+0.5376, +0.5728] | +0.5541 [+0.5362, +0.5728] |
| AUROC marginal | +0.6002 [+0.5862, +0.6150] | +0.6026 [+0.5883, +0.6172] |
| AUROC within distance bin | +0.5722 [+0.5576, +0.5869] | +0.5745 [+0.5591, +0.5898] |
| AUROC distance-matched | +0.5559 [+0.5392, +0.5720] | +0.5629 [+0.5473, +0.5784] |
| **AUROC, distance alone** | **+0.5953 [+0.5822, +0.6099]** | |

Marginal AUROC does not exceed the distance-only baseline. Signal appears only
after matching.

### Dilution gradient — GENOA reaches chance in the null stratum

| stratum | n | signed rho | direction agreement |
|---|---|---|---|
| p < 5e-8 | 4,037 | +0.1487 [+0.112, +0.183] | 0.5559 [0.537, 0.574] |
| 5e-8 – 1e-5 | 1,319 | +0.0886 [+0.031, +0.144] | 0.5406 [0.512, 0.569] |
| 1e-5 – 1e-3 | 2,347 | +0.0962 [+0.052, +0.141] | 0.5467 [0.526, 0.568] |
| 1e-3 – 1e-2 | 2,392 | +0.0900 [+0.047, +0.135] | 0.5326 [0.511, 0.557] |
| 1e-2 – 0.05 | 3,550 | +0.0723 [+0.034, +0.111] | 0.5313 [0.513, 0.550] |
| 0.05 – 0.5 | 15,891 | +0.0222 [+0.005, +0.039] | 0.5085 [0.501, 0.516] |
| **p > 0.5** | 13,330 | **−0.0006 [−0.017, +0.016]** | **0.4980 [0.490, 0.507]** |

**eGTEx does not.** `p > 0.5` gives +0.0176 [+0.005, +0.030] and `0.05–0.5`
gives +0.0304 [+0.017, +0.043]. State it. The interpretation is power: eGTEx has
418 significant pairs of 47,991 against GENOA's 4,037 of 42,866, so its
non-significant strata contain real but undetected effects and are not a true
null. **GENOA carries the dilution argument**; eGTEx is the tissue-matched
replication of discrimination.

### Meta-analysis over independent LD blocks

Rerun 12 Sep on the new context, GENOA (132,990 rows, p < 5e-8) and eGTEx
(153,786 rows, p < 1.483e-5):

    fusion    rho_meta 0.1723 [0.0624, 0.2780]  p = 0.0022  I2 = 0.0  Q p = 0.479  k = 2

Previous configuration, for reference: fusion 0.1775 [0.0678, 0.2829], sequence
0.1698 [0.0599, 0.2757]. The point estimate is essentially unchanged.

**The sequence arm is missing from this rerun** — the synthesis ran at 16:53,
before the sequence rescore completed, so only fusion is in `meta_analysis`.
Rerun `30_transfer_synthesis` now that both arms have current scores; it is CPU
and takes minutes. Until then the fusion-vs-sequence equivalence in R7 rests on
the previous values.

    eGTEx conditional   breast, European-dominant      81  rho 0.610 [0.452, 0.731]
    eGTEx regular       breast, European-dominant     418  rho 0.246 [0.114, 0.402]
    GENOA               whole blood, African American 4037  rho 0.152 [0.117, 0.188]
    GENOA null control  tested but null pairs            -  rho -0.004 [-0.021, +0.013]

**I² = 0.** No detectable heterogeneity between cohorts differing in tissue,
platform AND ancestry. Weighting is by independent LD blocks, not pair counts —
keep the run summary's sentence that pair-count weighting "would understate the
variance by roughly an order of magnitude and manufacture significance".
Self-imposed conservatism is worth more than the extra stars.

Two things it confirms without being designed to: fusion and sequence are
indistinguishable, so **R2 replicates in an independent cohort**; and GENOA is
blood scored with breast context, so **R3 holds cross-cohort**.

### The ancestry claim, stated exactly

GENOA is African American and eGTEx European-dominant, so this is the
cross-ancestry comparison — but GENOA also changes tissue and platform, so
ancestry cannot be isolated.

**A within-cohort ancestry contrast was attempted and is not possible.** GDC open
ancestry calls (CCG-AIM-2020, no controlled access) on the TCGA-BRCA training
normals give **84 EUR, 4 AFR, 1 SAS, 8 unlabelled of 97 donors**. Four donors
cannot support an error estimate and `18_ancestry_stratified_error.py` refuses to
compute one. Write the limitation with those numbers in it — it shows the
alternative was checked rather than overlooked. Variant-effect stratification by
ancestry would additionally need per-donor genotypes, which are
controlled-access and deliberately not used.

R3 narrows this by one dimension: it varies tissue alone across nine
European-dominant eGTEx cohorts, so tissue is measured on its own and GENOA's
marginal contribution is ancestry plus platform.

---

## Superseded — remove in one coordinated edit

`70_mqtl_positive_control` (81 pairs) and `71_mqtl_matched_negative` (35 pairs),
together with Supplementary S2/S3. The text still cites them.

---

## Writing guardrails

1. **Never claim context improves variant-effect prediction.** R1 yes, R2 no.
2. **Never claim better methylation-level prediction than Melody.** It was never
   tested, and the defence against being asked for it depends on not having
   claimed it. Grep both drafts.
3. **Never claim SilentMethyl transfers best.** Melody-ST beats it in 4 of 9.
4. **"Statistically indistinguishable"**, never "fusion is worse".
5. **Paired statistics, not marginal AUROCs**, for fusion-vs-sequence. The
   ~0.012 run-to-run noise floor is the reason.
6. **The permutation result is a ratio, not a null.** 0.23 SD is real.
7. **Report the Pearson discrepancy in R2** rather than the three metrics that
   agree.
8. eGTEx's null stratum, R4's n = 24 arm, and the ancestry n = 4 are the three
   places a careful reader will push. All are stated; keep them stated.
