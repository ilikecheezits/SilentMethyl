# SilentMethyl — requirements tracker

Living record of (A) the seven strengthening requirements from mentor feedback and
(B) the stated criteria of each candidate journal. Update the **Status** and
**Evidence** columns as work lands. Evidence should point at a path under
`results/` or `data/external/`, never at a claim.

Status vocabulary: `not-started` · `in-progress` · `done` · `n/a`

Plan: Stage A (data, no GPU) → Stage B (inference on frozen checkpoints) →
Stage C (one training pass) → Stage D (re-score + write).

---

## A. Mentor's seven requirements

| # | Requirement | Needs retraining? | Stage | Status | Evidence |
|---|---|---|---|---|---|
| 1 | Multi-cohort testing | No — inference | B.2 | not-started | |
| 2 | Repeated chromosome-blocked splits | **Yes — the only one** | C.1 | not-started | |
| 3 | Stronger baselines and ablations | No / cheap re-heads | A.2, B.5, C.2–C.3 | not-started | |
| 4 | Uncertainty calibration | No — post-hoc | B.3 | **done (analysis)** | `results/journal/rc_uncertainty{,_conditional}/` |
| 5 | Ancestry analyses | No — analysis | B.1 | in-progress | `results/journal/genoa_variant_evaluation/` (AFR arm) |
| 6 | Independent variant evaluation | No — inference | B.1 | **done** | `results/journal/genoa_variant_{scoring,evaluation}/` |
| 7 | Regulatory enrichment | No — inference | B.4 | not-started | |

### Detail

**1. Multi-cohort testing.** eGTEx 9 tissues; public GEO breast EPIC/450K series;
TCGA-BRCA tumours as a shifted domain; ENCODE/Roadmap WGBS. Zero-shot scoring of
existing checkpoints. Include the head-to-head against DeepMethylation's published
numbers (avg AUROC 0.909, EPIC R² 0.58, genome-wide AUROC 0.618, WGBS R² 0.17–0.20).

**2. Repeated chromosome-blocked splits.** Design: 4 additional folds at seed 42
only, combined with the existing 3-seed chr8–9 fold. Report seed SD from fold 1
(n=3) and fold SD across 5 folds (n=5). 12 runs, not 45.
*Known limitation:* no fold × seed interaction estimate. State this explicitly in
the methods rather than letting a reviewer find it.

**3. Stronger baselines and ablations.** Three layers:
- L1 competitors on our data — CpGenie (`gifford-lab/CpGenie`), DeepCpG (`PMBio/deepcpg`)
- L2 our model on their data — DeepMethylation's 9-tissue EPIC/WGBS setup
- L3 neutral ground — BEND (`frederikkemarin/BEND`), 7 tasks, 13 models pre-scored
Plus CPU baselines: CpG density, GC, k-mers, manifest annotation, and
**neighbouring-probe methylation** (currently absent; strongest known predictor).
Ablations: window at 400 / 1000 / 2000 bp only; all fusion-mechanism and
feature-group variants are free re-heads on frozen towers.

**4. Uncertainty calibration.** Temperature scaling on validation; conformal
prediction for distribution-free intervals on Δβ̂; ECE, Brier, PICP, CRPS.
Deep ensembles and a heteroscedastic head were dropped — both need retraining and
conformal gives a coverage guarantee without either.

*Result, script 16 (2026-08-24, `results/journal/rc_uncertainty/`).*
Spearman(FWD–RC disagreement, |error|) = **0.50** for sequence and fusion
(0.35–0.39 for epi), consistent across all 3 seeds, block-bootstrap CIs tight.
So the disagreement genuinely tracks local error.

But on selective prediction the ranking was:
`boundary_distance` < `combined` < `cross_seed_sd` < `rc_disagreement` << `random`.
The zero-parameter control **−|β̂ − 0.5| beat everything in 8/9 runs, including
the 3-seed ensemble SD.** RC disagreement beat random 9/9 but lost to cross-seed
SD 9/9 and to the heuristic 9/9.

*Resolved, script 17 (2026-08-24, `results/journal/rc_uncertainty_conditional/`).*
**The heuristic's advantage was the bounded-range artifact.** Evidence:

| | β error (bounded) | M error (unbounded) |
|---|---|---|
| boundary within-stratum ρ | +0.1569 | **+0.0461** |
| cross_seed_sd within-stratum ρ | +0.1381 | +0.1179 |
| rc_disagreement within-stratum ρ | +0.1069 | +0.0997 |
| cross_seed_sd beats boundary (stratified AURC) | 5/9 | **9/9** |
| rc_disagreement beats boundary | 0/9 | **9/9** |

boundary_distance's within-stratum correlation collapses 3.4× when the target is
moved to the unbounded logit (M) scale, while both real estimators hold. On M,
both beat the heuristic 9/9 in stratified selection and add incremental value
(cross_seed_sd 6/9, rc_disagreement 7/9). Absolute β error is mechanically
compressed near 0 and 1 — the heuristic was measuring headroom, not difficulty.

**Decisions that follow:**
1. Report uncertainty on **M-value error**, not β. State why explicitly.
2. Expected ordering holds on M: cross_seed_sd > rc_disagreement > boundary.
   Uncertainty estimation works; it was the β metric that was confounded.
3. **RC disagreement retains ~75% of the ensemble's incremental information**
   (partial ρ +0.0936 vs +0.1238) from a SINGLE model at zero extra cost.
   Report as a cheap single-model alternative, not as an ensemble replacement.
4. Conformal intervals still carry the section — coverage holds regardless.

**Spin-off contribution (free, worth a paragraph + supplementary figure):**
*Uncertainty evaluation on bounded bimodal targets is confounded by range
compression.* The field routinely reports β MAE; any model that is merely
confident at extreme β̂ will look well-calibrated on β and be exposed on M. We
recommend logit-scale evaluation. This generalises beyond SilentMethyl and costs
nothing to state.

*Robustness confirmed, script 17 strata sweep + script 18 figure
(`results/journal/rc_uncertainty_conditional_s{20,50}/`, `rc_uncertainty_figure/`).*
The 10-decile β ambiguity WAS under-stratification. Within-stratum ρ, pooled:

| estimator | β s=10 | β s=20 | β s=50 | M s=10 | M s=20 | M s=50 |
|---|---|---|---|---|---|---|
| cross_seed_sd | +0.1381 | +0.1049 | +0.0915 | +0.1179 | +0.1020 | +0.0952 |
| rc_disagreement | +0.1069 | +0.0789 | +0.0678 | +0.0997 | +0.0859 | +0.0793 |
| boundary_distance | +0.1569 | +0.0852 | **+0.0369** | +0.0461 | +0.0217 | **+0.0099** |

At 50 strata both real estimators beat the heuristic on **both** scales
(β: cross_seed_sd 9/9, rc_disagreement 8/9; M: 9/9 and 9/9). Requirement 4 is
closed — no outstanding checks.

**The crisp diagnostic to state in the paper:** a genuine uncertainty signal ranks
error about equally well on either scale. cross_seed_sd (+0.0915 β / +0.0952 M) and
rc_disagreement (+0.0678 / +0.0793) are scale-stable; boundary_distance decays to
near zero (+0.0369 / +0.0099). Scale-instability is the signature of a metric
artifact. Figure: `rc_uncertainty_figure/uncertainty_scale_stability.{png,pdf}`.

RC disagreement retains **~75–83%** of the ensemble's signal depending on measure
(partial ρ +0.0936 vs +0.1238 = 76%; within-stratum at s=50, 0.0793/0.0952 = 83%)
from a single model at zero extra cost.

**5. Ancestry analyses.** Cross-ancestry replication from published summary
statistics. GoDMC, the 2024 East Asian/European study and the gnomAD audit were all
dropped — GoDMC and gnomAD bought less than they cost in GB, and the mQTL summary
statistics already carry ancestry-matched allele frequencies (`af_genoa`) while the
HM450 manifest carries `MASK_snp5_common` / `MASK_snp5_GMAF1p`.

**The AFR arm is done** (see §6): GENOA is African American, and the requirement-6
result above *is* the African American replication — 4,037 genome-wide significant
pairs, ρ = +0.152, distance-matched AUROC 0.570. The comparison arm is eGTEx
(European-dominant), already in `results/journal/egtex_mqtl_positive_control/`.

**TCGA is not powered and must not carry this requirement.** GDC open ancestry calls
joined to the 97 training normals give EUR 84, AFR 4, SAS 1, unlabelled 8
(`data/external/tcga_ancestry/ancestry_summary.json`). Any per-group error estimate
on n=4 is noise. Report the group sizes, state plainly which strata are powered, and
rest the claim on GENOA vs eGTEx.

*Remaining:* the formal GENOA-vs-eGTEx contrast on a common metric, with the caveat
that the two differ in tissue as well as ancestry, so the comparison is not a clean
ancestry contrast and should not be presented as one.

**6. Independent variant evaluation.** From n=81 to millions. Report signed ρ,
direction agreement, AUROC, magnitude correlation — stratified by CpG-alteration
status, distance bin, effect decile, cohort ancestry.

*Cohort built, scripts 19 + `data/build_genoa_scoring_input.py` (2026-08-25).*
GENOA (Shang et al., *Nat Commun* 2023; Zenodo 10.5281/zenodo.7697509), African
American blood cohort, harmonised to hg38. 5.3 GB raw → 941,455 pairs → **66,495
held-out pairs, 39,657 unique variants, 19,081 of 26,570 test probes**. Allele
mismatch 0.1% (731/942,186), which certifies the liftover. CpGs were never lifted:
probe IDs are platform-stable and join the hg38 manifest directly.

**Effect-allele hazard, fixed.** GENOA is GEMMA output, so `beta` is keyed to the
minor allele; the model's Δ is keyed to hg38 REF→ALT. **11.3% of pairs (106,628)
have REF as the minor allele**, and for those the two run opposite. Always compare
against `beta_genoa_ref_to_alt`. Raw `beta_genoa` mixes conventions, does not
raise, and silently deflates both headline metrics.

*Result, script 20 (2026-08-25, `results/journal/genoa_variant_evaluation/`).*
Seed ensemble, non-CpG-altering, 500 block-bootstrap resamples over 1 Mb blocks.

| metric | fusion | sequence |
|---|---|---|
| signed ρ, p<5e-8 | **+0.1523** [+0.117, +0.188] | +0.1497 [+0.109, +0.184] |
| direction agreement | **0.5534** [0.536, 0.570] | 0.5541 [0.538, 0.573] |
| AUROC, distance-matched | **0.5700** [0.554, 0.585] | 0.5623 [0.546, 0.578] |
| AUROC, marginal | 0.6002 [0.586, 0.614] | 0.6026 [0.589, 0.617] |
| AUROC, distance alone | 0.5953 [0.583, 0.609] | — |

**Report the distance-matched AUROC, not the marginal one.** Significant meQTLs sit
closer to their CpG (median 191 vs 258 bp), |Δ| is larger for nearer variants
(ρ = −0.29), and distance *alone* classifies at 0.5953 — statistically
indistinguishable from the model's marginal 0.6002. State that explicitly and then
show the matched result: the signal is orthogonal to distance, not absent.
Volunteering the baseline is far stronger than having it extracted in review.

**Dilution gradient — the evidence the signal is real.** GENOA ships every cis pair
tested, not the meQTLs discovered; 71% of held-out non-CpG-altering pairs have
p > 0.05, which is why the pooled ρ is only 0.056.

| stratum | n | signed ρ | direction |
|---|---|---|---|
| p < 5e-8 | 4,037 | +0.1523 [+0.117, +0.184] | 0.5534 |
| 0.05 – 0.5 | 15,891 | +0.0232 [+0.006, +0.040] | 0.5084 |
| **p > 0.5** | 13,330 | **−0.0037 [−0.020, +0.014]** | **0.4966** |

The null stratum is indistinguishable from zero on both metrics — a negative
control inside the same data. Middle strata (0.085 / 0.099 / 0.097) are not
strictly monotone and their CIs overlap; the meaningful contrast is significant
versus null. Figure: `plots/significance_gradient.{png,pdf}`.

**Honest framing.** ρ ≈ 0.15 and 55% direction agreement is a *weak* predictor.
What makes it publishable is scale, independence and rigour — 4,037 genome-wide
significant pairs from a different tissue, ancestry and platform generation, on
probes the model never saw, against the previous n=81 and AUROC 0.512.

**Second spin-off contribution (see §4 for the first).** *Gated multimodal fusion
improves absolute methylation prediction but contributes nothing to variant-effect
prediction, because the context features are allele-invariant.* Paired equivalence
intervals, fusion − sequence: signed ρ +0.0026 [−0.0020, +0.0070]; direction
−0.0007 [−0.0064, +0.0051]; AUROC within distance bin −0.0021 [−0.0053, +0.0011].
Tight intervals, so this is equivalence, not failure to detect. Mechanism: the
context vector is identical for REF and ALT, so in
Δ = [dna_mut·g_mut − dna_wt·g_wt] + epi·[g_mut − g_wt] it survives only through a
gate shift driven by the sequence change. `gate_modulation.csv` closes that route
too — flat across all four gate-share quartiles, eight of eight intervals covering
zero, largest |effect| 0.007 (a non-significant trend in the predicted direction,
worth one sentence and no more). Applies to every sequence-plus-context model doing
variant scoring.

**Consequence: do not acquire blood ENCODE tracks.** The blood/breast mismatch is
not what limits variant-effect prediction here; allele-invariance is. Tissue-matched
context would not change these numbers.

**7. Regulatory enrichment.** Attribution + saturation mutagenesis matched against
JASPAR and HOCOMOCO, testing recovery of methylation-sensitive factors (CTCF,
NRF1, CEBPB, ZBTB33, BANP). Enrichment of high-|Δβ̂| variants against TFBS,
enhancers, CTCF sites, chromHMM states, with matched backgrounds.

---

## B. Journal criteria

| Journal | IF | Stated criterion | Met by | Status |
|---|---|---|---|---|
| Nature Machine Intelligence | 29.8 | Broadly generalizable AI framework validated across several genomic tasks | B.5 (BEND, 7 tasks) | at-risk — see note |
| Nature Genetics | 25.5 | Major genetic or disease discovery from large independent public cohorts | — | not pursued (deliberate) |
| Nature Communications | 18.1 | Cross-tissue + cross-cohort + cross-ancestry validation, plus substantial new biological findings derived computationally | B.1, B.2, B.4 | triad on track; biology clause is the weak leg |
| Genome Medicine | 10.8 | Clearer clinical relevance, multiple external cohorts, comparisons with leading methylation and variant-effect predictors | D.2, B.2, L1–L3 | best-served target |

### Notes on risk

**NMI — weakened by the compute compression.** BEND uses a frozen-embedding
protocol, so we enter as an *embedder*, and our embedder is DNABERT-2, which is
already on the BEND leaderboard. The gated fusion cannot help on BEND tasks
because BEND supplies no epigenomic context features. "Several genomic tasks" is
satisfied in letter, not in spirit. Treat NMI as a low-probability first
submission, not a plan target. Recovering it would require multi-task *training*,
which was cut on cost grounds.

**Nature Communications — the biology clause was always the weak leg.** The
validation triad (cross-tissue, cross-cohort, cross-ancestry) is now met cheaply
and completely. "Substantial new biological findings derived computationally"
rests entirely on requirement 7. Motif recovery is confirmatory by nature; to make
the clause land, requirement 7 needs at least one finding that is novel rather
than confirmatory — a tissue-differential pattern, or a characterized class of
variants nobody has described. Track that as its own line item.

**Genome Medicine — now the strongest fit.** Two of three criteria are met
emphatically ("multiple external cohorts" and "comparisons with leading
predictors" are the best-supported parts of the whole plan). Only clinical
relevance is thin, and it is the cheapest gap to close (one ClinVar figure,
~1 week). Mentor called this "the best-matching journal above 10" — the
compressed plan happens to optimise for exactly that.

**Also in range:** Nucleic Acids Research (~15) is above the >10 bar, is the
natural venue for a benchmark-plus-resource paper, and has materially better odds
than anything Nature-branded.

---

## C. Prior art to check before claiming novelty

| Territory | Status | Holder |
|---|---|---|
| Multi-modality variant effect, 1 Mb | taken | AlphaGenome, Borzoi, Enformer — **none predict methylation** |
| Multi-tissue sequence+context methylation with variant deltas | taken | DeepMethylation (9 tissues) |
| Methylation foundation models on profiles | adjacent | MethylGPT, CpGPT — no sequence input, cannot score variants |
| RC-equivariant architectures | taken | Shrikumar 2017; Zhou 2022; Caduceus 2024 |
| RC-consistency as fine-tuning objective | taken | RCCR (arXiv 2509.18529) — excludes variant effects and methylation |
| Sequence→methylation at foundation-model scale | **open** | the gap AlphaGenome left |
| Calibrated genomic variant effects | **open** | no major predictor ships uncertainty |

**Correction on record:** SilentMethyl averages FWD and RC predictions, and
½[f(x) + f(RC(x))] is exactly RC-invariant. The reported outputs are *not*
strand-inconsistent. Do not claim otherwise. An equivariant trunk is justified
only on efficiency (1× vs 2× inference), not correctness — default is skip.

---

## D. Data acquisition status

Run `python -u data/acquire_external_cohorts.py --check` for live status.
The frozen manifest is written to `data/external/external_manifest.json`.

| Source | Build | Liftover needed | Acquisition | Status |
|---|---|---|---|---|
| GENOA meQTL | hg19 | **yes** | automatic (Zenodo) | **done** — raw deleted, 30 MB harmonized retained |
| TCGA-BRCA tumours (from the matrix on disk) | hg38 | no | none | **done** — 699 unpaired of 791 |
| TCGA ancestry calls (GDC open) | n/a | no | manual | **done** — not powered, see §5 |
| ClinVar GRCh38 VCF | hg38 | no | automatic | **done** |
| JASPAR CORE vertebrates | n/a | no | manual | **done** |
| HOCOMOCO core | n/a | no | manual | **done** |
| CpGenie, DeepCpG | n/a | no | git clone | **done** — environments not yet stood up |
| GoDMC mQTL | hg19 | yes | manual | **dropped** — redundant with GENOA + eGTEx |
| gnomAD population AF | hg38 | no | manual | **dropped** — `af_genoa` + HM450 SNP masks cover it |
| BEND task data | hg38 | no | git clone | **dropped** — see NMI note in §B |
| ENCODE blood chromatin tracks | hg38 | no | manual | **not needed** — see §6, allele-invariance |

Stage A is closed. 31 GB → 26 GB, 728 → 629 files after cleanup. Raw GENOA is gone;
provenance survives via SHA-256 in `external_manifest.json` plus the Zenodo DOI.
`data/reference/*.bw` (14 GB) is deliberately retained: the C.2 window ablation may
need to re-extract context features at 400 and 2,000 bp.

**Build hazard.** GENOA and GoDMC are hg19; this project is hg38. The chain file
`data/reference/hg19ToHg38.over.chain.gz` is already in the repo. Liftover happens
in the harmonization step and writes new files — raw downloads stay byte-identical
to the published release so they remain checksum-verifiable against the source.
