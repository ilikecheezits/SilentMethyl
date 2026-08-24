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
| 4 | Uncertainty calibration | No — post-hoc | B.3 | **in-progress** | `results/journal/rc_uncertainty/` |
| 5 | Ancestry analyses | No — analysis | B.1 | not-started | |
| 6 | Independent variant evaluation | No — inference | B.1 | not-started | |
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

*Open question, script 17.* Two explanations must be separated before this is
written up: (a) intermediate-methylation probes are genuinely harder, so distance
from 0.5 is a legitimate difficulty signal; or (b) absolute β error is
**mechanically bounded** near 0 and 1, so the heuristic is measuring headroom
rather than uncertainty. Script 17 tests this with partial correlation given the
heuristic, within-β̂-stratum correlation, stratified selective prediction, and a
repeat on unbounded M-value error.

*Consequence either way.* If nothing beats the heuristic inside strata, that is a
publishable negative — ensemble uncertainty adds little over a trivial baseline
on this task — and the calibration section rests on conformal intervals, whose
coverage guarantee holds regardless. Do not write the calibration section until
script 17 has run.

**5. Ancestry analyses.** Cross-ancestry replication from published summary
statistics: GoDMC (European-dominant blood), GENOA (African American, EPIC), the
2024 East Asian/European study. Plus TCGA ancestry-stratified prediction error
using the GDC's open per-sample ancestry calls, and a gnomAD population-frequency
audit of SNPs under probes.

**6. Independent variant evaluation.** From n=81 to millions. Report signed ρ,
direction agreement, AUROC, magnitude correlation — stratified by CpG-alteration
status, distance bin, effect decile, cohort ancestry.

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
| GENOA meQTL (5.3 GB, 22 files) | hg19 | **yes** | automatic (Zenodo) | not-started |
| GoDMC mQTL | hg19 (verify) | **yes** | manual | not-started |
| eGTEx mQTL, 9 tissues | hg38 | no | manual | not-started |
| ClinVar GRCh38 VCF | hg38 | no | automatic | not-started |
| JASPAR CORE vertebrates | n/a | no | manual | not-started |
| HOCOMOCO core | n/a | no | manual | not-started |
| gnomAD population AF (probe-window subset) | hg38 | no | manual | not-started |
| BEND task data | hg38 | no | manual (git clone) | not-started |

**Build hazard.** GENOA and GoDMC are hg19; this project is hg38. The chain file
`data/reference/hg19ToHg38.over.chain.gz` is already in the repo. Liftover happens
in the harmonization step and writes new files — raw downloads stay byte-identical
to the published release so they remain checksum-verifiable against the source.
