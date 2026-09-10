**Subject:** SilentMethyl — all analyses finished, and two results that change the framing

Hi [name],

Short version: everything you asked for is done. Two of the results came back
differently than I expected. I think they make the paper stronger, but they do
change what we can claim, so I wanted to lay it out before writing the final
version.

**Your list**

- Multi-cohort testing — done, four independent data sources.
- Repeated chromosome splits — done. Four separate splits, model retrained from
  scratch on each one.
- Stronger baselines — done. Six comparison models, including the two best
  published ones.
- Uncertainty — done.
- Independent variant evaluation — done, two cohorts.
- Regulatory enrichment — done, though the signal turned out to be simpler than
  we thought (it is explained by base composition alone).
- Ancestry — partly; the sample sizes cap what is possible. Explained below.
- Multi-tissue joint training — I did not build this, for a specific reason I
  want to check with you.
- Functional validation — the one real gap.

**First result: the chromatin half of the model helps predict methylation, but
not variant effects.**

The model has two halves, one reading DNA sequence and one reading chromatin
context. Combining them clearly improves prediction of methylation level, and
that improvement now holds up across all four chromosome splits.

For predicting what a *variant* does, though, the chromatin half contributes
almost nothing. There is a structural reason. We score a variant by running the
reference and mutant sequences through the model and taking the difference, and
the chromatin context is identical for both of them. So it can scale the answer
up or down, but it cannot change what the answer is. I tested this three
different ways and got the same result each time.

This is also why I did not build the multi-tissue joint model. Giving the model
more tissues' chromatin cannot fix a pathway that cancels out in the
subtraction — it would need a different architecture. I would rather say that
plainly than build it and report a null.

**Second result: a competing model was published in August, and we are about
even with it rather than ahead.**

Melody (Nature Communications, August) predicts methylation across 39 cell
types. I scored both models on exactly the same variant–CpG pairs across eight
tissues. It is a tie in seven of the eight, with their estimates slightly ahead
in all eight.

I want to be careful not to spin this. The fair reading is that our model was
trained on breast tissue only and given no information about the other eight
tissues, while theirs was trained on all of them. Coming out even under that
handicap is a genuine result. Coming out "better" is not supported by the
numbers, and we should not write it.

That comparison also produced a number nobody had measured: training on the
matched tissue is worth only about 0.02–0.03 in accuracy. Four independent
routes to that estimate agree. That is small, and it matters — it suggests the
field is spending a great deal of effort on tissue-specific training for very
little return.

**The biological question came back no, for both models.**

We asked whether predicted variant effects can distinguish mQTLs that act across
many tissues from those confined to one. The answer is no.

More usefully, I ran the identical test on Melody, and the answer is no there
too. Both models show a weak apparent signal, and in both cases that signal
points toward whichever group happened to be measured in the smallest cohorts.
The two models point in *opposite* directions, which rules out "one model is
simply worse" and puts the problem in the benchmark data rather than in either
model. The discovery cohorts are too small to answer this question at present.

One thing I should own here: the check I used to detect this problem was
originally written to look in only one direction — the direction our own model
fails in. It stayed silent for Melody, and I first read that silence as Melody
not having the problem. I rewrote it to look both ways, reran both models, and
got the result above. The paper states this.

**Where that leaves the paper**

I think this is a methods paper, and a better one than what we started with. The
contribution is:

1. A way of evaluating these models that removes a confound nobody controls
   for — simply how close the variant sits to the CpG. Distance alone gets you
   most of the way to the published accuracy figures. Once it is removed, every
   model drops sharply, including the published ones.
2. A structural explanation of why the obvious architecture — attaching
   chromatin data to a variant-effect model — cannot work. This applies well
   beyond our model.
3. The measurement that tissue-matched training buys very little.
4. A demonstration that two very different models both fail the biological test,
   in a way that points at the benchmarks rather than the models.

What it is not is "we built a better variant-effect predictor." I do not want to
write that sentence.

**Two open items, and what I need from you**

Functional validation is the remaining gap — everything so far is computational.
I can add colocalisation with public eQTL and eQTM data without needing any
controlled-access sources. Anything experimental is out of reach for now.

Ancestry is limited by sample size. GENOA gives us an African American cohort,
but ancestry, tissue and platform all vary together there, so they cannot be
separated. Within TCGA, the breast normal samples are 84 European and 4
African — four donors cannot support the analysis, and I would rather say so
than produce something that looks like a disparity estimate but is not one.

So, three questions:

1. Are you comfortable with the reframing — methods and negative results rather
   than a benchmark win?
2. Should I add the eQTL/eQTM colocalisation before we submit, or list
   functional validation as future work?
3. Still Nature Communications, or is there a better fit given the change of
   emphasis?

Happy to walk through any of it whenever suits you.

Best,
Samuel
