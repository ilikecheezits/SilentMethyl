#!/usr/bin/env python3
"""Aggregate the R4 directions into one verdict, under a pre-specified rule.

Why this exists
---------------
scripts/40_meqtl_tissue_specificity.py compares model accuracy on tissue-SHARED
versus tissue-SPECIFIC meQTLs, one ordered cohort pair at a time. With two
cohorts that was two directions and a reader could hold both in their head. With
nine tissues it is up to 72, of which roughly 32 are informative -- and 32
directions x 2 metrics is 64 intervals, which will produce apparent findings by
chance alone if each is read as a separate test.

This applies script 40's OWN stated standard, unchanged, to nine tissues:

    "A difference in the SAME direction in BOTH analyses is the finding.
     One direction only means a cohort artifact."

So the unit of evidence is the unordered TISSUE PAIR, not the direction. A pair
counts as support only when both of its directions show a same-signed difference
whose interval excludes zero. Everything else is reported but does not count.

This rule is fixed before the numbers are read. It is the same rule whether the
answer is yes or no, and it is not adjusted afterwards.

Sign convention: difference = shared - specific. POSITIVE supports the
hypothesis (a sequence-only pathway predicts sequence-intrinsic meQTLs better).

Usage (run from the repository root)
------------------------------------
    python -u scripts/41_tissue_specificity_summary.py
    python -u scripts/41_tissue_specificity_summary.py \\
        --results results/journal/tissue_specificity_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

DEFAULT = Path("results/journal/tissue_shared_meqtls")
METRICS = ("direction_agreement", "signed_rho")


def verdict(diff, lo, hi):
    """Three-way call on one difference. None when the interval is unusable."""
    if any(x is None or pd.isna(x) for x in (diff, lo, hi)):
        return None
    if lo > 0:
        return "shared"
    if hi < 0:
        return "specific"
    return "none"


def collect(results, metric):
    rows = []
    for r in results:
        if r.get("note") or f"{metric}_difference" not in r:
            continue
        ci = r.get(f"{metric}_difference_ci") or [None, None]
        rows.append({
            "discovery": r["discovery"],
            "replication": r["replication"],
            "matched": r.get("matched"),
            "shared_n": r.get("shared"),
            "specific_n": r.get("specific"),
            "z_gap": r.get("median_absZ_gap"),
            "z_ok": r.get("z_balance_ok"),
            "diff": r.get(f"{metric}_difference"),
            "lo": ci[0], "hi": ci[1],
            "verdict": verdict(r.get(f"{metric}_difference"), ci[0], ci[1]),
        })
    return pd.DataFrame(rows)


def pair_concordance(frame):
    """Apply the pre-specified rule: both directions, same sign, both exclude 0."""
    out = []
    tissues = sorted(set(frame["discovery"]) | set(frame["replication"]))
    for a, b in combinations(tissues, 2):
        ab = frame[(frame["discovery"] == a) & (frame["replication"] == b)]
        ba = frame[(frame["discovery"] == b) & (frame["replication"] == a)]
        if ab.empty or ba.empty:
            continue  # only one direction informative -> cannot satisfy the rule
        va, vb = ab.iloc[0]["verdict"], ba.iloc[0]["verdict"]
        supports = (va == vb) and va in ("shared", "specific")
        out.append({"pair": f"{a} / {b}",
                    "dir1": va, "dir2": vb,
                    "both_directions_agree": bool(supports),
                    "class_favoured": va if supports else None})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    summary = args.results / "run_summary.json"
    if not summary.is_file():
        raise SystemExit(f"STOP: {summary} not found -- run scripts/40 first")
    with summary.open() as fh:
        payload = json.load(fh)
    results = payload.get("results", [])
    if not results:
        raise SystemExit(f"STOP: {summary} has no results")

    skipped = [r for r in results if r.get("note")]
    print()
    print("=" * 92)
    print("R4  SHARED vs TISSUE-SPECIFIC meQTLs")
    print("=" * 92)
    print(f"directions attempted        {len(results)}")
    print(f"informative                 {len(results) - len(skipped)}")
    print(f"skipped (too few in a class){len(skipped):>4}")
    if skipped:
        who = sorted({r['discovery'] for r in skipped})
        print(f"  underpowered as discovery: {', '.join(who)}")

    everything, decision = {}, {}
    for metric in METRICS:
        frame = collect(results, metric)
        if frame.empty:
            print(f"\n[{metric}] no usable directions")
            continue
        everything[metric] = frame

        # A direction whose arms are not |Z|-balanced is not a controlled
        # comparison; surface it rather than averaging it in silently.
        bad = frame[frame["z_ok"] == False]  # noqa: E712
        print(f"\n[{metric}]  difference = shared - specific")
        print("-" * 92)
        print(f"{'discovery':<22}{'replication':<22}{'n/arm':>7}"
              f"{'  difference [95% CI]':<30}{'call':>10}")
        for _, r in frame.iterrows():
            call = r["verdict"] or "unusable"
            flag = "" if r["z_ok"] in (True, None) else "  !Z"
            if r["diff"] is None or pd.isna(r["diff"]):
                print(f"{r['discovery']:<22}{r['replication']:<22}"
                      f"{r['matched']:>7}  {'(no interval)':<28}{call:>10}{flag}")
            else:
                print(f"{r['discovery']:<22}{r['replication']:<22}"
                      f"{r['matched']:>7}  {r['diff']:>+8.4f} "
                      f"[{r['lo']:+.4f},{r['hi']:+.4f}]{call:>11}{flag}")
        if len(bad):
            print(f"  !Z = |Z| medians outside the matching tolerance; "
                  f"{len(bad)} direction(s) NOT controlled")

        conc = pair_concordance(frame)
        n_support = int(conc["both_directions_agree"].sum()) if len(conc) else 0
        decision[metric] = {
            "directions": int(len(frame)),
            "directions_excluding_zero": int((frame["verdict"] != "none").sum()),
            "favouring_shared": int((frame["verdict"] == "shared").sum()),
            "favouring_specific": int((frame["verdict"] == "specific").sum()),
            "pairs_with_both_directions": int(len(conc)),
            "pairs_meeting_rule": n_support,
            "classes_favoured": sorted(
                {c for c in conc.get("class_favoured", pd.Series(dtype=object))
                 .dropna().tolist()}) if len(conc) else [],
        }
        d = decision[metric]
        print(f"\n  directions excluding zero   {d['directions_excluding_zero']} "
              f"of {d['directions']}  "
              f"(shared {d['favouring_shared']}, specific {d['favouring_specific']})")
        print(f"  expected by chance at 95%   ~{0.05 * d['directions']:.1f}")
        print(f"  PRE-SPECIFIED RULE: pairs with both directions agreeing "
              f"{d['pairs_meeting_rule']} of {d['pairs_with_both_directions']}")

    print()
    print("=" * 92)
    passes = [m for m, d in decision.items() if d["pairs_meeting_rule"] > 0]
    if not passes:
        print("VERDICT: no tissue pair meets the pre-specified rule on either")
        print("metric. The sequence-only pathway does NOT separate meQTLs by")
        print("mechanism. This is a real answer -- report it as one, and do not")
        print("promote individual directions that happened to exclude zero.")
    else:
        for m in passes:
            d = decision[m]
            print(f"VERDICT [{m}]: {d['pairs_meeting_rule']} pair(s) meet the rule, "
                  f"favouring {', '.join(d['classes_favoured'])}.")
        print("Check that the favoured class is the SAME across metrics and pairs")
        print("before calling this a finding.")
    print("=" * 92)

    out_dir = args.out or args.results
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, frame in everything.items():
        frame.to_csv(out_dir / f"summary_{metric}.csv", index=False)
    with (out_dir / "decision.json").open("w") as fh:
        json.dump({
            "rule": ("a tissue PAIR counts as support only when both ordered "
                     "directions show a same-signed difference whose 95% "
                     "block-bootstrap interval excludes zero; fixed before the "
                     "numbers were read"),
            "sign_convention": "difference = shared - specific; positive supports the hypothesis",
            "per_metric": decision,
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
