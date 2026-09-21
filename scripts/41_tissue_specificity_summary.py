#!/usr/bin/env python3
"""Aggregate the per-direction outputs of 40_meqtl_tissue_specificity.py into one verdict
under a pre-specified rule: the same class must be favoured across metrics and
directions. Reports the verdict and the evidence behind it, including disagreements.
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


def power_diagnostic(frame, results, metric):
    """Does the SIGN of the difference track cohort power rather than biology?

    Added after the nine-tissue run, where every direction calling `shared`
    had <= 55 pairs per arm and every direction calling `specific` had >= 404 --
    a clean separation with no overlap. That is not a biological gradient, it is
    a statement about which cohorts were underpowered.

    The mechanism is winner's curse. In a discovery cohort with ~130 significant
    pairs, a pair that clears p<5e-8 and then fails to replicate anywhere is
    more likely a false positive than a real meQTL of that class -- and the
    model ANTI-predicts noise, so that arm goes negative. Reporting the per-arm
    sign alongside n makes that visible instead of leaving it to be discovered
    by a reviewer.

    TWO-SIDED, and deliberately so
    ------------------------------
    The first version of this function tested only

        calls['shared'].max_n < calls['specific'].min_n

    i.e. it could only detect the confound when the `shared` calls were the
    underpowered ones. That is the shape SilentMethyl happens to produce. Run
    against Melody, whose R4 failure runs the OTHER way (it calls `specific`),
    the test stayed silent -- and silence was then read as "Melody has no power
    confound", which the test never checked. Melody in fact separates just as
    cleanly in reverse.

    A one-sided diagnostic applied to two models is not a comparison, so both
    the n-separation test and the anti-prediction test below now examine both
    arms and report WHICH call is the underpowered one.
    """
    calls = {}
    for call in ("shared", "specific"):
        sub = frame[frame["verdict"] == call]
        if sub.empty:
            continue
        calls[call] = {"n_directions": int(len(sub)),
                       "min_n_per_arm": int(sub["matched"].min()),
                       "max_n_per_arm": int(sub["matched"].max()),
                       "median_n_per_arm": float(sub["matched"].median())}

    separated, low_power_call = False, None
    if len(calls) == 2:
        s, p = calls["shared"], calls["specific"]
        if s["max_n_per_arm"] < p["min_n_per_arm"]:
            separated, low_power_call = True, "shared"
        elif p["max_n_per_arm"] < s["min_n_per_arm"]:
            separated, low_power_call = True, "specific"

    lookup = {(r["discovery"], r["replication"]): r for r in results}
    arm_key = {"shared": f"{metric}_shared",
               "specific": f"{metric}_tissue_specific"}
    negative = {"shared": [], "specific": []}
    for _, row in frame.iterrows():
        r = lookup.get((row["discovery"], row["replication"]), {})
        for arm, key in arm_key.items():
            v = r.get(key)
            if v is not None and not pd.isna(v) and v < 0:
                negative[arm].append({
                    "direction": f"{row['discovery']}->{row['replication']}",
                    "n": row["matched"], "arm_value": float(v)})
    return {"by_call": calls,
            "cleanly_separated_by_n": bool(separated),
            "low_power_call": low_power_call,
            "directions_with_negative_arm": negative,
            "directions_with_negative_specific_arm": negative["specific"]}


def pair_concordance(frame):
    """Apply the pre-specified rule: both directions, same sign, both exclude 0."""
    out = []
    tissues = sorted(set(frame["discovery"]) | set(frame["replication"]))
    for a, b in combinations(tissues, 2):
        ab = frame[(frame["discovery"] == a) & (frame["replication"] == b)]
        ba = frame[(frame["discovery"] == b) & (frame["replication"] == a)]
        if ab.empty or ba.empty:
            continue
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
    ap.add_argument("--label", default="The scored model",
                    help="Name of the model these results came from, so two "
                         "runs are distinguishable in a scrollback "
                         "(e.g. --label 'SilentMethyl fusion' / 'Melody-MT').")
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
    print(f"R4  SHARED vs TISSUE-SPECIFIC meQTLs   [{args.label}]")
    print("=" * 92)
    print(f"results                     {args.results}")
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

        diag = power_diagnostic(frame, results, metric)
        if diag["by_call"]:
            print("\n  sign of the difference vs cohort power:")
            for call, s in diag["by_call"].items():
                print(f"    calls '{call}'  {s['n_directions']:>2} direction(s), "
                      f"n per arm {s['min_n_per_arm']}-{s['max_n_per_arm']} "
                      f"(median {s['median_n_per_arm']:.0f})")
            if diag["cleanly_separated_by_n"]:
                lp = diag["low_power_call"]
                print(f"    -> the two calls do NOT overlap in n, and '{lp}' is "
                      f"the LOW-power side.")
                print("       The SIGN is determined by discovery-cohort power, "
                      "not by biology.")
                print("       Do not report either as a mechanism difference.")
            elif len(diag["by_call"]) == 2:
                print("    -> the n ranges overlap; the call is not explained "
                      "by power alone.")
        for arm, neg in diag["directions_with_negative_arm"].items():
            if not neg:
                continue
            print(f"\n  {arm} arm is NEGATIVE in {len(neg)} direction(s) "
                  f"-- the model anti-predicts them,")
            print("  which is the winner's-curse signature, not chromatin mediation:")
            for x in sorted(neg, key=lambda d: d["arm_value"])[:6]:
                print(f"    {x['direction']:<44} n={x['n']:>4}  "
                      f"{arm} arm {x['arm_value']:+.4f}")

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
            "power_diagnostic": diag,
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
        print(f"VERDICT: no tissue pair meets the pre-specified rule on either")
        print(f"metric. {args.label} does NOT separate meQTLs by mechanism.")
        print("This is a real answer -- report it as one, and do not promote")
        print("individual directions that happened to exclude zero.")
    else:
        for m in passes:
            d = decision[m]
            print(f"VERDICT [{m}]: {d['pairs_meeting_rule']} pair(s) meet the rule, "
                  f"favouring {', '.join(d['classes_favoured'])}.")
        print("Check that the favoured class is the SAME across metrics and pairs")
        print("before calling this a finding.")
        lp = {decision[m]["power_diagnostic"]["low_power_call"] for m in passes
              if decision[m]["power_diagnostic"]["cleanly_separated_by_n"]}
        if lp:
            side = ", ".join(sorted(x for x in lp if x))
            print()
            print("BUT: the sign of the difference separates cleanly by cohort")
            print(f"power (see above), with '{side}' on the low-power side. A pair")
            print("meeting the rule only among the least-powered cohorts, whose")
            print("arm is anti-predicted, is a winner's-curse artifact -- not")
            print("evidence that the model separates meQTLs by mechanism. Treat")
            print("the overall answer as NO and report the artifact explicitly.")
    print("=" * 92)

    out_dir = args.out or args.results
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, frame in everything.items():
        frame.to_csv(out_dir / f"summary_{metric}.csv", index=False)
    with (out_dir / "decision.json").open("w") as fh:
        json.dump({
            "label": args.label,
            "results_dir": str(args.results),
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
