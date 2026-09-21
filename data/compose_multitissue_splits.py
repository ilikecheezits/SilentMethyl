#!/usr/bin/env python3
"""Compose per-tissue builds into one joint training set with disjoint probes. Giving every
tissue the same probes would make the sequence tower see identical DNA carrying conflicting
targets and cut unique sequence coverage to budget/N, so each training probe is assigned to
exactly one tissue, drawn from the intersection of what all included tissues cover. Held-out
chromosomes are checked to agree across tissues, and test is each tissue's full test-
chromosome set so joint numbers stay comparable to the single-tissue ones. --holdout
reallocates the budget across the remaining tissues; --dry-run prints the allocation without
writing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_TISSUES = {
    "BreastEpithelium": SCRIPT_DIR / "datafiles_breast_epithelium",
    "KidneyCortex": SCRIPT_DIR / "datafiles_multitissue" / "KidneyCortex",
    "Lung": SCRIPT_DIR / "datafiles_multitissue" / "Lung",
    "ColonTransverse": SCRIPT_DIR / "datafiles_multitissue" / "ColonTransverse",
}

DEFAULT_BUDGET = 345_359


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissue", action="append", default=None, metavar="NAME=DIR",
                    help="Override or add a tissue, e.g. "
                         "--tissue Lung=data/datafiles_multitissue/Lung. "
                         "Repeatable. Omit to use the four defaults.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where the composed train/val/test are written.")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help=f"Total training rows across all tissues "
                         f"(default {DEFAULT_BUDGET}, the single-tissue count).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Probe-assignment seed. Independent of the model "
                         "training seed -- vary it to test whether a result "
                         "depends on which probes landed in which tissue.")
    ap.add_argument("--holdout", default=None,
                    help="Tissue to exclude from training and validation. Test "
                         "becomes that tissue alone, at its full probe set.")
    ap.add_argument("--val-budget", type=int, default=None,
                    help="Total validation rows (default: the single-tissue "
                         "validation size, split disjointly across tissues).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the allocation and exit without writing.")
    return ap.parse_args()


def resolve_tissues(overrides: list[str] | None) -> dict[str, Path]:
    tissues = dict(DEFAULT_TISSUES)
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"STOP: --tissue expects NAME=DIR, got {item!r}")
        name, _, path = item.partition("=")
        tissues[name.strip()] = Path(path.strip()).resolve()
    return tissues


def load_split(directory: Path, split: str) -> pd.DataFrame:
    path = directory / f"{split}.csv"
    if not path.is_file():
        raise SystemExit(
            f"STOP: {path} is missing. Build this tissue first with "
            f"data/build_training_data.py --reference-dir ... --out-dir {directory}")
    return pd.read_csv(path)


def check_chromosome_agreement(tissues: dict[str, Path]) -> dict:
    """Every tissue must hold out the same chromosomes, or the blocking leaks.

    A probe on chr8 is test in breast; if lung put chr8 in train, that probe's
    sequence is trained on and the joint test set is contaminated. Chromosome
    blocking across tissues is the ONLY thing preventing this, and it is silent
    when it fails.
    """
    seen: dict[str, dict] = {}
    for name, directory in tissues.items():
        path = directory / "split_manifest.json"
        if not path.is_file():
            raise SystemExit(f"STOP: {path} is missing; cannot verify the split.")
        m = json.loads(path.read_text())
        seen[name] = {
            "val": sorted(m.get("validation_chromosomes", [])),
            "test": sorted(m.get("test_chromosomes", [])),
        }
    reference_name, reference = next(iter(seen.items()))
    for name, block in seen.items():
        if block != reference:
            raise SystemExit(
                "STOP: tissues do not share held-out chromosomes.\n"
                f"  {reference_name}: val={reference['val']} test={reference['test']}\n"
                f"  {name}: val={block['val']} test={block['test']}\n"
                "A probe held out in one tissue would be trained on in another, "
                "contaminating the joint test set. Rebuild with matching "
                "--val-chroms / --test-chroms.")
    return reference


def allocate(probes: np.ndarray, names: list[str], budget: int,
             seed: int) -> dict[str, np.ndarray]:
    """Shuffle once, cut into equal disjoint blocks, one per tissue.

    Cutting a single shuffled array is what makes the blocks disjoint by
    construction rather than by a check that could be wrong. The per-tissue size
    is floor(budget / n) so the total never exceeds the budget; the remainder
    (at most n-1 probes) is dropped rather than given to an arbitrary tissue.
    """
    rng = np.random.default_rng(seed)
    shuffled = probes.copy()
    rng.shuffle(shuffled)

    n = len(names)
    per_tissue = min(budget // n, len(shuffled) // n)
    if per_tissue == 0:
        raise SystemExit(
            f"STOP: budget {budget} across {n} tissues leaves no probes each.")
    return {name: shuffled[i * per_tissue:(i + 1) * per_tissue]
            for i, name in enumerate(names)}


def main() -> None:
    args = parse_args()
    tissues = resolve_tissues(args.tissue)

    if args.holdout and args.holdout not in tissues:
        raise SystemExit(f"STOP: --holdout {args.holdout} is not one of "
                         f"{', '.join(tissues)}")

    chroms = check_chromosome_agreement(tissues)
    print(f"held-out chromosomes agree across all tissues: "
          f"val={chroms['val']} test={chroms['test']}")

    training_names = [t for t in tissues if t != args.holdout]
    print(f"training tissues: {', '.join(training_names)}")
    if args.holdout:
        print(f"held out entirely: {args.holdout}  "
              f"(test is this tissue alone, at its full probe set)")

    train_frames = {n: load_split(tissues[n], "train") for n in training_names}
    for name, frame in train_frames.items():
        print(f"  {name:20s} {len(frame):>8,} train rows available")

    universe = None
    for frame in train_frames.values():
        ids = set(frame["probeID"].astype(str))
        universe = ids if universe is None else (universe & ids)
    universe_array = np.array(sorted(universe))
    print(f"probe universe (covered in every training tissue): "
          f"{len(universe_array):,}")
    for name, frame in train_frames.items():
        dropped = len(set(frame["probeID"].astype(str))) - len(universe)
        if dropped:
            print(f"    {name}: {dropped:,} probes not covered everywhere, "
                  f"excluded from assignment")

    assignment = allocate(universe_array, training_names, args.budget, args.seed)

    pooled: set[str] = set()
    for name, probes in assignment.items():
        overlap = pooled.intersection(probes)
        if overlap:
            raise SystemExit(f"STOP: {name} shares {len(overlap)} probes with "
                             "another tissue; assignment is not disjoint.")
        pooled.update(probes)

    val_frames = {n: load_split(tissues[n], "val") for n in training_names}
    val_budget = args.val_budget or min(len(f) for f in val_frames.values())
    val_universe = None
    for frame in val_frames.values():
        ids = set(frame["probeID"].astype(str))
        val_universe = ids if val_universe is None else (val_universe & ids)
    val_assignment = allocate(np.array(sorted(val_universe)), training_names,
                              val_budget, args.seed)

    test_names = [args.holdout] if args.holdout else list(tissues)
    test_frames = {n: load_split(tissues[n], "test") for n in test_names}

    print()
    print(f"{'tissue':<20}{'train':>10}{'val':>10}{'test':>10}")
    for name in training_names:
        print(f"{name:<20}{len(assignment[name]):>10,}"
              f"{len(val_assignment[name]):>10,}"
              f"{len(test_frames.get(name, [])):>10,}")
    if args.holdout:
        print(f"{args.holdout:<20}{0:>10}{0:>10}"
              f"{len(test_frames[args.holdout]):>10,}")
    totals = (sum(len(v) for v in assignment.values()),
              sum(len(v) for v in val_assignment.values()),
              sum(len(f) for f in test_frames.values()))
    print(f"{'TOTAL':<20}{totals[0]:>10,}{totals[1]:>10,}{totals[2]:>10,}")
    print(f"genome coverage in training: "
          f"{len(pooled) / len(universe_array) * 100:.1f}% of the shared universe")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(split: str, parts: list[pd.DataFrame]) -> int:
        frame = pd.concat(parts, ignore_index=True)
        if frame["probeID"].duplicated().any():
            n = int(frame["probeID"].duplicated(keep=False).sum())
            raise SystemExit(
                f"STOP: {split} has {n} duplicated probe rows. In a disjoint "
                "composition every probe appears once; this means two tissues "
                "were given the same probe.")
        frame.to_csv(args.out_dir / f"{split}.csv", index=False)
        return len(frame)

    train_parts = []
    for name in training_names:
        keep = set(assignment[name])
        part = train_frames[name][train_frames[name]["probeID"].astype(str).isin(keep)].copy()
        part["Tissue"] = name
        train_parts.append(part)

    val_parts = []
    for name in training_names:
        keep = set(val_assignment[name])
        part = val_frames[name][val_frames[name]["probeID"].astype(str).isin(keep)].copy()
        part["Tissue"] = name
        val_parts.append(part)

    test_parts = []
    for name in test_names:
        part = test_frames[name].copy()
        part["Tissue"] = name
        if len(test_names) > 1:
            part["probeID"] = part["probeID"].astype(str) + f"__{name}"
        test_parts.append(part)

    n_train = write("train", train_parts)
    n_val = write("val", val_parts)
    n_test = write("test", test_parts)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "composition": "disjoint training probes, one tissue per probe",
        "assignment_seed": args.seed,
        "row_budget": args.budget,
        "holdout_tissue": args.holdout,
        "training_tissues": training_names,
        "test_tissues": test_names,
        "held_out_chromosomes": chroms,
        "sources": {n: str(p) for n, p in tissues.items()},
        "shared_probe_universe": int(len(universe_array)),
        "rows": {"train": n_train, "val": n_val, "test": n_test},
        "probes_per_tissue": {n: int(len(v)) for n, v in assignment.items()},
        "probe_id_suffixed_by_tissue_in_test": len(test_names) > 1,
    }
    (args.out_dir / "composition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    (args.out_dir / "probe_assignment.json").write_text(
        json.dumps({n: sorted(v.tolist()) for n, v in assignment.items()},
                   indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {args.out_dir}")
    print(f"  train.csv {n_train:,}   val.csv {n_val:,}   test.csv {n_test:,}")
    print(f"  composition_manifest.json, probe_assignment.json")


if __name__ == "__main__":
    main()
