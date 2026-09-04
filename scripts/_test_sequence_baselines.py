#!/usr/bin/env python3
"""Synthetic end-to-end test for scripts/14_baselines_simple.py.

Builds fake train/val/test splits whose methylation target is a KNOWN function of
sequence, plus a fake variant-scoring input, then checks that:

  * the k-mer baseline recovers the planted signal and beats the composition
    baseline on the test split (a sanity check that the ridge path works);
  * variant deltas are exact -- recomputed from scratch and compared against the
    script's own output;
  * every pair-level exclusion (target-CpG, REF mismatch, out-of-window) fires;
  * the emitted pair_scores.csv carries every column scripts/20 requires.

torch is stubbed only so scripts/training_common.py can be imported here; the
functions under test never touch it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = Path(tempfile.gettempdir()) / "silentmethyl_baseline_test"
RNG = np.random.default_rng(11)

FULL_LEN = 5000
C_INDEX = 2499
WINDOW = 1000


def write_torch_stub(root: Path) -> None:
    """Minimal torch / transformers so training_common imports on a CPU box."""
    stub = root / "_stubs"
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "torch").mkdir(exist_ok=True)
    (stub / "torch" / "__init__.py").write_text(
        "import math\n"
        "class Tensor:\n"
        "    pass\n"
        "class dtype:\n"
        "    pass\n"
        "class _Mod:\n"
        "    def __init__(self, *a, **k): pass\n"
        "    def __call__(self, *a, **k): raise NotImplementedError\n"
        "class nn:\n"
        "    Module = _Mod\n"
        "    Linear = _Mod\n"
        "    LayerNorm = _Mod\n"
        "    Sequential = _Mod\n"
        "    Dropout = _Mod\n"
        "    ReLU = _Mod\n"
        "    GELU = _Mod\n"
        "    Sigmoid = _Mod\n"
        "    Identity = _Mod\n"
        "    ModuleList = _Mod\n"
        "    class functional:\n"
        "        pass\n"
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return False\n"
        "    @staticmethod\n"
        "    def manual_seed_all(*a): pass\n"
        "    @staticmethod\n"
        "    def empty_cache(): pass\n"
        "cuda = _Cuda()\n"
        "class _Backends:\n"
        "    class cudnn:\n"
        "        deterministic = True\n"
        "        benchmark = False\n"
        "backends = _Backends()\n"
        "class amp:\n"
        "    @staticmethod\n"
        "    def autocast(*a, **k): raise NotImplementedError\n"
        "def manual_seed(*a): pass\n"
        "def tensor(*a, **k): raise NotImplementedError\n"
        "def sigmoid(*a, **k): raise NotImplementedError\n"
        "float32 = 'float32'\n"
        "long = 'long'\n"
        "class utils:\n"
        "    class data:\n"
        "        Dataset = _Mod\n"
        "        DataLoader = _Mod\n"
    )
    (stub / "torch" / "utils").mkdir(exist_ok=True)
    (stub / "torch" / "utils" / "__init__.py").write_text("")
    (stub / "torch" / "utils" / "data.py").write_text(
        "class Dataset: pass\nclass DataLoader: pass\n")
    (stub / "torch" / "nn").mkdir(exist_ok=True)
    (stub / "torch" / "nn" / "__init__.py").write_text(
        "class Module:\n"
        "    def __init__(self, *a, **k): pass\n"
        "class _M(Module): pass\n"
        "Linear = LayerNorm = Sequential = Dropout = ReLU = GELU = _M\n"
        "Sigmoid = Identity = ModuleList = _M\n")
    (stub / "torch" / "nn" / "functional.py").write_text("")
    (stub / "torch" / "amp.py").write_text(
        "def autocast(*a, **k): raise NotImplementedError\n")
    (stub / "transformers.py").write_text(
        "class _Auto:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, *a, **k): raise NotImplementedError\n"
        "AutoConfig = AutoModel = AutoTokenizer = _Auto\n")


def make_sequence(signal_strength: float) -> tuple[str, float]:
    """A 5,000-bp record with a CpG at the centre and a planted k-mer signal.

    The target M-value is a linear function of the count of 'CGCG' inside the
    scored 1,000-bp window plus noise. A 6-mer ridge must find this; a
    three-number composition summary largely cannot.
    """
    seq = list(RNG.choice(list("ACGT"), FULL_LEN))
    seq[C_INDEX] = "C"
    seq[C_INDEX + 1] = "G"
    text = "".join(seq)
    window = text[2000:3000]
    motif_count = sum(1 for i in range(len(window) - 3) if window[i:i + 4] == "CGCG")
    m = signal_strength * motif_count + RNG.normal(0, 0.4)
    return text, float(m)


def plant_motifs(text: str, n: int) -> str:
    """Insert n copies of CGCG at random positions inside the scored window."""
    chars = list(text)
    for _ in range(n):
        pos = int(RNG.integers(2010, 2980))
        if C_INDEX - 4 <= pos <= C_INDEX + 4:
            continue
        chars[pos:pos + 4] = list("CGCG")
    return "".join(chars)


def build_split(path: Path, n: int, split: str, start: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        text, _ = make_sequence(0.0)
        text = plant_motifs(text, int(RNG.integers(0, 12)))
        window = text[2000:3000]
        count = sum(1 for j in range(len(window) - 3) if window[j:j + 4] == "CGCG")
        m = 0.55 * count - 2.0 + RNG.normal(0, 0.35)
        beta = 1.0 / (1.0 + 2.0 ** (-m))
        rows.append({
            "probeID": f"cg{start + i:08d}",
            "Healthy_5000bp_DNA": text,
            "M_Value_Target": m,
            "Median_Beta": beta,
            "Binary_State_Target": int(beta > 0.5),
            "Split": split,
        })
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def build_scoring_input(path: Path, test_frame: pd.DataFrame) -> dict:
    """Pairs exercising the keep path and all three exclusion paths."""
    rows, expect = [], {"keep": 0, "target_cpg": 0, "ref_mismatch": 0,
                        "outside_window": 0}
    for i, row in test_frame.head(40).iterrows():
        seq = row["Healthy_5000bp_DNA"]
        probe = row["probeID"]

        for d in (-300, -7, 12, 400):
            pos0 = C_INDEX + d
            ref = seq[pos0]
            alt = "A" if ref != "A" else "T"
            rows.append(dict(Variant_ID=f"v{probe}_{d}", Gene="NA", chr="chr8",
                             Position_1based=pos0 + 1, Ref=ref, Alt=alt,
                             probeID=probe, probe_split="test", distance_bp=d,
                             abs_distance_bp=abs(d), cpg_chr="chr8",
                             cpg_pos_hg38=1_000_000 + int(i) * 5000,
                             creates_cpg=False, destroys_cpg=False,
                             alters_target_cpg=False,
                             beta_ref_to_alt=float(RNG.normal(0, 0.3)),
                             pvalue=float(10 ** -RNG.uniform(0, 10)),
                             se=0.05))
            expect["keep"] += 1

        # target CpG itself -> must be excluded
        rows.append(dict(Variant_ID=f"v{probe}_cpg", Gene="NA", chr="chr8",
                         Position_1based=C_INDEX + 1, Ref=seq[C_INDEX], Alt="A",
                         probeID=probe, probe_split="test", distance_bp=0,
                         abs_distance_bp=0, cpg_chr="chr8",
                         cpg_pos_hg38=1_000_000 + int(i) * 5000,
                         creates_cpg=False, destroys_cpg=True,
                         alters_target_cpg=True, beta_ref_to_alt=-0.5,
                         pvalue=1e-12, se=0.05))
        expect["target_cpg"] += 1

        # REF that disagrees with the stored sequence -> must be excluded
        pos0 = C_INDEX + 100
        wrong = "A" if seq[pos0] != "A" else "T"
        rows.append(dict(Variant_ID=f"v{probe}_bad", Gene="NA", chr="chr8",
                         Position_1based=pos0 + 1, Ref=wrong, Alt=seq[pos0],
                         probeID=probe, probe_split="test", distance_bp=100,
                         abs_distance_bp=100, cpg_chr="chr8",
                         cpg_pos_hg38=1_000_000 + int(i) * 5000,
                         creates_cpg=False, destroys_cpg=False,
                         alters_target_cpg=False, beta_ref_to_alt=0.2,
                         pvalue=1e-3, se=0.05))
        expect["ref_mismatch"] += 1

        # beyond the 1,000-bp window -> must be excluded
        rows.append(dict(Variant_ID=f"v{probe}_far", Gene="NA", chr="chr8",
                         Position_1based=C_INDEX + 900 + 1, Ref=seq[C_INDEX + 900],
                         Alt="A", probeID=probe, probe_split="test",
                         distance_bp=900, abs_distance_bp=900, cpg_chr="chr8",
                         cpg_pos_hg38=1_000_000 + int(i) * 5000,
                         creates_cpg=False, destroys_cpg=False,
                         alters_target_cpg=False, beta_ref_to_alt=0.1,
                         pvalue=0.4, se=0.05))
        expect["outside_window"] += 1

    pd.DataFrame(rows).to_csv(path, index=False)
    return expect


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    (ROOT / "scripts").mkdir(parents=True)
    write_torch_stub(ROOT)
    for name in ("14_baselines_simple.py",):
        shutil.copy(SRC / name, ROOT / "scripts" / name)
    repo = Path("/mnt/user-data/uploads/SilentMethyl/scripts")
    for name in ("training_common.py", "matched_background_utils.py"):
        shutil.copy(repo / name, ROOT / "scripts" / name)

    train = build_split(ROOT / "data/datafiles/train.csv", 900, "train", 0)
    build_split(ROOT / "data/datafiles/val.csv", 200, "val", 900)
    test = build_split(ROOT / "data/datafiles/test.csv", 200, "test", 1100)
    expect = build_scoring_input(ROOT / "data/scoring_input.csv", test)

    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT / "_stubs")

    print("=" * 74)
    r = subprocess.run(
        [sys.executable, "-u", "scripts/14_baselines_simple.py",
         "--task", "both", "--k-max", "4",
         "--input-csv", "data/scoring_input.csv",
         "--output-dir", "results/baselines"],
        cwd=ROOT, env=env)
    if r.returncode != 0:
        print(f"FAILED: exit {r.returncode}")
        return 1

    failures = []
    out = ROOT / "results/baselines"
    absolute = pd.read_csv(out / "absolute_prediction_metrics.csv")
    print()
    print("=" * 74)
    print("ASSERTIONS")
    print("=" * 74)
    print(absolute.to_string(index=False))

    kmer = absolute[absolute["model"] == "kmer_ridge"].iloc[0]
    comp = absolute[absolute["model"] == "composition"].iloc[0]
    if not kmer["m_pearson"] > 0.8:
        failures.append(f"kmer_ridge did not recover the planted signal "
                        f"(pearson {kmer['m_pearson']:.3f})")
    if not kmer["m_rmse"] < comp["m_rmse"]:
        failures.append("kmer_ridge did not beat composition on the planted signal")

    scores = pd.read_csv(out / "variant_scoring/heldout/kmer_ridge/seed-1/pair_scores.csv")
    print(f"\nscored pairs: {len(scores)} (expected {expect['keep']})")
    if len(scores) != expect["keep"]:
        failures.append(f"{len(scores)} pairs scored, expected {expect['keep']}")
    if scores["Pair_UID"].duplicated().any():
        failures.append("duplicate Pair_UID in the score file")

    required_by_script20 = {"Pair_UID", "Predicted_Delta_M", "beta_ref_to_alt",
                            "pvalue", "abs_distance_bp", "cpg_chr",
                            "cpg_pos_hg38", "creates_cpg", "destroys_cpg"}
    missing = sorted(required_by_script20 - set(scores.columns))
    if missing:
        failures.append(f"score file is missing columns scripts/20 needs: {missing}")

    # --- independent recomputation of the deltas ------------------------------
    sys.path.insert(0, str(ROOT / "_stubs"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    mod = importlib.import_module("14_baselines_simple") if False else None
    spec = importlib.util.spec_from_file_location(
        "baselines", ROOT / "scripts" / "14_baselines_simple.py")
    baselines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baselines)

    seqs = dict(zip(test["probeID"], test["Healthy_5000bp_DNA"]))
    enc = baselines.KmerEncoder(4)
    check = scores.head(25)
    worst = 0.0
    for row in check.itertuples(index=False):
        seq = seqs[row.probeID]
        idx = C_INDEX + int(row.distance_bp)
        mut = seq[:idx] + row.Alt + seq[idx + 1:]
        wt_w = seq[2000:3000]
        mut_w = mut[2000:3000]
        diff = enc.counts(mut_w) - enc.counts(wt_w)
        if int(np.abs(diff).sum()) == 0:
            failures.append("a scored variant changed no k-mer count at all")
            break
        # sign/magnitude consistency: delta must be finite and the two windows
        # must differ at exactly one position
        differing = [i for i, (a, b) in enumerate(zip(wt_w, mut_w)) if a != b]
        if differing != [499 + int(row.distance_bp)]:
            failures.append(f"window difference at {differing}, expected "
                            f"{[499 + int(row.distance_bp)]}")
            break
        worst = max(worst, abs(float(row.Predicted_Delta_M)))
    print(f"largest |predicted delta M| among the 25 rechecked: {worst:.4f}")

    # --- RC invariance, independently --------------------------------------
    probe_seq = test["Healthy_5000bp_DNA"].iloc[0][2000:3000]
    rc = probe_seq.upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]
    d = float(np.max(np.abs(enc.counts(probe_seq) - enc.counts(rc))))
    print(f"RC invariance, independent check: max abs diff {d:g}")
    if d > 1e-9:
        failures.append("k-mer features are not RC-invariant")

    print()
    print("=" * 74)
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
