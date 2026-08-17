"""Utilities for matched synonymous-background comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd

DNA_BASES = frozenset("ACGT")
CENTER_C_INDEX = 499
CENTER_G_INDEX = 500
PROTECTED_CPG_INDICES = frozenset({CENTER_C_INDEX, CENTER_G_INDEX})

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(_COMPLEMENT)[::-1]


def find_single_snv_difference(wt: str, mut: str) -> tuple[int, str, str]:
    wt = str(wt).upper()
    mut = str(mut).upper()
    if len(wt) != len(mut):
        raise ValueError("WT and mutant sequences have different lengths; expected an SNV.")
    differences = [i for i, (a, b) in enumerate(zip(wt, mut)) if a != b]
    if len(differences) != 1:
        raise ValueError(f"Expected exactly one changed base, found {len(differences)}.")
    index = differences[0]
    ref, alt = wt[index], mut[index]
    if ref not in DNA_BASES or alt not in DNA_BASES:
        raise ValueError(f"Non-ACGT allele at index {index}: {ref}>{alt}.")
    return index, ref, alt


def trinucleotide_context(sequence: str, index: int) -> str:
    sequence = str(sequence).upper()
    if index <= 0 or index >= len(sequence) - 1:
        return "NNN"
    return sequence[index - 1 : index + 2]


def canonical_sbs_context(trinucleotide: str, ref: str, alt: str) -> tuple[str, str]:
    tri = str(trinucleotide).upper()
    ref = str(ref).upper()
    alt = str(alt).upper()
    if len(tri) != 3 or any(base not in DNA_BASES for base in tri):
        return f"{ref}>{alt}", f"N[{ref}>{alt}]N"

    if ref in {"A", "G"}:
        tri = reverse_complement(tri)
        ref = ref.translate(_COMPLEMENT)
        alt = alt.translate(_COMPLEMENT)

    sbs6 = f"{ref}>{alt}"
    sbs96 = f"{tri[0]}[{sbs6}]{tri[2]}"
    return sbs6, sbs96


def transition_transversion(ref: str, alt: str) -> str:
    return "transition" if {ref, alt} in ({"A", "G"}, {"C", "T"}) else "transversion"


def cpg_effect(wt: str, mut: str, index: int) -> str:
    wt = str(wt).upper()
    mut = str(mut).upper()
    starts = [start for start in (index - 1, index) if 0 <= start < len(wt) - 1]
    wt_sites = {start for start in starts if wt[start : start + 2] == "CG"}
    mut_sites = {start for start in starts if mut[start : start + 2] == "CG"}
    created = mut_sites - wt_sites
    destroyed = wt_sites - mut_sites
    if created and destroyed:
        return "both_created_and_destroyed"
    if created:
        return "created"
    if destroyed:
        return "destroyed"
    return "unchanged"


def _first_present(row: pd.Series, names: tuple[str, ...], default):
    for name in names:
        if name in row.index:
            value = row.get(name)
            if not pd.isna(value):
                return value
    return default


def make_variant_uid(row: pd.Series) -> str:
    candidate_id = _first_present(row, ("Candidate_ID",), None)
    if candidate_id is not None:
        return str(candidate_id)
    genomic = _first_present(row, ("GDC_Genomic_DNA_Change",), "NA")
    probe = _first_present(row, ("probeID", "Target_CpG"), "NA")
    return f"{genomic}|{probe}"


def annotate_variant(
    row: pd.Series,
    wt_sequence: str,
    mutant_sequence: str,
    window_size: int = 1000,
) -> dict:
    wt_sequence = str(wt_sequence).upper()
    mutant_sequence = str(mutant_sequence).upper()
    if len(wt_sequence) != window_size or len(mutant_sequence) != window_size:
        raise ValueError(
            f"Expected {window_size}-bp model inputs, got {len(wt_sequence)} and "
            f"{len(mutant_sequence)}."
        )
    if wt_sequence[CENTER_C_INDEX : CENTER_G_INDEX + 1] != "CG":
        raise ValueError("WT sequence is not centered on a CpG at indices 499:501.")
    if mutant_sequence[CENTER_C_INDEX : CENTER_G_INDEX + 1] != "CG":
        raise ValueError("Mutant sequence alters the protected centered CpG.")

    mutation_index, ref, alt = find_single_snv_difference(wt_sequence, mutant_sequence)
    if mutation_index in PROTECTED_CPG_INDICES:
        raise ValueError("Observed mutation alters the protected centered CpG.")

    tri = trinucleotide_context(wt_sequence, mutation_index)
    sbs6, sbs96 = canonical_sbs_context(tri, ref, alt)
    signed_offset = mutation_index - CENTER_C_INDEX

    gene = _first_present(row, ("Selected_Gene_Name", "Gene"), "Unknown")
    genomic_change = _first_present(row, ("GDC_Genomic_DNA_Change",), np.nan)
    probe = _first_present(row, ("probeID", "Target_CpG"), np.nan)

    return {
        "Variant_UID": make_variant_uid(row),
        "Gene": str(gene),
        "GDC_Genomic_DNA_Change": genomic_change,
        "Target_CpG": probe,
        "Mutation_Window_Index_0based": int(mutation_index),
        "Signed_Offset_From_Target_CpG_C": int(signed_offset),
        "Absolute_Distance_From_Target_CpG": int(abs(signed_offset)),
        "Ref": ref,
        "Alt": alt,
        "Substitution": f"{ref}>{alt}",
        "Trinucleotide_Context": tri,
        "Canonical_SBS6": sbs6,
        "Canonical_SBS96": sbs96,
        "Transition_Transversion": transition_transversion(ref, alt),
        "CpG_Effect": cpg_effect(wt_sequence, mutant_sequence, mutation_index),
    }


@dataclass(frozen=True)
class MatchTier:
    name: str
    same_gene: bool = False
    same_sbs96: bool = False
    same_sbs6: bool = False
    same_titv: bool = False
    same_cpg_effect: bool = False
    max_distance_difference: int | None = None


MATCH_TIERS: tuple[MatchTier, ...] = (
    MatchTier(
        "T1_same_gene_SBS96_CpG_distance25",
        same_gene=True,
        same_sbs96=True,
        same_cpg_effect=True,
        max_distance_difference=25,
    ),
    MatchTier(
        "T2_SBS96_CpG_distance50",
        same_sbs96=True,
        same_cpg_effect=True,
        max_distance_difference=50,
    ),
    MatchTier(
        "T3_SBS96_CpG_distance100",
        same_sbs96=True,
        same_cpg_effect=True,
        max_distance_difference=100,
    ),
    MatchTier(
        "T4_SBS6_CpG_distance100",
        same_sbs6=True,
        same_cpg_effect=True,
        max_distance_difference=100,
    ),
    MatchTier(
        "T5_SBS6_CpG_distance250",
        same_sbs6=True,
        same_cpg_effect=True,
        max_distance_difference=250,
    ),
    MatchTier(
        "T6_TiTv_CpG_distance250",
        same_titv=True,
        same_cpg_effect=True,
        max_distance_difference=250,
    ),
    MatchTier(
        "T7_CpG_distance250",
        same_cpg_effect=True,
        max_distance_difference=250,
    ),
    MatchTier("T8_all_observed_synonymous"),
)


def _tier_mask(candidates: pd.DataFrame, target: pd.Series, tier: MatchTier) -> pd.Series:
    mask = candidates["Variant_UID"] != target["Variant_UID"]
    if tier.same_gene:
        mask &= candidates["Gene"] == target["Gene"]
    if tier.same_sbs96:
        mask &= candidates["Canonical_SBS96"] == target["Canonical_SBS96"]
    if tier.same_sbs6:
        mask &= candidates["Canonical_SBS6"] == target["Canonical_SBS6"]
    if tier.same_titv:
        mask &= candidates["Transition_Transversion"] == target["Transition_Transversion"]
    if tier.same_cpg_effect:
        mask &= candidates["CpG_Effect"] == target["CpG_Effect"]
    if tier.max_distance_difference is not None:
        distance_difference = (
            candidates["Absolute_Distance_From_Target_CpG"]
            - target["Absolute_Distance_From_Target_CpG"]
        ).abs()
        mask &= distance_difference <= tier.max_distance_difference
    return mask


def _stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**32)


def choose_matched_comparators(
    candidates: pd.DataFrame,
    target_index: int,
    min_comparators: int = 20,
    max_comparators: int = 1000,
    random_seed: int = 42,
) -> tuple[np.ndarray, str, bool]:
    target = candidates.iloc[target_index]
    chosen_indices: np.ndarray | None = None
    chosen_tier = MATCH_TIERS[-1].name

    for tier in MATCH_TIERS:
        indices = np.flatnonzero(_tier_mask(candidates, target, tier).to_numpy())
        chosen_indices = indices
        chosen_tier = tier.name
        if len(indices) >= min_comparators:
            break

    if chosen_indices is None or len(chosen_indices) == 0:
        return np.array([], dtype=int), chosen_tier, True

    if len(chosen_indices) > max_comparators:
        rng = np.random.default_rng(_stable_seed(str(target["Variant_UID"]), random_seed))
        chosen_indices = np.sort(
            rng.choice(chosen_indices, size=max_comparators, replace=False)
        )

    insufficient = len(chosen_indices) < min_comparators
    return chosen_indices, chosen_tier, insufficient


def empirical_two_sided_tail_probability(observed: float, controls: np.ndarray) -> float:
    if controls.size == 0 or not np.isfinite(observed):
        return np.nan
    controls = np.asarray(controls, dtype=float)
    controls = controls[np.isfinite(controls)]
    if controls.size == 0:
        return np.nan
    exceedances = np.count_nonzero(np.abs(controls) >= abs(observed))
    return (1.0 + exceedances) / (controls.size + 1.0)


def empirical_percentile(observed: float, controls: np.ndarray) -> float:
    if controls.size == 0 or not np.isfinite(observed):
        return np.nan
    controls = np.asarray(controls, dtype=float)
    controls = controls[np.isfinite(controls)]
    if controls.size == 0:
        return np.nan
    absolute_controls = np.abs(controls)
    less = np.count_nonzero(absolute_controls < abs(observed))
    tied = np.count_nonzero(absolute_controls == abs(observed))
    return 100.0 * (less + 0.5 * tied) / controls.size


def compute_matched_background_statistics(
    scored_variants: pd.DataFrame,
    delta_column: str = "Predicted_Delta_Beta",
    min_comparators: int = 20,
    max_comparators: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    required = {
        "Variant_UID",
        "Gene",
        "Canonical_SBS6",
        "Canonical_SBS96",
        "Transition_Transversion",
        "CpG_Effect",
        "Absolute_Distance_From_Target_CpG",
        delta_column,
    }
    missing = required - set(scored_variants.columns)
    if missing:
        raise ValueError(f"Missing columns for matched-background analysis: {sorted(missing)}")

    results = scored_variants.reset_index(drop=True).copy()
    control_indices_by_uid: dict[str, np.ndarray] = {}
    tail_probabilities: list[float] = []
    tiers: list[str] = []
    counts: list[int] = []
    insufficient_flags: list[bool] = []
    null_means: list[float] = []
    null_medians: list[float] = []
    null_stds: list[float] = []
    null_percentiles: list[float] = []

    all_deltas = pd.to_numeric(results[delta_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(all_deltas).all():
        raise ValueError(f"{delta_column} contains non-finite values")

    for target_index in range(len(results)):
        control_indices, tier, insufficient = choose_matched_comparators(
            results,
            target_index,
            min_comparators=min_comparators,
            max_comparators=max_comparators,
            random_seed=random_seed,
        )
        uid = str(results.iloc[target_index]["Variant_UID"])
        control_indices_by_uid[uid] = control_indices
        controls = all_deltas[control_indices]
        observed = all_deltas[target_index]

        tail_probabilities.append(empirical_two_sided_tail_probability(observed, controls))
        tiers.append(tier)
        counts.append(len(controls))
        insufficient_flags.append(insufficient)
        null_means.append(float(np.mean(controls)) if len(controls) else np.nan)
        null_medians.append(float(np.median(controls)) if len(controls) else np.nan)
        null_stds.append(float(np.std(controls, ddof=1)) if len(controls) > 1 else np.nan)
        null_percentiles.append(empirical_percentile(observed, controls))

    results["Matched_Background_Tier"] = tiers
    results["Matched_Comparator_Count"] = counts
    results["Matched_Comparator_Count_Below_Minimum"] = insufficient_flags
    results["Matched_Background_Mean"] = null_means
    results["Matched_Background_Median"] = null_medians
    results["Matched_Background_STD"] = null_stds
    results["Matched_Background_Absolute_Effect_Percentile"] = null_percentiles
    results["Matched_Empirical_Tail_Probability"] = tail_probabilities
    return results, control_indices_by_uid
