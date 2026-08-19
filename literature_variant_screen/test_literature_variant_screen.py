#!/usr/bin/env python3
"""Small offline tests for coordinate, consequence, and exclusion logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import pandas as pd


SCRIPT = Path(__file__).with_name("15_literature_variant_screen.py")
SPEC = importlib.util.spec_from_file_location("literature_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_spdi_is_zero_based_and_grch38_specific() -> None:
    assert MODULE.parse_spdi("NC_000017.11:43071076:A:G") == (
        "chr17", 43071077, "A", "G"
    )
    assert MODULE.parse_spdi("NC_000017.10:43071076:A:G") is None


def test_locus_key_ignores_allele_direction() -> None:
    assert MODULE.locus_key("17", 10, "A", "G") == MODULE.locus_key(
        "chr17", 10, "G", "A"
    )


def test_clinvar_summary_filter_and_parse() -> None:
    sample = {
        "uid": "1",
        "title": "NM_1(GENE):c.1A>G (p.Lys1Glu)",
        "genes": [{"symbol": "GENE"}],
        "molecular_consequence_list": ["missense variant"],
        "protein_change": "K1E",
        "germline_classification": {"description": "Pathogenic"},
        "variation_set": [{
            "canonical_spdi": "NC_000017.11:43071076:A:G",
            "variation_xrefs": [{"db_source": "dbSNP", "db_id": "123"}],
        }],
    }
    rows = MODULE.clinvar_to_candidates([sample], "GENE", "pathogenic")
    assert len(rows) == 1
    assert rows[0]["Rsid"] == "rs123"
    assert rows[0]["Position_1based"] == 43071077


def test_pathogenic_query_rechecks_current_classification() -> None:
    sample = {
        "uid": "2",
        "title": "NM_1(GENE):c.1A>G (p.Lys1Glu)",
        "genes": [{"symbol": "GENE"}],
        "molecular_consequence_list": ["missense variant"],
        "germline_classification": {"description": "Uncertain significance"},
        "variation_set": [{
            "canonical_spdi": "NC_000017.11:43071076:A:G",
        }],
    }
    assert MODULE.clinvar_to_candidates([sample], "GENE", "pathogenic") == []


def test_current_data_and_window_exclusions() -> None:
    candidates = pd.DataFrame([
        {
            "Variant_ID": "current:A>G", "Locus_Key": "chr8:100:A:G",
            "Gene": "A", "chr": "chr8", "Position_1based": 100,
            "Ref": "A", "Alt": "G", "Discovery_Source": "test",
        },
        {
            "Variant_ID": "eligible:C>T", "Locus_Key": "chr8:500:C:T",
            "Gene": "B", "chr": "chr8", "Position_1based": 500,
            "Ref": "C", "Alt": "T", "Discovery_Source": "test",
        },
        {
            "Variant_ID": "far:G>A", "Locus_Key": "chr8:9000:A:G",
            "Gene": "C", "chr": "chr8", "Position_1based": 9000,
            "Ref": "G", "Alt": "A", "Discovery_Source": "test",
        },
    ])
    probes = pd.DataFrame([
        {"probeID": "cg1", "chr": "chr8", "pos": 499, "Model_Split": "test"},
    ])
    eligible, audit = MODULE.audit_and_prescreen(
        candidates, {"chr8:100:A:G"}, set(), probes, False
    )
    assert eligible["Variant_ID"].tolist() == ["eligible:C>T"]
    status = audit.set_index("Variant_ID")["Status"].to_dict()
    assert status["current:A>G"].startswith("excluded_variant_already")
    assert status["far:G>A"] == "excluded_no_HM450_CpG_within_model_window"


def test_ranking_uses_prespecified_target_not_largest_nearby_delta() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        scorer = root / "scorer"
        scorer.mkdir()
        pd.DataFrame([
            {
                "Published_Variant_ID": "rs1:A>G", "probeID": "cg_nearest",
                "Predicted_Delta_Beta": 0.02, "Model_Split": "test",
                "Is_Primary_Nearest_Target": True,
            },
            {
                "Published_Variant_ID": "rs1:A>G", "probeID": "cg_larger",
                "Predicted_Delta_Beta": 0.20, "Model_Split": "test",
                "Is_Primary_Nearest_Target": False,
            },
        ]).to_csv(scorer / "known_variant_predictions_ensemble.csv", index=False)
        metadata = pd.DataFrame([{
            "Variant_ID": "rs1:A>G", "Locus_Key": "chr8:100:A:G",
        }])
        ranked = MODULE.rank_predictions(
            scorer, metadata, pd.DataFrame(), set(), root / "out",
        )
        assert ranked["probeID"].tolist() == ["cg_nearest"]
        all_pairs = pd.read_csv(root / "out" / "literature_variant_predictions_all_pairs.csv")
        assert set(all_pairs["probeID"]) == {"cg_nearest", "cg_larger"}


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} offline tests passed")
