import json

import pandas as pd
import pytest

from fly_sniff.staged_trace import run_staged_trace


def _annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5, 6],
            "type": [
                "ORN_demo",
                "PN_demo",
                "hDeltaC_demo",
                "PFL3_demo",
                "DNa02_demo",
                "PFN_demo",
            ],
            "side": ["L", "L", "L", "R", "R", "R"],
        }
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [1, 2, 6, 3, 4],
            "target": [2, 3, 3, 4, 5],
            "weight": [10, 10, 12, 11, 9],
        }
    )


def _config() -> dict:
    return {
        "dataset": "synthetic",
        "purpose": "test",
        "stages": [
            {
                "name": "odor_goal",
                "source": ["^ORN"],
                "target": ["^hDeltaC"],
                "max_hops": 3,
                "min_weight": 5,
                "fanout": 10,
                "hypothesis_class": "primary_navigation",
            },
            {
                "name": "wind_goal",
                "source": ["^PFN"],
                "target": ["^hDeltaC"],
                "max_hops": 1,
                "min_weight": 5,
                "fanout": 10,
                "hypothesis_class": "primary_navigation",
            },
            {
                "name": "missing_memory_seed",
                "source": ["^DOES_NOT_EXIST"],
                "target": ["^DNa02"],
                "max_hops": 2,
                "min_weight": 5,
                "fanout": 10,
                "hypothesis_class": "odor_blank_memory",
                "required_for_primary_hypothesis": False,
            },
        ],
    }


def test_staged_trace_seals_seeds_audits_and_keeps_optional_stages_independent(tmp_path):
    provenance = {
        "annotations": {"name": "annotations.feather", "sha256": "a" * 64},
        "weights": {"name": "weights.feather", "sha256": "b" * 64},
        "config": {"name": "config.json", "sha256": "c" * 64},
    }
    summary = run_staged_trace(
        _annotations(),
        _weights(),
        _config(),
        tmp_path,
        input_provenance=provenance,
    )
    by_name = {stage["name"]: stage for stage in summary["stages"]}

    assert by_name["odor_goal"]["status"] == "candidate_corridor"
    assert by_name["odor_goal"]["structural_audit_passed"] is True
    assert by_name["wind_goal"]["status"] == "candidate_corridor"
    assert by_name["wind_goal"]["structural_audit_passed"] is True
    assert by_name["missing_memory_seed"]["status"] == "empty_source_seed"
    assert by_name["missing_memory_seed"]["structural_audit_passed"] is None
    assert by_name["missing_memory_seed"]["required_for_primary_hypothesis"] is False

    assert summary["protocol"] == "staged-structural-discovery-v2"
    assert summary["candidate_stage_count"] == 2
    assert not summary["all_stages_have_candidate_corridors"]
    assert summary["required_stage_count"] == 2
    assert summary["required_candidate_stage_count"] == 2
    assert summary["required_stage_audit_pass_count"] == 2
    assert summary["all_required_stages_have_audited_candidate_corridors"]
    assert summary["input_provenance"] == provenance

    odor_dir = tmp_path / "odor_goal"
    assert (odor_dir / "nodes.parquet").exists()
    assert (odor_dir / "source_seeds.csv").exists()
    assert (odor_dir / "target_seeds.csv").exists()
    assert (odor_dir / "structural_audit.json").exists()
    assert (tmp_path / "staged_trace_report.json").exists()

    source_seeds = pd.read_csv(odor_dir / "source_seeds.csv")
    assert source_seeds.bodyId.tolist() == [1]
    assert source_seeds.type.tolist() == ["ORN_demo"]
    audit = json.loads((odor_dir / "structural_audit.json").read_text())
    assert audit["passed"] is True


def test_staged_trace_rejects_config_without_primary_required_stage(tmp_path):
    config = _config()
    for stage in config["stages"]:
        stage["required_for_primary_hypothesis"] = False

    with pytest.raises(ValueError, match="at least one primary required stage"):
        run_staged_trace(_annotations(), _weights(), config, tmp_path)
