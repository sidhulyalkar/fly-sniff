import pandas as pd

from fly_sniff.staged_trace import run_staged_trace


def _annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5, 6],
            "type": ["ORN_demo", "PN_demo", "hDeltaC_demo", "PFL3_demo", "DNa02_demo", "PFN_demo"],
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


def test_staged_trace_keeps_stages_independent(tmp_path):
    config = {
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
            },
            {
                "name": "wind_goal",
                "source": ["^PFN"],
                "target": ["^hDeltaC"],
                "max_hops": 1,
                "min_weight": 5,
                "fanout": 10,
            },
            {
                "name": "missing_seed",
                "source": ["^DOES_NOT_EXIST"],
                "target": ["^DNa02"],
                "max_hops": 2,
                "min_weight": 5,
                "fanout": 10,
            },
        ],
    }

    summary = run_staged_trace(_annotations(), _weights(), config, tmp_path)
    by_name = {stage["name"]: stage for stage in summary["stages"]}

    assert by_name["odor_goal"]["status"] == "candidate_corridor"
    assert by_name["wind_goal"]["status"] == "candidate_corridor"
    assert by_name["missing_seed"]["status"] == "empty_source_seed"
    assert summary["candidate_stage_count"] == 2
    assert not summary["all_stages_have_candidate_corridors"]
    assert (tmp_path / "odor_goal" / "nodes.parquet").exists()
    assert (tmp_path / "staged_trace_report.json").exists()
