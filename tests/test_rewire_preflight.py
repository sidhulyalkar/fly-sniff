from __future__ import annotations

import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.rewire_preflight import PROTOCOL, run_preflight


def _bundle() -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": list(range(20))})
    rows = []
    for i in range(20):
        rows.append({"source": i, "target": (i + 1) % 20, "weight": 1.0 + (i % 3), "sign": 1})
        rows.append({"source": i, "target": (i + 5) % 20, "weight": 2.0 + (i % 4), "sign": -1 if i % 2 else 1})
    return GraphBundle(
        nodes=nodes,
        edges=pd.DataFrame(rows),
        roles={},
        manifest={"protocol": "toy", "qualification_status": "candidate"},
    )


def _config() -> dict:
    return {
        "protocol": PROTOCOL,
        "dataset": "male-cns:v1.0",
        "rewire": {
            "swaps_per_edge": 1,
            "minimum_changed_edge_fraction": 0.25,
            "seeds": [11, 12],
            "minimum_independent_rewires_for_final_claim": 2,
        },
        "lesions": ["a", "b", "c", "d"],
        "equal_compute_contract": {
            "same_episode_seeds": True,
            "same_plume_realizations": True,
            "same_initial_states": True,
            "same_controller_step_budget": True,
            "same_optimization_budget": True,
            "same_model_parameters": True,
            "no_variant_specific_early_stopping": True,
        },
        "claim_boundary": "test",
    }


def test_rewire_preflight_preserves_contract_and_is_deterministic() -> None:
    report = run_preflight(_bundle(), _config())
    assert report["passed"] is True
    assert report["passed_gate_count"] == report["gate_count"]
    for row in report["rewire_reports"]:
        assert row["accepted_swaps"] == row["required_swaps"]
        assert row["changed_edge_fraction"] >= 0.25
        assert all(row["checks"].values())


def test_rewire_preflight_fails_an_impossible_mixing_floor() -> None:
    config = _config()
    config["rewire"]["minimum_changed_edge_fraction"] = 1.01
    report = run_preflight(_bundle(), config)
    assert report["passed"] is False
    assert any(not row["checks"]["minimum_changed_edge_fraction"] for row in report["rewire_reports"])
