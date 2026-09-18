from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.controllers import BilateralProxyController
from fly_sniff.env import Observation
from fly_sniff.graph import GraphBundle
from fly_sniff.live_showcase import SCHEMA, SensoryMaskController, export_live_showcase


def _write_graph(root: Path, *, qualified: bool) -> Path:
    root.mkdir()
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4, 1, 3],
            "target": [5, 5, 6, 6, 6, 5],
            "weight": [10, 8, 10, 8, 3, 3],
            "sign": [1, 1, 1, 1, 1, 1],
        }
    )
    roles = {
        "odor_left": [1],
        "odor_right": [2],
        "wind_forward": [3],
        "wind_backward": [4],
        "wind_left": [3],
        "wind_right": [4],
        "steer_left": [5],
        "steer_right": [6],
    }
    nodes.to_parquet(root / "nodes.parquet", index=False)
    edges.to_parquet(root / "edges.parquet", index=False)
    (root / "roles.json").write_text(json.dumps(roles))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "qualification_status": "qualified" if qualified else "candidate",
                "scientific_claim_allowed": qualified,
                "qualified_experiments": ["odor-plume"] if qualified else [],
            }
        )
    )
    return root


def test_development_showcase_is_visibly_non_claim_bearing(tmp_path: Path) -> None:
    output = tmp_path / "showcase.json"
    payload = export_live_showcase(output, seconds=0.5, sample_hz=5)

    assert payload["schema"] == SCHEMA
    assert payload["mode"] == "development-proxy"
    assert payload["claim_allowed"] is False
    assert payload["evidence_level"] == "development_proxy"
    assert payload["stimulus_contract"]["odor_identity_mode"] == "generic_scalar_concentration"
    assert payload["stimulus_contract"]["identity_specific_behavior_claim_allowed"] is False
    assert payload["connectome"]["available"] is False
    assert len(payload["conditions"]) == 3
    assert {row["key"] for row in payload["conditions"]} == {
        "proxy-intact",
        "left-antenna-off",
        "odor-blind",
    }
    assert len(payload["frames"]) > 0
    assert output.is_file()


def test_development_conditions_share_one_exogenous_plume_clock(tmp_path: Path) -> None:
    payload = export_live_showcase(tmp_path / "showcase.json", seconds=0.5, sample_hz=10)
    frame = payload["frames"][0]
    assert frame["plume"]
    assert set(frame["agents"]) == {
        "proxy-intact",
        "left-antenna-off",
        "odor-blind",
    }
    starts = {
        (row["x"], row["y"], row["heading"])
        for row in frame["agents"].values()
    }
    assert len(starts) == 1
    assert payload["pairing_checks"]["identical_initial_agent_state"] is True
    assert payload["pairing_checks"]["identical_exogenous_plume_at_exported_frames"] is True


def test_sensory_mask_reports_effective_controller_input() -> None:
    controller = SensoryMaskController(
        BilateralProxyController(),
        left_scale=0.0,
        right_scale=1.0,
    )
    controller.reset(7)
    observation = Observation(
        left_odor=0.8,
        right_odor=0.2,
        mean_odor=0.5,
        odor_delta=-0.6,
        wind_x_body=0.1,
        wind_y_body=-0.2,
        heading=0.3,
    )
    controller.act(observation)
    diagnostics = controller.diagnostics()

    assert diagnostics["input_left_odor"] == 0.0
    assert diagnostics["input_right_odor"] == 0.2


def test_candidate_graph_refuses_claim_bearing_export(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "candidate", qualified=False)
    with pytest.raises(ValueError, match="requires an odor-plume-qualified GraphBundle"):
        export_live_showcase(
            tmp_path / "showcase.json",
            seconds=0.2,
            sample_hz=5,
            graph_dir=graph,
        )


def test_candidate_graph_can_render_only_when_explicitly_allowed(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "candidate", qualified=False)
    payload = export_live_showcase(
        tmp_path / "showcase.json",
        seconds=0.2,
        sample_hz=5,
        graph_dir=graph,
        allow_candidate=True,
    )

    assert payload["claim_allowed"] is False
    assert payload["connectome"]["available"] is True
    assert payload["connectome"]["geometry_kind"] == "topology_only"
    assert "not anatomical XYZ morphology" in payload["connectome"]["claim_boundary"]



def test_generic_qualification_does_not_unlock_odor_navigation(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "generic-qualified", qualified=True)
    manifest_path = graph / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["qualified_experiments"] = ["loom-escape"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="odor-plume-qualified"):
        export_live_showcase(
            tmp_path / "showcase.json",
            seconds=0.2,
            sample_hz=5,
            graph_dir=graph,
        )


def test_qualified_graph_exports_paired_topology_controls(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "qualified", qualified=True)
    payload = export_live_showcase(
        tmp_path / "showcase.json",
        seconds=0.2,
        sample_hz=5,
        graph_dir=graph,
    )

    assert payload["claim_allowed"] is True
    assert payload["evidence_level"] == "qualified_modeled_circuit"
    keys = {row["key"] for row in payload["conditions"]}
    assert keys == {
        "malecns-intact",
        "degree-preserving-rewire",
        "malecns-odor-blind",
    }
    GraphBundle.load(graph).validate(require_sign=True, require_qualified=True)
