import pandas as pd
import pytest

from fly_sniff.graph import GraphBundle
from fly_sniff.neural_scene import build_neural_scene


def _scene_bundle(*, with_positions: bool = True) -> GraphBundle:
    nodes = pd.DataFrame(
        {
            "bodyId": [101, 202, 303],
            "type": ["ORN-A", "hDeltaC-1", "DNa02-1"],
            "instance": ["ORN-A_R", "hDeltaC-1", "DNa02-1_R"],
        }
    )
    if with_positions:
        nodes["somaLocation"] = [
            [100.0, 50.0, 300.0],
            {"coordinates": [200.0, 60.0, 200.0]},
            "[300.0, 70.0, 100.0]",
        ]
    edges = pd.DataFrame(
        {
            "source": [101, 202],
            "target": [202, 303],
            "weight": [12, 9],
            "sign": [1, -1],
        }
    )
    roles = {
        "odor_right": [101],
        "navigation": [202],
        "steer_right": [303],
    }
    manifest = {"dataset": "male-cns:v1.0", "qualification_status": "candidate"}
    return GraphBundle(nodes, edges, roles, manifest)


def test_scene_uses_source_coordinates_and_exact_body_ids():
    scene = build_neural_scene(_scene_bundle())
    assert scene["dataset"] == "male-cns:v1.0"
    assert scene["nodes_positioned"] == 3
    assert scene["nodes_missing_position"] == 0
    assert {node["body_id"] for node in scene["nodes"]} == {101, 202, 303}
    assert len(scene["edges"]) == 2
    assert "NOT A QUALIFIED RESULT" in scene["claim_label"]
    assert all(-1.2 <= node["screen_x"] <= 1.2 for node in scene["nodes"])
    assert all(-1.2 <= node["screen_y"] <= 1.2 for node in scene["nodes"])


def test_scene_refuses_to_fabricate_anatomical_positions():
    with pytest.raises(ValueError, match="synthetic positions are forbidden"):
        build_neural_scene(_scene_bundle(with_positions=False))
