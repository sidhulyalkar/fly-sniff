import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.r002_probe import run_escape_probe


def _escape_bundle() -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4],
            "target": [5, 5, 6, 6],
            "weight": [12, 12, 12, 12],
            "sign": [1, 1, 1, 1],
        }
    )
    roles = {
        "loom_size_left": [1],
        "loom_velocity_left": [2],
        "loom_size_right": [3],
        "loom_velocity_right": [4],
        "escape_left": [5],
        "escape_right": [6],
    }
    return GraphBundle(nodes, edges, roles, {"qualification_status": "qualified"})


def test_escape_probe_does_not_require_steering_roles():
    report = run_escape_probe(_escape_bundle(), trials=6)
    assert report["direct_stronger_than_near_rate"] > 0.5
    assert report["mean_direct_minus_near_target"] > 0.0
    assert report["mean_direct_lateralization"] > 0.0
    assert "does not infer a turn command" in report["interpretation"]
