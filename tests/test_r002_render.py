import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.r002_render import collect_trace


def _render_bundle() -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6, 7, 8]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4, 1, 3, 7, 8],
            "target": [5, 5, 6, 6, 7, 8, 6, 5],
            "weight": [12, 10, 12, 10, 3, 3, 2, 2],
            "sign": [1, 1, 1, 1, 1, 1, 1, 1],
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


def test_collect_trace_uses_identical_frames_for_intact_and_rewire():
    trace = collect_trace(_render_bundle(), side="left", rewire_seed=99)
    assert trace["trajectory"] == "direct-hit"
    assert len(trace["frames"]) == len(trace["intact"]["target"])
    assert len(trace["frames"]) == len(trace["rewired"]["target"])
    assert max(trace["intact"]["target"]) > 0.0
    assert trace["frames"][0]["inputs"] == {
        "loom_size_left": 0.0,
        "loom_size_right": 0.0,
        "loom_velocity_left": 0.0,
        "loom_velocity_right": 0.0,
    }
