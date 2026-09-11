import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.looming import benchmark_loom, generate_loom_frames


def _loom_bundle() -> GraphBundle:
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
        "steer_left": [5],
        "steer_right": [6],
        "escape": [5, 6],
    }
    return GraphBundle(nodes, edges, roles, {"qualification_status": "qualified"})


def test_direct_hit_expands_more_than_near_miss():
    hit = generate_loom_frames("left", "direct-hit")
    miss = generate_loom_frames("left", "near-miss")
    assert max(frame.angular_size_rad for frame in hit) > max(
        frame.angular_size_rad for frame in miss
    )
    assert max(frame.angular_velocity_rad_s for frame in hit) > 0.0


def test_lateral_adapter_only_drives_stimulated_side():
    frames = generate_loom_frames("right", "direct-hit")
    active = frames[-1].inputs
    assert active["loom_size_left"] == 0.0
    assert active["loom_velocity_left"] == 0.0
    assert active["loom_size_right"] > 0.0


def test_synthetic_escape_graph_turns_away_from_loom():
    report = benchmark_loom(_loom_bundle(), trials=6)
    assert report["away_accuracy_direct_hit"] == 1.0
    assert report["mean_signed_away_margin"] > 0.0
    assert report["escape_direct_minus_near_miss"] is not None
    assert report["escape_direct_minus_near_miss"] > 0.0
