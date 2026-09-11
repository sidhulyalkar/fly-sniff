import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.qualification import probe_candidate, qualify_candidate


def candidate_bundle() -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 3, 4, 5],
            "target": [4, 5, 4, 5, 4, 5],
            "weight": [10, 10, 4, 4, 16, 16],
            "sign": [1, 1, 1, 1, 1, 1],
        }
    )
    roles = {
        "odor_left": [1],
        "odor_right": [2],
        "wind_forward": [3],
        "wind_backward": [],
        "wind_left": [],
        "wind_right": [],
        "steer_left": [4],
        "steer_right": [5],
    }
    return GraphBundle(nodes, edges, roles, {"dataset": "male-cns:v1.0", "qualification_status": "candidate"})


def test_probe_is_bilateral_deterministic_and_lesion_sensitive():
    result = probe_candidate(candidate_bundle(), seed=7)
    assert result.left_turn < 0
    assert result.right_turn > 0
    assert result.opposite_sign
    assert result.separation > 0.05
    assert result.lesioned_peak_turn <= 1e-12
    assert result.deterministic_error <= 1e-12
    assert result.blank_retention > 0.05


def test_candidate_can_pass_model_sanity_without_becoming_qualified():
    bundle = candidate_bundle()
    report = qualify_candidate(bundle, seed=7)
    assert report["passed"]
    assert report["passed_gate_count"] == report["gate_count"]
    assert bundle.manifest["qualification_status"] == "candidate"


def test_missing_role_fails_gate():
    bundle = candidate_bundle()
    bundle.roles["odor_right"] = []
    report = qualify_candidate(bundle, seed=7)
    assert not report["passed"]
    role_gate = next(g for g in report["gates"] if g["name"] == "required_roles")
    assert not role_gate["passed"]
