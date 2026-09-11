import pandas as pd
import pytest

from fly_sniff.graph import GraphBundle
from fly_sniff.replay import replay_frames
from fly_sniff.runtime import ConnectomeRuntime
from fly_sniff.showcase import SPECS, assess


def _bundle(*, qualified: bool = True) -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3],
            "target": [3, 3, 4],
            "weight": [10, 4, 8],
            "sign": [1, -1, 1],
        }
    )
    roles = {
        "odor_left": [1],
        "odor_right": [2],
        "steer_left": [3],
        "steer_right": [4],
    }
    manifest = {"qualification_status": "qualified"} if qualified else None
    return GraphBundle(nodes, edges, roles, manifest)


def test_runtime_rejects_unknown_input_role():
    runtime = ConnectomeRuntime(_bundle())
    with pytest.raises(ValueError, match="unknown roles"):
        runtime.step({"invented_sensor": 1.0})


def test_runtime_rejects_non_finite_input():
    runtime = ConnectomeRuntime(_bundle())
    with pytest.raises(ValueError, match="finite"):
        runtime.step({"odor_left": float("nan")})


def test_runtime_produces_named_readouts_and_resets():
    runtime = ConnectomeRuntime(_bundle())
    first = runtime.step({"odor_left": 1.0}, readouts=["steer_left", "steer_right"])
    assert first.step == 1
    assert set(first.readouts) == {"steer_left", "steer_right"}
    assert first.activity_max > 0.0

    runtime.reset()
    assert runtime.step_index == 0
    assert runtime.role_mean("steer_left") == 0.0


def test_replay_is_deterministic_for_identical_frames():
    frames = [
        {"t": 0.0, "inputs": {"odor_left": 1.0, "odor_right": 0.0}},
        {"t": 0.1, "inputs": {"odor_left": 0.5, "odor_right": 0.2}},
    ]
    a = replay_frames(_bundle(), frames, readouts=["steer_left", "steer_right"])
    b = replay_frames(_bundle(), frames, readouts=["steer_left", "steer_right"])
    assert a == b
    assert [row["source_t"] for row in a] == [0.0, 0.1]


def test_candidate_replay_requires_explicit_override():
    frames = [{"inputs": {"odor_left": 1.0}}]
    with pytest.raises(ValueError, match="not sealed"):
        replay_frames(_bundle(qualified=False), frames, readouts=["steer_left"])

    trace = replay_frames(
        _bundle(qualified=False),
        frames,
        readouts=["steer_left"],
        require_qualified=False,
    )
    assert len(trace) == 1


def test_showcase_audit_separates_interface_ready_from_claim_ready():
    candidate = _bundle(qualified=False)
    report = assess(candidate, SPECS["odor-choice"])
    assert report["interface_ready"] is True
    assert report["male_cns_claim_allowed"] is False

    qualified = _bundle(qualified=True)
    report = assess(qualified, SPECS["odor-choice"])
    assert report["interface_ready"] is True
    assert report["male_cns_claim_allowed"] is True


def test_showcase_audit_reports_missing_visual_roles():
    report = assess(_bundle(), SPECS["visual-replay"])
    assert report["interface_ready"] is False
    assert report["missing_required_roles"] == ["vision_left", "vision_right"]
