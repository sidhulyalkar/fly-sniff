import pandas as pd
import pytest

from fly_sniff.env import Observation
from fly_sniff.graph import GraphBundle, MaleCNSRateController


def _bundle(*, signed: bool, qualified: bool) -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2]})
    edges = pd.DataFrame({"source": [1], "target": [2], "weight": [7]})
    if signed:
        edges["sign"] = [1]
    manifest = {"qualification_status": "qualified"} if qualified else None
    roles = {"odor_left": [1], "steer_right": [2], "steer_left": [1]}
    return GraphBundle(nodes, edges, roles, manifest)


def test_qualified_controller_rejects_unsigned_graph():
    with pytest.raises(ValueError, match="sign"):
        MaleCNSRateController(_bundle(signed=False, qualified=True))


def test_qualified_controller_rejects_candidate_manifest():
    with pytest.raises(ValueError, match="not sealed"):
        MaleCNSRateController(_bundle(signed=True, qualified=False))


def test_development_override_is_explicit():
    controller = MaleCNSRateController(_bundle(signed=True, qualified=False), require_qualified=False)
    assert controller.name == "malecns-rate-v0"


def test_activity_snapshot_uses_real_body_ids_and_declares_modeled_state():
    controller = MaleCNSRateController(_bundle(signed=True, qualified=False), require_qualified=False)
    controller.reset(3)
    obs = Observation(
        left_odor=0.8,
        right_odor=0.0,
        mean_odor=0.4,
        odor_delta=-0.8,
        wind_x_body=0.7,
        wind_y_body=0.0,
        heading=0.0,
    )
    controller.act(obs)
    snapshot = controller.activity_snapshot(limit=2)
    assert snapshot is not None
    assert snapshot["signal_kind"] == "modeled_rate_state"
    assert snapshot["claim_status"] == "candidate"
    assert {cell["body_id"] for cell in snapshot["cells"]}.issubset({1, 2})
    assert snapshot["cells"]
