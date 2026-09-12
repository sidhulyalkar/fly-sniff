import numpy as np
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


def _observation(left: float, right: float) -> Observation:
    return Observation(
        left_odor=left,
        right_odor=right,
        mean_odor=0.5 * (left + right),
        odor_delta=right - left,
        wind_x_body=0.0,
        wind_y_body=0.0,
        heading=0.0,
    )


def _drive_only_bundle() -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2]})
    edges = pd.DataFrame(
        {
            "source": pd.Series(dtype=int),
            "target": pd.Series(dtype=int),
            "weight": pd.Series(dtype=float),
            "sign": pd.Series(dtype=int),
        }
    )
    roles = {
        "odor_left": [1],
        "odor_right": [2],
        "steer_left": [1],
        "steer_right": [2],
    }
    return GraphBundle(nodes, edges, roles, {"qualification_status": "candidate"})


def test_qualified_controller_rejects_unsigned_graph():
    with pytest.raises(ValueError, match="sign"):
        MaleCNSRateController(_bundle(signed=False, qualified=True))


def test_qualified_controller_rejects_candidate_manifest():
    with pytest.raises(ValueError, match="not sealed"):
        MaleCNSRateController(_bundle(signed=True, qualified=False))


def test_development_override_is_explicit():
    controller = MaleCNSRateController(_bundle(signed=True, qualified=False), require_qualified=False)
    assert controller.name == "malecns-rate-v0"


def test_graph_rejects_duplicate_body_ids():
    bundle = _bundle(signed=True, qualified=False)
    bundle.nodes.loc[1, "bodyId"] = 1
    with pytest.raises(ValueError, match="duplicate body IDs"):
        bundle.validate(require_sign=True)


def test_graph_rejects_null_signs_instead_of_propagating_nan_activity():
    bundle = _bundle(signed=True, qualified=False)
    bundle.edges.loc[0, "sign"] = np.nan
    with pytest.raises(ValueError, match="null"):
        bundle.validate(require_sign=True)


def test_graph_rejects_nonpositive_structural_weights():
    bundle = _bundle(signed=True, qualified=False)
    bundle.edges.loc[0, "weight"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        bundle.validate(require_sign=True)


def test_role_semantics_map_left_activity_to_positive_left_turn():
    controller = MaleCNSRateController(_drive_only_bundle(), require_qualified=False)
    controller.reset(1)
    left_turn = controller.act(_observation(1.0, 0.0)).turn

    controller.reset(1)
    right_turn = controller.act(_observation(0.0, 1.0)).turn

    assert left_turn > 0.0
    assert right_turn < 0.0


def test_rate_relaxation_is_time_consistent_for_constant_drive():
    bundle = _drive_only_bundle()
    slow = MaleCNSRateController(bundle, model_dt_s=0.05, require_qualified=False)
    fast = MaleCNSRateController(bundle, model_dt_s=0.025, require_qualified=False)
    slow.reset(3)
    fast.reset(3)
    obs = _observation(1.0, 0.0)

    for _ in range(20):
        slow.act(obs)
    for _ in range(40):
        fast.act(obs)

    assert np.allclose(slow.activity, fast.activity, rtol=0.0, atol=1e-12)


def test_activity_snapshot_uses_real_body_ids_and_declares_modeled_state():
    controller = MaleCNSRateController(
        _bundle(signed=True, qualified=False),
        require_qualified=False,
    )
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


def test_input_snapshot_binds_observation_values_to_exact_role_body_ids():
    controller = MaleCNSRateController(_drive_only_bundle(), require_qualified=False)
    controller.reset(4)
    assert controller.input_snapshot() is None

    obs = Observation(
        left_odor=0.8,
        right_odor=0.2,
        mean_odor=0.5,
        odor_delta=-0.6,
        wind_x_body=0.7,
        wind_y_body=-0.3,
        heading=0.0,
    )
    controller.act(obs)
    snapshot = controller.input_snapshot()
    assert snapshot is not None
    assert snapshot["signal_kind"] == "modeled_role_drive"
    assert snapshot["interface_status"] == "modeled_interface_not_peripheral_sensory_qualification"
    by_role = {entry["role"]: entry for entry in snapshot["roles"]}
    assert by_role["odor_left"] == {"role": "odor_left", "value": 0.8, "body_ids": [1]}
    assert by_role["odor_right"] == {"role": "odor_right", "value": 0.2, "body_ids": [2]}
    assert by_role["wind_forward"]["value"] == pytest.approx(0.7)
    assert by_role["wind_right"]["value"] == pytest.approx(0.3)
