import numpy as np
import pandas as pd

from fly_sniff.flyarc import (
    ARCFrameEncoder,
    FlyARCReservoir,
    LinearQPolicy,
    RunConfig,
    degree_preserving_rewire,
    select_structural_core,
    shaped_reward,
)
from fly_sniff.graph import GraphBundle


def _bundle() -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "target": [2, 3, 4, 5, 6, 1, 4, 5, 6, 1, 2, 3],
            "weight": [9, 8, 7, 6, 5, 4, 7, 7, 6, 6, 5, 5],
            "sign": [1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1],
        }
    )
    return GraphBundle(nodes=nodes, edges=edges, roles={}, manifest=None)


def _degree_pairs(edges: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    out_degree = edges.groupby("source").size().astype(int).to_dict()
    in_degree = edges.groupby("target").size().astype(int).to_dict()
    return out_degree, in_degree


def test_arc_frame_encoder_keeps_colors_categorical_and_tracks_change():
    encoder = ARCFrameEncoder(pool=2, colors=16)
    frame = np.array([[1, 1], [2, 2]], dtype=int)
    first = encoder.encode(frame)
    second = encoder.encode(np.array([[1, 3], [2, 2]], dtype=int))

    assert encoder.feature_dim == 2 * 2 * 17
    assert first.shape == (encoder.feature_dim,)
    assert np.isclose(first[-4:].sum(), 0.0)
    assert second[-4:].sum() > 0.0


def test_structural_core_selection_is_deterministic():
    a = select_structural_core(_bundle(), max_nodes=4)
    b = select_structural_core(_bundle(), max_nodes=4)

    assert a.body_ids == b.body_ids
    assert a.policy == b.policy
    assert len(a.body_ids) == 4
    assert set(a.edges["source"]).issubset(set(a.body_ids))
    assert set(a.edges["target"]).issubset(set(a.body_ids))


def test_degree_preserving_rewire_preserves_exact_directed_degrees():
    original = _bundle().edges
    rewired, swaps = degree_preserving_rewire(
        original,
        seed=17,
        swaps_per_edge=2,
    )

    assert swaps > 0
    assert _degree_pairs(rewired) == _degree_pairs(original)
    assert len(rewired) == len(original)
    assert not (rewired["source"] == rewired["target"]).any()
    assert not rewired.duplicated(["source", "target"]).any()
    assert rewired[["source", "target"]].values.tolist() != (
        original[["source", "target"]].values.tolist()
    )


def test_variants_share_input_projection_and_state_dimension():
    core = select_structural_core(_bundle(), max_nodes=6)
    intact = FlyARCReservoir(
        core,
        variant="intact",
        feature_dim=32,
        projection_seed=31,
        topology_seed=47,
    )
    rewired = FlyARCReservoir(
        core,
        variant="rewire",
        feature_dim=32,
        projection_seed=31,
        topology_seed=47,
    )
    stateless = FlyARCReservoir(
        core,
        variant="stateless",
        feature_dim=32,
        projection_seed=31,
        topology_seed=47,
    )

    assert intact.activity.shape == rewired.activity.shape == stateless.activity.shape
    assert (intact.projection != rewired.projection).nnz == 0
    assert (intact.projection != stateless.projection).nnz == 0


def test_stateless_control_has_no_hidden_recurrent_memory():
    core = select_structural_core(_bundle(), max_nodes=6)
    features = np.zeros(32, dtype=np.float32)
    features[3] = 1.0

    stateless = FlyARCReservoir(
        core,
        variant="stateless",
        feature_dim=32,
        projection_seed=7,
        topology_seed=11,
    )
    first = stateless.step(features)
    second = stateless.step(features)

    assert np.allclose(first, second)


def test_linear_q_policy_updates_only_selected_action_row():
    policy = LinearQPolicy(3, 8, seed=5, epsilon=0.0)
    state = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    next_state = state[::-1].copy()

    before = policy.weights.copy()
    td = policy.update(
        state,
        1,
        1.0,
        next_state,
        done=True,
        valid_next=range(3),
    )

    assert td > 0.0
    assert np.array_equal(policy.weights[0], before[0])
    assert not np.array_equal(policy.weights[1], before[1])
    assert np.array_equal(policy.weights[2], before[2])


def test_reward_contract_is_explicit_and_monotonic_for_progress():
    config = RunConfig()
    baseline = shaped_reward(
        previous_levels=0,
        next_levels=0,
        state_name="NOT_FINISHED",
        novel_frame=False,
        config=config,
    )
    novel = shaped_reward(
        previous_levels=0,
        next_levels=0,
        state_name="NOT_FINISHED",
        novel_frame=True,
        config=config,
    )
    level = shaped_reward(
        previous_levels=0,
        next_levels=1,
        state_name="NOT_FINISHED",
        novel_frame=False,
        config=config,
    )

    assert novel > baseline
    assert level > novel
