from __future__ import annotations

import pytest

from fly_sniff.topology_inference import GraphEvaluation, topology_randomization_inference


CONDITIONS = tuple(f"episode-{index:03d}" for index in range(20))


def _evaluation(
    graph_id: str,
    statistic: float,
    index: int,
    *,
    family: str | None,
    conditions: tuple[str, ...] = CONDITIONS,
) -> GraphEvaluation:
    return GraphEvaluation(
        graph_id=graph_id,
        graph_sha256=f"{index:064x}",
        statistic_name="mean_spl",
        statistic=statistic,
        condition_ids=conditions,
        null_family=family,
    )


def _nulls(count: int = 31) -> tuple[GraphEvaluation, ...]:
    return tuple(
        _evaluation(
            f"null-{index}",
            statistic=index / 100.0,
            index=index + 1,
            family="directed_degree",
        )
        for index in range(count)
    )


def test_confirmatory_inference_uses_graphs_as_units() -> None:
    intact = _evaluation("intact", 1.0, 999, family=None)
    receipt = topology_randomization_inference(intact, _nulls())

    assert receipt.empirical_p == pytest.approx(1.0 / 32.0)
    assert receipt.intact_rank_best_is_one == 1
    assert receipt.intact_percentile == 1.0
    assert receipt.extreme_or_tied_null_count == 0
    assert receipt.signed_effect_vs_null_mean > 0
    assert receipt.to_dict()["experimental_unit"] == "graph_realization"


def test_ties_count_against_intact_in_empirical_tail_probability() -> None:
    intact = _evaluation("intact", 0.30, 999, family=None)
    nulls = list(_nulls())
    nulls[-1] = _evaluation("null-tie", 0.30, 1001, family="directed_degree")
    receipt = topology_randomization_inference(intact, tuple(nulls))

    assert receipt.extreme_or_tied_null_count == 1
    assert receipt.empirical_p == pytest.approx(2.0 / 32.0)


def test_confirmatory_inference_rejects_too_few_null_graphs() -> None:
    intact = _evaluation("intact", 1.0, 999, family=None)
    with pytest.raises(ValueError, match="at least 31 null graphs"):
        topology_randomization_inference(intact, _nulls(30))


def test_inference_rejects_mismatched_episode_sets() -> None:
    intact = _evaluation("intact", 1.0, 999, family=None)
    nulls = list(_nulls())
    nulls[7] = _evaluation(
        "null-bad-conditions",
        0.07,
        1200,
        family="directed_degree",
        conditions=CONDITIONS[:-1] + ("different-episode",),
    )
    with pytest.raises(ValueError, match="exact paired condition_ids"):
        topology_randomization_inference(intact, tuple(nulls))


def test_inference_rejects_duplicate_null_realizations() -> None:
    intact = _evaluation("intact", 1.0, 999, family=None)
    nulls = list(_nulls())
    nulls[-1] = GraphEvaluation(
        graph_id="duplicate-hash",
        graph_sha256=nulls[0].graph_sha256,
        statistic_name="mean_spl",
        statistic=0.5,
        condition_ids=CONDITIONS,
        null_family="directed_degree",
    )
    with pytest.raises(ValueError, match="must be unique"):
        topology_randomization_inference(intact, tuple(nulls))


def test_inference_rejects_mixed_null_families() -> None:
    intact = _evaluation("intact", 1.0, 999, family=None)
    nulls = list(_nulls())
    nulls[-1] = _evaluation("type-null", 0.5, 1201, family="cell_type")
    with pytest.raises(ValueError, match="one common null_family"):
        topology_randomization_inference(intact, tuple(nulls))


def test_lower_is_better_has_symmetric_tail_test() -> None:
    intact = GraphEvaluation(
        graph_id="intact",
        graph_sha256="f" * 64,
        statistic_name="error",
        statistic=0.0,
        condition_ids=CONDITIONS,
    )
    nulls = tuple(
        GraphEvaluation(
            graph_id=f"null-{index}",
            graph_sha256=f"{index + 1:064x}",
            statistic_name="error",
            statistic=1.0 + index,
            condition_ids=CONDITIONS,
            null_family="directed_degree",
        )
        for index in range(31)
    )
    receipt = topology_randomization_inference(
        intact,
        nulls,
        higher_is_better=False,
    )
    assert receipt.empirical_p == pytest.approx(1.0 / 32.0)
    assert receipt.intact_rank_best_is_one == 1
    assert receipt.signed_effect_vs_null_mean > 0
