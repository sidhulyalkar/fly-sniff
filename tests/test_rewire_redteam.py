import pandas as pd
import pytest

from fly_sniff.graph import GraphBundle
from fly_sniff.rewire_redteam import summarize_rewire_ensemble


def _bundle(targets, *, weights=(1.0, 2.0, 3.0, 4.0)):
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4],
            "target": list(targets),
            "weight": list(weights),
            "sign": [1, -1, 1, -1],
        }
    )
    return GraphBundle(
        nodes,
        edges,
        {},
        {"qualification_status": "candidate", "dataset": "synthetic-rewire-redteam"},
    )


def test_rewire_similarity_report_requires_unique_nonintact_topologies():
    intact = _bundle([2, 3, 4, 1])
    first = _bundle([3, 4, 1, 2])
    second = _bundle([4, 1, 2, 3])
    report = summarize_rewire_ensemble(
        intact,
        [(101, first), (102, second)],
        expected_count=2,
    )
    assert report["all_rewire_fingerprints_unique"] is True
    assert report["all_rewires_distinct_from_intact"] is True
    assert report["status"] == "identity_checks_passed_similarity_diagnostic_only"
    assert report["summary"]["mean_edge_overlap_vs_intact"] == pytest.approx(0.0)
    assert report["summary"]["mean_pairwise_edge_overlap"] == pytest.approx(0.0)
    assert "stationarity" in report["claim_boundary"]


def test_duplicate_rewire_fingerprint_is_rejected():
    intact = _bundle([2, 3, 4, 1])
    first = _bundle([3, 4, 1, 2])
    with pytest.raises(RuntimeError, match="duplicate topology fingerprints"):
        summarize_rewire_ensemble(
            intact,
            [(101, first), (102, first)],
            expected_count=2,
        )


def test_rewire_identical_to_intact_is_rejected_even_with_different_seed():
    intact = _bundle([2, 3, 4, 1])
    identical = _bundle([2, 3, 4, 1])
    with pytest.raises(RuntimeError, match="intact topology exactly"):
        summarize_rewire_ensemble(intact, [(999, identical)])


def test_rewire_cannot_mutate_presynaptic_weight_or_sign_contract():
    intact = _bundle([2, 3, 4, 1])
    altered = _bundle([3, 4, 1, 2], weights=(1.0, 2.0, 30.0, 4.0))
    with pytest.raises(RuntimeError, match="weight/sign"):
        summarize_rewire_ensemble(intact, [(101, altered)])
