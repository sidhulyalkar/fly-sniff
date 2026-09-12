import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text())


def test_hdelta_c_addendum_is_part_of_literature_authority() -> None:
    authority = _load_json("authority/olfactory-navigation-literature-v1.json")
    by_key = {entry["key"]: entry for entry in authority["evidence"]}

    addendum = by_key["matheson_2024_hdelta_addendum"]
    assert addendum["doi"] == "10.1038/s41467-024-46225-8"
    assert "hDeltaK" in " ".join(addendum["claims_used_as_constraints"])


def test_hdelta_c_is_not_promoted_to_proven_functional_integrator() -> None:
    literature = _load_json("authority/olfactory-navigation-literature-v1.json")
    type_evidence = _load_json("authority/malecns-v1.0-type-evidence.json")

    serialized = json.dumps([literature, type_evidence]).lower()
    disallowed_phrases = [
        "hdelta c activity is consistent with odor-gated wind-direction integration",
        "hdeltac activity is consistent with odor-gated wind-direction integration",
        "hdeltac can specify an odor-gated goal direction",
        "hdeltac itself computes odor-gated wind direction",
    ]
    for phrase in disallowed_phrases:
        assert phrase not in serialized

    assert "structural convergence candidate" in serialized
    assert "confound" in serialized


def test_pfn_laterality_keeps_soma_side_as_primary_authority() -> None:
    role_policy = _load_json("configs/role_review_v1.json")
    assert role_policy["side_inference"]["priority"][0] == "somaSide"

    rationale = role_policy["side_inference"]["rationale"].lower()
    assert "cell body" in rationale or "soma" in rationale
    assert "fallback" in rationale


def test_model_inputs_remain_explicitly_distinct_from_measured_activity() -> None:
    role_policy = _load_json("configs/role_review_v1.json")
    odor_rationale = role_policy["roles"]["odor_context_left"]["rationale"].lower()
    hdelta_rationale = role_policy["structural_context_roles"][
        "integration_hdelta_c"
    ]["rationale"].lower()

    assert "model abstraction" in odor_rationale
    assert "not measured" in odor_rationale
    assert "structural" in hdelta_rationale
    assert "physiological" in hdelta_rationale
