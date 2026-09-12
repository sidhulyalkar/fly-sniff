import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text())


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def test_hdelta_c_addendum_is_part_of_literature_authority() -> None:
    authority = _load_json("authority/olfactory-navigation-literature-v1.json")
    by_key = {entry["key"]: entry for entry in authority["evidence"]}

    addendum = by_key["matheson_2024_hdelta_addendum"]
    assert addendum["doi"] == "10.1038/s41467-024-46225-8"
    assert "hDeltaK" in " ".join(addendum["claims_used_as_constraints"])


def test_role_policy_is_bound_to_current_literature_authority() -> None:
    role_policy = _load_json("configs/role_review_v1.json")
    binding = role_policy["literature_authority"]
    authority_path = ROOT / binding["path"]

    assert authority_path.is_file()
    assert _git_blob_sha1(authority_path) == binding["git_blob_sha1"]


def test_hdelta_c_is_not_promoted_to_proven_functional_integrator() -> None:
    literature = _load_json("authority/olfactory-navigation-literature-v1.json")
    type_evidence = _load_json("authority/malecns-v1.0-type-evidence.json")
    staged_route = _load_json("configs/staged_route_v1.json")

    serialized = json.dumps([literature, type_evidence, staged_route]).lower()
    disallowed_phrases = [
        "hdelta c activity is consistent with odor-gated wind-direction integration",
        "hdeltac activity is consistent with odor-gated wind-direction integration",
        "hdeltac can specify an odor-gated goal direction",
        "candidate odor-gated wind representation",
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


def test_pfl3_steering_side_is_not_equated_with_generic_anatomical_side() -> None:
    authority = _load_json("authority/olfactory-navigation-literature-v1.json")
    by_key = {entry["key"]: entry for entry in authority["evidence"]}
    westeinde = by_key["westeinde_2024_goal_to_steering"]

    claims = " ".join(westeinde["claims_used_as_constraints"]).lower()
    boundary = westeinde["boundary"].lower()
    assert "lateral accessory lobe" in claims
    assert "opposite" in claims
    assert "do not infer pfl3 steering laterality" in boundary


def test_indirect_local_fb_route_is_prior_not_posthoc_whitelist() -> None:
    authority = _load_json("authority/olfactory-navigation-literature-v1.json")
    by_key = {entry["key"]: entry for entry in authority["evidence"]}
    hulse = by_key["hulse_2021_central_complex_connectome"]

    claims = " ".join(hulse["claims_used_as_search_priors"]).lower()
    boundary = hulse["boundary"].lower()
    assert "two- and three-step pathways" in claims
    assert "does not authorize requiring" in boundary


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


def test_scientific_docs_acknowledge_hdelta_c_functional_confound() -> None:
    required_docs = [
        "docs/E001_E002_PROTOCOL.md",
        "docs/TASK_OPTIMIZATION_V1.md",
    ]
    for relative_path in required_docs:
        text = (ROOT / relative_path).read_text().lower()
        assert "2024" in text
        assert "hdelta" in text
        assert "h∆" in text or "hdelta" in text
        assert "confound" in text or "cannot be assigned specifically" in text
