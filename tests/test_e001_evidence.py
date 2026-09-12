import hashlib
import json
from pathlib import Path

import pandas as pd

from fly_sniff.e001_evidence import build_evidence_pack, git_blob_sha1, sha256_file


def _write_fixture(tmp_path: Path):
    staged = tmp_path / "staged"
    stage = staged / "stage-a"
    stage.mkdir(parents=True)

    nodes = pd.DataFrame(
        [
            {"bodyId": 1, "type": "FB5AB", "instance": "FB5AB_L", "somaSide": "L"},
            {"bodyId": 2, "type": "FB5AB", "instance": "FB5AB_R", "somaSide": "R"},
            {"bodyId": 3, "type": "PFNa", "instance": "PFNa_L", "somaSide": "L"},
            {"bodyId": 4, "type": "PFNa", "instance": "PFNa_R", "somaSide": "R"},
            {"bodyId": 5, "type": "DNa02", "instance": "DNa02_L", "somaSide": "L"},
            {"bodyId": 6, "type": "DNa02", "instance": "DNa02_R", "somaSide": "R"},
            {"bodyId": 7, "type": "hDeltaC", "instance": "hDeltaC_1"},
            {"bodyId": 8, "type": "PFL3", "instance": "PFL3_1"},
        ]
    )
    nodes.to_parquet(stage / "nodes.parquet", index=False)
    pd.DataFrame(
        [
            {"source": 1, "target": 7, "weight": 10},
            {"source": 3, "target": 7, "weight": 8},
            {"source": 7, "target": 8, "weight": 6},
            {"source": 8, "target": 5, "weight": 9},
        ]
    ).to_parquet(stage / "edges.parquet", index=False)

    staged_report = {
        "protocol": "staged-structural-discovery-v3",
        "dataset": "male-cns:v1.0",
        "primary_structural_hypothesis_passed": True,
        "input_provenance": {"annotations": {"sha256": "a"}, "weights": {"sha256": "b"}},
        "stages": [{"name": "stage-a", "required_for_primary_hypothesis": True}],
    }
    (staged / "staged_trace_report.json").write_text(json.dumps(staged_report))
    (staged / "handoff_audit.json").write_text(json.dumps({"handoffs": []}))

    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "qualification_status": "hypothesis-prior-only-not-malecns-body-id-evidence",
                "forbidden_inferences": ["structure is not physiology"],
            }
        )
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "protocol": "e001-role-review-v1",
                "literature_authority": {
                    "path": str(authority),
                    "git_blob_sha1": git_blob_sha1(authority),
                },
            }
        )
    )

    role_draft = {
        "odor_context_left": [1],
        "odor_context_right": [2],
        "wind_basis_left": [3],
        "wind_basis_right": [4],
        "steer_left": [5],
        "steer_right": [6],
        "integration_hdelta_c": [7],
        "pfl3_output": [8],
    }
    evidence = {}
    for role, body_ids in role_draft.items():
        if role in {"integration_hdelta_c", "pfl3_output"}:
            evidence[role] = {"body_ids": body_ids, "passed": True}
            continue
        body_id = body_ids[0]
        row = nodes.loc[nodes.bodyId.eq(body_id)].iloc[0]
        evidence[role] = {
            "passed": True,
            "evidence": [
                {
                    "body_id": body_id,
                    "type": row["type"],
                    "instance": row["instance"],
                    "soma_side": row.get("somaSide"),
                    "root_side": None,
                    "inferred_side": row.get("somaSide"),
                    "side_evidence_source": "somaSide",
                }
            ],
        }

    review_dir = tmp_path / "review"
    review_dir.mkdir()
    review = {
        "protocol": "e001-role-review-v1",
        "dataset": "male-cns:v1.0",
        "review_status": "candidate_needs_human_review",
        "automatic_qualification": False,
        "all_model_role_rules_pass": True,
        "all_disjoint_checks_pass": True,
        "policy_sha256": sha256_file(policy),
        "staged_report_sha256": sha256_file(staged / "staged_trace_report.json"),
        "handoff_audit_sha256": sha256_file(staged / "handoff_audit.json"),
        "role_draft": role_draft,
        "role_evidence": evidence,
        "wind_resolution": {"side_unresolved_count": 0},
        "primary_stage_summaries": [],
        "claim_boundary": "candidate only",
    }
    (review_dir / "role_review.json").write_text(json.dumps(review))
    return staged, review_dir, policy, authority


def test_git_blob_sha_matches_git_object_formula(tmp_path):
    path = tmp_path / "authority.json"
    path.write_bytes(b"abc")
    expected = hashlib.sha1(b"blob 3\0abc").hexdigest()
    assert git_blob_sha1(path) == expected


def test_e001_pack_allows_structural_claim_but_blocks_functional_promotion(tmp_path):
    staged, review, policy, authority = _write_fixture(tmp_path)
    pack = build_evidence_pack(
        staged,
        review,
        policy_path=policy,
        authority_path=authority,
    )
    assert pack["claim_level"] == "BODY_ID_RESOLVED_STRUCTURAL_CANDIDATE"
    assert pack["ship_gate"]["structural_evidence_pack_ready"] is True
    assert pack["ship_gate"]["training_promotion_ready"] is False
    assert pack["ship_gate"]["public_functional_result_ready"] is False
    assert len(pack["body_id_claims"]) == 8
    hdelta = next(item for item in pack["body_id_claims"] if item["role"] == "integration_hdelta_c")
    assert hdelta["functional_verdict"] == "INSUFFICIENT_HDELTAC_SPECIFIC_EVIDENCE"
    assert hdelta["edge_context"]["incoming_edge_count"] == 2


def test_authority_mutation_breaks_structural_ship_gate(tmp_path):
    staged, review, policy, authority = _write_fixture(tmp_path)
    authority.write_text(
        json.dumps(
            {
                "qualification_status": "hypothesis-prior-only-not-malecns-body-id-evidence",
                "changed": True,
            }
        )
    )
    pack = build_evidence_pack(
        staged,
        review,
        policy_path=policy,
        authority_path=authority,
    )
    assert pack["checks"]["literature_authority_git_blob_matches"] is False
    assert pack["ship_gate"]["structural_evidence_pack_ready"] is False
    assert pack["claim_level"] == "DEVELOPMENT_ONLY"
