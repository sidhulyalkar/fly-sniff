import json

import pandas as pd
import pytest

from fly_sniff import candidate_seal
from fly_sniff.sign_authority import write_sign_authority_report

STAGES = [
    "odor_value_to_fb5ab",
    "fb5ab_to_hDeltaC",
    "wind_to_hDeltaC",
    "hDeltaC_to_pfl3",
    "pfl3_to_dna02",
]


def _fixture(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    nodes = pd.DataFrame(
        [
            {"bodyId": 1, "type": "FB5AB"},
            {"bodyId": 2, "type": "FB5AB"},
            {"bodyId": 3, "type": "PFNa"},
            {"bodyId": 4, "type": "PFNa"},
            {"bodyId": 5, "type": "DNa02"},
            {"bodyId": 6, "type": "DNa02"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": 1, "target": 5, "weight": 5.0, "sign": 1},
            {"source": 2, "target": 6, "weight": 5.0, "sign": 1},
            {"source": 3, "target": 5, "weight": 4.0, "sign": 1},
            {"source": 4, "target": 6, "weight": 4.0, "sign": 1},
        ]
    )
    roles = {
        "odor_context_left": [1],
        "odor_context_right": [2],
        "wind_basis_left": [3],
        "wind_basis_right": [4],
        "steer_left": [5],
        "steer_right": [6],
    }
    nodes.to_parquet(bundle_dir / "nodes.parquet", index=False)
    edges.to_parquet(bundle_dir / "edges.parquet", index=False)
    (bundle_dir / "roles.json").write_text(json.dumps(roles))
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"dataset": "male-cns:v1.0", "qualification_status": "qualified"})
    )

    staged = tmp_path / "staged"
    for name in STAGES:
        stage = staged / name
        stage.mkdir(parents=True)
        nodes.to_parquet(stage / "nodes.parquet", index=False)
        edges.drop(columns=["sign"]).to_parquet(stage / "edges.parquet", index=False)
    (staged / "staged_trace_report.json").write_text(
        json.dumps(
            {
                "primary_structural_hypothesis_passed": True,
                "stages": [
                    {
                        "name": name,
                        "required_for_primary_hypothesis": True,
                        "structural_audit_passed": True,
                    }
                    for name in STAGES
                ],
            }
        )
    )
    (staged / "handoff_audit.json").write_text(
        json.dumps({"all_required_handoffs_pass": True})
    )

    role_review = tmp_path / "role_review.json"
    role_review.write_text(json.dumps({"role_draft": roles}))
    evidence = tmp_path / "e001_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "claim_level": "BODY_ID_RESOLVED_STRUCTURAL_CANDIDATE",
                "ship_gate": {"structural_evidence_pack_ready": True},
            }
        )
    )

    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "authority_kind": "type-level-transmitter-evidence",
                "model_sign_rule": {"ACh": 1, "GABA": -1, "Glu": 0, "other_or_unclear": 0},
                "types": {
                    name: {
                        "predicted_neurotransmitter": "ACh",
                        "confidence": 0.9,
                        "source": "fixture",
                    }
                    for name in ("FB5AB", "PFNa", "DNa02")
                },
            }
        )
    )
    sign_dir = tmp_path / "sign-authority"
    write_sign_authority_report(bundle_dir, [authority], sign_dir)
    sign_path = sign_dir / "source_sign_authority.json"

    policy = tmp_path / "candidate_policy.json"
    policy.write_text(
        json.dumps(
            {
                "protocol": "sealed-candidate-graph-v1",
                "dataset": "male-cns:v1.0",
                "claim_boundary": "fixture",
                "required_primary_stages": STAGES,
                "required_model_roles": list(roles),
                "allowed_additional_roles": [],
                "sign_authority": {"minimum_signed_edge_fraction_for_training": 0.6},
                "sensory_drive_normalization": {
                    "method": "role_total_l1",
                    "roles": [
                        "odor_context_left",
                        "odor_context_right",
                        "wind_basis_left",
                        "wind_basis_right",
                    ],
                },
                "performance_freeze": {"allow_graph_change": False},
            }
        )
    )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "connectome_sensory_interface": {
                    "drive_normalization": "role_total_l1",
                    "role_membership_source": "sealed_candidate_manifest_exact_ids",
                }
            }
        )
    )
    return bundle_dir, staged, role_review, evidence, sign_path, policy, task


def test_candidate_manifest_binds_exact_graph_and_roles(tmp_path):
    bundle, staged, review, evidence, sign, policy, task = _fixture(tmp_path)
    manifest = candidate_seal.build_candidate_manifest(
        bundle_dir=bundle,
        staged_root=staged,
        role_review_path=review,
        e001_evidence_path=evidence,
        sign_authority_path=sign,
        policy_path=policy,
        task_config_path=task,
        code_ref="fixture-commit",
    )
    assert manifest["training_ready"] is True
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(manifest))
    verified = candidate_seal.verify_candidate_manifest(bundle, path, task_config_path=task)
    assert verified["manifest_sha256"] == manifest["manifest_sha256"]


def test_candidate_manifest_rejects_graph_change(tmp_path):
    bundle, staged, review, evidence, sign, policy, task = _fixture(tmp_path)
    manifest = candidate_seal.build_candidate_manifest(
        bundle_dir=bundle,
        staged_root=staged,
        role_review_path=review,
        e001_evidence_path=evidence,
        sign_authority_path=sign,
        policy_path=policy,
        task_config_path=task,
        code_ref="fixture-commit",
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(manifest))
    edges = pd.read_parquet(bundle / "edges.parquet")
    edges.loc[0, "weight"] = 6.0
    edges.to_parquet(bundle / "edges.parquet", index=False)
    with pytest.raises(ValueError, match="GraphBundle fingerprint differs"):
        candidate_seal.verify_candidate_manifest(bundle, path, task_config_path=task)


def test_candidate_manifest_rejects_sign_report_from_different_nodes_file(tmp_path):
    bundle, staged, review, evidence, sign, policy, task = _fixture(tmp_path)
    report = json.loads(sign.read_text())
    report["node_annotation_authority"]["sha256"] = "not-the-bundle-nodes-sha"
    report.pop("report_sha256", None)
    report["report_sha256"] = candidate_seal.canonical_sha256(report)
    sign.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different nodes.parquet annotations"):
        candidate_seal.build_candidate_manifest(
            bundle_dir=bundle,
            staged_root=staged,
            role_review_path=review,
            e001_evidence_path=evidence,
            sign_authority_path=sign,
            policy_path=policy,
            task_config_path=task,
            code_ref="fixture-commit",
        )
