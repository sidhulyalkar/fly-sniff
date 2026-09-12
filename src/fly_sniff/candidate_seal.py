from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .freeze import current_git_ref
from .graph import GraphBundle

MODEL_ROLES = (
    "odor_context_left",
    "odor_context_right",
    "wind_basis_left",
    "wind_basis_right",
    "steer_left",
    "steer_right",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _canonical_edge_identity(bundle: GraphBundle) -> list[tuple[int, int, float, int]]:
    if "sign" not in bundle.edges.columns:
        raise ValueError("candidate seal requires an explicit sign column")
    rows = [
        (int(row.source), int(row.target), float(row.weight), int(row.sign))
        for row in bundle.edges[["source", "target", "weight", "sign"]].itertuples(
            index=False
        )
    ]
    return sorted(rows)


def _required_stage_union(
    staged_root: Path,
    staged_report: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[set[int], dict[tuple[int, int], dict[str, Any]]]:
    required = list(policy["required_primary_stages"])
    stage_reports = {
        str(stage.get("name")): stage for stage in staged_report.get("stages", [])
    }
    nodes: set[int] = set()
    edges: dict[tuple[int, int], dict[str, Any]] = {}

    for stage_name in required:
        stage = stage_reports.get(stage_name)
        if not isinstance(stage, dict):
            raise ValueError(f"candidate seal is missing required E001 stage {stage_name!r}")
        if stage.get("required_for_primary_hypothesis") is not True:
            raise ValueError(f"stage {stage_name!r} is no longer a required primary stage")
        if stage.get("structural_audit_passed") is not True:
            raise ValueError(f"required E001 stage {stage_name!r} did not pass its structural audit")

        node_path = staged_root / stage_name / "nodes.parquet"
        edge_path = staged_root / stage_name / "edges.parquet"
        if not node_path.exists() or not edge_path.exists():
            raise FileNotFoundError(f"missing staged artifacts for {stage_name!r}")
        node_frame = pd.read_parquet(node_path)
        edge_frame = pd.read_parquet(edge_path)
        if "bodyId" not in node_frame.columns:
            raise ValueError(f"stage {stage_name!r} nodes are missing bodyId")
        if not {"source", "target", "weight"}.issubset(edge_frame.columns):
            raise ValueError(f"stage {stage_name!r} edges are missing required columns")
        nodes.update(int(value) for value in node_frame.bodyId)

        for row in edge_frame[["source", "target", "weight"]].itertuples(index=False):
            source = int(row.source)
            target = int(row.target)
            weight = float(row.weight)
            key = (source, target)
            prior = edges.get(key)
            if prior is None:
                edges[key] = {"weight": weight, "stages_seen": {stage_name}}
                continue
            if not np.isclose(float(prior["weight"]), weight, rtol=0.0, atol=0.0):
                raise ValueError(
                    "the same biological edge has inconsistent structural weight across E001 stages: "
                    f"{source}->{target}: {prior['weight']} vs {weight}"
                )
            prior["stages_seen"].add(stage_name)
    return nodes, edges


def _verify_sign_report(bundle: GraphBundle, report: dict[str, Any]) -> dict[str, Any]:
    stored_sha = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if not stored_sha or stored_sha != canonical_sha256(unsigned):
        raise ValueError("source sign-authority report hash mismatch")
    if report.get("passed") is not True:
        raise ValueError("source sign-authority report contains graph/sign mismatches")
    if report.get("graph_sha256") != bundle.replay_fingerprint():
        raise ValueError("source sign-authority report was produced for a different graph")

    source_records = report.get("source_records")
    if not isinstance(source_records, list):
        raise TypeError("source sign-authority report is missing source_records")
    sign_by_source = {
        int(record["source_body_id"]): int(record["model_sign"])
        for record in source_records
    }
    actual_sources = set(bundle.edges.source.astype(int))
    if set(sign_by_source) != actual_sources:
        raise ValueError("sign-authority source-body set does not match the sealed GraphBundle")
    for row in bundle.edges[["source", "sign"]].itertuples(index=False):
        if int(row.sign) != sign_by_source[int(row.source)]:
            raise ValueError("GraphBundle edge sign disagrees with source-body sign authority")
    return report["coverage"]


def _role_receipt(roles: dict[str, list[int]]) -> dict[str, Any]:
    canonical = {name: sorted(int(value) for value in ids) for name, ids in roles.items()}
    return {
        "roles": canonical,
        "role_counts": {name: len(ids) for name, ids in canonical.items()},
        "roles_sha256": canonical_sha256(canonical),
    }


def build_candidate_manifest(
    *,
    bundle_dir: str | Path,
    staged_root: str | Path,
    role_review_path: str | Path,
    e001_evidence_path: str | Path,
    sign_authority_path: str | Path,
    policy_path: str | Path,
    task_config_path: str | Path,
    code_ref: str | None = None,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    staged_root = Path(staged_root)
    role_review_path = Path(role_review_path)
    e001_evidence_path = Path(e001_evidence_path)
    sign_authority_path = Path(sign_authority_path)
    policy_path = Path(policy_path)
    task_config_path = Path(task_config_path)

    bundle = GraphBundle.load(bundle_dir)
    bundle.validate(require_sign=True, require_qualified=False)
    staged_report_path = staged_root / "staged_trace_report.json"
    handoff_path = staged_root / "handoff_audit.json"
    staged_report = _load_json(staged_report_path)
    handoffs = _load_json(handoff_path)
    role_review = _load_json(role_review_path)
    e001_evidence = _load_json(e001_evidence_path)
    sign_report = _load_json(sign_authority_path)
    policy = _load_json(policy_path)
    task_config = _load_json(task_config_path)

    if policy.get("protocol") != "sealed-candidate-graph-v1":
        raise ValueError("unsupported candidate graph policy")
    if staged_report.get("primary_structural_hypothesis_passed") is not True:
        raise ValueError("candidate graph cannot seal before E001 primary structure passes")
    if handoffs.get("all_required_handoffs_pass") is not True:
        raise ValueError("candidate graph cannot seal before required body-ID handoffs pass")
    if e001_evidence.get("claim_level") != "BODY_ID_RESOLVED_STRUCTURAL_CANDIDATE":
        raise ValueError("candidate graph requires the exact E001 structural evidence claim level")
    if e001_evidence.get("ship_gate", {}).get("structural_evidence_pack_ready") is not True:
        raise ValueError("candidate graph requires a passing E001 evidence pack")

    allowed_nodes, stage_edges = _required_stage_union(staged_root, staged_report, policy)
    bundle_nodes = set(bundle.nodes.bodyId.astype(int))
    outside_nodes = sorted(bundle_nodes - allowed_nodes)
    if outside_nodes:
        raise ValueError(
            f"sealed GraphBundle contains {len(outside_nodes)} nodes outside required E001 stages"
        )

    if bundle.edges.duplicated(["source", "target"]).any():
        raise ValueError("sealed GraphBundle contains duplicate biological source-target edges")
    missing_edges: list[tuple[int, int]] = []
    weight_mismatches: list[tuple[int, int, float, float]] = []
    for row in bundle.edges[["source", "target", "weight"]].itertuples(index=False):
        source = int(row.source)
        target = int(row.target)
        weight = float(row.weight)
        staged = stage_edges.get((source, target))
        if staged is None:
            missing_edges.append((source, target))
            continue
        expected_weight = float(staged["weight"])
        if not np.isclose(weight, expected_weight, rtol=0.0, atol=0.0):
            weight_mismatches.append((source, target, weight, expected_weight))
    if missing_edges:
        raise ValueError(
            f"sealed GraphBundle contains {len(missing_edges)} edges outside required E001 stages"
        )
    if weight_mismatches:
        raise ValueError("sealed GraphBundle structural weights differ from E001 stage evidence")

    expected_roles = role_review.get("role_draft", {})
    required_roles = list(policy["required_model_roles"])
    for role in required_roles:
        expected = sorted(int(value) for value in expected_roles.get(role, []))
        observed = sorted(int(value) for value in bundle.roles.get(role, []))
        if expected != observed:
            raise ValueError(f"sealed role membership differs from E001 review for {role!r}")
    allowed_roles = set(required_roles) | set(policy.get("allowed_additional_roles", []))
    unexpected_roles = sorted(set(bundle.roles) - allowed_roles)
    if unexpected_roles:
        raise ValueError(f"sealed GraphBundle contains unexpected roles: {unexpected_roles}")

    coverage = _verify_sign_report(bundle, sign_report)
    minimum_signed = float(
        policy["sign_authority"]["minimum_signed_edge_fraction_for_training"]
    )
    signed_fraction = float(coverage["signed_edge_fraction"])

    normalization = policy["sensory_drive_normalization"]
    interface = task_config.get("connectome_sensory_interface", {})
    if interface.get("drive_normalization") != normalization["method"]:
        raise ValueError("task config sensory normalization differs from candidate policy")
    if interface.get("role_membership_source") != "sealed_candidate_manifest_exact_ids":
        raise ValueError("task config does not bind sensory roles to the sealed candidate manifest")

    exact_node_ids = sorted(bundle_nodes)
    role_ids = {
        int(value)
        for role in required_roles
        for value in bundle.roles.get(role, [])
    }
    intermediate_ids = sorted(bundle_nodes - role_ids)
    edge_identity = _canonical_edge_identity(bundle)
    role_receipt = _role_receipt(bundle.roles)
    bundle_files = {}
    for name in ("nodes.parquet", "edges.parquet", "roles.json", "manifest.json"):
        path = bundle_dir / name
        if path.exists():
            bundle_files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    checks = {
        "e001_primary_structure_passed": True,
        "e001_required_handoffs_passed": True,
        "bundle_nodes_within_required_stages": True,
        "bundle_edges_within_required_stages": True,
        "bundle_weights_match_e001": True,
        "bundle_edges_unique_by_source_target": True,
        "model_roles_match_e001_role_review": True,
        "sign_authority_matches_every_graph_edge": True,
        "task_config_uses_role_total_l1": True,
        "signed_edge_fraction_meets_training_gate": signed_fraction >= minimum_signed,
    }
    training_ready = all(checks.values())
    ref = code_ref or current_git_ref()
    if not ref or ref == "UNKNOWN":
        raise ValueError("candidate manifest requires an exact git commit")

    payload = {
        "schema": "fly-sniff-sealed-candidate-v1",
        "status": "sealed",
        "dataset": policy["dataset"],
        "code_ref": ref,
        "graph_sha256": bundle.replay_fingerprint(),
        "bundle_files": bundle_files,
        "node_count": len(bundle_nodes),
        "exact_node_body_ids": exact_node_ids,
        "intermediate_body_ids": intermediate_ids,
        "edge_count": len(bundle.edges),
        "edge_identity_sha256": canonical_sha256(edge_identity),
        "role_receipt": role_receipt,
        "signed_edge_fraction": signed_fraction,
        "signed_edge_minimum_for_training": minimum_signed,
        "training_ready": training_ready,
        "checks": checks,
        "inclusion_policy": policy,
        "inclusion_policy_sha256": sha256_file(policy_path),
        "sensory_drive_normalization": normalization,
        "task_config_sha256": sha256_file(task_config_path),
        "sign_authority_report_sha256": sha256_file(sign_authority_path),
        "sign_authority_internal_sha256": sign_report["report_sha256"],
        "source_lineage": {
            "staged_trace_report_sha256": sha256_file(staged_report_path),
            "handoff_audit_sha256": sha256_file(handoff_path),
            "role_review_sha256": sha256_file(role_review_path),
            "e001_evidence_sha256": sha256_file(e001_evidence_path),
        },
        "performance_freeze": policy["performance_freeze"],
        "claim_boundary": policy["claim_boundary"],
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def verify_candidate_manifest(
    bundle_dir: str | Path,
    manifest_path: str | Path,
    *,
    task_config_path: str | Path | None = None,
    require_training_ready: bool = True,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    stored = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not stored or stored != canonical_sha256(unsigned):
        raise ValueError("candidate manifest hash mismatch")
    if manifest.get("schema") != "fly-sniff-sealed-candidate-v1":
        raise ValueError("unsupported candidate manifest schema")
    if manifest.get("status") != "sealed":
        raise ValueError("candidate manifest is not sealed")
    if require_training_ready and manifest.get("training_ready") is not True:
        raise ValueError("candidate manifest is sealed but not training-ready")

    bundle = GraphBundle.load(bundle_dir)
    bundle.validate(require_sign=True, require_qualified=False)
    if bundle.replay_fingerprint() != manifest.get("graph_sha256"):
        raise ValueError("GraphBundle fingerprint differs from sealed candidate manifest")
    if sorted(bundle.nodes.bodyId.astype(int)) != manifest.get("exact_node_body_ids"):
        raise ValueError("GraphBundle node set differs from sealed candidate manifest")
    if canonical_sha256(_canonical_edge_identity(bundle)) != manifest.get(
        "edge_identity_sha256"
    ):
        raise ValueError("GraphBundle edge identity differs from sealed candidate manifest")
    if _role_receipt(bundle.roles)["roles_sha256"] != manifest["role_receipt"][
        "roles_sha256"
    ]:
        raise ValueError("GraphBundle roles differ from sealed candidate manifest")

    for name, receipt in manifest.get("bundle_files", {}).items():
        path = bundle_dir / name
        if not path.exists() or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"sealed GraphBundle file changed: {name}")
    if task_config_path is not None:
        if sha256_file(task_config_path) != manifest.get("task_config_sha256"):
            raise ValueError("task optimization config differs from candidate seal")
    return manifest


def seal_main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal the exact pre-performance GraphBundle and all scientific authorities"
    )
    parser.add_argument("bundle")
    parser.add_argument("staged_root")
    parser.add_argument("role_review")
    parser.add_argument("e001_evidence")
    parser.add_argument("sign_authority")
    parser.add_argument("--policy", default="configs/candidate_graph_v1.json")
    parser.add_argument("--task-config", default="configs/task_optimization_v1.json")
    parser.add_argument("--output", default="manifests/candidate-graph-v1.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite sealed candidate manifest: {output}")
    manifest = build_candidate_manifest(
        bundle_dir=args.bundle,
        staged_root=args.staged_root,
        role_review_path=args.role_review,
        e001_evidence_path=args.e001_evidence,
        sign_authority_path=args.sign_authority,
        policy_path=args.policy,
        task_config_path=args.task_config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "training_ready": manifest["training_ready"],
                "signed_edge_fraction": manifest["signed_edge_fraction"],
                "node_count": manifest["node_count"],
                "edge_count": manifest["edge_count"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not manifest["training_ready"]:
        raise SystemExit(2)


def verify_main() -> None:
    parser = argparse.ArgumentParser(description="Verify an immutable candidate GraphBundle seal")
    parser.add_argument("bundle")
    parser.add_argument("manifest")
    parser.add_argument("--task-config", default="configs/task_optimization_v1.json")
    args = parser.parse_args()
    manifest = verify_candidate_manifest(
        args.bundle,
        args.manifest,
        task_config_path=args.task_config,
        require_training_ready=True,
    )
    print(
        json.dumps(
            {
                "verified": True,
                "manifest_sha256": manifest["manifest_sha256"],
                "graph_sha256": manifest["graph_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    seal_main()
