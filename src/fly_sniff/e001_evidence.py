from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_ROLES = (
    "odor_context_left",
    "odor_context_right",
    "wind_basis_left",
    "wind_basis_right",
    "steer_left",
    "steer_right",
)
STRUCTURAL_ROLES = ("integration_hdelta_c", "pfl3_output")
ALL_ROLES = MODEL_ROLES + STRUCTURAL_ROLES

ROLE_CLAIMS = {
    "odor_context_left": {
        "evidence_class": "cross-dataset functional prior + MaleCNS structural identity",
        "structural_verdict": "DIRECTLY_SUPPORTED_STRUCTURE",
        "functional_verdict": "SUPPORTED_AS_CROSS_DATASET_PRIOR",
        "allowed_wording": (
            "MaleCNS contains a left FB5AB structural candidate corresponding by type annotation "
            "to neurons implicated in olfactory navigation in earlier work."
        ),
        "forbidden_wording": "This MaleCNS neuron detects odor or encodes the modeled odor scalar.",
    },
    "odor_context_right": {
        "evidence_class": "cross-dataset functional prior + MaleCNS structural identity",
        "structural_verdict": "DIRECTLY_SUPPORTED_STRUCTURE",
        "functional_verdict": "SUPPORTED_AS_CROSS_DATASET_PRIOR",
        "allowed_wording": (
            "MaleCNS contains a right FB5AB structural candidate corresponding by type annotation "
            "to neurons implicated in olfactory navigation in earlier work."
        ),
        "forbidden_wording": "This MaleCNS neuron detects odor or encodes the modeled odor scalar.",
    },
    "wind_basis_left": {
        "evidence_class": "cross-animal PFN airflow physiology + MaleCNS structural identity",
        "structural_verdict": "DIRECTLY_SUPPORTED_STRUCTURE",
        "functional_verdict": "SUPPORTED_AS_CROSS_DATASET_PRIOR",
        "allowed_wording": (
            "The model assigns retained left-side ventral-PFN candidates to a literature-motivated "
            "airflow basis; soma-side assignments have the strongest physiological support."
        ),
        "forbidden_wording": "Every selected MaleCNS PFN has the measured airflow tuning used by the model.",
    },
    "wind_basis_right": {
        "evidence_class": "cross-animal PFN airflow physiology + MaleCNS structural identity",
        "structural_verdict": "DIRECTLY_SUPPORTED_STRUCTURE",
        "functional_verdict": "SUPPORTED_AS_CROSS_DATASET_PRIOR",
        "allowed_wording": (
            "The model assigns retained right-side ventral-PFN candidates to a literature-motivated "
            "airflow basis; soma-side assignments have the strongest physiological support."
        ),
        "forbidden_wording": "Every selected MaleCNS PFN has the measured airflow tuning used by the model.",
    },
    "integration_hdelta_c": {
        "evidence_class": "MaleCNS body-ID structural convergence + confounded cross-dataset physiology",
        "structural_verdict": "DIRECTLY_SUPPORTED_STRUCTURE",
        "functional_verdict": "INSUFFICIENT_HDELTAC_SPECIFIC_EVIDENCE",
        "allowed_wording": (
            "The exact MaleCNS hDeltaC body ID is retained by the odor-input, PFN-input, and outgoing "
            "structural searches, making it a body-ID-resolved convergence candidate."
        ),
        "forbidden_wording": (
            "This hDeltaC neuron is experimentally proven to compute odor-gated wind direction or source direction."
        ),
    },
    "pfl3_output": {
        "evidence_class": "MaleCNS structural continuity + cross-dataset steering literature",
        "structural_verdict": "DIRECTLY_SUPPORTED_STRUCTURE",
        "functional_verdict": "SUPPORTED_AS_CROSS_DATASET_PRIOR",
        "allowed_wording": (
            "The exact MaleCNS PFL3 body ID is structurally continuous across the central-complex-to-"
            "descending-neuron corridor and belongs to a steering-related type."
        ),
        "forbidden_wording": (
            "Soma, root, or bridge side alone proves this PFL3 neuron's steering-output laterality."
        ),
    },
    "steer_left": {
        "evidence_class": "MaleCNS structural identity + cross-animal direct DNa02 physiology",
        "structural_verdict": "DIRECTLY_SUPPORTED_STRUCTURE",
        "functional_verdict": "SUPPORTED_AS_CROSS_DATASET_PRIOR",
        "allowed_wording": (
            "The left MaleCNS DNa02 is a body-ID-resolved candidate steering readout; bilateral DNa02 "
            "physiology in other animals strongly links ipsilateral activity to turning."
        ),
        "forbidden_wording": "This MaleCNS DNa02 alone controls left turning.",
    },
    "steer_right": {
        "evidence_class": "MaleCNS structural identity + cross-animal direct DNa02 physiology",
        "structural_verdict": "DIRECTLY_SUPPORTED_STRUCTURE",
        "functional_verdict": "SUPPORTED_AS_CROSS_DATASET_PRIOR",
        "allowed_wording": (
            "The right MaleCNS DNa02 is a body-ID-resolved candidate steering readout; bilateral DNa02 "
            "physiology in other animals strongly links ipsilateral activity to turning."
        ),
        "forbidden_wording": "This MaleCNS DNa02 alone controls right turning.",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: str | Path) -> str:
    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _review_json_path(review: str | Path) -> Path:
    path = Path(review)
    return path / "role_review.json" if path.is_dir() else path


def _stage_nodes(staged_root: Path, staged_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    columns = (
        "type",
        "instance",
        "class",
        "subclass",
        "side",
        "somaSide",
        "rootSide",
        "predictedNt",
        "predictedNtProb",
        "predictedNT",
        "predictedNTProb",
        "predictedNeurotransmitter",
        "predictedNeurotransmitterConfidence",
    )
    for stage in staged_report.get("stages", []):
        name = str(stage.get("name", ""))
        path = staged_root / name / "nodes.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "bodyId" not in frame.columns:
            continue
        for row in frame.itertuples(index=False):
            body_id = int(row.bodyId)
            record = index.setdefault(body_id, {"body_id": body_id, "stages": []})
            if name and name not in record["stages"]:
                record["stages"].append(name)
            for column in columns:
                if column not in frame.columns:
                    continue
                value = _json_scalar(getattr(row, column, None))
                if value is not None and record.get(column) in {None, ""}:
                    record[column] = value
    for record in index.values():
        record["stages"] = sorted(record["stages"])
    return index


def _edge_context(
    staged_root: Path,
    staged_report: dict[str, Any],
    selected_ids: set[int],
    node_index: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {
        body_id: {
            "incoming_edge_count": 0,
            "outgoing_edge_count": 0,
            "incoming_weight": 0.0,
            "outgoing_weight": 0.0,
            "top_incoming": [],
            "top_outgoing": [],
        }
        for body_id in selected_ids
    }
    incoming: dict[int, list[tuple[float, int, str]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[float, int, str]]] = defaultdict(list)

    for stage in staged_report.get("stages", []):
        name = str(stage.get("name", ""))
        path = staged_root / name / "edges.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if not {"source", "target", "weight"}.issubset(frame.columns):
            continue
        for row in frame.itertuples(index=False):
            source = int(row.source)
            target = int(row.target)
            weight = float(row.weight)
            if source in selected_ids:
                entry = stats[source]
                entry["outgoing_edge_count"] += 1
                entry["outgoing_weight"] += weight
                outgoing[source].append((weight, target, name))
            if target in selected_ids:
                entry = stats[target]
                entry["incoming_edge_count"] += 1
                entry["incoming_weight"] += weight
                incoming[target].append((weight, source, name))

    def neighbor_record(item: tuple[float, int, str]) -> dict[str, Any]:
        weight, neighbor, stage = item
        annotation = node_index.get(neighbor, {})
        return {
            "body_id": neighbor,
            "type": annotation.get("type"),
            "instance": annotation.get("instance"),
            "weight": weight,
            "stage": stage,
        }

    for body_id in selected_ids:
        stats[body_id]["top_incoming"] = [
            neighbor_record(item) for item in sorted(incoming[body_id], reverse=True)[:5]
        ]
        stats[body_id]["top_outgoing"] = [
            neighbor_record(item) for item in sorted(outgoing[body_id], reverse=True)[:5]
        ]
        stats[body_id]["incoming_weight"] = round(stats[body_id]["incoming_weight"], 3)
        stats[body_id]["outgoing_weight"] = round(stats[body_id]["outgoing_weight"], 3)
    return stats


def build_evidence_pack(
    staged_root: str | Path,
    review: str | Path,
    *,
    policy_path: str | Path,
    authority_path: str | Path,
) -> dict[str, Any]:
    staged_root = Path(staged_root)
    review_path = _review_json_path(review)
    policy_path = Path(policy_path)
    authority_path = Path(authority_path)

    staged_report_path = staged_root / "staged_trace_report.json"
    handoff_path = staged_root / "handoff_audit.json"
    staged_report = _load_json(staged_report_path)
    role_review = _load_json(review_path)
    policy = _load_json(policy_path)
    authority = _load_json(authority_path)

    declared_authority = policy.get("literature_authority", {})
    checks = {
        "primary_structural_hypothesis_passed": bool(
            staged_report.get("primary_structural_hypothesis_passed")
        ),
        "role_review_is_human_pending_candidate": (
            role_review.get("review_status") == "candidate_needs_human_review"
        ),
        "automatic_qualification_is_false": role_review.get("automatic_qualification") is False,
        "all_model_role_rules_pass": bool(role_review.get("all_model_role_rules_pass")),
        "all_left_right_disjoint_checks_pass": bool(role_review.get("all_disjoint_checks_pass")),
        "role_policy_sha256_matches": role_review.get("policy_sha256") == sha256_file(policy_path),
        "staged_report_sha256_matches": (
            role_review.get("staged_report_sha256") == sha256_file(staged_report_path)
        ),
        "handoff_audit_sha256_matches": (
            handoff_path.exists()
            and role_review.get("handoff_audit_sha256") == sha256_file(handoff_path)
        ),
        "literature_authority_path_matches": (
            str(declared_authority.get("path", "")) == str(authority_path)
            or Path(str(declared_authority.get("path", ""))).name == authority_path.name
        ),
        "literature_authority_git_blob_matches": (
            declared_authority.get("git_blob_sha1") == git_blob_sha1(authority_path)
        ),
        "literature_authority_remains_nonqualifying": (
            authority.get("qualification_status")
            == "hypothesis-prior-only-not-malecns-body-id-evidence"
        ),
    }

    node_index = _stage_nodes(staged_root, staged_report)
    role_draft = role_review.get("role_draft", {})
    selected_ids = {
        int(body_id)
        for role in ALL_ROLES
        for body_id in role_draft.get(role, [])
    }
    edge_context = _edge_context(staged_root, staged_report, selected_ids, node_index)

    role_evidence = role_review.get("role_evidence", {})
    body_records: list[dict[str, Any]] = []
    for role in ALL_ROLES:
        per_body_review = {
            int(item["body_id"]): item
            for item in role_evidence.get(role, {}).get("evidence", [])
            if "body_id" in item
        }
        for body_id_raw in role_draft.get(role, []):
            body_id = int(body_id_raw)
            annotation = dict(node_index.get(body_id, {"body_id": body_id, "stages": []}))
            review_evidence = per_body_review.get(body_id, {})
            claim = ROLE_CLAIMS[role]
            body_records.append(
                {
                    "role": role,
                    "body_id": body_id,
                    "type": review_evidence.get("type") or annotation.get("type"),
                    "instance": review_evidence.get("instance") or annotation.get("instance"),
                    "soma_side": review_evidence.get("soma_side") or annotation.get("somaSide"),
                    "root_side": review_evidence.get("root_side") or annotation.get("rootSide"),
                    "inferred_side": review_evidence.get("inferred_side"),
                    "side_evidence_source": review_evidence.get("side_evidence_source"),
                    "stage_membership": annotation.get("stages", []),
                    "predicted_neurotransmitter": (
                        annotation.get("predictedNt")
                        or annotation.get("predictedNT")
                        or annotation.get("predictedNeurotransmitter")
                    ),
                    "predicted_neurotransmitter_confidence": (
                        annotation.get("predictedNtProb")
                        or annotation.get("predictedNTProb")
                        or annotation.get("predictedNeurotransmitterConfidence")
                    ),
                    "edge_context": edge_context.get(body_id, {}),
                    **claim,
                }
            )

    structural_ready = all(checks.values()) and bool(body_records)
    blockers: list[str] = []
    if not structural_ready:
        blockers.append("one or more E001 provenance/role/authority checks failed")
    blockers.extend(
        [
            "human body-ID and intermediate-population review has not been promoted to qualified",
            "candidate GraphBundle sign coverage has not been supplied to this E001 evidence pack",
            "matched intact/rewire/lesion task-optimization evidence is not part of E001",
            "held-out/OOD final evaluation receipt is not part of E001",
        ]
    )

    max_public_wording = (
        "We extracted a body-ID-resolved MaleCNS structural candidate linking literature-anchored "
        "odor-context, airflow, central-complex, and steering-related populations. This is structural "
        "evidence, not proof that the biological circuit performs the modeled computation."
        if structural_ready
        else "Development evidence only; do not make a body-ID-resolved MaleCNS circuit claim yet."
    )

    return {
        "protocol": "e001-body-id-evidence-pack-v1",
        "dataset": staged_report.get("dataset"),
        "claim_level": (
            "BODY_ID_RESOLVED_STRUCTURAL_CANDIDATE" if structural_ready else "DEVELOPMENT_ONLY"
        ),
        "checks": checks,
        "ship_gate": {
            "structural_evidence_pack_ready": structural_ready,
            "training_promotion_ready": False,
            "public_functional_result_ready": False,
            "max_public_wording": max_public_wording,
            "forbidden_public_wording": [
                "the fly connectome found the odor source",
                "the MaleCNS hDeltaC neurons compute odor-gated wind direction",
                "the selected MaleCNS PFNs were measured sensing the modeled airflow directions",
                "the connectome is firing in the visualization",
            ],
            "remaining_blockers": blockers,
        },
        "provenance": {
            "staged_report_sha256": sha256_file(staged_report_path),
            "handoff_audit_sha256": sha256_file(handoff_path) if handoff_path.exists() else None,
            "role_review_sha256": sha256_file(review_path),
            "role_policy_sha256": sha256_file(policy_path),
            "literature_authority_sha256": sha256_file(authority_path),
            "literature_authority_git_blob_sha1": git_blob_sha1(authority_path),
            "raw_input_provenance": staged_report.get("input_provenance"),
        },
        "role_counts": {
            role: len(role_draft.get(role, [])) for role in ALL_ROLES
        },
        "wind_resolution": role_review.get("wind_resolution"),
        "primary_stage_summaries": role_review.get("primary_stage_summaries", []),
        "body_id_claims": body_records,
        "claim_boundary": role_review.get("claim_boundary"),
    }


def _render_markdown(pack: dict[str, Any]) -> str:
    gate = pack["ship_gate"]
    lines = [
        "# E001 exact-body-ID evidence audit",
        "",
        f"**Dataset:** `{pack.get('dataset')}`",
        f"**Claim level:** `{pack['claim_level']}`",
        f"**Structural evidence pack ready:** `{gate['structural_evidence_pack_ready']}`",
        f"**Training promotion ready:** `{gate['training_promotion_ready']}`",
        f"**Public functional result ready:** `{gate['public_functional_result_ready']}`",
        "",
        "## Maximum allowed public wording",
        "",
        f"> {gate['max_public_wording']}",
        "",
        "## Provenance checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in pack["checks"].items():
        lines.append(f"| `{name}` | {'YES' if passed else 'NO'} |")
    lines.extend(
        [
            "",
            "## Exact selected body IDs",
            "",
            "| role | body ID | type | soma | root | side evidence | structural verdict | functional verdict |",
            "|---|---:|---|---|---|---|---|---|",
        ]
    )
    for item in pack["body_id_claims"]:
        lines.append(
            "| {role} | {body_id} | {type} | {soma} | {root} | {source} | {structural} | {functional} |".format(
                role=item["role"],
                body_id=item["body_id"],
                type=item.get("type") or "?",
                soma=item.get("soma_side") or "?",
                root=item.get("root_side") or "?",
                source=item.get("side_evidence_source") or "n/a",
                structural=item["structural_verdict"],
                functional=item["functional_verdict"],
            )
        )
    lines.extend(["", "## Claim table", ""])
    lines.append("| claim | direct evidence | indirect evidence | uncertainty | allowed wording | forbidden wording |")
    lines.append("|---|---|---|---|---|---|")
    for item in pack["body_id_claims"]:
        direct = f"MaleCNS body ID {item['body_id']} retained in {', '.join(item['stage_membership']) or 'reviewed role'}"
        indirect = item["evidence_class"]
        uncertainty = item["functional_verdict"]
        lines.append(
            f"| `{item['role']}` / `{item['body_id']}` | {direct} | {indirect} | {uncertainty} | "
            f"{item['allowed_wording']} | {item['forbidden_wording']} |"
        )
    lines.extend(["", "## Remaining blockers", ""])
    for blocker in gate["remaining_blockers"]:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Forbidden public wording", ""])
    for wording in gate["forbidden_public_wording"]:
        lines.append(f"- `{wording}`")
    return "\n".join(lines) + "\n"


def write_evidence_pack(
    staged_root: str | Path,
    review: str | Path,
    *,
    policy_path: str | Path,
    authority_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    pack = build_evidence_pack(
        staged_root,
        review,
        policy_path=policy_path,
        authority_path=authority_path,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "e001_evidence.json").write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")
    (output / "E001_BODY_ID_AUDIT.md").write_text(_render_markdown(pack))
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a provenance-checked, exact-body-ID E001 evidence and claim-boundary pack"
    )
    parser.add_argument("staged_root", help="passing staged-route-v1 artifact directory")
    parser.add_argument("role_review", help="role-review directory or role_review.json")
    parser.add_argument("--policy", default="configs/role_review_v1.json")
    parser.add_argument(
        "--authority", default="authority/olfactory-navigation-literature-v1.json"
    )
    parser.add_argument("--output", default="results/e001/evidence-pack")
    args = parser.parse_args()
    pack = write_evidence_pack(
        args.staged_root,
        args.role_review,
        policy_path=args.policy,
        authority_path=args.authority,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "claim_level": pack["claim_level"],
                "structural_evidence_pack_ready": pack["ship_gate"][
                    "structural_evidence_pack_ready"
                ],
                "body_id_count": len(pack["body_id_claims"]),
                "output": str(Path(args.output)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
