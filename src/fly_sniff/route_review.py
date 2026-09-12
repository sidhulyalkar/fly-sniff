from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def infer_instance_side(
    instance: str,
    *,
    left_pattern: str,
    right_pattern: str,
) -> str | None:
    text = str(instance or "")
    left = bool(re.search(left_pattern, text))
    right = bool(re.search(right_pattern, text))
    if left and right:
        return None
    if left:
        return "L"
    if right:
        return "R"
    return None


def _retained_seed_ids(stage_dir: Path, seed_role: str) -> set[int]:
    provenance = pd.read_csv(stage_dir / "path_provenance.csv")
    column = "is_source_seed" if seed_role == "source" else "is_target_seed"
    if column not in provenance.columns:
        raise ValueError(f"{stage_dir.name} provenance is missing {column}")
    return set(provenance.loc[_truthy(provenance[column]), "bodyId"].astype(int))


def _seed_table(stage_dir: Path, seed_role: str) -> pd.DataFrame:
    path = stage_dir / f"{seed_role}_seeds.csv"
    table = pd.read_csv(path)
    if "bodyId" not in table.columns:
        raise ValueError(f"seed table is missing bodyId: {path}")
    table = table.copy()
    table["bodyId"] = table.bodyId.astype(int)
    return table


def derive_role_ids(
    root: str | Path,
    policy: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    root = Path(root)
    side_cfg = policy["side_inference"]
    left_pattern = str(side_cfg["left_pattern"])
    right_pattern = str(side_cfg["right_pattern"])
    roles: dict[str, list[int]] = {}
    role_evidence: dict[str, Any] = {}

    for role_name, rule in policy["roles"].items():
        stage_dir = root / str(rule["stage"])
        seed_role = str(rule["seed_role"])
        retained = _retained_seed_ids(stage_dir, seed_role)
        seeds = _seed_table(stage_dir, seed_role)
        if "type" not in seeds.columns or "instance" not in seeds.columns:
            raise ValueError(
                f"{stage_dir.name} {seed_role} seed table requires type and instance columns"
            )
        type_regex = re.compile(str(rule["type_regex"]), re.IGNORECASE)
        required_side = str(rule["required_side"])
        selected: list[int] = []
        evidence_rows: list[dict[str, Any]] = []
        for row in seeds.itertuples(index=False):
            body_id = int(row.bodyId)
            if body_id not in retained:
                continue
            neuron_type = str(getattr(row, "type", "") or "")
            if not type_regex.search(neuron_type):
                continue
            instance = str(getattr(row, "instance", "") or "")
            side = infer_instance_side(
                instance,
                left_pattern=left_pattern,
                right_pattern=right_pattern,
            )
            if side != required_side:
                continue
            selected.append(body_id)
            evidence_rows.append(
                {
                    "body_id": body_id,
                    "type": neuron_type,
                    "instance": instance,
                    "inferred_side": side,
                    "stage": stage_dir.name,
                    "seed_role": seed_role,
                }
            )
        selected = sorted(set(selected))
        minimum = int(rule.get("minimum_count", 1))
        maximum = rule.get("maximum_count")
        passed = len(selected) >= minimum and (
            maximum is None or len(selected) <= int(maximum)
        )
        roles[role_name] = selected
        role_evidence[role_name] = {
            "passed": passed,
            "count": len(selected),
            "minimum_count": minimum,
            "maximum_count": int(maximum) if maximum is not None else None,
            "selection_rule": rule,
            "body_ids": selected,
            "evidence": evidence_rows,
        }

    handoff_report = json.loads((root / "handoff_audit.json").read_text())
    handoff_lookup = {
        str(item["name"]): item for item in handoff_report.get("handoffs", [])
    }
    for role_name, rule in policy.get("structural_context_roles", {}).items():
        handoff_name = str(rule["handoff"])
        if handoff_name not in handoff_lookup:
            raise ValueError(f"missing configured handoff {handoff_name!r}")
        source_key = str(rule.get("source", "shared_body_ids"))
        values = sorted(int(x) for x in handoff_lookup[handoff_name].get(source_key, []))
        roles[role_name] = values
        role_evidence[role_name] = {
            "passed": bool(values),
            "count": len(values),
            "body_ids": values,
            "handoff": handoff_name,
            "rationale": rule.get("rationale", ""),
        }

    return roles, role_evidence


def _stage_summary(root: Path, stage: dict[str, Any]) -> dict[str, Any]:
    name = str(stage["name"])
    stage_dir = root / name
    nodes = pd.read_parquet(stage_dir / "nodes.parquet")
    edges = pd.read_parquet(stage_dir / "edges.parquet")
    source = _seed_table(stage_dir, "source")
    target = _seed_table(stage_dir, "target")
    retained_source = _retained_seed_ids(stage_dir, "source")
    retained_target = _retained_seed_ids(stage_dir, "target")
    seed_ids = retained_source | retained_target
    node_ids = nodes.bodyId.astype(int) if "bodyId" in nodes.columns else pd.Series(dtype=int)
    intermediate = nodes.loc[~node_ids.isin(seed_ids)].copy()

    if "type" in intermediate.columns:
        types = intermediate["type"].fillna("").astype(str).str.strip()
        unknown_type_count = int(types.eq("").sum())
        type_counts = Counter(value for value in types if value)
        top_types = [
            {"type": neuron_type, "count": int(count)}
            for neuron_type, count in type_counts.most_common(20)
        ]
    else:
        unknown_type_count = len(intermediate)
        top_types = []

    return {
        "name": name,
        "required_for_primary_hypothesis": bool(
            stage.get("required_for_primary_hypothesis", True)
        ),
        "status": stage.get("status"),
        "structural_audit_passed": stage.get("structural_audit_passed"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "retained_source_seed_count": len(retained_source),
        "retained_target_seed_count": len(retained_target),
        "intermediate_node_count": len(intermediate),
        "unknown_intermediate_type_count": unknown_type_count,
        "top_intermediate_types": top_types,
        "source_type_counts": source.get("type", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .value_counts()
        .to_dict(),
        "target_type_counts": target.get("type", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .value_counts()
        .to_dict(),
    }


def _wind_resolution_report(root: Path, policy: dict[str, Any]) -> dict[str, Any]:
    stage_dir = root / "wind_to_hDeltaC"
    seeds = _seed_table(stage_dir, "source")
    retained = _retained_seed_ids(stage_dir, "source")
    seeds = seeds.loc[seeds.bodyId.isin(retained)].copy()
    side_cfg = policy["side_inference"]
    left_pattern = str(side_cfg["left_pattern"])
    right_pattern = str(side_cfg["right_pattern"])

    rows: list[dict[str, Any]] = []
    counts = Counter()
    type_side_counts: dict[str, Counter[str]] = {}
    for row in seeds.itertuples(index=False):
        neuron_type = str(getattr(row, "type", "") or "")
        instance = str(getattr(row, "instance", "") or "")
        side = infer_instance_side(
            instance,
            left_pattern=left_pattern,
            right_pattern=right_pattern,
        )
        label = side or "unresolved"
        counts[label] += 1
        type_side_counts.setdefault(neuron_type, Counter())[label] += 1
        if side is None:
            rows.append(
                {
                    "body_id": int(row.bodyId),
                    "type": neuron_type,
                    "instance": instance,
                }
            )
    return {
        "retained_pfn_count": len(seeds),
        "side_counts": dict(counts),
        "type_side_counts": {
            key: dict(value) for key, value in sorted(type_side_counts.items())
        },
        "side_unresolved_count": len(rows),
        "side_unresolved": rows,
        "training_policy": policy.get("wind_unresolved_policy", {}),
    }


def build_review(
    staged_root: str | Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    root = Path(staged_root)
    staged_report = json.loads((root / "staged_trace_report.json").read_text())
    if staged_report.get("protocol") != policy.get("input_protocol"):
        raise ValueError(
            f"role review expected {policy.get('input_protocol')!r}, "
            f"found {staged_report.get('protocol')!r}"
        )
    if not staged_report.get("primary_structural_hypothesis_passed"):
        raise ValueError("role review requires a passing preregistered E001 structural result")

    roles, role_evidence = derive_role_ids(root, policy)
    model_role_names = list(policy["roles"])
    all_model_roles_pass = all(role_evidence[name]["passed"] for name in model_role_names)
    disjoint_checks = {
        "odor_context_left_right_disjoint": set(roles["odor_context_left"]).isdisjoint(
            roles["odor_context_right"]
        ),
        "wind_basis_left_right_disjoint": set(roles["wind_basis_left"]).isdisjoint(
            roles["wind_basis_right"]
        ),
        "steer_left_right_disjoint": set(roles["steer_left"]).isdisjoint(
            roles["steer_right"]
        ),
    }
    primary_stages = [
        stage
        for stage in staged_report.get("stages", [])
        if stage.get("required_for_primary_hypothesis")
    ]
    stage_summaries = [_stage_summary(root, stage) for stage in primary_stages]
    wind_resolution = _wind_resolution_report(root, policy)

    role_draft = {
        name: roles[name]
        for name in (
            "odor_context_left",
            "odor_context_right",
            "wind_basis_left",
            "wind_basis_right",
            "steer_left",
            "steer_right",
            "integration_hdelta_c",
            "pfl3_output",
        )
    }
    return {
        "protocol": policy["protocol"],
        "dataset": staged_report.get("dataset"),
        "review_status": "candidate_needs_human_review",
        "automatic_qualification": False,
        "staged_report_sha256": _sha256_file(root / "staged_trace_report.json"),
        "handoff_audit_sha256": _sha256_file(root / "handoff_audit.json"),
        "input_provenance": staged_report.get("input_provenance"),
        "primary_structural_hypothesis_passed": True,
        "all_model_role_rules_pass": all_model_roles_pass,
        "disjoint_checks": disjoint_checks,
        "all_disjoint_checks_pass": all(disjoint_checks.values()),
        "role_draft": role_draft,
        "role_evidence": role_evidence,
        "wind_resolution": wind_resolution,
        "primary_stage_summaries": stage_summaries,
        "required_human_review": [
            "confirm the PFNa/PFNm side-to-preferred-airflow mapping for this MaleCNS release",
            "review dominant and unknown intermediate cell types in every primary stage",
            "construct the candidate graph from an explicitly frozen node/edge inclusion policy",
            "attach conservative presynaptic transmitter signs and inspect signed-edge coverage",
            "verify DNa02 left/right steering semantics against the direct-recording convention",
        ],
        "claim_boundary": policy["claim_boundary"],
    }


def write_review(
    staged_root: str | Path,
    policy_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    policy_path = Path(policy_path)
    policy = json.loads(policy_path.read_text())
    report = build_review(staged_root, policy)
    report["policy_sha256"] = _sha256_file(policy_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "role_review.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "roles.candidate.json").write_text(
        json.dumps(report["role_draft"], indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive and audit candidate training roles from a passing E001 route"
    )
    parser.add_argument("staged_root", help="E001 staged-route artifact directory")
    parser.add_argument(
        "--policy",
        default="configs/role_review_v1.json",
        help="frozen body-ID role selection policy",
    )
    parser.add_argument("--output", default="results/e001-role-review-v1")
    args = parser.parse_args()
    report = write_review(args.staged_root, args.policy, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_model_role_rules_pass"] or not report["all_disjoint_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
