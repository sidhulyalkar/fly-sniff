from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .literature_route_audit import _git_output, _normalize_edges, _sha256_file

DEFAULT_CONFIG = Path("configs/integration_sign_audit_v2.json")


def _population_rows(integration_audit: dict[str, Any], name: str) -> list[dict[str, Any]]:
    populations = integration_audit.get("populations", {})
    if name not in populations:
        raise ValueError(f"integration audit missing population {name!r}")
    rows = [dict(row) for row in populations[name].get("rows", [])]
    expected_count = int(populations[name].get("count", len(rows)))
    body_ids = [int(row["bodyId"]) for row in rows if row.get("bodyId") is not None]
    if len(set(body_ids)) != expected_count or len(body_ids) != expected_count:
        raise ValueError(
            f"integration audit population {name!r} body-ID count mismatch: "
            f"rows={len(body_ids)} unique={len(set(body_ids))} expected={expected_count}"
        )
    missing_type = [
        body_id
        for body_id, row in zip(body_ids, rows)
        if not str(row.get("type", "")).strip()
    ]
    if missing_type:
        raise ValueError(
            f"integration audit population {name!r} has rows without type: {missing_type}"
        )
    return rows


def _verify_authority_source(
    authority: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    source = authority.get("source", {})
    mismatches = {
        key: {"observed": source.get(key), "expected": value}
        for key, value in expected.items()
        if source.get(key) != value
    }
    if mismatches:
        raise ValueError(f"transmitter authority source mismatch: {mismatches}")


def _type_evidence(
    authority: dict[str, Any],
    cell_type: str,
) -> dict[str, Any] | None:
    row = authority.get("types", {}).get(cell_type)
    return dict(row) if isinstance(row, dict) else None


def _population_summary(
    rows: list[dict[str, Any]],
    authority: dict[str, Any],
) -> dict[str, Any]:
    type_counts = Counter(str(row["type"]).strip() for row in rows)
    type_evidence: dict[str, dict[str, Any]] = {}
    unresolved_types: list[str] = []

    for cell_type, count in sorted(type_counts.items()):
        evidence = _type_evidence(authority, cell_type)
        if evidence is None:
            unresolved_types.append(cell_type)
            type_evidence[cell_type] = {
                "body_id_count": int(count),
                "status": "missing_from_authority",
            }
            continue

        modeled_sign = int(evidence.get("modeled_sign", 0))
        confidence = evidence.get("confidence")
        if modeled_sign == 0:
            unresolved_types.append(cell_type)
        type_evidence[cell_type] = {
            "body_id_count": int(count),
            "predicted_neurotransmitter": evidence.get("predicted_neurotransmitter"),
            "confidence": None if confidence is None else float(confidence),
            "modeled_sign": modeled_sign,
            "source_path": evidence.get("source_path"),
            "status": "resolved" if modeled_sign != 0 else "unresolved_sign",
        }

    confidences = [
        float(row["confidence"])
        for row in type_evidence.values()
        if row.get("confidence") is not None
    ]
    body_ids = sorted(int(row["bodyId"]) for row in rows)
    return {
        "body_id_count": len(body_ids),
        "body_ids": body_ids,
        "type_counts": dict(sorted(type_counts.items())),
        "type_evidence": type_evidence,
        "minimum_prediction_confidence": min(confidences) if confidences else None,
        "unresolved_types": sorted(unresolved_types),
        "ready": not unresolved_types,
    }


def _body_type_map(
    integration_audit: dict[str, Any],
    population_names: list[str],
) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name in population_names:
        for row in _population_rows(integration_audit, name):
            body_id = int(row["bodyId"])
            cell_type = str(row["type"]).strip()
            previous = mapping.get(body_id)
            if previous is not None and previous != cell_type:
                raise ValueError(
                    f"body ID {body_id} has conflicting frozen types: {previous!r} vs {cell_type!r}"
                )
            mapping[body_id] = cell_type
    return mapping


def _edge_family_summary(
    edges: pd.DataFrame,
    integration_audit: dict[str, Any],
    authority: dict[str, Any],
    source_name: str,
    target_name: str,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    source_rows = _population_rows(integration_audit, source_name)
    target_rows = _population_rows(integration_audit, target_name)
    source_ids = {int(row["bodyId"]) for row in source_rows}
    target_ids = {int(row["bodyId"]) for row in target_rows}
    source_types = {int(row["bodyId"]): str(row["type"]).strip() for row in source_rows}

    detail = edges.loc[
        edges.source.astype(int).isin(source_ids)
        & edges.target.astype(int).isin(target_ids)
    ].copy()
    detail["source_type"] = detail.source.astype(int).map(source_types)

    def modeled_sign(cell_type: str) -> int:
        evidence = _type_evidence(authority, str(cell_type))
        return int(evidence.get("modeled_sign", 0)) if evidence else 0

    detail["modeled_sign"] = detail.source_type.map(modeled_sign).fillna(0).astype(int)

    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        retained = detail.loc[detail.weight.astype(float).ge(threshold)]
        resolved = retained.modeled_sign.ne(0)
        sweep.append(
            {
                "min_weight": float(threshold),
                "edge_pairs": len(retained),
                "resolved_signed_edges": int(resolved.sum()),
                "unresolved_edges": int((~resolved).sum()),
                "signed_fraction": float(resolved.mean()) if len(retained) else 0.0,
                "modeled_sign_counts": {
                    str(key): int(value)
                    for key, value in sorted(Counter(retained.modeled_sign.astype(int)).items())
                },
            }
        )

    return {
        "source": source_name,
        "target": target_name,
        "edge_pairs": len(detail),
        "weight_sum": float(detail.weight.sum()) if len(detail) else 0.0,
        "threshold_sweep": sweep,
    }


def _sensitivity_requirements(
    authority: dict[str, Any],
    integration_audit: dict[str, Any],
    population_names: list[str],
) -> list[dict[str, Any]]:
    body_types = _body_type_map(integration_audit, population_names)
    by_type: dict[str, list[int]] = {}
    for body_id, cell_type in body_types.items():
        by_type.setdefault(cell_type, []).append(body_id)

    output: list[dict[str, Any]] = []
    for case in authority.get("mandatory_sensitivity_cases", []):
        row = dict(case)
        if row.get("name") == "PFNp_b_unresolved_sign":
            row["affected_body_ids"] = sorted(by_type.get("PFNp_b", []))
        output.append(row)
    return output


def build_external_integration_sign_audit(
    weights: pd.DataFrame,
    integration_audit: dict[str, Any],
    authority: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    edges = _normalize_edges(weights)
    required = [str(name) for name in config.get("required_presynaptic_populations", [])]
    thresholds = tuple(float(value) for value in config.get("thresholds", [1, 3, 5, 10]))
    _verify_authority_source(authority, dict(config.get("expected_transmitter_source", {})))

    population_signs = {
        name: _population_summary(_population_rows(integration_audit, name), authority)
        for name in required
    }
    edge_families = {
        f"{row['source']}->{row['target']}": _edge_family_summary(
            edges,
            integration_audit,
            authority,
            str(row["source"]),
            str(row["target"]),
            thresholds,
        )
        for row in config.get("edge_families", [])
    }

    ready = all(row["ready"] for row in population_signs.values())
    confidence_rows = [
        {
            "population": population,
            "type": cell_type,
            "confidence": evidence.get("confidence"),
        }
        for population, summary in population_signs.items()
        for cell_type, evidence in summary["type_evidence"].items()
        if evidence.get("confidence") is not None
    ]
    confidence_rows.sort(key=lambda row: (float(row["confidence"]), row["population"], row["type"]))

    return {
        "protocol": str(config.get("protocol", "malecns-integration-sign-audit-v2")),
        "dataset": str(config.get("dataset", "male-cns:v1.0")),
        "sign_evidence_source": "pinned_external_type_authority",
        "transmitter_authority_source": authority.get("source", {}),
        "required_presynaptic_populations": required,
        "population_signs": population_signs,
        "edge_families": edge_families,
        "thresholds": list(thresholds),
        "prediction_confidence_order": confidence_rows,
        "mandatory_sensitivity_cases": _sensitivity_requirements(
            authority,
            integration_audit,
            required,
        ),
        "ready_for_modeled_sign_probe": ready,
        "ready_with_sensitivity_requirements": ready
        and bool(authority.get("mandatory_sensitivity_cases")),
        "readiness_rule": str(config.get("readiness_rule", "")),
        "claim_boundary": str(config.get("claim_boundary", "Model sign audit only.")),
    }


def _require_sha256(path: Path, expected: str, label: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: observed={observed} expected={expected}"
        )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit modeled signs for the integration route using pinned external "
            "MaleCNS type-level transmitter evidence"
        )
    )
    parser.add_argument("weights")
    parser.add_argument("integration_audit")
    parser.add_argument("transmitter_authority")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output",
        default="results/route/integration-sign-audit-v2.json",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    audit_path = Path(args.integration_audit)
    authority_path = Path(args.transmitter_authority)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())

    observed_audit_sha = _require_sha256(
        audit_path,
        str(config["expected_integration_audit_sha256"]),
        "integration audit",
    )
    observed_authority_sha = _require_sha256(
        authority_path,
        str(config["expected_transmitter_authority_sha256"]),
        "transmitter authority",
    )

    integration_audit = json.loads(audit_path.read_text())
    authority = json.loads(authority_path.read_text())
    expected_weights_sha = str(
        integration_audit.get("inputs", {}).get("weights", {}).get("sha256", "")
    )
    if not expected_weights_sha:
        raise ValueError("integration audit does not seal the source weights SHA-256")
    observed_weights_sha = _require_sha256(weights_path, expected_weights_sha, "weights")

    weights = pd.read_feather(weights_path)
    report = build_external_integration_sign_audit(
        weights,
        integration_audit,
        authority,
        config,
    )
    report["runtime"] = {
        "git_branch": _git_output("branch", "--show-current"),
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_dirty_paths": (_git_output("status", "--porcelain") or "").splitlines(),
    }
    report["inputs"] = {
        "weights": {"path": str(weights_path), "sha256": observed_weights_sha},
        "integration_audit": {"path": str(audit_path), "sha256": observed_audit_sha},
        "transmitter_authority": {
            "path": str(authority_path),
            "sha256": observed_authority_sha,
        },
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

    if not report.get("ready_for_modeled_sign_probe", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
