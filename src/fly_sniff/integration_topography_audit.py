from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .literature_route_audit import _git_output, _sha256_file

DEFAULT_CONFIG = Path("configs/integration_topography_audit_v1.json")


def _match_int(value: Any, pattern: str | None) -> int | None:
    if not pattern:
        return None
    match = re.search(pattern, str(value or ""))
    return int(match.group(1)) if match else None


def _match_text(value: Any, pattern: str | None) -> str | None:
    if not pattern:
        return None
    match = re.search(pattern, str(value or ""))
    return str(match.group(1)) if match else None


def _population_rows(integration_audit: dict[str, Any], name: str) -> list[dict[str, Any]]:
    populations = integration_audit.get("populations", {})
    if name not in populations:
        raise ValueError(f"integration audit missing population {name!r}")
    rows = [dict(row) for row in populations[name].get("rows", [])]
    expected = int(populations[name].get("count", len(rows)))
    if len(rows) != expected:
        raise ValueError(
            f"integration audit population {name!r} row count mismatch: "
            f"rows={len(rows)} expected={expected}"
        )
    return rows


def _population_topography(
    integration_audit: dict[str, Any],
    name: str,
    rule: dict[str, Any],
) -> dict[str, Any]:
    rows = _population_rows(integration_audit, name)
    column_pattern = rule.get("column_regex")
    side_pattern = rule.get("side_label_regex")

    parsed_columns: dict[int, int] = {}
    parsed_sides: dict[int, str] = {}
    for row in rows:
        body_id = int(row["bodyId"])
        column = _match_int(row.get("instance"), column_pattern)
        side = _match_text(row.get("instance"), side_pattern)
        if column is not None:
            parsed_columns[body_id] = column
        if side is not None:
            parsed_sides[body_id] = side

    return {
        "body_id_count": len(rows),
        "column_parsed_count": len(parsed_columns),
        "column_parse_fraction": len(parsed_columns) / len(rows) if rows else 0.0,
        "columns": sorted(set(parsed_columns.values())),
        "unparsed_column_body_ids": sorted(
            int(row["bodyId"])
            for row in rows
            if int(row["bodyId"]) not in parsed_columns
        ),
        "side_label_parsed_count": len(parsed_sides),
        "side_label_counts": {
            side: sum(value == side for value in parsed_sides.values())
            for side in sorted(set(parsed_sides.values()))
        },
        "body_columns": {str(key): value for key, value in sorted(parsed_columns.items())},
        "body_side_labels": {str(key): value for key, value in sorted(parsed_sides.items())},
    }


def _direct_prediction(
    integration_audit: dict[str, Any],
    source: str,
    target: str,
) -> dict[str, Any]:
    for row in integration_audit.get("direct_predictions", []):
        if str(row.get("source")) == source and str(row.get("target")) == target:
            edges = list(row.get("edges", []))
            expected = int(row.get("observed_edge_pairs", len(edges)))
            if len(edges) != expected:
                raise ValueError(
                    f"direct prediction {source}->{target} is truncated: "
                    f"stored={len(edges)} expected={expected}"
                )
            return dict(row)
    raise ValueError(f"integration audit missing direct prediction {source}->{target}")


def _matrix_entries(
    edges: list[dict[str, Any]],
    source_rule: dict[str, Any],
    target_rule: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    matrix: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: {"edge_pairs": 0.0, "weight_sum": 0.0}
    )
    parse = {
        "source_column_edges": 0,
        "target_column_edges": 0,
        "both_column_edges": 0,
    }
    for edge in edges:
        source_column = _match_int(edge.get("source_instance"), source_rule.get("column_regex"))
        target_column = _match_int(edge.get("target_instance"), target_rule.get("column_regex"))
        if source_column is not None:
            parse["source_column_edges"] += 1
        if target_column is not None:
            parse["target_column_edges"] += 1
        if source_column is None or target_column is None:
            continue
        parse["both_column_edges"] += 1
        cell = matrix[(source_column, target_column)]
        cell["edge_pairs"] += 1.0
        cell["weight_sum"] += float(edge["weight"])

    entries = [
        {
            "source_column": source_column,
            "target_column": target_column,
            "edge_pairs": int(values["edge_pairs"]),
            "weight_sum": float(values["weight_sum"]),
        }
        for (source_column, target_column), values in matrix.items()
    ]
    entries.sort(
        key=lambda row: (
            -row["weight_sum"],
            row["source_column"],
            row["target_column"],
        )
    )
    return entries, parse


def _instance_side_summary(
    edges: list[dict[str, Any]],
    target_rule: dict[str, Any],
) -> dict[str, Any]:
    pattern = target_rule.get("side_label_regex")
    counts: dict[str, int] = defaultdict(int)
    weights: dict[str, float] = defaultdict(float)
    body_ids: dict[str, set[int]] = defaultdict(set)
    unparsed: set[int] = set()

    for edge in edges:
        side = _match_text(edge.get("target_instance"), pattern)
        body_id = int(edge["target"])
        if side is None:
            if pattern:
                unparsed.add(body_id)
            continue
        counts[side] += 1
        weights[side] += float(edge["weight"])
        body_ids[side].add(body_id)

    return {
        "edge_counts": dict(sorted(counts.items())),
        "weight_sums": {key: float(value) for key, value in sorted(weights.items())},
        "target_body_ids": {
            key: sorted(values) for key, values in sorted(body_ids.items())
        },
        "unparsed_target_body_ids": sorted(unparsed),
    }


def _family_topography(
    integration_audit: dict[str, Any],
    source: str,
    target: str,
    rules: dict[str, dict[str, Any]],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    prediction = _direct_prediction(integration_audit, source, target)
    source_rule = rules.get(source, {})
    target_rule = rules.get(target, {})
    edges = list(prediction.get("edges", []))

    sweeps: list[dict[str, Any]] = []
    for threshold in thresholds:
        retained = [edge for edge in edges if float(edge["weight"]) >= threshold]
        matrix, parse = _matrix_entries(retained, source_rule, target_rule)
        sweeps.append(
            {
                "min_weight": float(threshold),
                "edge_pairs": len(retained),
                "source_column_parse_fraction": (
                    parse["source_column_edges"] / len(retained) if retained else 0.0
                ),
                "target_column_parse_fraction": (
                    parse["target_column_edges"] / len(retained) if retained else 0.0
                ),
                "both_column_parse_fraction": (
                    parse["both_column_edges"] / len(retained) if retained else 0.0
                ),
                "column_matrix": matrix,
                "target_instance_side": _instance_side_summary(retained, target_rule),
            }
        )

    return {
        "source": source,
        "target": target,
        "observed_edge_pairs": int(prediction["observed_edge_pairs"]),
        "observed_weight_sum": float(prediction["observed_weight_sum"]),
        "threshold_sweep": sweeps,
    }


def build_integration_topography_audit(
    integration_audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rules = {
        str(name): dict(rule)
        for name, rule in config.get("population_instance_rules", {}).items()
    }
    thresholds = tuple(float(value) for value in config.get("thresholds", [1, 3, 5, 10]))

    populations = {
        name: _population_topography(integration_audit, name, rule)
        for name, rule in rules.items()
    }
    families = [
        _family_topography(
            integration_audit,
            str(row["source"]),
            str(row["target"]),
            rules,
            thresholds,
        )
        for row in config.get("edge_families", [])
    ]

    return {
        "protocol": str(config.get("protocol", "malecns-integration-topography-audit-v1")),
        "dataset": str(config.get("dataset", "male-cns:v1.0")),
        "thresholds": list(thresholds),
        "populations": populations,
        "edge_families": families,
        "directional_role_status": "unresolved",
        "interpretation_rules": list(config.get("interpretation_rules", [])),
        "claim_boundary": str(config.get("claim_boundary", "Descriptive topology audit only.")),
    }


def _require_sha256(path: Path, expected: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            "integration audit SHA-256 mismatch: "
            f"observed={observed} expected={expected}"
        )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit anatomical instance-column topology along the frozen integration route"
    )
    parser.add_argument("integration_audit")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output",
        default="results/route/integration-topography-audit-v1.json",
    )
    args = parser.parse_args()

    audit_path = Path(args.integration_audit)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    observed_audit_sha = _require_sha256(
        audit_path,
        str(config["expected_integration_audit_sha256"]),
    )
    integration_audit = json.loads(audit_path.read_text())
    report = build_integration_topography_audit(integration_audit, config)
    report["runtime"] = {
        "git_branch": _git_output("branch", "--show-current"),
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_dirty_paths": (_git_output("status", "--porcelain") or "").splitlines(),
    }
    report["inputs"] = {
        "integration_audit": {
            "path": str(audit_path),
            "sha256": observed_audit_sha,
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
