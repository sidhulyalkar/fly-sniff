from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .literature_route_audit import _git_output, _sha256_file


def _rows(audit: dict[str, Any], name: str) -> list[dict[str, Any]]:
    block = audit.get("populations", {}).get(name)
    if block is None:
        raise ValueError(f"missing population {name}")
    rows = [dict(row) for row in block.get("rows", [])]
    if len(rows) != int(block.get("count", len(rows))):
        raise ValueError(f"truncated population {name}")
    ids = [int(row["bodyId"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate body IDs in {name}")
    return rows


def _prediction(audit: dict[str, Any], source: str, target: str) -> dict[str, Any]:
    matches = [row for row in audit.get("direct_predictions", []) if row.get("source") == source and row.get("target") == target]
    if len(matches) != 1:
        raise ValueError(f"expected one {source} to {target} prediction")
    result = dict(matches[0])
    edges = list(result.get("edges", []))
    if len(edges) != int(result.get("observed_edge_pairs", len(edges))):
        raise ValueError(f"truncated prediction {source} to {target}")
    return result


def _columns(rows: list[dict[str, Any]], pattern: str) -> dict[str, Any]:
    parsed: dict[int, int] = {}
    for row in rows:
        match = re.search(pattern, str(row.get("instance", "")))
        if match:
            parsed[int(row["bodyId"])] = int(match.group(1))
    return {
        "parsed_count": len(parsed),
        "parse_fraction": len(parsed) / len(rows) if rows else 0.0,
        "columns": sorted(set(parsed.values())),
        "body_columns": {str(k): v for k, v in sorted(parsed.items())},
        "unparsed_body_ids": sorted(int(row["bodyId"]) for row in rows if int(row["bodyId"]) not in parsed),
    }


def build_report(audit: dict[str, Any], config: dict[str, Any], signs: dict[str, Any]) -> dict[str, Any]:
    if audit.get("protocol") != "malecns-goal-relay-audit-v1":
        raise ValueError("unexpected goal-relay protocol")
    if audit.get("dataset") != config.get("dataset") or signs.get("dataset") != config.get("dataset"):
        raise ValueError("dataset mismatch")

    target = str(config["target_population"])
    thresholds = [float(x) for x in config["thresholds"]]
    pattern = str(config["instance_column_regex"])
    target_rows = _rows(audit, target)
    target_ids = {int(row["bodyId"]) for row in target_rows}
    populations: dict[str, Any] = {
        target: {"count": len(target_rows), "body_ids": sorted(target_ids), "instance_columns": _columns(target_rows, pattern)}
    }
    interfaces: dict[str, Any] = {}

    for source in config["source_populations"]:
        source = str(source)
        source_rows = _rows(audit, source)
        source_ids = {int(row["bodyId"]) for row in source_rows}
        sign = signs.get("predictions", {}).get(source)
        if sign is None or int(sign.get("neuron_count", -1)) != len(source_rows) or int(sign.get("modeled_sign", 0)) != 1:
            raise ValueError(f"sign authority mismatch for {source}")
        populations[source] = {
            "count": len(source_rows),
            "body_ids": sorted(source_ids),
            "instance_columns": _columns(source_rows, pattern),
            "transmitter_prediction": dict(sign),
        }
        pred = _prediction(audit, source, target)
        edges = [dict(edge) for edge in pred.get("edges", [])]
        sweeps: dict[str, Any] = {}
        for threshold in thresholds:
            kept = [edge for edge in edges if float(edge["weight"]) >= threshold]
            src = sorted({int(edge["source"]) for edge in kept})
            dst = sorted({int(edge["target"]) for edge in kept})
            if not set(src).issubset(source_ids) or not set(dst).issubset(target_ids):
                raise ValueError(f"edge endpoint mismatch for {source}")
            key = str(int(threshold) if threshold.is_integer() else threshold)
            sweeps[key] = {
                "structural_threshold": threshold,
                "edge_pairs": len(kept),
                "weight_sum": float(sum(float(edge["weight"]) for edge in kept)),
                "source_body_ids": src,
                "target_body_ids": dst,
                "source_coverage": f"{len(src)}/{len(source_ids)}",
                "target_coverage": f"{len(dst)}/{len(target_ids)}",
            }
        interfaces[source] = {"source": source, "target": target, "threshold_reports": sweeps}

    return {
        "protocol": config["protocol"],
        "dataset": config["dataset"],
        "thresholds": thresholds,
        "populations": populations,
        "interfaces": interfaces,
        "phase_mapping_status": "unresolved",
        "interpretation_rules": config.get("interpretation_rules", []),
        "claim_boundary": config.get("claim_boundary", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("goal_relay_audit")
    parser.add_argument("--config", default="configs/fc2_goal_interface_audit_v1.json")
    parser.add_argument("--sign-authority", default="authority/malecns-v1.0-fc2-transmitter-evidence.json")
    parser.add_argument("--output", default="results/route/fc2-goal-interface-audit-v1.json")
    args = parser.parse_args()

    audit_path = Path(args.goal_relay_audit)
    config_path = Path(args.config)
    sign_path = Path(args.sign_authority)
    config = json.loads(config_path.read_text())
    observed = _sha256_file(audit_path)
    expected = str(config["expected_goal_relay_audit_sha256"])
    if observed != expected:
        raise ValueError(f"goal-relay audit hash mismatch: {observed} != {expected}")

    report = build_report(json.loads(audit_path.read_text()), config, json.loads(sign_path.read_text()))
    report["runtime"] = {
        "git_branch": _git_output("branch", "--show-current"),
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_dirty_paths": (_git_output("status", "--porcelain") or "").splitlines(),
    }
    report["inputs"] = {
        "goal_relay_audit": {"path": str(audit_path), "sha256": observed},
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "sign_authority": {"path": str(sign_path), "sha256": _sha256_file(sign_path)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"protocol": report["protocol"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
