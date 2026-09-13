from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _pb_token(instance: str) -> tuple[str, int] | None:
    match = re.search(r"_(L|R)(\d+)(?:_|$)", str(instance))
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def review_heading_topography(audit: dict[str, Any]) -> dict[str, Any]:
    if audit.get("protocol") != "malecns-heading-route-audit-v1":
        raise ValueError("unexpected heading route audit protocol")

    matches = [
        row
        for row in audit.get("direct_predictions", [])
        if row.get("source") == "EPG" and row.get("target") == "PFL3"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one EPG->PFL3 prediction, observed {len(matches)}")
    prediction = matches[0]

    parsed_edges: list[dict[str, Any]] = []
    unparsed_edges: list[dict[str, Any]] = []
    for edge in prediction.get("edges", []):
        source_token = _pb_token(str(edge.get("source_instance", "")))
        target_token = _pb_token(str(edge.get("target_instance", "")))
        row = {
            "source": int(edge["source"]),
            "target": int(edge["target"]),
            "source_instance": str(edge.get("source_instance", "")),
            "target_instance": str(edge.get("target_instance", "")),
            "weight": float(edge["weight"]),
            "source_pb": source_token,
            "target_pb": target_token,
        }
        if source_token is None or target_token is None:
            unparsed_edges.append(row)
            continue
        row["matching_pb_glomerulus"] = source_token == target_token
        parsed_edges.append(row)

    threshold_reports: dict[str, Any] = {}
    for threshold_value in audit.get("thresholds", [1, 3, 5, 10]):
        threshold = float(threshold_value)
        retained = [row for row in parsed_edges if row["weight"] >= threshold]
        unparsed_retained = [row for row in unparsed_edges if row["weight"] >= threshold]
        matching = [row for row in retained if row["matching_pb_glomerulus"]]
        total_weight = sum(row["weight"] for row in retained)
        matching_weight = sum(row["weight"] for row in matching)
        unmatched = [
            {
                "source": row["source"],
                "target": row["target"],
                "source_instance": row["source_instance"],
                "target_instance": row["target_instance"],
                "weight": row["weight"],
            }
            for row in retained
            if not row["matching_pb_glomerulus"]
        ]
        key = str(int(threshold) if threshold.is_integer() else threshold)
        threshold_reports[key] = {
            "structural_threshold": threshold,
            "retained_parsed_edges": len(retained),
            "retained_unparsed_edges": len(unparsed_retained),
            "matching_pb_edges": len(matching),
            "matching_pb_edge_fraction": len(matching) / len(retained) if retained else 0.0,
            "retained_weight_sum": total_weight,
            "matching_pb_weight_sum": matching_weight,
            "matching_pb_weight_fraction": matching_weight / total_weight if total_weight else 0.0,
            "unmatched_edges": unmatched,
        }

    return {
        "protocol": "malecns-heading-topography-review-v1",
        "dataset": audit.get("dataset", "unknown"),
        "source_protocol": audit["protocol"],
        "parsed_edge_count": len(parsed_edges),
        "unparsed_edge_count": len(unparsed_edges),
        "threshold_reports": threshold_reports,
        "interpretation": (
            "This review measures whether direct EPG->PFL3 structural edges preserve the same "
            "PB side/index label in source and target instance metadata. It does not assign a "
            "physical heading angle or preferred direction to any neuron."
        ),
        "claim_boundary": (
            "Descriptive anatomical topography only. Matching PB labels do not establish tuning, "
            "effective sign, neural dynamics, steering direction, or behavior."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review direct EPG->PFL3 PB-glomerulus alignment in a heading route audit"
    )
    parser.add_argument("audit")
    parser.add_argument(
        "--output", default="results/route/heading-topography-review-v1.json"
    )
    args = parser.parse_args()

    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text())
    report = review_heading_topography(audit)
    report["source_artifact"] = {
        "path": str(audit_path),
        "sha256": _sha256_file(audit_path),
    }
    report["runtime"] = {
        "git_branch": _git_output("branch", "--show-current"),
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_dirty_paths": (_git_output("status", "--porcelain") or "").splitlines(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
