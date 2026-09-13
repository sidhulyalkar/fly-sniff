from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROTOCOL = "odor-motion-qualification-v2"
SCHEMA_VERSION = 1


def qualification_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def build_receipt(q, motion_config, arena, plume, sensors, edge, fixed):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": q["status"],
        "claim_boundary": q["claim_boundary"],
        "controller_access_in_v1": False,
        "navigation_performance_used": False,
        "may_retune_estimator_from_this_report": False,
        "environment": {
            "arena": asdict(arena),
            "plume": asdict(plume),
            "sensor": asdict(sensors),
        },
        "estimator": asdict(motion_config),
        "traveling_edge_assay": edge,
        "fixed_plume_probe_assay": fixed,
        "functional_v2_allowed": bool(
            edge["status"] == "pass" and fixed["status"] == "signal_present_requires_review"
        ),
    }
    return {"qualification": payload, "qualification_sha256": qualification_sha256(payload)}


def write_qualification(path: str | Path, bundle: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def load_qualification(path: str | Path):
    bundle = json.loads(Path(path).read_text())
    payload = bundle.get("qualification")
    expected = bundle.get("qualification_sha256")
    if not isinstance(payload, dict) or not isinstance(expected, str):
        raise TypeError("invalid odor-motion qualification bundle")
    if qualification_sha256(payload) != expected:
        raise ValueError("odor-motion qualification SHA-256 mismatch")
    if payload.get("protocol") != PROTOCOL or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported odor-motion qualification protocol/schema")
    return bundle
