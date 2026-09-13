from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_BASE = Path("configs/showcase_evidence_v3.json")
DEFAULT_E002C = Path("results/e002/pfl3-convergence-v1.json")
DEFAULT_FC2 = Path("results/route/fc2-goal-interface-audit-v1.json")
DEFAULT_OUTPUT = Path("results/showcase/showcase-evidence-local-v3.json")


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_showcase_inputs(base: dict[str, Any], e002c: dict[str, Any], fc2: dict[str, Any]) -> None:
    if base.get("protocol") != "who-farted-showcase-evidence-v3":
        raise ValueError("unexpected showcase evidence protocol")
    if base.get("comparison", {}).get("real_vs_rewire_headline_allowed", False):
        raise ValueError("local seal cannot enable real-vs-rewire headline")
    if base.get("behavioral_state", {}).get("malecns_behavior_claim_allowed", False):
        raise ValueError("local seal cannot enable MaleCNS behavior claim")
    if base.get("modeled_activity", {}).get("behavior_sync_allowed", True):
        raise ValueError("local seal requires mechanism probe to remain separate from chase")

    mechanism = base.get("mechanism_probe", {})
    if e002c.get("protocol") != mechanism.get("e002c_protocol"):
        raise ValueError("unexpected E002c protocol")
    if not bool(e002c.get("passed")):
        raise ValueError("local seal requires passed E002c")
    gates = {str(row.get("name")): bool(row.get("passed")) for row in e002c.get("gates", [])}
    if mechanism.get("e002b_required", True) and not gates.get("e002b_qualified", False):
        raise ValueError("local seal requires E002b-qualified E002c")
    expected_thresholds = {str(int(x)) for x in mechanism.get("all_thresholds_must_be_reported", [])}
    observed_thresholds = {str(key) for key in e002c.get("threshold_reports", {})}
    if not expected_thresholds <= observed_thresholds:
        raise ValueError("E002c is missing preregistered threshold reports")

    if fc2.get("protocol") != mechanism.get("fc2_protocol"):
        raise ValueError("unexpected FC2 goal-interface protocol")
    if mechanism.get("phase_mapping_must_remain_unresolved", True) and fc2.get("phase_mapping_status") != "unresolved":
        raise ValueError("FC2 phase mapping must remain unresolved in showcase v3/v4")
    for family in ("FC2A", "FC2B", "FC2C"):
        population = fc2.get("populations", {}).get(family, {})
        columns = population.get("instance_columns", {})
        if float(columns.get("parse_fraction", 0.0)) != 1.0:
            raise ValueError(f"{family} instance columns are not fully resolved")
        report = fc2.get("interfaces", {}).get(family, {}).get("threshold_reports", {}).get("10", {})
        if report.get("target_coverage") != "24/24":
            raise ValueError(f"{family}->PFL3 does not retain 24/24 target coverage at threshold 10")


def build_local_seal(
    base: dict[str, Any],
    e002c: dict[str, Any],
    fc2: dict[str, Any],
    *,
    e002c_sha256: str,
    fc2_sha256: str,
) -> dict[str, Any]:
    validate_showcase_inputs(base, e002c, fc2)
    sealed = deepcopy(base)
    sealed["mechanism_probe"]["e002c_sha256"] = str(e002c_sha256)
    sealed["mechanism_probe"]["fc2_sha256"] = str(fc2_sha256)
    sealed["local_evidence_seal"] = {
        "protocol": "who-farted-showcase-local-seal-v1",
        "semantic_validation": "passed",
        "scope": "renderer provenance only; does not promote scientific claims",
    }
    return sealed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and seal local showcase evidence artifacts")
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--e002c", default=str(DEFAULT_E002C))
    parser.add_argument("--fc2", default=str(DEFAULT_FC2))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    base = _load(args.base)
    e002c = _load(args.e002c)
    fc2 = _load(args.fc2)
    report = build_local_seal(
        base,
        e002c,
        fc2,
        e002c_sha256=_sha256(args.e002c),
        fc2_sha256=_sha256(args.fc2),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    print(f"e002c_sha256={report['mechanism_probe']['e002c_sha256']}")
    print(f"fc2_sha256={report['mechanism_probe']['fc2_sha256']}")


if __name__ == "__main__":
    main()
