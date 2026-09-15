from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .graph import GraphBundle
from .pfl3_descending_probe import run_e002e

PROTOCOL = "E002e-pfl3-descending-steering-v2"
PARENT_PROTOCOL = "E002e-pfl3-descending-steering-v1"
DEFAULT_PROTOCOL = Path("configs/e002e_descending_steering_protocol_v2.json")
DEFAULT_E002D = Path("results/e002/pfl3-phase-comparison-v1.json")
DEFAULT_CROSSWALK = Path("results/e002/pfl3-phase-crosswalk-v1.json")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonzero_mirrored_phases(protocol: dict[str, Any]) -> list[str]:
    phases: list[str] = []
    for negative, positive in protocol["mirrored_phase_pairs_deg"]:
        phases.extend([str(int(negative)), str(int(positive))])
    return phases


def _side_swap_confirmation(
    phase_reports: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    phase_keys = _nonzero_mirrored_phases(protocol)
    rows: list[dict[str, Any]] = []
    for key in phase_keys:
        row = phase_reports[key]
        intact = float(row["turn"])
        swapped = float(row["side_swap_turn"])
        reversed_sign = intact * swapped < 0.0
        rows.append(
            {
                "phase_deg": int(key),
                "turn": intact,
                "side_swap_turn": swapped,
                "reversed_sign": reversed_sign,
            }
        )

    zero = phase_reports.get("0")
    zero_report = None
    if zero is not None:
        zero_report = {
            "turn": float(zero["turn"]),
            "side_swap_turn": float(zero["side_swap_turn"]),
            "gate_role": "descriptive_baseline_only",
        }
    return all(bool(row["reversed_sign"]) for row in rows), {
        "required_phase_count": len(rows),
        "reversed_phase_count": sum(bool(row["reversed_sign"]) for row in rows),
        "phase_reports": rows,
        "zero_phase": zero_report,
    }


def run_e002e_v2(
    protocol: dict[str, Any],
    e002d: dict[str, Any],
    crosswalk: dict[str, Any],
    bundle: GraphBundle,
    *,
    seed: int,
    steps: int = 32,
) -> dict[str, Any]:
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("unexpected E002e v2 protocol")
    if int(seed) != int(protocol["confirmation_seed"]):
        raise ValueError("E002e v2 must use the frozen confirmation seed")

    parent = dict(protocol)
    parent["protocol"] = PARENT_PROTOCOL
    parent["claim_boundary"] = protocol["claim_boundary"]
    report = run_e002e(parent, e002d, crosswalk, bundle, seed=seed, steps=steps)
    report["protocol"] = PROTOCOL
    report["parent_protocol"] = PARENT_PROTOCOL
    report["design_status"] = protocol["design_status"]

    primary = report["threshold_reports"][str(int(protocol["primary_structural_threshold"]))][
        "phase_reports"
    ]
    side_swap_ok, side_swap_detail = _side_swap_confirmation(primary, protocol)
    for gate in report["gates"]:
        if gate["name"] == "left_right_group_swap_reverses_turn_curve":
            gate.clear()
            gate.update(
                {
                    "name": "left_right_group_swap_reverses_nonzero_mirrored_phases",
                    "passed": side_swap_ok,
                    **side_swap_detail,
                }
            )
            break
    else:
        raise RuntimeError("parent E002e side-swap gate missing")

    report["passed"] = all(bool(row["passed"]) for row in report["gates"])
    report["passed_gate_count"] = sum(bool(row["passed"]) for row in report["gates"])
    report["run_config"]["confirmation_seed_frozen"] = True
    report["claim_boundary"] = protocol["claim_boundary"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent E002e v2 confirmation")
    parser.add_argument("bundle")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--e002d", default=str(DEFAULT_E002D))
    parser.add_argument("--crosswalk", default=str(DEFAULT_CROSSWALK))
    parser.add_argument("--output", default="results/e002/pfl3-descending-steering-v2.json")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    e002d = json.loads(Path(args.e002d).read_text())
    crosswalk = json.loads(Path(args.crosswalk).read_text())
    bundle = GraphBundle.load(args.bundle)
    seed = int(protocol["confirmation_seed"])
    report = run_e002e_v2(
        protocol,
        e002d,
        crosswalk,
        bundle,
        seed=seed,
        steps=args.steps,
    )
    report["input_sha256"] = {
        "protocol": _sha256(args.protocol),
        "e002d": _sha256(args.e002d),
        "crosswalk": _sha256(args.crosswalk),
        "bundle_manifest": _sha256(Path(args.bundle) / "manifest.json"),
        "bundle_edges": _sha256(Path(args.bundle) / "edges.parquet"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    print(f"passed={report['passed']} gates={report['passed_gate_count']}/{report['gate_count']}")
    for gate in report["gates"]:
        print(f"  {'PASS' if gate['passed'] else 'FAIL'}  {gate['name']}")


if __name__ == "__main__":
    main()
