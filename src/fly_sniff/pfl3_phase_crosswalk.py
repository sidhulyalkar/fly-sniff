from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("configs/e002d_pfl3_phase_comparison_protocol_v3.json")
DEFAULT_E002C = Path("results/e002/pfl3-convergence-v1.json")
DEFAULT_FC2 = Path("results/route/fc2-goal-interface-audit-v1.json")
DEFAULT_HEADING = Path("results/route/heading-route-audit-v1.json")
DEFAULT_STEERING = Path("configs/steering_scaffold_candidate_v1.json")

_PFL3_RE = re.compile(r"^PFL3\(PB12c\)_([LR])(\d+)_C(\d+)(?:_irreg)?$")
_PB_ORDER = [f"L{i}" for i in range(9, 0, -1)] + [f"R{i}" for i in range(1, 10)]


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def build_crosswalk(
    config: dict[str, Any],
    e002c: dict[str, Any],
    fc2: dict[str, Any],
    heading: dict[str, Any],
    steering: dict[str, Any],
) -> dict[str, Any]:
    if config.get("protocol") != "E002d-pfl3-relative-phase-comparison-v3":
        raise ValueError("unexpected E002d protocol")
    if not bool(e002c.get("passed")):
        raise ValueError("E002d requires passed E002c")
    if fc2.get("protocol") != "malecns-fc2-goal-interface-audit-v1":
        raise ValueError("unexpected FC2 audit protocol")
    if fc2.get("phase_mapping_status") != "unresolved":
        raise ValueError("FC2 phase mapping must remain unresolved before E002d")
    if heading.get("protocol") != "malecns-heading-route-audit-v1":
        raise ValueError("unexpected heading route protocol")
    if steering.get("protocol") != "malecns-steering-scaffold-candidate-v1":
        raise ValueError("unexpected steering scaffold protocol")

    goal_phase = config["mapping"]["nine_column_goal_sensitivity_degrees"]
    heading_phase = config["mapping"]["heading_glomerulus_sequence_degrees_left_to_right"]
    if len(goal_phase) != 9 or len(heading_phase) != 18:
        raise ValueError("phase mapping dimensions drifted")
    pb_to_phase = dict(zip(_PB_ORDER, heading_phase, strict=True))

    fc2_columns = {
        int(k): int(v)
        for k, v in fc2["populations"]["PFL3"]["instance_columns"]["body_columns"].items()
    }
    left_group = {int(x) for x in steering["candidate_roles"]["turn_drive_left"]}
    right_group = {int(x) for x in steering["candidate_roles"]["turn_drive_right"]}
    if left_group & right_group or len(left_group) != 12 or len(right_group) != 12:
        raise ValueError("frozen PFL3 steering-side groups must be disjoint 12+12")

    rows = heading["populations"]["PFL3"]["rows"]
    records: list[dict[str, Any]] = []
    for row in rows:
        body_id = int(row["bodyId"])
        instance = str(row["instance"])
        match = _PFL3_RE.fullmatch(instance)
        if match is None:
            raise ValueError(f"unparsed PFL3 instance: {instance}")
        pb = f"{match.group(1)}{int(match.group(2))}"
        column = int(match.group(3))
        if fc2_columns.get(body_id) != column:
            raise ValueError(f"PFL3 column mismatch for body {body_id}")
        if pb not in pb_to_phase:
            raise ValueError(f"PB label outside frozen 18-glomerulus order: {pb}")
        if body_id in left_group:
            readout_side = "turn_drive_left"
        elif body_id in right_group:
            readout_side = "turn_drive_right"
        else:
            raise ValueError(f"PFL3 body missing from frozen steering-side groups: {body_id}")
        records.append(
            {
                "body_id": body_id,
                "instance": instance,
                "pb_label": pb,
                "column": column,
                "instance_pb_side": match.group(1),
                "readout_side": readout_side,
                "irregular_instance": instance.endswith("_irreg"),
                "heading_phase_deg": float(pb_to_phase[pb]),
                "goal_phase_sensitivity_deg": float(goal_phase[column - 1]),
            }
        )

    records.sort(key=lambda row: (row["column"], row["body_id"]))
    body_ids = {row["body_id"] for row in records}
    if body_ids != left_group | right_group or len(records) != 24:
        raise ValueError("phase crosswalk does not cover the exact frozen 24 PFL3 bodies")

    return {
        "protocol": "E002d-pfl3-phase-crosswalk-v1",
        "dataset": "male-cns:v1.0",
        "ready_for_phase_probe": True,
        "absolute_world_offset_status": "unresolved-nuisance-parameter",
        "pb_order_left_to_right": _PB_ORDER,
        "nine_column_mapping_status": "literature-sensitivity-only",
        "records": records,
        "counts": {
            "PFL3": len(records),
            "turn_drive_left": len(left_group),
            "turn_drive_right": len(right_group),
            "irregular_instances": sum(bool(row["irregular_instance"]) for row in records),
        },
        "claim_boundary": "Body-ID phase crosswalk only. It does not establish absolute preferred angles, physiological tuning, a steering command, or behavior.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the preregistered E002d PFL3 phase crosswalk")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--e002c", default=str(DEFAULT_E002C))
    parser.add_argument("--fc2", default=str(DEFAULT_FC2))
    parser.add_argument("--heading", default=str(DEFAULT_HEADING))
    parser.add_argument("--steering", default=str(DEFAULT_STEERING))
    parser.add_argument("--output", default="results/e002/pfl3-phase-crosswalk-v1.json")
    args = parser.parse_args()
    report = build_crosswalk(_load(args.config), _load(args.e002c), _load(args.fc2), _load(args.heading), _load(args.steering))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
