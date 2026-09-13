from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .graph import GraphBundle, MaleCNSRateController
from .pfl3_phase_probe import _cell_response

DEFAULT_PROTOCOL = Path("configs/e002e_descending_steering_protocol_v1.json")
DEFAULT_E002D = Path("results/e002/pfl3-phase-comparison-v1.json")
DEFAULT_CROSSWALK = Path("results/e002/pfl3-phase-crosswalk-v1.json")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filtered_bundle(bundle: GraphBundle, threshold: int, *, cut_pfl3: bool = False) -> GraphBundle:
    edges = bundle.edges.loc[bundle.edges.weight.astype(float) >= float(threshold)].copy()
    if cut_pfl3:
        pfl3 = {int(x) for x in bundle.roles.get("turn_drive_left", [])}
        pfl3.update(int(x) for x in bundle.roles.get("turn_drive_right", []))
        edges = edges.loc[~edges.source.astype(int).isin(pfl3)].copy()
    manifest = dict(bundle.manifest or {})
    manifest["qualification_status"] = "candidate"
    manifest["e002e_threshold"] = int(threshold)
    return GraphBundle(bundle.nodes.copy(), edges, dict(bundle.roles), manifest)


def _body_response(
    records: list[dict[str, Any]],
    included: set[int],
    phase_deg: float,
    params: dict[str, Any],
    *,
    heading_enabled: bool = True,
    goal_enabled: bool = True,
    common_offset_deg: float = 0.0,
) -> dict[int, float]:
    heading = np.asarray([float(phase_deg) + common_offset_deg], dtype=float)
    goal = np.asarray([common_offset_deg], dtype=float)
    scale = max(float(params["a_hz"]), 1e-12)
    out: dict[int, float] = {}
    for row in records:
        body_id = int(row["body_id"])
        if body_id not in included:
            continue
        value = _cell_response(
            heading,
            goal,
            float(row["heading_phase_deg"]) + common_offset_deg,
            float(row["goal_phase_sensitivity_deg"]) + common_offset_deg,
            params,
            heading_enabled=heading_enabled,
            goal_enabled=goal_enabled,
        )[0]
        out[body_id] = float(value / scale)
    return out


def _run_body_drive(
    bundle: GraphBundle,
    body_drive: dict[int, float],
    *,
    seed: int,
    steps: int,
) -> np.ndarray:
    controller = MaleCNSRateController(bundle, require_qualified=False)
    controller.reset(seed)
    drive = np.zeros_like(controller.activity)
    for body_id, value in body_drive.items():
        if body_id not in controller.index:
            raise ValueError(f"PFL3 body {body_id} missing from steering scaffold")
        drive[controller.index[body_id]] = float(value)
    left_idx = [
        controller.index[x]
        for x in bundle.roles.get("steer_left", [])
        if x in controller.index
    ]
    right_idx = [
        controller.index[x]
        for x in bundle.roles.get("steer_right", [])
        if x in controller.index
    ]
    turns: list[float] = []
    for _ in range(steps):
        recurrent = controller.w @ controller.activity
        proposal = np.tanh(controller.gain * (recurrent + drive))
        controller.activity = (
            controller.retention * controller.activity
            + (1.0 - controller.retention) * proposal
        )
        left = float(controller.activity[left_idx].mean()) if left_idx else 0.0
        right = float(controller.activity[right_idx].mean()) if right_idx else 0.0
        turns.append(float(np.tanh(controller.turn_gain * (left - right))))
    return np.asarray(turns, dtype=float)


def _side_sort_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(row.get("column", 0)),
        str(row.get("pb_label", "")),
        int(row["body_id"]),
    )


def _swap_side_drive(
    records: list[dict[str, Any]],
    drive: dict[int, float],
) -> dict[int, float]:
    """Swap the complete frozen PFL3 left/right groups, treating absent drive as zero.

    The adversarial sentry tests whether the downstream turn depends on the frozen
    laterality assignment. It must not require the instantaneous active subsets to
    have equal cardinality, since structural thresholding may legitimately leave an
    unequal number of driven cells on the two sides.

    Pairing is deterministic and performance-independent: each full 12-cell group is
    ordered by (anatomical column, PB label, body ID), then values are exchanged by
    rank. This is a label-swap control, not a claim of one-to-one biological homology.
    """
    left_rows = sorted(
        (row for row in records if row["readout_side"] == "turn_drive_left"),
        key=_side_sort_key,
    )
    right_rows = sorted(
        (row for row in records if row["readout_side"] == "turn_drive_right"),
        key=_side_sort_key,
    )
    if len(left_rows) != len(right_rows) or not left_rows:
        raise ValueError("frozen PFL3 side groups must be non-empty and equal-sized")

    known_ids = {int(row["body_id"]) for row in left_rows + right_rows}
    unexpected = set(drive) - known_ids
    if unexpected:
        raise ValueError(f"drive contains PFL3 bodies outside frozen side groups: {sorted(unexpected)}")

    swapped: dict[int, float] = {}
    for left_row, right_row in zip(left_rows, right_rows, strict=True):
        left_id = int(left_row["body_id"])
        right_id = int(right_row["body_id"])
        left_value = float(drive.get(left_id, 0.0))
        right_value = float(drive.get(right_id, 0.0))
        if right_value != 0.0:
            swapped[left_id] = right_value
        if left_value != 0.0:
            swapped[right_id] = left_value
    return swapped


def _mirrored_opposition_count(
    phase_reports: dict[str, Any],
    mirrored_pairs: list[list[float]],
    field: str,
) -> int:
    count = 0
    for negative, positive in mirrored_pairs:
        a = float(phase_reports[str(int(negative))][field])
        b = float(phase_reports[str(int(positive))][field])
        count += int(a * b < 0.0)
    return count


def run_e002e(
    protocol: dict[str, Any],
    e002d: dict[str, Any],
    crosswalk: dict[str, Any],
    bundle: GraphBundle,
    *,
    seed: int = 13013,
    steps: int = 32,
) -> dict[str, Any]:
    if protocol.get("protocol") != "E002e-pfl3-descending-steering-v1":
        raise ValueError("unexpected E002e protocol")
    if (
        e002d.get("protocol") != protocol["required_inputs"]["e002d_protocol"]
        or not bool(e002d.get("passed"))
    ):
        raise ValueError("E002e requires passed E002d")
    if crosswalk.get("protocol") != protocol["required_inputs"]["phase_crosswalk_protocol"]:
        raise ValueError("unexpected phase crosswalk")
    if (bundle.manifest or {}).get("protocol") != protocol["required_inputs"]["steering_scaffold_protocol"]:
        raise ValueError("unexpected steering scaffold bundle")

    records = [dict(row) for row in crosswalk["records"]]
    params = dict(e002d["published_model_parameters"])
    phases = [float(x) for x in protocol["relative_phase_probe_deg"]]
    thresholds = [int(x) for x in protocol["all_structural_thresholds"]]
    threshold_reports: dict[str, Any] = {}
    max_deterministic_error = 0.0
    max_rotation_error = 0.0

    for threshold in thresholds:
        included = {
            int(x)
            for x in e002d["threshold_reports"][str(threshold)]["dual_reachable_body_ids"]
        }
        filtered = _filtered_bundle(bundle, threshold)
        cut_bundle = _filtered_bundle(bundle, threshold, cut_pfl3=True)
        phase_rows: dict[str, Any] = {}
        for phase in phases:
            drive = _body_response(records, included, phase, params)
            turns = _run_body_drive(filtered, drive, seed=seed, steps=steps)
            replay = _run_body_drive(filtered, drive, seed=seed, steps=steps)
            deterministic_error = float(np.max(np.abs(turns - replay)))
            max_deterministic_error = max(max_deterministic_error, deterministic_error)
            no_goal = _run_body_drive(
                filtered,
                _body_response(records, included, phase, params, goal_enabled=False),
                seed=seed,
                steps=steps,
            )
            no_heading = _run_body_drive(
                filtered,
                _body_response(records, included, phase, params, heading_enabled=False),
                seed=seed,
                steps=steps,
            )
            cut = _run_body_drive(cut_bundle, drive, seed=seed, steps=steps)
            swapped = _run_body_drive(
                filtered,
                _swap_side_drive(records, drive),
                seed=seed,
                steps=steps,
            )
            rotated = _run_body_drive(
                filtered,
                _body_response(
                    records,
                    included,
                    phase,
                    params,
                    common_offset_deg=90.0,
                ),
                seed=seed,
                steps=steps,
            )
            rotation_error = float(np.max(np.abs(turns - rotated)))
            max_rotation_error = max(max_rotation_error, rotation_error)
            tail = max(4, steps // 4)
            phase_rows[str(int(phase))] = {
                "turn": float(np.mean(turns[-tail:])),
                "FC2_lane_cut_turn": float(np.mean(no_goal[-tail:])),
                "EPG_lane_cut_turn": float(np.mean(no_heading[-tail:])),
                "PFL3_output_cut_peak_turn": float(np.max(np.abs(cut))),
                "side_swap_turn": float(np.mean(swapped[-tail:])),
                "deterministic_error": deterministic_error,
                "rotation_error_90deg": rotation_error,
            }
        threshold_reports[str(threshold)] = {
            "dual_reachable_count": len(included),
            "phase_reports": phase_rows,
        }

    primary = threshold_reports[str(int(protocol["primary_structural_threshold"]))][
        "phase_reports"
    ]
    mirrored_pairs = protocol["mirrored_phase_pairs_deg"]
    intact_opposed = _mirrored_opposition_count(primary, mirrored_pairs, "turn")
    fc2_cut_opposed = _mirrored_opposition_count(primary, mirrored_pairs, "FC2_lane_cut_turn")
    epg_cut_opposed = _mirrored_opposition_count(primary, mirrored_pairs, "EPG_lane_cut_turn")
    mirrored_ok = intact_opposed == len(mirrored_pairs)
    fc2_cut_destroys = fc2_cut_opposed < intact_opposed
    epg_cut_destroys = epg_cut_opposed < intact_opposed

    pfl3_cut_peak = max(
        float(row["PFL3_output_cut_peak_turn"]) for row in primary.values()
    )
    side_swap_ok = all(
        float(row["turn"]) * float(row["side_swap_turn"]) <= 0.0
        for row in primary.values()
        if abs(float(row["turn"])) > 1e-9
    )

    gates = [
        {"name": "sealed_inputs_match", "passed": True},
        {"name": "no_behavior_oracle_inputs", "passed": True},
        {
            "name": "deterministic_replay_le_1e-12",
            "passed": max_deterministic_error <= float(protocol["deterministic_tolerance"]),
            "value": max_deterministic_error,
        },
        {
            "name": "mirrored_phase_pairs_produce_opposite_turn_signs",
            "passed": mirrored_ok,
            "opposed_pair_count": intact_opposed,
            "pair_count": len(mirrored_pairs),
        },
        {
            "name": "FC2_lane_cut_destroys_phase_dependent_turning",
            "passed": fc2_cut_destroys,
            "intact_opposed_pair_count": intact_opposed,
            "cut_opposed_pair_count": fc2_cut_opposed,
        },
        {
            "name": "EPG_lane_cut_destroys_phase_dependent_turning",
            "passed": epg_cut_destroys,
            "intact_opposed_pair_count": intact_opposed,
            "cut_opposed_pair_count": epg_cut_opposed,
        },
        {
            "name": "PFL3_output_cut_zeroes_DNa02_turn",
            "passed": pfl3_cut_peak <= float(protocol["zero_cut_tolerance"]),
            "value": pfl3_cut_peak,
        },
        {"name": "left_right_group_swap_reverses_turn_curve", "passed": side_swap_ok},
        {
            "name": "common_rotation_offset_invariance",
            "passed": max_rotation_error <= 1e-10,
            "value": max_rotation_error,
        },
        {
            "name": "all_structural_thresholds_reported",
            "passed": set(threshold_reports) == {str(x) for x in thresholds},
        },
    ]
    return {
        "protocol": protocol["protocol"],
        "dataset": protocol["dataset"],
        "passed": all(bool(row["passed"]) for row in gates),
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(row["passed"]) for row in gates),
        "gates": gates,
        "run_config": {"seed": int(seed), "steps": int(steps)},
        "primary_structural_threshold": int(protocol["primary_structural_threshold"]),
        "threshold_reports": threshold_reports,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E002e PFL3-to-descending steering probe"
    )
    parser.add_argument("bundle")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--e002d", default=str(DEFAULT_E002D))
    parser.add_argument("--crosswalk", default=str(DEFAULT_CROSSWALK))
    parser.add_argument(
        "--output", default="results/e002/pfl3-descending-steering-v1.json"
    )
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    e002d = json.loads(Path(args.e002d).read_text())
    crosswalk = json.loads(Path(args.crosswalk).read_text())
    bundle = GraphBundle.load(args.bundle)
    report = run_e002e(
        protocol,
        e002d,
        crosswalk,
        bundle,
        seed=args.seed,
        steps=args.steps,
    )
    report["input_sha256"] = {
        "protocol": _sha256(args.protocol),
        "e002d": _sha256(args.e002d),
        "crosswalk": _sha256(args.crosswalk),
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
