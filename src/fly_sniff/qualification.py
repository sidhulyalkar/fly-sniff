from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .env import Observation
from .graph import GraphBundle, MaleCNSRateController

REQUIRED_ROLES = ("odor_left", "odor_right", "steer_left", "steer_right")
WIND_ROLES = ("wind_forward", "wind_backward", "wind_left", "wind_right")


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    value: float | int | str | None
    criterion: str


@dataclass(frozen=True)
class ProbeResult:
    left_turn: float
    right_turn: float
    separation: float
    opposite_sign: bool
    laterality_correct: bool
    lesioned_peak_turn: float
    deterministic_error: float
    blank_retention: float


def _observation(left: float, right: float, *, wind_x: float = 1.0, wind_y: float = 0.0) -> Observation:
    return Observation(
        left_odor=float(left),
        right_odor=float(right),
        mean_odor=0.5 * (left + right),
        odor_delta=float(right - left),
        wind_x_body=float(wind_x),
        wind_y_body=float(wind_y),
        heading=0.0,
    )


def _rollout_turn(
    bundle: GraphBundle,
    sequence: list[Observation],
    *,
    seed: int,
) -> tuple[np.ndarray, MaleCNSRateController]:
    controller = MaleCNSRateController(bundle, require_qualified=False)
    controller.reset(seed)
    turns = np.asarray([controller.act(obs).turn for obs in sequence], dtype=float)
    return turns, controller


def _lesion_incoming(bundle: GraphBundle, roles: tuple[str, ...]) -> GraphBundle:
    lesioned = set()
    for role in roles:
        lesioned.update(int(x) for x in bundle.roles.get(role, []))
    edges = bundle.edges.loc[~bundle.edges.target.astype(int).isin(lesioned)].copy()
    manifest = dict(bundle.manifest or {})
    manifest["qualification_status"] = "candidate"
    manifest["lesion"] = {"incoming_to_roles": list(roles)}
    return GraphBundle(bundle.nodes.copy(), edges, dict(bundle.roles), manifest)


def _reachable(edges, starts: set[int]) -> set[int]:
    adjacency: dict[int, list[int]] = {}
    for row in edges[["source", "target", "weight"]].itertuples(index=False):
        if float(row.weight) <= 0:
            continue
        adjacency.setdefault(int(row.source), []).append(int(row.target))
    seen = {int(x) for x in starts}
    queue = deque(seen)
    while queue:
        node = queue.popleft()
        for nxt in adjacency.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def probe_candidate(
    bundle: GraphBundle,
    *,
    seed: int = 13013,
    pulse_steps: int = 32,
    blank_steps: int = 24,
) -> ProbeResult:
    """Run model-level E002 probes without claiming physiological validation.

    Core E002 asks whether modeled dynamics over the extracted topology produce
    bilateral discrimination, correct left/right steering semantics,
    deterministic behavior, and steering-output dependence. Blank persistence is
    reported as a *separate memory hypothesis* because recent work implicates
    persistent local-FB circuitry that need not be identical to the hDeltaC/PFL3
    steering pathway.
    """
    bundle.validate(require_sign=True, require_qualified=False)
    left_seq = [_observation(1.0, 0.0)] * pulse_steps
    right_seq = [_observation(0.0, 1.0)] * pulse_steps
    left_turns, left_ctl = _rollout_turn(bundle, left_seq, seed=seed)
    right_turns, _ = _rollout_turn(bundle, right_seq, seed=seed)
    left_turn = float(np.mean(left_turns[-max(4, pulse_steps // 4) :]))
    right_turn = float(np.mean(right_turns[-max(4, pulse_steps // 4) :]))
    laterality_correct = bool(left_turn > 0.0 and right_turn < 0.0)

    replay, _ = _rollout_turn(bundle, left_seq, seed=seed)
    deterministic_error = float(np.max(np.abs(left_turns - replay))) if len(replay) else 0.0

    lesioned = _lesion_incoming(bundle, ("steer_left", "steer_right"))
    lesioned_turns, _ = _rollout_turn(lesioned, left_seq, seed=seed)
    lesioned_peak = float(np.max(np.abs(lesioned_turns))) if len(lesioned_turns) else 0.0

    controller = left_ctl
    pulse_end = max(abs(float(left_turns[-1])), 1e-12)
    blank = _observation(0.0, 0.0)
    blank_turns = np.asarray([controller.act(blank).turn for _ in range(blank_steps)], dtype=float)
    late = float(np.mean(np.abs(blank_turns[-max(4, blank_steps // 4) :]))) if len(blank_turns) else 0.0
    retention = float(np.clip(late / pulse_end, 0.0, 10.0))

    return ProbeResult(
        left_turn=left_turn,
        right_turn=right_turn,
        separation=abs(right_turn - left_turn),
        opposite_sign=bool(left_turn * right_turn < 0.0),
        laterality_correct=laterality_correct,
        lesioned_peak_turn=lesioned_peak,
        deterministic_error=deterministic_error,
        blank_retention=retention,
    )


def qualify_candidate(
    bundle: GraphBundle,
    *,
    min_sign_fraction: float = 0.60,
    min_turn_separation: float = 0.05,
    max_lesioned_turn: float = 0.02,
    min_blank_retention: float = 0.05,
    seed: int = 13013,
) -> dict:
    """Return an auditable E002 report; never mutates a graph into qualified state."""
    bundle.validate(require_sign=True, require_qualified=False)
    ids = set(bundle.nodes.bodyId.astype(int))
    roles = {k: {int(x) for x in v} for k, v in bundle.roles.items()}
    missing_roles = [role for role in REQUIRED_ROLES if not roles.get(role)]
    steering_disjoint = roles.get("steer_left", set()).isdisjoint(roles.get("steer_right", set()))
    sensory_disjoint = roles.get("odor_left", set()).isdisjoint(roles.get("odor_right", set()))

    signed_fraction = float(bundle.edges.sign.astype(int).ne(0).mean()) if len(bundle.edges) else 0.0
    sensory = roles.get("odor_left", set()) | roles.get("odor_right", set())
    wind = set().union(*(roles.get(role, set()) for role in WIND_ROLES))
    reached = _reachable(bundle.edges, sensory | wind)
    steer_left_reached = bool(roles.get("steer_left", set()) & reached)
    steer_right_reached = bool(roles.get("steer_right", set()) & reached)

    probe = probe_candidate(bundle, seed=seed)
    core_gates = [
        Gate("required_roles", not missing_roles, ",".join(missing_roles) if missing_roles else "complete", "all bilateral odor and steering roles are non-empty"),
        Gate("role_disjointness", steering_disjoint and sensory_disjoint, int(steering_disjoint and sensory_disjoint), "left/right odor and steering role sets do not overlap"),
        Gate("body_id_closure", all(x in ids for values in roles.values() for x in values), len(ids), "all role body IDs exist in nodes.parquet"),
        Gate("signed_edge_fraction", signed_fraction >= min_sign_fraction, signed_fraction, f">= {min_sign_fraction:.2f}"),
        Gate("structural_reachability_left", steer_left_reached, int(steer_left_reached), "sensory/wind seeds structurally reach steer_left"),
        Gate("structural_reachability_right", steer_right_reached, int(steer_right_reached), "sensory/wind seeds structurally reach steer_right"),
        Gate("bilateral_turn_separation", probe.separation >= min_turn_separation, probe.separation, f">= {min_turn_separation:.3f}"),
        Gate("bilateral_turn_opposition", probe.opposite_sign, int(probe.opposite_sign), "mirrored odor perturbations produce opposite steering signs"),
        Gate(
            "bilateral_turn_laterality",
            probe.laterality_correct,
            int(probe.laterality_correct),
            "left-only odor produces positive/left turn and right-only odor negative/right turn",
        ),
        Gate("steering_lesion", probe.lesioned_peak_turn <= max_lesioned_turn, probe.lesioned_peak_turn, f"<= {max_lesioned_turn:.3f} after bilateral steering-input lesion"),
        Gate("determinism", probe.deterministic_error <= 1e-12, probe.deterministic_error, "exact-seed replay max error <= 1e-12"),
    ]
    memory_gate = Gate(
        "blank_bridge_memory_hypothesis",
        probe.blank_retention >= min_blank_retention,
        probe.blank_retention,
        f">= {min_blank_retention:.3f} late-blank / pulse-end modeled turn magnitude",
    )
    core_passed = all(gate.passed for gate in core_gates)
    all_gates = core_gates + [memory_gate]
    return {
        "protocol": "E002-circuit-sanity-v1",
        "dataset": (bundle.manifest or {}).get("dataset", "unknown"),
        "passed": core_passed,
        "core_passed": core_passed,
        "memory_hypothesis_passed": memory_gate.passed,
        "core_gate_count": len(core_gates),
        "core_passed_gate_count": sum(g.passed for g in core_gates),
        "gate_count": len(all_gates),
        "passed_gate_count": sum(g.passed for g in all_gates),
        "gates": [asdict(g) for g in all_gates],
        "probe": asdict(probe),
        "coordinate_convention": {
            "positive_turn": "left/counterclockwise",
            "negative_turn": "right/clockwise",
            "role_semantics": "steer_left and steer_right are ipsilateral steering roles",
        },
        "warning": (
            "E002 core passing qualifies modeled propagation sanity only. The blank-bridge result is a "
            "separate memory hypothesis and is not required for steering qualification. Neither result "
            "validates physiological dynamics, receptor kinetics, or sensory-pathway identity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E002 sanity gates on a candidate MaleCNS graph bundle")
    parser.add_argument("bundle")
    parser.add_argument("--output", default="results/e002/qualification.json")
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--min-sign-fraction", type=float, default=0.60)
    parser.add_argument("--min-turn-separation", type=float, default=0.05)
    parser.add_argument("--min-blank-retention", type=float, default=0.05)
    args = parser.parse_args()
    bundle = GraphBundle.load(args.bundle)
    report = qualify_candidate(
        bundle,
        min_sign_fraction=args.min_sign_fraction,
        min_turn_separation=args.min_turn_separation,
        min_blank_retention=args.min_blank_retention,
        seed=args.seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["core_passed"]:
        raise SystemExit(2)
