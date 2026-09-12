from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .graph import GraphBundle, MaleCNSRateController


@dataclass(frozen=True)
class SteeringProbeResult:
    left_turn: float
    right_turn: float
    separation: float
    opposite_sign: bool
    laterality_correct: bool
    deterministic_error: float
    pfl3_cut_peak_turn: float


def _run_role(
    bundle: GraphBundle,
    role: str,
    *,
    seed: int,
    steps: int,
) -> np.ndarray:
    controller = MaleCNSRateController(bundle, require_qualified=False)
    controller.reset(seed)
    return np.asarray(
        [controller.act_role_drive({role: 1.0}).turn for _ in range(steps)],
        dtype=float,
    )


def _without_edge_families(bundle: GraphBundle, families: set[str]) -> GraphBundle:
    if "edge_family" not in bundle.edges.columns:
        raise ValueError("steering scaffold edges must carry edge_family provenance")
    edges = bundle.edges.loc[~bundle.edges.edge_family.isin(families)].copy()
    manifest = dict(bundle.manifest or {})
    manifest["qualification_status"] = "candidate"
    manifest["mechanistic_lesion"] = {"removed_edge_families": sorted(families)}
    return GraphBundle(bundle.nodes.copy(), edges, dict(bundle.roles), manifest)


def _cut_pfl3_output(bundle: GraphBundle) -> GraphBundle:
    drive_ids = set(int(x) for x in bundle.roles.get("turn_drive_left", []))
    drive_ids.update(int(x) for x in bundle.roles.get("turn_drive_right", []))
    edges = bundle.edges.loc[~bundle.edges.source.astype(int).isin(drive_ids)].copy()
    manifest = dict(bundle.manifest or {})
    manifest["qualification_status"] = "candidate"
    manifest["mechanistic_lesion"] = {"removed": "all scaffold output from PFL3 drive roles"}
    return GraphBundle(bundle.nodes.copy(), edges, dict(bundle.roles), manifest)


def probe_steering_scaffold(
    bundle: GraphBundle,
    *,
    seed: int = 13013,
    steps: int = 32,
) -> dict:
    """Probe modeled propagation inside the restricted steering scaffold.

    This is intentionally *not* the odor-navigation E002 assay. PFL3 drive is
    injected directly through dedicated mechanistic-probe roles so the result
    cannot be misread as evidence that PFL3 is a sensory population.
    """
    if steps < 4:
        raise ValueError("steps must be >= 4")
    bundle.validate(require_sign=True, require_qualified=False)

    required_roles = ("turn_drive_left", "turn_drive_right", "steer_left", "steer_right")
    missing_roles = [role for role in required_roles if not bundle.roles.get(role)]
    if missing_roles:
        raise ValueError(f"steering scaffold missing required roles: {missing_roles}")

    tail = max(4, steps // 4)
    left = _run_role(bundle, "turn_drive_left", seed=seed, steps=steps)
    right = _run_role(bundle, "turn_drive_right", seed=seed, steps=steps)
    replay = _run_role(bundle, "turn_drive_left", seed=seed, steps=steps)
    left_turn = float(np.mean(left[-tail:]))
    right_turn = float(np.mean(right[-tail:]))
    deterministic_error = float(np.max(np.abs(left - replay)))

    cut = _cut_pfl3_output(bundle)
    cut_left = _run_role(cut, "turn_drive_left", seed=seed, steps=steps)
    cut_right = _run_role(cut, "turn_drive_right", seed=seed, steps=steps)
    cut_peak = float(max(np.max(np.abs(cut_left)), np.max(np.abs(cut_right))))

    probe = SteeringProbeResult(
        left_turn=left_turn,
        right_turn=right_turn,
        separation=abs(left_turn - right_turn),
        opposite_sign=bool(left_turn * right_turn < 0.0),
        laterality_correct=bool(left_turn > 0.0 and right_turn < 0.0),
        deterministic_error=deterministic_error,
        pfl3_cut_peak_turn=cut_peak,
    )

    signed_fraction = float(bundle.edges.sign.astype(int).ne(0).mean()) if len(bundle.edges) else 0.0
    gates = [
        {
            "name": "candidate_not_promoted",
            "passed": (bundle.manifest or {}).get("qualification_status") == "candidate",
            "criterion": "input scaffold remains qualification_status='candidate'",
        },
        {
            "name": "no_sensory_role_aliasing",
            "passed": not any(role in bundle.roles for role in ("odor_left", "odor_right")),
            "criterion": "restricted steering scaffold does not alias PFL3 drive as odor roles",
        },
        {
            "name": "explicit_signed_edges",
            "passed": signed_fraction == 1.0,
            "value": signed_fraction,
            "criterion": "all included scaffold edges have explicit non-zero modeled sign",
        },
        {
            "name": "modeled_left_right_laterality",
            "passed": probe.laterality_correct,
            "criterion": "left-drive -> positive/left turn and right-drive -> negative/right turn",
        },
        {
            "name": "modeled_left_right_opposition",
            "passed": probe.opposite_sign,
            "criterion": "mirrored PFL3 drive produces opposite modeled turn signs",
        },
        {
            "name": "deterministic_role_drive",
            "passed": deterministic_error <= 1e-12,
            "value": deterministic_error,
            "criterion": "same seed and drive sequence replay with max error <= 1e-12",
        },
        {
            "name": "pfl3_output_cut",
            "passed": cut_peak <= 1e-12,
            "value": cut_peak,
            "criterion": "removing all PFL3 scaffold outputs abolishes modeled DNa02 turning",
        },
    ]

    lesion_reports: dict[str, dict[str, float]] = {}
    lesion_sets = {
        "no_direct_PFL3_DNa02": {"PFL3->DNa02"},
        "no_DNa03_relay": {"PFL3->DNa03", "DNa03->DNa02"},
        "no_LAL010_relay": {"PFL3->LAL010", "LAL010->DNa02"},
        "direct_only": {
            "PFL3->DNa03",
            "DNa03->DNa02",
            "PFL3->LAL010",
            "LAL010->DNa02",
        },
    }
    for name, removed in lesion_sets.items():
        lesioned = _without_edge_families(bundle, removed)
        lturn = _run_role(lesioned, "turn_drive_left", seed=seed, steps=steps)
        rturn = _run_role(lesioned, "turn_drive_right", seed=seed, steps=steps)
        lesion_reports[name] = {
            "left_turn": float(np.mean(lturn[-tail:])),
            "right_turn": float(np.mean(rturn[-tail:])),
            "removed_edge_family_count": float(len(removed)),
        }

    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "protocol": "E002a-steering-scaffold-propagation-v1",
        "dataset": (bundle.manifest or {}).get("dataset", "unknown"),
        "passed": passed,
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(gate["passed"]) for gate in gates),
        "gates": gates,
        "probe": asdict(probe),
        "descriptive_lesions": lesion_reports,
        "claim_boundary": (
            "Passing demonstrates deterministic modeled propagation and laterality within the restricted "
            "body-ID-resolved steering scaffold only. It does not qualify odor sensing, hDeltaC-to-PFL3 "
            "transmission, the inhibitory see-saw arm, physiological firing, or behavioral performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated E002a modeled-propagation probe on a steering scaffold"
    )
    parser.add_argument("bundle")
    parser.add_argument("--output", default="results/e002/steering-scaffold-v1.json")
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    bundle = GraphBundle.load(args.bundle)
    report = probe_steering_scaffold(bundle, seed=args.seed, steps=args.steps)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
