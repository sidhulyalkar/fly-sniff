from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL = "E002e-pfl3-descending-steering-v2"
QUALIFICATION_PROTOCOL = "E002e-qualification-seal-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_dirty_paths() -> list[str]:
    text = _git_output("status", "--porcelain")
    return text.splitlines() if text else []


def _opposed_pairs(phase_reports: dict[str, Any], pairs: list[list[int]]) -> int:
    return sum(
        float(phase_reports[str(int(negative))]["turn"])
        * float(phase_reports[str(int(positive))]["turn"])
        < 0.0
        for negative, positive in pairs
    )


def qualify_e002e_v2(
    result: dict[str, Any],
    protocol: dict[str, Any],
    *,
    actual_input_sha256: dict[str, str],
    result_sha256: str,
    expected_git_sha: str | None,
    git_branch: str | None,
    git_sha: str | None,
    git_dirty_paths: list[str] | None,
) -> dict[str, Any]:
    if result.get("protocol") != PROTOCOL:
        raise ValueError("unexpected E002e v2 result protocol")
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("unexpected E002e v2 protocol config")

    gate_names = [str(row["name"]) for row in result.get("gates", [])]
    expected_gates = [str(x) for x in protocol["qualification_gates"]]
    thresholds = {str(x) for x in protocol["all_structural_thresholds"]}
    observed_thresholds = set(result.get("threshold_reports", {}))
    primary = str(int(protocol["primary_structural_threshold"]))
    pairs = [[int(a), int(b)] for a, b in protocol["mirrored_phase_pairs_deg"]]
    primary_reports = result["threshold_reports"][primary]["phase_reports"]
    high_reports = result["threshold_reports"]["10"]["phase_reports"]

    recorded_inputs = dict(result.get("input_sha256", {}))
    input_hashes_match = all(
        recorded_inputs.get(name) == digest for name, digest in actual_input_sha256.items()
    )

    side_swap_gate = next(
        row
        for row in result["gates"]
        if row["name"] == "left_right_group_swap_reverses_nonzero_mirrored_phases"
    )
    zero_phase = dict(side_swap_gate.get("zero_phase") or {})
    dirty = list(git_dirty_paths or [])
    clean_runtime = (
        not dirty
        and bool(git_sha)
        and (expected_git_sha is None or git_sha == expected_git_sha)
    )

    checks = [
        {
            "name": "confirmation_science_gates_pass",
            "passed": bool(result.get("passed"))
            and int(result.get("passed_gate_count", -1)) == 10
            and int(result.get("gate_count", -1)) == 10,
        },
        {
            "name": "gate_set_matches_protocol",
            "passed": gate_names == expected_gates,
        },
        {
            "name": "frozen_confirmation_runtime_matches",
            "passed": int(result["run_config"]["seed"]) == int(protocol["confirmation_seed"])
            and int(result["run_config"]["steps"]) == 32
            and bool(result["run_config"].get("confirmation_seed_frozen")),
        },
        {
            "name": "primary_threshold_remains_5",
            "passed": int(result["primary_structural_threshold"]) == 5
            and int(protocol["primary_structural_threshold"]) == 5,
        },
        {
            "name": "all_threshold_sensitivities_present",
            "passed": observed_thresholds == thresholds,
        },
        {
            "name": "input_hashes_match_local_authorities",
            "passed": input_hashes_match,
        },
        {
            "name": "side_swap_confirmation_complete",
            "passed": bool(side_swap_gate.get("passed"))
            and int(side_swap_gate.get("reversed_phase_count", -1)) == 10
            and int(side_swap_gate.get("required_phase_count", -2)) == 10
            and zero_phase.get("gate_role") == "descriptive_baseline_only",
        },
        {
            "name": "clean_exact_git_runtime",
            "passed": clean_runtime,
        },
    ]

    primary_opposed = _opposed_pairs(primary_reports, pairs)
    high_opposed = _opposed_pairs(high_reports, pairs)
    high_dual = int(result["threshold_reports"]["10"]["dual_reachable_count"])
    robustness_status = (
        "primary-threshold-qualified-high-threshold-fragile"
        if high_opposed < len(pairs)
        else "primary-and-high-threshold-consistent"
    )

    return {
        "protocol": QUALIFICATION_PROTOCOL,
        "dataset": result.get("dataset", "unknown"),
        "qualification_ready": all(bool(row["passed"]) for row in checks),
        "source_result": {
            "protocol": result["protocol"],
            "sha256": result_sha256,
            "passed": bool(result.get("passed")),
            "passed_gate_count": int(result.get("passed_gate_count", 0)),
            "gate_count": int(result.get("gate_count", 0)),
        },
        "checks": checks,
        "robustness": {
            "primary_threshold": 5,
            "primary_mirrored_opposed_pairs": primary_opposed,
            "primary_pair_count": len(pairs),
            "threshold_10_dual_reachable_count": high_dual,
            "threshold_10_mirrored_opposed_pairs": high_opposed,
            "threshold_10_pair_count": len(pairs),
            "status": robustness_status,
            "interpretation": (
                "Qualification is tied to the frozen primary threshold 5. Threshold 10 is a "
                "reported sensitivity and is not used to retune or promote the result."
            ),
        },
        "runtime": {
            "git_branch": git_branch,
            "git_sha": git_sha,
            "git_dirty_paths": dirty,
            "expected_git_sha": expected_git_sha,
        },
        "claim_boundary": (
            "A qualified E002e-v2 seal supports a deterministic modeled PFL3-to-descending "
            "steering transformation at the frozen primary structural threshold 5. It does not "
            "establish measured firing, sensory-to-goal encoding, closed-loop odor navigation, "
            "or intact-over-rewire superiority. High-threshold fragility remains explicit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal an E002e-v2 confirmation run")
    parser.add_argument("result")
    parser.add_argument("--protocol", default="configs/e002e_descending_steering_protocol_v2.json")
    parser.add_argument("--e002d", default="results/e002/pfl3-phase-comparison-v1.json")
    parser.add_argument("--crosswalk", default="results/e002/pfl3-phase-crosswalk-v1.json")
    parser.add_argument("--bundle", default="data/cache/steering-scaffold-v1")
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--output", default="results/e002/pfl3-descending-steering-qualified-v1.json")
    args = parser.parse_args()

    result_path = Path(args.result)
    protocol_path = Path(args.protocol)
    bundle = Path(args.bundle)
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    actual_inputs = {
        "protocol": _sha256(protocol_path),
        "e002d": _sha256(args.e002d),
        "crosswalk": _sha256(args.crosswalk),
        "bundle_manifest": _sha256(bundle / "manifest.json"),
        "bundle_edges": _sha256(bundle / "edges.parquet"),
    }
    report = qualify_e002e_v2(
        result,
        protocol,
        actual_input_sha256=actual_inputs,
        result_sha256=_sha256(result_path),
        expected_git_sha=args.expected_git_sha,
        git_branch=_git_output("branch", "--show-current"),
        git_sha=_git_output("rev-parse", "HEAD"),
        git_dirty_paths=_git_dirty_paths(),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["qualification_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
