from __future__ import annotations

import argparse
import hashlib
import json
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


def _git_dirty_paths() -> list[str]:
    text = _git_output("status", "--porcelain")
    return text.splitlines() if text else []


def qualify_e002b_result(
    result: dict[str, Any],
    runtime_config: dict[str, Any],
    *,
    source_result_sha256: str,
    expected_git_sha: str | None = None,
    git_branch: str | None = None,
    git_sha: str | None = None,
    git_dirty_paths: list[str] | None = None,
) -> dict[str, Any]:
    if result.get("protocol") != "E002b-goal-channel-propagation-v1":
        raise ValueError("unexpected E002b result protocol")

    expected_thresholds = [float(x) for x in runtime_config["structural_thresholds"]]
    observed_thresholds = [float(x) for x in result["run_config"]["thresholds"]]
    runtime_matches = (
        int(result["run_config"]["seed"]) == int(runtime_config["seed"])
        and int(result["run_config"]["steps"]) == int(runtime_config["steps"])
        and int(result["run_config"]["pulse_steps"]) == int(runtime_config["pulse_steps"])
        and float(result["run_config"]["drive_amplitude"])
        == float(runtime_config["drive_amplitude"])
        and observed_thresholds == expected_thresholds
        and bool(runtime_config.get("frozen_before_first_real_probe"))
    )

    comparator_complete = True
    comparator_frozen = True
    for threshold in expected_thresholds:
        key = str(int(threshold) if threshold.is_integer() else threshold)
        comparator = result["threshold_reports"][key]["hDeltaM_relay_comparator"]
        comparator_complete &= comparator.get("status") == "modeled"
        comparator_frozen &= (
            comparator.get("selection_status")
            == "frozen_comparator_not_eligible_for_promotion"
        )

    sensitivity_present = all(
        "PFNp_b_sign_zero_sensitivity" in result["threshold_reports"][
            str(int(t) if t.is_integer() else t)
        ]
        for t in expected_thresholds
    )

    dirty = list(git_dirty_paths or [])
    sha_matches = expected_git_sha is None or git_sha == expected_git_sha
    clean_runtime = not dirty and bool(git_sha) and sha_matches

    checks = [
        {
            "name": "preregistered_science_gates_pass",
            "passed": bool(result.get("passed"))
            and int(result.get("passed_gate_count", -1)) == int(result.get("gate_count", -2))
            and int(result.get("gate_count", -1)) == 7,
        },
        {"name": "frozen_runtime_matches", "passed": runtime_matches},
        {"name": "hDeltaM_comparator_complete", "passed": comparator_complete},
        {"name": "hDeltaM_selection_remains_frozen", "passed": comparator_frozen},
        {"name": "PFNp_b_sensitivity_present", "passed": sensitivity_present},
        {"name": "clean_exact_git_runtime", "passed": clean_runtime},
    ]
    qualification_ready = all(bool(row["passed"]) for row in checks)

    return {
        "protocol": "E002b-qualification-seal-v1",
        "dataset": result.get("dataset", "unknown"),
        "qualification_ready": qualification_ready,
        "source_result": {
            "protocol": result["protocol"],
            "sha256": source_result_sha256,
            "passed": bool(result.get("passed")),
            "passed_gate_count": int(result.get("passed_gate_count", 0)),
            "gate_count": int(result.get("gate_count", 0)),
        },
        "runtime": {
            "git_branch": git_branch,
            "git_sha": git_sha,
            "git_dirty_paths": dirty,
            "expected_git_sha": expected_git_sha,
        },
        "checks": checks,
        "claim_boundary": (
            "A qualified E002b seal establishes deterministic modeled propagation through the "
            "restricted signed goal-channel candidate under the preregistered gating abstraction. "
            "It does not establish sensory tuning, heading comparison, physiological firing, "
            "steering direction, source finding, or intact-versus-rewire behavioral superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal an E002b result to a clean exact git runtime")
    parser.add_argument("result")
    parser.add_argument(
        "--runtime-config", default="configs/e002b_probe_runtime_v1.json"
    )
    parser.add_argument("--expected-git-sha")
    parser.add_argument(
        "--output", default="results/e002/goal-channel-qualified-v1.json"
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    runtime_path = Path(args.runtime_config)
    result = json.loads(result_path.read_text())
    runtime_config = json.loads(runtime_path.read_text())

    report = qualify_e002b_result(
        result,
        runtime_config,
        source_result_sha256=_sha256_file(result_path),
        expected_git_sha=args.expected_git_sha,
        git_branch=_git_output("branch", "--show-current"),
        git_sha=_git_output("rev-parse", "HEAD"),
        git_dirty_paths=_git_dirty_paths(),
    )
    report["inputs"] = {
        "result": {"path": str(result_path), "sha256": _sha256_file(result_path)},
        "runtime_config": {
            "path": str(runtime_path),
            "sha256": _sha256_file(runtime_path),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["qualification_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
