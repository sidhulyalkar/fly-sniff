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


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def build_e002c_input_seal(paths: dict[str, str | Path]) -> dict[str, Any]:
    loaded = {name: _load(path) for name, path in paths.items()}
    qualified = loaded["e002b_qualification"]
    heading = loaded["heading_route"]
    heading_topography = loaded["heading_topography"]
    heading_authority = loaded["heading_route_authority"]
    heading_sign = loaded["heading_sign_authority"]
    protocol = loaded["protocol"]
    runtime = loaded["runtime"]

    if not bool(qualified.get("qualification_ready")):
        raise ValueError("E002c requires qualification_ready=true from E002b")
    if heading.get("protocol") != "malecns-heading-route-audit-v1":
        raise ValueError("unexpected heading route protocol")
    if heading_topography.get("protocol") != "malecns-heading-topography-review-v1":
        raise ValueError("unexpected heading topography protocol")
    if protocol.get("protocol") != "E002c-pfl3-goal-heading-convergence-v1":
        raise ValueError("unexpected E002c protocol")
    if runtime.get("protocol") != protocol.get("protocol"):
        raise ValueError("E002c runtime protocol mismatch")
    if not bool(runtime.get("frozen_before_first_real_probe")):
        raise ValueError("E002c runtime is not frozen before first real probe")
    if runtime.get("structural_thresholds") != protocol.get("fixed_structural_thresholds"):
        raise ValueError("E002c runtime thresholds drifted from preregistration")

    heading_sha = _sha256_file(paths["heading_route"])
    expected_heading_sha = str(heading_authority["source_artifact"]["sha256"])
    if heading_sha != expected_heading_sha:
        raise ValueError("heading route artifact hash does not match sealed authority")
    topo_source_sha = str(heading_topography["source_artifact"]["sha256"])
    if topo_source_sha != heading_sha:
        raise ValueError("heading topography review was not derived from the sealed heading audit")

    epg = heading_sign["predictions"]["EPG"]
    delta7 = heading_sign["predictions"]["Delta7"]
    if int(epg["modeled_sign"]) != 1:
        raise ValueError("E002c primary EPG sign must remain +1")
    if int(delta7["modeled_sign"]) != 0:
        raise ValueError("Delta7 must remain sign-unresolved for E002c")

    dirty = (_git_output("status", "--porcelain") or "").splitlines()
    if dirty:
        raise ValueError(f"E002c input seal requires a clean worktree: {dirty[:5]}")

    return {
        "protocol": "E002c-input-seal-v1",
        "dataset": "male-cns:v1.0",
        "git_runtime": {
            "git_branch": _git_output("branch", "--show-current"),
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_dirty_paths": [],
        },
        "files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "semantic_checks": {
            "e002b_qualified": True,
            "heading_route_matches_authority": True,
            "heading_topography_matches_heading_route": True,
            "EPG_modeled_sign": 1,
            "Delta7_modeled_sign": 0,
            "runtime_frozen": True,
        },
        "claim_boundary": (
            "This seal binds E002c inputs before the convergence probe. It does not qualify "
            "heading tuning, goal-heading comparison, steering, or behavior."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal all E002c inputs before the first run")
    parser.add_argument("--e002b-qualification", required=True)
    parser.add_argument("--heading-route", required=True)
    parser.add_argument("--heading-topography", required=True)
    parser.add_argument("--integration-route", required=True)
    parser.add_argument("--integration-sign", required=True)
    parser.add_argument("--integration-topography", required=True)
    parser.add_argument(
        "--heading-route-authority",
        default="authority/malecns-v1.0-heading-route-evidence.json",
    )
    parser.add_argument(
        "--heading-sign-authority",
        default="authority/malecns-v1.0-heading-transmitter-evidence.json",
    )
    parser.add_argument(
        "--protocol", default="configs/e002c_pfl3_convergence_protocol_v1.json"
    )
    parser.add_argument("--runtime", default="configs/e002c_probe_runtime_v1.json")
    parser.add_argument("--output", default="results/e002/e002c-input-seal-v1.json")
    args = parser.parse_args()

    paths = {
        "e002b_qualification": args.e002b_qualification,
        "heading_route": args.heading_route,
        "heading_topography": args.heading_topography,
        "integration_route": args.integration_route,
        "integration_sign": args.integration_sign,
        "integration_topography": args.integration_topography,
        "heading_route_authority": args.heading_route_authority,
        "heading_sign_authority": args.heading_sign_authority,
        "protocol": args.protocol,
        "runtime": args.runtime,
    }
    report = build_e002c_input_seal(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
