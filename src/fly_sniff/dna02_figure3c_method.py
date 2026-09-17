from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .freeze import canonical_sha256

DEFAULT_CONTRACT = Path("authority/program-a-dna02-figure3c-method-contract-v1.json")
DEFAULT_HASH = Path("authority/program-a-dna02-figure3c-method-contract-v1.sha256")

_EXPECTED_SCHEMA = "fly-sniff-dna02-figure3c-method-contract-v1"
_EXPECTED_STATUS = "PARAMETERS_RESOLVED_EDGE_SEMANTICS_PENDING"


def load_method_contract(
    path: str | Path = DEFAULT_CONTRACT,
    hash_path: str | Path = DEFAULT_HASH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Figure 3C method contract must be a JSON object")

    expected_hash = Path(hash_path).read_text(encoding="utf-8").strip()
    observed_hash = canonical_sha256(payload)
    if observed_hash != expected_hash:
        raise ValueError("Figure 3C method contract canonical hash does not match")
    if payload.get("schema") != _EXPECTED_SCHEMA:
        raise ValueError("unsupported Figure 3C method-contract schema")
    if payload.get("status") != _EXPECTED_STATUS:
        raise ValueError("Figure 3C method-contract status has drifted")

    published = payload.get("version_of_record")
    if not isinstance(published, dict):
        raise TypeError("version_of_record must be an object")
    expected_published = {
        "doi": "10.7554/eLife.102230.3",
        "figure": "Figure 3B-C",
        "spike_detection": "relative-prominence peak detection using MATLAB findpeaks",
        "spike_bin_ms": 10,
        "smoothing_function": "MATLAB smoothts",
        "smoothing_method": "exponential",
        "smoothing_window_ms": 30,
        "neural_to_behavior_alignment_ms": 150,
        "figure_average_bin_ms": 50,
        "primary_predictor": "right_firing_rate_hz - left_firing_rate_hz",
        "primary_outcome": "rotational_velocity_deg_per_s",
    }
    for key, value in expected_published.items():
        if published.get(key) != value:
            raise ValueError(f"Figure 3C published method drifted at {key}")

    smoothing = payload.get("resolved_smoothing_parameters")
    if not isinstance(smoothing, dict):
        raise TypeError("resolved_smoothing_parameters must be an object")
    if smoothing.get("period_bins") != 3:
        raise ValueError("30 ms smoothing over 10 ms bins must remain period length 3")
    if smoothing.get("alpha") != 0.5:
        raise ValueError("smoothts period length 3 must remain alpha 0.5")
    expected_alpha = 2.0 / (float(smoothing["period_bins"]) + 1.0)
    if float(smoothing["alpha"]) != expected_alpha:
        raise ValueError("smoothts alpha no longer matches 2/(period+1)")

    historical = payload.get("historical_matlab_context")
    if not isinstance(historical, dict):
        raise TypeError("historical_matlab_context must be an object")
    if historical.get("repository") != "SashaRayshubskiy/eLife_102230_analysis_code":
        raise ValueError("historical MATLAB repository authority has drifted")
    if historical.get("commit") != "55e30c19b1a18f601df1803295f0c401aec3c167":
        raise ValueError("historical MATLAB commit authority has drifted")
    if historical.get("path") != "calculate_psth_A2.m":
        raise ValueError("historical DNa02 smoothing helper has drifted")

    unresolved = payload.get("unresolved_before_numeric_extraction")
    if not isinstance(unresolved, list) or len(unresolved) < 4:
        raise ValueError("numeric extraction blockers must remain explicit")

    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise TypeError("gates must be an object")
    if gates.get("numeric_smoothing_implementation_allowed") is not False:
        raise ValueError("numeric smoothing must remain blocked before conformance evidence")
    if gates.get("figure3c_behavior_allowed") is not False:
        raise ValueError("Figure 3C behavior must remain sealed before method conformance")
    if gates.get("thresholds_must_be_frozen_first") is not True:
        raise ValueError("threshold freeze must remain a prerequisite")
    if gates.get("navigation_performance_used") is not False:
        raise ValueError("navigation performance may not resolve Figure 3C preprocessing")
    return payload


def method_status(
    path: str | Path = DEFAULT_CONTRACT,
    hash_path: str | Path = DEFAULT_HASH,
) -> dict[str, Any]:
    payload = load_method_contract(path, hash_path)
    return {
        "schema": payload["schema"],
        "contract_id": payload["contract_id"],
        "contract_sha256": canonical_sha256(payload),
        "status": payload["status"],
        "resolved_smoothing_parameters": payload["resolved_smoothing_parameters"],
        "unresolved_before_numeric_extraction": payload["unresolved_before_numeric_extraction"],
        "gates": payload["gates"],
        "allowed_next_action": payload["allowed_next_action"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen DNa02 Figure 3C preprocessing contract without opening behavior"
        )
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--hash", dest="hash_path", default=str(DEFAULT_HASH))
    args = parser.parse_args(argv)
    print(
        json.dumps(
            method_status(args.contract, args.hash_path),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
