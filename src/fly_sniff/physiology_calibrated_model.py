from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .physiology_calibration import validate_protocol

PROTOCOL = "physiology-calibrated-model-v1"
FIT_REPORT_PROTOCOL = "physiology-fit-report-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_calibrated_model_artifact(
    calibration_config: dict[str, Any],
    binding_report: dict[str, Any],
    fit_report: dict[str, Any],
    *,
    calibration_config_sha256: str,
    binding_report_sha256: str,
    fit_report_sha256: str,
) -> dict[str, Any]:
    preregistration = validate_protocol(calibration_config)
    if not preregistration["valid_for_preregistration"]:
        raise ValueError("calibration config does not pass preregistration gates")
    if binding_report.get("protocol") != "physiology-probe-binding-v1":
        raise ValueError("unexpected physiology binding protocol")
    if binding_report.get("ready_to_define_fit_objective") is not True:
        unresolved = binding_report.get("unresolved_probes", [])
        raise ValueError(f"cannot seal calibrated model with unresolved probe authorities: {unresolved}")
    if fit_report.get("protocol") != FIT_REPORT_PROTOCOL:
        raise ValueError("unexpected physiology fit report protocol")
    if fit_report.get("calibration_protocol") != calibration_config["protocol"]:
        raise ValueError("fit report calibration protocol mismatch")
    if fit_report.get("fit_status") != "accepted_under_preregistered_objectives":
        raise ValueError("fit report is not accepted under preregistered objectives")
    if float(fit_report.get("navigation_objective_weight", float("nan"))) != 0.0:
        raise ValueError("navigation objective must have exactly zero weight in physiology calibration")
    if fit_report.get("topology_variant_visible_during_fit") is not False:
        raise ValueError("topology variant must remain hidden during physiology calibration")
    if fit_report.get("final_evaluation_visible_during_fit") is not False:
        raise ValueError("final evaluation must remain hidden during physiology calibration")

    permitted = set(str(x) for x in calibration_config["permitted_parameters"])
    parameters = {str(key): float(value) for key, value in fit_report["parameters"].items()}
    if set(parameters) != permitted:
        raise ValueError(
            f"calibrated parameter set must exactly match preregistered permitted parameters: "
            f"expected={sorted(permitted)} observed={sorted(parameters)}"
        )
    probe_ids = [str(row["id"]) for row in calibration_config["probes"]]
    probe_results = dict(fit_report["probe_results"])
    if set(probe_results) != set(probe_ids):
        raise ValueError("fit report must retain every preregistered physiology probe")
    if not all(bool(probe_results[probe_id].get("objective_defined_before_fit")) for probe_id in probe_ids):
        raise ValueError("every physiology objective must be marked as defined before fitting")

    return {
        "protocol": PROTOCOL,
        "dataset": calibration_config["dataset"],
        "calibration_protocol": calibration_config["protocol"],
        "fit_status": fit_report["fit_status"],
        "parameters": parameters,
        "probe_results": probe_results,
        "parameter_sharing": calibration_config["parameter_sharing"],
        "forbidden_calibration_signals": calibration_config["forbidden_calibration_signals"],
        "input_sha256": {
            "calibration_config": calibration_config_sha256,
            "probe_bindings": binding_report_sha256,
            "fit_report": fit_report_sha256,
        },
        "claim_boundary": (
            "This artifact freezes one model parameterization accepted under preregistered, "
            "independent physiology objectives. It is not measured MaleCNS activity and does not "
            "establish navigation performance or topology dependence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal a physiology-calibrated dynamics artifact")
    parser.add_argument("--config", default="configs/physiology_calibration_v1.json")
    parser.add_argument("--bindings", default="authority/physiology-probe-bindings-v1.json")
    parser.add_argument("--fit-report", required=True)
    parser.add_argument("--output", default="results/calibration/physiology-calibrated-model-v1.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    bindings = json.loads(Path(args.bindings).read_text())
    fit_report = json.loads(Path(args.fit_report).read_text())
    artifact = build_calibrated_model_artifact(
        config,
        bindings,
        fit_report,
        calibration_config_sha256=file_sha256(args.config),
        binding_report_sha256=file_sha256(args.bindings),
        fit_report_sha256=file_sha256(args.fit_report),
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibrated model artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"{output}")
    print("status=sealed physiology-calibrated-model-v1")


if __name__ == "__main__":
    main()
