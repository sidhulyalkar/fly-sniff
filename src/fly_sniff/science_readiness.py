from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .connectome_necessity import validate_protocol as validate_connectome_necessity
from .evidence import EvidenceLedger
from .experiment_protocol import ExperimentSpec
from .null_factory_v2 import validate_protocol as validate_null_factory
from .physiology_calibration import validate_protocol as validate_physiology
from .physiology_probe_binding import bind_probe_authorities
from .zero_shot_artifacts import validate_zero_shot_artifact
from .zero_shot_experiment import validate_zero_shot_spec

PROTOCOL = "fly-sniff-science-readiness-v1"

DEFAULT_ARTIFACTS = {
    "evidence_ledger": Path("authority/evidence-ledger-v1.json"),
    "physiology_calibrated_model": Path("results/calibration/physiology-calibrated-model-v1.json"),
    "null_factory_protocol": Path("configs/null_factory_v2.json"),
    "connectome_necessity_protocol": Path("configs/connectome_necessity_v1.json"),
}


def _parse_artifacts(values: list[str]) -> dict[str, Path]:
    result = dict(DEFAULT_ARTIFACTS)
    for value in values:
        if "=" not in value:
            raise ValueError("artifact arguments must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        result[name] = Path(raw_path)
    return result


def build_readiness_report(
    *,
    zero_shot_spec: ExperimentSpec,
    physiology_config: dict[str, Any],
    null_config: dict[str, Any],
    necessity_config: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    zero_shot = validate_zero_shot_spec(zero_shot_spec)
    physiology = validate_physiology(physiology_config)
    null_factory = validate_null_factory(null_config)
    necessity = validate_connectome_necessity(necessity_config)

    blockers: list[dict[str, str]] = []
    if not zero_shot["valid_for_preregistration"]:
        blockers.append({"kind": "protocol", "detail": "zero-shot specification failed preregistration"})
    if not physiology["valid_for_preregistration"]:
        blockers.append({"kind": "protocol", "detail": "physiology calibration failed preregistration"})
    if not null_factory["valid_for_preregistration"]:
        blockers.append({"kind": "protocol", "detail": "NullFactory v2 failed preregistration"})
    if not necessity["valid_for_preregistration"]:
        blockers.append({"kind": "protocol", "detail": "ConnectomeNecessity v1 failed preregistration"})

    required = set(zero_shot["required_artifacts"])
    artifact_reports: dict[str, Any] = {}
    for name in sorted(required):
        path = artifact_paths.get(name)
        if path is None:
            artifact_reports[name] = {"status": "path-unspecified"}
            blockers.append({"kind": "artifact", "detail": f"{name}: path unspecified"})
            continue
        if not path.exists():
            artifact_reports[name] = {"status": "missing", "path": str(path)}
            blockers.append({"kind": "artifact", "detail": f"{name}: missing at {path}"})
            continue
        try:
            validation = validate_zero_shot_artifact(name, path)
        except (ValueError, TypeError, KeyError) as exc:
            artifact_reports[name] = {
                "status": "invalid",
                "path": str(path),
                "error": str(exc),
            }
            blockers.append({"kind": "artifact", "detail": f"{name}: invalid ({exc})"})
            continue
        artifact_reports[name] = {
            "status": "valid" if validation["valid"] else "invalid",
            "path": str(path),
            "validation": validation,
        }
        if not validation["valid"]:
            blockers.append({"kind": "artifact", "detail": f"{name}: semantic validation failed"})

    probe_binding: dict[str, Any] | None = None
    ledger_path = artifact_paths.get("evidence_ledger")
    if ledger_path is not None and ledger_path.exists():
        try:
            ledger = EvidenceLedger.load(ledger_path)
            probe_binding = bind_probe_authorities(physiology_config, ledger)
            if probe_binding["unresolved_probes"]:
                blockers.append(
                    {
                        "kind": "science",
                        "detail": "unresolved physiology probes: "
                        + ",".join(probe_binding["unresolved_probes"]),
                    }
                )
        except (ValueError, TypeError, KeyError) as exc:
            blockers.append({"kind": "science", "detail": f"physiology evidence binding failed: {exc}"})

    blocked_nulls = list(null_factory.get("blocked_families", []))
    if blocked_nulls:
        blockers.append(
            {
                "kind": "software-or-metadata",
                "detail": "confirmatory null families not ready: " + ",".join(blocked_nulls),
            }
        )

    return {
        "protocol": PROTOCOL,
        "experiment_id": zero_shot_spec.experiment_id,
        "ready_to_seal_zero_shot": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "protocol_reports": {
            "zero_shot": zero_shot,
            "physiology_calibration": physiology,
            "null_factory": null_factory,
            "connectome_necessity": necessity,
        },
        "artifact_reports": artifact_reports,
        "physiology_probe_binding": probe_binding,
        "claim_boundary": (
            "Readiness means all declared software, evidence, and artifact gates required to seal "
            "the zero-shot experiment are satisfied. It is not a navigation result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report blockers for the zero-shot science program")
    parser.add_argument("--zero-shot", default="configs/zero_shot_latent_wiring_v1.json")
    parser.add_argument("--physiology", default="configs/physiology_calibration_v1.json")
    parser.add_argument("--nulls", default="configs/null_factory_v2.json")
    parser.add_argument("--necessity", default="configs/connectome_necessity_v1.json")
    parser.add_argument("--artifact", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--output", default="results/science-readiness-v1.json")
    args = parser.parse_args()

    report = build_readiness_report(
        zero_shot_spec=ExperimentSpec.load(args.zero_shot),
        physiology_config=json.loads(Path(args.physiology).read_text()),
        null_config=json.loads(Path(args.nulls).read_text()),
        necessity_config=json.loads(Path(args.necessity).read_text()),
        artifact_paths=_parse_artifacts(args.artifact),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{output}")
    print(
        f"ready_to_seal_zero_shot={report['ready_to_seal_zero_shot']} "
        f"blockers={report['blocker_count']}"
    )
    for blocker in report["blockers"]:
        print(f"  BLOCK  [{blocker['kind']}] {blocker['detail']}")


if __name__ == "__main__":
    main()
