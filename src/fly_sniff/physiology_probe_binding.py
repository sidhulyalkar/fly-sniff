from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evidence import EvidenceClass, EvidenceLedger
from .physiology_calibration import validate_protocol

PROTOCOL = "physiology-probe-binding-v1"
ELIGIBLE_CLASSES = {
    EvidenceClass.MEASURED_PHYSIOLOGY,
    EvidenceClass.CROSS_DATASET_PRIOR,
}


def bind_probe_authorities(
    calibration_config: dict[str, Any],
    ledger: EvidenceLedger,
) -> dict[str, Any]:
    preregistration = validate_protocol(calibration_config)
    if not preregistration["valid_for_preregistration"]:
        raise ValueError("physiology calibration protocol must pass preregistration gates before binding")

    ledger.validate()
    facts = [
        (record.entity, fact)
        for record in ledger.records
        for fact in record.facts
    ]
    probe_reports: list[dict[str, Any]] = []

    for probe in calibration_config["probes"]:
        probe_id = str(probe["id"])
        candidates = [
            (entity, fact)
            for entity, fact in facts
            if probe_id in tuple(str(x) for x in fact.metadata.get("probe_ids", []))
        ]
        eligible = [
            (entity, fact)
            for entity, fact in candidates
            if fact.evidence_class in ELIGIBLE_CLASSES
            and fact.metadata.get("calibration_eligible") is True
        ]
        rejected = [
            {
                "statement": fact.statement,
                "evidence_class": str(fact.evidence_class),
                "authority": fact.authority,
                "reason": (
                    "calibration_eligible=false"
                    if fact.metadata.get("calibration_eligible") is not True
                    else "evidence_class_not_permitted"
                ),
            }
            for _, fact in candidates
            if (fact.evidence_class not in ELIGIBLE_CLASSES)
            or fact.metadata.get("calibration_eligible") is not True
        ]
        bindings = [
            {
                "entity": {
                    "dataset": entity.dataset,
                    "body_id": entity.body_id,
                    "type": entity.type,
                    "name": entity.name,
                },
                "statement": fact.statement,
                "evidence_class": str(fact.evidence_class),
                "authority": fact.authority,
                "authority_sha256": fact.authority_sha256,
                "confidence": fact.confidence,
                "caveat": fact.caveat,
                "source_key": fact.metadata.get("source_key"),
                "underlying_evidence": fact.metadata.get("underlying_evidence"),
            }
            for entity, fact in eligible
        ]
        probe_reports.append(
            {
                "id": probe_id,
                "target": probe["target"],
                "status": "sealed" if bindings else "unresolved",
                "binding_count": len(bindings),
                "bindings": bindings,
                "rejected_candidate_facts": rejected,
                "cross_dataset_only": bool(bindings)
                and all(row["evidence_class"] == str(EvidenceClass.CROSS_DATASET_PRIOR) for row in bindings),
            }
        )

    sealed = [row["id"] for row in probe_reports if row["status"] == "sealed"]
    unresolved = [row["id"] for row in probe_reports if row["status"] != "sealed"]
    return {
        "protocol": PROTOCOL,
        "calibration_protocol": calibration_config["protocol"],
        "evidence_ledger_sha256": ledger.to_dict()["ledger_sha256"],
        "probe_count": len(probe_reports),
        "sealed_probe_count": len(sealed),
        "sealed_probes": sealed,
        "unresolved_probes": unresolved,
        "ready_to_define_fit_objective": not unresolved,
        "probe_reports": probe_reports,
        "claim_boundary": (
            "A sealed binding means an independently sourced physiological or cross-dataset prior "
            "exists for the probe. It does not define a numeric fitting objective, establish "
            "MaleCNS-specific physiology, or authorize navigation-based calibration."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind physiology probes to EvidenceLedger authorities")
    parser.add_argument("--config", default="configs/physiology_calibration_v1.json")
    parser.add_argument("--ledger", default="authority/evidence-ledger-v1.json")
    parser.add_argument("--output", default="authority/physiology-probe-bindings-v1.json")
    args = parser.parse_args()

    calibration = json.loads(Path(args.config).read_text())
    ledger = EvidenceLedger.load(args.ledger)
    report = bind_probe_authorities(calibration, ledger)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{output}")
    print(
        f"sealed={report['sealed_probe_count']}/{report['probe_count']} "
        f"ready_to_define_fit_objective={report['ready_to_define_fit_objective']}"
    )
    if report["unresolved_probes"]:
        print("unresolved=" + ",".join(report["unresolved_probes"]))


if __name__ == "__main__":
    main()
