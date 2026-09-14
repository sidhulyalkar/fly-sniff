from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .evidence import EvidenceClass, EvidenceLedger
from .freeze import canonical_sha256


class CalibrationStrength(str, Enum):
    """How strongly one evidence-backed target may constrain the model."""

    QUANTITATIVE_FIT = "quantitative_fit"
    QUALITATIVE_SENSITIVITY = "qualitative_sensitivity"


class ParameterScope(str, Enum):
    """Allowed v1 parameter classes. Individual edges are intentionally absent."""

    GLOBAL_DYNAMICS = "global_dynamics"
    SENSORY_GAIN = "sensory_gain"
    POPULATION_GAIN = "population_gain"
    POPULATION_TIMESCALE = "population_timescale"


_SUPPORTED_CALIBRATION_EVIDENCE = {
    EvidenceClass.MEASURED_PHYSIOLOGY,
    EvidenceClass.CROSS_DATASET_PRIOR,
}


@dataclass(frozen=True)
class CalibrationTarget:
    target_id: str
    biological_entity: str
    phenomenon: str
    strength: CalibrationStrength
    evidence_record_ids: tuple[str, ...]
    model_observable: str
    objective: str
    parameter_scopes: tuple[ParameterScope, ...]
    uncertainty_policy: str
    allowed_interpretation: str
    forbidden_interpretations: tuple[str, ...]

    def validate(self, ledger: EvidenceLedger) -> None:
        if not self.target_id.strip():
            raise ValueError("calibration target_id must be non-empty")
        if not self.biological_entity.strip() or not self.phenomenon.strip():
            raise ValueError(f"calibration target {self.target_id}: biological description is required")
        if not self.evidence_record_ids:
            raise ValueError(f"calibration target {self.target_id}: evidence records are required")
        if len(self.evidence_record_ids) != len(set(self.evidence_record_ids)):
            raise ValueError(f"calibration target {self.target_id}: duplicate evidence record IDs")
        if not self.model_observable.strip() or not self.objective.strip():
            raise ValueError(f"calibration target {self.target_id}: model observable/objective required")
        if not self.parameter_scopes:
            raise ValueError(f"calibration target {self.target_id}: parameter scopes are required")
        if not self.uncertainty_policy.strip():
            raise ValueError(f"calibration target {self.target_id}: uncertainty policy required")
        if not self.allowed_interpretation.strip():
            raise ValueError(f"calibration target {self.target_id}: allowed interpretation required")
        if not self.forbidden_interpretations:
            raise ValueError(f"calibration target {self.target_id}: forbidden interpretations required")

        records_by_id = {record.record_id: record for record in ledger.records}
        missing = [record_id for record_id in self.evidence_record_ids if record_id not in records_by_id]
        if missing:
            raise ValueError(f"calibration target {self.target_id}: missing evidence records {missing}")
        evidence_classes = {records_by_id[record_id].evidence_class for record_id in self.evidence_record_ids}
        if not evidence_classes & _SUPPORTED_CALIBRATION_EVIDENCE:
            raise ValueError(
                f"calibration target {self.target_id}: structural/model evidence alone cannot calibrate dynamics"
            )
        if (
            self.strength is CalibrationStrength.QUANTITATIVE_FIT
            and EvidenceClass.MEASURED_PHYSIOLOGY not in evidence_classes
        ):
            raise ValueError(
                f"calibration target {self.target_id}: quantitative fit requires measured physiology"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "biological_entity": self.biological_entity,
            "phenomenon": self.phenomenon,
            "strength": self.strength.value,
            "evidence_record_ids": list(self.evidence_record_ids),
            "model_observable": self.model_observable,
            "objective": self.objective,
            "parameter_scopes": [scope.value for scope in self.parameter_scopes],
            "uncertainty_policy": self.uncertainty_policy,
            "allowed_interpretation": self.allowed_interpretation,
            "forbidden_interpretations": list(self.forbidden_interpretations),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationTarget:
        return cls(
            target_id=str(payload["target_id"]),
            biological_entity=str(payload["biological_entity"]),
            phenomenon=str(payload["phenomenon"]),
            strength=CalibrationStrength(payload["strength"]),
            evidence_record_ids=tuple(str(value) for value in payload["evidence_record_ids"]),
            model_observable=str(payload["model_observable"]),
            objective=str(payload["objective"]),
            parameter_scopes=tuple(ParameterScope(value) for value in payload["parameter_scopes"]),
            uncertainty_policy=str(payload["uncertainty_policy"]),
            allowed_interpretation=str(payload["allowed_interpretation"]),
            forbidden_interpretations=tuple(str(value) for value in payload["forbidden_interpretations"]),
        )


@dataclass(frozen=True)
class PhysiologyCalibrationProtocol:
    protocol_id: str
    evidence_ledger_sha256: str
    targets: tuple[CalibrationTarget, ...]
    model_family_ids: tuple[str, ...]
    navigation_reward_allowed: bool = False
    topology_specific_fit_allowed: bool = False
    whole_graph_backprop_allowed: bool = False
    navigation_performance_selection_allowed: bool = False
    individual_edge_weight_fit_allowed: bool = False
    body_membership_fit_allowed: bool = False
    role_assignment_fit_allowed: bool = False
    schema: str = "fly-sniff-physiology-calibration-v1"

    def validate(self, ledger: EvidenceLedger) -> None:
        ledger.validate()
        if self.schema != "fly-sniff-physiology-calibration-v1":
            raise ValueError(f"unsupported physiology calibration schema: {self.schema}")
        if not self.protocol_id.strip():
            raise ValueError("physiology calibration protocol_id must be non-empty")
        if self.evidence_ledger_sha256 != ledger.sha256:
            raise ValueError("physiology calibration protocol is bound to a different evidence ledger")
        if not self.targets:
            raise ValueError("physiology calibration protocol requires at least one target")
        if not self.model_family_ids:
            raise ValueError("physiology calibration protocol requires a prespecified model family")
        if len(self.model_family_ids) != len(set(self.model_family_ids)):
            raise ValueError("model_family_ids must be unique")

        forbidden_flags = {
            "navigation_reward_allowed": self.navigation_reward_allowed,
            "topology_specific_fit_allowed": self.topology_specific_fit_allowed,
            "whole_graph_backprop_allowed": self.whole_graph_backprop_allowed,
            "navigation_performance_selection_allowed": self.navigation_performance_selection_allowed,
            "individual_edge_weight_fit_allowed": self.individual_edge_weight_fit_allowed,
            "body_membership_fit_allowed": self.body_membership_fit_allowed,
            "role_assignment_fit_allowed": self.role_assignment_fit_allowed,
        }
        enabled = [name for name, value in forbidden_flags.items() if value]
        if enabled:
            raise ValueError(
                "Program A physiology calibration forbids these freedoms: " + ", ".join(enabled)
            )

        seen_targets: set[str] = set()
        for target in self.targets:
            if target.target_id in seen_targets:
                raise ValueError(f"duplicate calibration target_id: {target.target_id}")
            target.validate(ledger)
            seen_targets.add(target.target_id)

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_id": self.protocol_id,
            "evidence_ledger_sha256": self.evidence_ledger_sha256,
            "targets": [target.to_dict() for target in self.targets],
            "model_family_ids": list(self.model_family_ids),
            "navigation_reward_allowed": self.navigation_reward_allowed,
            "topology_specific_fit_allowed": self.topology_specific_fit_allowed,
            "whole_graph_backprop_allowed": self.whole_graph_backprop_allowed,
            "navigation_performance_selection_allowed": self.navigation_performance_selection_allowed,
            "individual_edge_weight_fit_allowed": self.individual_edge_weight_fit_allowed,
            "body_membership_fit_allowed": self.body_membership_fit_allowed,
            "role_assignment_fit_allowed": self.role_assignment_fit_allowed,
            "claim_boundary": [
                "calibration constrains modeled dynamics; it does not validate navigation",
                "quantitative fits require measured physiology",
                "cross-dataset priors may support qualitative sensitivity but cannot substitute for measured physiology in a quantitative fit",
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["protocol_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PhysiologyCalibrationProtocol:
        protocol = cls(
            protocol_id=str(payload["protocol_id"]),
            evidence_ledger_sha256=str(payload["evidence_ledger_sha256"]),
            targets=tuple(CalibrationTarget.from_dict(item) for item in payload["targets"]),
            model_family_ids=tuple(str(value) for value in payload["model_family_ids"]),
            navigation_reward_allowed=bool(payload.get("navigation_reward_allowed", False)),
            topology_specific_fit_allowed=bool(payload.get("topology_specific_fit_allowed", False)),
            whole_graph_backprop_allowed=bool(payload.get("whole_graph_backprop_allowed", False)),
            navigation_performance_selection_allowed=bool(
                payload.get("navigation_performance_selection_allowed", False)
            ),
            individual_edge_weight_fit_allowed=bool(
                payload.get("individual_edge_weight_fit_allowed", False)
            ),
            body_membership_fit_allowed=bool(payload.get("body_membership_fit_allowed", False)),
            role_assignment_fit_allowed=bool(payload.get("role_assignment_fit_allowed", False)),
            schema=str(payload.get("schema", "fly-sniff-physiology-calibration-v1")),
        )
        claimed_hash = payload.get("protocol_sha256")
        if claimed_hash is not None and claimed_hash != protocol.sha256:
            raise ValueError("physiology calibration protocol hash mismatch")
        return protocol


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a navigation-independent physiology calibration protocol."
    )
    parser.add_argument("--ledger", required=True, help="EvidenceLedger JSON")
    parser.add_argument("--protocol", required=True, help="Physiology calibration protocol JSON")
    args = parser.parse_args(argv)

    ledger = EvidenceLedger.from_dict(_load_json(args.ledger))
    protocol = PhysiologyCalibrationProtocol.from_dict(_load_json(args.protocol))
    protocol.validate(ledger)
    print(
        json.dumps(
            {
                "status": "valid_navigation_independent_calibration_contract",
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.sha256,
                "target_count": len(protocol.targets),
                "model_family_count": len(protocol.model_family_ids),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())