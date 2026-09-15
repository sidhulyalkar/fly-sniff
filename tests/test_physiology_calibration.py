from __future__ import annotations

import pytest

from fly_sniff.evidence import EvidenceClass, EvidenceLedger, EvidenceRecord
from fly_sniff.physiology_calibration import (
    CalibrationStrength,
    CalibrationTarget,
    ParameterScope,
    PhysiologyCalibrationProtocol,
)


def _ledger() -> EvidenceLedger:
    return EvidenceLedger.build(
        [
            EvidenceRecord(
                record_id="pfn-physiology",
                subject="PFN",
                predicate="airflow_tuning_basis",
                value={"qualitative": "lateralized airflow tuning"},
                evidence_class=EvidenceClass.MEASURED_PHYSIOLOGY,
                authority="independent published physiology",
                dataset="external-preparation",
            ),
            EvidenceRecord(
                record_id="pfl3-prior",
                subject="PFL3",
                predicate="steering_relationship",
                value={"direction": "lateralized"},
                evidence_class=EvidenceClass.CROSS_DATASET_PRIOR,
                authority="cross-dataset literature prior",
                dataset="external-connectome-preparation",
            ),
            EvidenceRecord(
                record_id="structure-only",
                subject="candidate-edge",
                predicate="connected",
                value=True,
                evidence_class=EvidenceClass.MEASURED_STRUCTURE,
                authority="MaleCNS",
                dataset="male-cns:v1.0",
            ),
        ]
    )


def _target(
    *,
    record_ids: tuple[str, ...] = ("pfn-physiology",),
    strength: CalibrationStrength = CalibrationStrength.QUANTITATIVE_FIT,
) -> CalibrationTarget:
    return CalibrationTarget(
        target_id="pfn-airflow",
        biological_entity="PFN airflow-responsive population",
        phenomenon="airflow tuning geometry",
        strength=strength,
        evidence_record_ids=record_ids,
        model_observable="population response across frozen airflow directions",
        objective="match independently measured tuning geometry within prespecified uncertainty",
        parameter_scopes=(ParameterScope.SENSORY_GAIN, ParameterScope.POPULATION_TIMESCALE),
        uncertainty_policy="report fit and sensitivity across prespecified measurement uncertainty",
        allowed_interpretation="independent physiology constrains modeled PFN response parameters",
        forbidden_interpretations=(
            "this proves MaleCNS navigation",
            "this identifies a unique biological dynamical model",
        ),
    )


def _protocol(ledger: EvidenceLedger, **overrides: object) -> PhysiologyCalibrationProtocol:
    values: dict[str, object] = {
        "protocol_id": "program-a-calibration-v1",
        "evidence_ledger_sha256": ledger.sha256,
        "targets": (_target(),),
        "model_family_ids": ("rate-normalized-v1", "rate-unnormalized-sensitivity-v1"),
    }
    values.update(overrides)
    return PhysiologyCalibrationProtocol(**values)  # type: ignore[arg-type]


def test_quantitative_calibration_accepts_measured_physiology() -> None:
    ledger = _ledger()
    protocol = _protocol(ledger)
    protocol.validate(ledger)
    assert protocol.to_dict()["navigation_reward_allowed"] is False
    assert protocol.sha256 == protocol.to_dict()["protocol_sha256"]


def test_quantitative_fit_rejects_cross_dataset_prior_without_measured_physiology() -> None:
    ledger = _ledger()
    target = _target(record_ids=("pfl3-prior",))
    protocol = _protocol(ledger, targets=(target,))
    with pytest.raises(ValueError, match="quantitative fit requires measured physiology"):
        protocol.validate(ledger)


def test_qualitative_sensitivity_can_use_cross_dataset_prior() -> None:
    ledger = _ledger()
    target = _target(
        record_ids=("pfl3-prior",),
        strength=CalibrationStrength.QUALITATIVE_SENSITIVITY,
    )
    protocol = _protocol(ledger, targets=(target,))
    protocol.validate(ledger)


def test_structural_evidence_alone_cannot_calibrate_dynamics() -> None:
    ledger = _ledger()
    target = _target(
        record_ids=("structure-only",),
        strength=CalibrationStrength.QUALITATIVE_SENSITIVITY,
    )
    protocol = _protocol(ledger, targets=(target,))
    with pytest.raises(ValueError, match="structural/model evidence alone"):
        protocol.validate(ledger)


@pytest.mark.parametrize(
    "field",
    [
        "navigation_reward_allowed",
        "topology_specific_fit_allowed",
        "whole_graph_backprop_allowed",
        "navigation_performance_selection_allowed",
        "individual_edge_weight_fit_allowed",
        "body_membership_fit_allowed",
        "role_assignment_fit_allowed",
    ],
)
def test_program_a_calibration_rejects_performance_or_topology_freedom(field: str) -> None:
    ledger = _ledger()
    protocol = _protocol(ledger, **{field: True})
    with pytest.raises(ValueError, match=field):
        protocol.validate(ledger)


def test_protocol_rejects_wrong_evidence_ledger_binding() -> None:
    ledger = _ledger()
    protocol = _protocol(ledger, evidence_ledger_sha256="f" * 64)
    with pytest.raises(ValueError, match="different evidence ledger"):
        protocol.validate(ledger)


def test_target_rejects_missing_evidence_record() -> None:
    ledger = _ledger()
    protocol = _protocol(ledger, targets=(_target(record_ids=("not-present",)),))
    with pytest.raises(ValueError, match="missing evidence records"):
        protocol.validate(ledger)
