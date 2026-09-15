from __future__ import annotations

import dataclasses
import enum
from typing import Any

from .freeze import canonical_sha256


class ConditionRegime(str, enum.Enum):
    ID = "id"
    OOD = "ood"


class InterventionKind(str, enum.Enum):
    CIRCUIT_LESION = "circuit_lesion"
    SENSORY_ABLATION = "sensory_ablation"
    OUTPUT_IDENTITY_SCRAMBLE = "output_identity_scramble"
    TEMPORAL_SHUFFLE = "temporal_shuffle"


@dataclasses.dataclass(frozen=True)
class ConditionFamily:
    family_id: str
    regime: ConditionRegime
    scientific_rationale: str
    source_rule: str
    start_rule: str
    wind_rule: str
    plume_rule: str
    sensor_rule: str
    episode_rule: str
    success_rule: str
    hidden_seed_namespace: str

    def validate(self) -> None:
        fields = {
            "family_id": self.family_id,
            "scientific_rationale": self.scientific_rationale,
            "source_rule": self.source_rule,
            "start_rule": self.start_rule,
            "wind_rule": self.wind_rule,
            "plume_rule": self.plume_rule,
            "sensor_rule": self.sensor_rule,
            "episode_rule": self.episode_rule,
            "success_rule": self.success_rule,
            "hidden_seed_namespace": self.hidden_seed_namespace,
        }
        empty = [name for name, value in fields.items() if not value.strip()]
        if empty:
            raise ValueError(
                f"condition family {self.family_id!r} has empty required fields: {empty}"
            )
        if ":" in self.family_id:
            raise ValueError("condition family_id may not contain ':'")
        if self.hidden_seed_namespace.startswith("development"):
            raise ValueError("final evaluation condition families may not use development seed namespaces")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "family_id": self.family_id,
            "regime": self.regime.value,
            "scientific_rationale": self.scientific_rationale,
            "source_rule": self.source_rule,
            "start_rule": self.start_rule,
            "wind_rule": self.wind_rule,
            "plume_rule": self.plume_rule,
            "sensor_rule": self.sensor_rule,
            "episode_rule": self.episode_rule,
            "success_rule": self.success_rule,
            "hidden_seed_namespace": self.hidden_seed_namespace,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConditionFamily:
        family = cls(
            family_id=str(payload["family_id"]),
            regime=ConditionRegime(payload["regime"]),
            scientific_rationale=str(payload["scientific_rationale"]),
            source_rule=str(payload["source_rule"]),
            start_rule=str(payload["start_rule"]),
            wind_rule=str(payload["wind_rule"]),
            plume_rule=str(payload["plume_rule"]),
            sensor_rule=str(payload["sensor_rule"]),
            episode_rule=str(payload["episode_rule"]),
            success_rule=str(payload["success_rule"]),
            hidden_seed_namespace=str(payload["hidden_seed_namespace"]),
        )
        family.validate()
        return family


@dataclasses.dataclass(frozen=True)
class AcuteIntervention:
    intervention_id: str
    kind: InterventionKind
    transformation: str
    scientific_rationale: str
    affected_entities: tuple[str, ...]
    parameter_refit_allowed: bool = False
    topology_reselection_allowed: bool = False
    role_reselection_allowed: bool = False

    def validate(self) -> None:
        if not self.intervention_id.strip():
            raise ValueError("intervention_id must be non-empty")
        if ":" in self.intervention_id:
            raise ValueError("intervention_id may not contain ':'")
        if not self.transformation.strip() or not self.scientific_rationale.strip():
            raise ValueError(f"intervention {self.intervention_id}: transformation/rationale required")
        if not self.affected_entities:
            raise ValueError(f"intervention {self.intervention_id}: affected_entities required")
        if len(self.affected_entities) != len(set(self.affected_entities)):
            raise ValueError(f"intervention {self.intervention_id}: affected_entities must be unique")
        if self.parameter_refit_allowed:
            raise ValueError(
                f"acute intervention {self.intervention_id}: parameter refitting is forbidden"
            )
        if self.topology_reselection_allowed:
            raise ValueError(
                f"acute intervention {self.intervention_id}: topology reselection is forbidden"
            )
        if self.role_reselection_allowed:
            raise ValueError(
                f"acute intervention {self.intervention_id}: role reselection is forbidden"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "intervention_id": self.intervention_id,
            "kind": self.kind.value,
            "transformation": self.transformation,
            "scientific_rationale": self.scientific_rationale,
            "affected_entities": list(self.affected_entities),
            "parameter_refit_allowed": self.parameter_refit_allowed,
            "topology_reselection_allowed": self.topology_reselection_allowed,
            "role_reselection_allowed": self.role_reselection_allowed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AcuteIntervention:
        intervention = cls(
            intervention_id=str(payload["intervention_id"]),
            kind=InterventionKind(payload["kind"]),
            transformation=str(payload["transformation"]),
            scientific_rationale=str(payload["scientific_rationale"]),
            affected_entities=tuple(str(value) for value in payload["affected_entities"]),
            parameter_refit_allowed=bool(payload.get("parameter_refit_allowed", False)),
            topology_reselection_allowed=bool(payload.get("topology_reselection_allowed", False)),
            role_reselection_allowed=bool(payload.get("role_reselection_allowed", False)),
        )
        intervention.validate()
        return intervention


@dataclasses.dataclass(frozen=True)
class ProgramAEvaluationContract:
    contract_id: str
    plume_contract_sha256: str
    bilateral_sensor_geometry_sha256: str
    condition_families: tuple[ConditionFamily, ...]
    interventions: tuple[AcuteIntervention, ...]
    paired_condition_ids_required: bool = True
    source_coordinates_visible_to_controller: bool = False
    final_seeds_materialized_before_entropy_reveal: bool = False
    topology_specific_condition_selection_allowed: bool = False
    outcome_driven_ood_selection_allowed: bool = False
    acute_intervention_retraining_allowed: bool = False
    schema: str = "fly-sniff-program-a-evaluation-contract-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-program-a-evaluation-contract-v1":
            raise ValueError(f"unsupported evaluation contract schema: {self.schema}")
        if not self.contract_id.strip():
            raise ValueError("evaluation contract_id must be non-empty")
        _validate_sha256(self.plume_contract_sha256, field="plume_contract_sha256")
        _validate_sha256(
            self.bilateral_sensor_geometry_sha256,
            field="bilateral_sensor_geometry_sha256",
        )
        if not self.condition_families:
            raise ValueError("evaluation contract requires condition families")
        if not self.interventions:
            raise ValueError("evaluation contract requires at least one acute intervention")

        family_ids: set[str] = set()
        namespaces: set[str] = set()
        regimes: set[ConditionRegime] = set()
        for family in self.condition_families:
            family.validate()
            if family.family_id in family_ids:
                raise ValueError(f"duplicate condition family_id: {family.family_id}")
            if family.hidden_seed_namespace in namespaces:
                raise ValueError(
                    "condition families require distinct hidden seed namespaces: "
                    f"{family.hidden_seed_namespace}"
                )
            family_ids.add(family.family_id)
            namespaces.add(family.hidden_seed_namespace)
            regimes.add(family.regime)

        if ConditionRegime.ID not in regimes:
            raise ValueError("evaluation contract requires at least one ID condition family")
        if ConditionRegime.OOD not in regimes:
            raise ValueError("evaluation contract requires at least one OOD condition family")

        intervention_ids: set[str] = set()
        for intervention in self.interventions:
            intervention.validate()
            if intervention.intervention_id in intervention_ids:
                raise ValueError(f"duplicate intervention_id: {intervention.intervention_id}")
            intervention_ids.add(intervention.intervention_id)

        if not self.paired_condition_ids_required:
            raise ValueError("Program A requires exact paired condition IDs across graph realizations")
        if self.source_coordinates_visible_to_controller:
            raise ValueError("source coordinates may exist in evaluator metadata but not controller input")
        if self.final_seeds_materialized_before_entropy_reveal:
            raise ValueError("final seeds may not be materialized before hidden entropy reveal")
        if self.topology_specific_condition_selection_allowed:
            raise ValueError("topology-specific condition selection is forbidden")
        if self.outcome_driven_ood_selection_allowed:
            raise ValueError("OOD conditions may not be selected from topology outcomes")
        if self.acute_intervention_retraining_allowed:
            raise ValueError("acute Program A interventions may not retrain")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "contract_id": self.contract_id,
            "plume_contract_sha256": self.plume_contract_sha256,
            "bilateral_sensor_geometry_sha256": self.bilateral_sensor_geometry_sha256,
            "condition_families": [
                family.to_dict() for family in sorted(self.condition_families, key=lambda item: item.family_id)
            ],
            "interventions": [
                intervention.to_dict()
                for intervention in sorted(self.interventions, key=lambda item: item.intervention_id)
            ],
            "paired_condition_ids_required": self.paired_condition_ids_required,
            "source_coordinates_visible_to_controller": self.source_coordinates_visible_to_controller,
            "final_seeds_materialized_before_entropy_reveal": self.final_seeds_materialized_before_entropy_reveal,
            "topology_specific_condition_selection_allowed": self.topology_specific_condition_selection_allowed,
            "outcome_driven_ood_selection_allowed": self.outcome_driven_ood_selection_allowed,
            "acute_intervention_retraining_allowed": self.acute_intervention_retraining_allowed,
            "claim_boundary": [
                "condition metadata may contain source coordinates while controller observations may not",
                "acute interventions are evaluated with frozen parameters and no retraining",
                "OOD dimensions are frozen for scientific interpretability before topology performance",
                "condition IDs are nuisance-condition identities shared exactly across graph realizations",
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["contract_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProgramAEvaluationContract:
        contract = cls(
            contract_id=str(payload["contract_id"]),
            plume_contract_sha256=str(payload["plume_contract_sha256"]),
            bilateral_sensor_geometry_sha256=str(payload["bilateral_sensor_geometry_sha256"]),
            condition_families=tuple(
                ConditionFamily.from_dict(item) for item in payload["condition_families"]
            ),
            interventions=tuple(
                AcuteIntervention.from_dict(item) for item in payload["interventions"]
            ),
            paired_condition_ids_required=bool(
                payload.get("paired_condition_ids_required", True)
            ),
            source_coordinates_visible_to_controller=bool(
                payload.get("source_coordinates_visible_to_controller", False)
            ),
            final_seeds_materialized_before_entropy_reveal=bool(
                payload.get("final_seeds_materialized_before_entropy_reveal", False)
            ),
            topology_specific_condition_selection_allowed=bool(
                payload.get("topology_specific_condition_selection_allowed", False)
            ),
            outcome_driven_ood_selection_allowed=bool(
                payload.get("outcome_driven_ood_selection_allowed", False)
            ),
            acute_intervention_retraining_allowed=bool(
                payload.get("acute_intervention_retraining_allowed", False)
            ),
            schema=str(
                payload.get(
                    "schema",
                    "fly-sniff-program-a-evaluation-contract-v1",
                )
            ),
        )
        contract.validate()
        claimed_hash = payload.get("contract_sha256")
        if claimed_hash is not None and claimed_hash != contract.sha256:
            raise ValueError("Program A evaluation contract hash mismatch")
        return contract

    def condition_id(self, *, family_id: str, final_seed: int, replicate_index: int = 0) -> str:
        """Derive a stable condition identity after final entropy has produced a seed.

        This function does not derive or expose a final seed. The seed must come from the
        separately locked commit-reveal protocol after model/spec freeze.
        """

        self.validate()
        family_ids = {family.family_id for family in self.condition_families}
        if family_id not in family_ids:
            raise ValueError(f"unknown condition family_id: {family_id}")
        if final_seed < 0:
            raise ValueError("final_seed must be non-negative")
        if replicate_index < 0:
            raise ValueError("replicate_index must be non-negative")
        identity = canonical_sha256(
            {
                "schema": "fly-sniff-program-a-condition-id-v1",
                "evaluation_contract_sha256": self.sha256,
                "family_id": family_id,
                "final_seed": final_seed,
                "replicate_index": replicate_index,
            }
        )
        return f"{family_id}:{identity}"


def assert_paired_condition_ids(*cohort_condition_ids: tuple[str, ...]) -> None:
    if len(cohort_condition_ids) < 2:
        raise ValueError("paired-condition audit requires at least two cohorts")
    reference = cohort_condition_ids[0]
    if not reference:
        raise ValueError("paired condition IDs may not be empty")
    for condition_ids in cohort_condition_ids[1:]:
        if condition_ids != reference:
            raise ValueError("all Program A cohorts must use exact ordered paired condition IDs")


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in frozenset("0123456789abcdef") for char in value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")
