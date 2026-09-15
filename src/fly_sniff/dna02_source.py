from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from .freeze import canonical_sha256

_HEX = frozenset("0123456789abcdef")

BLOCKED_COHORT = "BLOCKED_FIGURE3C_COHORT_UNRESOLVED"
BLOCKED_FILE_MAP = "BLOCKED_DATAVERSE_FILE_MAP_UNRESOLVED"
READY = "READY_FOR_EXTRACTION"


def _require_text(value: str, *, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _validate_hex(value: str, *, field: str, length: int) -> None:
    if len(value) != length or any(char not in _HEX for char in value):
        raise ValueError(f"{field} must be a lowercase {length}-character hexadecimal digest")


@dataclasses.dataclass(frozen=True)
class CodeAuthority:
    repository: str
    commit: str
    path: str
    git_blob_sha1: str
    role: str

    def validate(self) -> None:
        for field, value in (
            ("repository", self.repository),
            ("path", self.path),
            ("role", self.role),
        ):
            _require_text(value, field=field)
        _validate_hex(self.commit, field="code authority commit", length=40)
        _validate_hex(self.git_blob_sha1, field="code authority git_blob_sha1", length=40)

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CodeAuthority:
        authority = cls(
            repository=str(payload["repository"]),
            commit=str(payload["commit"]),
            path=str(payload["path"]),
            git_blob_sha1=str(payload["git_blob_sha1"]),
            role=str(payload["role"]),
        )
        authority.validate()
        return authority


@dataclasses.dataclass(frozen=True)
class DataverseFileRef:
    fly_alias: str
    file_id: int
    filename: str
    sha256: str

    def validate(self) -> None:
        _require_text(self.fly_alias, field="fly_alias")
        if self.file_id <= 0:
            raise ValueError("Dataverse file_id must be positive")
        _require_text(self.filename, field="Dataverse filename")
        _validate_hex(self.sha256, field="Dataverse file sha256", length=64)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataverseFileRef:
        ref = cls(
            fly_alias=str(payload["fly_alias"]),
            file_id=int(payload["file_id"]),
            filename=str(payload["filename"]),
            sha256=str(payload["sha256"]),
        )
        ref.validate()
        return ref


@dataclasses.dataclass(frozen=True)
class DNa02SourceContract:
    contract_id: str
    paper_doi: str
    paper_title: str
    figure: str
    published_n_flies: int
    published_claim: str
    dataverse_doi: str
    data_file_map: tuple[DataverseFileRef, ...]
    code_authorities: tuple[CodeAuthority, ...]
    candidate_bilateral_aliases: tuple[str, ...]
    known_raw_session_candidates: tuple[tuple[str, tuple[str, ...]], ...]
    figure3c_cohort: tuple[str, ...] | None
    figure3c_cohort_authority: str | None
    cohort_review_note: str
    source_fields: tuple[tuple[str, str], ...]
    preprocessing: tuple[tuple[str, Any], ...]
    evidence_ledger_record_ids: tuple[str, ...]
    navigation_performance_used: bool
    allowed_interpretation: tuple[str, ...]
    forbidden_interpretation: tuple[str, ...]
    schema: str = "fly-sniff-dna02-calibration-source-v1"

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.figure3c_cohort is None or self.figure3c_cohort_authority is None:
            blockers.append(BLOCKED_COHORT)
        if self.figure3c_cohort is None:
            blockers.append(BLOCKED_FILE_MAP)
        else:
            mapped = {ref.fly_alias for ref in self.data_file_map}
            if mapped != set(self.figure3c_cohort):
                blockers.append(BLOCKED_FILE_MAP)
        return tuple(sorted(set(blockers)))

    @property
    def status(self) -> str:
        return READY if not self.blockers else "BLOCKED"

    def validate(self) -> None:
        if self.schema != "fly-sniff-dna02-calibration-source-v1":
            raise ValueError(f"unsupported DNa02 source schema: {self.schema}")
        for field, value in (
            ("contract_id", self.contract_id),
            ("paper_doi", self.paper_doi),
            ("paper_title", self.paper_title),
            ("figure", self.figure),
            ("published_claim", self.published_claim),
            ("dataverse_doi", self.dataverse_doi),
            ("cohort_review_note", self.cohort_review_note),
        ):
            _require_text(value, field=field)
        if self.paper_doi != "10.7554/eLife.102230.3":
            raise ValueError("DNa02 source contract must bind the version-of-record DOI")
        if self.dataverse_doi != "10.7910/DVN/0NCLP1":
            raise ValueError("DNa02 source contract must bind the published Harvard Dataverse DOI")
        if self.figure != "Figure 3C":
            raise ValueError("Program A DNa02 v1 source target is frozen to Figure 3C")
        if self.published_n_flies != 4:
            raise ValueError("Figure 3C source contract must retain published n=4")
        if self.navigation_performance_used:
            raise ValueError("DNa02 source selection may not use navigation performance")

        if not self.code_authorities:
            raise ValueError("at least one immutable code authority is required")
        code_keys: set[tuple[str, str]] = set()
        for authority in self.code_authorities:
            authority.validate()
            key = (authority.repository, authority.path)
            if key in code_keys:
                raise ValueError(f"duplicate code authority: {key}")
            code_keys.add(key)

        if not self.candidate_bilateral_aliases:
            raise ValueError("candidate bilateral aliases may not be empty")
        if len(self.candidate_bilateral_aliases) != len(set(self.candidate_bilateral_aliases)):
            raise ValueError("candidate bilateral aliases must be unique")
        aliases = set(self.candidate_bilateral_aliases)

        raw_aliases = [alias for alias, _ in self.known_raw_session_candidates]
        if len(raw_aliases) != len(set(raw_aliases)):
            raise ValueError("known raw-session candidate aliases must be unique")
        if not set(raw_aliases).issubset(aliases):
            raise ValueError("raw-session candidate aliases must be candidate bilateral aliases")

        if self.figure3c_cohort is not None:
            if len(self.figure3c_cohort) != self.published_n_flies:
                raise ValueError("resolved Figure 3C cohort must contain exactly four flies")
            if len(self.figure3c_cohort) != len(set(self.figure3c_cohort)):
                raise ValueError("resolved Figure 3C cohort must contain unique fly aliases")
            if not set(self.figure3c_cohort).issubset(aliases):
                raise ValueError("resolved Figure 3C cohort must come from audited bilateral aliases")
            if self.figure3c_cohort_authority is None:
                raise ValueError("resolved Figure 3C cohort requires explicit authority")
            _require_text(self.figure3c_cohort_authority, field="figure3c_cohort_authority")
        elif self.figure3c_cohort_authority is not None:
            raise ValueError("cohort authority may not be supplied while figure3c_cohort is unresolved")

        file_aliases: set[str] = set()
        file_ids: set[int] = set()
        for ref in self.data_file_map:
            ref.validate()
            if ref.fly_alias not in aliases:
                raise ValueError(f"Dataverse mapping uses unknown fly alias: {ref.fly_alias}")
            if ref.fly_alias in file_aliases:
                raise ValueError(f"duplicate Dataverse fly mapping: {ref.fly_alias}")
            if ref.file_id in file_ids:
                raise ValueError(f"duplicate Dataverse file_id: {ref.file_id}")
            file_aliases.add(ref.fly_alias)
            file_ids.add(ref.file_id)
        if self.figure3c_cohort is None and self.data_file_map:
            raise ValueError("do not freeze Dataverse fly mapping before the Figure 3C cohort is resolved")
        if self.figure3c_cohort is not None and file_aliases - set(self.figure3c_cohort):
            raise ValueError("Dataverse file map may include only the frozen Figure 3C cohort")

        field_names = [name for name, _ in self.source_fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("source_fields contains duplicate names")
        required_fields = {
            "left_neuron",
            "right_neuron",
            "rotational_velocity",
            "ephys_timebase",
            "ball_timebase",
        }
        if not required_fields.issubset(field_names):
            raise ValueError("source_fields is missing required bilateral steering variables")

        preprocessing = dict(self.preprocessing)
        required_preprocessing = {
            "spike_detection",
            "firing_rate_bin_ms",
            "firing_rate_smoothing",
            "firing_rate_smoothing_window_ms",
            "neural_to_behavior_alignment_ms",
            "alignment_interpretation",
            "figure_average_bin_ms",
            "primary_predictor",
            "primary_outcome",
            "relationship",
            "normalization_policy",
        }
        missing = required_preprocessing - set(preprocessing)
        if missing:
            raise ValueError(f"published preprocessing is missing fields: {sorted(missing)}")
        if int(preprocessing["firing_rate_bin_ms"]) != 10:
            raise ValueError("Figure 3B-C firing-rate bin is frozen to 10 ms")
        if str(preprocessing["firing_rate_smoothing"]).lower() != "exponential":
            raise ValueError("Figure 3B-C firing-rate smoothing must remain exponential")
        if int(preprocessing["firing_rate_smoothing_window_ms"]) != 30:
            raise ValueError("Figure 3B-C firing-rate smoothing is frozen to 30 ms")
        if int(preprocessing["neural_to_behavior_alignment_ms"]) != 150:
            raise ValueError("Figure 3 reproduction alignment is frozen to 150 ms")
        if int(preprocessing["figure_average_bin_ms"]) != 50:
            raise ValueError("Figure 3 colormap averaging window is frozen to 50 ms")
        if str(preprocessing["primary_predictor"]) != (
            "right_firing_rate_hz - left_firing_rate_hz"
        ):
            raise ValueError("Figure 3C predictor must remain right-minus-left firing rate")
        interpretation = str(preprocessing["alignment_interpretation"]).lower()
        if "universal" not in interpretation or "must not" not in interpretation:
            raise ValueError("150 ms alignment must explicitly reject universal-delay interpretation")
        if "treadmill inertia" not in interpretation:
            raise ValueError("150 ms alignment must preserve the spherical-treadmill inertia caveat")

        required_evidence = {
            "rayshubskiy2025-dna02-bilateral-steering",
            "rayshubskiy2025-dna02-temporal-precedence",
        }
        if not required_evidence.issubset(self.evidence_ledger_record_ids):
            raise ValueError("DNa02 source contract must retain steering and timing-caveat evidence IDs")
        if not self.allowed_interpretation or not self.forbidden_interpretation:
            raise ValueError("allowed and forbidden interpretation boundaries are required")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "contract_id": self.contract_id,
            "status": self.status,
            "blockers": list(self.blockers),
            "paper": {
                "doi": self.paper_doi,
                "title": self.paper_title,
                "figure": self.figure,
                "published_n_flies": self.published_n_flies,
                "published_claim": self.published_claim,
            },
            "data_authority": {
                "repository": "Harvard Dataverse",
                "doi": self.dataverse_doi,
                "exact_file_map": [ref.to_dict() for ref in self.data_file_map],
                "mapping_status": READY if not self.blockers else BLOCKED_FILE_MAP,
            },
            "code_authorities": [authority.to_dict() for authority in self.code_authorities],
            "candidate_bilateral_aliases": list(self.candidate_bilateral_aliases),
            "known_raw_session_candidates": {
                alias: list(values) for alias, values in self.known_raw_session_candidates
            },
            "figure3c_cohort": (
                None if self.figure3c_cohort is None else list(self.figure3c_cohort)
            ),
            "figure3c_cohort_authority": self.figure3c_cohort_authority,
            "cohort_status": READY if self.figure3c_cohort is not None else BLOCKED_COHORT,
            "cohort_review_note": self.cohort_review_note,
            "source_fields": dict(self.source_fields),
            "published_figure3c_preprocessing": dict(self.preprocessing),
            "evidence_ledger_record_ids": list(self.evidence_ledger_record_ids),
            "navigation_performance_used": self.navigation_performance_used,
            "allowed_interpretation": list(self.allowed_interpretation),
            "forbidden_interpretation": list(self.forbidden_interpretation),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["source_contract_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DNa02SourceContract:
        paper = payload["paper"]
        data = payload["data_authority"]
        raw_sessions = payload.get("known_raw_session_candidates", {})
        contract = cls(
            contract_id=str(payload["contract_id"]),
            paper_doi=str(paper["doi"]),
            paper_title=str(paper["title"]),
            figure=str(paper["figure"]),
            published_n_flies=int(paper["published_n_flies"]),
            published_claim=str(paper["published_claim"]),
            dataverse_doi=str(data["doi"]),
            data_file_map=tuple(
                DataverseFileRef.from_dict(item) for item in data.get("exact_file_map", [])
            ),
            code_authorities=tuple(
                CodeAuthority.from_dict(item) for item in payload.get("code_authorities", [])
            ),
            candidate_bilateral_aliases=tuple(
                str(value) for value in payload.get("candidate_bilateral_aliases", [])
            ),
            known_raw_session_candidates=tuple(
                (str(alias), tuple(str(value) for value in values))
                for alias, values in sorted(raw_sessions.items())
            ),
            figure3c_cohort=(
                None
                if payload.get("figure3c_cohort") is None
                else tuple(str(value) for value in payload["figure3c_cohort"])
            ),
            figure3c_cohort_authority=(
                None
                if payload.get("figure3c_cohort_authority") is None
                else str(payload["figure3c_cohort_authority"])
            ),
            cohort_review_note=str(payload["cohort_review_note"]),
            source_fields=tuple(
                (str(name), str(value))
                for name, value in sorted(payload.get("source_fields", {}).items())
            ),
            preprocessing=tuple(
                (str(name), value)
                for name, value in sorted(
                    payload.get("published_figure3c_preprocessing", {}).items()
                )
            ),
            evidence_ledger_record_ids=tuple(
                str(value) for value in payload.get("evidence_ledger_record_ids", [])
            ),
            navigation_performance_used=bool(payload.get("navigation_performance_used", False)),
            allowed_interpretation=tuple(
                str(value) for value in payload.get("allowed_interpretation", [])
            ),
            forbidden_interpretation=tuple(
                str(value) for value in payload.get("forbidden_interpretation", [])
            ),
            schema=str(payload.get("schema", "fly-sniff-dna02-calibration-source-v1")),
        )
        contract.validate()
        claimed_hash = payload.get("source_contract_sha256")
        if claimed_hash is not None and claimed_hash != contract.sha256:
            raise ValueError("DNa02 source contract hash mismatch")
        claimed_status = payload.get("status")
        if claimed_status is not None and claimed_status != contract.status:
            raise ValueError("DNa02 source contract status does not match blockers")
        claimed_blockers = payload.get("blockers")
        if claimed_blockers is not None and tuple(claimed_blockers) != contract.blockers:
            raise ValueError("DNa02 source contract blockers do not match resolved inputs")
        return contract


def load_contract(path: str | Path) -> DNa02SourceContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DNa02 source contract must be a JSON object")
    return DNa02SourceContract.from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Program A DNa02 source ingress")
    parser.add_argument("manifest")
    args = parser.parse_args(argv)
    contract = load_contract(args.manifest)
    print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    return 0 if contract.status == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
