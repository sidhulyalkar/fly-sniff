from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from fly_sniff.dna02_source import (
    BLOCKED_FILE_MAP,
    READY,
    DataverseFileRef,
    DNa02SourceContract,
    load_contract,
)

AUTHORITY_PATH = Path("authority/program-a-dna02-source-contract-v1.json")
EXPECTED_COHORT = ("a2_d_08", "a2_d_12", "a2_d_13", "a2_d_14")
EXPECTED_SHA256 = {
    "a2_d_08": "f02a345effb2fe722800dd1f43dce2c876878424307cb1fa486b5974f3d5922b",
    "a2_d_12": "0f9a9251262f528f0b415d67cdb169bc1236a5157029ffeebace6d86136f60f7",
    "a2_d_13": "a460eb313b503b039b3eeec554efd902b54119aadfd783c72210efaee2daee24",
    "a2_d_14": "6e5aabc4bc45d76fd24baa81a9351cc6d29530c72e697038b78f07a0da36623f",
}


def _real_contract() -> DNa02SourceContract:
    return load_contract(AUTHORITY_PATH)


def _file(alias: str, index: int) -> DataverseFileRef:
    return DataverseFileRef(
        fly_alias=alias,
        file_id=1000 + index,
        filename=f"{alias}.mat",
        sha256=f"{index + 1:064x}",
    )


def _synthetic_ready_contract() -> DNa02SourceContract:
    contract = _real_contract()
    return dataclasses.replace(
        contract,
        data_file_map=tuple(_file(alias, index) for index, alias in enumerate(EXPECTED_COHORT)),
    )


def test_real_contract_is_ready_for_extraction_with_exact_sha_map() -> None:
    contract = _real_contract()
    contract.validate()

    assert contract.status == READY
    assert contract.blockers == ()
    assert contract.figure3c_cohort == EXPECTED_COHORT
    assert contract.figure3c_cohort_authority is not None
    assert tuple(ref.fly_alias for ref in contract.data_file_map) == EXPECTED_COHORT
    assert {ref.fly_alias: ref.sha256 for ref in contract.data_file_map} == EXPECTED_SHA256
    assert contract.navigation_performance_used is False


def test_real_contract_pins_version_of_record_and_dataverse() -> None:
    contract = _real_contract()
    assert contract.paper_doi == "10.7554/eLife.102230.3"
    assert contract.dataverse_doi == "10.7910/DVN/0NCLP1"
    assert contract.figure == "Figure 3C"
    assert contract.published_n_flies == 4


def test_code_authorities_are_immutable_and_include_primary_and_secondary_paths() -> None:
    contract = _real_contract()
    by_path = {authority.path: authority for authority in contract.code_authorities}

    primary = by_path["A2_dual_patch_analysis_v2.m"]
    assert primary.commit == "55e30c19b1a18f601df1803295f0c401aec3c167"
    assert primary.git_blob_sha1 == "3dd3d22135bf399d70dc3c41172eef965e1abbf7"

    secondary = by_path["import_preprocess_data.ipynb"]
    assert secondary.commit == "7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8"
    assert secondary.git_blob_sha1 == "0da2089b468c172f700881b714bfdda99a6fe424"


def test_candidate_aliases_equal_adjudicated_figure3c_cohort() -> None:
    contract = _real_contract()
    assert tuple(contract.candidate_bilateral_aliases) == EXPECTED_COHORT
    assert contract.figure3c_cohort == EXPECTED_COHORT


def test_cohort_resolution_requires_explicit_authority() -> None:
    contract = _real_contract()
    with pytest.raises(ValueError, match="requires explicit authority"):
        dataclasses.replace(contract, figure3c_cohort_authority=None).validate()


def test_cohort_must_match_published_n_four() -> None:
    contract = _real_contract()
    with pytest.raises(ValueError, match="exactly four"):
        dataclasses.replace(
            contract,
            figure3c_cohort=EXPECTED_COHORT[:3],
            figure3c_cohort_authority="synthetic authority",
            data_file_map=(),
        ).validate()


def test_removing_sha_file_map_returns_contract_to_blocked() -> None:
    contract = dataclasses.replace(_real_contract(), data_file_map=())
    contract.validate()
    assert contract.blockers == (BLOCKED_FILE_MAP,)
    assert contract.status == "BLOCKED"


def test_dataverse_map_may_not_be_frozen_before_cohort() -> None:
    contract = _real_contract()
    with pytest.raises(ValueError, match="before the Figure 3C cohort is resolved"):
        dataclasses.replace(
            contract,
            figure3c_cohort=None,
            figure3c_cohort_authority=None,
            data_file_map=(_file("a2_d_08", 0),),
        ).validate()


def test_incomplete_dataverse_map_stays_blocked() -> None:
    contract = _real_contract()
    partial = dataclasses.replace(
        contract,
        data_file_map=contract.data_file_map[:-1],
    )
    partial.validate()
    assert partial.blockers == (BLOCKED_FILE_MAP,)


def test_complete_hypothetical_contract_is_ready_for_extraction_only() -> None:
    resolved = _synthetic_ready_contract()
    resolved.validate()

    assert resolved.status == READY
    assert resolved.blockers == ()
    payload = resolved.to_dict()
    assert payload["status"] == READY
    assert payload["blockers"] == []
    assert payload["source_contract_sha256"] == resolved.sha256


def test_published_figure3c_preprocessing_is_frozen_without_making_lag_biology() -> None:
    preprocessing = dict(_real_contract().preprocessing)

    assert preprocessing["firing_rate_bin_ms"] == 10
    assert preprocessing["firing_rate_smoothing"] == "exponential"
    assert preprocessing["firing_rate_smoothing_window_ms"] == 30
    assert preprocessing["neural_to_behavior_alignment_ms"] == 150
    assert preprocessing["figure_average_bin_ms"] == 50
    assert preprocessing["primary_predictor"] == "right_firing_rate_hz - left_firing_rate_hz"
    interpretation = preprocessing["alignment_interpretation"].lower()
    assert "universal" in interpretation
    assert "treadmill inertia" in interpretation


def test_150_ms_alignment_cannot_be_rewritten_as_universal_delay() -> None:
    contract = _real_contract()
    preprocessing = dict(contract.preprocessing)
    preprocessing["alignment_interpretation"] = "universal biological delay"
    mutated = dataclasses.replace(
        contract,
        preprocessing=tuple(sorted(preprocessing.items())),
    )
    with pytest.raises(ValueError, match="reject universal-delay interpretation"):
        mutated.validate()


def test_navigation_performance_cannot_enter_source_selection() -> None:
    with pytest.raises(ValueError, match="may not use navigation performance"):
        dataclasses.replace(_real_contract(), navigation_performance_used=True).validate()


def test_dataverse_refs_require_content_hashes_and_unique_file_ids() -> None:
    contract = _real_contract()
    first = contract.data_file_map[0]
    invalid = dataclasses.replace(first, sha256="not-a-sha")
    with pytest.raises(ValueError, match="64-character"):
        dataclasses.replace(
            contract,
            data_file_map=(invalid,) + contract.data_file_map[1:],
        ).validate()

    duplicate_id = dataclasses.replace(contract.data_file_map[1], file_id=first.file_id)
    with pytest.raises(ValueError, match="duplicate Dataverse file_id"):
        dataclasses.replace(
            contract,
            data_file_map=(first, duplicate_id) + contract.data_file_map[2:],
        ).validate()


def test_contract_identity_changes_when_source_selection_changes() -> None:
    resolved = _real_contract()
    changed = dataclasses.replace(
        resolved,
        data_file_map=(
            dataclasses.replace(resolved.data_file_map[0], sha256="f" * 64),
            *resolved.data_file_map[1:],
        ),
    )
    assert changed.sha256 != resolved.sha256


def test_contract_roundtrip_rejects_tampered_hash() -> None:
    resolved = _real_contract()
    payload = resolved.to_dict()
    restored = DNa02SourceContract.from_dict(payload)
    assert restored.sha256 == resolved.sha256

    tampered = json.loads(json.dumps(payload))
    tampered["source_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        DNa02SourceContract.from_dict(tampered)


def test_source_fields_preserve_left_right_and_timebases() -> None:
    fields = dict(_real_contract().source_fields)
    assert fields["left_neuron"] == "ephys_A / a2_l"
    assert fields["right_neuron"] == "ephys_B / a2_r"
    assert fields["rotational_velocity"] == "yaw"
    assert fields["ephys_timebase"] == "t_ephys"
    assert fields["ball_timebase"] == "t_ball"


def test_all_four_raw_session_names_are_resolved() -> None:
    raw = dict(_real_contract().known_raw_session_candidates)
    assert raw == {
        "a2_d_08": ("180410_gfp_3G_ss730_dual_08",),
        "a2_d_12": ("180430_gfp_3G_ss730_dual_12",),
        "a2_d_13": ("180501_gfp_3G_ss730_dual_13",),
        "a2_d_14": ("180517_gfp_3G_ss730_dual_14",),
    }


def test_ready_for_extraction_does_not_mean_calibrated_or_navigation_ready() -> None:
    contract = _real_contract()
    boundary = " ".join(contract.forbidden_interpretation)
    assert "READY_FOR_EXTRACTION" in boundary
    assert "calibrated" in boundary
    assert "odor-navigation" in boundary
