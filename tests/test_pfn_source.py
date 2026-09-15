from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fly_sniff.pfn_source import (
    BLOCKED_MEMBER_MAP,
    BLOCKED_RECORDINGS,
    READY,
    ArchiveMemberRef,
    RecordingRef,
    bootstrap_circular_mean_ci_deg,
    circular_mean_deg,
    fold_angle_to_ipsilateral_deg,
    load_contract,
    mean_response_vector_angle_deg,
)

AUTHORITY_PATH = Path("authority/program-a-pfn-source-contract-v1.json")
DIRECTIONS = (-135, -90, -45, 0, 45, 90, 135, 180)
README_SHA256 = "6313f97ff216f29e5457502f80524edbc3ec0801dc0cb3b7fd57d08cebc31692"


def _real_contract():
    return load_contract(AUTHORITY_PATH)


def _resolved_contract():
    contract = _real_contract()
    members = tuple(
        ArchiveMemberRef(
            member_path=f"figure4/recording_{index:02d}.mat",
            container_files=("Currier2020.zip.001",),
            sha256=f"{index + 1:064x}",
        )
        for index in range(12)
    )
    recordings = tuple(
        RecordingRef(
            recording_id=f"figure4-recording-{index:02d}",
            soma_side="R" if index == 11 else "L",
            archive_member_path=members[index].member_path,
        )
        for index in range(12)
    )
    return dataclasses.replace(
        contract,
        archive_member_map=members,
        recording_set=recordings,
    )


def test_real_contract_is_blocked_only_on_member_and_recording_maps() -> None:
    contract = _real_contract()
    contract.validate()
    assert contract.status == "BLOCKED"
    assert contract.blockers == (BLOCKED_MEMBER_MAP, BLOCKED_RECORDINGS)
    assert contract.readme_sha256 == README_SHA256
    assert contract.navigation_performance_used is False


def test_public_repository_identity_and_archive_inventory_are_frozen() -> None:
    contract = _real_contract()
    assert contract.paper_doi == "10.7554/eLife.61510"
    assert contract.dryad_doi == "10.5061/dryad.vq83bk3rh"
    assert contract.readme_filename == "Currier2020README.rtf"
    assert contract.readme_file_stream_id == 536042
    assert contract.readme_md5 == "91e5213503788fcde11c0f5aa3e91f43"
    assert contract.readme_sha256 == README_SHA256
    assert len(contract.archive_inventory) == 14
    assert contract.archive_inventory[0][0] == "Currier2020.z01"
    assert contract.archive_inventory[-1][0] == "Currier2020.zip.013"


def test_published_figure4_protocol_is_frozen() -> None:
    contract = _real_contract()
    assert contract.directions_deg == DIRECTIONS
    assert contract.airflow_speed_cm_per_s == 25
    assert contract.repetitions_per_direction == 5
    assert contract.trials_per_recording == 40
    assert contract.baseline_window_s == (-1.5, -0.5)
    assert contract.response_window_s == (0.5, 1.5)
    assert contract.negative_responses_are_clipped is False


def test_signed_response_vector_recovers_known_preferred_direction() -> None:
    directions = np.asarray(DIRECTIONS, dtype=np.float64)
    responses = np.cos(np.deg2rad(directions - 45.0))
    preferred = mean_response_vector_angle_deg(directions, responses)
    assert preferred == pytest.approx(45.0)
    assert np.any(responses < 0)


def test_negative_responses_cannot_be_reclassified_as_clipped() -> None:
    with pytest.raises(ValueError, match="may not be clipped"):
        dataclasses.replace(_real_contract(), negative_responses_are_clipped=True).validate()


def test_soma_side_fold_makes_both_hemispheres_ipsilateral_positive() -> None:
    assert fold_angle_to_ipsilateral_deg(45.0, soma_side="L") == pytest.approx(45.0)
    assert fold_angle_to_ipsilateral_deg(-45.0, soma_side="R") == pytest.approx(45.0)
    with pytest.raises(ValueError, match="L or R"):
        fold_angle_to_ipsilateral_deg(45.0, soma_side="unknown")


def test_direction_order_or_set_cannot_drift() -> None:
    contract = _real_contract()
    with pytest.raises(ValueError, match="published eight-direction set"):
        dataclasses.replace(contract, directions_deg=(-180, -135, -90, -45, 0, 45, 90, 135)).validate()

    responses = np.ones(8)
    with pytest.raises(ValueError, match="published eight-direction set"):
        mean_response_vector_angle_deg((-180, -135, -90, -45, 0, 45, 90, 135), responses)


def test_verified_readme_does_not_promote_member_or_recording_maps() -> None:
    contract = _real_contract()
    assert contract.readme_sha256 == README_SHA256
    assert contract.blockers == (BLOCKED_MEMBER_MAP, BLOCKED_RECORDINGS)
    assert contract.status == "BLOCKED"


def test_recording_set_cannot_reference_unresolved_archive_member() -> None:
    contract = _real_contract()
    recording = RecordingRef(
        recording_id="figure4-recording-00",
        soma_side="L",
        archive_member_path="missing.mat",
    )
    with pytest.raises(ValueError, match="unresolved archive member"):
        dataclasses.replace(contract, recording_set=(recording,)).validate()


def test_archive_members_require_content_hashes() -> None:
    member = ArchiveMemberRef(
        member_path="figure4/recording.mat",
        container_files=("Currier2020.zip.001",),
        sha256="not-a-sha",
    )
    with pytest.raises(ValueError, match="64-character"):
        member.validate()


def test_complete_hypothetical_contract_is_ready_for_extraction_only() -> None:
    contract = _resolved_contract()
    contract.validate()
    assert contract.status == READY
    assert contract.blockers == ()
    assert len(contract.recording_set) == 12
    assert len(contract.archive_member_map) == 12


def test_wrong_recording_count_cannot_be_promoted() -> None:
    contract = _resolved_contract()
    with pytest.raises(ValueError, match="exactly 12 recordings"):
        dataclasses.replace(contract, recording_set=contract.recording_set[:-1]).validate()


def test_navigation_performance_cannot_enter_pfn_source_selection() -> None:
    with pytest.raises(ValueError, match="may not use navigation performance"):
        dataclasses.replace(_real_contract(), navigation_performance_used=True).validate()


def test_absolute_gain_and_temporal_shape_remain_outside_v1_transfer() -> None:
    with pytest.raises(ValueError, match="geometry, not absolute gain or dynamics"):
        dataclasses.replace(_real_contract(), absolute_gain_transferred=True).validate()
    with pytest.raises(ValueError, match="geometry, not absolute gain or dynamics"):
        dataclasses.replace(_real_contract(), temporal_shape_transferred=True).validate()


def test_circular_aggregation_and_bootstrap_are_deterministic() -> None:
    angles = np.asarray([38.0, 42.0, 44.0, 45.0, 47.0, 49.0, 51.0, 41.0, 46.0, 48.0, 43.0, 45.0])
    center = circular_mean_deg(angles)
    first = bootstrap_circular_mean_ci_deg(angles)
    second = bootstrap_circular_mean_ci_deg(angles)
    assert first == pytest.approx(second)
    assert first[0] == pytest.approx(center)
    assert first[1] < first[0] < first[2]

    with pytest.raises(ValueError, match="frozen to 10000"):
        bootstrap_circular_mean_ci_deg(angles, resamples=1000)


def test_contract_identity_changes_when_source_bytes_change() -> None:
    contract = _resolved_contract()
    changed_member = dataclasses.replace(contract.archive_member_map[0], sha256="f" * 64)
    changed = dataclasses.replace(
        contract,
        archive_member_map=(changed_member,) + contract.archive_member_map[1:],
    )
    assert changed.sha256 != contract.sha256


def test_zero_resultant_vector_fails_closed() -> None:
    with pytest.raises(ValueError, match="undefined"):
        mean_response_vector_angle_deg(DIRECTIONS, np.zeros(8))
