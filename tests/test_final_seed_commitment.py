from __future__ import annotations

import pytest

from fly_sniff.final_seed_commitment import (
    create_commitment,
    derive_seed_stream,
    reveal_seed_streams,
    verify_commitment,
)


def test_commitment_hides_secret_and_reveal_is_deterministic() -> None:
    secret = bytes(range(32))
    commitment = create_commitment(secret)
    assert "secret" not in commitment
    assert commitment["secret_bytes"] == 32

    lock_sha = "11" * 32
    first = derive_seed_stream(
        secret,
        experiment_lock_sha256=lock_sha,
        stream_name="heldout",
        count=64,
    )
    second = derive_seed_stream(
        secret,
        experiment_lock_sha256=lock_sha,
        stream_name="heldout",
        count=64,
    )
    assert first == second
    assert len(first) == 64
    assert len(first) == len(set(first))

    reveal = reveal_seed_streams(
        commitment,
        secret,
        experiment_lock_sha256=lock_sha,
        streams={"heldout": 64, "topology": 63},
    )
    assert reveal["streams"]["heldout"] == first
    assert len(reveal["streams"]["topology"]) == 63


def test_wrong_reveal_secret_is_rejected() -> None:
    commitment = create_commitment(b"a" * 32)
    with pytest.raises(ValueError, match="does not match published commitment"):
        verify_commitment(commitment, b"b" * 32)


def test_seed_stream_is_bound_to_experiment_lock() -> None:
    secret = b"c" * 32
    a = derive_seed_stream(
        secret,
        experiment_lock_sha256="22" * 32,
        stream_name="heldout",
        count=16,
    )
    b = derive_seed_stream(
        secret,
        experiment_lock_sha256="33" * 32,
        stream_name="heldout",
        count=16,
    )
    assert a != b
