from __future__ import annotations

import copy
import hashlib

import pytest

from fly_sniff.final_entropy import (
    FinalEntropyCommitment,
    FinalSeedReceipt,
    build_final_seed_receipt,
    derive_unique_seed_stream,
)


SECRET = b"s" * 32
OTHER_SECRET = b"t" * 32
LOCK_A = "a" * 64
LOCK_B = "b" * 64


def test_commitment_hides_secret_and_verifies_reveal() -> None:
    commitment = FinalEntropyCommitment.build(
        experiment_id="odor-zero-shot-v1",
        secret=SECRET,
    )
    payload = commitment.to_dict()

    assert SECRET.hex() not in str(payload)
    assert payload["commitment_sha256"] != hashlib.sha256(SECRET).hexdigest()
    commitment.verify_secret(SECRET)
    with pytest.raises(ValueError, match="does not match commitment"):
        commitment.verify_secret(OTHER_SECRET)


def test_seed_stream_is_deterministic_unique_and_lock_bound() -> None:
    first = derive_unique_seed_stream(
        secret=SECRET,
        lock_sha256=LOCK_A,
        namespace="heldout",
        count=128,
    )
    replay = derive_unique_seed_stream(
        secret=SECRET,
        lock_sha256=LOCK_A,
        namespace="heldout",
        count=128,
    )
    other_lock = derive_unique_seed_stream(
        secret=SECRET,
        lock_sha256=LOCK_B,
        namespace="heldout",
        count=128,
    )

    assert first == replay
    assert first != other_lock
    assert len(first) == len(set(first)) == 128
    assert all(1 <= seed <= 1_999_999_999 for seed in first)


def test_seed_stream_namespace_separates_heldout_and_ood() -> None:
    heldout = derive_unique_seed_stream(
        secret=SECRET,
        lock_sha256=LOCK_A,
        namespace="heldout",
        count=64,
    )
    ood = derive_unique_seed_stream(
        secret=SECRET,
        lock_sha256=LOCK_A,
        namespace="ood",
        count=64,
    )
    assert heldout != ood


def test_final_seed_receipt_verifies_exact_reveal() -> None:
    commitment = FinalEntropyCommitment.build(
        experiment_id="odor-zero-shot-v1",
        secret=SECRET,
    )
    receipt = build_final_seed_receipt(
        secret=SECRET,
        commitment=commitment,
        lock_sha256=LOCK_A,
        namespace="heldout",
        count=32,
    )
    receipt.verify_reveal(secret=SECRET, commitment=commitment)

    restored = FinalSeedReceipt.from_dict(receipt.to_dict())
    assert restored == receipt


def test_final_seed_receipt_rejects_wrong_secret_and_tampering() -> None:
    commitment = FinalEntropyCommitment.build(
        experiment_id="odor-zero-shot-v1",
        secret=SECRET,
    )
    receipt = build_final_seed_receipt(
        secret=SECRET,
        commitment=commitment,
        lock_sha256=LOCK_A,
        namespace="heldout",
        count=16,
    )

    with pytest.raises(ValueError, match="does not match commitment"):
        receipt.verify_reveal(secret=OTHER_SECRET, commitment=commitment)

    payload = copy.deepcopy(receipt.to_dict())
    payload["seeds"][0] = payload["seeds"][0] + 1
    with pytest.raises(ValueError, match="seed hash mismatch|receipt hash mismatch"):
        FinalSeedReceipt.from_dict(payload)


def test_final_entropy_rejects_short_secret_and_impossible_unique_range() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        FinalEntropyCommitment.build(experiment_id="x", secret=b"short")

    with pytest.raises(ValueError, match="too small"):
        derive_unique_seed_stream(
            secret=SECRET,
            lock_sha256=LOCK_A,
            namespace="tiny",
            count=4,
            min_seed=1,
            max_seed=3,
        )
