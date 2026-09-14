from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_io import file_sha256, load_json
from .freeze import canonical_sha256


_COMMIT_DOMAIN = b"fly-sniff-final-entropy-commitment-v1\x00"
_SEED_DOMAIN = b"fly-sniff-final-seed-stream-v1\x00"
_MIN_SECRET_BYTES = 32


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")


def _validate_secret(secret: bytes) -> None:
    if len(secret) < _MIN_SECRET_BYTES:
        raise ValueError(f"final entropy secret must contain at least {_MIN_SECRET_BYTES} bytes")


def secret_commitment_sha256(secret: bytes) -> str:
    """Commit to hidden entropy without exposing the entropy itself."""

    _validate_secret(secret)
    return hashlib.sha256(_COMMIT_DOMAIN + secret).hexdigest()


@dataclass(frozen=True)
class FinalEntropyCommitment:
    """Public artifact created before the model/final experiment is frozen."""

    experiment_id: str
    commitment_sha256: str
    algorithm: str = "sha256-domain-separated-secret-v1"
    schema: str = "fly-sniff-final-entropy-commitment-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-final-entropy-commitment-v1":
            raise ValueError(f"unsupported final entropy commitment schema: {self.schema}")
        if not self.experiment_id.strip():
            raise ValueError("final entropy commitment experiment_id must be non-empty")
        if self.algorithm != "sha256-domain-separated-secret-v1":
            raise ValueError(f"unsupported final entropy commitment algorithm: {self.algorithm}")
        _validate_sha256(self.commitment_sha256, field="commitment_sha256")

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "commitment_sha256": self.commitment_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["artifact_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FinalEntropyCommitment:
        commitment = cls(
            experiment_id=str(payload["experiment_id"]),
            commitment_sha256=str(payload["commitment_sha256"]),
            algorithm=str(payload.get("algorithm", "sha256-domain-separated-secret-v1")),
            schema=str(payload.get("schema", "fly-sniff-final-entropy-commitment-v1")),
        )
        commitment.validate()
        claimed_hash = payload.get("artifact_sha256")
        if claimed_hash is not None and claimed_hash != commitment.sha256:
            raise ValueError("final entropy commitment artifact hash mismatch")
        return commitment

    @classmethod
    def build(cls, *, experiment_id: str, secret: bytes) -> FinalEntropyCommitment:
        commitment = cls(
            experiment_id=experiment_id,
            commitment_sha256=secret_commitment_sha256(secret),
        )
        commitment.validate()
        return commitment

    def verify_secret(self, secret: bytes) -> None:
        actual = secret_commitment_sha256(secret)
        if not hmac.compare_digest(actual, self.commitment_sha256):
            raise ValueError("revealed final entropy secret does not match commitment")


def _uniform_candidate(
    *,
    secret: bytes,
    lock_sha256: str,
    namespace: str,
    counter: int,
    min_seed: int,
    max_seed: int,
) -> int | None:
    """Derive an unbiased candidate seed with deterministic rejection sampling."""

    width = max_seed - min_seed + 1
    if width <= 0:
        raise ValueError("max_seed must be >= min_seed")
    data = (
        _SEED_DOMAIN
        + bytes.fromhex(lock_sha256)
        + b"\x00"
        + namespace.encode("utf-8")
        + b"\x00"
        + counter.to_bytes(8, "big", signed=False)
    )
    value = int.from_bytes(hmac.new(secret, data, hashlib.sha256).digest()[:8], "big")
    space = 1 << 64
    limit = space - (space % width)
    if value >= limit:
        return None
    return min_seed + (value % width)


def derive_unique_seed_stream(
    *,
    secret: bytes,
    lock_sha256: str,
    namespace: str,
    count: int,
    min_seed: int = 1,
    max_seed: int = 1_999_999_999,
) -> tuple[int, ...]:
    """Derive a deterministic, unique seed list bound to an exact experiment lock."""

    _validate_secret(secret)
    _validate_sha256(lock_sha256, field="lock_sha256")
    if not namespace.strip():
        raise ValueError("seed namespace must be non-empty")
    if count <= 0:
        raise ValueError("seed count must be positive")
    width = max_seed - min_seed + 1
    if width < count:
        raise ValueError("seed range is too small for requested unique seed count")

    seeds: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(seeds) < count:
        candidate = _uniform_candidate(
            secret=secret,
            lock_sha256=lock_sha256,
            namespace=namespace,
            counter=counter,
            min_seed=min_seed,
            max_seed=max_seed,
        )
        counter += 1
        if candidate is None or candidate in seen:
            continue
        seeds.append(candidate)
        seen.add(candidate)
    return tuple(seeds)


@dataclass(frozen=True)
class FinalSeedReceipt:
    """Post-reveal receipt proving how final seeds were derived."""

    experiment_id: str
    commitment_sha256: str
    lock_sha256: str
    namespace: str
    min_seed: int
    max_seed: int
    seeds: tuple[int, ...]
    reveal_fingerprint_sha256: str
    algorithm: str = "hmac-sha256-lock-bound-unique-seeds-v1"
    schema: str = "fly-sniff-final-seed-receipt-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-final-seed-receipt-v1":
            raise ValueError(f"unsupported final seed receipt schema: {self.schema}")
        if not self.experiment_id.strip():
            raise ValueError("final seed receipt experiment_id must be non-empty")
        if not self.namespace.strip():
            raise ValueError("final seed receipt namespace must be non-empty")
        if self.algorithm != "hmac-sha256-lock-bound-unique-seeds-v1":
            raise ValueError(f"unsupported final seed receipt algorithm: {self.algorithm}")
        _validate_sha256(self.commitment_sha256, field="commitment_sha256")
        _validate_sha256(self.lock_sha256, field="lock_sha256")
        _validate_sha256(self.reveal_fingerprint_sha256, field="reveal_fingerprint_sha256")
        if self.max_seed < self.min_seed:
            raise ValueError("max_seed must be >= min_seed")
        if not self.seeds:
            raise ValueError("final seed receipt must contain at least one seed")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("final seed receipt contains duplicate seeds")
        if any(seed < self.min_seed or seed > self.max_seed for seed in self.seeds):
            raise ValueError("final seed receipt contains an out-of-range seed")

    @property
    def seeds_sha256(self) -> str:
        return canonical_sha256(list(self.seeds))

    def _payload_without_hash(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "commitment_sha256": self.commitment_sha256,
            "lock_sha256": self.lock_sha256,
            "namespace": self.namespace,
            "min_seed": self.min_seed,
            "max_seed": self.max_seed,
            "count": len(self.seeds),
            "seeds": list(self.seeds),
            "seeds_sha256": self.seeds_sha256,
            "reveal_fingerprint_sha256": self.reveal_fingerprint_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["receipt_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FinalSeedReceipt:
        receipt = cls(
            experiment_id=str(payload["experiment_id"]),
            commitment_sha256=str(payload["commitment_sha256"]),
            lock_sha256=str(payload["lock_sha256"]),
            namespace=str(payload["namespace"]),
            min_seed=int(payload["min_seed"]),
            max_seed=int(payload["max_seed"]),
            seeds=tuple(int(seed) for seed in payload["seeds"]),
            reveal_fingerprint_sha256=str(payload["reveal_fingerprint_sha256"]),
            algorithm=str(payload.get("algorithm", "hmac-sha256-lock-bound-unique-seeds-v1")),
            schema=str(payload.get("schema", "fly-sniff-final-seed-receipt-v1")),
        )
        receipt.validate()
        if int(payload.get("count", len(receipt.seeds))) != len(receipt.seeds):
            raise ValueError("final seed receipt count mismatch")
        if payload.get("seeds_sha256") not in {None, receipt.seeds_sha256}:
            raise ValueError("final seed receipt seed hash mismatch")
        claimed_hash = payload.get("receipt_sha256")
        if claimed_hash is not None and claimed_hash != receipt.sha256:
            raise ValueError("final seed receipt hash mismatch")
        return receipt

    def verify_reveal(self, *, secret: bytes, commitment: FinalEntropyCommitment) -> None:
        commitment.verify_secret(secret)
        if commitment.experiment_id != self.experiment_id:
            raise ValueError("commitment and seed receipt experiment IDs differ")
        if commitment.commitment_sha256 != self.commitment_sha256:
            raise ValueError("commitment digest does not match seed receipt")
        expected_fingerprint = hashlib.sha256(secret).hexdigest()
        if not hmac.compare_digest(expected_fingerprint, self.reveal_fingerprint_sha256):
            raise ValueError("reveal fingerprint does not match supplied secret")
        expected = derive_unique_seed_stream(
            secret=secret,
            lock_sha256=self.lock_sha256,
            namespace=self.namespace,
            count=len(self.seeds),
            min_seed=self.min_seed,
            max_seed=self.max_seed,
        )
        if expected != self.seeds:
            raise ValueError("final seed receipt does not match deterministic derivation")


def build_final_seed_receipt(
    *,
    secret: bytes,
    commitment: FinalEntropyCommitment,
    lock_sha256: str,
    namespace: str,
    count: int,
    min_seed: int = 1,
    max_seed: int = 1_999_999_999,
) -> FinalSeedReceipt:
    commitment.verify_secret(secret)
    seeds = derive_unique_seed_stream(
        secret=secret,
        lock_sha256=lock_sha256,
        namespace=namespace,
        count=count,
        min_seed=min_seed,
        max_seed=max_seed,
    )
    receipt = FinalSeedReceipt(
        experiment_id=commitment.experiment_id,
        commitment_sha256=commitment.commitment_sha256,
        lock_sha256=lock_sha256,
        namespace=namespace,
        min_seed=min_seed,
        max_seed=max_seed,
        seeds=seeds,
        reveal_fingerprint_sha256=hashlib.sha256(secret).hexdigest(),
    )
    receipt.validate()
    return receipt


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_secret(path: str) -> bytes:
    secret = Path(path).read_bytes()
    _validate_secret(secret)
    return secret


def _commit_main(args: argparse.Namespace) -> int:
    secret = _read_secret(args.secret_file)
    commitment = FinalEntropyCommitment.build(
        experiment_id=args.experiment_id,
        secret=secret,
    )
    output = Path(args.output)
    _write_json(output, commitment.to_dict())
    print(
        json.dumps(
            {
                "status": "committed_hidden_final_entropy",
                "output": str(output),
                "commitment_sha256": commitment.commitment_sha256,
                "artifact_sha256": commitment.sha256,
                "file_sha256": file_sha256(output),
                "secret_emitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _load_commitment(path: str) -> FinalEntropyCommitment:
    return FinalEntropyCommitment.from_dict(load_json(path))


def _derive_main(args: argparse.Namespace) -> int:
    secret = _read_secret(args.secret_file)
    commitment = _load_commitment(args.commitment)
    lock_payload = load_json(args.lock)
    if lock_payload.get("schema") != "fly-sniff-experiment-lock-v1":
        raise ValueError("--lock must be a fly-sniff-experiment-lock-v1 artifact")
    lock_sha256 = str(lock_payload.get("lock_sha256", ""))
    _validate_sha256(lock_sha256, field="lock_sha256")
    receipt = build_final_seed_receipt(
        secret=secret,
        commitment=commitment,
        lock_sha256=lock_sha256,
        namespace=args.namespace,
        count=args.count,
        min_seed=args.min_seed,
        max_seed=args.max_seed,
    )
    output = Path(args.output)
    _write_json(output, receipt.to_dict())
    print(
        json.dumps(
            {
                "status": "derived_final_seed_receipt",
                "output": str(output),
                "count": len(receipt.seeds),
                "seeds_sha256": receipt.seeds_sha256,
                "receipt_sha256": receipt.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_main(args: argparse.Namespace) -> int:
    secret = _read_secret(args.secret_file)
    commitment = _load_commitment(args.commitment)
    receipt = FinalSeedReceipt.from_dict(load_json(args.seed_receipt))
    receipt.verify_reveal(secret=secret, commitment=commitment)
    print(
        json.dumps(
            {
                "status": "verified_final_entropy_reveal",
                "receipt_sha256": receipt.sha256,
                "seeds_sha256": receipt.seeds_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Commit hidden final entropy and derive lock-bound seeds only after reveal."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commit = subparsers.add_parser("commit", help="Create a public commitment without emitting the secret")
    commit.add_argument("--secret-file", required=True)
    commit.add_argument("--experiment-id", required=True)
    commit.add_argument("--output", required=True)
    commit.set_defaults(func=_commit_main)

    derive = subparsers.add_parser("derive", help="Derive final seeds after model/experiment lock")
    derive.add_argument("--secret-file", required=True)
    derive.add_argument("--commitment", required=True)
    derive.add_argument("--lock", required=True)
    derive.add_argument("--namespace", default="final")
    derive.add_argument("--count", type=int, required=True)
    derive.add_argument("--min-seed", type=int, default=1)
    derive.add_argument("--max-seed", type=int, default=1_999_999_999)
    derive.add_argument("--output", required=True)
    derive.set_defaults(func=_derive_main)

    verify = subparsers.add_parser("verify", help="Verify a reveal and exact derived seed receipt")
    verify.add_argument("--secret-file", required=True)
    verify.add_argument("--commitment", required=True)
    verify.add_argument("--seed-receipt", required=True)
    verify.set_defaults(func=_verify_main)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
