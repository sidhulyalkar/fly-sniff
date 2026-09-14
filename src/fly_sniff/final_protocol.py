from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_io import file_sha256, load_json
from .experiment import ExperimentLock
from .final_entropy import (
    FinalEntropyCommitment,
    FinalSeedReceipt,
    build_final_seed_receipt,
)


def load_commitment(path: str | Path) -> FinalEntropyCommitment:
    return FinalEntropyCommitment.from_dict(load_json(path))


def load_lock(path: str | Path) -> ExperimentLock:
    return ExperimentLock.from_dict(load_json(path))


def verify_commitment_binding(
    *,
    lock: ExperimentLock,
    commitment: FinalEntropyCommitment,
    commitment_file_sha256: str,
) -> None:
    """Require the pre-reveal commitment to be an exact input to the sealed lock."""

    lock.validate()
    commitment.validate()
    if commitment.experiment_id != lock.spec.experiment_id:
        raise ValueError("final entropy commitment experiment_id does not match experiment lock")

    refs = [
        artifact
        for artifact in lock.spec.artifacts
        if artifact.kind == "final_entropy_commitment"
    ]
    if len(refs) != 1:
        raise ValueError(
            "experiment lock must bind exactly one final_entropy_commitment artifact"
        )
    if refs[0].sha256 != commitment_file_sha256:
        raise ValueError("experiment lock does not bind the exact commitment file bytes")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_secret(path: str | Path) -> bytes:
    secret = Path(path).read_bytes()
    if len(secret) < 32:
        raise ValueError("final entropy secret must contain at least 32 bytes")
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
                "experiment_id": commitment.experiment_id,
                "commitment_sha256": commitment.commitment_sha256,
                "scientific_sha256": commitment.sha256,
                "file_sha256": file_sha256(output),
                "secret_emitted": False,
                "next_step": (
                    "bind this exact file_sha256 as an ArtifactRef with "
                    "kind=final_entropy_commitment before creating ExperimentLock"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _derive_main(args: argparse.Namespace) -> int:
    secret = _read_secret(args.secret_file)
    commitment_path = Path(args.commitment)
    commitment = load_commitment(commitment_path)
    lock = load_lock(args.lock)
    verify_commitment_binding(
        lock=lock,
        commitment=commitment,
        commitment_file_sha256=file_sha256(commitment_path),
    )
    receipt = build_final_seed_receipt(
        secret=secret,
        commitment=commitment,
        lock_sha256=lock.sha256,
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
                "status": "derived_lock_bound_final_seed_receipt",
                "output": str(output),
                "lock_sha256": lock.sha256,
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
    commitment_path = Path(args.commitment)
    commitment = load_commitment(commitment_path)
    lock = load_lock(args.lock)
    verify_commitment_binding(
        lock=lock,
        commitment=commitment,
        commitment_file_sha256=file_sha256(commitment_path),
    )
    receipt = FinalSeedReceipt.from_dict(load_json(args.seed_receipt))
    if receipt.lock_sha256 != lock.sha256:
        raise ValueError("final seed receipt is not bound to the supplied experiment lock")
    receipt.verify_reveal(secret=secret, commitment=commitment)
    print(
        json.dumps(
            {
                "status": "verified_lock_bound_final_entropy_reveal",
                "lock_sha256": lock.sha256,
                "receipt_sha256": receipt.sha256,
                "seeds_sha256": receipt.seeds_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Commit hidden final entropy before freeze, then derive seeds only from a "
            "commitment that is exact-file-hash-bound inside ExperimentLock."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commit = subparsers.add_parser(
        "commit",
        help="Create the public pre-freeze commitment without emitting the secret",
    )
    commit.add_argument("--secret-file", required=True)
    commit.add_argument("--experiment-id", required=True)
    commit.add_argument("--output", required=True)
    commit.set_defaults(func=_commit_main)

    derive = subparsers.add_parser(
        "derive",
        help="After freeze/reveal, derive final seeds from the exact bound commitment",
    )
    derive.add_argument("--secret-file", required=True)
    derive.add_argument("--commitment", required=True)
    derive.add_argument("--lock", required=True)
    derive.add_argument("--namespace", default="final")
    derive.add_argument("--count", type=int, required=True)
    derive.add_argument("--min-seed", type=int, default=1)
    derive.add_argument("--max-seed", type=int, default=1_999_999_999)
    derive.add_argument("--output", required=True)
    derive.set_defaults(func=_derive_main)

    verify = subparsers.add_parser(
        "verify",
        help="Verify commitment binding, revealed entropy, and exact derived seeds",
    )
    verify.add_argument("--secret-file", required=True)
    verify.add_argument("--commitment", required=True)
    verify.add_argument("--lock", required=True)
    verify.add_argument("--seed-receipt", required=True)
    verify.set_defaults(func=_verify_main)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
