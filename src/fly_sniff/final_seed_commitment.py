from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any

COMMIT_PROTOCOL = "fly-sniff-final-seed-commitment-v1"
REVEAL_PROTOCOL = "fly-sniff-final-seed-reveal-v1"
INFO_PREFIX = b"fly-sniff-final-seeds-v1:"
MAX_STREAM_SEEDS = 2000


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")


def create_commitment(secret: bytes) -> dict[str, Any]:
    if len(secret) < 32:
        raise ValueError("final-seed secret must contain at least 256 bits of entropy")
    return {
        "protocol": COMMIT_PROTOCOL,
        "hash": "sha256",
        "secret_bytes": len(secret),
        "commitment_sha256": hashlib.sha256(secret).hexdigest(),
        "claim_boundary": (
            "Publishing this artifact commits to hidden entropy only. It reveals no final seeds "
            "and does not prove that model freeze predates the later reveal unless repository or "
            "external archive timestamps establish that ordering."
        ),
    }


def verify_commitment(commitment: dict[str, Any], secret: bytes) -> None:
    if commitment.get("protocol") != COMMIT_PROTOCOL:
        raise ValueError("unexpected final-seed commitment protocol")
    expected = str(commitment["commitment_sha256"])
    _validate_sha256(expected, field="commitment_sha256")
    actual = hashlib.sha256(secret).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ValueError("revealed secret does not match published commitment")


def _hkdf_expand(secret: bytes, *, salt: bytes, info: bytes, length: int) -> bytes:
    if length < 0 or length > 255 * hashlib.sha256().digest_size:
        raise ValueError("requested HKDF output length is unsupported")
    prk = hmac.new(salt, secret, hashlib.sha256).digest()
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(
            prk,
            previous + info + bytes([counter]),
            hashlib.sha256,
        ).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def derive_seed_stream(
    secret: bytes,
    *,
    experiment_lock_sha256: str,
    stream_name: str,
    count: int,
) -> list[int]:
    _validate_sha256(experiment_lock_sha256, field="experiment_lock_sha256")
    if not stream_name.strip():
        raise ValueError("stream_name must be non-empty")
    if count <= 0 or count > MAX_STREAM_SEEDS:
        raise ValueError(f"count must be in [1, {MAX_STREAM_SEEDS}]")
    raw = _hkdf_expand(
        secret,
        salt=bytes.fromhex(experiment_lock_sha256),
        info=INFO_PREFIX + stream_name.encode(),
        length=count * 4,
    )
    seeds = [
        1 + int.from_bytes(raw[offset : offset + 4], "big") % 2_000_000_000
        for offset in range(0, len(raw), 4)
    ]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(
            "derived seed stream contains a collision; use a different committed secret rather "
            "than silently editing the stream"
        )
    return seeds


def reveal_seed_streams(
    commitment: dict[str, Any],
    secret: bytes,
    *,
    experiment_lock_sha256: str,
    streams: dict[str, int],
) -> dict[str, Any]:
    verify_commitment(commitment, secret)
    if not streams:
        raise ValueError("at least one seed stream is required")
    derived = {
        name: derive_seed_stream(
            secret,
            experiment_lock_sha256=experiment_lock_sha256,
            stream_name=name,
            count=int(count),
        )
        for name, count in sorted(streams.items())
    }
    return {
        "protocol": REVEAL_PROTOCOL,
        "commitment_sha256": commitment["commitment_sha256"],
        "experiment_lock_sha256": experiment_lock_sha256,
        "revealed_secret_hex": secret.hex(),
        "streams": derived,
        "derivation": "HKDF-SHA256(secret, salt=experiment_lock_sha256, info=stream-name)",
        "claim_boundary": (
            "The reveal proves that these seed streams derive from the previously committed "
            "secret and the exact experiment lock. Chronological blinding additionally depends "
            "on the commitment being published before the model/experiment freeze."
        ),
    }


def _parse_streams(values: list[str]) -> dict[str, int]:
    streams: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("stream arguments must use NAME=COUNT")
        name, raw_count = value.split("=", 1)
        if name in streams:
            raise ValueError(f"duplicate stream name: {name}")
        streams[name] = int(raw_count)
    return streams


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit or reveal hidden final-evaluation entropy")
    sub = parser.add_subparsers(dest="command", required=True)

    commit = sub.add_parser("commit")
    commit.add_argument("--commitment-out", default="commitments/final-seed-v1.json")
    commit.add_argument("--secret-out", default=".private/final-seed-v1.hex")

    reveal = sub.add_parser("reveal")
    reveal.add_argument("--commitment", required=True)
    reveal.add_argument("--secret", required=True)
    reveal.add_argument("--lock-sha256", required=True)
    reveal.add_argument("--stream", action="append", default=[], help="NAME=COUNT")
    reveal.add_argument("--output", default="manifests/final-seeds-v1.json")

    args = parser.parse_args()
    if args.command == "commit":
        secret = secrets.token_bytes(32)
        commitment = create_commitment(secret)
        commitment_path = Path(args.commitment_out)
        secret_path = Path(args.secret_out)
        if commitment_path.exists() or secret_path.exists():
            raise FileExistsError("refusing to overwrite existing commitment or secret")
        commitment_path.parent.mkdir(parents=True, exist_ok=True)
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        commitment_path.write_text(json.dumps(commitment, indent=2, sort_keys=True) + "\n")
        secret_path.write_text(secret.hex() + "\n")
        print(f"commitment={commitment_path}")
        print(f"secret={secret_path} (private; do not commit before reveal)")
        print(f"sha256={commitment['commitment_sha256']}")
        return

    commitment = json.loads(Path(args.commitment).read_text())
    secret = bytes.fromhex(Path(args.secret).read_text().strip())
    payload = reveal_seed_streams(
        commitment,
        secret,
        experiment_lock_sha256=args.lock_sha256,
        streams=_parse_streams(args.stream),
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final seed reveal: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"reveal={output}")
    print("streams=" + ",".join(f"{name}:{len(values)}" for name, values in payload["streams"].items()))


if __name__ == "__main__":
    main()
