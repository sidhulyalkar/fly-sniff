from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "door-response-authority-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_authority(payload: dict[str, Any]) -> None:
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("unexpected odor authority protocol")
    receptors = payload.get("receptors")
    odorants = payload.get("odorants")
    if not isinstance(receptors, list) or not receptors:
        raise ValueError("odor authority requires a non-empty receptor list")
    if len(set(map(str, receptors))) != len(receptors):
        raise ValueError("receptor names must be unique")
    if not isinstance(odorants, dict) or len(odorants) < 2:
        raise ValueError("odor authority requires at least two odorants")
    for name, values in odorants.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape != (len(receptors),):
            raise ValueError(f"odor {name!r} response length does not match receptors")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"odor {name!r} contains non-finite responses")
        if np.min(arr) < 0.0 or np.max(arr) > 1.0:
            raise ValueError(f"odor {name!r} responses must be normalized to [0, 1]")
    source = payload.get("source", {})
    if not source.get("name") or not source.get("version"):
        raise ValueError("odor authority source name/version are required")


def build_authority_from_long_csv(
    csv_path: str | Path,
    *,
    source_name: str,
    source_version: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Build a compact receptor authority from receptor,odorant,response rows.

    This function intentionally performs no imputation. Missing receptor/odorant
    pairs remain zero only when the CSV explicitly contains a zero; otherwise the
    odorant is rejected as incomplete. Scientific experiments should therefore
    choose odorants with preregistered coverage rather than silently filling gaps.
    """
    rows: list[tuple[str, str, float]] = []
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"receptor", "odorant", "response"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain receptor, odorant, response columns")
        for row in reader:
            receptor = str(row["receptor"]).strip()
            odorant = str(row["odorant"]).strip()
            response = float(row["response"])
            if not receptor or not odorant or not np.isfinite(response):
                raise ValueError("invalid odor authority row")
            rows.append((receptor, odorant, response))

    receptors = sorted({row[0] for row in rows})
    odorant_names = sorted({row[1] for row in rows})
    matrix: dict[str, dict[str, float]] = {name: {} for name in odorant_names}
    for receptor, odorant, response in rows:
        if receptor in matrix[odorant]:
            raise ValueError(f"duplicate receptor/odorant pair: {receptor}, {odorant}")
        matrix[odorant][receptor] = response

    incomplete = [
        odorant
        for odorant, values in matrix.items()
        if set(values) != set(receptors)
    ]
    if incomplete:
        raise ValueError(
            "incomplete odorants are not silently imputed: " + ", ".join(incomplete[:8])
        )

    odorants = {
        odorant: [float(matrix[odorant][receptor]) for receptor in receptors]
        for odorant in odorant_names
    }
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "source": {
            "name": source_name,
            "version": source_version,
            "commit": source_commit,
            "input_sha256": sha256_file(csv_path),
        },
        "receptors": receptors,
        "odorants": odorants,
        "missing_value_policy": "reject-incomplete-odorants; no imputation",
        "claim_boundary": (
            "Normalized receptor-response authority only. Values are external sensory evidence, "
            "not measured activity from the MaleCNS simulation and not a navigation result."
        ),
    }
    validate_authority(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pinned odor receptor authority from long CSV")
    parser.add_argument("csv")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", default="authority/door-response-authority-v1.json")
    args = parser.parse_args()

    payload = build_authority_from_long_csv(
        args.csv,
        source_name=args.source_name,
        source_version=args.source_version,
        source_commit=args.source_commit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
