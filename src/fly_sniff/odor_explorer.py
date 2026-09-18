from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .olfactory_door import sha256_file
from .olfactory_e006_audit import _canonical_sha


def _validate_v1(root: Path) -> dict[str, Any]:
    receipt_path = root / "o002-development-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing O002 v1 receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("protocol") != "o002-within-study-development-v1":
        raise ValueError("named-odor explorer requires O002 v1 development receipt")
    observed = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if observed != _canonical_sha(unhashed):
        raise ValueError("O002 v1 canonical receipt hash mismatch")
    if receipt.get("development_only") is not True:
        raise ValueError("O002 v1 must remain development-only")
    if receipt.get("confirmatory_use_allowed") is not False:
        raise ValueError("O002 v1 unexpectedly permits confirmatory use")
    return receipt


def build_odor_explorer(
    o002_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(o002_dir).expanduser().resolve()
    receipt = _validate_v1(root)

    matrix_path = root / "o002-complete-response-matrix.csv"
    metadata_path = root / "o002-odor-metadata.csv"
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("O002 v1 matrix or metadata is missing")

    if sha256_file(matrix_path) != str(receipt["outputs"]["matrix"]["sha256"]):
        raise ValueError("O002 matrix hash mismatch")
    if sha256_file(metadata_path) != str(receipt["outputs"]["odor_metadata"]["sha256"]):
        raise ValueError("O002 metadata hash mismatch")

    matrix = pd.read_csv(matrix_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)
    matrix.index = matrix.index.astype(str)
    metadata.index = metadata.index.astype(str)
    metadata = metadata.reindex(matrix.index)

    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("O002 explorer matrix contains non-finite values")

    channel_min = values.min(axis=0)
    channel_max = values.max(axis=0)
    global_abs = float(max(abs(values.min()), abs(values.max()), 1e-12))

    odors: list[dict[str, Any]] = []
    for source_row_id, row in matrix.iterrows():
        meta = metadata.loc[source_row_id]
        odor_name = str(meta.get("odor_name", "") or "").strip()
        odor_class = str(meta.get("odor_class", "") or "").strip()
        if not odor_name:
            odor_name = f"source row {source_row_id}"
        vector = [round(float(x), 6) for x in row.to_numpy(dtype=float)]
        odors.append(
            {
                "source_row_id": str(source_row_id),
                "odor_name": odor_name,
                "odor_class": odor_class or "unlabeled",
                "responses": vector,
                "l2_norm": round(float(np.linalg.norm(row.to_numpy(dtype=float))), 6),
            }
        )

    odors.sort(key=lambda item: (item["odor_class"].casefold(), item["odor_name"].casefold()))

    payload: dict[str, Any] = {
        "schema": "fly-sniff-o002-odor-explorer-v1",
        "program_id": "olfactory-computation-v0",
        "study_id": str(receipt["selected_study"]["study_id"]),
        "development_only": True,
        "confirmatory_use_allowed": False,
        "source_receipt_sha256": receipt["receipt_sha256"],
        "matrix_sha256": receipt["outputs"]["matrix"]["sha256"],
        "metadata_sha256": receipt["outputs"]["odor_metadata"]["sha256"],
        "responding_units": [str(x) for x in matrix.columns],
        "channel_min": [round(float(x), 6) for x in channel_min],
        "channel_max": [round(float(x), 6) for x in channel_max],
        "global_abs_max": round(global_abs, 6),
        "odors": odors,
        "claim_boundary": (
            "This explorer displays measured within-study O002 response vectors from the frozen "
            "Hallem/DoOR development matrix. Selecting a named odor does not drive the movement replay. "
            "No named-odor behavioral response, valence, receptor-specific mechanism, or connectome "
            "causal claim is implied."
        ),
    }

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export frozen O002 named-odor response vectors for the live browser showcase"
    )
    parser.add_argument("o002_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = build_odor_explorer(args.o002_dir, args.output)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "study_id": payload["study_id"],
                "odors": len(payload["odors"]),
                "responding_units": len(payload["responding_units"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
