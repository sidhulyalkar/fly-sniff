from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.odor_explorer import build_odor_explorer
from fly_sniff.olfactory_door import sha256_file
from fly_sniff.olfactory_e006_audit import _canonical_sha


def _fixture(root: Path) -> Path:
    root.mkdir()
    matrix = pd.DataFrame(
        {
            "OrA": [1.0, 0.2, -0.1],
            "OrB": [0.1, 0.9, 0.4],
            "OrC": [-0.2, 0.1, 1.2],
        },
        index=["1", "2", "3"],
    )
    matrix.index.name = "source_row_id"
    metadata = pd.DataFrame(
        {
            "source_row_id": ["1", "2", "3"],
            "odor_name": ["odor-a", "odor-b", "odor-c"],
            "odor_class": ["acid", "ester", "acid"],
        }
    ).set_index("source_row_id")

    matrix_path = root / "o002-complete-response-matrix.csv"
    metadata_path = root / "o002-odor-metadata.csv"
    matrix.to_csv(matrix_path)
    metadata.to_csv(metadata_path)

    receipt = {
        "protocol": "o002-within-study-development-v1",
        "development_only": True,
        "confirmatory_use_allowed": False,
        "selected_study": {"study_id": "Synthetic.OdorExplorer"},
        "outputs": {
            "matrix": {"sha256": sha256_file(matrix_path)},
            "odor_metadata": {"sha256": sha256_file(metadata_path)},
        },
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    (root / "o002-development-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return root


def test_odor_explorer_preserves_frozen_vectors(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "o002")
    payload = build_odor_explorer(root, tmp_path / "explorer.json")

    assert payload["study_id"] == "Synthetic.OdorExplorer"
    assert payload["development_only"] is True
    assert payload["confirmatory_use_allowed"] is False
    assert payload["responding_units"] == ["OrA", "OrB", "OrC"]
    assert len(payload["odors"]) == 3
    odor_a = next(row for row in payload["odors"] if row["odor_name"] == "odor-a")
    assert odor_a["responses"] == [1.0, 0.1, -0.2]
    assert "does not drive the movement replay" in payload["claim_boundary"]


def test_odor_explorer_refuses_tampered_matrix(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "o002")
    path = root / "o002-complete-response-matrix.csv"
    path.write_text(path.read_text() + "tampered\n")

    with pytest.raises(ValueError, match="matrix hash mismatch"):
        build_odor_explorer(root, tmp_path / "out.json")


def test_odor_explorer_refuses_promoted_receipt(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "o002")
    path = root / "o002-development-receipt.json"
    receipt = json.loads(path.read_text())
    receipt.pop("receipt_sha256")
    receipt["confirmatory_use_allowed"] = True
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    path.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="unexpectedly permits confirmatory use"):
        build_odor_explorer(root, tmp_path / "out.json")
