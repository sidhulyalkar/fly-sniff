from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fly_sniff.olfactory_door import sha256_file
from fly_sniff.olfactory_e006_audit import _canonical_sha
from fly_sniff.olfactory_o002_robustness import (
    _row_l2_normalize,
    run_o002_robustness,
)


def _write_v1_fixture(root: Path) -> Path:
    root.mkdir()
    units = [f"Or{i}" for i in range(8)]
    rows = []
    meta = []
    source_row = 1
    for label, active in (("class-a", 0), ("class-b", 1)):
        for replicate in range(6):
            vector = np.zeros(8, dtype=float)
            vector[active] = 10.0 + replicate * 0.1
            vector[2:] = np.arange(6, dtype=float) * 0.01
            rows.append(pd.Series(vector, index=units, name=str(source_row)))
            meta.append(
                {
                    "source_row_id": str(source_row),
                    "odor_class": label,
                    "odor_name": f"{label}-{replicate}",
                    "inchikey": f"IK-{label}-{replicate}",
                    "cid": str(source_row),
                    "cas": f"CAS-{source_row}",
                }
            )
            source_row += 1

    matrix = pd.DataFrame(rows)
    matrix.index.name = "source_row_id"
    metadata = pd.DataFrame(meta).set_index("source_row_id")

    matrix_path = root / "o002-complete-response-matrix.csv"
    metadata_path = root / "o002-odor-metadata.csv"
    matrix.to_csv(matrix_path)
    metadata.to_csv(metadata_path)

    receipt = {
        "protocol": "o002-within-study-development-v1",
        "development_only": True,
        "confirmatory_use_allowed": False,
        "feature_identity": "source_responding_unit",
        "selected_study": {"study_id": "Synthetic.Dev"},
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


def test_row_l2_normalize_preserves_zero_rows() -> None:
    x = np.asarray([[3.0, 4.0], [0.0, 0.0]])
    normalized = _row_l2_normalize(x)
    assert normalized[0].tolist() == pytest.approx([0.6, 0.8])
    assert normalized[1].tolist() == [0.0, 0.0]


def test_robustness_detects_channel_identity_signal(tmp_path: Path) -> None:
    source = _write_v1_fixture(tmp_path / "v1")
    output = tmp_path / "v2"
    report = run_o002_robustness(source, output_dir=output)

    assert report["status"] == "development_complete_not_confirmatory"
    assert report["development_only"] is True
    assert report["confirmatory_use_allowed"] is False
    assert report["o003_accessed"] is False
    assert report["frozen_before_v2_outcomes"] is True
    assert report["full_pattern"]["balanced_accuracy"] > 0.9
    assert (
        report["identity_erased_sorted_profile"]["balanced_accuracy"]
        < report["full_pattern"]["balanced_accuracy"]
    )
    assert report["channel_identity_shuffle_null"]["count"] == 256
    assert report["sample"]["eligible_odors"] == 12
    assert report["sample"]["responding_units"] == 8
    assert (output / "o002-robustness-receipt.json").is_file()
    assert (output / "o002-v2-leave-one-unit.csv").is_file()
    assert (output / "o002-v2-feature-subsets.csv").is_file()
    assert (output / "o002-v2-channel-shuffle-null.csv").is_file()
    assert (output / "SUMMARY.txt").is_file()


def test_robustness_is_deterministic(tmp_path: Path) -> None:
    source = _write_v1_fixture(tmp_path / "v1")
    a = run_o002_robustness(source, output_dir=tmp_path / "a")
    b = run_o002_robustness(source, output_dir=tmp_path / "b")

    assert a["full_pattern"] == b["full_pattern"]
    assert a["channel_identity_shuffle_null"] == b["channel_identity_shuffle_null"]
    assert a["feature_subset_robustness"] == b["feature_subset_robustness"]


def test_robustness_refuses_tampered_v1_matrix(tmp_path: Path) -> None:
    source = _write_v1_fixture(tmp_path / "v1")
    matrix_path = source / "o002-complete-response-matrix.csv"
    matrix_path.write_text(matrix_path.read_text() + "\n")

    with pytest.raises(ValueError, match="matrix hash mismatch"):
        run_o002_robustness(source, output_dir=tmp_path / "v2")


def test_robustness_refuses_overwrite(tmp_path: Path) -> None:
    source = _write_v1_fixture(tmp_path / "v1")
    output = tmp_path / "v2"
    output.mkdir()
    (output / "existing.txt").write_text("keep\n")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        run_o002_robustness(source, output_dir=output)
