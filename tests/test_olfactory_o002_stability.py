from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fly_sniff.olfactory_door import sha256_file
from fly_sniff.olfactory_e006_audit import _canonical_sha
from fly_sniff.olfactory_o002_stability import run_o002_stability


def _write_fixture(root: Path) -> tuple[Path, Path]:
    v1 = root / "v1"
    v2 = root / "v2"
    v1.mkdir()
    v2.mkdir()

    units = [f"Or{i:02d}" for i in range(24)]
    matrix_rows = []
    meta_rows = []
    source_row = 1
    class_units = {"class-a": 0, "class-b": 1, "class-c": 2}
    for label, active in class_units.items():
        for replicate in range(4):
            vector = np.zeros(24, dtype=float)
            vector[active] = 10.0
            vector[3:] = 0.05 * np.sin(np.arange(21) + replicate)
            matrix_rows.append(pd.Series(vector, index=units, name=str(source_row)))
            meta_rows.append(
                {
                    "source_row_id": str(source_row),
                    "odor_class": label,
                    "odor_name": f"{label}-{replicate}",
                    "inchikey": f"IK-{source_row}",
                    "cid": str(source_row),
                    "cas": f"CAS-{source_row}",
                }
            )
            source_row += 1

    matrix = pd.DataFrame(matrix_rows)
    matrix.index.name = "source_row_id"
    metadata = pd.DataFrame(meta_rows).set_index("source_row_id")

    matrix_path = v1 / "o002-complete-response-matrix.csv"
    metadata_path = v1 / "o002-odor-metadata.csv"
    matrix.to_csv(matrix_path)
    metadata.to_csv(metadata_path)

    v1_receipt = {
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
    v1_receipt["receipt_sha256"] = _canonical_sha(v1_receipt)
    v1_path = v1 / "o002-development-receipt.json"
    v1_path.write_text(json.dumps(v1_receipt, indent=2, sort_keys=True) + "\n")

    v2_receipt = {
        "protocol": "o002-coding-robustness-development-v2",
        "development_only": True,
        "confirmatory_use_allowed": False,
        "input": {"v1_receipt_sha256": v1_receipt["receipt_sha256"]},
    }
    v2_receipt["receipt_sha256"] = _canonical_sha(v2_receipt)
    (v2 / "o002-robustness-receipt.json").write_text(
        json.dumps(v2_receipt, indent=2, sort_keys=True) + "\n"
    )
    return v1, v2


def test_v3_runs_frozen_stability_diagnostics(tmp_path: Path) -> None:
    v1, v2 = _write_fixture(tmp_path)
    output = tmp_path / "v3"
    report = run_o002_stability(v1, v2, output_dir=output)

    assert report["status"] == "development_complete_not_confirmatory"
    assert report["development_only"] is True
    assert report["confirmatory_use_allowed"] is False
    assert report["o003_accessed"] is False
    assert report["frozen_before_v3_outcomes"] is True
    assert report["balanced_holdout_stability"]["draws"] == 512
    assert report["balanced_holdout_stability"]["seed"] == 24020
    assert len(report["subspace_curve"]) == 9
    assert report["subspace_curve"][0]["components"] == 1
    assert report["subspace_curve"][-1]["components"] == 24
    assert report["sample"]["eligible_odors"] == 12
    assert report["sample"]["responding_units"] == 24
    assert (output / "o002-v3-full-pattern-per-class.csv").is_file()
    assert (output / "o002-v3-direction-only-per-class.csv").is_file()
    assert (output / "o002-v3-balanced-holdout.csv").is_file()
    assert (output / "o002-v3-subspace-curve.csv").is_file()
    assert (output / "o002-stability-receipt.json").is_file()
    assert (output / "SUMMARY.txt").is_file()


def test_v3_is_deterministic(tmp_path: Path) -> None:
    v1, v2 = _write_fixture(tmp_path)
    a = run_o002_stability(v1, v2, output_dir=tmp_path / "a")
    b = run_o002_stability(v1, v2, output_dir=tmp_path / "b")

    assert a["balanced_holdout_stability"] == b["balanced_holdout_stability"]
    assert a["subspace_curve"] == b["subspace_curve"]
    assert a["class_diagnostics"] == b["class_diagnostics"]


def test_v3_refuses_mismatched_v2_lineage(tmp_path: Path) -> None:
    v1, v2 = _write_fixture(tmp_path)
    path = v2 / "o002-robustness-receipt.json"
    receipt = json.loads(path.read_text())
    receipt.pop("receipt_sha256")
    receipt["input"]["v1_receipt_sha256"] = "wrong"
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="does not point to the supplied O002 v1 receipt"):
        run_o002_stability(v1, v2, output_dir=tmp_path / "out")


def test_v3_refuses_overwrite(tmp_path: Path) -> None:
    v1, v2 = _write_fixture(tmp_path)
    output = tmp_path / "v3"
    output.mkdir()
    (output / "existing.txt").write_text("keep\n")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        run_o002_stability(v1, v2, output_dir=output)
