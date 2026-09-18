from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fly_sniff.olfactory_door import sha256_file
from fly_sniff.olfactory_e006_audit import _canonical_sha
from fly_sniff.olfactory_o002_visual import (
    BUNDLE_FILENAME,
    CLASS_RECALL_FILENAME,
    DEEP_DIVE_FILENAME,
    HERO_FILENAME,
    RECEIPT_FILENAME,
    ROADMAP_FILENAME,
    SUMMARY_FILENAME,
    load_o002_visual_data,
    render_showcase,
)


def _seal(payload: dict) -> dict:
    result = dict(payload)
    result["receipt_sha256"] = _canonical_sha(result)
    return result


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    v1 = root / "v1"
    v2 = root / "v2"
    v3 = root / "v3"
    v1.mkdir()
    v2.mkdir()
    v3.mkdir()

    classes = ["acid", "alcohol", "ester"]
    rows = []
    metadata = []
    rng = np.random.default_rng(7)
    source_id = 1
    for class_index, label in enumerate(classes):
        for rep in range(4):
            vector = rng.normal(0, 0.04, 6)
            vector[class_index] += 1.0
            rows.append(pd.Series(vector, name=str(source_id)))
            metadata.append(
                {
                    "source_row_id": str(source_id),
                    "odor_name": f"{label}-{rep}",
                    "odor_class": label,
                }
            )
            source_id += 1

    matrix = pd.DataFrame(rows)
    centered = matrix.to_numpy() - matrix.to_numpy().mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = pd.DataFrame(
        centered @ vt.T,
        index=matrix.index,
        columns=[f"PC{i + 1}" for i in range(vt.shape[0])],
    )
    coords.index.name = "source_row_id"
    meta = pd.DataFrame(metadata).set_index("source_row_id")

    coords_path = v1 / "o002-pca-coordinates.csv"
    meta_path = v1 / "o002-odor-metadata.csv"
    coords.to_csv(coords_path)
    meta.to_csv(meta_path)

    v1_receipt = _seal(
        {
            "protocol": "o002-within-study-development-v1",
            "development_only": True,
            "confirmatory_use_allowed": False,
            "selected_study": {"study_id": "Synthetic.Visual"},
            "outputs": {
                "pca_coordinates": {"sha256": sha256_file(coords_path)},
                "odor_metadata": {"sha256": sha256_file(meta_path)},
            },
        }
    )
    (v1 / "o002-development-receipt.json").write_text(
        json.dumps(v1_receipt, indent=2, sort_keys=True) + "\n"
    )

    v2_receipt = _seal(
        {
            "protocol": "o002-coding-robustness-development-v2",
            "development_only": True,
            "confirmatory_use_allowed": False,
            "input": {"v1_receipt_sha256": v1_receipt["receipt_sha256"]},
            "sample": {
                "study_id": "Synthetic.Visual",
                "eligible_odors": 12,
                "responding_units": 6,
                "eligible_classes": classes,
            },
            "full_pattern": {"balanced_accuracy": 0.72},
            "direction_only": {"balanced_accuracy": 0.78},
            "identity_erased_sorted_profile": {"balanced_accuracy": 0.44},
            "amplitude_only": {
                "mean_response_balanced_accuracy": 0.38,
                "l2_norm_balanced_accuracy": 0.36,
                "peak_to_peak_balanced_accuracy": 0.32,
            },
            "channel_identity_shuffle_null": {
                "mean": 0.31,
                "q95": 0.39,
            },
        }
    )
    (v2 / "o002-robustness-receipt.json").write_text(
        json.dumps(v2_receipt, indent=2, sort_keys=True) + "\n"
    )

    holdout = pd.DataFrame(
        {
            "full_pattern_accuracy": [0.66, 0.75, 0.58, 0.83],
            "direction_only_accuracy": [0.75, 0.83, 0.66, 0.91],
            "identity_erased_sorted_accuracy": [0.41, 0.33, 0.50, 0.42],
        }
    )
    subspace = pd.DataFrame(
        {
            "components": [1, 2, 3, 5, 6],
            "full_pattern_balanced_accuracy": [0.35, 0.50, 0.63, 0.70, 0.72],
            "direction_only_balanced_accuracy": [0.32, 0.55, 0.69, 0.76, 0.78],
        }
    )
    confusion = pd.DataFrame(
        [[3, 1, 0], [0, 4, 0], [0, 1, 3]],
        index=classes,
        columns=classes,
    )
    direction_class = pd.DataFrame(
        {
            "odor_class": classes,
            "count": [4, 4, 4],
            "recall": [0.75, 1.0, 0.75],
        }
    )

    holdout_path = v3 / "o002-v3-balanced-holdout.csv"
    subspace_path = v3 / "o002-v3-subspace-curve.csv"
    confusion_path = v3 / "o002-v3-direction-only-confusion.csv"
    direction_class_path = v3 / "o002-v3-direction-only-per-class.csv"
    holdout.to_csv(holdout_path, index=False)
    subspace.to_csv(subspace_path, index=False)
    confusion.to_csv(confusion_path)
    direction_class.to_csv(direction_class_path, index=False)

    v3_receipt = _seal(
        {
            "protocol": "o002-subspace-stability-development-v3",
            "development_only": True,
            "confirmatory_use_allowed": False,
            "input": {
                "v1_receipt_sha256": v1_receipt["receipt_sha256"],
                "v2_receipt_sha256": v2_receipt["receipt_sha256"],
            },
            "outputs": {
                "balanced_holdout": {"sha256": sha256_file(holdout_path)},
                "subspace_curve": {"sha256": sha256_file(subspace_path)},
                "direction_only_confusion": {"sha256": sha256_file(confusion_path)},
                "direction_only_per_class": {"sha256": sha256_file(direction_class_path)},
            },
            "claim_boundary": "synthetic development fixture",
        }
    )
    (v3 / "o002-stability-receipt.json").write_text(
        json.dumps(v3_receipt, indent=2, sort_keys=True) + "\n"
    )
    return v1, v2, v3


def test_load_visual_data_verifies_lineage_and_hashes(tmp_path: Path) -> None:
    v1, v2, v3 = _write_fixture(tmp_path)
    data = load_o002_visual_data(v1, v2, v3)

    assert data.study_id == "Synthetic.Visual"
    assert data.eligible_odors == 12
    assert data.responding_units == 6
    assert data.eligible_classes == ["acid", "alcohol", "ester"]


def test_static_showcase_renders_receipt_and_bundle(tmp_path: Path) -> None:
    v1, v2, v3 = _write_fixture(tmp_path)
    out = tmp_path / "showcase"

    result = render_showcase(v1, v2, v3, output_dir=out, video=False)

    for name in (
        HERO_FILENAME,
        DEEP_DIVE_FILENAME,
        CLASS_RECALL_FILENAME,
        ROADMAP_FILENAME,
        RECEIPT_FILENAME,
        SUMMARY_FILENAME,
        BUNDLE_FILENAME,
    ):
        path = out / name
        assert path.is_file()
        assert path.stat().st_size > 0

    receipt = json.loads((out / RECEIPT_FILENAME).read_text())
    assert receipt["protocol"] == "o002-visual-showcase-v1"
    assert receipt["development_only"] is True
    assert receipt["confirmatory_use_allowed"] is False
    assert receipt["scientific_metrics_recomputed"] is False
    assert receipt["selection_from_visual_outcomes_allowed"] is False
    assert receipt["input"]["v1_receipt_sha256"]
    assert result["social_video"] is None


def test_visual_showcase_refuses_overwrite_without_force(tmp_path: Path) -> None:
    v1, v2, v3 = _write_fixture(tmp_path)
    out = tmp_path / "showcase"
    render_showcase(v1, v2, v3, output_dir=out, video=False)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        render_showcase(v1, v2, v3, output_dir=out, video=False)


def test_visual_showcase_force_replaces_only_known_outputs(tmp_path: Path) -> None:
    v1, v2, v3 = _write_fixture(tmp_path)
    out = tmp_path / "showcase"
    render_showcase(v1, v2, v3, output_dir=out, video=False)
    sentinel = out / "keep-me.txt"
    sentinel.write_text("do not delete\n")

    render_showcase(v1, v2, v3, output_dir=out, video=False, force=True)

    assert sentinel.read_text() == "do not delete\n"
    assert (out / HERO_FILENAME).is_file()


def test_visual_showcase_refuses_mismatched_lineage(tmp_path: Path) -> None:
    v1, v2, v3 = _write_fixture(tmp_path)
    receipt_path = v3 / "o002-stability-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("receipt_sha256")
    receipt["input"]["v2_receipt_sha256"] = "wrong"
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="v3 lineage"):
        load_o002_visual_data(v1, v2, v3)


def test_visual_showcase_refuses_tampered_input(tmp_path: Path) -> None:
    v1, v2, v3 = _write_fixture(tmp_path)
    path = v3 / "o002-v3-direction-only-per-class.csv"
    path.write_text(path.read_text() + "tampered\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_o002_visual_data(v1, v2, v3)
