from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_video_neural.registry import load_registry, validate_registry


def test_shipped_registry_has_one_paired_primary():
    path = Path(__file__).resolve().parents[1] / "configs" / "datasets_v1.json"
    document = load_registry(path)
    primary = [d for d in document["datasets"] if d["role"] == "primary_paired_benchmark"]
    assert [d["id"] for d in primary] == ["mc2p_v1"]
    assert all(d["download_in_ci"] is False for d in document["datasets"])


def test_registry_rejects_neural_free_primary():
    path = Path(__file__).resolve().parents[1] / "configs" / "datasets_v1.json"
    document = json.loads(path.read_text())
    primary = next(d for d in document["datasets"] if d["role"] == "primary_paired_benchmark")
    primary["modalities"]["neural"] = False
    with pytest.raises(ValueError, match="measured video and neural"):
        validate_registry(document)
