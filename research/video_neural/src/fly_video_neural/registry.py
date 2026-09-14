from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_ROLES = {"primary_paired_benchmark", "behavior_pretraining", "pose_pretraining", "future_neural_validation"}


def load_registry(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_registry(document)
    return document


def validate_registry(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("dataset registry requires schema_version=1")
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset registry requires a non-empty datasets list")
    ids = [item.get("id") for item in datasets]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("every dataset requires a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("dataset ids must be unique")
    paired = 0
    for item in datasets:
        role = item.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"unsupported dataset role {role!r}")
        if item.get("download_in_ci") is not False:
            raise ValueError("large/public datasets must never download in CI")
        sources = item.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"dataset {item['id']} requires at least one source")
        modalities = item.get("modalities", {})
        if role == "primary_paired_benchmark":
            paired += 1
            if modalities.get("video") is not True or modalities.get("neural") is not True:
                raise ValueError("primary paired benchmark must contain measured video and neural activity")
            if item.get("synchronized") is not True:
                raise ValueError("primary paired benchmark must be synchronized")
    if paired != 1:
        raise ValueError("registry must define exactly one primary_paired_benchmark")
