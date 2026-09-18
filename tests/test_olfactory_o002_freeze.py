from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_sniff.olfactory_o002_freeze import validate_o002_freeze

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "authority" / "o002-development-freeze-v1.json"


def _payload() -> dict:
    return json.loads(FREEZE.read_text())


def test_o002_freeze_closes_same_table_model_search() -> None:
    report = validate_o002_freeze(_payload())
    assert report["status"] == "frozen_development_finding_not_confirmatory"
    assert report["study_id"] == "Hallem.2006.EN"
    assert report["responding_units"] == 24
    assert report["complete_odors"] == 110
    assert report["same_table_model_search_closed"] is True
    assert report["confirmatory_usable"] is False


def test_o002_freeze_cannot_reopen_model_search() -> None:
    payload = _payload()
    payload["continuation_policy"][
        "further_model_or_threshold_search_on_same_O002_table_allowed"
    ] = True
    with pytest.raises(ValueError, match="cannot reopen"):
        validate_o002_freeze(payload)


def test_o002_freeze_cannot_promote_topology_claim() -> None:
    payload = _payload()
    payload["forbidden_promotions"].remove("connectome topology effect")
    with pytest.raises(ValueError, match="lost a claim boundary"):
        validate_o002_freeze(payload)


def test_o002_freeze_lineage_is_immutable() -> None:
    payload = _payload()
    payload["lineage"]["v3"]["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage changed"):
        validate_o002_freeze(payload)
