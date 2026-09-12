import json

import pandas as pd
import pytest

from fly_sniff.freeze import build_manifest
from fly_sniff.qualified_eval import (
    _paired_against_null_mean,
    circuit_digest,
    verify_sealed_manifest,
)


def test_sealed_manifest_verification_detects_mutation():
    manifest = build_manifest(seed=9, n_id=3, n_ood=2, code_ref="abc", circuit_sha256="def")
    assert manifest["schema"] == "fly-sniff-final-v2"
    assert manifest["lesion"] == {
        "kind": "remove-incoming-edges-to-roles",
        "roles": ["steer_left", "steer_right"],
    }
    assert manifest["rewire"]["seed"] == manifest["rewire_ensemble"]["seeds"][0]
    assert manifest["rewire_ensemble"]["count"] == 8
    assert len(set(manifest["rewire_ensemble"]["seeds"])) == 8
    verify_sealed_manifest(manifest)
    mutated = dict(manifest)
    mutated["gold"] = dict(mutated["gold"])
    mutated["gold"]["success_rate_min"] = 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_sealed_manifest(mutated)


def test_paired_null_ensemble_uses_per_environment_mean():
    frame = pd.DataFrame(
        [
            {"seed": 1, "label": "malecns", "spl": 0.9},
            {"seed": 1, "label": "rewire", "spl": 0.4},
            {"seed": 1, "label": "rewire_01", "spl": 0.6},
            {"seed": 2, "label": "malecns", "spl": 0.7},
            {"seed": 2, "label": "rewire", "spl": 0.3},
            {"seed": 2, "label": "rewire_01", "spl": 0.5},
        ]
    )
    report = _paired_against_null_mean(
        frame,
        metric="spl",
        intact_label="malecns",
        null_labels=["rewire", "rewire_01"],
        n_boot=1000,
    )
    # Per seed: 0.9 - mean(0.4, 0.6) = 0.4; 0.7 - mean(0.3, 0.5) = 0.3.
    assert report["mean_delta"] == pytest.approx(0.35)
    assert report["n_environment_seeds"] == 2
    assert report["n_null_topologies"] == 2


def test_circuit_digest_is_stable(tmp_path):
    pd.DataFrame({"bodyId": [1, 2]}).to_parquet(tmp_path / "nodes.parquet", index=False)
    pd.DataFrame({"source": [1], "target": [2], "weight": [5], "sign": [1]}).to_parquet(
        tmp_path / "edges.parquet", index=False
    )
    (tmp_path / "roles.json").write_text(json.dumps({"odor_left": [1], "steer_right": [2]}))
    assert circuit_digest(tmp_path) == circuit_digest(tmp_path)
