import json

import pandas as pd
import pytest

from fly_sniff.freeze import build_manifest
from fly_sniff.qualified_eval import circuit_digest, verify_sealed_manifest


def test_sealed_manifest_verification_detects_mutation():
    manifest = build_manifest(seed=9, n_id=3, n_ood=2, code_ref="abc", circuit_sha256="def")
    assert manifest["lesion"] == {
        "kind": "remove-incoming-edges-to-roles",
        "roles": ["steer_left", "steer_right"],
    }
    verify_sealed_manifest(manifest)
    mutated = dict(manifest)
    mutated["gold"] = dict(mutated["gold"])
    mutated["gold"]["success_rate_min"] = 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_sealed_manifest(mutated)


def test_circuit_digest_is_stable(tmp_path):
    pd.DataFrame({"bodyId": [1, 2]}).to_parquet(tmp_path / "nodes.parquet", index=False)
    pd.DataFrame({"source": [1], "target": [2], "weight": [5], "sign": [1]}).to_parquet(
        tmp_path / "edges.parquet", index=False
    )
    (tmp_path / "roles.json").write_text(json.dumps({"odor_left": [1], "steer_right": [2]}))
    assert circuit_digest(tmp_path) == circuit_digest(tmp_path)
