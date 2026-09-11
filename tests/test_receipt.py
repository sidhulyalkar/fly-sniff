import json

import pytest

from fly_sniff.evaluate import sealed_manifest_digest, write_receipt
from fly_sniff.freeze import build_manifest


def test_receipt_preserves_sealed_manifest_identity(tmp_path):
    manifest = build_manifest(seed=7, n_id=3, n_ood=2, code_ref="abc", circuit_sha256="def")
    assert sealed_manifest_digest(manifest) == manifest["manifest_sha256"]
    path = tmp_path / "receipt.json"
    write_receipt(path, manifest, {"ok": True})
    receipt = json.loads(path.read_text())
    assert receipt["manifest_sha256"] == manifest["manifest_sha256"]


def test_sealed_manifest_digest_rejects_mutation():
    manifest = build_manifest(seed=7, n_id=3, n_ood=2, code_ref="abc", circuit_sha256="def")
    manifest["heldout_seeds"][0] += 1
    with pytest.raises(ValueError, match="self-hash mismatch"):
        sealed_manifest_digest(manifest)
