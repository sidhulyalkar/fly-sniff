from fly_sniff.freeze import build_manifest, canonical_sha256


def test_manifest_is_deterministic_and_self_identifying():
    a = build_manifest(seed=7, n_id=5, n_ood=3, code_ref="abc", circuit_sha256="def")
    b = build_manifest(seed=7, n_id=5, n_ood=3, code_ref="abc", circuit_sha256="def")
    assert a == b
    digest = a["manifest_sha256"]
    unsigned = dict(a)
    unsigned.pop("manifest_sha256")
    assert digest == canonical_sha256(unsigned)
    assert len(a["heldout_seeds"]) == 5
    assert len(a["ood_seeds"]) == 3
