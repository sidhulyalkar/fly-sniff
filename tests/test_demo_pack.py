from fly_sniff.choice import benchmark_choice
from fly_sniff.controllers import BilateralProxyController, RandomWalkController
from fly_sniff.demo_pack import make_demo_manifest


def test_demo_manifest_is_explicitly_non_claim_bearing(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "merge123")
    monkeypatch.setenv("SOURCE_HEAD_SHA", "head456")
    proxy = benchmark_choice(BilateralProxyController(), trials=4, seed=7)
    random = benchmark_choice(RandomWalkController(), trials=4, seed=7)
    manifest = make_demo_manifest(
        seed=7,
        trials=4,
        seconds=6,
        fps=10,
        proxy_report=proxy,
        random_report=random,
        clip_name="who-farted-proxy.gif",
    )

    assert manifest["claim_status"] == "development-only"
    assert "NOT A MALECNS RESULT" in manifest["claim_label"]
    assert "simulated odor-source localization" in manifest["scientific_scope"]
    assert manifest["git_sha"] == "merge123"
    assert manifest["source_head_sha"] == "head456"
    assert "synthetic pull-request merge" in manifest["provenance_note"]
    assert manifest["choice_summary"]["chance_accuracy"] == 0.5
    assert manifest["artifacts"]["clip"] == "who-farted-proxy.gif"
