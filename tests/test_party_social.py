from fly_sniff.config import ArenaConfig
from fly_sniff.party_social import CULPRIT_INDEX, PEOPLE, PROXY_CLAIM_LABEL, _should_reveal


def test_party_culprit_is_exact_benchmark_source():
    arena = ArenaConfig()
    x, y, _ = PEOPLE[CULPRIT_INDEX]
    assert x == arena.source_x
    assert y == arena.source_y


def test_party_has_multiple_visual_decoys():
    assert len(PEOPLE) >= 6
    assert 0 <= CULPRIT_INDEX < len(PEOPLE)


def test_proxy_label_cannot_be_mistaken_for_malecns_result():
    assert "NOT A MALECNS RESULT" in PROXY_CLAIM_LABEL


def test_party_reveals_ground_truth_at_end_without_requiring_success():
    assert not _should_reveal(frame=79, frames=100, fps=10, done=False)
    assert _should_reveal(frame=80, frames=100, fps=10, done=False)
    assert _should_reveal(frame=1, frames=100, fps=10, done=True)
