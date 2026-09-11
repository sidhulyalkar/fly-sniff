from fly_sniff.config import ArenaConfig
from fly_sniff.party_social import CULPRIT_INDEX, PEOPLE


def test_party_culprit_is_exact_benchmark_source():
    arena = ArenaConfig()
    x, y, _ = PEOPLE[CULPRIT_INDEX]
    assert x == arena.source_x
    assert y == arena.source_y


def test_party_has_multiple_visual_decoys():
    assert len(PEOPLE) >= 6
    assert 0 <= CULPRIT_INDEX < len(PEOPLE)
