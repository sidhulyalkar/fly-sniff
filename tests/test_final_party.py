from fly_sniff.config import ArenaConfig
from fly_sniff.final_party import _people_for_arena


def test_final_party_first_suspect_is_exact_source():
    arena = ArenaConfig(source_x=2.25, source_y=4.1)
    people = _people_for_arena(arena)
    assert people[0][:2] == (arena.source_x, arena.source_y)
    assert len(people) == 6
