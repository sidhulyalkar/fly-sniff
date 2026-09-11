import numpy as np

from fly_sniff.config import ArenaConfig, PlumeConfig
from fly_sniff.plume import TurbulentPlume


def test_plume_is_deterministic_for_seed():
    arena = ArenaConfig()
    cfg = PlumeConfig()
    a = TurbulentPlume(arena, cfg, seed=42)
    b = TurbulentPlume(arena, cfg, seed=42)
    for _ in range(50):
        a.step()
        b.step()
    assert np.allclose(a.snapshot(), b.snapshot())
    assert a.concentration(5.0, 3.0) == b.concentration(5.0, 3.0)


def test_source_is_upwind_of_downstream_agent():
    arena = ArenaConfig()
    assert arena.source_x < arena.start_x
