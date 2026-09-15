import numpy as np

from fly_sniff.config import ArenaConfig, PlumeConfig
from fly_sniff.plume import TurbulentPlume, concentration_from_snapshot


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


def test_single_puff_center_matches_normalized_2d_gaussian():
    arena = ArenaConfig()
    plume = TurbulentPlume(arena, PlumeConfig(), seed=1)
    sigma = 0.25
    mass = 2.5
    plume.x = np.array([4.0])
    plume.y = np.array([3.0])
    plume.sigma = np.array([sigma])
    plume.mass = np.array([mass])
    plume.age = np.array([0.0])

    expected = mass / (2.0 * np.pi * sigma**2)
    assert np.isclose(plume.concentration(4.0, 3.0), expected, rtol=1e-12, atol=0.0)


def test_complete_snapshot_reconstructs_live_concentration_exactly():
    arena = ArenaConfig()
    cfg = PlumeConfig(max_puffs=600)
    plume = TurbulentPlume(arena, cfg, seed=17)
    plume.warmup()
    snapshot = plume.snapshot(max_points=cfg.max_puffs)

    assert plume.snapshot_is_complete(cfg.max_puffs)
    for x, y in [(1.5, 3.0), (3.0, 2.5), (6.5, 3.5)]:
        expected = plume.concentration(x, y)
        replayed = concentration_from_snapshot(snapshot, cfg.puff_mass, x, y)
        assert np.isclose(replayed, expected, rtol=1e-12, atol=1e-12)


def test_snapshot_reports_when_display_puffs_are_sampled():
    arena = ArenaConfig()
    cfg = PlumeConfig(max_puffs=600)
    plume = TurbulentPlume(arena, cfg, seed=23)
    plume.warmup()

    assert len(plume.x) > 10
    assert not plume.snapshot_is_complete(10)
    assert plume.snapshot(10).shape[0] <= 10


def test_single_puff_concentration_is_radially_symmetric_and_decays():
    arena = ArenaConfig()
    plume = TurbulentPlume(arena, PlumeConfig(), seed=2)
    plume.x = np.array([5.0])
    plume.y = np.array([2.0])
    plume.sigma = np.array([0.4])
    plume.mass = np.array([1.0])
    plume.age = np.array([0.0])

    center = plume.concentration(5.0, 2.0)
    right = plume.concentration(5.3, 2.0)
    up = plume.concentration(5.0, 2.3)
    farther = plume.concentration(5.6, 2.0)

    assert np.isclose(right, up, rtol=1e-12, atol=1e-15)
    assert center > right > farther > 0.0


def test_puff_width_obeys_diffusion_variance_law_without_noise_dependence():
    arena = ArenaConfig(dt=0.05)
    cfg = PlumeConfig(initial_sigma=0.1, diffusion_rate=0.035, emission_rate_hz=0.0)
    plume = TurbulentPlume(arena, cfg, seed=3)
    plume.x = np.array([3.0])
    plume.y = np.array([2.0])
    plume.age = np.array([0.0])
    plume.sigma = np.array([cfg.initial_sigma])
    plume.mass = np.array([1.0])

    for _ in range(20):
        plume.step()

    expected_variance = cfg.initial_sigma**2 + 2.0 * cfg.diffusion_rate * plume.age[0]
    assert np.isclose(plume.sigma[0] ** 2, expected_variance, rtol=1e-12, atol=1e-15)


def test_source_is_upwind_of_downstream_agent():
    arena = ArenaConfig()
    assert arena.source_x < arena.start_x
