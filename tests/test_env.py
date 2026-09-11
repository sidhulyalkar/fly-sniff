import numpy as np

from fly_sniff.config import ArenaConfig
from fly_sniff.env import FlySniffEnv


def test_environment_starts_with_developed_plume():
    env = FlySniffEnv(seed=123)
    assert env.plume.t >= env.plume_config.warmup_s
    assert len(env.plume.x) > 0


def test_controller_observation_exposes_no_source_coordinates():
    env = FlySniffEnv(seed=5)
    obs = env.observe()
    assert not hasattr(obs, "source_x")
    assert not hasattr(obs, "source_y")


def test_wall_collision_reflects_heading_without_leaving_arena():
    arena = ArenaConfig()
    env = FlySniffEnv(seed=7, arena=arena)
    env.agent.x = arena.width - 1e-4
    env.agent.y = arena.height / 2
    env.agent.heading = 0.0
    _, _ = env.step(turn_command=0.0, speed_scale=1.0)
    assert 0.0 <= env.agent.x <= arena.width
    assert np.cos(env.agent.heading) < 0.0
