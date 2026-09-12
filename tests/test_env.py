import numpy as np

from fly_sniff.config import ArenaConfig, SensorConfig
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


def test_observe_is_idempotent_at_fixed_physical_state():
    env = FlySniffEnv(seed=11)
    first = env.observe()
    first_adapt = (env.agent.left_adapt, env.agent.right_adapt)

    second = env.observe()
    second_adapt = (env.agent.left_adapt, env.agent.right_adapt)

    assert first == second
    assert first_adapt == second_adapt


def test_transduction_outputs_from_prior_adaptation_then_updates_state():
    arena = ArenaConfig(dt=0.05)
    sensors = SensorConfig(
        concentration_gain=3.0,
        concentration_half_sat=0.18,
        adaptation_tau=0.8,
    )
    env = FlySniffEnv(seed=19, arena=arena, sensors=sensors)
    env.agent.left_adapt = 0.0

    concentration = 1.0
    raw = sensors.concentration_gain * concentration
    sat = raw / (sensors.concentration_half_sat + raw + 1e-12)
    response = env._transduce(concentration, "left")

    expected_response = sat
    expected_next_adapt = env._adaptation_alpha() * sat
    assert np.isclose(response, expected_response, rtol=0.0, atol=1e-12)
    assert np.isclose(env.agent.left_adapt, expected_next_adapt, rtol=0.0, atol=1e-12)


def test_sensor_adaptation_advances_once_per_simulator_step():
    env = FlySniffEnv(seed=17)

    def constant_concentration(x: float, y: float) -> float:
        return 1.0

    env.plume.concentration = constant_concentration
    env.observe()
    before = (env.agent.left_adapt, env.agent.right_adapt)

    next_obs, _ = env.step(turn_command=0.0, speed_scale=0.0)
    after_step = (env.agent.left_adapt, env.agent.right_adapt)
    repeated = env.observe()
    after_repeat = (env.agent.left_adapt, env.agent.right_adapt)

    assert next_obs == repeated
    assert after_repeat == after_step
    assert after_step[0] > before[0]
    assert after_step[1] > before[1]


def test_sensor_trace_matches_cached_observation_without_advancing_state():
    env = FlySniffEnv(seed=18)

    def constant_concentration(x: float, y: float) -> float:
        return 0.25 if y >= env.agent.y else 0.10

    env.plume.concentration = constant_concentration
    obs = env.observe()
    adapt_after_observe = (env.agent.left_adapt, env.agent.right_adapt)
    trace = env.sensor_trace()
    trace_again = env.sensor_trace()

    assert trace == trace_again
    assert (env.agent.left_adapt, env.agent.right_adapt) == adapt_after_observe
    assert np.isclose(trace.left.concentration, 0.25)
    assert np.isclose(trace.right.concentration, 0.10)
    assert np.isclose(trace.left.response, obs.left_odor)
    assert np.isclose(trace.right.response, obs.right_odor)
    assert trace.left.adaptation_before == 0.0
    assert trace.right.adaptation_before == 0.0
    assert trace.left.adaptation_after == env.agent.left_adapt
    assert trace.right.adaptation_after == env.agent.right_adapt
    assert trace.signal_kind == "modeled_antenna_transduction"


def test_adaptation_discretization_is_exact_first_order_hold():
    arena = ArenaConfig(dt=0.05)
    sensors = SensorConfig(adaptation_tau=0.8)
    env = FlySniffEnv(seed=23, arena=arena, sensors=sensors)
    expected = 1.0 - np.exp(-arena.dt / sensors.adaptation_tau)
    assert np.isclose(env._adaptation_alpha(), expected, rtol=0.0, atol=1e-15)


def test_antennae_are_symmetric_about_body_center():
    arena = ArenaConfig(antenna_separation=0.12)
    env = FlySniffEnv(seed=29, arena=arena)
    env.agent.x = 4.0
    env.agent.y = 2.0
    env.agent.heading = 0.0

    left, right = env._antenna_positions()
    midpoint = 0.5 * (np.asarray(left) + np.asarray(right))
    separation = np.linalg.norm(np.asarray(left) - np.asarray(right))

    assert np.allclose(midpoint, [env.agent.x, env.agent.y])
    assert np.isclose(separation, arena.antenna_separation)
    assert left[1] > env.agent.y
    assert right[1] < env.agent.y


def test_body_frame_wind_rotates_with_heading_without_changing_magnitude():
    env = FlySniffEnv(seed=31)
    wind_speed = env.plume_config.wind_speed

    env.agent.heading = 0.0
    downwind = env.observe()
    assert np.allclose([downwind.wind_x_body, downwind.wind_y_body], [wind_speed, 0.0])

    env.agent.heading = np.pi / 2
    crosswind = env.observe()
    assert np.allclose(
        [crosswind.wind_x_body, crosswind.wind_y_body],
        [0.0, -wind_speed],
        atol=1e-12,
    )
    assert np.isclose(
        np.hypot(crosswind.wind_x_body, crosswind.wind_y_body),
        wind_speed,
    )


def test_wall_collision_reflects_heading_without_leaving_arena():
    arena = ArenaConfig()
    env = FlySniffEnv(seed=7, arena=arena)
    env.agent.x = arena.width - 1e-4
    env.agent.y = arena.height / 2
    env.agent.heading = 0.0
    _, _ = env.step(turn_command=0.0, speed_scale=1.0)
    assert 0.0 <= env.agent.x <= arena.width
    assert np.cos(env.agent.heading) < 0.0
