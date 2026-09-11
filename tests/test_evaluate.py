from fly_sniff.config import ArenaConfig, PlumeConfig, SensorConfig
from fly_sniff.controllers import RandomWalkController
from fly_sniff.evaluate import evaluate


def test_evaluate_accepts_frozen_environment_configs():
    arena = ArenaConfig(max_steps=2)
    plume = PlumeConfig(warmup_s=0.0, emission_rate_hz=0.0)
    sensors = SensorConfig()
    frame = evaluate(
        {"random": RandomWalkController},
        [1, 2],
        arena=arena,
        plume=plume,
        sensors=sensors,
    )
    assert len(frame) == 2
    assert set(frame.label) == {"random"}
