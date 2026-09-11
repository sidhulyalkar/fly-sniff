from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArenaConfig:
    width: float = 12.0
    height: float = 6.0
    source_x: float = 1.0
    source_y: float = 3.0
    source_radius: float = 0.32
    start_x: float = 10.5
    antenna_separation: float = 0.12
    dt: float = 0.05
    max_steps: int = 900
    speed: float = 0.9
    max_turn_rate: float = 2.8


@dataclass(frozen=True)
class PlumeConfig:
    wind_speed: float = 0.70
    emission_rate_hz: float = 8.0
    initial_sigma: float = 0.10
    diffusion_rate: float = 0.035
    crosswind_noise: float = 0.14
    meander_amplitude: float = 0.18
    meander_frequency: float = 0.55
    puff_mass: float = 1.0
    max_puffs: int = 600
    warmup_s: float = 18.0


@dataclass(frozen=True)
class SensorConfig:
    concentration_gain: float = 3.0
    concentration_half_sat: float = 0.18
    adaptation_tau: float = 0.8


@dataclass(frozen=True)
class EvaluationConfig:
    dev_episodes: int = 64
    heldout_episodes: int = 1000
    ood_episodes: int = 400
    success_target: float = 0.70
    ood_success_target: float = 0.60
    spl_delta_target: float = 0.10


def default_config_dict() -> dict:
    return {
        "arena": asdict(ArenaConfig()),
        "plume": asdict(PlumeConfig()),
        "sensor": asdict(SensorConfig()),
        "evaluation": asdict(EvaluationConfig()),
    }
