from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .plume import TurbulentPlume


@dataclass
class Observation:
    left_odor: float
    right_odor: float
    mean_odor: float
    odor_delta: float
    wind_x_body: float
    wind_y_body: float
    heading: float


@dataclass
class AgentState:
    x: float
    y: float
    heading: float
    path_length: float = 0.0
    steps: int = 0
    found: bool = False
    left_adapt: float = 0.0
    right_adapt: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)


class FlySniffEnv:
    def __init__(
        self,
        seed: int,
        arena: ArenaConfig | None = None,
        plume: PlumeConfig | None = None,
        sensors: SensorConfig | None = None,
    ):
        self.seed = int(seed)
        self.arena = arena or ArenaConfig()
        self.plume_config = plume or PlumeConfig()
        self.sensor_config = sensors or SensorConfig()
        self.rng = np.random.default_rng(self.seed + 17)
        self.plume = TurbulentPlume(self.arena, self.plume_config, self.seed)
        self.agent = self._new_agent()

    def _new_agent(self) -> AgentState:
        y = float(self.rng.uniform(0.8, self.arena.height - 0.8))
        heading = float(self.rng.uniform(-np.pi, np.pi))
        state = AgentState(self.arena.start_x, y, heading)
        state.history.append((state.x, state.y))
        return state

    def _antenna_positions(self) -> tuple[tuple[float, float], tuple[float, float]]:
        a = self.agent
        half = 0.5 * self.arena.antenna_separation
        # Lateral axis relative to heading.
        lx = a.x - np.sin(a.heading) * half
        ly = a.y + np.cos(a.heading) * half
        rx = a.x + np.sin(a.heading) * half
        ry = a.y - np.cos(a.heading) * half
        return (lx, ly), (rx, ry)

    def _transduce(self, concentration: float, side: str) -> float:
        cfg = self.sensor_config
        raw = cfg.concentration_gain * concentration
        sat = raw / (cfg.concentration_half_sat + raw + 1e-12)
        alpha = self.arena.dt / max(cfg.adaptation_tau, self.arena.dt)
        attr = "left_adapt" if side == "left" else "right_adapt"
        adapt = getattr(self.agent, attr)
        adapt += alpha * (sat - adapt)
        setattr(self.agent, attr, adapt)
        # Preserve onset sensitivity while avoiding negative receptor activity.
        return float(np.clip(0.72 * sat + 0.28 * max(sat - adapt, 0.0), 0.0, 1.0))

    def observe(self) -> Observation:
        (lx, ly), (rx, ry) = self._antenna_positions()
        left = self._transduce(self.plume.concentration(lx, ly), "left")
        right = self._transduce(self.plume.concentration(rx, ry), "right")
        wind = self.plume.wind_vector
        c, s = np.cos(-self.agent.heading), np.sin(-self.agent.heading)
        wx = c * wind[0] - s * wind[1]
        wy = s * wind[0] + c * wind[1]
        return Observation(
            left_odor=left,
            right_odor=right,
            mean_odor=0.5 * (left + right),
            odor_delta=right - left,
            wind_x_body=float(wx),
            wind_y_body=float(wy),
            heading=self.agent.heading,
        )

    def step(self, turn_command: float, speed_scale: float = 1.0) -> tuple[Observation, bool]:
        a = self.agent
        turn = float(np.clip(turn_command, -1.0, 1.0)) * self.arena.max_turn_rate
        a.heading = float((a.heading + turn * self.arena.dt + np.pi) % (2 * np.pi) - np.pi)
        speed = self.arena.speed * float(np.clip(speed_scale, 0.0, 1.5))
        old_x, old_y = a.x, a.y
        a.x += np.cos(a.heading) * speed * self.arena.dt
        a.y += np.sin(a.heading) * speed * self.arena.dt
        a.x = float(np.clip(a.x, 0.0, self.arena.width))
        a.y = float(np.clip(a.y, 0.0, self.arena.height))
        a.path_length += float(np.hypot(a.x - old_x, a.y - old_y))
        a.steps += 1
        a.history.append((a.x, a.y))
        distance = np.hypot(a.x - self.arena.source_x, a.y - self.arena.source_y)
        a.found = bool(distance <= self.arena.source_radius)
        self.plume.step()
        done = a.found or a.steps >= self.arena.max_steps
        return self.observe(), done

    @property
    def distance_to_source(self) -> float:
        return float(
            np.hypot(
                self.agent.x - self.arena.source_x,
                self.agent.y - self.arena.source_y,
            )
        )
