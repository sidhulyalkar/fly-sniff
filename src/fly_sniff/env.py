from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .plume import TurbulentPlume


@dataclass(frozen=True)
class Observation:
    left_odor: float
    right_odor: float
    mean_odor: float
    odor_delta: float
    wind_x_body: float
    wind_y_body: float
    heading: float


@dataclass(frozen=True)
class AntennaSample:
    x: float
    y: float
    concentration: float
    gain_scaled_concentration: float
    saturation: float
    adaptation_before: float
    adaptation_after: float
    response: float


@dataclass(frozen=True)
class SensorTrace:
    """Auditable phenomenological sensor state for one cached observation.

    These values describe the benchmark's physical antenna sample locations and
    transduction model. They are not receptor currents, ORN spikes, or measured
    neural activity.
    """

    left: AntennaSample
    right: AntennaSample
    adaptation_alpha: float
    model_id: str = "bilateral-phenomenological-v1"
    signal_kind: str = "modeled_antenna_transduction"


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
        self.plume.warmup()
        self.agent = self._new_agent()
        self._observation_key: tuple[int, float, float, float, float] | None = None
        self._observation_cache: Observation | None = None
        self._sensor_trace_cache: SensorTrace | None = None

    def _new_agent(self) -> AgentState:
        y = float(self.rng.uniform(0.8, self.arena.height - 0.8))
        heading = float(self.rng.uniform(-np.pi, np.pi))
        state = AgentState(self.arena.start_x, y, heading)
        state.history.append((state.x, state.y))
        return state

    def _antenna_positions(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return left/right antenna sample points in world coordinates.

        The antenna baseline is perpendicular to the recorded body heading. For
        heading zero (+x), left lies at +y and right at -y.
        """
        a = self.agent
        half = 0.5 * self.arena.antenna_separation
        lx = a.x - np.sin(a.heading) * half
        ly = a.y + np.cos(a.heading) * half
        rx = a.x + np.sin(a.heading) * half
        ry = a.y - np.cos(a.heading) * half
        return (lx, ly), (rx, ry)

    def _adaptation_alpha(self) -> float:
        """Exact zero-order-hold discretization of da/dt=(s-a)/tau."""
        tau = float(self.sensor_config.adaptation_tau)
        if tau <= 0.0:
            return 1.0
        return float(1.0 - np.exp(-self.arena.dt / tau))

    def _transduction_components(self, concentration: float) -> tuple[float, float]:
        cfg = self.sensor_config
        raw = max(0.0, cfg.concentration_gain * float(concentration))
        sat = raw / (cfg.concentration_half_sat + raw + 1e-12)
        return float(raw), float(sat)

    def _transduce(self, concentration: float, side: str) -> float:
        _, sat = self._transduction_components(concentration)
        attr = "left_adapt" if side == "left" else "right_adapt"
        adapt = float(getattr(self.agent, attr))

        # Output at time t depends on the current drive and the adaptation state
        # carried into the sample. This avoids using a future-updated state in
        # the same observation.
        response = 0.72 * sat + 0.28 * max(sat - adapt, 0.0)

        # Then advance adaptation for the next sample. The exact zero-order-hold
        # update solves da/dt=(s-a)/tau for a piecewise-constant drive over dt.
        alpha = self._adaptation_alpha()
        next_adapt = adapt + alpha * (sat - adapt)
        setattr(self.agent, attr, next_adapt)

        # This is an explicit phenomenological benchmark transduction, not a
        # receptor-kinetics or ORN firing-rate claim.
        return float(np.clip(response, 0.0, 1.0))

    def _current_observation_key(self) -> tuple[int, float, float, float, float]:
        a = self.agent
        return (
            int(a.steps),
            float(a.x),
            float(a.y),
            float(a.heading),
            float(self.plume.t),
        )

    def _compute_observation(self) -> Observation:
        (lx, ly), (rx, ry) = self._antenna_positions()
        left_concentration = float(self.plume.concentration(lx, ly))
        right_concentration = float(self.plume.concentration(rx, ry))
        left_raw, left_sat = self._transduction_components(left_concentration)
        right_raw, right_sat = self._transduction_components(right_concentration)
        left_before = float(self.agent.left_adapt)
        right_before = float(self.agent.right_adapt)
        left = self._transduce(left_concentration, "left")
        right = self._transduce(right_concentration, "right")
        left_after = float(self.agent.left_adapt)
        right_after = float(self.agent.right_adapt)

        self._sensor_trace_cache = SensorTrace(
            left=AntennaSample(
                x=float(lx),
                y=float(ly),
                concentration=left_concentration,
                gain_scaled_concentration=left_raw,
                saturation=left_sat,
                adaptation_before=left_before,
                adaptation_after=left_after,
                response=left,
            ),
            right=AntennaSample(
                x=float(rx),
                y=float(ry),
                concentration=right_concentration,
                gain_scaled_concentration=right_raw,
                saturation=right_sat,
                adaptation_before=right_before,
                adaptation_after=right_after,
                response=right,
            ),
            adaptation_alpha=self._adaptation_alpha(),
        )

        # Rotate the world-frame downwind vector into body coordinates by -heading.
        # Controllers therefore receive local airflow, not privileged world heading.
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

    def observe(self) -> Observation:
        """Return the sensory state for the current physical simulator state.

        Observation is idempotent at a fixed (agent state, plume time). This is
        scientifically important because sensory adaptation is stateful: logging,
        rendering, or debugging must not advance adaptation merely by reading the
        same observation more than once.
        """
        key = self._current_observation_key()
        if key == self._observation_key and self._observation_cache is not None:
            return self._observation_cache
        observation = self._compute_observation()
        self._observation_key = key
        self._observation_cache = observation
        return observation

    def sensor_trace(self) -> SensorTrace:
        """Return the trace associated with the exact cached observation.

        Calling this accessor cannot advance plume time, adaptation, or agent
        state. It forces `observe()` only when no observation exists for the
        current physical state, then returns the trace generated by that sample.
        """
        self.observe()
        if self._sensor_trace_cache is None:
            raise RuntimeError("sensor trace was not produced with the observation")
        return self._sensor_trace_cache

    def step(self, turn_command: float, speed_scale: float = 1.0) -> tuple[Observation, bool]:
        a = self.agent
        turn = float(np.clip(turn_command, -1.0, 1.0)) * self.arena.max_turn_rate
        a.heading = float((a.heading + turn * self.arena.dt + np.pi) % (2 * np.pi) - np.pi)
        speed = self.arena.speed * float(np.clip(speed_scale, 0.0, 1.5))
        old_x, old_y = a.x, a.y
        a.x += np.cos(a.heading) * speed * self.arena.dt
        a.y += np.sin(a.heading) * speed * self.arena.dt

        # Reflect rather than pinning an agent against a boundary. The arena wall
        # is part of the simulator, not a source-position cue exposed to controllers.
        if a.x < 0.0 or a.x > self.arena.width:
            a.x = float(np.clip(a.x, 0.0, self.arena.width))
            a.heading = float((np.pi - a.heading + np.pi) % (2 * np.pi) - np.pi)
        if a.y < 0.0 or a.y > self.arena.height:
            a.y = float(np.clip(a.y, 0.0, self.arena.height))
            a.heading = float((-a.heading + np.pi) % (2 * np.pi) - np.pi)

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
