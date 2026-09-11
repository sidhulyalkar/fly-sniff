from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ArenaConfig, PlumeConfig


@dataclass
class Puff:
    x: float
    y: float
    sigma: float
    mass: float
    age: float = 0.0


class TurbulentPlume:
    """Deterministic 2-D stochastic puff plume for paired controller evaluation.

    The plume is intentionally a benchmark model, not a CFD claim. All controllers
    evaluated under one episode seed see the exact same exogenous plume realization.
    """

    def __init__(self, arena: ArenaConfig, config: PlumeConfig, seed: int):
        self.arena = arena
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.puffs: list[Puff] = []
        self.t = 0.0
        self._emit_accumulator = 0.0

    @property
    def wind_vector(self) -> np.ndarray:
        # Air and odor travel downwind in +x. The source is therefore upwind at -x.
        return np.array([self.config.wind_speed, 0.0], dtype=float)

    def step(self) -> None:
        dt = self.arena.dt
        self.t += dt
        self._emit_accumulator += self.config.emission_rate_hz * dt
        n_emit = int(self._emit_accumulator)
        self._emit_accumulator -= n_emit
        if self.rng.random() < self._emit_accumulator:
            n_emit += 1
            self._emit_accumulator = 0.0

        for _ in range(n_emit):
            self.puffs.append(
                Puff(
                    x=self.arena.source_x,
                    y=self.arena.source_y + self.rng.normal(0.0, 0.03),
                    sigma=self.config.initial_sigma,
                    mass=self.config.puff_mass,
                )
            )

        updated: list[Puff] = []
        for puff in self.puffs:
            puff.age += dt
            phase = self.config.meander_frequency * self.t + 0.19 * puff.age
            dy = self.config.meander_amplitude * np.sin(phase) * dt
            dy += self.rng.normal(0.0, self.config.crosswind_noise * np.sqrt(dt))
            puff.x += self.config.wind_speed * dt
            puff.y += dy
            puff.sigma = np.sqrt(
                self.config.initial_sigma**2 + 2.0 * self.config.diffusion_rate * puff.age
            )
            if (
                -0.5 <= puff.x <= self.arena.width + 0.5
                and -1.0 <= puff.y <= self.arena.height + 1.0
            ):
                updated.append(puff)
        self.puffs = updated[-self.config.max_puffs :]

    def concentration(self, x: float, y: float) -> float:
        if not self.puffs:
            return 0.0
        px = np.fromiter((p.x for p in self.puffs), dtype=float)
        py = np.fromiter((p.y for p in self.puffs), dtype=float)
        sigma = np.fromiter((p.sigma for p in self.puffs), dtype=float)
        mass = np.fromiter((p.mass for p in self.puffs), dtype=float)
        d2 = (px - x) ** 2 + (py - y) ** 2
        # A 2-D Gaussian puff field. Absolute units are arbitrary and explicitly
        # treated as simulator units; sensory transduction is a separate model.
        c = mass * np.exp(-0.5 * d2 / np.maximum(sigma**2, 1e-9))
        c /= 2.0 * np.pi * np.maximum(sigma**2, 1e-9)
        return float(c.sum())

    def snapshot(self, max_points: int = 450) -> np.ndarray:
        if not self.puffs:
            return np.empty((0, 3), dtype=float)
        stride = max(1, len(self.puffs) // max_points)
        chosen = self.puffs[::stride][:max_points]
        return np.asarray([[p.x, p.y, p.sigma] for p in chosen], dtype=float)
