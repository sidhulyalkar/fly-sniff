from __future__ import annotations

import numpy as np

from .config import ArenaConfig, PlumeConfig


def concentration_from_snapshot(
    snapshot: np.ndarray,
    puff_mass: float,
    x: float,
    y: float,
) -> float:
    """Reconstruct concentration from a complete ``[x, y, sigma]`` puff snapshot.

    This is the same normalized 2-D Gaussian mixture used by ``TurbulentPlume``.
    It is exact only when the supplied snapshot contains every retained puff.
    Recording metadata therefore marks whether a frame is complete or sampled.
    """
    snapshot = np.asarray(snapshot, dtype=float)
    if snapshot.size == 0:
        return 0.0
    if snapshot.ndim != 2 or snapshot.shape[1] != 3:
        raise ValueError("plume snapshot must have shape (n, 3) with x/y/sigma columns")
    if not np.isfinite(snapshot).all() or not np.isfinite(puff_mass) or puff_mass < 0.0:
        raise ValueError("plume snapshot and puff_mass must be finite; puff_mass must be >= 0")

    dx = snapshot[:, 0] - float(x)
    dy = snapshot[:, 1] - float(y)
    sigma = snapshot[:, 2]
    if (sigma <= 0.0).any():
        raise ValueError("snapshot sigma values must be > 0")
    reach = 4.0 * sigma
    local = (np.abs(dx) <= reach) & (np.abs(dy) <= reach)
    if not local.any():
        return 0.0
    variance = np.maximum(sigma[local] ** 2, 1e-9)
    d2 = dx[local] ** 2 + dy[local] ** 2
    c = float(puff_mass) * np.exp(-0.5 * d2 / variance)
    c /= 2.0 * np.pi * variance
    return float(c.sum())


class TurbulentPlume:
    """Vectorized 2-D stochastic puff plume for paired controller evaluation.

    This is a benchmark model, not a computational-fluid-dynamics claim. All
    controllers evaluated under one episode seed receive the same exogenous
    plume realization. A deterministic pre-roll creates a developed plume
    before the agent is released.
    """

    def __init__(self, arena: ArenaConfig, config: PlumeConfig, seed: int):
        self.arena = arena
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.x = np.empty(0, dtype=float)
        self.y = np.empty(0, dtype=float)
        self.age = np.empty(0, dtype=float)
        self.sigma = np.empty(0, dtype=float)
        self.mass = np.empty(0, dtype=float)

    @property
    def wind_vector(self) -> np.ndarray:
        # Air and odor travel downwind in +x. The source is therefore upwind at -x.
        return np.array([self.config.wind_speed, 0.0], dtype=float)

    def warmup(self) -> None:
        for _ in range(round(self.config.warmup_s / self.arena.dt)):
            self.step()

    def step(self) -> None:
        dt = self.arena.dt
        self.t += dt
        n_emit = int(self.rng.poisson(self.config.emission_rate_hz * dt))
        if n_emit:
            self.x = np.concatenate((self.x, np.full(n_emit, self.arena.source_x)))
            self.y = np.concatenate(
                (self.y, self.arena.source_y + self.rng.normal(0.0, 0.03, n_emit))
            )
            self.age = np.concatenate((self.age, np.zeros(n_emit)))
            self.sigma = np.concatenate(
                (self.sigma, np.full(n_emit, self.config.initial_sigma))
            )
            self.mass = np.concatenate((self.mass, np.full(n_emit, self.config.puff_mass)))

        if not len(self.x):
            return

        self.age += dt
        phase = self.config.meander_frequency * self.t + 0.19 * self.age
        self.y += self.config.meander_amplitude * np.sin(phase) * dt
        self.y += self.rng.normal(
            0.0,
            self.config.crosswind_noise * np.sqrt(dt),
            len(self.x),
        )
        self.x += self.config.wind_speed * dt
        self.sigma = np.sqrt(
            self.config.initial_sigma**2 + 2.0 * self.config.diffusion_rate * self.age
        )

        keep = (
            (self.x >= -0.5)
            & (self.x <= self.arena.width + 0.5)
            & (self.y >= -1.0)
            & (self.y <= self.arena.height + 1.0)
        )
        kept_idx = np.flatnonzero(keep)
        if len(kept_idx) > self.config.max_puffs:
            kept_idx = kept_idx[-self.config.max_puffs :]
        self.x = self.x[kept_idx]
        self.y = self.y[kept_idx]
        self.age = self.age[kept_idx]
        self.sigma = self.sigma[kept_idx]
        self.mass = self.mass[kept_idx]

    def concentration(self, x: float, y: float) -> float:
        if not len(self.x):
            return 0.0
        dx = self.x - x
        dy = self.y - y
        reach = 4.0 * self.sigma
        local = (np.abs(dx) <= reach) & (np.abs(dy) <= reach)
        if not local.any():
            return 0.0
        sigma = self.sigma[local]
        d2 = dx[local] ** 2 + dy[local] ** 2
        variance = np.maximum(sigma**2, 1e-9)
        c = self.mass[local] * np.exp(-0.5 * d2 / variance)
        c /= 2.0 * np.pi * variance
        return float(c.sum())

    def snapshot(self, max_points: int = 450) -> np.ndarray:
        if max_points < 1:
            raise ValueError("max_points must be >= 1")
        if not len(self.x):
            return np.empty((0, 3), dtype=float)
        stride = max(1, len(self.x) // max_points)
        return np.column_stack(
            (self.x[::stride], self.y[::stride], self.sigma[::stride])
        )[:max_points]

    def snapshot_is_complete(self, max_points: int) -> bool:
        """Whether ``snapshot(max_points)`` contains every retained puff."""
        if max_points < 1:
            raise ValueError("max_points must be >= 1")
        return len(self.x) <= max_points
