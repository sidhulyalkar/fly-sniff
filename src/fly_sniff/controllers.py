from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .env import Observation


@dataclass(frozen=True)
class Action:
    turn: float
    speed: float = 1.0


class Controller(ABC):
    name = "controller"

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float]:
        return {}


class RandomWalkController(Controller):
    name = "random-walk"

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.turn = 0.0

    def act(self, obs: Observation) -> Action:
        self.turn = 0.88 * self.turn + 0.12 * float(self.rng.normal())
        return Action(float(np.tanh(self.turn)))


class CastSurgeController(Controller):
    """Transparent non-connectomic plume-search baseline.

    Odor present -> turn toward upwind heading with a small bilateral correction.
    Odor absent -> cast crosswind with an expanding sinusoidal sweep.
    """

    name = "cast-surge"

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.phase = float(self.rng.uniform(0, 2 * np.pi))
        self.last_odor = 0.0

    @staticmethod
    def _angle_error(target: float, current: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))

    def act(self, obs: Observation) -> Action:
        self.last_odor = obs.mean_odor
        if obs.mean_odor > 0.06:
            # Wind travels +x; source is upwind at pi.
            err = self._angle_error(np.pi, obs.heading)
            turn = 0.72 * np.tanh(err) + 0.28 * np.tanh(5.0 * obs.odor_delta)
            return Action(float(np.clip(turn, -1, 1)), 1.05)
        self.phase += 0.16
        return Action(float(0.78 * np.sin(self.phase)), 0.78)

    def diagnostics(self) -> dict[str, float]:
        return {"odor": self.last_odor}


class BilateralProxyController(Controller):
    """Biology-inspired proxy used only to develop plumbing and visualization.

    This is deliberately labelled PROXY in every rendered output. It is not a
    MaleCNS result and must never be renamed to imply otherwise.
    """

    name = "bilateral-proxy"

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.memory = 0.0
        self.cast_phase = float(self.rng.uniform(0, 2 * np.pi))
        self._diag: dict[str, float] = {}

    def act(self, obs: Observation) -> Action:
        self.memory = 0.94 * self.memory + 0.06 * obs.mean_odor
        if obs.mean_odor > 0.045 or self.memory > 0.025:
            upwind_err = np.arctan2(np.sin(np.pi - obs.heading), np.cos(np.pi - obs.heading))
            hdc = float(np.tanh(2.0 * self.memory + 4.0 * obs.odor_delta))
            left_dn = float(np.clip(0.5 - 0.28 * upwind_err - 0.22 * hdc, 0, 1))
            right_dn = float(np.clip(0.5 + 0.28 * upwind_err + 0.22 * hdc, 0, 1))
            turn = right_dn - left_dn
            self._diag = {"hdc_proxy": hdc, "dn_left": left_dn, "dn_right": right_dn}
            return Action(float(np.clip(turn, -1, 1)), 1.0)
        self.cast_phase += 0.12
        turn = float(0.68 * np.sin(self.cast_phase))
        self._diag = {"hdc_proxy": 0.0, "dn_left": 0.5 - turn / 2, "dn_right": 0.5 + turn / 2}
        return Action(turn, 0.76)

    def diagnostics(self) -> dict[str, float]:
        return self._diag.copy()
