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

    Odor present -> surge upwind with bilateral gradient correction.
    Odor absent -> cast crosswind, periodically reversing direction.
    """

    name = "cast-surge"

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.cast_dir = 1 if self.rng.random() < 0.5 else -1
        self.lost_steps = 0
        self.last_odor = 0.0

    @staticmethod
    def _angle_error(target: float, current: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))

    def act(self, obs: Observation) -> Action:
        self.last_odor = obs.mean_odor
        if obs.mean_odor > 0.06:
            self.lost_steps = 0
            upwind_err = self._angle_error(np.pi, obs.heading)
            # odor_delta = right - left. Positive therefore calls for a
            # clockwise/rightward correction, which is negative heading change.
            turn = 0.70 * np.tanh(1.4 * upwind_err)
            turn -= 0.30 * np.tanh(18.0 * obs.odor_delta)
            return Action(float(np.clip(turn, -1.0, 1.0)), 1.05)

        self.lost_steps += 1
        if self.lost_steps % 70 == 0:
            self.cast_dir *= -1
        target = self.cast_dir * np.pi / 2.0
        err = self._angle_error(target, obs.heading)
        return Action(float(np.tanh(1.5 * err)), 0.72)

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
        self.cast_dir = 1 if self.rng.random() < 0.5 else -1
        self.lost_steps = 0
        self._diag: dict[str, float] = {}

    @staticmethod
    def _angle_error(target: float, current: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))

    def act(self, obs: Observation) -> Action:
        self.memory = 0.93 * self.memory + 0.07 * obs.mean_odor
        recently_detected = self.memory > 0.03 and self.lost_steps < 20
        if obs.mean_odor > 0.045 or recently_detected:
            if obs.mean_odor > 0.045:
                self.lost_steps = 0
            else:
                self.lost_steps += 1
            upwind_err = self._angle_error(np.pi, obs.heading)
            bilateral = float(-np.tanh(18.0 * obs.odor_delta))
            turn = float(
                np.clip(
                    0.72 * np.tanh(1.35 * upwind_err) + 0.28 * bilateral,
                    -1.0,
                    1.0,
                )
            )
            left_dn = float(np.clip(0.5 - turn / 2.0, 0.0, 1.0))
            right_dn = float(np.clip(0.5 + turn / 2.0, 0.0, 1.0))
            self._diag = {
                "hdc_proxy": bilateral,
                "dn_left": left_dn,
                "dn_right": right_dn,
            }
            return Action(turn, 1.0)

        self.lost_steps += 1
        if self.lost_steps % 60 == 0:
            self.cast_dir *= -1
        target = self.cast_dir * np.pi / 2.0
        err = self._angle_error(target, obs.heading)
        turn = float(np.tanh(1.5 * err))
        self._diag = {
            "hdc_proxy": 0.0,
            "dn_left": 0.5 - turn / 2.0,
            "dn_right": 0.5 + turn / 2.0,
        }
        return Action(turn, 0.70)

    def diagnostics(self) -> dict[str, float]:
        return self._diag.copy()
