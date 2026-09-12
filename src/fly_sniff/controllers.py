from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .env import Observation


@dataclass(frozen=True)
class Action:
    turn: float
    speed: float = 1.0


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _wind_target_error(obs: Observation, offset_from_downwind: float) -> float:
    """Return a target bearing in the fly's body frame.

    `Observation.wind_*_body` is the sensed downwind airflow vector after the
    environment rotates the global wind into the animal's body frame. Adding
    pi points upwind; +/- pi/2 points crosswind. The controller therefore does
    not need privileged world heading or source geometry to orient to airflow.
    """
    if float(np.hypot(obs.wind_x_body, obs.wind_y_body)) < 1e-12:
        return 0.0
    downwind_body = float(np.arctan2(obs.wind_y_body, obs.wind_x_body))
    return _wrap_angle(downwind_body + offset_from_downwind)


class Controller(ABC):
    name = "controller"

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float]:
        return {}

    def activity_snapshot(self, *, limit: int = 256) -> dict[str, Any] | None:
        """Return sparse neural activity for visualization when it really exists.

        Non-neural controllers intentionally return ``None``. Renderers must not
        synthesize neuron firing from proxy diagnostics or steering commands.
        """
        return None


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

    Odor present -> surge upwind with bilateral correction.
    Odor absent -> cast crosswind, periodically reversing direction.

    Airflow orientation comes only from the body-frame wind observation. This
    avoids quietly giving the baseline a privileged world-frame wind bearing.
    """

    name = "cast-surge"

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.cast_dir = 1 if self.rng.random() < 0.5 else -1
        self.lost_steps = 0
        self.last_odor = 0.0
        self._diag: dict[str, float] = {}

    def act(self, obs: Observation) -> Action:
        self.last_odor = obs.mean_odor
        if obs.mean_odor > 0.06:
            self.lost_steps = 0
            upwind_err = _wind_target_error(obs, np.pi)
            # odor_delta = right - left. Positive therefore calls for a
            # clockwise/rightward correction, which is negative heading change.
            bilateral = float(-np.tanh(18.0 * obs.odor_delta))
            turn = 0.70 * np.tanh(1.4 * upwind_err) + 0.30 * bilateral
            self._diag = {
                "odor": float(obs.mean_odor),
                "odor_delta": float(obs.odor_delta),
                "upwind_error": float(upwind_err),
                "mode_surge": 1.0,
            }
            return Action(float(np.clip(turn, -1.0, 1.0)), 1.05)

        self.lost_steps += 1
        if self.lost_steps % 70 == 0:
            self.cast_dir *= -1
        crosswind_err = _wind_target_error(obs, self.cast_dir * np.pi / 2.0)
        turn = float(np.tanh(1.5 * crosswind_err))
        self._diag = {
            "odor": float(obs.mean_odor),
            "odor_delta": float(obs.odor_delta),
            "upwind_error": float(_wind_target_error(obs, np.pi)),
            "mode_surge": 0.0,
        }
        return Action(turn, 0.72)

    def diagnostics(self) -> dict[str, float]:
        return self._diag.copy()


class BilateralProxyController(Controller):
    """Biology-inspired proxy used only to develop plumbing and visualization.

    Odor gates anemotaxis; body-frame airflow provides the upwind bearing; the
    instantaneous bilateral odor difference provides a small steering bias.
    This remains a transparent engineering proxy, not a MaleCNS result.
    """

    name = "bilateral-proxy"

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.memory = 0.0
        self.cast_dir = 1 if self.rng.random() < 0.5 else -1
        self.lost_steps = 0
        self._diag: dict[str, float] = {}

    def act(self, obs: Observation) -> Action:
        self.memory = 0.93 * self.memory + 0.07 * obs.mean_odor
        recently_detected = self.memory > 0.03 and self.lost_steps < 20
        if obs.mean_odor > 0.045 or recently_detected:
            if obs.mean_odor > 0.045:
                self.lost_steps = 0
            else:
                self.lost_steps += 1
            upwind_err = _wind_target_error(obs, np.pi)
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
                "odor_delta": float(obs.odor_delta),
                "odor_memory": float(self.memory),
                "upwind_error": float(upwind_err),
                "dn_left": left_dn,
                "dn_right": right_dn,
                "mode_surge": 1.0,
            }
            return Action(turn, 1.0)

        self.lost_steps += 1
        if self.lost_steps % 60 == 0:
            self.cast_dir *= -1
        crosswind_err = _wind_target_error(obs, self.cast_dir * np.pi / 2.0)
        turn = float(np.tanh(1.5 * crosswind_err))
        self._diag = {
            "hdc_proxy": 0.0,
            "odor_delta": float(obs.odor_delta),
            "odor_memory": float(self.memory),
            "upwind_error": float(_wind_target_error(obs, np.pi)),
            "dn_left": float(np.clip(0.5 - turn / 2.0, 0.0, 1.0)),
            "dn_right": float(np.clip(0.5 + turn / 2.0, 0.0, 1.0)),
            "mode_surge": 0.0,
        }
        return Action(turn, 0.70)

    def diagnostics(self) -> dict[str, float]:
        return self._diag.copy()
