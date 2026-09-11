from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .controllers import (
    BilateralProxyController,
    CastSurgeController,
    Controller,
    RandomWalkController,
)
from .party_social import PartyAgent, _make_agent

SCHEMA_VERSION = 1
DEFAULT_SIM_SECONDS = 45.0
DEFAULT_PLUME_POINTS = 220

_CONTROLLER_FACTORIES: dict[str, tuple[str, type[Controller], str]] = {
    "proxy": ("PROXY", BilateralProxyController, "#38BDF8"),
    "random": ("RANDOM", RandomWalkController, "#FB7185"),
    "cast-surge": ("CAST-SURGE", CastSurgeController, "#FBBF24"),
}


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def recording_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _obs_payload(obs) -> dict[str, float]:
    return {
        "left_odor": float(obs.left_odor),
        "right_odor": float(obs.right_odor),
        "mean_odor": float(obs.mean_odor),
        "odor_delta": float(obs.odor_delta),
        "wind_x_body": float(obs.wind_x_body),
        "wind_y_body": float(obs.wind_y_body),
        "heading": float(obs.heading),
    }


def _agent_payload(
    live: PartyAgent,
    *,
    action: dict[str, float] | None = None,
) -> dict[str, Any]:
    agent = live.env.agent
    return {
        "label": live.label,
        "controller": live.controller.name,
        "x": float(agent.x),
        "y": float(agent.y),
        "heading": float(agent.heading),
        "path_length": float(agent.path_length),
        "distance_to_source": float(live.env.distance_to_source),
        "found": bool(agent.found),
        "done": bool(live.done),
        "observation": _obs_payload(live.obs),
        "action": action or {"turn": 0.0, "speed": 0.0},
        "diagnostics": {
            key: float(value)
            for key, value in live.controller.diagnostics().items()
        },
    }


def _plume_payload(live: PartyAgent, max_points: int) -> list[list[float]]:
    return live.env.plume.snapshot(max_points=max_points).astype(float).tolist()


def _assert_shared_plume(agents: list[PartyAgent], max_points: int) -> None:
    if len(agents) < 2:
        return
    reference = agents[0].env.plume.snapshot(max_points=max_points)
    for live in agents[1:]:
        candidate = live.env.plume.snapshot(max_points=max_points)
        same_shape = reference.shape == candidate.shape
        same_values = same_shape and np.allclose(
            reference,
            candidate,
            rtol=0.0,
            atol=0.0,
        )
        if not same_values:
            raise RuntimeError(
                "paired showcase agents no longer share the same exogenous plume state"
            )


def build_recording(
    *,
    seed: int = 13013,
    sim_seconds: float = DEFAULT_SIM_SECONDS,
    plume_points: int = DEFAULT_PLUME_POINTS,
    controller_names: tuple[str, ...] = ("proxy", "random"),
) -> dict[str, Any]:
    if sim_seconds <= 0:
        raise ValueError("sim_seconds must be > 0")
    if plume_points < 1:
        raise ValueError("plume_points must be >= 1")
    if not controller_names:
        raise ValueError("at least one controller is required")

    agents: list[PartyAgent] = []
    colors: dict[str, str] = {}
    for name in controller_names:
        try:
            label, factory, color = _CONTROLLER_FACTORIES[name]
        except KeyError as exc:
            raise ValueError(f"unknown controller {name!r}") from exc
        agents.append(_make_agent(label, factory, seed, color))
        colors[label] = color

    arena = agents[0].env.arena
    steps = min(arena.max_steps, int(round(sim_seconds / arena.dt)))
    frames: list[dict[str, Any]] = []

    def capture(actions: list[dict[str, float]] | None = None) -> None:
        _assert_shared_plume(agents, plume_points)
        if actions is None:
            actions = [{"turn": 0.0, "speed": 0.0} for _ in agents]
        frames.append(
            {
                "t": float(agents[0].env.agent.steps * arena.dt),
                "plume_t": float(agents[0].env.plume.t),
                "plume": _plume_payload(agents[0], plume_points),
                "agents": [
                    _agent_payload(live, action=action)
                    for live, action in zip(agents, actions, strict=True)
                ],
            }
        )

    capture()
    for _ in range(steps):
        actions: list[dict[str, float]] = []
        for live in agents:
            if live.done:
                actions.append({"turn": 0.0, "speed": 0.0})
            else:
                action = live.controller.act(live.obs)
                actions.append(
                    {
                        "turn": float(action.turn),
                        "speed": float(action.speed),
                    }
                )

        # Store the exact observation -> command pair at the state that generated it.
        _assert_shared_plume(agents, plume_points)
        frames[-1]["agents"] = [
            _agent_payload(live, action=action)
            for live, action in zip(agents, actions, strict=True)
        ]

        for live, action in zip(agents, actions, strict=True):
            if live.done:
                # The animal stays fixed after success, but exogenous plume time
                # keeps moving so paired controllers remain on one frozen plume.
                live.env.plume.step()
            else:
                live.obs, live.done = live.env.step(
                    action["turn"],
                    action["speed"],
                )
        capture()
        if all(live.done for live in agents):
            break

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": "DEVELOPMENT PROXY • NOT A MALECNS RESULT",
        "seed": int(seed),
        "dt": float(arena.dt),
        "arena": {
            "width": float(arena.width),
            "height": float(arena.height),
            "source_x": float(arena.source_x),
            "source_y": float(arena.source_y),
            "source_radius": float(arena.source_radius),
            "antenna_separation": float(arena.antenna_separation),
        },
        "controllers": [
            {
                "label": live.label,
                "name": live.controller.name,
                "color": colors[live.label],
            }
            for live in agents
        ],
        "frames": frames,
    }
    return {
        "recording": payload,
        "recording_sha256": recording_sha256(payload),
    }


def write_recording(path: str | Path, bundle: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return output


def load_recording(path: str | Path) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text())
    payload = bundle.get("recording")
    expected = bundle.get("recording_sha256")
    if not isinstance(payload, dict) or not isinstance(expected, str):
        raise ValueError("invalid recording bundle")
    actual = recording_sha256(payload)
    if actual != expected:
        raise ValueError(
            f"recording SHA-256 mismatch: expected {expected}, got {actual}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported recording schema {payload.get('schema_version')!r}"
        )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record an auditable fly-sniff social episode"
    )
    parser.add_argument(
        "--output",
        default="artifacts/showcase/who-farted-run.json",
    )
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--sim-seconds", type=float, default=DEFAULT_SIM_SECONDS)
    parser.add_argument("--plume-points", type=int, default=DEFAULT_PLUME_POINTS)
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["proxy", "random"],
        choices=sorted(_CONTROLLER_FACTORIES),
    )
    args = parser.parse_args()
    bundle = build_recording(
        seed=args.seed,
        sim_seconds=args.sim_seconds,
        plume_points=args.plume_points,
        controller_names=tuple(args.controllers),
    )
    path = write_recording(args.output, bundle)
    print(path)
    print(bundle["recording_sha256"])


if __name__ == "__main__":
    main()
