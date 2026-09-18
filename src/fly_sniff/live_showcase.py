from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .controllers import Action, BilateralProxyController, Controller
from .env import FlySniffEnv, Observation
from .graph import GraphBundle, MaleCNSRateController
from .rewire import degree_preserving_rewire


SCHEMA = "fly-sniff-live-showcase-v1"


class SensoryMaskController(Controller):
    """Apply an explicit sensory lesion before delegating to another controller."""

    def __init__(self, inner: Controller, *, left_scale: float = 1.0, right_scale: float = 1.0):
        self.inner = inner
        self.left_scale = float(left_scale)
        self.right_scale = float(right_scale)
        self.name = f"{inner.name}-masked"

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.inner.reset(seed)

    def act(self, obs: Observation) -> Action:
        left = float(np.clip(obs.left_odor * self.left_scale, 0.0, 1.0))
        right = float(np.clip(obs.right_odor * self.right_scale, 0.0, 1.0))
        masked = Observation(
            left_odor=left,
            right_odor=right,
            mean_odor=0.5 * (left + right),
            odor_delta=right - left,
            wind_x_body=obs.wind_x_body,
            wind_y_body=obs.wind_y_body,
            heading=obs.heading,
        )
        return self.inner.act(masked)

    def diagnostics(self) -> dict[str, float]:
        return self.inner.diagnostics()


def _condition(
    key: str,
    label: str,
    controller: Controller,
    *,
    intervention: str,
    evidence_level: str,
    description: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "controller": controller,
        "intervention": intervention,
        "evidence_level": evidence_level,
        "description": description,
    }


def development_conditions() -> list[dict[str, Any]]:
    return [
        _condition(
            "proxy-intact",
            "Bilateral proxy",
            BilateralProxyController(),
            intervention="none",
            evidence_level="development_proxy",
            description=(
                "Biology-inspired bilateral odor controller used only to develop the showcase. "
                "It is not a MaleCNS result."
            ),
        ),
        _condition(
            "left-antenna-off",
            "Left antenna off",
            SensoryMaskController(BilateralProxyController(), left_scale=0.0),
            intervention="left odor channel set to zero before the controller",
            evidence_level="development_proxy",
            description=(
                "Engineered sensory ablation used to make the behavioral consequence of asymmetric "
                "odor input visible. It is not a biological lesion experiment."
            ),
        ),
        _condition(
            "odor-blind",
            "Odor blind",
            SensoryMaskController(BilateralProxyController(), left_scale=0.0, right_scale=0.0),
            intervention="both odor channels set to zero before the controller",
            evidence_level="development_proxy",
            description=(
                "Negative-control controller receives wind but no odor. It tests whether the visual "
                "demo leaks source information through another channel."
            ),
        ),
    ]


def graph_conditions(
    bundle: GraphBundle,
    *,
    rewire_seed: int,
    allow_candidate: bool,
) -> list[dict[str, Any]]:
    status = str((bundle.manifest or {}).get("qualification_status", "unsealed"))
    if status != "qualified" and not allow_candidate:
        raise ValueError(
            "claim-bearing live showcase requires a qualified GraphBundle; "
            "pass --allow-candidate only for visibly labelled development output"
        )

    require_qualified = status == "qualified"
    rewired = degree_preserving_rewire(bundle, seed=rewire_seed)
    evidence = "qualified_modeled_circuit" if require_qualified else "candidate_modeled_circuit"

    return [
        _condition(
            "malecns-intact",
            "MaleCNS topology",
            MaleCNSRateController(bundle, require_qualified=require_qualified),
            intervention="none",
            evidence_level=evidence,
            description="Modeled rate dynamics over the supplied MaleCNS graph topology.",
        ),
        _condition(
            "degree-preserving-rewire",
            "Degree-preserving rewire",
            MaleCNSRateController(rewired, require_qualified=False),
            intervention=f"directed graph degree-preserving rewire, seed {rewire_seed}",
            evidence_level="matched_topology_control",
            description=(
                "Same nodes and directed degree structure with topology disrupted by the frozen "
                "rewire procedure."
            ),
        ),
        _condition(
            "malecns-odor-blind",
            "MaleCNS odor blind",
            SensoryMaskController(
                MaleCNSRateController(bundle, require_qualified=require_qualified),
                left_scale=0.0,
                right_scale=0.0,
            ),
            intervention="both odor channels set to zero before sensory injection",
            evidence_level=evidence,
            description="Explicit sensory-ablation control over the same modeled circuit.",
        ),
    ]


def _sample_plume(env: FlySniffEnv, max_points: int = 90) -> list[list[float]]:
    snap = env.plume.snapshot(max_points=max_points)
    return [[round(float(x), 4), round(float(y), 4), round(float(s), 4)] for x, y, s in snap]


def _record_frame(
    env: FlySniffEnv,
    obs: Observation,
    action: Action,
    diag: dict[str, float],
) -> dict[str, Any]:
    return {
        "x": round(float(env.agent.x), 5),
        "y": round(float(env.agent.y), 5),
        "heading": round(float(env.agent.heading), 5),
        "found": bool(env.agent.found),
        "left_odor": round(float(obs.left_odor), 5),
        "right_odor": round(float(obs.right_odor), 5),
        "mean_odor": round(float(obs.mean_odor), 5),
        "odor_delta": round(float(obs.odor_delta), 5),
        "wind_x_body": round(float(obs.wind_x_body), 5),
        "wind_y_body": round(float(obs.wind_y_body), 5),
        "turn": round(float(action.turn), 5),
        "speed": round(float(action.speed), 5),
        "dn_left": round(float(diag.get("dn_left", 0.0)), 5),
        "dn_right": round(float(diag.get("dn_right", 0.0)), 5),
        "activity_mean": round(float(diag.get("activity_mean", 0.0)), 6),
    }


def _topology_payload(bundle: GraphBundle | None) -> dict[str, Any]:
    if bundle is None:
        return {
            "available": False,
            "geometry_kind": "none",
            "label": "No connectome asset loaded",
            "claim_boundary": (
                "The development proxy has no MaleCNS graph behind it. The live page must not "
                "draw decorative neurons and call them anatomy."
            ),
        }

    node_ids = bundle.nodes["bodyId"].astype(int).tolist()
    role_by_id: dict[int, list[str]] = {body_id: [] for body_id in node_ids}
    for role, body_ids in bundle.roles.items():
        for body_id in body_ids:
            if int(body_id) in role_by_id:
                role_by_id[int(body_id)].append(str(role))

    nodes = [
        {"body_id": body_id, "roles": sorted(role_by_id.get(body_id, []))}
        for body_id in node_ids
    ]
    edges = [
        {
            "source": int(row.source),
            "target": int(row.target),
            "weight": float(row.weight),
            "sign": int(getattr(row, "sign", 0)),
        }
        for row in bundle.edges.itertuples(index=False)
    ]
    return {
        "available": True,
        "geometry_kind": "topology_only",
        "label": "Connectome topology",
        "nodes": nodes,
        "edges": edges,
        "qualification_status": str((bundle.manifest or {}).get("qualification_status", "unsealed")),
        "claim_boundary": (
            "This payload contains graph topology and role labels, not anatomical XYZ morphology. "
            "The browser must label it TOPOLOGY, NOT MORPHOLOGY unless an independently sourced "
            "morphology asset is loaded."
        ),
    }


def export_live_showcase(
    output: str | Path,
    *,
    seed: int = 13013,
    seconds: float = 18.0,
    sample_hz: float = 10.0,
    graph_dir: str | Path | None = None,
    rewire_seed: int = 24018,
    allow_candidate: bool = False,
) -> dict[str, Any]:
    if seconds <= 0 or sample_hz <= 0:
        raise ValueError("seconds and sample_hz must be positive")

    arena = ArenaConfig()
    plume = PlumeConfig()
    sensors = SensorConfig()

    bundle = GraphBundle.load(graph_dir) if graph_dir is not None else None
    conditions = (
        graph_conditions(bundle, rewire_seed=rewire_seed, allow_candidate=allow_candidate)
        if bundle is not None
        else development_conditions()
    )

    envs: dict[str, FlySniffEnv] = {}
    controllers: dict[str, Controller] = {}
    observations: dict[str, Observation] = {}
    for condition in conditions:
        key = str(condition["key"])
        env = FlySniffEnv(seed=seed, arena=arena, plume=plume, sensors=sensors)
        controller = condition["controller"]
        # Pair controller-internal stochastic state across conditions so the
        # declared intervention remains the only intentional difference.
        controller.reset(seed + 101)
        envs[key] = env
        controllers[key] = controller
        observations[key] = env.observe()

    stride = max(1, round(1.0 / (arena.dt * sample_hz)))
    max_steps = min(arena.max_steps, max(1, round(seconds / arena.dt)))
    frames: list[dict[str, Any]] = []

    for step in range(max_steps):
        actions: dict[str, Action] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for condition in conditions:
            key = str(condition["key"])
            controller = controllers[key]
            action = controller.act(observations[key])
            actions[key] = action
            diagnostics[key] = controller.diagnostics()

        if step % stride == 0:
            first_key = str(conditions[0]["key"])
            frames.append(
                {
                    "t": round(step * arena.dt, 4),
                    "plume": _sample_plume(envs[first_key]),
                    "agents": {
                        str(condition["key"]): _record_frame(
                            envs[str(condition["key"])],
                            observations[str(condition["key"])],
                            actions[str(condition["key"])],
                            diagnostics[str(condition["key"])],
                        )
                        for condition in conditions
                    },
                }
            )

        done_all = True
        for condition in conditions:
            key = str(condition["key"])
            obs, done = envs[key].step(actions[key].turn, actions[key].speed)
            observations[key] = obs
            done_all = done_all and done
        if done_all:
            break

    public_conditions = [
        {key: value for key, value in condition.items() if key != "controller"}
        for condition in conditions
    ]
    graph_status = str((bundle.manifest or {}).get("qualification_status", "none")) if bundle else "none"
    claim_allowed = bool(bundle and graph_status == "qualified")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "program_id": "olfactory-computation-v0",
        "mode": "graph" if bundle is not None else "development-proxy",
        "claim_allowed": claim_allowed,
        "evidence_level": (
            "qualified_modeled_circuit" if claim_allowed else "development_proxy"
        ),
        "seed": seed,
        "sample_hz": sample_hz,
        "arena": asdict(arena),
        "plume_model": {
            **asdict(plume),
            "claim_boundary": (
                "2-D stochastic puff benchmark, not room CFD. The source is visible to the viewer "
                "only; controllers never receive source coordinates."
            ),
        },
        "sensor_model": asdict(sensors),
        "conditions": public_conditions,
        "frames": frames,
        "connectome": _topology_payload(bundle),
        "claim_boundary": (
            "The live page replays frozen simulator state. Browser animation is presentation only. "
            "Development proxy conditions are not MaleCNS evidence. When a qualified graph is supplied, "
            "neural values are modeled dynamics over structural connectivity, not neural recordings. "
            "A topology view must not be described as anatomical morphology."
        ),
    }

    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export frozen replay data for the browser-based fly-sniff experiment theater"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--graph", help="optional GraphBundle directory")
    parser.add_argument("--rewire-seed", type=int, default=24018)
    parser.add_argument("--allow-candidate", action="store_true")
    args = parser.parse_args(argv)

    payload = export_live_showcase(
        args.output,
        seed=args.seed,
        seconds=args.seconds,
        sample_hz=args.sample_hz,
        graph_dir=args.graph,
        rewire_seed=args.rewire_seed,
        allow_candidate=args.allow_candidate,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "mode": payload["mode"],
                "claim_allowed": payload["claim_allowed"],
                "frames": len(payload["frames"]),
                "conditions": [c["key"] for c in payload["conditions"]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
