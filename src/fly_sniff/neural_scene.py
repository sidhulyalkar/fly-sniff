from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph import GraphBundle

SCENE_SCHEMA_VERSION = 1


def _coerce_xyz(value: Any) -> tuple[float, float, float] | None:
    """Parse a neuPrint-style soma location without inventing coordinates."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        if "coordinates" in value:
            value = value["coordinates"]
        elif all(key in value for key in ("x", "y", "z")):
            value = [value["x"], value["y"], value["z"]]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None
        return _coerce_xyz(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            xyz = tuple(float(component) for component in value[:3])
        except (TypeError, ValueError):
            return None
        if all(np.isfinite(component) for component in xyz):
            return xyz
    return None


def _position_column(nodes: pd.DataFrame) -> str:
    for column in ("somaLocation", "soma_location", "soma"):
        if column in nodes.columns:
            return column
    raise ValueError(
        "nodes have no soma coordinate column. Enrich the graph from neuPrint "
        "before exporting an anatomical scene; synthetic positions are forbidden."
    )


def _node_roles(bundle: GraphBundle) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for role, body_ids in bundle.roles.items():
        for body_id in body_ids:
            result.setdefault(int(body_id), []).append(str(role))
    return result


def _normalize_projection(points: np.ndarray) -> np.ndarray:
    """Robustly scale an anatomical X/Z projection to [-1, 1]."""
    if not len(points):
        return points
    lo = np.quantile(points, 0.01, axis=0)
    hi = np.quantile(points, 0.99, axis=0)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1.0)
    scaled = 2.0 * (points - center) / span
    return np.clip(scaled, -1.2, 1.2)


def build_neural_scene(bundle: GraphBundle) -> dict[str, Any]:
    """Export exact graph geometry for an anatomical social replay.

    Only neurons with source-provided soma coordinates are included. Missing
    positions are reported, never filled with fabricated force-layout points.
    """
    bundle.validate(require_sign="sign" in bundle.edges.columns)
    position_column = _position_column(bundle.nodes)
    roles = _node_roles(bundle)
    type_column = "type" if "type" in bundle.nodes.columns else None
    instance_column = "instance" if "instance" in bundle.nodes.columns else None

    positioned: list[dict[str, Any]] = []
    missing_position = 0
    raw_projection: list[tuple[float, float]] = []
    for row in bundle.nodes.itertuples(index=False):
        body_id = int(getattr(row, "bodyId"))
        xyz = _coerce_xyz(getattr(row, position_column))
        if xyz is None:
            missing_position += 1
            continue
        raw_projection.append((xyz[0], xyz[2]))
        positioned.append(
            {
                "body_id": body_id,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "type": str(getattr(row, type_column)) if type_column else "",
                "instance": str(getattr(row, instance_column)) if instance_column else "",
                "roles": sorted(roles.get(body_id, [])),
            }
        )

    projection = _normalize_projection(np.asarray(raw_projection, dtype=float))
    for node, projected in zip(positioned, projection, strict=True):
        node["screen_x"] = float(projected[0])
        node["screen_y"] = float(projected[1])

    positioned_ids = {node["body_id"] for node in positioned}
    edges: list[dict[str, Any]] = []
    for row in bundle.edges.itertuples(index=False):
        source = int(getattr(row, "source"))
        target = int(getattr(row, "target"))
        if source not in positioned_ids or target not in positioned_ids:
            continue
        edge = {
            "source": source,
            "target": target,
            "weight": float(getattr(row, "weight")),
        }
        if hasattr(row, "sign"):
            edge["sign"] = int(getattr(row, "sign"))
        edges.append(edge)

    qualification = (bundle.manifest or {}).get("qualification_status", "candidate")
    claim_label = (
        "QUALIFIED MALECNS MODELED ACTIVITY"
        if qualification == "qualified"
        else "CANDIDATE MALECNS STRUCTURE • NOT A QUALIFIED RESULT"
    )
    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "dataset": (bundle.manifest or {}).get("dataset", "male-cns:v1.0"),
        "qualification_status": str(qualification),
        "claim_label": claim_label,
        "coordinate_source": position_column,
        "projection": "source anatomical x/z coordinates, robustly normalized",
        "nodes_total": int(len(bundle.nodes)),
        "nodes_positioned": int(len(positioned)),
        "nodes_missing_position": int(missing_position),
        "nodes": positioned,
        "edges": edges,
    }


def write_neural_scene(path: str | Path, scene: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scene, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export real MaleCNS neuron coordinates for cinematic replay"
    )
    parser.add_argument("graph", help="candidate or qualified GraphBundle directory")
    parser.add_argument(
        "--output",
        default="artifacts/showcase/malecns-neural-scene.json",
    )
    args = parser.parse_args()
    scene = build_neural_scene(GraphBundle.load(args.graph))
    output = write_neural_scene(args.output, scene)
    print(output)
    print(
        f"positioned {scene['nodes_positioned']}/{scene['nodes_total']} neurons; "
        f"{len(scene['edges'])} drawable edges"
    )
    print(scene["claim_label"])


if __name__ == "__main__":
    main()
