from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph import GraphBundle


ANNOTATION_SOURCE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
SKELETON_BASE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
    "skeletons-malecns/skeletons-swc"
)
SKELETON_MAX_BYTES = 64 * 1024 * 1024
USER_AGENT = "fly-sniff-live-showcase/1.0"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyz(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        return None
    xyz = tuple(float(x) for x in value)
    if not all(np.isfinite(xyz)):
        return None
    return xyz


def build_soma_context(
    annotations: str | Path,
    output: str | Path,
    *,
    max_points: int = 12000,
) -> dict[str, Any]:
    if max_points < 1:
        raise ValueError("max_points must be positive")

    source = Path(annotations).expanduser().resolve()
    frame = pd.read_feather(source)
    required = {"bodyId", "somaLocation"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MaleCNS annotations missing required columns: {sorted(missing)}")

    if "status" in frame.columns:
        traced = frame[frame["status"].fillna("").astype(str).eq("Traced")].copy()
    else:
        traced = frame.copy()

    rows: list[dict[str, Any]] = []
    for row in traced.itertuples(index=False):
        body_id = int(getattr(row, "bodyId"))
        position = _xyz(getattr(row, "somaLocation"))
        if position is None:
            continue
        rows.append(
            {
                "body_id": body_id,
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "type": str(getattr(row, "type", "") or ""),
                "superclass": str(getattr(row, "superclass", "") or ""),
            }
        )

    if not rows:
        raise ValueError("no valid somaLocation rows found")

    rows.sort(key=lambda item: item["body_id"])
    available = len(rows)
    if available > max_points:
        indices = np.linspace(0, available - 1, max_points, dtype=int)
        sampled = [rows[int(index)] for index in indices]
    else:
        sampled = rows

    xyz = np.asarray([[row["x"], row["y"], row["z"]] for row in sampled], dtype=float)
    lower = xyz.min(axis=0)
    upper = xyz.max(axis=0)

    payload: dict[str, Any] = {
        "schema": "fly-sniff-malecns-soma-context-v1",
        "dataset": "male-cns:v1.0",
        "geometry_kind": "measured_soma_xyz",
        "coordinate_space": "MaleCNS EM",
        "coordinate_units": "8nm voxels",
        "source": {
            "url": ANNOTATION_SOURCE,
            "local_sha256": _sha256_file(source),
            "path": str(source),
        },
        "selection": {
            "status_filter": "Traced when status column is present",
            "valid_soma_rows": available,
            "exported_points": len(sampled),
            "sampling": "bodyId-sorted deterministic linspace when bounded",
            "max_points": max_points,
        },
        "bounds": {
            "min": [float(x) for x in lower],
            "max": [float(x) for x in upper],
        },
        "point_meaning": (
            "One measured somaLocation per exported body. Marker size is a display device, "
            "not anatomical cell size. Soma points are context, not neurite morphology."
        ),
        "points": sampled,
        "license": "CC BY",
        "attribution": "MaleCNS / FlyEM / HHMI Janelia and collaborators",
    }

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    return payload


def _download_swc(body_id: int, cache_dir: Path) -> tuple[bytes, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{body_id}.swc"
    url = f"{SKELETON_BASE}/{body_id}.swc"
    if path.is_file() and path.stat().st_size > 0:
        return path.read_bytes(), url

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120.0) as response:
        raw = response.read(SKELETON_MAX_BYTES + 1)
    if len(raw) > SKELETON_MAX_BYTES:
        raise ValueError(f"SWC for body {body_id} exceeds byte cap")
    if not raw.strip():
        raise ValueError(f"SWC for body {body_id} is empty")
    path.write_bytes(raw)
    return raw, url


def _parse_swc_segments(raw: bytes) -> list[list[float]]:
    nodes: dict[int, tuple[float, float, float, int]] = {}
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 7:
            continue
        node_id = int(fields[0])
        x, y, z = float(fields[2]), float(fields[3]), float(fields[4])
        parent = int(fields[6])
        if not all(np.isfinite((x, y, z))):
            continue
        nodes[node_id] = (x, y, z, parent)

    segments: list[list[float]] = []
    for node_id, (x, y, z, parent) in nodes.items():
        if parent < 0 or parent not in nodes:
            continue
        px, py, pz, _ = nodes[parent]
        segments.append([px, py, pz, x, y, z])
    if not segments:
        raise ValueError("SWC contains no valid parent-child segments")
    return segments


def build_selected_skeletons(
    graph_dir: str | Path,
    output: str | Path,
    *,
    cache_dir: str | Path,
    max_neurons: int = 128,
    max_segments_per_neuron: int = 6000,
) -> dict[str, Any]:
    if max_neurons < 1 or max_segments_per_neuron < 1:
        raise ValueError("max_neurons and max_segments_per_neuron must be positive")

    graph = GraphBundle.load(graph_dir)
    body_ids = sorted(graph.nodes["bodyId"].astype(int).unique().tolist())
    if len(body_ids) > max_neurons:
        raise ValueError(
            f"graph contains {len(body_ids)} neurons; refusing to fetch more than "
            f"{max_neurons} without an explicit higher bound"
        )

    cache = Path(cache_dir).expanduser().resolve()
    neurons: list[dict[str, Any]] = []
    for body_id in body_ids:
        raw, url = _download_swc(body_id, cache)
        segments = _parse_swc_segments(raw)
        original_segments = len(segments)
        if original_segments > max_segments_per_neuron:
            indices = np.linspace(
                0,
                original_segments - 1,
                max_segments_per_neuron,
                dtype=int,
            )
            segments = [segments[int(index)] for index in indices]

        roles = sorted(
            role for role, ids in graph.roles.items() if int(body_id) in {int(x) for x in ids}
        )
        neurons.append(
            {
                "body_id": int(body_id),
                "roles": roles,
                "source_url": url,
                "source_sha256": _sha256_bytes(raw),
                "original_segments": original_segments,
                "exported_segments": len(segments),
                "segments": segments,
            }
        )

    payload: dict[str, Any] = {
        "schema": "fly-sniff-malecns-selected-skeletons-v1",
        "dataset": "male-cns:v1.0",
        "geometry_kind": "measured_skeleton_xyz",
        "coordinate_space": "MaleCNS EM",
        "coordinate_units": "8nm voxels",
        "neurons": neurons,
        "decimation": (
            "Each neuron is unmodified when within the segment cap. Larger neurons are "
            "deterministically subsampled for browser presentation only."
        ),
        "claim_boundary": (
            "These are measured centerline skeleton coordinates for the selected bodies. "
            "Animated activity overlays remain modeled dynamics unless independently measured."
        ),
        "license": "CC BY",
        "attribution": "MaleCNS / FlyEM / HHMI Janelia and collaborators",
    }

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build real MaleCNS spatial context assets for the live showcase"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    soma = sub.add_parser("soma", help="export measured somaLocation context from annotations")
    soma.add_argument("annotations")
    soma.add_argument("--output", required=True)
    soma.add_argument("--max-points", type=int, default=12000)

    skel = sub.add_parser("skeletons", help="fetch exact SWC skeletons for a GraphBundle")
    skel.add_argument("graph")
    skel.add_argument("--output", required=True)
    skel.add_argument("--cache-dir", required=True)
    skel.add_argument("--max-neurons", type=int, default=128)
    skel.add_argument("--max-segments-per-neuron", type=int, default=6000)

    args = parser.parse_args(argv)
    if args.command == "soma":
        payload = build_soma_context(
            args.annotations,
            args.output,
            max_points=args.max_points,
        )
        print(
            json.dumps(
                {
                    "output": str(Path(args.output).expanduser().resolve()),
                    "points": len(payload["points"]),
                    "source_sha256": payload["source"]["local_sha256"],
                },
                indent=2,
            )
        )
        return 0

    payload = build_selected_skeletons(
        args.graph,
        args.output,
        cache_dir=args.cache_dir,
        max_neurons=args.max_neurons,
        max_segments_per_neuron=args.max_segments_per_neuron,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "neurons": len(payload["neurons"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
