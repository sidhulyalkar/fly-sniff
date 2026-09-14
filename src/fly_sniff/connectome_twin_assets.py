from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

ASSET_PROTOCOL = "male-cns-connectome-twin-assets-v1"


def read_swc(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an SWC centerline into nodes and parent-index edges.

    MaleCNS SWCs use body-coordinate values in 8 nm units. The caller may
    transform units for rendering, but topology remains the source centerline.
    """
    rows: list[tuple[int, float, float, float, int]] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            raise ValueError(f"invalid SWC row in {path}: {line}")
        node_id = int(fields[0])
        x, y, z = map(float, fields[2:5])
        parent = int(fields[6])
        rows.append((node_id, x, y, z, parent))
    if not rows:
        raise ValueError(f"empty SWC: {path}")
    ids = [row[0] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate SWC node id in {path}")
    index = {node_id: i for i, node_id in enumerate(ids)}
    nodes = np.asarray([[row[1], row[2], row[3]] for row in rows], dtype=float)
    edges: list[tuple[int, int]] = []
    for i, row in enumerate(rows):
        parent = row[4]
        if parent < 0:
            continue
        if parent not in index:
            raise ValueError(f"SWC parent {parent} missing in {path}")
        edges.append((index[parent], i))
    return nodes, np.asarray(edges, dtype=int)


def _decimate_edges(nodes: np.ndarray, edges: np.ndarray, max_segments: int) -> np.ndarray:
    if max_segments <= 0:
        raise ValueError("max_segments must be > 0")
    if len(edges) <= max_segments:
        return edges
    # Deterministic edge sampling for display LOD only. Exact route bundles can
    # set max_segments high enough to retain every centerline edge.
    idx = np.linspace(0, len(edges) - 1, max_segments, dtype=int)
    return edges[idx]


def pack_swc_directory(
    directory: str | Path,
    body_ids: list[int],
    *,
    max_segments_per_neuron: int,
    units_um_per_coordinate: float = 0.008,
) -> dict[str, Any]:
    root = Path(directory)
    neurons: list[dict[str, Any]] = []
    bounds_min = np.full(3, np.inf)
    bounds_max = np.full(3, -np.inf)
    for body_id in body_ids:
        path = root / f"{int(body_id)}.swc"
        if not path.exists():
            raise FileNotFoundError(f"missing MaleCNS skeleton for body {body_id}: {path}")
        nodes, edges = read_swc(path)
        nodes_um = nodes * float(units_um_per_coordinate)
        shown_edges = _decimate_edges(nodes_um, edges, max_segments_per_neuron)
        vertices = np.empty((len(shown_edges) * 2, 3), dtype=np.float32)
        if len(shown_edges):
            vertices[0::2] = nodes_um[shown_edges[:, 0]]
            vertices[1::2] = nodes_um[shown_edges[:, 1]]
            bounds_min = np.minimum(bounds_min, vertices.min(axis=0))
            bounds_max = np.maximum(bounds_max, vertices.max(axis=0))
        neurons.append(
            {
                "body_id": int(body_id),
                "source_file": path.name,
                "source_node_count": len(nodes),
                "source_segment_count": len(edges),
                "render_segment_count": len(shown_edges),
                "lod_is_exact": bool(len(shown_edges) == len(edges)),
                "line_vertices_um": vertices.round(4).tolist(),
            }
        )
    return {
        "protocol": ASSET_PROTOCOL,
        "dataset": "male-cns:v1.0",
        "coordinate_source": "MaleCNS SWC centerlines",
        "source_coordinate_units": "8nm units",
        "render_coordinate_units": "micrometers",
        "body_ids": [int(x) for x in body_ids],
        "neuron_count": len(neurons),
        "bounds_um": {
            "min": bounds_min.round(4).tolist() if np.all(np.isfinite(bounds_min)) else None,
            "max": bounds_max.round(4).tolist() if np.all(np.isfinite(bounds_max)) else None,
        },
        "neurons": neurons,
        "claim_boundary": (
            "Measured centerline geometry for visualization. Decimated geometry is display LOD only. "
            "Colors, emissive intensity, and animation are separate modeled/display state and are not measured firing."
        ),
    }


def _load_body_ids(path: str | Path | None, csv_ids: str | None) -> list[int]:
    values: list[int] = []
    if path is not None:
        payload = json.loads(Path(path).read_text())
        if isinstance(payload, list):
            values.extend(int(x) for x in payload)
        elif isinstance(payload, dict) and isinstance(payload.get("body_ids"), list):
            values.extend(int(x) for x in payload["body_ids"])
        else:
            raise ValueError("body-id file must be a JSON list or contain body_ids")
    if csv_ids:
        values.extend(int(x.strip()) for x in csv_ids.split(",") if x.strip())
    result = list(dict.fromkeys(values))
    if not result:
        raise ValueError("no body IDs provided")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack exact MaleCNS SWCs for the connectome-twin renderer")
    parser.add_argument("swc_directory")
    parser.add_argument("--body-ids-file")
    parser.add_argument("--body-ids")
    parser.add_argument("--max-segments-per-neuron", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/connectome-twin/route-skeletons-v1.json.gz")
    args = parser.parse_args()

    report = pack_swc_directory(
        args.swc_directory,
        _load_body_ids(args.body_ids_file, args.body_ids),
        max_segments_per_neuron=args.max_segments_per_neuron,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, separators=(",", ":")) + "\n").encode()
    if output.suffix == ".gz":
        with gzip.open(output, "wb", compresslevel=6) as handle:
            handle.write(raw)
    else:
        output.write_bytes(raw)
    print(output)
    print(f"neurons={report['neuron_count']} body_ids={report['body_ids']}")


if __name__ == "__main__":
    main()
