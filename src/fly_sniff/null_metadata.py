from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .graph import GraphBundle
from .rewire import save_bundle

PROTOCOL = "malecns-null-metadata-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_side(value: Any) -> str:
    text = str(value).strip().upper()
    aliases = {
        "L": "L",
        "LEFT": "L",
        "R": "R",
        "RIGHT": "R",
    }
    if text not in aliases:
        raise ValueError(f"unresolved somaSide for confirmatory hemisphere null: {value!r}")
    return aliases[text]


def attach_null_metadata(bundle: GraphBundle) -> tuple[GraphBundle, dict[str, Any]]:
    """Copy exact MaleCNS annotation fields into frozen null-model metadata.

    No value is inferred from connectivity or navigation performance. `cell_type` is an
    exact copy of the release `type` field. `hemisphere_class` is a normalization of the
    release `somaSide` field restricted to explicit left/right values. Spatial bins are
    intentionally not created here because their biological definition is still unresolved.
    """
    bundle.validate(require_sign="sign" in bundle.edges.columns)
    required = {"bodyId", "type", "somaSide"}
    missing = required - set(bundle.nodes.columns)
    if missing:
        raise ValueError(f"MaleCNS null metadata requires node columns: {sorted(missing)}")

    nodes = bundle.nodes.copy()
    types = nodes["type"].astype("string").str.strip()
    if types.isna().any() or (types == "").any():
        bad = nodes.loc[types.isna() | (types == ""), "bodyId"].astype(int).tolist()[:10]
        raise ValueError(f"unresolved MaleCNS type prevents confirmatory type null: {bad}")
    nodes["cell_type"] = types.astype(str)
    nodes["hemisphere_class"] = [_normalize_side(value) for value in nodes["somaSide"]]

    connected = set(bundle.edges.source.astype(int)) | set(bundle.edges.target.astype(int))
    connected_nodes = nodes.loc[nodes.bodyId.astype(int).isin(connected)]
    report = {
        "protocol": PROTOCOL,
        "dataset": (bundle.manifest or {}).get("dataset", "male-cns:v1.0"),
        "node_count": int(len(nodes)),
        "connected_node_count": int(len(connected_nodes)),
        "cell_type_count": int(nodes["cell_type"].nunique()),
        "hemisphere_counts": {
            str(key): int(value)
            for key, value in nodes["hemisphere_class"].value_counts().sort_index().items()
        },
        "mapping": {
            "cell_type": "exact copy of MaleCNS nodes.type",
            "hemisphere_class": "nodes.somaSide normalized only from explicit L/LEFT/R/RIGHT",
            "spatial_bin": "unresolved-not-created",
        },
        "selection_used_navigation_performance": False,
        "claim_boundary": (
            "These fields are frozen metadata constraints for topology null generation only. "
            "They do not imply functional equivalence within a cell type or hemisphere."
        ),
    }
    manifest = dict(bundle.manifest or {})
    manifest["null_metadata_protocol"] = PROTOCOL
    return GraphBundle(nodes, bundle.edges.copy(), dict(bundle.roles), manifest), report


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach frozen MaleCNS metadata for constrained nulls")
    parser.add_argument("bundle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = GraphBundle.load(args.bundle)
    mapped, report = attach_null_metadata(source)
    save_bundle(mapped, args.output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(report_path)
    print(file_sha256(report_path))


if __name__ == "__main__":
    main()
