from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .graph import GraphBundle


def degree_preserving_rewire(
    bundle: GraphBundle,
    seed: int,
    swaps_per_edge: int = 8,
) -> GraphBundle:
    """Directed double-edge swaps preserving exact in/out degree.

    Edge attributes remain attached to their presynaptic edge record while targets
    are swapped. Roles remain on the same neurons. Self-loops and duplicate edges
    are rejected. This is a topology control, not a perfect null for every graph statistic.
    """
    bundle.validate(require_sign="sign" in bundle.edges.columns)
    rng = np.random.default_rng(seed)
    edges = bundle.edges.copy().reset_index(drop=True)
    occupied = set(zip(edges.source.astype(int), edges.target.astype(int), strict=True))
    n = len(edges)
    target_swaps = swaps_per_edge * n
    accepted = 0
    attempts = 0
    max_attempts = max(target_swaps * 30, 1000)
    while accepted < target_swaps and attempts < max_attempts and n >= 2:
        attempts += 1
        i, j = rng.choice(n, size=2, replace=False)
        a, b = int(edges.at[i, "source"]), int(edges.at[i, "target"])
        c, d = int(edges.at[j, "source"]), int(edges.at[j, "target"])
        if len({a, b, c, d}) < 4:
            continue
        p1, p2 = (a, d), (c, b)
        if p1 in occupied or p2 in occupied or a == d or c == b:
            continue
        occupied.remove((a, b))
        occupied.remove((c, d))
        edges.at[i, "target"] = d
        edges.at[j, "target"] = b
        occupied.add(p1)
        occupied.add(p2)
        accepted += 1

    manifest = copy.deepcopy(bundle.manifest) if bundle.manifest else None
    if manifest is not None:
        manifest["graph_role"] = "degree-preserving-rewire"
        manifest["rewire"] = {
            "seed": int(seed),
            "swaps_per_edge": int(swaps_per_edge),
            "accepted_swaps": int(accepted),
            "attempted_swaps": int(attempts),
            "exact_in_out_degree_preserved": True,
        }
    return GraphBundle(
        bundle.nodes.copy(),
        edges,
        {k: list(v) for k, v in bundle.roles.items()},
        manifest,
    )


def save_bundle(bundle: GraphBundle, directory: str | Path) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    bundle.nodes.to_parquet(root / "nodes.parquet", index=False)
    bundle.edges.to_parquet(root / "edges.parquet", index=False)
    (root / "roles.json").write_text(json.dumps(bundle.roles, indent=2, sort_keys=True) + "\n")
    if bundle.manifest is not None:
        (root / "manifest.json").write_text(
            json.dumps(bundle.manifest, indent=2, sort_keys=True) + "\n"
        )
