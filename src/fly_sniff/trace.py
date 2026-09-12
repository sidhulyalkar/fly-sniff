from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

EDGE_COLUMN_ALIASES = {
    "source": ("source", "body_pre", "bodyId_pre", "pre", "pre_root_id"),
    "target": ("target", "body_post", "bodyId_post", "post", "post_root_id"),
    "weight": ("weight", "syn_count", "count", "n_synapses"),
}


def _resolve_column(frame: pd.DataFrame, role: str) -> str:
    aliases = EDGE_COLUMN_ALIASES[role]
    match = next((column for column in aliases if column in frame.columns), None)
    if match is None:
        raise ValueError(
            f"cannot resolve {role} edge column; expected one of {list(aliases)}, "
            f"found {list(frame.columns)}"
        )
    return match


def resolve_edge_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    """Resolve supported edge-table schemas to source, target, and weight columns.

    The public Janelia MaleCNS flat-connectome tables use ``body_pre`` and
    ``body_post``. Other supported inputs include normalized ``source`` / ``target``
    tables and common neuPrint-style aliases. Downstream tracing always renames the
    resolved columns to the canonical ``source`` / ``target`` / ``weight`` schema.
    """
    return (
        _resolve_column(frame, "source"),
        _resolve_column(frame, "target"),
        _resolve_column(frame, "weight"),
    )


def body_ids_matching(annotations: pd.DataFrame, patterns: list[str]) -> set[int]:
    if "bodyId" not in annotations.columns:
        raise ValueError("annotations require bodyId")
    cols = [c for c in ["type", "instance", "class", "subclass"] if c in annotations.columns]
    if not cols:
        raise ValueError("annotations lack searchable type/instance/class/subclass columns")
    text = annotations[cols].fillna("").astype(str).agg(" | ".join, axis=1)
    union = "(?:" + ")|(?:".join(patterns) + ")"
    mask = text.str.contains(re.compile(union, re.IGNORECASE), regex=True)
    return set(annotations.loc[mask, "bodyId"].astype(int))


def _frontier_depths(
    edges: pd.DataFrame,
    seeds: set[int],
    *,
    source_col: str,
    target_col: str,
    weight_col: str,
    reverse: bool,
    max_hops: int,
    min_weight: float,
    fanout_per_node: int,
) -> dict[int, int]:
    depths = {int(x): 0 for x in seeds}
    frontier = set(depths)
    src_col, dst_col = (target_col, source_col) if reverse else (source_col, target_col)
    eligible = edges.loc[edges[weight_col] >= min_weight, [src_col, dst_col, weight_col]].copy()
    for depth in range(1, max_hops + 1):
        if not frontier:
            break
        step = eligible[eligible[src_col].isin(frontier)]
        if step.empty:
            break
        step = step.sort_values([src_col, weight_col], ascending=[True, False])
        step = step.groupby(src_col, sort=False).head(fanout_per_node)
        discovered = set(step[dst_col].astype(int)) - set(depths)
        for body_id in discovered:
            depths[int(body_id)] = depth
        frontier = discovered
    return depths


def trace_corridor(
    annotations: pd.DataFrame,
    weights: pd.DataFrame,
    source_ids: set[int],
    target_ids: set[int],
    *,
    max_hops: int = 6,
    min_weight: float = 5,
    fanout_per_node: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract a bounded bidirectional structural corridor between seed populations.

    Forward and reverse searches retain only the strongest local fanout at each
    expansion step. A node survives if its forward distance plus reverse distance
    can participate in a path no longer than max_hops. Saved edges obey the same
    ``min_weight`` threshold used during discovery. This is a discovery tool, not
    proof of functional influence.
    """
    source_col, target_col, weight_col = resolve_edge_columns(weights)
    normalized = weights[[source_col, target_col, weight_col]].rename(
        columns={source_col: "source", target_col: "target", weight_col: "weight"}
    )
    normalized["source"] = normalized.source.astype(int)
    normalized["target"] = normalized.target.astype(int)
    normalized["weight"] = normalized.weight.astype(float)

    forward = _frontier_depths(
        normalized,
        source_ids,
        source_col="source",
        target_col="target",
        weight_col="weight",
        reverse=False,
        max_hops=max_hops,
        min_weight=min_weight,
        fanout_per_node=fanout_per_node,
    )
    reverse = _frontier_depths(
        normalized,
        target_ids,
        source_col="source",
        target_col="target",
        weight_col="weight",
        reverse=True,
        max_hops=max_hops,
        min_weight=min_weight,
        fanout_per_node=fanout_per_node,
    )
    corridor = {
        node
        for node in set(forward) & set(reverse)
        if forward[node] + reverse[node] <= max_hops
    }
    corridor |= source_ids & set(reverse)
    corridor |= target_ids & set(forward)

    # The persisted edge table must obey the same edge-strength contract that
    # created the frontier. Otherwise a trace reported as min_weight=N could
    # silently contain weaker edges that never participated in discovery.
    edges = normalized.loc[
        (normalized.weight >= min_weight)
        & normalized.source.isin(corridor)
        & normalized.target.isin(corridor)
    ].copy()

    # Vectorized bounded-path filter. This is equivalent to the former row-wise
    # apply but substantially faster on the full MaleCNS corridor.
    source_forward = edges.source.map(forward).fillna(max_hops + 1).astype(int)
    target_reverse = edges.target.map(reverse).fillna(max_hops + 1).astype(int)
    edges = edges.loc[(source_forward + 1 + target_reverse) <= max_hops].copy()

    nodes = annotations[annotations.bodyId.astype(int).isin(corridor)].copy()
    provenance = pd.DataFrame(
        {
            "bodyId": sorted(corridor),
            "forward_depth": [forward.get(x) for x in sorted(corridor)],
            "reverse_depth": [reverse.get(x) for x in sorted(corridor)],
            "is_source_seed": [x in source_ids for x in sorted(corridor)],
            "is_target_seed": [x in target_ids for x in sorted(corridor)],
        }
    )
    return nodes, edges, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace a bounded MaleCNS structural corridor offline")
    parser.add_argument("annotations", help="MaleCNS body annotation Feather file")
    parser.add_argument("weights", help="MaleCNS connection-weight Feather file")
    parser.add_argument("--source", action="append", required=True, help="source annotation regex; repeatable")
    parser.add_argument("--target", action="append", required=True, help="target annotation regex; repeatable")
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--min-weight", type=float, default=5)
    parser.add_argument("--fanout", type=int, default=30)
    parser.add_argument("--output", default="data/cache/corridor-v0")
    args = parser.parse_args()

    annotations = pd.read_feather(args.annotations)
    weights = pd.read_feather(args.weights)
    source_ids = body_ids_matching(annotations, args.source)
    target_ids = body_ids_matching(annotations, args.target)
    if not source_ids or not target_ids:
        raise SystemExit(
            f"empty seed set: {len(source_ids)} source IDs, {len(target_ids)} target IDs; inspect regexes"
        )
    nodes, edges, provenance = trace_corridor(
        annotations,
        weights,
        source_ids,
        target_ids,
        max_hops=args.max_hops,
        min_weight=args.min_weight,
        fanout_per_node=args.fanout,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(out / "nodes.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)
    provenance.to_csv(out / "path_provenance.csv", index=False)

    retained_source_seeds = int(provenance.is_source_seed.sum())
    retained_target_seeds = int(provenance.is_target_seed.sum())
    report = {
        "source_regex": args.source,
        "target_regex": args.target,
        # Compatibility keys: these are the regex-matched input populations.
        "source_seed_count": len(source_ids),
        "target_seed_count": len(target_ids),
        "input_source_seed_count": len(source_ids),
        "input_target_seed_count": len(target_ids),
        # These are the seed neurons that actually survive the bounded corridor.
        "retained_source_seed_count": retained_source_seeds,
        "retained_target_seed_count": retained_target_seeds,
        "corridor_nodes": len(nodes),
        "corridor_edges": len(edges),
        "max_hops": args.max_hops,
        "min_weight": args.min_weight,
        "fanout_per_node": args.fanout,
        "saved_edge_policy": "weight>=min_weight and bounded source-target path",
        "warning": "structural corridor only; not evidence of functional influence",
    }
    (out / "trace_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
