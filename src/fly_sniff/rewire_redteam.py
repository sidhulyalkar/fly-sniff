from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .graph import GraphBundle
from .rewire import degree_preserving_rewire
from .training import (
    FROZEN_V1_REWIRE_COUNT,
    FROZEN_V1_SWAPS_PER_EDGE,
    canonical_sha256,
)

PROTOCOL = "rewire-ensemble-redteam-v1"


def _edge_pairs(bundle: GraphBundle) -> list[tuple[int, int]]:
    return [
        (int(row.source), int(row.target))
        for row in bundle.edges[["source", "target"]].itertuples(index=False)
    ]


def _edge_set(bundle: GraphBundle) -> set[tuple[int, int]]:
    return set(_edge_pairs(bundle))


def _degree_signature(bundle: GraphBundle) -> tuple[Counter[int], Counter[int]]:
    incoming: Counter[int] = Counter()
    outgoing: Counter[int] = Counter()
    for source, target in _edge_pairs(bundle):
        outgoing[source] += 1
        incoming[target] += 1
    for body_id in bundle.nodes.bodyId.astype(int):
        incoming[int(body_id)] += 0
        outgoing[int(body_id)] += 0
    return incoming, outgoing


def _presynaptic_attribute_multiset(bundle: GraphBundle) -> Counter[tuple[Any, ...]]:
    columns = ["source", "weight"]
    if "sign" in bundle.edges.columns:
        columns.append("sign")
    values: Counter[tuple[Any, ...]] = Counter()
    for row in bundle.edges[columns].itertuples(index=False, name=None):
        normalized = [int(row[0]), float(row[1])]
        if len(row) == 3:
            normalized.append(int(row[2]))
        values[tuple(normalized)] += 1
    return values


def _overlap_fraction(a: GraphBundle, b: GraphBundle) -> float:
    a_edges = _edge_set(a)
    b_edges = _edge_set(b)
    if not a_edges:
        return 1.0 if not b_edges else 0.0
    return float(len(a_edges & b_edges) / len(a_edges))


def _jaccard(a: GraphBundle, b: GraphBundle) -> float:
    a_edges = _edge_set(a)
    b_edges = _edge_set(b)
    union = a_edges | b_edges
    return float(len(a_edges & b_edges) / len(union)) if union else 1.0


def _validate_rewire_against_intact(intact: GraphBundle, rewired: GraphBundle) -> None:
    intact.validate(require_sign="sign" in intact.edges.columns)
    rewired.validate(require_sign="sign" in rewired.edges.columns)
    if sorted(intact.nodes.bodyId.astype(int)) != sorted(rewired.nodes.bodyId.astype(int)):
        raise RuntimeError("rewire changed the body-ID set")
    if intact.roles != rewired.roles:
        raise RuntimeError("rewire changed role membership")
    if len(intact.edges) != len(rewired.edges):
        raise RuntimeError("rewire changed edge count")
    if _degree_signature(intact) != _degree_signature(rewired):
        raise RuntimeError("rewire changed exact directed in/out degree")
    if _presynaptic_attribute_multiset(intact) != _presynaptic_attribute_multiset(rewired):
        raise RuntimeError("rewire changed the presynaptic structural weight/sign multiset")
    pairs = _edge_pairs(rewired)
    if len(set(pairs)) != len(pairs):
        raise RuntimeError("rewire contains duplicate directed edges")
    if any(source == target for source, target in pairs):
        raise RuntimeError("rewire contains self-loops")


def summarize_rewire_ensemble(
    intact: GraphBundle,
    rewires: list[tuple[int, GraphBundle]],
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Audit exact identity and report topology similarity without a mixing threshold."""
    if not rewires:
        raise ValueError("rewire audit requires at least one topology")
    if expected_count is not None and len(rewires) != int(expected_count):
        raise ValueError(
            f"rewire audit expected {expected_count} topologies, received {len(rewires)}"
        )
    seeds = [int(seed) for seed, _ in rewires]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("rewire ensemble contains duplicate seeds")

    intact_fingerprint = intact.replay_fingerprint()
    rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    for seed, rewired in rewires:
        _validate_rewire_against_intact(intact, rewired)
        fingerprint = rewired.replay_fingerprint()
        if fingerprint == intact_fingerprint:
            raise RuntimeError(
                f"rewire seed {seed} regenerated the intact topology exactly; null is not disrupted"
            )
        fingerprints.append(fingerprint)
        rows.append(
            {
                "seed": int(seed),
                "graph_sha256": fingerprint,
                "directed_edge_overlap_fraction_vs_intact": _overlap_fraction(intact, rewired),
                "directed_edge_jaccard_vs_intact": _jaccard(intact, rewired),
                "edge_count": len(rewired.edges),
                "self_loop_count": sum(
                    int(source == target) for source, target in _edge_pairs(rewired)
                ),
                "duplicate_edge_count": len(_edge_pairs(rewired)) - len(_edge_set(rewired)),
            }
        )

    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("rewire ensemble contains duplicate topology fingerprints")

    pairwise: list[dict[str, Any]] = []
    for (seed_a, graph_a), (seed_b, graph_b) in combinations(rewires, 2):
        pairwise.append(
            {
                "seed_a": int(seed_a),
                "seed_b": int(seed_b),
                "directed_edge_overlap_fraction": _overlap_fraction(graph_a, graph_b),
                "directed_edge_jaccard": _jaccard(graph_a, graph_b),
            }
        )

    intact_overlaps = np.asarray(
        [row["directed_edge_overlap_fraction_vs_intact"] for row in rows],
        dtype=float,
    )
    pairwise_overlaps = np.asarray(
        [row["directed_edge_overlap_fraction"] for row in pairwise],
        dtype=float,
    )
    return {
        "protocol": PROTOCOL,
        "status": "identity_checks_passed_similarity_diagnostic_only",
        "intact_graph_sha256": intact_fingerprint,
        "rewire_count": len(rewires),
        "all_rewire_fingerprints_unique": True,
        "all_rewires_distinct_from_intact": True,
        "rewires": rows,
        "pairwise": pairwise,
        "summary": {
            "mean_edge_overlap_vs_intact": float(intact_overlaps.mean()),
            "max_edge_overlap_vs_intact": float(intact_overlaps.max()),
            "min_edge_overlap_vs_intact": float(intact_overlaps.min()),
            "mean_pairwise_edge_overlap": (
                float(pairwise_overlaps.mean()) if len(pairwise_overlaps) else None
            ),
            "max_pairwise_edge_overlap": (
                float(pairwise_overlaps.max()) if len(pairwise_overlaps) else None
            ),
        },
        "claim_boundary": (
            "Exact identity/degree/attribute checks pass and edge-overlap diagnostics quantify "
            "disruption. These measurements do not prove that the double-edge-swap Markov chain "
            "reached stationarity or that this null preserves every graph statistic relevant to "
            "navigation. No post-hoc overlap threshold is applied."
        ),
    }


def regenerate_and_audit_v1(
    intact: GraphBundle,
    matched_report: dict[str, Any],
) -> dict[str, Any]:
    if matched_report.get("protocol") != "matched-task-optimization-controls-v1":
        raise ValueError("not a matched task-optimization v1 report")
    if int(matched_report.get("rewire_count", -1)) != FROZEN_V1_REWIRE_COUNT:
        raise ValueError("matched report does not contain the frozen eight-rewire ensemble")
    if int(matched_report.get("swaps_per_edge", -1)) != FROZEN_V1_SWAPS_PER_EDGE:
        raise ValueError("matched report does not contain the frozen 8-swaps/edge budget")
    seeds = [int(seed) for seed in matched_report.get("rewire_seeds", [])]
    if len(seeds) != FROZEN_V1_REWIRE_COUNT or len(set(seeds)) != len(seeds):
        raise ValueError("matched report rewire seed receipt is missing, incomplete, or duplicated")
    if matched_report.get("intact_graph_sha256") != intact.replay_fingerprint():
        raise ValueError("matched report intact graph fingerprint does not match supplied bundle")

    rewires: list[tuple[int, GraphBundle]] = []
    report_rewires = matched_report.get("results", {}).get("rewires", {})
    for seed in seeds:
        rewired = degree_preserving_rewire(
            intact,
            seed=seed,
            swaps_per_edge=FROZEN_V1_SWAPS_PER_EDGE,
        )
        stored = report_rewires.get(str(seed))
        if not isinstance(stored, dict):
            raise TypeError(f"matched report is missing rewire result for seed {seed}")
        if stored.get("graph_sha256") != rewired.replay_fingerprint():
            raise ValueError(f"rewire seed {seed} does not reproduce its stored graph fingerprint")
        rewires.append((seed, rewired))

    audit = summarize_rewire_ensemble(
        intact,
        rewires,
        expected_count=FROZEN_V1_REWIRE_COUNT,
    )
    audit["matched_report_sha256"] = canonical_sha256(matched_report)
    audit["swaps_per_edge"] = FROZEN_V1_SWAPS_PER_EDGE
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate and red-team the frozen v1 degree-preserving rewire ensemble"
    )
    parser.add_argument("bundle", help="intact reviewed signed GraphBundle directory")
    parser.add_argument("matched_report", help="matched task-optimization report JSON")
    parser.add_argument(
        "--output",
        default="results/training-redteam/rewire-ensemble-diagnostics-v1.json",
    )
    args = parser.parse_args()

    intact = GraphBundle.load(args.bundle)
    matched_report = json.loads(Path(args.matched_report).read_text())
    audit = regenerate_and_audit_v1(intact, matched_report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)
    print(canonical_sha256(audit))


if __name__ == "__main__":
    main()
