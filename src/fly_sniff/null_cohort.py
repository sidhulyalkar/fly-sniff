from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .graph import GraphBundle
from .null_generation import audit_degree_null, exact_max_distance_constrained_rewire
from .rewire import save_bundle

PROTOCOL = "confirmatory-null-cohort-v1"
FAMILY_METADATA = {
    "degree_preserving": (),
    "degree_sign_preserving": (),
    "degree_sign_hemisphere_preserving": ("hemisphere_class",),
    "within_cell_type_rewire": ("cell_type",),
    "spatially_constrained_rewire": ("cell_type", "spatial_bin"),
}


def _topology_sha256(bundle: GraphBundle) -> str:
    pairs = sorted(
        (int(source), int(target))
        for source, target in zip(bundle.edges.source, bundle.edges.target, strict=True)
    )
    raw = json.dumps(pairs, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _seed_stream(payload: dict[str, Any], stream: str) -> list[int]:
    if payload.get("protocol") != "fly-sniff-final-seed-reveal-v1":
        raise ValueError("confirmatory null cohorts require a commit-reveal seed receipt")
    streams = payload.get("streams")
    if not isinstance(streams, dict) or stream not in streams:
        raise ValueError(f"seed reveal does not contain stream {stream!r}")
    values = [int(x) for x in streams[stream]]
    if len(values) != len(set(values)):
        raise ValueError("seed stream contains duplicate values")
    return values


def build_null_cohort(
    intact: GraphBundle,
    *,
    family: str,
    seeds: list[int],
    minimum_changed_edge_fraction: float,
    minimum_topologies: int,
    output_root: str | Path,
    max_candidate_pairs: int = 2_000_000,
) -> dict[str, Any]:
    if family not in FAMILY_METADATA:
        raise ValueError(f"unsupported null family: {family}")
    if len(seeds) < minimum_topologies:
        raise ValueError(
            f"confirmatory cohort requires at least {minimum_topologies} seeds; got {len(seeds)}"
        )
    if len(seeds) != len(set(seeds)):
        raise ValueError("confirmatory null seeds must be unique")
    if minimum_topologies < 63:
        raise ValueError("Program A confirmatory topology cohorts may not use fewer than 63 nulls")

    metadata_columns = FAMILY_METADATA[family]
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    intact_sha = _topology_sha256(intact)
    topology_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []

    for index, seed in enumerate(seeds[:minimum_topologies]):
        rewired = exact_max_distance_constrained_rewire(
            intact,
            seed=int(seed),
            minimum_changed_edge_fraction=minimum_changed_edge_fraction,
            target_metadata_columns=metadata_columns,
            max_candidate_pairs=max_candidate_pairs,
        )
        audit = audit_degree_null(
            intact,
            rewired,
            target_metadata_columns=metadata_columns,
        )
        if not audit["passed"]:
            raise RuntimeError(f"null seed {seed} failed invariant audit")
        topology_sha = _topology_sha256(rewired)
        if topology_sha == intact_sha:
            raise RuntimeError(f"null seed {seed} reproduced the intact topology")
        if topology_sha in topology_hashes:
            raise RuntimeError(
                f"null seed {seed} produced a duplicate topology; confirmatory nulls must be distinct"
            )
        topology_hashes.add(topology_sha)

        bundle_dir = root / f"null-{index:03d}-seed-{int(seed)}"
        if bundle_dir.exists():
            raise FileExistsError(f"refusing to overwrite null bundle: {bundle_dir}")
        save_bundle(rewired, bundle_dir)
        audit_path = bundle_dir / "null-audit.json"
        audit_payload = {
            **audit,
            "family": family,
            "seed": int(seed),
            "topology_sha256": topology_sha,
            "selection_used_behavior_performance": False,
        }
        audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n")
        rows.append(
            {
                "index": index,
                "seed": int(seed),
                "bundle": str(bundle_dir),
                "topology_sha256": topology_sha,
                "changed_edge_fraction": float(audit["changed_edge_fraction"]),
                "intact_overlap_fraction": float(audit["intact_overlap_fraction"]),
                "audit_passed": True,
            }
        )

    changed = [float(row["changed_edge_fraction"]) for row in rows]
    return {
        "protocol": PROTOCOL,
        "family": family,
        "metadata_constraints": list(metadata_columns),
        "intact_topology_sha256": intact_sha,
        "topology_count": len(rows),
        "unique_topology_count": len(topology_hashes),
        "minimum_changed_edge_fraction": float(minimum_changed_edge_fraction),
        "observed_changed_edge_fraction_min": min(changed),
        "observed_changed_edge_fraction_max": max(changed),
        "all_invariant_audits_passed": all(bool(row["audit_passed"]) for row in rows),
        "all_topologies_unique": len(topology_hashes) == len(rows),
        "selection_used_behavior_performance": False,
        "nulls": rows,
        "claim_boundary": (
            "This receipt qualifies the generated graphs as topology-null realizations under the "
            "declared invariants and distance floor. It contains no navigation performance and "
            "does not support a topology-dependence claim by itself."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an audited confirmatory topology-null cohort")
    parser.add_argument("bundle")
    parser.add_argument("--family", choices=tuple(FAMILY_METADATA), required=True)
    parser.add_argument("--seed-reveal", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--minimum-topologies", type=int, default=63)
    parser.add_argument("--minimum-changed", type=float, default=0.80)
    parser.add_argument("--max-candidate-pairs", type=int, default=2_000_000)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    seed_payload = json.loads(Path(args.seed_reveal).read_text())
    report = build_null_cohort(
        GraphBundle.load(args.bundle),
        family=args.family,
        seeds=_seed_stream(seed_payload, args.stream),
        minimum_changed_edge_fraction=args.minimum_changed,
        minimum_topologies=args.minimum_topologies,
        output_root=args.output_root,
        max_candidate_pairs=args.max_candidate_pairs,
    )
    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite cohort receipt: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)
    print(
        f"family={report['family']} topologies={report['topology_count']} "
        f"unique={report['unique_topology_count']} "
        f"changed_min={report['observed_changed_edge_fraction_min']:.6f}"
    )


if __name__ == "__main__":
    main()
