from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph import GraphBundle
from .rewire import degree_preserving_rewire

PROTOCOL = "E004a-rewire-prequalification-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _degree_map(edges: pd.DataFrame, column: str) -> dict[int, int]:
    counts = edges[column].astype(int).value_counts()
    return {int(k): int(v) for k, v in counts.items()}


def _edge_pairs(edges: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(edges.source.astype(int), edges.target.astype(int), strict=True))


def _sorted_numeric(values: pd.Series) -> list[float]:
    return sorted(float(x) for x in values.tolist())


def _same_rewire(a: GraphBundle, b: GraphBundle) -> bool:
    cols = [column for column in ("source", "target", "weight", "sign") if column in a.edges.columns]
    if cols != [column for column in ("source", "target", "weight", "sign") if column in b.edges.columns]:
        return False
    left = a.edges[cols].reset_index(drop=True)
    right = b.edges[cols].reset_index(drop=True)
    return bool(left.equals(right))


def run_preflight(bundle: GraphBundle, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected E004a protocol")
    bundle.validate(require_sign="sign" in bundle.edges.columns, require_qualified=False)
    rules = config["rewire"]
    seeds = [int(x) for x in rules["seeds"]]
    swaps_per_edge = int(rules["swaps_per_edge"])
    required_swaps = swaps_per_edge * len(bundle.edges)
    original_pairs = _edge_pairs(bundle.edges)
    original_in = _degree_map(bundle.edges, "target")
    original_out = _degree_map(bundle.edges, "source")
    original_weights = _sorted_numeric(bundle.edges.weight)
    original_signs = _sorted_numeric(bundle.edges.sign) if "sign" in bundle.edges else None

    reports: list[dict[str, Any]] = []
    for seed in seeds:
        rewired = degree_preserving_rewire(bundle, seed=seed, swaps_per_edge=swaps_per_edge)
        replay = degree_preserving_rewire(bundle, seed=seed, swaps_per_edge=swaps_per_edge)
        rewired.validate(require_sign="sign" in bundle.edges.columns, require_qualified=False)
        pairs = _edge_pairs(rewired.edges)
        changed_fraction = 1.0 - len(original_pairs & pairs) / max(len(original_pairs), 1)
        manifest = rewired.manifest or {}
        rewire_meta = manifest.get("rewire", {})
        accepted = int(rewire_meta.get("accepted_swaps", -1))
        self_loops = int((rewired.edges.source.astype(int) == rewired.edges.target.astype(int)).sum())
        duplicate_count = int(rewired.edges.duplicated(subset=["source", "target"]).sum())
        checks = {
            "in_degree_identity": _degree_map(rewired.edges, "target") == original_in,
            "out_degree_identity": _degree_map(rewired.edges, "source") == original_out,
            "edge_count_identity": len(rewired.edges) == len(bundle.edges),
            "weight_multiset_identity": _sorted_numeric(rewired.edges.weight) == original_weights,
            "sign_multiset_identity": (
                True
                if original_signs is None
                else _sorted_numeric(rewired.edges.sign) == original_signs
            ),
            "no_self_loops": self_loops == 0,
            "no_duplicate_edges": duplicate_count == 0,
            "deterministic_same_seed_rewire": _same_rewire(rewired, replay),
            "full_requested_swap_count": accepted == required_swaps,
            "minimum_changed_edge_fraction": changed_fraction
            >= float(rules["minimum_changed_edge_fraction"]),
        }
        reports.append(
            {
                "seed": seed,
                "accepted_swaps": accepted,
                "required_swaps": required_swaps,
                "changed_edge_fraction": float(changed_fraction),
                "self_loop_count": self_loops,
                "duplicate_edge_count": duplicate_count,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    gates = [
        {
            "name": "all_frozen_rewire_seeds_reported",
            "passed": [row["seed"] for row in reports] == seeds,
        },
        {
            "name": "minimum_independent_rewire_count",
            "passed": len(reports) >= int(rules["minimum_independent_rewires_for_final_claim"]),
        },
        {
            "name": "all_rewires_pass_preflight",
            "passed": all(bool(row["passed"]) for row in reports),
        },
        {
            "name": "equal_compute_contract_frozen",
            "passed": all(bool(value) for value in config["equal_compute_contract"].values()),
        },
        {
            "name": "lesion_set_frozen",
            "passed": len(config.get("lesions", [])) >= 4,
        },
    ]
    return {
        "protocol": PROTOCOL,
        "dataset": config["dataset"],
        "passed": all(bool(row["passed"]) for row in gates),
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(row["passed"]) for row in gates),
        "gates": gates,
        "rewire_reports": reports,
        "equal_compute_contract": config["equal_compute_contract"],
        "lesions": config["lesions"],
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E004a degree-preserving rewire prequalification")
    parser.add_argument("bundle")
    parser.add_argument("--config", default="configs/e004a_rewire_prequalification_v1.json")
    parser.add_argument("--output", default="results/e004/rewire-prequalification-v1.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    bundle = GraphBundle.load(args.bundle)
    config = json.loads(config_path.read_text())
    report = run_preflight(bundle, config)
    report["input_sha256"] = {
        "config": _sha256(config_path),
        "bundle_manifest": _sha256(Path(args.bundle) / "manifest.json"),
        "bundle_edges": _sha256(Path(args.bundle) / "edges.parquet"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    print(f"passed={report['passed']} gates={report['passed_gate_count']}/{report['gate_count']}")
    for row in report["rewire_reports"]:
        print(
            f"seed={row['seed']} passed={row['passed']} "
            f"changed={row['changed_edge_fraction']:.3f} swaps={row['accepted_swaps']}/{row['required_swaps']}"
        )


if __name__ == "__main__":
    main()
