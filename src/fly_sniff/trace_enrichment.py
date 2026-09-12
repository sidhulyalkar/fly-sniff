from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import hypergeom

DEFAULT_COLUMNS = ("type", "class", "subclass")


def _bh_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 1.0
    total = len(p_values)
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = total - reverse_rank + 1
        candidate = min(1.0, p_values[index] * total / rank)
        running = min(running, candidate)
        adjusted[index] = running
    return adjusted


def enrichment_for_column(
    annotations: pd.DataFrame,
    retained_ids: set[int],
    column: str,
    *,
    top_n: int = 50,
) -> dict[str, Any]:
    if "bodyId" not in annotations.columns:
        raise ValueError("annotations require bodyId")
    if column not in annotations.columns:
        return {
            "column": column,
            "available": False,
            "rows": [],
        }

    universe = annotations[["bodyId", column]].copy()
    universe["bodyId"] = universe.bodyId.astype(int)
    universe = universe.drop_duplicates("bodyId")
    universe[column] = universe[column].fillna("<NA>").astype(str)
    retained = universe[universe.bodyId.isin(retained_ids)].copy()

    population_size = len(universe)
    sample_size = len(retained)
    if sample_size == 0 or population_size == 0:
        return {
            "column": column,
            "available": True,
            "population_size": population_size,
            "corridor_size": sample_size,
            "rows": [],
        }

    background_counts = universe[column].value_counts()
    retained_counts = retained[column].value_counts()
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []

    for label, observed_value in retained_counts.items():
        observed = int(observed_value)
        background = int(background_counts[label])
        expected = sample_size * background / population_size
        fold = (observed / sample_size) / (background / population_size)
        p_value = float(
            hypergeom.sf(
                observed - 1,
                population_size,
                background,
                sample_size,
            )
        )
        p_values.append(p_value)
        rows.append(
            {
                "label": str(label),
                "corridor_count": observed,
                "background_count": background,
                "expected_count": float(expected),
                "fold_enrichment": float(fold),
                "log2_fold_enrichment": float(math.log2(fold)),
                "p_value": p_value,
            }
        )

    q_values = _bh_adjust(p_values)
    for row, q_value in zip(rows, q_values, strict=True):
        row["fdr_bh"] = float(q_value)

    rows.sort(
        key=lambda row: (
            row["fdr_bh"],
            row["p_value"],
            -row["fold_enrichment"],
            -row["corridor_count"],
            row["label"],
        )
    )
    return {
        "column": column,
        "available": True,
        "population_size": population_size,
        "corridor_size": sample_size,
        "tested_labels": len(rows),
        "rows": rows[: max(0, int(top_n))],
        "interpretation": (
            "One-sided hypergeometric over-representation relative to all annotated body IDs; "
            "FDR is Benjamini-Hochberg within this annotation column. Enrichment is descriptive "
            "structural evidence, not functional evidence."
        ),
    }


def audit_enrichment(
    annotations: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    columns: tuple[str, ...] = DEFAULT_COLUMNS,
    top_n: int = 50,
) -> dict[str, Any]:
    if "bodyId" not in nodes.columns:
        raise ValueError("trace nodes require bodyId")
    retained_ids = set(nodes.bodyId.astype(int))
    return {
        "protocol": "structural-corridor-enrichment-v1",
        "retained_node_count": len(retained_ids),
        "columns": {
            column: enrichment_for_column(
                annotations,
                retained_ids,
                column,
                top_n=top_n,
            )
            for column in columns
        },
        "claim_boundary": (
            "Over-representation in a structural search corridor does not establish neural activity, "
            "causal influence, odor tuning, or navigation function."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure annotation enrichment in a traced MaleCNS structural corridor"
    )
    parser.add_argument("annotations", help="MaleCNS body annotation Feather file")
    parser.add_argument("trace_dir", help="completed fly-sniff-trace output directory")
    parser.add_argument(
        "--output",
        default="results/trace/staged-route-v1-enrichment.json",
    )
    parser.add_argument("--column", action="append", default=None)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    annotations = pd.read_feather(args.annotations)
    nodes = pd.read_parquet(Path(args.trace_dir) / "nodes.parquet")
    report = audit_enrichment(
        annotations,
        nodes,
        columns=tuple(args.column or DEFAULT_COLUMNS),
        top_n=args.top_n,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
