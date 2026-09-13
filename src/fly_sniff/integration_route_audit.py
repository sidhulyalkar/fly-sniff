from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .literature_route_audit import (
    ANNOTATION_ID_COLUMNS,
    _edge_detail,
    _git_output,
    _json_records,
    _normalize_edges,
    _sha256_file,
    _threshold_summary,
)

DEFAULT_CONFIG = Path("configs/integration_route_audit_v1.json")


def _population_rows(
    annotations: pd.DataFrame,
    name: str,
    selector: dict[str, Any],
) -> pd.DataFrame:
    column = str(selector.get("column", "type"))
    if column not in annotations.columns:
        raise ValueError(f"population {name!r} requires missing annotation column {column!r}")

    exact = selector.get("exact")
    regex = selector.get("regex")
    if (exact is None) == (regex is None):
        raise ValueError(f"population {name!r} must define exactly one of exact or regex")

    values = annotations[column].fillna("").astype(str).str.strip()
    if exact is not None:
        mask = values.eq(str(exact))
    else:
        mask = values.str.match(str(regex), case=True, na=False)

    columns = [c for c in ANNOTATION_ID_COLUMNS if c in annotations.columns]
    rows = annotations.loc[mask, columns].copy()
    if rows.empty:
        rows = pd.DataFrame(columns=columns)
        if "bodyId" not in rows.columns:
            rows["bodyId"] = pd.Series(dtype="int64")
        return rows

    rows["bodyId"] = rows.bodyId.astype(int)
    if rows.bodyId.duplicated().any():
        raise ValueError(f"population {name!r} contains duplicate body IDs")
    return rows.sort_values("bodyId").reset_index(drop=True)


def _population_summary(
    rows: pd.DataFrame,
    selector: dict[str, Any],
) -> dict[str, Any]:
    type_counts: list[dict[str, Any]] = []
    if "type" in rows.columns and len(rows):
        counts = rows["type"].fillna("").astype(str).value_counts().sort_index()
        type_counts = [
            {"type": str(cell_type), "count": int(count)}
            for cell_type, count in counts.items()
        ]
    return {
        "selector": selector,
        "count": len(rows),
        "type_counts": type_counts,
        "rows": _json_records(rows, limit=1000),
    }


def _direct_prediction(
    edges: pd.DataFrame,
    populations: dict[str, pd.DataFrame],
    prediction: dict[str, Any],
    thresholds: tuple[float, ...],
    *,
    max_edges: int,
) -> dict[str, Any]:
    source_name = str(prediction["source"])
    target_name = str(prediction["target"])
    detail = _edge_detail(edges, populations[source_name], populations[target_name])
    return {
        **prediction,
        "source_population_count": len(populations[source_name]),
        "target_population_count": len(populations[target_name]),
        "observed_edge_pairs": len(detail),
        "observed_weight_sum": float(detail.weight.sum()) if len(detail) else 0.0,
        "threshold_sweep": _threshold_summary(detail, thresholds),
        "edges": _json_records(detail, limit=max_edges),
        "claim_boundary": (
            "Structural support only. Observed edges do not establish effective connectivity, "
            "physiology, stimulus tuning, or behavioral contribution."
        ),
    }


def _three_hop_prediction(
    edges: pd.DataFrame,
    populations: dict[str, pd.DataFrame],
    prediction: dict[str, Any],
    thresholds: tuple[float, ...],
    *,
    max_paths: int,
) -> dict[str, Any]:
    source_name = str(prediction["source"])
    via1_name = str(prediction["via1"])
    via2_name = str(prediction["via2"])
    target_name = str(prediction["target"])

    first = _edge_detail(edges, populations[source_name], populations[via1_name])
    second = _edge_detail(edges, populations[via1_name], populations[via2_name])
    third = _edge_detail(edges, populations[via2_name], populations[target_name])

    if first.empty or second.empty or third.empty:
        paths = pd.DataFrame(
            columns=[
                "source",
                "via1",
                "via2",
                "target",
                "weight_1",
                "weight_2",
                "weight_3",
                "bottleneck_weight",
            ]
        )
    else:
        first_small = first[["source", "target", "weight"]].rename(
            columns={"target": "via1", "weight": "weight_1"}
        )
        second_small = second[["source", "target", "weight"]].rename(
            columns={"source": "via1", "target": "via2", "weight": "weight_2"}
        )
        third_small = third[["source", "target", "weight"]].rename(
            columns={"source": "via2", "weight": "weight_3"}
        )
        paths = first_small.merge(second_small, on="via1", how="inner")
        paths = paths.merge(third_small, on="via2", how="inner")
        paths["bottleneck_weight"] = paths[["weight_1", "weight_2", "weight_3"]].min(
            axis=1
        )
        paths = paths.sort_values(
            ["bottleneck_weight", "weight_1", "weight_2", "weight_3"],
            ascending=False,
        )

    threshold_sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        retained = paths[
            (paths.weight_1 >= threshold)
            & (paths.weight_2 >= threshold)
            & (paths.weight_3 >= threshold)
        ]
        threshold_sweep.append(
            {
                "min_weight_each_edge": float(threshold),
                "path_count": len(retained),
                "source_body_ids": sorted(set(retained.source.astype(int)))
                if len(retained)
                else [],
                "via1_body_ids": sorted(set(retained.via1.astype(int)))
                if len(retained)
                else [],
                "via2_body_ids": sorted(set(retained.via2.astype(int)))
                if len(retained)
                else [],
                "target_body_ids": sorted(set(retained.target.astype(int)))
                if len(retained)
                else [],
            }
        )

    return {
        **prediction,
        "source_population_count": len(populations[source_name]),
        "via1_population_count": len(populations[via1_name]),
        "via2_population_count": len(populations[via2_name]),
        "target_population_count": len(populations[target_name]),
        "observed_path_count": len(paths),
        "threshold_sweep": threshold_sweep,
        "paths": _json_records(paths, limit=max_paths),
        "claim_boundary": (
            "Three-edge structural support is descriptive. It does not establish signal "
            "transmission, sign, timing, tuning, necessity, or sufficiency."
        ),
    }


def build_integration_route_audit(
    annotations: pd.DataFrame,
    weights: pd.DataFrame,
    config: dict[str, Any],
    *,
    max_edges: int = 500,
    max_paths: int = 1000,
) -> dict[str, Any]:
    if "bodyId" not in annotations.columns:
        raise ValueError("annotations require bodyId")
    annotations = annotations.copy()
    annotations["bodyId"] = annotations.bodyId.astype(int)
    edges = _normalize_edges(weights)

    selectors = config.get("population_selectors", {})
    if not selectors:
        raise ValueError("integration-route config requires population_selectors")
    populations = {
        str(name): _population_rows(annotations, str(name), dict(selector))
        for name, selector in selectors.items()
    }
    thresholds = tuple(float(value) for value in config.get("thresholds", [1, 3, 5, 10]))

    direct = [
        _direct_prediction(
            edges,
            populations,
            prediction,
            thresholds,
            max_edges=max_edges,
        )
        for prediction in config.get("direct_predictions", [])
    ]
    three_hop = [
        _three_hop_prediction(
            edges,
            populations,
            prediction,
            thresholds,
            max_paths=max_paths,
        )
        for prediction in config.get("three_hop_predictions", [])
    ]

    return {
        "protocol": str(config.get("protocol", "malecns-integration-route-audit-v1")),
        "dataset": str(config.get("dataset", "male-cns:v1.0")),
        "runtime": {
            "git_branch": _git_output("branch", "--show-current"),
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_dirty_paths": (_git_output("status", "--porcelain") or "").splitlines(),
        },
        "populations": {
            name: _population_summary(populations[name], dict(selectors[name]))
            for name in populations
        },
        "direct_predictions": direct,
        "three_hop_predictions": three_hop,
        "thresholds": list(thresholds),
        "claim_boundary": str(config.get("claim_boundary", "Structural audit only.")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen MaleCNS odor/wind -> hDeltaC -> hDeltaG -> PFL route"
    )
    parser.add_argument("annotations")
    parser.add_argument("weights")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output",
        default="results/route/integration-route-audit-v1.json",
    )
    parser.add_argument("--max-edges", type=int, default=500)
    parser.add_argument("--max-paths", type=int, default=1000)
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    weights_path = Path(args.weights)
    config_path = Path(args.config)
    annotations = pd.read_feather(annotations_path)
    weights = pd.read_feather(weights_path)
    config = json.loads(config_path.read_text())
    report = build_integration_route_audit(
        annotations,
        weights,
        config,
        max_edges=args.max_edges,
        max_paths=args.max_paths,
    )
    report["inputs"] = {
        "annotations": {
            "path": str(annotations_path),
            "sha256": _sha256_file(annotations_path),
        },
        "weights": {"path": str(weights_path), "sha256": _sha256_file(weights_path)},
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    concise = {
        "protocol": report["protocol"],
        "git_sha": report["runtime"]["git_sha"],
        "population_counts": {
            name: row["count"] for name, row in report["populations"].items()
        },
        "direct_support": {
            f"{row['source']}->{row['target']}": row["observed_weight_sum"]
            for row in report["direct_predictions"]
        },
        "three_hop_paths": {
            f"{row['source']}->{row['via1']}->{row['via2']}->{row['target']}": (
                row["observed_path_count"]
            )
            for row in report["three_hop_predictions"]
        },
        "output": str(output),
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
