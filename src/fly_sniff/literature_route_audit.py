from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .trace import body_ids_matching, resolve_edge_columns

DEFAULT_CONFIG = Path("configs/literature_route_audit_v1.json")
ANNOTATION_ID_COLUMNS = (
    "bodyId",
    "type",
    "instance",
    "class",
    "subclass",
    "superclass",
    "hemibrainType",
    "flywireType",
    "somaSide",
    "rootSide",
    "status",
    "statusLabel",
    "receptorType",
)


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if pd.isna(value):
        return None
    return str(value)


def _json_records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        frame = frame.head(max(0, limit))
    return [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _population_mask(annotations: pd.DataFrame, name: str) -> pd.Series:
    """Match a named cell type exactly across canonical type-name columns.

    This deliberately does not search free text or receptor fields. Literature
    anchors are type identities, not regex fragments.
    """
    columns = [
        column
        for column in ("type", "hemibrainType", "flywireType")
        if column in annotations.columns
    ]
    if not columns:
        raise ValueError("annotations lack type/hemibrainType/flywireType columns")
    target = name.strip().lower()
    mask = pd.Series(False, index=annotations.index)
    for column in columns:
        values = annotations[column].fillna("").astype(str).str.strip().str.lower()
        mask |= values.eq(target)
    return mask


def _population_rows(
    annotations: pd.DataFrame,
    name: str,
    *,
    max_rows: int = 100,
) -> pd.DataFrame:
    mask = _population_mask(annotations, name)
    columns = [column for column in ANNOTATION_ID_COLUMNS if column in annotations.columns]
    rows = annotations.loc[mask, columns].copy()
    rows["bodyId"] = rows.bodyId.astype(int)
    return rows.sort_values("bodyId").head(max_rows)


def _source_population_audit(annotations: pd.DataFrame) -> dict[str, Any]:
    if "bodyId" not in annotations.columns:
        raise ValueError("annotations require bodyId")

    body_ids = annotations.bodyId.astype(int)
    olfactory_mask = (
        annotations["class"].fillna("").astype(str).str.strip().str.lower().eq("olfactory")
        if "class" in annotations.columns
        else pd.Series(False, index=annotations.index)
    )
    orn_mask = (
        annotations["type"].fillna("").astype(str).str.strip().str.match(
            r"^ORN(?:_|$)", case=False
        )
        if "type" in annotations.columns
        else pd.Series(False, index=annotations.index)
    )

    olfactory_ids = set(body_ids[olfactory_mask])
    orn_ids = set(body_ids[orn_mask])
    legacy_patterns = ("ORN", "^Or", "^Ir")
    legacy_ids = {
        pattern: body_ids_matching(annotations, [pattern]) for pattern in legacy_patterns
    }

    columns = [column for column in ANNOTATION_ID_COLUMNS if column in annotations.columns]
    olfactory_not_orn = annotations.loc[
        body_ids.isin(olfactory_ids - orn_ids), columns
    ].copy()
    orn_not_olfactory = annotations.loc[
        body_ids.isin(orn_ids - olfactory_ids), columns
    ].copy()

    receptor_summary: list[dict[str, Any]] = []
    receptor_nonnull = 0
    if "receptorType" in annotations.columns:
        receptor_values = annotations.loc[olfactory_mask, "receptorType"]
        receptor_nonnull = int(receptor_values.notna().sum())
        counts = receptor_values.dropna().astype(str).value_counts().head(50)
        receptor_summary = [
            {"receptorType": str(value), "count": int(count)}
            for value, count in counts.items()
        ]

    return {
        "olfactory_class_count": len(olfactory_ids),
        "orn_type_prefix_count": len(orn_ids),
        "olfactory_not_orn_count": len(olfactory_ids - orn_ids),
        "orn_not_olfactory_count": len(orn_ids - olfactory_ids),
        "olfactory_not_orn_rows": _json_records(
            olfactory_not_orn.sort_values("bodyId"), limit=50
        ),
        "orn_not_olfactory_rows": _json_records(
            orn_not_olfactory.sort_values("bodyId"), limit=50
        ),
        "receptor_type_nonnull_in_olfactory": receptor_nonnull,
        "top_receptor_types_in_olfactory": receptor_summary,
        "legacy_regex_counts": {
            pattern: len(ids) for pattern, ids in legacy_ids.items()
        },
        "legacy_regex_overlaps": {
            "ORN_and_^Or": len(legacy_ids["ORN"] & legacy_ids["^Or"]),
            "ORN_and_^Ir": len(legacy_ids["ORN"] & legacy_ids["^Ir"]),
            "^Or_and_^Ir": len(legacy_ids["^Or"] & legacy_ids["^Ir"]),
        },
        "interpretation": (
            "Source-set audit only. Type/class membership and receptor annotations do not "
            "establish odor tuning, attraction, or functional contribution."
        ),
    }


def _normalize_edges(weights: pd.DataFrame) -> pd.DataFrame:
    source_col, target_col, weight_col = resolve_edge_columns(weights)
    normalized = weights[[source_col, target_col, weight_col]].rename(
        columns={source_col: "source", target_col: "target", weight_col: "weight"}
    )
    normalized["source"] = normalized.source.astype(int)
    normalized["target"] = normalized.target.astype(int)
    normalized["weight"] = pd.to_numeric(normalized.weight, errors="raise").astype(float)
    if not (normalized.weight > 0).all():
        raise ValueError("full structural weight table contains non-positive weights")
    return normalized


def _edge_detail(
    edges: pd.DataFrame,
    source_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    source_ids = set(source_rows.bodyId.astype(int))
    target_ids = set(target_rows.bodyId.astype(int))
    subset = edges[
        edges.source.isin(source_ids) & edges.target.isin(target_ids)
    ].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "weight",
                "source_type",
                "source_instance",
                "source_somaSide",
                "target_type",
                "target_instance",
                "target_somaSide",
            ]
        )

    pairwise = (
        subset.groupby(["source", "target"], as_index=False)
        .agg(weight=("weight", "sum"))
        .sort_values(["weight", "source", "target"], ascending=[False, True, True])
    )

    source_meta = source_rows.set_index("bodyId")
    target_meta = target_rows.set_index("bodyId")
    pairwise["source_type"] = pairwise.source.map(source_meta.get("type", pd.Series(dtype=object)))
    pairwise["source_instance"] = pairwise.source.map(
        source_meta.get("instance", pd.Series(dtype=object))
    )
    pairwise["source_somaSide"] = pairwise.source.map(
        source_meta.get("somaSide", pd.Series(dtype=object))
    )
    pairwise["target_type"] = pairwise.target.map(target_meta.get("type", pd.Series(dtype=object)))
    pairwise["target_instance"] = pairwise.target.map(
        target_meta.get("instance", pd.Series(dtype=object))
    )
    pairwise["target_somaSide"] = pairwise.target.map(
        target_meta.get("somaSide", pd.Series(dtype=object))
    )
    return pairwise


def _threshold_summary(detail: pd.DataFrame, thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        retained = detail[detail.weight >= threshold]
        rows.append(
            {
                "min_weight": float(threshold),
                "edge_pairs": len(retained),
                "weight_sum": float(retained.weight.sum()) if len(retained) else 0.0,
                "source_body_ids": sorted(set(retained.source.astype(int))) if len(retained) else [],
                "target_body_ids": sorted(set(retained.target.astype(int))) if len(retained) else [],
            }
        )
    return rows


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
            "Observed structural edges test the preregistered route prediction only. "
            "They do not establish effective connectivity or physiological influence."
        ),
    }


def _two_hop_prediction(
    edges: pd.DataFrame,
    populations: dict[str, pd.DataFrame],
    prediction: dict[str, Any],
    thresholds: tuple[float, ...],
    *,
    max_paths: int,
) -> dict[str, Any]:
    source_name = str(prediction["source"])
    via_name = str(prediction["via"])
    target_name = str(prediction["target"])
    first = _edge_detail(edges, populations[source_name], populations[via_name])
    second = _edge_detail(edges, populations[via_name], populations[target_name])

    if first.empty or second.empty:
        paths = pd.DataFrame(
            columns=[
                "source",
                "via",
                "target",
                "weight_1",
                "weight_2",
                "bottleneck_weight",
            ]
        )
    else:
        first_small = first[["source", "target", "weight"]].rename(
            columns={"target": "via", "weight": "weight_1"}
        )
        second_small = second[["source", "target", "weight"]].rename(
            columns={"source": "via", "weight": "weight_2"}
        )
        paths = first_small.merge(second_small, on="via", how="inner")
        paths["bottleneck_weight"] = paths[["weight_1", "weight_2"]].min(axis=1)
        paths = paths.sort_values(
            ["bottleneck_weight", "weight_1", "weight_2"],
            ascending=False,
        )

    threshold_sweep = []
    for threshold in thresholds:
        retained = paths[
            (paths.weight_1 >= threshold) & (paths.weight_2 >= threshold)
        ]
        threshold_sweep.append(
            {
                "min_weight_each_edge": float(threshold),
                "path_count": len(retained),
                "source_body_ids": sorted(set(retained.source.astype(int)))
                if len(retained)
                else [],
                "via_body_ids": sorted(set(retained.via.astype(int))) if len(retained) else [],
                "target_body_ids": sorted(set(retained.target.astype(int)))
                if len(retained)
                else [],
            }
        )

    return {
        **prediction,
        "source_population_count": len(populations[source_name]),
        "via_population_count": len(populations[via_name]),
        "target_population_count": len(populations[target_name]),
        "observed_path_count": len(paths),
        "threshold_sweep": threshold_sweep,
        "paths": _json_records(paths, limit=max_paths),
        "claim_boundary": (
            "Two-hop structural support is descriptive. It does not establish signal "
            "transmission, sign, timing, or necessity."
        ),
    }


def build_route_audit(
    annotations: pd.DataFrame,
    weights: pd.DataFrame,
    config: dict[str, Any],
    *,
    max_edges: int = 200,
    max_paths: int = 200,
) -> dict[str, Any]:
    if "bodyId" not in annotations.columns:
        raise ValueError("annotations require bodyId")
    annotations = annotations.copy()
    annotations["bodyId"] = annotations.bodyId.astype(int)
    edges = _normalize_edges(weights)

    population_names = tuple(str(name) for name in config["populations"])
    populations = {
        name: _population_rows(annotations, name, max_rows=1000)
        for name in population_names
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
    two_hop = [
        _two_hop_prediction(
            edges,
            populations,
            prediction,
            thresholds,
            max_paths=max_paths,
        )
        for prediction in config.get("two_hop_predictions", [])
    ]

    return {
        "protocol": str(config.get("protocol", "malecns-literature-route-audit-v1")),
        "dataset": str(config.get("dataset", "male-cns:v1.0")),
        "runtime": {
            "git_branch": _git_output("branch", "--show-current"),
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_dirty_paths": (_git_output("status", "--porcelain") or "").splitlines(),
        },
        "source_population_audit": _source_population_audit(annotations),
        "populations": {
            name: {
                "count": len(rows),
                "rows": _json_records(rows, limit=100),
            }
            for name, rows in populations.items()
        },
        "direct_predictions": direct,
        "two_hop_predictions": two_hop,
        "thresholds": list(thresholds),
        "claim_boundary": str(
            config.get(
                "claim_boundary",
                "Literature-guided structural audit only; no functional claim is qualified.",
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit literature-predicted MaleCNS route segments without shortest-path "
            "or fanout pruning"
        )
    )
    parser.add_argument("annotations", help="MaleCNS body annotation Feather file")
    parser.add_argument("weights", help="MaleCNS connection-weight Feather file")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output",
        default="results/route/literature-route-audit-v1.json",
    )
    parser.add_argument("--max-edges", type=int, default=200)
    parser.add_argument("--max-paths", type=int, default=200)
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    weights_path = Path(args.weights)
    config_path = Path(args.config)
    annotations = pd.read_feather(annotations_path)
    weights = pd.read_feather(weights_path)
    config = json.loads(config_path.read_text())
    report = build_route_audit(
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
        "weights": {
            "path": str(weights_path),
            "sha256": _sha256_file(weights_path),
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    concise = {
        "protocol": report["protocol"],
        "git_sha": report["runtime"]["git_sha"],
        "source_population_audit": {
            "olfactory_class_count": report["source_population_audit"]["olfactory_class_count"],
            "orn_type_prefix_count": report["source_population_audit"]["orn_type_prefix_count"],
            "olfactory_not_orn_count": report["source_population_audit"]["olfactory_not_orn_count"],
            "legacy_regex_counts": report["source_population_audit"]["legacy_regex_counts"],
        },
        "population_counts": {
            name: value["count"] for name, value in report["populations"].items()
        },
        "direct_support": {
            f"{row['source']}->{row['target']}": row["observed_weight_sum"]
            for row in report["direct_predictions"]
        },
        "two_hop_support": {
            f"{row['source']}->{row['via']}->{row['target']}": row["observed_path_count"]
            for row in report["two_hop_predictions"]
        },
        "output": str(output),
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
