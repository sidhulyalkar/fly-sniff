from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _finite_int_hist(values: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    counts = numeric.value_counts().sort_index()
    return {str(int(index)): int(value) for index, value in counts.items()}


def _quantiles(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {}
    quantile_points = {
        "min": 0.0,
        "q25": 0.25,
        "median": 0.50,
        "q75": 0.75,
        "q95": 0.95,
        "q99": 0.99,
        "max": 1.0,
    }
    return {label: float(numeric.quantile(q)) for label, q in quantile_points.items()}


def _bool_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = sorted(set(normalized.dropna()) - allowed)
    if unexpected:
        raise ValueError(f"cannot interpret boolean provenance values {unexpected}")
    return normalized.isin({"true", "1"})


def _annotation_label(row: pd.Series) -> str | None:
    for column in ("type", "instance", "class", "subclass"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return None


def audit_corridor(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    provenance: pd.DataFrame,
    trace_report: dict[str, Any] | None = None,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """Audit a traced structural corridor without assigning functional meaning.

    The report checks closure and summarizes graph geometry. High degree, high
    weight, or short structural distance are descriptive properties only; they
    are not interpreted as neural importance or physiological influence.
    """
    required_node_columns = {"bodyId"}
    required_edge_columns = {"source", "target", "weight"}
    required_provenance_columns = {
        "bodyId",
        "forward_depth",
        "reverse_depth",
        "is_source_seed",
        "is_target_seed",
    }
    if not required_node_columns.issubset(nodes.columns):
        raise ValueError(f"nodes missing columns {sorted(required_node_columns - set(nodes.columns))}")
    if not required_edge_columns.issubset(edges.columns):
        raise ValueError(f"edges missing columns {sorted(required_edge_columns - set(edges.columns))}")
    if not required_provenance_columns.issubset(provenance.columns):
        raise ValueError(
            "provenance missing columns "
            f"{sorted(required_provenance_columns - set(provenance.columns))}"
        )

    node_ids = set(nodes.bodyId.astype(int))
    provenance_ids = set(provenance.bodyId.astype(int))
    source_mask = _bool_mask(provenance.is_source_seed)
    target_mask = _bool_mask(provenance.is_target_seed)
    source_ids = set(provenance.loc[source_mask, "bodyId"].astype(int))
    target_ids = set(provenance.loc[target_mask, "bodyId"].astype(int))

    edge_sources = edges.source.astype(int)
    edge_targets = edges.target.astype(int)
    endpoint_ids = set(edge_sources) | set(edge_targets)

    minimum_weight = None
    max_hops = None
    if trace_report is not None:
        if trace_report.get("min_weight") is not None:
            minimum_weight = float(trace_report["min_weight"])
        if trace_report.get("max_hops") is not None:
            max_hops = int(trace_report["max_hops"])

    weight_threshold_ok = True
    if minimum_weight is not None and len(edges):
        weight_threshold_ok = bool((edges.weight.astype(float) >= minimum_weight).all())

    prov = provenance.set_index(provenance.bodyId.astype(int), drop=False)
    forward = pd.to_numeric(prov.forward_depth, errors="coerce")
    reverse = pd.to_numeric(prov.reverse_depth, errors="coerce")
    bounded_length = forward + reverse
    bounded_hop_limit_ok = True
    if max_hops is not None and bounded_length.notna().any():
        bounded_hop_limit_ok = bool((bounded_length.dropna() <= max_hops).all())

    edge_geometry = pd.DataFrame(
        {
            "source": edge_sources.to_numpy(),
            "target": edge_targets.to_numpy(),
            "weight": edges.weight.astype(float).to_numpy(),
        }
    )
    edge_geometry["source_forward_depth"] = edge_geometry.source.map(forward)
    edge_geometry["target_forward_depth"] = edge_geometry.target.map(forward)
    edge_geometry["source_reverse_depth"] = edge_geometry.source.map(reverse)
    edge_geometry["target_reverse_depth"] = edge_geometry.target.map(reverse)

    fwd_delta = edge_geometry.target_forward_depth - edge_geometry.source_forward_depth
    rev_delta = edge_geometry.source_reverse_depth - edge_geometry.target_reverse_depth
    finite_geometry = fwd_delta.notna() & rev_delta.notna()
    on_shortest_layer_step = finite_geometry & (fwd_delta == 1) & (rev_delta == 1)
    forward_progress = finite_geometry & (fwd_delta > 0)

    degree = pd.DataFrame(index=sorted(node_ids))
    degree.index.name = "bodyId"
    degree["in_degree"] = edge_targets.value_counts().reindex(degree.index, fill_value=0).astype(int)
    degree["out_degree"] = edge_sources.value_counts().reindex(degree.index, fill_value=0).astype(int)
    weighted_in = edges.groupby(edge_targets).weight.sum() if len(edges) else pd.Series(dtype=float)
    weighted_out = edges.groupby(edge_sources).weight.sum() if len(edges) else pd.Series(dtype=float)
    degree["weighted_in"] = weighted_in.reindex(degree.index, fill_value=0.0).astype(float)
    degree["weighted_out"] = weighted_out.reindex(degree.index, fill_value=0.0).astype(float)
    degree["total_degree"] = degree.in_degree + degree.out_degree
    degree["weighted_total"] = degree.weighted_in + degree.weighted_out

    annotation_rows = nodes.copy()
    annotation_rows["bodyId"] = annotation_rows.bodyId.astype(int)
    annotation_rows = annotation_rows.drop_duplicates("bodyId").set_index("bodyId")

    hubs: list[dict[str, Any]] = []
    for body_id, row in degree.sort_values(
        ["total_degree", "weighted_total"], ascending=False
    ).head(max(0, int(top_n))).iterrows():
        annotation = (
            annotation_rows.loc[body_id]
            if body_id in annotation_rows.index
            else pd.Series(dtype=object)
        )
        hubs.append(
            {
                "bodyId": int(body_id),
                "label": _annotation_label(annotation),
                "in_degree": int(row.in_degree),
                "out_degree": int(row.out_degree),
                "weighted_in": float(row.weighted_in),
                "weighted_out": float(row.weighted_out),
                "warning": "structural degree only; not functional importance",
            }
        )

    report_counts_match = True
    report_mismatches: dict[str, dict[str, int]] = {}
    if trace_report is not None:
        observed = {
            "source_seed_count": len(source_ids),
            "target_seed_count": len(target_ids),
            "corridor_nodes": len(nodes),
            "corridor_edges": len(edges),
        }
        for key, value in observed.items():
            if key in trace_report and int(trace_report[key]) != int(value):
                report_counts_match = False
                report_mismatches[key] = {
                    "trace_report": int(trace_report[key]),
                    "observed": int(value),
                }

    checks = {
        "node_ids_unique": bool(nodes.bodyId.astype(int).is_unique),
        "provenance_ids_unique": bool(provenance.bodyId.astype(int).is_unique),
        "node_provenance_id_match": node_ids == provenance_ids,
        "edge_endpoint_closure": endpoint_ids.issubset(node_ids),
        "source_seed_closure": source_ids.issubset(node_ids),
        "target_seed_closure": target_ids.issubset(node_ids),
        "nonnegative_weights": bool((edges.weight.astype(float) >= 0.0).all()),
        "trace_min_weight_respected": weight_threshold_ok,
        "bounded_hop_limit_respected": bounded_hop_limit_ok,
        "trace_report_counts_match": report_counts_match,
    }

    passed = all(checks.values())
    max_bounded = float(bounded_length.max()) if bounded_length.notna().any() else None

    return {
        "protocol": "structural-corridor-audit-v1",
        "passed": passed,
        "checks": checks,
        "report_mismatches": report_mismatches,
        "counts": {
            "nodes": int(len(nodes)),
            "edges": int(len(edges)),
            "source_seeds": int(len(source_ids)),
            "target_seeds": int(len(target_ids)),
        },
        "depths": {
            "forward": _finite_int_hist(provenance.forward_depth),
            "reverse": _finite_int_hist(provenance.reverse_depth),
            "bounded_path_length": _finite_int_hist(bounded_length),
            "max_bounded_path_length": max_bounded,
        },
        "edge_geometry": {
            "finite_depth_edges": int(finite_geometry.sum()),
            "forward_progress_edges": int(forward_progress.sum()),
            "forward_progress_fraction": (
                float(forward_progress.sum() / finite_geometry.sum())
                if finite_geometry.any()
                else None
            ),
            "shortest_layer_step_edges": int(on_shortest_layer_step.sum()),
            "shortest_layer_step_fraction": (
                float(on_shortest_layer_step.sum() / finite_geometry.sum())
                if finite_geometry.any()
                else None
            ),
            "warning": (
                "depth progression describes the bounded graph search only; it is not a functional "
                "or temporal neural pathway"
            ),
        },
        "edge_weight_quantiles": _quantiles(edges.weight),
        "degree_quantiles": {
            "in_degree": _quantiles(degree.in_degree),
            "out_degree": _quantiles(degree.out_degree),
            "weighted_in": _quantiles(degree.weighted_in),
            "weighted_out": _quantiles(degree.weighted_out),
        },
        "top_structural_hubs": hubs,
        "warning": (
            "Structural audit only. Connectivity, degree, edge weight, and graph distance do not "
            "establish physiological activity, causal influence, or odor-navigation function."
        ),
    }


def audit_directory(path: str | Path, *, top_n: int = 20) -> dict[str, Any]:
    root = Path(path)
    nodes = pd.read_parquet(root / "nodes.parquet")
    edges = pd.read_parquet(root / "edges.parquet")
    provenance = pd.read_csv(root / "path_provenance.csv")
    trace_report_path = root / "trace_report.json"
    trace_report = json.loads(trace_report_path.read_text()) if trace_report_path.exists() else None
    return audit_corridor(nodes, edges, provenance, trace_report, top_n=top_n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a fly-sniff structural trace without assigning functional meaning"
    )
    parser.add_argument("trace_dir", help="directory containing trace parquet/csv artifacts")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    report = audit_directory(args.trace_dir, top_n=args.top_n)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
