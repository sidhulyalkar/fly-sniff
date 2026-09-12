from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .trace import body_ids_matching
from .trace_audit import audit_directory

DEFAULT_SOURCE_PATTERNS = ("ORN", "^Or", "^Ir")
DEFAULT_ANCHORS = ("FB5AB", "PFNa", "PFNm", "PFNp", "hDeltaC", "PFL3", "DNa02")
SEARCH_COLUMNS = ("type", "instance", "class", "subclass")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        frame = frame.head(max(0, int(limit)))
    normalized = frame.astype(object).where(pd.notna(frame), None)
    return normalized.to_dict(orient="records")


def _annotation_summary(annotations: pd.DataFrame) -> dict[str, Any]:
    searchable = [column for column in SEARCH_COLUMNS if column in annotations.columns]
    summary: dict[str, Any] = {
        "rows": int(len(annotations)),
        "columns": list(annotations.columns),
        "searchable_columns": searchable,
        "body_id_unique": bool(annotations.bodyId.astype(int).is_unique),
    }
    for column in searchable:
        counts = annotations[column].dropna().astype(str).value_counts().head(25)
        summary[f"top_{column}"] = [
            {"value": str(value), "count": int(count)} for value, count in counts.items()
        ]
    return summary


def _pattern_audit(
    annotations: pd.DataFrame,
    patterns: tuple[str, ...],
    *,
    max_examples: int,
) -> dict[str, Any]:
    per_pattern: dict[str, set[int]] = {}
    details: dict[str, Any] = {}
    useful_columns = ["bodyId", *[c for c in SEARCH_COLUMNS if c in annotations.columns]]
    body_ids = annotations.bodyId.astype(int)

    for pattern in patterns:
        ids = body_ids_matching(annotations, [pattern])
        per_pattern[pattern] = ids
        rows = annotations.loc[body_ids.isin(ids), useful_columns].copy()
        type_counts = (
            rows["type"].fillna("<NA>").astype(str).value_counts().head(25)
            if "type" in rows.columns
            else pd.Series(dtype=int)
        )
        details[pattern] = {
            "count": int(len(ids)),
            "top_types": [
                {"type": str(value), "count": int(count)}
                for value, count in type_counts.items()
            ],
            "examples": _json_records(rows.sort_values("bodyId"), limit=max_examples),
        }

    union = set().union(*per_pattern.values()) if per_pattern else set()
    overlaps: dict[str, int] = {}
    for i, left in enumerate(patterns):
        for right in patterns[i + 1 :]:
            overlaps[f"{left} & {right}"] = int(len(per_pattern[left] & per_pattern[right]))
    if len(patterns) > 2:
        overlaps["all_patterns"] = int(len(set.intersection(*(per_pattern[p] for p in patterns))))

    return {
        "patterns": list(patterns),
        "union_count": int(len(union)),
        "per_pattern": details,
        "overlap_counts": overlaps,
    }


def _anchor_audit(
    annotations: pd.DataFrame,
    anchors: tuple[str, ...],
    *,
    max_examples: int,
) -> dict[str, Any]:
    searchable = [column for column in SEARCH_COLUMNS if column in annotations.columns]
    useful_columns = ["bodyId", *searchable]
    result: dict[str, Any] = {}

    for anchor in anchors:
        exact_mask = pd.Series(False, index=annotations.index)
        family_mask = pd.Series(False, index=annotations.index)
        anchor_lower = anchor.lower()
        for column in searchable:
            values = annotations[column].fillna("").astype(str).str.strip()
            lower = values.str.lower()
            exact_mask |= lower.eq(anchor_lower)
            family_mask |= lower.str.match(rf"^{anchor_lower}(?:$|[_-])", case=False)

        exact = annotations.loc[exact_mask, useful_columns].copy()
        family = annotations.loc[family_mask, useful_columns].copy()
        result[anchor] = {
            "exact_count": int(len(exact)),
            "family_count": int(len(family)),
            "exact_rows": _json_records(exact.sort_values("bodyId"), limit=max_examples),
            "family_rows": _json_records(family.sort_values("bodyId"), limit=max_examples),
        }
    return result


def _depth_profiles(
    nodes: pd.DataFrame,
    provenance: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_examples: int,
) -> dict[str, Any]:
    node_table = nodes.copy()
    node_table["bodyId"] = node_table.bodyId.astype(int)
    prov = provenance.copy()
    prov["bodyId"] = prov.bodyId.astype(int)
    merged = prov.merge(node_table, on="bodyId", how="left", validate="one_to_one")

    profiles: dict[str, Any] = {}
    if "type" in merged.columns:
        by_forward: dict[str, Any] = {}
        for depth, group in merged.groupby("forward_depth", dropna=False):
            key = "null" if pd.isna(depth) else str(int(depth))
            counts = group["type"].fillna("<NA>").astype(str).value_counts().head(20)
            by_forward[key] = [
                {"type": str(value), "count": int(count)} for value, count in counts.items()
            ]
        profiles["top_types_by_forward_depth"] = by_forward

    forward_lookup = pd.to_numeric(prov.set_index("bodyId").forward_depth, errors="coerce")
    edge_table = edges[["source", "target", "weight"]].copy()
    edge_table["source"] = edge_table.source.astype(int)
    edge_table["target"] = edge_table.target.astype(int)
    edge_table["source_forward_depth"] = edge_table.source.map(forward_lookup)
    edge_table["target_forward_depth"] = edge_table.target.map(forward_lookup)
    transition = (
        edge_table.dropna(subset=["source_forward_depth", "target_forward_depth"])
        .groupby(["source_forward_depth", "target_forward_depth"], as_index=False)
        .agg(edges=("weight", "size"), weight_sum=("weight", "sum"))
        .sort_values(["source_forward_depth", "target_forward_depth"])
    )
    profiles["forward_depth_edge_transitions"] = _json_records(transition)

    if "type" in node_table.columns and len(edges):
        type_lookup = node_table.drop_duplicates("bodyId").set_index("bodyId")["type"]
        typed = edge_table.copy()
        typed["source_type"] = typed.source.map(type_lookup).fillna("<NA>").astype(str)
        typed["target_type"] = typed.target.map(type_lookup).fillna("<NA>").astype(str)
        type_pairs = (
            typed.groupby(["source_type", "target_type"], as_index=False)
            .agg(edges=("weight", "size"), weight_sum=("weight", "sum"))
            .sort_values(["weight_sum", "edges"], ascending=False)
            .head(max(20, max_examples))
        )
        profiles["top_type_to_type_connections"] = _json_records(type_pairs)

    return profiles


def _anchor_depths(
    annotations: pd.DataFrame,
    provenance: pd.DataFrame,
    anchors: tuple[str, ...],
) -> dict[str, Any]:
    searchable = [column for column in SEARCH_COLUMNS if column in annotations.columns]
    useful = ["bodyId", *searchable]
    prov = provenance.copy()
    prov["bodyId"] = prov.bodyId.astype(int)
    out: dict[str, Any] = {}
    for anchor in anchors:
        mask = pd.Series(False, index=annotations.index)
        anchor_lower = anchor.lower()
        for column in searchable:
            values = annotations[column].fillna("").astype(str).str.strip().str.lower()
            mask |= values.str.match(rf"^{anchor_lower}(?:$|[_-])", case=False)
        rows = annotations.loc[mask, useful].copy()
        rows["bodyId"] = rows.bodyId.astype(int)
        joined = rows.merge(prov, on="bodyId", how="inner")
        out[anchor] = {
            "annotation_population_count": int(len(rows)),
            "retained_in_corridor_count": int(len(joined)),
            "retained_rows": _json_records(joined.sort_values("bodyId")),
        }
    return out


def build_handoff(
    annotations_path: str | Path,
    weights_path: str | Path,
    trace_dir: str | Path,
    *,
    source_patterns: tuple[str, ...] = DEFAULT_SOURCE_PATTERNS,
    anchors: tuple[str, ...] = DEFAULT_ANCHORS,
    max_examples: int = 20,
) -> dict[str, Any]:
    annotations_path = Path(annotations_path)
    weights_path = Path(weights_path)
    trace_dir = Path(trace_dir)

    annotations = pd.read_feather(annotations_path)
    if "bodyId" not in annotations.columns:
        raise ValueError("annotations require bodyId")

    trace_report_path = trace_dir / "trace_report.json"
    nodes_path = trace_dir / "nodes.parquet"
    edges_path = trace_dir / "edges.parquet"
    provenance_path = trace_dir / "path_provenance.csv"
    required_trace = [trace_report_path, nodes_path, edges_path, provenance_path]
    missing = [str(path) for path in required_trace if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "science handoff requires a completed trace; missing: " + ", ".join(missing)
        )

    trace_report = json.loads(trace_report_path.read_text())
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    provenance = pd.read_csv(provenance_path)
    audit = audit_directory(trace_dir, top_n=max_examples)

    return {
        "protocol": "fly-sniff-science-handoff-v1",
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git_branch": _git_output("branch", "--show-current"),
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_dirty_paths": (_git_output("status", "--porcelain") or "").splitlines(),
        },
        "inputs": {
            "annotations": _file_identity(annotations_path),
            "weights": _file_identity(weights_path),
            "trace_report": _file_identity(trace_report_path),
            "trace_nodes": _file_identity(nodes_path),
            "trace_edges": _file_identity(edges_path),
            "trace_provenance": _file_identity(provenance_path),
        },
        "annotations": _annotation_summary(annotations),
        "olfactory_seed_hypothesis": _pattern_audit(
            annotations, source_patterns, max_examples=max_examples
        ),
        "anchor_populations": _anchor_audit(
            annotations, anchors, max_examples=max_examples
        ),
        "trace_report": trace_report,
        "trace_audit": audit,
        "corridor_profiles": _depth_profiles(
            nodes, provenance, edges, max_examples=max_examples
        ),
        "anchor_corridor_membership": _anchor_depths(
            annotations, provenance, anchors
        ),
        "claim_boundary": (
            "Local structural/evidence handoff only. Annotation matches, graph depth, synapse weight, "
            "and corridor membership do not establish physiological activity or odor-navigation function."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a shareable local science handoff from MaleCNS + trace artifacts"
    )
    parser.add_argument("annotations", help="MaleCNS body annotation Feather file")
    parser.add_argument("weights", help="MaleCNS connection-weight Feather file")
    parser.add_argument(
        "--trace-dir",
        default="data/cache/staged-route-v1",
        help="completed fly-sniff-trace output directory",
    )
    parser.add_argument(
        "--output",
        default="results/handoff/science-handoff-v1.json",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--anchor", action="append", default=None)
    args = parser.parse_args()

    payload = build_handoff(
        args.annotations,
        args.weights,
        args.trace_dir,
        source_patterns=tuple(args.source or DEFAULT_SOURCE_PATTERNS),
        anchors=tuple(args.anchor or DEFAULT_ANCHORS),
        max_examples=args.max_examples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

    concise = {
        "protocol": payload["protocol"],
        "git_sha": payload["runtime"]["git_sha"],
        "input_hashes": {
            key: value["sha256"] for key, value in payload["inputs"].items()
        },
        "olfactory_seed_union": payload["olfactory_seed_hypothesis"]["union_count"],
        "trace_counts": payload["trace_audit"]["counts"],
        "trace_passed": payload["trace_audit"]["passed"],
        "anchor_corridor_counts": {
            key: value["retained_in_corridor_count"]
            for key, value in payload["anchor_corridor_membership"].items()
        },
        "output": str(output),
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
