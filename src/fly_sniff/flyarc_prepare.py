from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import ipc

from .flyarc import SELECTION_POLICY

MALECNS_RELEASE = "male-cns:v1.0"
EXPECTED_RETAINED_NEURONS = 166_700
EXPECTED_RETAINED_EDGES = 25_582_938
SIGN_POLICY = "consensus-fallback-whole-neuron-sign-v1"
SOURCE_LOCKS = {
    "annotations": {
        "bytes": 14_483_314,
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    },
    "neurotransmitters": {
        "bytes": 43_282_834,
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    },
    "connectivity": {
        "bytes": 1_051_241_946,
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    },
}

POSITIVE_NT = {
    "acetylcholine",
    "dopamine",
    "octopamine",
    "serotonin",
}
NEGATIVE_NT = {
    "gaba",
    "glutamate",
    "histamine",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_lock(path: str | Path, source_key: str) -> dict[str, Any]:
    if source_key not in SOURCE_LOCKS:
        raise ValueError(f"unknown FlyARC source lock {source_key!r}")
    source = Path(path)
    expected = SOURCE_LOCKS[source_key]
    observed_bytes = source.stat().st_size
    observed_sha = sha256_file(source)
    if observed_bytes != expected["bytes"] or observed_sha != expected["sha256"]:
        raise ValueError(
            f"{source_key} source lock mismatch: "
            f"bytes expected={expected['bytes']} observed={observed_bytes}; "
            f"sha256 expected={expected['sha256']} observed={observed_sha}"
        )
    return {"bytes": observed_bytes, "sha256": observed_sha}


def _resolve_column(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"could not resolve {label}; candidates={list(candidates)}, columns={list(columns)}")


def _iter_feather_columns(
    path: str | Path,
    columns: Sequence[str],
) -> Iterator[dict[str, np.ndarray]]:
    path = Path(path)
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        indices = []
        for column in columns:
            index = reader.schema.get_field_index(column)
            if index < 0:
                raise ValueError(f"{path} is missing required column {column!r}")
            indices.append(index)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            yield {
                column: batch.column(index).to_numpy(zero_copy_only=False)
                for column, index in zip(columns, indices, strict=True)
            }


def _membership_positions(
    sorted_ids: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.int64)
    positions = np.searchsorted(sorted_ids, values)
    valid = positions < len(sorted_ids)
    safe = positions.copy()
    safe[~valid] = 0
    valid &= sorted_ids[safe] == values
    return positions, valid


def retained_annotations(path: str | Path) -> pd.DataFrame:
    frame = pd.read_feather(path)
    body_col = _resolve_column(frame.columns, ("bodyId", "body"), "annotation body ID")
    if "superclass" not in frame.columns:
        raise ValueError(
            "MaleCNS v1.0 FlyARC preparation requires the curated superclass column"
        )
    superclass = frame["superclass"].fillna("").astype(str).str.strip()
    retained = frame.loc[superclass != ""].copy()
    retained = retained.rename(columns={body_col: "bodyId"})
    retained["bodyId"] = retained["bodyId"].astype(np.int64)
    if retained["bodyId"].duplicated().any():
        raise ValueError("retained MaleCNS annotations contain duplicate body IDs")
    return retained.sort_values("bodyId").reset_index(drop=True)


def scan_structural_strength(
    connectivity: str | Path,
    retained_ids: np.ndarray,
    *,
    min_weight: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    if min_weight < 1:
        raise ValueError("min_weight must be at least one")
    retained_ids = np.asarray(retained_ids, dtype=np.int64)
    incoming = np.zeros(len(retained_ids), dtype=np.float64)
    outgoing = np.zeros(len(retained_ids), dtype=np.float64)
    retained_edge_count = 0

    columns = ("body_pre", "body_post", "weight")
    for batch in _iter_feather_columns(connectivity, columns):
        pre = np.asarray(batch["body_pre"], dtype=np.int64)
        post = np.asarray(batch["body_post"], dtype=np.int64)
        weight = np.asarray(batch["weight"], dtype=np.int64)

        pre_pos, pre_valid = _membership_positions(retained_ids, pre)
        post_pos, post_valid = _membership_positions(retained_ids, post)
        valid = pre_valid & post_valid & (weight >= min_weight)
        if not valid.any():
            continue

        source_index = pre_pos[valid]
        target_index = post_pos[valid]
        transformed = np.log1p(weight[valid].astype(np.float64))
        outgoing += np.bincount(
            source_index,
            weights=transformed,
            minlength=len(retained_ids),
        )
        incoming += np.bincount(
            target_index,
            weights=transformed,
            minlength=len(retained_ids),
        )
        retained_edge_count += int(valid.sum())

    return incoming, outgoing, retained_edge_count


def select_body_ids(
    retained_ids: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    *,
    max_nodes: int,
) -> np.ndarray:
    if max_nodes <= 1:
        raise ValueError("max_nodes must be greater than one")
    if len(retained_ids) != len(incoming) or len(retained_ids) != len(outgoing):
        raise ValueError("strength vectors must align with retained body IDs")

    scores = np.sqrt(incoming * outgoing)
    order = np.lexsort((retained_ids, -scores))
    return np.asarray(retained_ids[order[: min(max_nodes, len(order))]], dtype=np.int64)


def extract_induced_edges(
    connectivity: str | Path,
    selected_ids: np.ndarray,
    *,
    min_weight: int = 1,
) -> pd.DataFrame:
    selected_sorted = np.sort(np.asarray(selected_ids, dtype=np.int64))
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []

    columns = ("body_pre", "body_post", "weight")
    for batch in _iter_feather_columns(connectivity, columns):
        pre = np.asarray(batch["body_pre"], dtype=np.int64)
        post = np.asarray(batch["body_post"], dtype=np.int64)
        weight = np.asarray(batch["weight"], dtype=np.int64)

        _, pre_valid = _membership_positions(selected_sorted, pre)
        _, post_valid = _membership_positions(selected_sorted, post)
        valid = pre_valid & post_valid & (weight >= min_weight)
        if not valid.any():
            continue
        source_parts.append(pre[valid])
        target_parts.append(post[valid])
        weight_parts.append(weight[valid])

    if not source_parts:
        raise ValueError("selected FlyARC structural core contains no induced edges")

    edges = pd.DataFrame(
        {
            "source": np.concatenate(source_parts),
            "target": np.concatenate(target_parts),
            "weight": np.concatenate(weight_parts),
        }
    )
    if edges.duplicated(["source", "target"]).any():
        raise ValueError("official connectivity unexpectedly contains duplicate directed pairs")
    return edges.reset_index(drop=True)


def _normalize_nt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _choose_transmitter(row: pd.Series) -> tuple[str, str]:
    consensus = _normalize_nt(row.get("consensus_nt"))
    if consensus in POSITIVE_NT or consensus in NEGATIVE_NT:
        return consensus, "consensus_nt"

    predicted = _normalize_nt(row.get("predicted_nt"))
    confidence = row.get("predicted_nt_confidence")
    if (
        predicted in POSITIVE_NT | NEGATIVE_NT
        and confidence is not None
        and not pd.isna(confidence)
        and float(confidence) >= 0.5
    ):
        return predicted, "predicted_nt>=0.5"

    celltype = _normalize_nt(row.get("celltype_predicted_nt"))
    if celltype in POSITIVE_NT or celltype in NEGATIVE_NT:
        return celltype, "celltype_predicted_nt"

    return consensus or predicted or celltype or "unclear", "unresolved"


def selected_neurotransmitters(
    path: str | Path,
    selected_ids: Sequence[int],
    *,
    unresolved_sign: int,
) -> pd.DataFrame:
    if unresolved_sign not in {-1, 0, 1}:
        raise ValueError("unresolved_sign must be -1, 0, or +1")

    frame = pd.read_feather(path)
    body_col = _resolve_column(frame.columns, ("body", "bodyId"), "neurotransmitter body ID")
    frame[body_col] = frame[body_col].astype(np.int64)
    selected = {int(x) for x in selected_ids}
    frame = frame[frame[body_col].isin(selected)].copy()
    if frame.empty:
        raise ValueError("neurotransmitter table contains none of the selected neurons")

    records: list[dict[str, Any]] = []
    for body_id, group in frame.groupby(body_col, sort=False):
        choices = [_choose_transmitter(row) for _, row in group.iterrows()]
        resolved = {(nt, source) for nt, source in choices if source != "unresolved"}
        if len({nt for nt, _ in resolved}) > 1:
            raise ValueError(
                f"conflicting transmitter assignments for body {int(body_id)}: {sorted(resolved)}"
            )
        if resolved:
            nt, source = min(resolved)
        else:
            nt, source = choices[0]

        if nt in NEGATIVE_NT:
            sign = -1
        elif nt in POSITIVE_NT:
            sign = 1
        else:
            sign = unresolved_sign
        records.append(
            {
                "bodyId": int(body_id),
                "flyarc_nt": nt,
                "flyarc_nt_source": source,
                "flyarc_sign": int(sign),
            }
        )

    result = pd.DataFrame(records)
    missing = selected - set(result["bodyId"].astype(int))
    if missing:
        raise ValueError(
            f"neurotransmitter table is missing {len(missing)} selected body IDs"
        )
    return result


def prepare_graph(
    *,
    annotations: str | Path,
    connectivity: str | Path,
    neurotransmitters: str | Path,
    output: str | Path,
    max_nodes: int = 4096,
    min_weight: int = 1,
    unresolved_sign: int = 1,
    expected_neurons: int | None = EXPECTED_RETAINED_NEURONS,
    expected_edges: int | None = EXPECTED_RETAINED_EDGES,
    verify_sources: bool = True,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    source_locks = None
    if verify_sources:
        source_locks = {
            "annotations": verify_source_lock(annotations, "annotations"),
            "connectivity": verify_source_lock(connectivity, "connectivity"),
            "neurotransmitters": verify_source_lock(
                neurotransmitters,
                "neurotransmitters",
            ),
        }

    retained = retained_annotations(annotations)
    if expected_neurons is not None and len(retained) != expected_neurons:
        raise ValueError(
            f"retained MaleCNS neuron count mismatch: expected {expected_neurons}, "
            f"observed {len(retained)}"
        )

    retained_ids = retained["bodyId"].to_numpy(dtype=np.int64)
    incoming, outgoing, retained_edge_count = scan_structural_strength(
        connectivity,
        retained_ids,
        min_weight=min_weight,
    )
    if min_weight == 1 and expected_edges is not None and retained_edge_count != expected_edges:
        raise ValueError(
            f"retained MaleCNS edge count mismatch: expected {expected_edges}, "
            f"observed {retained_edge_count}"
        )

    selected_ids = select_body_ids(
        retained_ids,
        incoming,
        outgoing,
        max_nodes=max_nodes,
    )
    edges = extract_induced_edges(
        connectivity,
        selected_ids,
        min_weight=min_weight,
    )
    nt = selected_neurotransmitters(
        neurotransmitters,
        selected_ids,
        unresolved_sign=unresolved_sign,
    )

    selected_set = {int(x) for x in selected_ids}
    nodes = retained[retained["bodyId"].astype(int).isin(selected_set)].copy()
    order = {int(body_id): index for index, body_id in enumerate(selected_ids)}
    nodes["_flyarc_order"] = nodes["bodyId"].astype(int).map(order)
    nodes = nodes.sort_values("_flyarc_order").drop(columns="_flyarc_order")
    nodes = nodes.merge(nt, on="bodyId", how="left", validate="one_to_one")
    sign_map = nodes.set_index("bodyId")["flyarc_sign"].astype(int).to_dict()
    edges["sign"] = edges["source"].astype(int).map(sign_map)
    if edges["sign"].isna().any():
        raise ValueError("failed to assign a presynaptic sign to every selected edge")
    edges["sign"] = edges["sign"].astype(int)

    nodes_path = output / "nodes.parquet"
    edges_path = output / "edges.parquet"
    roles_path = output / "roles.json"
    manifest_path = output / "manifest.json"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    roles_path.write_text("{}\n")

    nt_counts = nodes["flyarc_nt"].fillna("missing").value_counts().sort_index().to_dict()
    sign_counts = nodes["flyarc_sign"].value_counts().sort_index().to_dict()
    manifest = {
        "schema_version": 1,
        "dataset": MALECNS_RELEASE,
        "qualification_status": "candidate",
        "purpose": "task-blind structural reservoir core for FlyARC development",
        "selection_policy": SELECTION_POLICY,
        "max_nodes": max_nodes,
        "min_weight": min_weight,
        "retention": "non-empty curated superclass",
        "retained_neurons_scanned": len(retained),
        "retained_edges_scanned": retained_edge_count,
        "selected_neurons": len(nodes),
        "selected_induced_edges": len(edges),
        "sign_policy": SIGN_POLICY,
        "unresolved_sign": unresolved_sign,
        "neurotransmitter_counts": {str(k): int(v) for k, v in nt_counts.items()},
        "sign_counts": {str(k): int(v) for k, v in sign_counts.items()},
        "source_lock_verified": bool(verify_sources),
        "source_locks": source_locks,
        "sources": {
            "annotations": {
                "path": str(Path(annotations)),
                "sha256": sha256_file(annotations),
            },
            "connectivity": {
                "path": str(Path(connectivity)),
                "sha256": sha256_file(connectivity),
            },
            "neurotransmitters": {
                "path": str(Path(neurotransmitters)),
                "sha256": sha256_file(neurotransmitters),
            },
        },
        "artifacts": {
            "nodes.parquet": sha256_file(nodes_path),
            "edges.parquet": sha256_file(edges_path),
            "roles.json": sha256_file(roles_path),
        },
        "claim_boundaries": [
            "This graph is selected without ARC observations or rewards.",
            "Neurotransmitter-to-sign conversion is a modeling assumption, not a receptor-specific conductance model.",
            "The selected core is a development reservoir and is not a natural ARC-processing circuit.",
            "qualification_status=candidate prevents promotion to a biological task claim.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a task-blind signed MaleCNS structural core for FlyARC from official "
            "MaleCNS v1.0 bulk Feather tables"
        )
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--connectivity", required=True)
    parser.add_argument("--neurotransmitters", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--min-weight", type=int, default=1)
    parser.add_argument(
        "--unresolved-sign",
        type=int,
        choices=(-1, 0, 1),
        default=1,
        help="modeled sign for unresolved transmitter identity; default +1",
    )
    parser.add_argument(
        "--skip-count-checks",
        action="store_true",
        help="development escape hatch for noncanonical/future source tables",
    )
    parser.add_argument(
        "--skip-source-locks",
        action="store_true",
        help="development only: bypass canonical MaleCNS v1.0 byte/SHA-256 locks",
    )
    args = parser.parse_args()

    manifest = prepare_graph(
        annotations=args.annotations,
        connectivity=args.connectivity,
        neurotransmitters=args.neurotransmitters,
        output=args.output,
        max_nodes=args.max_nodes,
        min_weight=args.min_weight,
        unresolved_sign=args.unresolved_sign,
        expected_neurons=None if args.skip_count_checks else EXPECTED_RETAINED_NEURONS,
        expected_edges=None if args.skip_count_checks else EXPECTED_RETAINED_EDGES,
        verify_sources=not args.skip_source_locks,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
