from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from .graph import GraphBundle
from .public_data import download_file, sha256_file
from .rewire import save_bundle

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
WEIGHTS_URL = BASE + "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NT_URL = BASE + "body-neurotransmitters-male-cns-v1.0.feather"

# These signs are modeling assumptions applied to transmitter annotations, not
# electrophysiological measurements from MaleCNS. R002's selected LPLC2/LC4
# pathway is expected to be cholinergic; ambiguous transmitters are kept as zero.
NT_SIGN = {
    "acetylcholine": 1,
    "ach": 1,
    "gaba": -1,
    "glutamate": -1,
    "glu": -1,
}


def _body_column(frame: pd.DataFrame) -> str:
    for column in ("body", "bodyId", "body_id"):
        if column in frame.columns:
            return column
    raise ValueError(f"could not find body ID column in neurotransmitter table: {list(frame.columns)}")


def _nt_column(frame: pd.DataFrame) -> str:
    for column in ("consensus_nt", "consensusNt", "predictedNt", "celltypePredictedNt"):
        if column in frame.columns:
            return column
    raise ValueError(
        f"could not find consensus neurotransmitter column in table: {list(frame.columns)}"
    )


def _normalize_nt(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    aliases = {
        "acetylcholine": "acetylcholine",
        "ach": "acetylcholine",
        "gaba": "gaba",
        "glutamate": "glutamate",
        "glu": "glutamate",
    }
    return aliases.get(text, text or None)


def _selected_ids(authority: dict) -> set[int]:
    ids: set[int] = set()
    for members in authority["populations"].values():
        ids.update(int(body_id) for body_id in members)
    return ids


def _read_induced_edges(weights_path: Path, selected: set[int]) -> pd.DataFrame:
    selected_array = pa.array(sorted(selected), type=pa.int64())
    reader = ipc.open_file(pa.memory_map(str(weights_path)))
    tables: list[pa.Table] = []
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        names = set(batch.schema.names)
        required = {"body_pre", "body_post", "weight"}
        if not required.issubset(names):
            raise ValueError(f"weights table missing columns: {sorted(required - names)}")
        pre = pc.fill_null(pc.index_in(batch["body_pre"], value_set=selected_array), -1)
        post = pc.fill_null(pc.index_in(batch["body_post"], value_set=selected_array), -1)
        mask = pc.and_(pc.greater_equal(pre, 0), pc.greater_equal(post, 0))
        if pc.any(mask).as_py():
            tables.append(pa.Table.from_batches([batch.filter(mask)]))
    if not tables:
        return pd.DataFrame(columns=["body_pre", "body_post", "weight"])
    frame = pa.concat_tables(tables).select(["body_pre", "body_post", "weight"]).to_pandas()
    if frame.duplicated(["body_pre", "body_post"]).any():
        raise ValueError("MaleCNS weights contain duplicate body_pre/body_post rows")
    return frame.sort_values(["body_pre", "body_post"]).reset_index(drop=True)


def _assert_structural_totals(authority: dict, edges: pd.DataFrame) -> list[dict]:
    checks: list[dict] = []
    populations = {key: set(map(int, value)) for key, value in authority["populations"].items()}
    for expected in authority["structural_summaries"]:
        source_key = f"{expected['source_type']}_{expected['source_side']}"
        target_key = f"{expected['target_type']}_{expected['target_side']}"
        subset = edges[
            edges.body_pre.astype(int).isin(populations[source_key])
            & edges.body_post.astype(int).isin(populations[target_key])
        ]
        observed = int(subset.weight.astype(int).sum())
        wanted = int(expected["aggregate_connections"])
        checks.append(
            {
                "source": source_key,
                "target": target_key,
                "expected_connections": wanted,
                "observed_connections": observed,
                "edge_rows": int(len(subset)),
                "match": observed == wanted,
            }
        )
        if observed != wanted:
            raise ValueError(
                f"MaleCNS structural total mismatch for {source_key}->{target_key}: "
                f"explorer={wanted} weights={observed}"
            )
    return checks


def build_r002_bundle(
    *,
    authority_path: str | Path,
    annotations_path: str | Path,
    neurotransmitters_path: str | Path,
    weights_path: str | Path,
) -> GraphBundle:
    authority_path = Path(authority_path)
    annotations_path = Path(annotations_path)
    neurotransmitters_path = Path(neurotransmitters_path)
    weights_path = Path(weights_path)
    authority = json.loads(authority_path.read_text())
    if authority.get("dataset") != "male-cns:v1.0":
        raise ValueError("R002 requires authority for male-cns:v1.0")

    selected = _selected_ids(authority)
    annotations = pd.read_feather(annotations_path)
    if "bodyId" not in annotations.columns:
        raise ValueError("annotations table missing bodyId")
    nodes = annotations[annotations.bodyId.astype(int).isin(selected)].copy()
    observed_ids = set(nodes.bodyId.astype(int))
    missing = selected - observed_ids
    if missing:
        raise ValueError(f"authority references {len(missing)} missing annotation bodies")
    nodes = nodes.drop_duplicates("bodyId").sort_values("bodyId").reset_index(drop=True)
    if len(nodes) != len(selected):
        raise ValueError("annotation body IDs are not unique for selected R002 population")

    nt = pd.read_feather(neurotransmitters_path)
    body_col = _body_column(nt)
    nt_col = _nt_column(nt)
    nt_lookup = {
        int(body): _normalize_nt(value)
        for body, value in zip(nt[body_col], nt[nt_col], strict=True)
        if pd.notna(body)
    }
    nodes["consensus_nt"] = nodes.bodyId.astype(int).map(nt_lookup)

    raw_edges = _read_induced_edges(weights_path, selected)
    if raw_edges.empty:
        raise ValueError("R002 selected MaleCNS populations have no induced edges")
    structural_checks = _assert_structural_totals(authority, raw_edges)

    source_nt = raw_edges.body_pre.astype(int).map(nt_lookup)
    normalized = source_nt.map(_normalize_nt)
    signs = normalized.map(lambda value: NT_SIGN.get(value or "", 0)).astype(int)
    edges = pd.DataFrame(
        {
            "source": raw_edges.body_pre.astype(int),
            "target": raw_edges.body_post.astype(int),
            "weight": raw_edges.weight.astype(int),
            "sign": signs,
            "source_nt": normalized,
        }
    )

    populations = {key: list(map(int, value)) for key, value in authority["populations"].items()}
    roles = {
        "loom_size_left": populations["LPLC2_L"],
        "loom_size_right": populations["LPLC2_R"],
        "loom_velocity_left": populations["LC4_L"],
        "loom_velocity_right": populations["LC4_R"],
        "escape_left": populations["DNp01_L"],
        "escape_right": populations["DNp01_R"],
        "escape": populations["DNp01_L"] + populations["DNp01_R"],
    }

    active_sources = set(roles["loom_size_left"] + roles["loom_size_right"] + roles["loom_velocity_left"] + roles["loom_velocity_right"])
    active = edges[edges.source.isin(active_sources) & edges.target.isin(roles["escape"])]
    if active.empty:
        raise ValueError("R002 extracted graph contains no looming-population -> DNp01 edges")
    if (active.sign != 1).any():
        bad = active[active.sign != 1][["source", "source_nt", "sign"]].head(10).to_dict("records")
        raise ValueError(
            "R002 direct LPLC2/LC4 -> DNp01 pathway is expected to be cholinergic/excitatory; "
            f"observed non-positive source signs: {bad}"
        )

    manifest = {
        "schema": "fly-sniff-r002-graph-bundle-v1",
        "dataset": "male-cns:v1.0",
        "experiment": "R002-loom-escape",
        "qualification_status": "candidate",
        "scientific_claim_allowed": False,
        "authority_sha256": sha256_file(authority_path),
        "sources": {
            "annotations": {"path": str(annotations_path), "sha256": sha256_file(annotations_path)},
            "neurotransmitters": {
                "path": str(neurotransmitters_path),
                "sha256": sha256_file(neurotransmitters_path),
            },
            "weights": {"path": str(weights_path), "sha256": sha256_file(weights_path)},
        },
        "population_counts": {key: len(value) for key, value in populations.items()},
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "signed_edge_counts": {
            str(sign): int(count) for sign, count in edges.sign.value_counts().sort_index().items()
        },
        "structural_checks": structural_checks,
        "modeling_assumptions": {
            "dynamics": "generic leaky rate model; not measured electrophysiology",
            "input_encoding": "angular-size and positive-expansion adapters are modeled features",
            "transmitter_to_sign": NT_SIGN,
            "unknown_or_ambiguous_transmitter_sign": 0,
            "behavioral_readout": "DNp01/GF escape activity; not a steering command",
        },
    }
    bundle = GraphBundle(nodes=nodes, edges=edges, roles=roles, manifest=manifest)
    bundle.validate(require_sign=True, require_qualified=False)
    return bundle


def _download_raw(raw_dir: Path) -> tuple[Path, Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotations = raw_dir / "body-annotations-male-cns-v1.0.feather"
    nt = raw_dir / "body-neurotransmitters-male-cns-v1.0.feather"
    weights = raw_dir / "connectome-weights-male-cns-v1.0.feather"
    from .public_data import ANNOTATIONS_URL

    for url, path in ((ANNOTATIONS_URL, annotations), (NT_URL, nt), (WEIGHTS_URL, weights)):
        if not path.exists():
            print(f"downloading {url}", flush=True)
            download_file(url, path)
    return annotations, nt, weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the body-level R002 MaleCNS graph")
    parser.add_argument("authority", help="R002 authority JSON")
    parser.add_argument("--annotations")
    parser.add_argument("--neurotransmitters")
    parser.add_argument("--weights")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--raw-dir", default="data/raw/r002")
    parser.add_argument("--output", default="artifacts/r002/circuit")
    args = parser.parse_args()

    if args.download:
        annotations, nt, weights = _download_raw(Path(args.raw_dir))
    else:
        if not args.annotations or not args.neurotransmitters or not args.weights:
            parser.error("provide --annotations, --neurotransmitters, --weights or pass --download")
        annotations = Path(args.annotations)
        nt = Path(args.neurotransmitters)
        weights = Path(args.weights)

    bundle = build_r002_bundle(
        authority_path=args.authority,
        annotations_path=annotations,
        neurotransmitters_path=nt,
        weights_path=weights,
    )
    save_bundle(bundle, args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "nodes": len(bundle.nodes),
                "edges": len(bundle.edges),
                "qualification_status": bundle.manifest["qualification_status"],
                "structural_checks": bundle.manifest["structural_checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
