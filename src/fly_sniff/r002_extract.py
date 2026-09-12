from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from .public_data import CONNECTOME_WEIGHTS_URL, download_file, sha256_file
from .r002_authority import DATASET

NT_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-neurotransmitters-male-cns-v1.0.feather"
)

# These are explicit model assumptions about the dominant fast transmitter effect.
# Unknown/modulatory identities are preserved as sign=0 rather than silently made excitatory.
NT_SIGN = {
    "acetylcholine": 1,
    "ach": 1,
    "gaba": -1,
    "glutamate": -1,
    "glu": -1,
}


def _consensus_nt_column(frame: pd.DataFrame) -> str:
    for name in ("consensus_nt", "consensusNt", "nt"):
        if name in frame.columns:
            return name
    raise ValueError(f"no consensus neurotransmitter column found: {list(frame.columns)}")


def _nt_body_column(frame: pd.DataFrame) -> str:
    for name in ("body", "bodyId"):
        if name in frame.columns:
            return name
    raise ValueError(f"no neurotransmitter body-id column found: {list(frame.columns)}")


def _canonical_nt(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    aliases = {
        "acetylcholine": "acetylcholine",
        "ach": "acetylcholine",
        "gaba": "gaba",
        "glutamate": "glutamate",
        "glu": "glutamate",
        "dopamine": "dopamine",
        "da": "dopamine",
        "serotonin": "serotonin",
        "5ht": "serotonin",
        "octopamine": "octopamine",
        "oa": "octopamine",
        "unclear": "unclear",
        "unc": "unclear",
    }
    return aliases.get(text, text or None)


def _read_selected_edges(weights_path: Path, selected_ids: set[int]) -> pd.DataFrame:
    """Stream only edges whose endpoints are both in the bounded R002 population set."""

    ids = pa.array(sorted(selected_ids), type=pa.int64())
    reader = ipc.open_file(pa.memory_map(str(weights_path)))
    tables: list[pa.Table] = []
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        pre = pc.fill_null(pc.index_in(batch["body_pre"], value_set=ids), -1)
        post = pc.fill_null(pc.index_in(batch["body_post"], value_set=ids), -1)
        mask = pc.and_(pc.greater_equal(pre, 0), pc.greater_equal(post, 0))
        if pc.any(mask).as_py():
            tables.append(pa.Table.from_batches([batch.filter(mask)]))
    if not tables:
        return pd.DataFrame(columns=["body_pre", "body_post", "weight"])
    frame = pa.concat_tables(tables).to_pandas()
    required = {"body_pre", "body_post", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MaleCNS weights missing required columns: {sorted(missing)}")
    if frame.duplicated(["body_pre", "body_post"]).any():
        raise ValueError("MaleCNS weight table contains duplicate body_pre/body_post pairs")
    return frame.sort_values(["body_pre", "body_post"]).reset_index(drop=True)


def _expected_structural_sums(authority: dict) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in authority["structural_summaries"]:
        source = f"{row['source_type']}_{row['source_side']}"
        target = f"{row['target_type']}_{row['target_side']}"
        out[(source, target)] = int(row["aggregate_connections"])
    return out


def build_r002_bundle(
    authority: dict,
    *,
    weights_path: str | Path,
    neurotransmitters_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]], dict]:
    """Build a bounded GraphBundle candidate and cross-check release-level edge sums."""

    if authority.get("dataset") != DATASET:
        raise ValueError(f"expected {DATASET}, got {authority.get('dataset')}")
    populations = {
        key: [int(value) for value in values]
        for key, values in authority["populations"].items()
    }
    selected_ids = {body for values in populations.values() for body in values}
    if len(selected_ids) != sum(len(values) for values in populations.values()):
        raise ValueError("R002 authority populations overlap unexpectedly")

    weights_path = Path(weights_path)
    nt_path = Path(neurotransmitters_path)
    raw_edges = _read_selected_edges(weights_path, selected_ids)

    # Verify the large body-level release reproduces the independent explorer summaries.
    expected = _expected_structural_sums(authority)
    for (source_key, target_key), expected_sum in expected.items():
        source_ids = set(populations[source_key])
        target_ids = set(populations[target_key])
        observed = int(
            raw_edges.loc[
                raw_edges.body_pre.isin(source_ids) & raw_edges.body_post.isin(target_ids),
                "weight",
            ].sum()
        )
        if observed != expected_sum:
            raise ValueError(
                f"structural cross-check failed for {source_key}->{target_key}: "
                f"weights={observed} explorer={expected_sum}"
            )

    nt = pd.read_feather(nt_path)
    body_col = _nt_body_column(nt)
    nt_col = _consensus_nt_column(nt)
    nt_map = {
        int(row[body_col]): _canonical_nt(row[nt_col])
        for _, row in nt[[body_col, nt_col]].iterrows()
    }

    membership: dict[int, str] = {}
    for population, bodies in populations.items():
        for body in bodies:
            membership[body] = population

    nodes = pd.DataFrame(
        {
            "bodyId": sorted(selected_ids),
        }
    )
    nodes["population"] = nodes.bodyId.map(membership)
    nodes["type"] = nodes.population.str.rsplit("_", n=1).str[0]
    nodes["side"] = nodes.population.str.rsplit("_", n=1).str[1]
    nodes["consensus_nt"] = nodes.bodyId.map(nt_map)

    edges = raw_edges.rename(columns={"body_pre": "source", "body_post": "target"})[
        ["source", "target", "weight"]
    ].copy()
    source_nt = edges.source.map(nt_map)
    edges["source_nt"] = source_nt
    edges["sign"] = source_nt.map(lambda value: NT_SIGN.get(str(value).lower(), 0) if value else 0)

    roles = {
        "loom_size_left": populations["LPLC2_L"],
        "loom_size_right": populations["LPLC2_R"],
        "loom_velocity_left": populations["LC4_L"],
        "loom_velocity_right": populations["LC4_R"],
        "escape_left": populations["DNp01_L"],
        "escape_right": populations["DNp01_R"],
        "escape": populations["DNp01_L"] + populations["DNp01_R"],
    }

    zero_sign_edges = int((edges.sign == 0).sum())
    manifest = {
        "schema": "fly-sniff-r002-graph-v1",
        "dataset": DATASET,
        "qualification_status": "candidate",
        "scientific_claim_allowed": False,
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "source_files": {
            "weights": {
                "url": CONNECTOME_WEIGHTS_URL,
                "sha256": sha256_file(weights_path),
            },
            "neurotransmitters": {
                "url": NT_URL,
                "sha256": sha256_file(nt_path),
            },
            "authority_schema": authority.get("schema"),
        },
        "sign_model": {
            "acetylcholine": 1,
            "gaba": -1,
            "glutamate": -1,
            "other_or_unknown": 0,
            "status": "explicit-dynamics-assumption-not-connectome-measurement",
        },
        "zero_sign_edges": zero_sign_edges,
        "structural_cross_checks": [
            {
                "source": source,
                "target": target,
                "expected_connections": expected_sum,
                "status": "exact-match",
            }
            for (source, target), expected_sum in sorted(expected.items())
        ],
        "behavioral_readout": {
            "primary": "DNp01/GF escape activation",
            "directional_steering_claim": False,
        },
        "qualification_blockers": [
            "pass R002 circuit-sanity and lesion tests on exact extracted graph",
            "freeze held-out direct-hit/near-miss benchmark seeds and thresholds",
            "demonstrate intact-vs-rewire and sensory-lesion effects without directional steering overclaim",
        ],
    }
    return nodes, edges, roles, manifest


def write_bundle(
    directory: str | Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    roles: dict[str, list[int]],
    manifest: dict,
) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(root / "nodes.parquet", index=False)
    edges.to_parquet(root / "edges.parquet", index=False)
    (root / "roles.json").write_text(json.dumps(roles, indent=2, sort_keys=True) + "\n")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _download(path: Path, url: str) -> Path:
    if not path.exists():
        download_file(url, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract bounded body-level R002 MaleCNS graph")
    parser.add_argument("authority", help="candidate R002 authority JSON")
    parser.add_argument("--weights")
    parser.add_argument("--neurotransmitters")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache-dir", default="data/raw/r002")
    parser.add_argument("--output", default="artifacts/r002/graph")
    args = parser.parse_args()

    authority = json.loads(Path(args.authority).read_text())
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if args.download:
        weights = _download(cache / "connectome-weights-male-cns-v1.0.feather", CONNECTOME_WEIGHTS_URL)
        nt = _download(cache / "body-neurotransmitters-male-cns-v1.0.feather", NT_URL)
    else:
        if not args.weights or not args.neurotransmitters:
            parser.error("provide --weights and --neurotransmitters or pass --download")
        weights = Path(args.weights)
        nt = Path(args.neurotransmitters)

    nodes, edges, roles, manifest = build_r002_bundle(
        authority,
        weights_path=weights,
        neurotransmitters_path=nt,
    )
    write_bundle(args.output, nodes, edges, roles, manifest)
    print(
        json.dumps(
            {
                "output": args.output,
                "nodes": len(nodes),
                "edges": len(edges),
                "zero_sign_edges": manifest["zero_sign_edges"],
                "qualification_status": manifest["qualification_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
