from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from .graph import GraphBundle
from .public_data import ANNOTATIONS_URL, CONNECTOME_WEIGHTS_URL, download_file, sha256_file

NEUROTRANSMITTERS_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-neurotransmitters-male-cns-v1.0.feather"
)

# Receptor effects are modeled assumptions, not measurements in the connectome.
# For this candidate circuit the relevant visual populations are cholinergic, but
# the policy remains explicit so unknown/modulatory sources cannot silently become
# excitatory edges.
NT_SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}


def _selected_ids(authority: dict) -> set[int]:
    populations = authority.get("populations", {})
    required = {"LPLC2_L", "LPLC2_R", "LC4_L", "LC4_R", "DNp01_L", "DNp01_R"}
    missing = required - set(populations)
    if missing:
        raise ValueError(f"authority missing R002 populations: {sorted(missing)}")
    return {int(body) for name in required for body in populations[name]}


def _stream_selected_edges(weights_path: Path, selected: set[int]) -> pd.DataFrame:
    values = pa.array(sorted(selected), type=pa.int64())
    reader = ipc.open_file(pa.memory_map(str(weights_path)))
    tables: list[pa.Table] = []
    for index in range(reader.num_record_batches):
        batch = reader.get_batch(index)
        names = set(batch.schema.names)
        required = {"body_pre", "body_post", "weight"}
        if not required.issubset(names):
            raise ValueError(f"weights file missing columns: {sorted(required - names)}")
        pre = pc.is_in(batch["body_pre"], value_set=values)
        post = pc.is_in(batch["body_post"], value_set=values)
        mask = pc.and_(pre, post)
        if bool(pc.any(mask).as_py()):
            filtered = batch.filter(mask)
            tables.append(pa.Table.from_batches([filtered]).select(["body_pre", "body_post", "weight"]))
    if not tables:
        return pd.DataFrame(columns=["source", "target", "weight"])
    frame = pa.concat_tables(tables).to_pandas()
    if frame.duplicated(["body_pre", "body_post"]).any():
        raise ValueError("MaleCNS weight table contains duplicate selected body pairs")
    return frame.rename(columns={"body_pre": "source", "body_post": "target"})


def _transmitter_signs(nt_path: Path, selected: set[int]) -> tuple[dict[int, str | None], dict[int, int]]:
    frame = pd.read_feather(nt_path)
    required = {"body", "consensus_nt"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"neurotransmitter table missing columns: {sorted(missing)}")
    frame = frame[frame["body"].astype(int).isin(selected)].copy()
    names: dict[int, str | None] = {}
    signs: dict[int, int] = {}
    for row in frame.itertuples(index=False):
        body = int(row.body)
        raw = row.consensus_nt
        name = None if pd.isna(raw) else str(raw).strip().lower()
        names[body] = name
        signs[body] = NT_SIGN.get(name or "", 0)
    for body in selected:
        names.setdefault(body, None)
        signs.setdefault(body, 0)
    return names, signs


def _validate_structural_sums(edges: pd.DataFrame, authority: dict) -> list[dict]:
    results: list[dict] = []
    populations = authority["populations"]
    for expected in authority["structural_summaries"]:
        source_key = f"{expected['source_type']}_{expected['source_side']}"
        target_key = f"{expected['target_type']}_{expected['target_side']}"
        source_ids = {int(value) for value in populations[source_key]}
        target_ids = {int(value) for value in populations[target_key]}
        mask = edges["source"].astype(int).isin(source_ids)
        mask &= edges["target"].astype(int).isin(target_ids)
        observed = int(edges.loc[mask, "weight"].astype(int).sum())
        wanted = int(expected["aggregate_connections"])
        if observed != wanted:
            raise ValueError(
                f"body-level sum mismatch for {source_key}->{target_key}: "
                f"weights={observed} explorer={wanted}"
            )
        results.append(
            {
                "source_population": source_key,
                "target_population": target_key,
                "observed_connections": observed,
                "expected_connections": wanted,
                "exact_match": True,
            }
        )
    return results


def build_r002_graph(
    authority_path: str | Path,
    annotations_path: str | Path,
    neurotransmitters_path: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
) -> dict:
    authority_path = Path(authority_path)
    annotations_path = Path(annotations_path)
    nt_path = Path(neurotransmitters_path)
    weights_path = Path(weights_path)
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"refusing to overwrite R002 graph directory: {output}")

    authority = json.loads(authority_path.read_text())
    if authority.get("dataset") != "male-cns:v1.0":
        raise ValueError("R002 graph extraction requires male-cns:v1.0 authority")
    selected = _selected_ids(authority)

    expected_annotations_sha = authority.get("source_hashes", {}).get("annotations_sha256")
    observed_annotations_sha = sha256_file(annotations_path)
    if expected_annotations_sha and observed_annotations_sha != expected_annotations_sha:
        raise ValueError(
            "annotation hash mismatch: "
            f"authority={expected_annotations_sha} observed={observed_annotations_sha}"
        )

    annotations = pd.read_feather(annotations_path)
    nodes = annotations[annotations["bodyId"].astype(int).isin(selected)].copy()
    if len(nodes) != len(selected):
        observed = set(nodes["bodyId"].astype(int))
        raise ValueError(f"annotations missing selected bodies: {sorted(selected - observed)[:10]}")
    nodes = nodes.sort_values("bodyId").reset_index(drop=True)

    edges = _stream_selected_edges(weights_path, selected)
    if edges.empty:
        raise ValueError("R002 extraction produced no body-level edges")
    edges["source"] = edges["source"].astype(int)
    edges["target"] = edges["target"].astype(int)
    edges["weight"] = edges["weight"].astype(int)

    nt_names, nt_signs = _transmitter_signs(nt_path, selected)
    edges["sign"] = edges["source"].map(nt_signs).astype(int)
    nodes["consensus_nt"] = nodes["bodyId"].astype(int).map(nt_names)
    nodes["modeled_fast_sign"] = nodes["bodyId"].astype(int).map(nt_signs).astype(int)

    direct_validation = _validate_structural_sums(edges, authority)
    populations = {key: [int(x) for x in value] for key, value in authority["populations"].items()}
    roles = {
        "loom_size_left": populations["LPLC2_L"],
        "loom_size_right": populations["LPLC2_R"],
        "loom_velocity_left": populations["LC4_L"],
        "loom_velocity_right": populations["LC4_R"],
        "escape_left": populations["DNp01_L"],
        "escape_right": populations["DNp01_R"],
    }
    manifest = {
        "schema": "fly-sniff-r002-graph-v1",
        "dataset": "male-cns:v1.0",
        "qualification_status": "candidate",
        "scientific_claim_allowed": False,
        "execution_mode": "open-loop-escape-readout",
        "authority_sha256": sha256_file(authority_path),
        "sources": {
            "annotations": {"url": ANNOTATIONS_URL, "sha256": observed_annotations_sha},
            "neurotransmitters": {"url": NEUROTRANSMITTERS_URL, "sha256": sha256_file(nt_path)},
            "weights": {"url": CONNECTOME_WEIGHTS_URL, "sha256": sha256_file(weights_path)},
        },
        "selected_body_count": len(selected),
        "selected_edge_count": len(edges),
        "direct_edge_validation": direct_validation,
        "sign_policy": {
            "acetylcholine": 1,
            "gaba": -1,
            "glutamate": -1,
            "histamine": -1,
            "other_or_missing": 0,
            "status": "modeled-fast-receptor-effect-assumption",
        },
        "behavioral_readout_status": (
            "DNp01/GF is retained as lateral escape activity only; it is not interpreted "
            "as a steering command in this candidate graph."
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    nodes.to_parquet(output / "nodes.parquet", index=False)
    edges.sort_values(["source", "target"]).to_parquet(output / "edges.parquet", index=False)
    (output / "roles.json").write_text(json.dumps(roles, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    bundle = GraphBundle.load(output)
    bundle.validate(require_sign=True, require_qualified=False)
    return manifest


def _download_raw(cache_dir: Path) -> tuple[Path, Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    annotations = cache_dir / "body-annotations-male-cns-v1.0.feather"
    nt = cache_dir / "body-neurotransmitters-male-cns-v1.0.feather"
    weights = cache_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    for url, path in (
        (ANNOTATIONS_URL, annotations),
        (NEUROTRANSMITTERS_URL, nt),
        (CONNECTOME_WEIGHTS_URL, weights),
    ):
        if not path.exists():
            download_file(url, path)
    return annotations, nt, weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract bounded body-level R002 MaleCNS graph")
    parser.add_argument("--authority", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--neurotransmitters")
    parser.add_argument("--weights")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache-dir", default="data/raw/r002")
    parser.add_argument("--output", default="artifacts/r002/graph")
    args = parser.parse_args()

    if args.download:
        annotations, nt, weights = _download_raw(Path(args.cache_dir))
    else:
        if not args.annotations or not args.neurotransmitters or not args.weights:
            parser.error("provide all raw tables or pass --download")
        annotations = Path(args.annotations)
        nt = Path(args.neurotransmitters)
        weights = Path(args.weights)

    manifest = build_r002_graph(
        args.authority,
        annotations,
        nt,
        weights,
        args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
