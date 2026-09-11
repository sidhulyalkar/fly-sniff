from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


DEFAULT_PATTERNS = {
    "navigation": ["hDeltaC.*", "PFN.*", "PFL2.*", "PFL3.*"],
    "descending": ["DNa02.*"],
    # Sensory names are intentionally broad discovery queries. The resulting IDs
    # must be reviewed and sealed into roles.json before benchmark use.
    "olfactory_discovery": [".*ORN.*", "Or.*", "Ir.*"],
}


def _client():
    try:
        from neuprint import Client
    except ImportError as exc:
        raise SystemExit("Install MaleCNS support with: pip install -e '.[malecns]'") from exc
    token = os.getenv("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit("NEUPRINT_TOKEN is required. Obtain a read token from neuprint.janelia.org.")
    server = os.getenv("NEUPRINT_SERVER", "https://neuprint.janelia.org")
    return Client(server, dataset="male-cns:v1.0", token=token)


def discover(patterns: dict[str, list[str]]) -> pd.DataFrame:
    from neuprint import NeuronCriteria, fetch_neurons

    client = _client()
    frames: list[pd.DataFrame] = []
    for family, regexes in patterns.items():
        for regex in regexes:
            neurons, _ = fetch_neurons(NeuronCriteria(type=regex, regex=True), client=client)
            if len(neurons):
                keep = [c for c in ["bodyId", "instance", "type", "pre", "post", "status"] if c in neurons]
                part = neurons[keep].copy()
                part["query_family"] = family
                part["query_regex"] = regex
                frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["bodyId", "instance", "type", "query_family", "query_regex"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["bodyId", "query_family"])


def extract_induced(ids: list[int], min_weight: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    from neuprint import fetch_adjacencies

    client = _client()
    neurons, conn = fetch_adjacencies(ids, ids, min_total_weight=min_weight, client=client)
    # neuprint-python commonly returns bodyId_pre/bodyId_post in the connection table.
    rename = {"bodyId_pre": "source", "bodyId_post": "target", "weight": "weight"}
    conn = conn.rename(columns=rename)
    required = ["source", "target", "weight"]
    missing = [c for c in required if c not in conn]
    if missing:
        raise RuntimeError(f"unexpected neuPrint adjacency columns; missing {missing}, got {list(conn.columns)}")
    return neurons, conn[required].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and extract MaleCNS v1.0 candidate circuits")
    parser.add_argument("--output", default="data/cache/malecns-candidate-v0")
    parser.add_argument("--patterns", help="JSON file overriding discovery regexes")
    parser.add_argument("--min-weight", type=int, default=2)
    parser.add_argument("--extract-induced", action="store_true", help="also fetch connectivity among discovered cells")
    args = parser.parse_args()

    patterns = DEFAULT_PATTERNS
    if args.patterns:
        patterns = json.loads(Path(args.patterns).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    found = discover(patterns)
    found.to_parquet(out / "discovery.parquet", index=False)
    (out / "discovery.json").write_text(found.to_json(orient="records", indent=2))
    print(f"discovered {len(found)} unique family assignments across {found.bodyId.nunique() if len(found) else 0} neurons")

    if args.extract_induced and len(found):
        nodes, edges = extract_induced(sorted(found.bodyId.astype(int).unique().tolist()), args.min_weight)
        nodes.to_parquet(out / "nodes.parquet", index=False)
        edges.to_parquet(out / "edges.parquet", index=False)
        print(f"wrote {len(nodes)} nodes and {len(edges)} induced edges")
        print("NOTE: this is a discovery graph, not a benchmark-qualified circuit. Review roles and pathway coverage first.")
