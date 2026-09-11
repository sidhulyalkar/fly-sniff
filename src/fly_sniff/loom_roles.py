from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from .explorer_data import (
    DEFAULT_EXPLORER_BLOB,
    DEFAULT_EXPLORER_REPO,
    DEFAULT_EXPLORER_URL,
    normalize_explorer_document,
)

PRIMARY_INPUT_TYPE = "LPLC2"
SUPPORT_INPUT_TYPE = "LC4"
PRIMARY_OUTPUT_TYPE = "DNp06"
OPTIONAL_OUTPUT_TYPES = ("DNp01", "DNp03", "DNp04")


def _read_source(source: str) -> bytes:
    if source.startswith(("https://", "http://")):
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "fly-sniff/0.1 looming-role-authority"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    return Path(source).read_bytes()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def _exact_type(frame: pd.DataFrame, type_name: str) -> pd.DataFrame:
    values = frame["type"].fillna("").astype(str).str.casefold()
    return frame.loc[values == type_name.casefold()].copy().reset_index(drop=True)


def _split_bilateral(frame: pd.DataFrame, type_name: str, *, required: bool) -> dict[str, list[int]]:
    population = _exact_type(frame, type_name)
    if population.empty:
        if required:
            raise ValueError(f"required MaleCNS type {type_name!r} is absent")
        return {"L": [], "R": []}

    unresolved = population.loc[~population["side"].isin(["L", "R"])]
    if not unresolved.empty:
        sample = unresolved.bodyId.astype(int).head(8).tolist()
        raise ValueError(
            f"MaleCNS type {type_name!r} has neurons without resolved L/R side; sample={sample}"
        )

    result = {
        side: sorted(population.loc[population.side == side, "bodyId"].astype(int).tolist())
        for side in ("L", "R")
    }
    if required and (not result["L"] or not result["R"]):
        raise ValueError(
            f"required MaleCNS type {type_name!r} must have candidates on both sides; "
            f"L={len(result['L'])} R={len(result['R'])}"
        )
    return result


def derive_loom_roles(frame: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Derive candidate role populations from exact MaleCNS type labels.

    This function establishes type membership and laterality only. It does not
    establish exact edges, edge signs, physiology, response selectivity, or the
    behavioral meaning of a hemisphere-specific descending-neuron readout.
    """

    lplc2 = _split_bilateral(frame, PRIMARY_INPUT_TYPE, required=True)
    dnp06 = _split_bilateral(frame, PRIMARY_OUTPUT_TYPE, required=True)
    lc4 = _split_bilateral(frame, SUPPORT_INPUT_TYPE, required=False)
    optional_outputs = {
        type_name: _split_bilateral(frame, type_name, required=False)
        for type_name in OPTIONAL_OUTPUT_TYPES
    }

    roles: dict[str, list[int]] = {
        "loom_left": lplc2["L"],
        "loom_right": lplc2["R"],
        "escape_left": dnp06["L"],
        "escape_right": dnp06["R"],
    }
    if lc4["L"]:
        roles["loom_lc4_left"] = lc4["L"]
    if lc4["R"]:
        roles["loom_lc4_right"] = lc4["R"]
    for type_name, split in optional_outputs.items():
        slug = type_name.casefold()
        if split["L"]:
            roles[f"escape_{slug}_left"] = split["L"]
        if split["R"]:
            roles[f"escape_{slug}_right"] = split["R"]

    all_role_ids = [body_id for body_ids in roles.values() for body_id in body_ids]
    if len(all_role_ids) != len(set(all_role_ids)):
        raise ValueError("looming role derivation produced overlapping body IDs across roles")

    populations = {
        PRIMARY_INPUT_TYPE: {"L": lplc2["L"], "R": lplc2["R"]},
        SUPPORT_INPUT_TYPE: {"L": lc4["L"], "R": lc4["R"]},
        PRIMARY_OUTPUT_TYPE: {"L": dnp06["L"], "R": dnp06["R"]},
        **{
            type_name: {"L": split["L"], "R": split["R"]}
            for type_name, split in optional_outputs.items()
        },
    }
    return roles, populations


def build_candidate_manifest(
    raw: bytes,
    *,
    source: str,
    upstream_repository: str,
    expected_blob_sha: str,
) -> dict[str, Any]:
    observed_blob_sha = git_blob_sha1(raw)
    if observed_blob_sha != expected_blob_sha:
        raise ValueError(
            "MaleCNS explorer source drifted from pinned Git blob: "
            f"expected={expected_blob_sha} observed={observed_blob_sha}"
        )
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MaleCNS explorer source is invalid JSON: {source}: {exc}") from exc

    frame = normalize_explorer_document(document)
    roles, populations = derive_loom_roles(frame)
    return {
        "authority_kind": "malecns-looming-role-candidates-v0",
        "dataset": "male-cns:v1.0",
        "qualification_status": "candidate",
        "scientific_claim_allowed": False,
        "source": source,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "upstream_repository": upstream_repository,
        "expected_git_blob_sha1": expected_blob_sha,
        "observed_git_blob_sha1": observed_blob_sha,
        "git_blob_verified": True,
        "normalized_neuron_count": int(len(frame)),
        "roles": roles,
        "populations": populations,
        "edge_authority": "pending",
        "sign_authority": "pending",
        "sensory_boundary": (
            "abstract positive angular-expansion drive injected at candidate LPLC2 populations"
        ),
        "output_boundary": (
            "candidate DNp06 population activity; no biological turn-direction decoder asserted"
        ),
        "warning": (
            "Pinned type membership and laterality are candidate structural authority only. "
            "Exact connectivity, edge signs, neural dynamics, looming physiology, and motor "
            "semantics must be qualified separately before a MaleCNS behavioral claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a pinned candidate MaleCNS LPLC2/LC4 -> descending escape role manifest"
    )
    parser.add_argument("--source", default=DEFAULT_EXPLORER_URL)
    parser.add_argument("--upstream-repository", default=DEFAULT_EXPLORER_REPO)
    parser.add_argument("--expected-blob-sha", default=DEFAULT_EXPLORER_BLOB)
    parser.add_argument("--output", default="artifacts/loom-roles-candidate.json")
    args = parser.parse_args()

    raw = _read_source(args.source)
    manifest = build_candidate_manifest(
        raw,
        source=args.source,
        upstream_repository=args.upstream_repository,
        expected_blob_sha=args.expected_blob_sha,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    counts = {role: len(body_ids) for role, body_ids in manifest["roles"].items()}
    print(json.dumps({"output": str(output), "role_counts": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
