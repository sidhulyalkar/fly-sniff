from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_EXPLORER_URL = (
    "https://raw.githubusercontent.com/reiserlab/"
    "celltype-explorer-drosophila-male-cns/main/data/neurons.json"
)
DEFAULT_EXPLORER_REPO = "reiserlab/celltype-explorer-drosophila-male-cns"
DEFAULT_EXPLORER_BLOB = "9f239760f25fd35bd00755bb97fac614f4b54b1a"

BODY_ID_KEYS = (
    "bodyId",
    "body_id",
    "bodyid",
    "root_id",
    "rootId",
    "rootid",
    "segment_id",
    "segmentId",
)
TYPE_KEYS = ("type", "cell_type", "cellType", "celltype")
INSTANCE_KEYS = ("instance", "name", "instance_name", "instanceName", "cell_name", "cellName")
SIDE_KEYS = ("side", "hemisphere", "somaSide", "soma_side")
NT_KEYS = (
    "predicted_nt",
    "predictedNeurotransmitter",
    "predicted_neurotransmitter",
    "neurotransmitter",
    "top_nt",
    "topNt",
    "nt",
)
NT_CONFIDENCE_KEYS = (
    "predicted_nt_confidence",
    "predictedNeurotransmitterConfidence",
    "predicted_neurotransmitter_confidence",
    "neurotransmitter_confidence",
    "top_nt_confidence",
    "topNtConfidence",
    "nt_confidence",
)


@dataclass(frozen=True)
class ExplorerAuthority:
    source: str
    sha256: str
    upstream_repository: str | None = None
    upstream_blob_sha: str | None = None


def _read_bytes(source: str) -> bytes:
    if source.startswith(("https://", "http://")):
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "fly-sniff/0.1 MaleCNS authority importer"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    return Path(source).read_bytes()


def _first_value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _as_body_id(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if not re.fullmatch(r"[0-9]+", value):
                return None
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if numeric >= 0 else None


def _iter_candidate_dicts(value: Any, *, map_key: str | None = None) -> Iterable[dict[str, Any]]:
    """Yield record-like dicts with explicit body IDs or numeric body-ID map keys.

    Generic metadata fields such as ``id`` are intentionally *not* accepted as
    body-level authority. If a JSON object is keyed by numeric body IDs and its
    values are metadata dicts, the numeric map key is injected as
    ``__map_body_id``. The raw source object is otherwise left untouched.
    """
    if isinstance(value, list):
        for item in value:
            yield from _iter_candidate_dicts(item)
        return
    if not isinstance(value, dict):
        return

    explicit_id = _first_value(value, BODY_ID_KEYS)
    injected_id = _as_body_id(map_key)
    if _as_body_id(explicit_id) is not None or injected_id is not None:
        row = dict(value)
        if injected_id is not None and _as_body_id(explicit_id) is None:
            row["__map_body_id"] = injected_id
        yield row
        return

    for key, child in value.items():
        if isinstance(child, (dict, list)):
            yield from _iter_candidate_dicts(child, map_key=str(key))


def _normalize_side(value: Any, *, type_name: str | None, instance: str | None) -> str | None:
    if value not in (None, ""):
        token = str(value).strip().lower()
        if token in {"l", "left", "lhs"}:
            return "L"
        if token in {"r", "right", "rhs"}:
            return "R"
        if token in {"m", "midline", "center", "central"}:
            return "M"
    for text in (instance, type_name):
        if not text:
            continue
        match = re.search(r"(?:^|[_\- ])([LR])$", str(text).strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def normalize_explorer_document(document: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in _iter_candidate_dicts(document):
        explicit = _first_value(raw, BODY_ID_KEYS)
        body_id = _as_body_id(explicit)
        id_source = next(
            (key for key in BODY_ID_KEYS if key in raw and _as_body_id(raw[key]) is not None),
            None,
        )
        if body_id is None:
            body_id = _as_body_id(raw.get("__map_body_id"))
            id_source = "json-map-key" if body_id is not None else None
        if body_id is None:
            continue
        type_name = _first_value(raw, TYPE_KEYS)
        instance = _first_value(raw, INSTANCE_KEYS)
        side_raw = _first_value(raw, SIDE_KEYS)
        nt = _first_value(raw, NT_KEYS)
        nt_conf = _as_float(_first_value(raw, NT_CONFIDENCE_KEYS))
        rows.append(
            {
                "bodyId": body_id,
                "type": None if type_name is None else str(type_name),
                "instance": None if instance is None else str(instance),
                "side": _normalize_side(side_raw, type_name=type_name, instance=instance),
                "predicted_nt": None if nt is None else str(nt),
                "predicted_nt_confidence": nt_conf,
                "body_id_source_field": id_source,
            }
        )

    if not rows:
        raise ValueError(
            "Could not find any record with a numeric unique body-ID field. "
            f"Recognized fields: {', '.join(BODY_ID_KEYS)} or numeric map keys."
        )

    frame = pd.DataFrame(rows)
    duplicated = frame.bodyId[frame.bodyId.duplicated(keep=False)]
    if not duplicated.empty:
        sample = sorted({int(x) for x in duplicated.head(10)})
        raise ValueError(f"Explorer document produced duplicate body IDs; sample={sample}")
    return frame.sort_values("bodyId", kind="stable").reset_index(drop=True)


def discover_candidates(frame: pd.DataFrame, pattern: str) -> pd.DataFrame:
    try:
        regex = re.compile(pattern, flags=re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid discovery regex {pattern!r}: {exc}") from exc
    searchable = (
        frame["type"].fillna("").astype(str)
        + "\t"
        + frame["instance"].fillna("").astype(str)
    )
    mask = searchable.map(lambda text: bool(regex.search(text)))
    return frame.loc[mask].copy().reset_index(drop=True)


def import_explorer(
    source: str,
    *,
    pattern: str,
    upstream_repository: str | None = None,
    upstream_blob_sha: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _read_bytes(source)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Explorer source is not valid JSON: {source}: {exc}") from exc
    all_neurons = normalize_explorer_document(document)
    selected = discover_candidates(all_neurons, pattern)
    authority = ExplorerAuthority(
        source=source,
        sha256=digest,
        upstream_repository=upstream_repository,
        upstream_blob_sha=upstream_blob_sha,
    )
    manifest = {
        "authority_kind": "malecns-explorer-neuron-candidates-v0",
        "dataset": "male-cns:v1.0",
        "qualification_status": "candidate",
        "source": authority.source,
        "source_sha256": authority.sha256,
        "upstream_repository": authority.upstream_repository,
        "upstream_blob_sha": authority.upstream_blob_sha,
        "selection_regex": pattern,
        "normalized_neuron_count": len(all_neurons),
        "selected_neuron_count": len(selected),
        "selected_body_ids": [int(x) for x in selected.bodyId.tolist()],
        "schema": {
            "body_id_candidates": list(BODY_ID_KEYS),
            "type_candidates": list(TYPE_KEYS),
            "instance_candidates": list(INSTANCE_KEYS),
            "side_candidates": list(SIDE_KEYS),
            "neurotransmitter_candidates": list(NT_KEYS),
            "neurotransmitter_confidence_candidates": list(NT_CONFIDENCE_KEYS),
        },
        "warning": (
            "This artifact identifies release-specific candidate neurons only. It does not prove "
            "functional roles, connectivity corridors, edge signs, or circuit qualification."
        ),
    }
    return selected, manifest


def write_candidate_artifact(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "candidates.parquet", index=False)
    frame.to_csv(out / "candidates.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize the upstream MaleCNS Cell Type Explorer neuron manifest "
            "and select exact v1.0 candidates"
        )
    )
    parser.add_argument("--source", default=DEFAULT_EXPLORER_URL)
    parser.add_argument("--patterns", required=True, help="Case-insensitive regex over type + instance")
    parser.add_argument("--output", default="data/cache/explorer-candidates-v0")
    parser.add_argument("--upstream-repository", default=DEFAULT_EXPLORER_REPO)
    parser.add_argument("--upstream-blob-sha", default=DEFAULT_EXPLORER_BLOB)
    args = parser.parse_args()
    selected, manifest = import_explorer(
        args.source,
        pattern=args.patterns,
        upstream_repository=args.upstream_repository,
        upstream_blob_sha=args.upstream_blob_sha,
    )
    out = write_candidate_artifact(selected, manifest, args.output)
    print(f"normalized={manifest['normalized_neuron_count']} selected={len(selected)} output={out}")
    if selected.empty:
        raise SystemExit(2)
