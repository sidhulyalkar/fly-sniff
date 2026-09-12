from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .graph import GraphBundle

DEFAULT_AUTHORITIES = (
    "authority/malecns-v1.0-steering-sign-evidence.json",
    "authority/malecns-v1.0-body-sign-overrides-v1.json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_transmitter(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    aliases = {
        "acetylcholine": "ACh",
        "ach": "ACh",
        "gaba": "GABA",
        "glutamate": "Glu",
        "glu": "Glu",
    }
    return aliases.get(text.lower(), text)


def _validate_confidence(value: Any) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"transmitter confidence must be in [0, 1], got {value!r}")
    return confidence


def _merge_record(
    table: dict[Any, dict[str, Any]],
    key: Any,
    record: dict[str, Any],
    *,
    label: str,
) -> None:
    prior = table.get(key)
    if prior is None:
        table[key] = record
        return
    comparable = ("predicted_neurotransmitter", "confidence")
    if any(prior.get(field) != record.get(field) for field in comparable):
        raise ValueError(f"conflicting {label} transmitter authority for {key!r}")


def load_authorities(paths: list[str | Path]) -> dict[str, Any]:
    type_records: dict[str, dict[str, Any]] = {}
    body_records: dict[int, dict[str, Any]] = {}
    authority_receipts: list[dict[str, Any]] = []
    sign_rules: list[dict[str, int]] = []

    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        authority_receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "authority_kind": payload.get("authority_kind") or payload.get("protocol"),
                "qualification_status": payload.get("qualification_status"),
            }
        )
        rule = payload.get("model_sign_rule")
        if isinstance(rule, dict):
            sign_rules.append(
                {
                    str(key): int(value)
                    for key, value in rule.items()
                    if key != "warning" and isinstance(value, (int, float))
                }
            )

        for cell_type, raw_record in payload.get("types", {}).items():
            record = dict(raw_record)
            record["predicted_neurotransmitter"] = _normalize_transmitter(
                record.get("predicted_neurotransmitter")
            )
            record["confidence"] = _validate_confidence(record.get("confidence"))
            record["authority_path"] = str(path)
            record["authority_sha256"] = sha256_file(path)
            record["evidence_level"] = "type_level_prediction_applied_to_body_by_annotation"
            _merge_record(type_records, str(cell_type), record, label="type-level")

        for raw_body_id, raw_record in payload.get("body_ids", {}).items():
            body_id = int(raw_body_id)
            record = dict(raw_record)
            record["predicted_neurotransmitter"] = _normalize_transmitter(
                record.get("predicted_neurotransmitter")
            )
            record["confidence"] = _validate_confidence(record.get("confidence"))
            record["authority_path"] = str(path)
            record["authority_sha256"] = sha256_file(path)
            record["evidence_level"] = "exact_body_id_authority"
            _merge_record(body_records, body_id, record, label="body-ID")

    canonical_rule = sign_rules[0] if sign_rules else {"ACh": 1, "GABA": -1, "Glu": 0}
    for rule in sign_rules[1:]:
        if rule != canonical_rule:
            raise ValueError("sign authority files disagree on the model sign convention")
    for sign in canonical_rule.values():
        if sign not in {-1, 0, 1}:
            raise ValueError("model sign authority must use only -1, 0, or +1")

    return {
        "type_records": type_records,
        "body_records": body_records,
        "authority_receipts": authority_receipts,
        "model_sign_rule": canonical_rule,
    }


def _sign_for_transmitter(transmitter: str | None, rule: dict[str, int]) -> int:
    if transmitter is None:
        return 0
    if transmitter in rule:
        return int(rule[transmitter])
    normalized = _normalize_transmitter(transmitter)
    if normalized in rule:
        return int(rule[normalized])
    return int(rule.get("other_or_unclear", 0))


def build_sign_authority_report(
    bundle: GraphBundle,
    authority_paths: list[str | Path],
) -> dict[str, Any]:
    bundle.validate(require_sign=True, require_qualified=False)
    if "type" not in bundle.nodes.columns:
        raise ValueError("sign authority audit requires bundle nodes to include a type column")

    authorities = load_authorities(authority_paths)
    node_rows = bundle.nodes.drop_duplicates("bodyId").set_index("bodyId")
    source_ids = sorted(set(bundle.edges.source.astype(int)))
    source_records: list[dict[str, Any]] = []
    source_sign: dict[int, int] = {}

    for body_id in source_ids:
        if body_id not in node_rows.index:
            raise ValueError(f"presynaptic source body ID {body_id} is absent from nodes.parquet")
        row = node_rows.loc[body_id]
        cell_type = str(row.get("type", ""))
        exact = authorities["body_records"].get(body_id)
        type_level = authorities["type_records"].get(cell_type)
        evidence = exact or type_level
        if evidence is None:
            transmitter = None
            confidence = None
            evidence_level = "unknown"
            authority_path = None
            authority_sha256 = None
            source_reference = None
        else:
            transmitter = evidence.get("predicted_neurotransmitter")
            confidence = evidence.get("confidence")
            evidence_level = evidence["evidence_level"]
            authority_path = evidence.get("authority_path")
            authority_sha256 = evidence.get("authority_sha256")
            source_reference = evidence.get("source")
        model_sign = _sign_for_transmitter(transmitter, authorities["model_sign_rule"])
        source_sign[body_id] = model_sign
        source_records.append(
            {
                "source_body_id": body_id,
                "type": cell_type or None,
                "transmitter": transmitter,
                "confidence": confidence,
                "evidence_level": evidence_level,
                "authority_path": authority_path,
                "authority_sha256": authority_sha256,
                "source_reference": source_reference,
                "model_sign": model_sign,
                "sign_status": "resolved" if model_sign != 0 else "unresolved_zero",
            }
        )

    mismatches: list[dict[str, Any]] = []
    signed_edges = 0
    signed_weight = 0.0
    total_weight = 0.0
    for row in bundle.edges[["source", "target", "weight", "sign"]].itertuples(index=False):
        source = int(row.source)
        target = int(row.target)
        expected = int(source_sign[source])
        observed = int(row.sign)
        weight = float(row.weight)
        total_weight += weight
        if expected != 0:
            signed_edges += 1
            signed_weight += weight
        if observed != expected:
            mismatches.append(
                {
                    "source_body_id": source,
                    "target_body_id": target,
                    "observed_sign": observed,
                    "authority_sign": expected,
                }
            )

    edge_count = len(bundle.edges)
    source_signed = sum(record["model_sign"] != 0 for record in source_records)
    coverage = {
        "source_body_count": len(source_records),
        "signed_source_body_count": source_signed,
        "unresolved_source_body_count": len(source_records) - source_signed,
        "edge_count": edge_count,
        "signed_edge_count": signed_edges,
        "unresolved_edge_count": edge_count - signed_edges,
        "signed_edge_fraction": float(signed_edges / edge_count) if edge_count else 0.0,
        "signed_structural_weight_fraction": (
            float(signed_weight / total_weight) if total_weight > 0.0 else 0.0
        ),
        "edge_sign_mismatch_count": len(mismatches),
    }
    report = {
        "protocol": "source-body-transmitter-sign-authority-v1",
        "dataset": (bundle.manifest or {}).get("dataset", "unspecified"),
        "graph_sha256": bundle.replay_fingerprint(),
        "model_sign_rule": authorities["model_sign_rule"],
        "authority_files": authorities["authority_receipts"],
        "source_records": source_records,
        "coverage": coverage,
        "edge_sign_mismatches": mismatches,
        "passed": len(mismatches) == 0,
        "claim_boundary": (
            "A nonzero sign is a conservative presynaptic transmitter-derived modeling convention. "
            "It is not a measurement of receptor-specific postsynaptic effect. Unknown evidence "
            "remains transmitter=null, confidence=null, sign=0."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def write_sign_authority_report(
    bundle_dir: str | Path,
    authority_paths: list[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    bundle = GraphBundle.load(bundle_dir)
    report = build_sign_authority_report(bundle, authority_paths)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_sign_authority.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output / "edge_sign_coverage.json").write_text(
        json.dumps(report["coverage"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit every sealed GraphBundle source body against transmitter/sign authority"
    )
    parser.add_argument("bundle", help="sealed signed GraphBundle directory")
    parser.add_argument(
        "--authority",
        action="append",
        default=None,
        help="authority JSON; repeat for multiple files",
    )
    parser.add_argument("--output", default="results/candidate/sign-authority-v1")
    args = parser.parse_args()
    paths = args.authority or list(DEFAULT_AUTHORITIES)
    report = write_sign_authority_report(args.bundle, paths, args.output)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "coverage": report["coverage"],
                "report_sha256": report["report_sha256"],
                "output": str(Path(args.output)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
