from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .freeze import canonical_sha256

_ALLOWED_SELECTION_BASIS = {
    "raw_trace_review",
    "waveform_stereotypy",
    "refractory_violations",
    "blockwise_stability",
    "bilateral_artifact_coincidence",
}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _validated_qc(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "fly-sniff-dna02-threshold-adjudication-qc-v1":
        raise ValueError("unexpected threshold-QC schema")
    observed = str(payload.get("qc_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("qc_sha256", None)
    if observed != canonical_sha256(unhashed):
        raise ValueError("threshold-QC hash mismatch")
    required_false = (
        "behavior_fields_loaded",
        "yaw_loaded",
        "navigation_performance_used",
        "figure3c_statistic_computed",
        "thresholds_frozen",
        "automatic_threshold_selection",
    )
    for field in required_false:
        if payload.get(field) is not False:
            raise ValueError(f"threshold QC violates frozen false flag: {field}")
    return payload


def _qc_channel_map(qc: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    mapped: dict[tuple[str, str], dict[str, Any]] = {}
    for file_item in qc.get("files", []):
        alias = str(file_item["fly_alias"])
        for channel in file_item.get("channels", []):
            key = (alias, str(channel["soma_side"]))
            if key in mapped:
                raise ValueError(f"duplicate QC channel: {key}")
            mapped[key] = {
                **channel,
                "filename": str(file_item["filename"]),
                "source_sha256": str(file_item["source_sha256"]),
            }
    if len(mapped) != 8:
        raise ValueError("threshold QC must contain exactly eight fly/side channels")
    return mapped


def _selected_candidate(
    qc_channel: dict[str, Any], *, quantile: float, threshold: float
) -> dict[str, Any]:
    matches = [
        item
        for item in qc_channel["candidate_qc"]
        if np.isclose(
            float(item["prominence_quantile"]),
            quantile,
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(float(item["threshold"]), threshold, rtol=0.0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selected q/threshold pair is not an exact audited candidate: "
            f"q={quantile:g}, threshold={threshold:.17g}"
        )
    return matches[0]


def freeze_manifest(
    *,
    qc_path: str | Path,
    decisions_path: str | Path,
    output_path: str | Path,
    code_ref: str,
) -> dict[str, Any]:
    if not _SHA40.fullmatch(code_ref):
        raise ValueError("code_ref must be an exact lowercase 40-character git commit SHA")

    qc = _validated_qc(Path(qc_path))
    qc_channels = _qc_channel_map(qc)
    decisions = json.loads(Path(decisions_path).read_text())
    if decisions.get("schema") != "fly-sniff-dna02-threshold-decisions-v1":
        raise ValueError("unexpected threshold-decisions schema")
    if str(decisions.get("qc_sha256")) != str(qc["qc_sha256"]):
        raise ValueError("threshold decisions are not bound to this QC receipt")
    if str(decisions.get("prominence_audit_sha256")) != str(qc["prominence_audit_sha256"]):
        raise ValueError("threshold decisions reference a different prominence audit")
    if decisions.get("adjudication_method") != "manual_ephys_only_review":
        raise ValueError("v1 supports only manual_ephys_only_review")
    reviewer = str(decisions.get("reviewer", "")).strip()
    if not reviewer:
        raise ValueError("threshold decisions require a non-empty reviewer")
    for field in (
        "behavior_fields_reviewed",
        "yaw_reviewed",
        "navigation_performance_used",
        "figure3c_statistic_reviewed",
    ):
        if decisions.get(field) is not False:
            raise ValueError(f"threshold decisions violate frozen false flag: {field}")

    entries = decisions.get("decisions")
    if not isinstance(entries, list) or len(entries) != 8:
        raise ValueError("threshold decisions must contain exactly eight entries")

    seen: set[tuple[str, str]] = set()
    frozen: list[dict[str, Any]] = []
    for entry in entries:
        alias = str(entry["fly_alias"])
        side = str(entry["soma_side"])
        key = (alias, side)
        if key in seen:
            raise ValueError(f"duplicate threshold decision: {key}")
        seen.add(key)
        if key not in qc_channels:
            raise ValueError(f"threshold decision is not present in QC: {key}")
        qc_channel = qc_channels[key]

        filename = str(entry["filename"])
        source_sha256 = str(entry["source_sha256"])
        source_field = str(entry["source_field"])
        if filename != qc_channel["filename"]:
            raise ValueError(f"filename mismatch for {key}")
        if source_sha256 != qc_channel["source_sha256"]:
            raise ValueError(f"source SHA-256 mismatch for {key}")
        if source_field != str(qc_channel["source_field"]):
            raise ValueError(f"source field mismatch for {key}")

        quantile_value = entry.get("selected_prominence_quantile")
        threshold_value = entry.get("selected_threshold")
        if quantile_value is None or threshold_value is None:
            raise ValueError(f"threshold decision is incomplete for {key}")
        quantile = float(quantile_value)
        threshold = float(threshold_value)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"selected threshold must be positive for {key}")
        selected_qc = _selected_candidate(
            qc_channel,
            quantile=quantile,
            threshold=threshold,
        )

        basis = entry.get("selection_basis")
        if not isinstance(basis, list) or not basis:
            raise ValueError(f"selection_basis must be non-empty for {key}")
        normalized_basis = [str(value) for value in basis]
        if len(normalized_basis) != len(set(normalized_basis)):
            raise ValueError(f"selection_basis contains duplicates for {key}")
        invalid_basis = set(normalized_basis) - _ALLOWED_SELECTION_BASIS
        if invalid_basis:
            raise ValueError(
                f"selection_basis contains forbidden values for {key}: {invalid_basis}"
            )
        rationale = str(entry.get("rationale", "")).strip()
        if len(rationale) < 20:
            raise ValueError(f"rationale must be at least 20 characters for {key}")

        frozen.append(
            {
                "fly_alias": alias,
                "filename": filename,
                "source_sha256": source_sha256,
                "source_field": source_field,
                "soma_side": side,
                "selected_prominence_quantile": quantile,
                "selected_threshold": threshold,
                "selection_basis": normalized_basis,
                "rationale": rationale,
                "selected_candidate_qc": {
                    "event_count": selected_qc["event_count"],
                    "event_rate_hz_descriptive_only": selected_qc["event_rate_hz"],
                    "refractory_violation_fraction_lt_1ms": selected_qc[
                        "refractory_violation_fraction_lt_1ms"
                    ],
                    "short_isi_fraction_lt_2ms": selected_qc["short_isi_fraction_lt_2ms"],
                    "block_rate_cv": selected_qc["block_rate_cv"],
                    "waveform_median_template_correlation": selected_qc["waveform"][
                        "median_template_correlation"
                    ],
                    "waveform_fwhm_ms": selected_qc["waveform"]["fwhm_ms"],
                },
            }
        )

    if seen != set(qc_channels):
        missing = sorted(set(qc_channels) - seen)
        raise ValueError(f"threshold decisions are missing QC channels: {missing}")

    decisions_for_hash = dict(decisions)
    decisions_sha256 = canonical_sha256(decisions_for_hash)
    payload: dict[str, Any] = {
        "schema": "fly-sniff-dna02-threshold-manifest-v1",
        "status": "THRESHOLDS_FROZEN_BEFORE_BEHAVIOR",
        "code_ref": code_ref,
        "source_contract_sha256": qc["source_contract_sha256"],
        "source_byte_evidence_sha256": qc["source_byte_evidence_sha256"],
        "prominence_audit_sha256": qc["prominence_audit_sha256"],
        "threshold_qc_sha256": qc["qc_sha256"],
        "threshold_decisions_sha256": decisions_sha256,
        "reviewer": reviewer,
        "adjudication_method": decisions["adjudication_method"],
        "behavior_fields_reviewed": False,
        "yaw_reviewed": False,
        "navigation_performance_used": False,
        "figure3c_statistic_reviewed": False,
        "thresholds_frozen": True,
        "threshold_count": len(frozen),
        "thresholds": sorted(
            frozen,
            key=lambda item: (item["fly_alias"], item["soma_side"]),
        ),
        "allowed_next_action": (
            "Run the independently frozen Figure 3C reproduction pipeline. Thresholds in this "
            "manifest may not be changed after behavior is opened for v1."
        ),
        "forbidden_actions": [
            "retune thresholds after inspecting yaw or Figure 3C fit",
            "retune thresholds using odor-navigation performance",
            "drop a Figure 3C fly because its downstream relationship is inconvenient",
        ],
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze exactly eight DNa02 prominence thresholds after ephys-only human review"
        )
    )
    parser.add_argument("--qc", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--code-ref", required=True)
    args = parser.parse_args(argv)
    result = freeze_manifest(
        qc_path=args.qc,
        decisions_path=args.decisions,
        output_path=args.out,
        code_ref=args.code_ref,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
