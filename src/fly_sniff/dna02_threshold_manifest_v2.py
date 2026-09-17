from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .freeze import canonical_sha256

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_BASIS = {
    "raw_trace_review",
    "waveform_stereotypy",
    "refractory_violations",
    "blockwise_stability",
    "bilateral_artifact_coincidence",
    "individual_waveform_distribution",
}
_Q_ORDER = (0.99, 0.995, 0.999, 0.9995, 0.9999)
_DISTRIBUTION_Q = {0.995, 0.999, 0.9995}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _validated_hashed_json(path: Path, *, schema: str, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected schema for {path}: {payload.get('schema')!r}")
    observed = str(payload.get(hash_field, ""))
    unhashed = dict(payload)
    unhashed.pop(hash_field, None)
    if observed != canonical_sha256(unhashed):
        raise ValueError(f"{hash_field} mismatch for {path}")
    return payload


def _require_false(payload: dict[str, Any], fields: tuple[str, ...], *, label: str) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise ValueError(f"{label} violates frozen false flag: {field}")


def _channel_map(payload: dict[str, Any], *, label: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for file_item in payload.get("files", []):
        alias = str(file_item["fly_alias"])
        for channel in file_item.get("channels", []):
            key = (alias, str(channel["soma_side"]))
            if key in result:
                raise ValueError(f"duplicate {label} channel: {key}")
            result[key] = {
                **channel,
                "filename": str(file_item["filename"]),
                "source_sha256": str(file_item["source_sha256"]),
            }
    if len(result) != 8:
        raise ValueError(f"{label} must contain exactly eight fly/side channels")
    return result


def _candidate(channel: dict[str, Any], quantile: float) -> dict[str, Any]:
    try:
        return channel["candidates"][str(float(quantile))]
    except KeyError as exc:
        raise ValueError(f"freeze evidence lacks q={quantile:g}") from exc


def _candidate_evidence(candidate: dict[str, Any]) -> dict[str, float]:
    return {"threshold": float(candidate["threshold"])}


def _compact_distribution(channel: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_quantile": float(channel["anchor_quantile"]),
        "bands": [dict(band) for band in channel["bands"]],
    }


def _profile_entry(channel: dict[str, Any], quantile: float) -> dict[str, float]:
    return {
        "prominence_quantile": float(quantile),
        "threshold": float(_candidate(channel, quantile)["threshold"]),
    }


def _sensitivity_profiles(
    selected: dict[tuple[str, str], float],
    channels: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {
        "lower_one_step": {},
        "primary": {},
        "higher_one_step": {},
    }
    for key in sorted(selected):
        quantile = selected[key]
        index = _Q_ORDER.index(quantile)
        label = f"{key[0]}:{key[1]}"
        result["lower_one_step"][label] = _profile_entry(
            channels[key], _Q_ORDER[max(0, index - 1)]
        )
        result["primary"][label] = _profile_entry(channels[key], quantile)
        result["higher_one_step"][label] = _profile_entry(
            channels[key], _Q_ORDER[min(len(_Q_ORDER) - 1, index + 1)]
        )
    return result


def freeze_manifest_v2(
    *,
    freeze_evidence_path: str | Path,
    distribution_evidence_path: str | Path,
    decisions_path: str | Path,
    output_path: str | Path,
    code_ref: str,
) -> dict[str, Any]:
    if not _SHA40.fullmatch(code_ref):
        raise ValueError("code_ref must be an exact lowercase 40-character git commit SHA")

    evidence = _validated_hashed_json(
        Path(freeze_evidence_path),
        schema="fly-sniff-dna02-threshold-freeze-evidence-v2",
        hash_field="evidence_sha256",
    )
    distribution = _validated_hashed_json(
        Path(distribution_evidence_path),
        schema="fly-sniff-dna02-threshold-distribution-freeze-evidence-v2",
        hash_field="evidence_sha256",
    )
    decisions = _validated_hashed_json(
        Path(decisions_path),
        schema="fly-sniff-dna02-threshold-decisions-v2",
        hash_field="decisions_sha256",
    )

    _require_false(
        evidence,
        (
            "behavior_fields_loaded",
            "yaw_loaded",
            "navigation_performance_used",
            "figure3c_statistic_computed",
            "thresholds_frozen",
            "automatic_threshold_selection",
            "event_rate_used_for_selection",
        ),
        label="freeze evidence",
    )
    _require_false(
        distribution,
        (
            "behavior_fields_loaded",
            "yaw_loaded",
            "navigation_performance_used",
            "figure3c_statistic_computed",
            "thresholds_frozen",
            "automatic_threshold_selection",
        ),
        label="distribution freeze evidence",
    )
    _require_false(
        decisions,
        (
            "behavior_fields_reviewed",
            "yaw_reviewed",
            "navigation_performance_used",
            "figure3c_statistic_reviewed",
            "event_rate_used_for_selection",
        ),
        label="threshold decisions",
    )

    if distribution.get("prominence_audit_sha256") != evidence.get("prominence_audit_sha256"):
        raise ValueError("distribution and threshold freeze evidence reference different audits")
    if decisions.get("prominence_audit_sha256") != evidence.get("prominence_audit_sha256"):
        raise ValueError("decisions reference a different prominence audit")
    if decisions.get("threshold_qc_sha256") != evidence.get("parent_threshold_qc_sha256"):
        raise ValueError("decisions reference a different threshold QC")
    if decisions.get("distribution_review_sha256") != distribution.get(
        "parent_distribution_review_sha256"
    ):
        raise ValueError("decisions reference a different distribution review")

    channels = _channel_map(evidence, label="freeze evidence")
    distribution_channels = _channel_map(distribution, label="distribution evidence")
    if set(channels) != set(distribution_channels):
        raise ValueError("threshold and distribution evidence channel sets differ")
    for key in channels:
        for field in ("filename", "source_sha256", "source_field"):
            if str(channels[key][field]) != str(distribution_channels[key][field]):
                raise ValueError(f"{field} mismatch between evidence sources for {key}")

    entries = decisions.get("decisions")
    if not isinstance(entries, list) or len(entries) != 8:
        raise ValueError("threshold decisions must contain exactly eight entries")
    reviewer = str(decisions.get("reviewer", "")).strip()
    if not reviewer:
        raise ValueError("threshold decisions require a reviewer")

    selected: dict[tuple[str, str], float] = {}
    frozen: list[dict[str, Any]] = []
    for entry in entries:
        key = (str(entry["fly_alias"]), str(entry["soma_side"]))
        if key in selected:
            raise ValueError(f"duplicate threshold decision: {key}")
        if key not in channels:
            raise ValueError(f"unknown threshold decision channel: {key}")
        channel = channels[key]
        distribution_channel = distribution_channels[key]
        for field in ("filename", "source_sha256", "source_field"):
            if str(entry[field]) != str(channel[field]):
                raise ValueError(f"{field} mismatch for {key}")

        quantile = float(entry["selected_prominence_quantile"])
        if quantile not in _DISTRIBUTION_Q:
            raise ValueError(f"v2 primary threshold must be distribution-reviewed for {key}")
        candidate = _candidate(channel, quantile)
        threshold = float(entry["selected_threshold"])
        if abs(threshold - float(candidate["threshold"])) > 1e-12:
            raise ValueError(f"selected threshold is not the exact audited candidate for {key}")

        basis = [str(value) for value in entry.get("selection_basis", [])]
        if not basis or len(basis) != len(set(basis)):
            raise ValueError(f"selection_basis must be non-empty and unique for {key}")
        invalid = set(basis) - _ALLOWED_BASIS
        if invalid:
            raise ValueError(f"forbidden selection_basis values for {key}: {sorted(invalid)}")
        if "individual_waveform_distribution" not in basis:
            raise ValueError(f"distribution-review evidence is required for {key}")
        rationale = str(entry.get("rationale", "")).strip()
        if len(rationale) < 40:
            raise ValueError(f"rationale must be at least 40 characters for {key}")

        selected[key] = quantile
        frozen.append(
            {
                "fly_alias": key[0],
                "soma_side": key[1],
                "filename": channel["filename"],
                "source_sha256": channel["source_sha256"],
                "source_field": channel["source_field"],
                "selected_prominence_quantile": quantile,
                "selected_threshold": threshold,
                "selection_basis": basis,
                "rationale": rationale,
                "selected_candidate_evidence": _candidate_evidence(candidate),
                "distribution_evidence": _compact_distribution(distribution_channel),
            }
        )

    if set(selected) != set(channels):
        raise ValueError("threshold decisions do not cover the complete eight-channel set")

    payload: dict[str, Any] = {
        "schema": "fly-sniff-dna02-threshold-manifest-v2",
        "status": "THRESHOLDS_FROZEN_BEFORE_BEHAVIOR_V2",
        "code_ref": code_ref,
        "source_contract_sha256": evidence["source_contract_sha256"],
        "source_byte_evidence_sha256": evidence["source_byte_evidence_sha256"],
        "prominence_audit_sha256": evidence["prominence_audit_sha256"],
        "threshold_qc_sha256": evidence["parent_threshold_qc_sha256"],
        "freeze_evidence_sha256": evidence["evidence_sha256"],
        "distribution_review_sha256": distribution["parent_distribution_review_sha256"],
        "distribution_freeze_evidence_sha256": distribution["evidence_sha256"],
        "threshold_decisions_sha256": decisions["decisions_sha256"],
        "reviewer": reviewer,
        "adjudication_method": decisions["adjudication_method"],
        "behavior_fields_reviewed": False,
        "yaw_reviewed": False,
        "navigation_performance_used": False,
        "figure3c_statistic_reviewed": False,
        "event_rate_used_for_selection": False,
        "thresholds_frozen": True,
        "threshold_count": 8,
        "thresholds": sorted(frozen, key=lambda item: (item["fly_alias"], item["soma_side"])),
        "pre_behavior_sensitivity_profiles": _sensitivity_profiles(selected, channels),
        "sensitivity_policy": (
            "Primary, one-step-lower, and one-step-higher prominence profiles are frozen "
            "before behavior is opened. Sensitivity results may describe robustness but may "
            "not replace the primary thresholds in v2."
        ),
        "allowed_next_action": (
            "Run the independently qualified Figure 3C reproduction pipeline. Opening yaw "
            "consumes the v2 threshold one-way door; the primary thresholds may not then change."
        ),
        "forbidden_actions": [
            "retune primary thresholds after inspecting yaw or Figure 3C fit",
            "retune primary thresholds using odor-navigation performance",
            "promote a sensitivity profile to primary because downstream results look better",
            "drop a Figure 3C fly because its downstream relationship is inconvenient",
        ],
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze eight DNa02 thresholds after the individual-waveform distribution gate"
    )
    parser.add_argument("--freeze-evidence", required=True)
    parser.add_argument("--distribution-evidence", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--code-ref", required=True)
    args = parser.parse_args(argv)
    result = freeze_manifest_v2(
        freeze_evidence_path=args.freeze_evidence,
        distribution_evidence_path=args.distribution_evidence,
        decisions_path=args.decisions,
        output_path=args.out,
        code_ref=args.code_ref,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
