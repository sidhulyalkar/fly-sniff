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
_Q_ORDER = (0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999)
_DISTRIBUTION_Q = {0.995, 0.999, 0.9995}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _validated_hashed_json(
    path: Path, *, schema: str, hash_field: str
) -> dict[str, Any]:
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


def _qc_channel_map(qc: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for file_item in qc.get("files", []):
        alias = str(file_item["fly_alias"])
        for channel in file_item.get("channels", []):
            key = (alias, str(channel["soma_side"]))
            if key in result:
                raise ValueError(f"duplicate QC channel: {key}")
            result[key] = {
                **channel,
                "filename": str(file_item["filename"]),
                "source_sha256": str(file_item["source_sha256"]),
            }
    if len(result) != 8:
        raise ValueError("threshold QC must contain exactly eight fly/side channels")
    return result


def _distribution_channel_map(
    review: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for file_item in review.get("files", []):
        alias = str(file_item["fly_alias"])
        for channel in file_item.get("channels", []):
            key = (alias, str(channel["soma_side"]))
            if key in result:
                raise ValueError(f"duplicate distribution-review channel: {key}")
            result[key] = {
                **channel,
                "filename": str(file_item["filename"]),
                "source_sha256": str(file_item["source_sha256"]),
            }
    if len(result) != 8:
        raise ValueError("distribution review must contain exactly eight fly/side channels")
    return result


def _candidate(qc_channel: dict[str, Any], quantile: float) -> dict[str, Any]:
    matches = [
        item
        for item in qc_channel["candidate_qc"]
        if abs(float(item["prominence_quantile"]) - quantile) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one QC candidate for q={quantile:g}")
    return matches[0]


def _compact_band(band: dict[str, Any]) -> dict[str, Any]:
    corr = band["anchor_correlation_quantiles"]
    amp = band["peak_amplitude_quantiles"]
    return {
        "label": str(band["label"]),
        "total_event_count_descriptive_only": int(band["total_event_count"]),
        "anchor_correlation_p10": corr["p10"],
        "anchor_correlation_p50": corr["p50"],
        "anchor_correlation_fraction_ge_0p90": band[
            "anchor_correlation_fraction_ge_0p90"
        ],
        "anchor_correlation_fraction_ge_0p95": band[
            "anchor_correlation_fraction_ge_0p95"
        ],
        "peak_amplitude_p10": amp["p10"],
        "peak_amplitude_p50": amp["p50"],
    }


def _distribution_evidence(channel: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_quantile": float(channel["anchor_quantile"]),
        "bands": [_compact_band(item) for item in channel["bands"]],
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_count_descriptive_only": int(candidate["event_count"]),
        "event_rate_hz_descriptive_only": float(candidate["event_rate_hz"]),
        "refractory_violation_fraction_lt_1ms": float(
            candidate["refractory_violation_fraction_lt_1ms"]
        ),
        "short_isi_fraction_lt_2ms": float(candidate["short_isi_fraction_lt_2ms"]),
        "block_rate_cv": float(candidate["block_rate_cv"]),
        "waveform_median_template_correlation": float(
            candidate["waveform"]["median_template_correlation"]
        ),
        "waveform_fwhm_ms": float(candidate["waveform"]["fwhm_ms"]),
    }


def _profile_entry(qc_channel: dict[str, Any], quantile: float) -> dict[str, float]:
    candidate = _candidate(qc_channel, quantile)
    return {
        "prominence_quantile": float(quantile),
        "threshold": float(candidate["threshold"]),
    }


def _sensitivity_profiles(
    selected: dict[tuple[str, str], float],
    qc_channels: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    profiles: dict[str, dict[str, dict[str, float]]] = {
        "lower_one_step": {},
        "primary": {},
        "higher_one_step": {},
    }
    for key in sorted(selected):
        quantile = selected[key]
        try:
            index = _Q_ORDER.index(quantile)
        except ValueError as exc:
            raise ValueError(f"selected quantile is outside frozen audit grid: {key}") from exc
        label = f"{key[0]}:{key[1]}"
        profiles["primary"][label] = _profile_entry(qc_channels[key], quantile)
        profiles["lower_one_step"][label] = _profile_entry(
            qc_channels[key], _Q_ORDER[max(0, index - 1)]
        )
        profiles["higher_one_step"][label] = _profile_entry(
            qc_channels[key], _Q_ORDER[min(len(_Q_ORDER) - 1, index + 1)]
        )
    return profiles


def freeze_manifest_v2(
    *,
    qc_path: str | Path,
    distribution_review_path: str | Path,
    decisions_path: str | Path,
    output_path: str | Path,
    code_ref: str,
) -> dict[str, Any]:
    if not _SHA40.fullmatch(code_ref):
        raise ValueError("code_ref must be an exact lowercase 40-character git commit SHA")

    qc = _validated_hashed_json(
        Path(qc_path),
        schema="fly-sniff-dna02-threshold-adjudication-qc-v1",
        hash_field="qc_sha256",
    )
    review = _validated_hashed_json(
        Path(distribution_review_path),
        schema="fly-sniff-dna02-threshold-distribution-review-v1",
        hash_field="review_sha256",
    )
    _require_false(
        qc,
        (
            "behavior_fields_loaded",
            "yaw_loaded",
            "navigation_performance_used",
            "figure3c_statistic_computed",
            "thresholds_frozen",
            "automatic_threshold_selection",
        ),
        label="threshold QC",
    )
    _require_false(
        review,
        (
            "behavior_fields_loaded",
            "yaw_loaded",
            "navigation_performance_used",
            "figure3c_statistic_computed",
            "thresholds_frozen",
            "automatic_threshold_selection",
        ),
        label="distribution review",
    )
    if review.get("prominence_audit_sha256") != qc.get("prominence_audit_sha256"):
        raise ValueError("distribution review and threshold QC reference different prominence audits")

    qc_channels = _qc_channel_map(qc)
    review_channels = _distribution_channel_map(review)
    if set(qc_channels) != set(review_channels):
        raise ValueError("QC and distribution-review channel sets differ")
    for key in qc_channels:
        qch = qc_channels[key]
        rch = review_channels[key]
        if rch["filename"] != qch["filename"]:
            raise ValueError(f"filename mismatch between QC and distribution review for {key}")
        if rch["source_sha256"] != qch["source_sha256"]:
            raise ValueError(f"source SHA-256 mismatch between QC and distribution review for {key}")
        if str(rch["source_field"]) != str(qch["source_field"]):
            raise ValueError(f"source field mismatch between QC and distribution review for {key}")

    decisions = _validated_hashed_json(
        Path(decisions_path),
        schema="fly-sniff-dna02-threshold-decisions-v2",
        hash_field="decisions_sha256",
    )
    if decisions.get("prominence_audit_sha256") != qc.get("prominence_audit_sha256"):
        raise ValueError("decisions reference a different prominence audit")
    if decisions.get("threshold_qc_sha256") != qc.get("qc_sha256"):
        raise ValueError("decisions reference a different threshold QC")
    if decisions.get("distribution_review_sha256") != review.get("review_sha256"):
        raise ValueError("decisions reference a different distribution review")
    if decisions.get("event_rate_used_for_selection") is not False:
        raise ValueError("event rate may not be used for threshold selection")
    _require_false(
        decisions,
        (
            "behavior_fields_reviewed",
            "yaw_reviewed",
            "navigation_performance_used",
            "figure3c_statistic_reviewed",
        ),
        label="threshold decisions",
    )

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
        if key not in qc_channels:
            raise ValueError(f"unknown threshold decision channel: {key}")
        qch = qc_channels[key]
        rch = review_channels[key]
        for field in ("filename", "source_sha256", "source_field"):
            expected = str(qch[field])
            if str(entry[field]) != expected:
                raise ValueError(f"{field} mismatch for {key}")

        quantile = float(entry["selected_prominence_quantile"])
        if quantile not in _DISTRIBUTION_Q:
            raise ValueError(f"v2 primary threshold must be distribution-reviewed for {key}")
        candidate = _candidate(qch, quantile)
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

        expected_distribution = _distribution_evidence(rch)
        if entry.get("distribution_evidence") != expected_distribution:
            raise ValueError(f"distribution evidence mismatch for {key}")

        selected[key] = quantile
        frozen.append(
            {
                "fly_alias": key[0],
                "soma_side": key[1],
                "filename": qch["filename"],
                "source_sha256": qch["source_sha256"],
                "source_field": qch["source_field"],
                "selected_prominence_quantile": quantile,
                "selected_threshold": threshold,
                "selection_basis": basis,
                "rationale": rationale,
                "selected_candidate_qc": _candidate_summary(candidate),
                "distribution_evidence": expected_distribution,
            }
        )

    if set(selected) != set(qc_channels):
        raise ValueError("threshold decisions do not cover the complete eight-channel QC set")

    payload: dict[str, Any] = {
        "schema": "fly-sniff-dna02-threshold-manifest-v2",
        "status": "THRESHOLDS_FROZEN_BEFORE_BEHAVIOR_V2",
        "code_ref": code_ref,
        "source_contract_sha256": qc["source_contract_sha256"],
        "source_byte_evidence_sha256": qc["source_byte_evidence_sha256"],
        "prominence_audit_sha256": qc["prominence_audit_sha256"],
        "threshold_qc_sha256": qc["qc_sha256"],
        "distribution_review_sha256": review["review_sha256"],
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
        "thresholds": sorted(
            frozen, key=lambda item: (item["fly_alias"], item["soma_side"])
        ),
        "pre_behavior_sensitivity_profiles": _sensitivity_profiles(
            selected, qc_channels
        ),
        "sensitivity_policy": (
            "Primary, one-step-lower, and one-step-higher prominence profiles are frozen "
            "before behavior is opened. Sensitivity results may describe robustness but may "
            "not be used to replace the primary thresholds in v2."
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
        description=(
            "Freeze eight DNa02 thresholds only after the individual-waveform "
            "distribution-review gate"
        )
    )
    parser.add_argument("--qc", required=True)
    parser.add_argument("--distribution-review", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--code-ref", required=True)
    args = parser.parse_args(argv)
    result = freeze_manifest_v2(
        qc_path=args.qc,
        distribution_review_path=args.distribution_review,
        decisions_path=args.decisions,
        output_path=args.out,
        code_ref=args.code_ref,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
