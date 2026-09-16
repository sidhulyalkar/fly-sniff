from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

from .dna02_source import READY, load_contract
from .dna02_source_inspect import DEFAULT_EVIDENCE, _hash_file, _load_byte_evidence
from .freeze import canonical_sha256

DEFAULT_CONTRACT = Path("authority/program-a-dna02-source-contract-v1.json")
_ALLOWED_MAT_FIELDS = ("ephys_SR", "ephys_A", "ephys_B")
_CHANNELS = (("ephys_A", "L"), ("ephys_B", "R"))
_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _source_map(source_paths: list[str | Path], expected: set[str]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for value in source_paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"DNa02 source path is not a file: {path}")
        if path.name not in expected:
            raise ValueError(f"unexpected DNa02 source filename: {path.name}")
        if path.name in mapped:
            raise ValueError(f"duplicate DNa02 source filename: {path.name}")
        mapped[path.name] = path
    missing = expected - set(mapped)
    if missing:
        raise ValueError(f"missing frozen DNa02 source files: {sorted(missing)}")
    return mapped


def _load_neural_channel(path: Path, channel: str) -> tuple[float, np.ndarray]:
    if channel not in {item[0] for item in _CHANNELS}:
        raise ValueError(f"channel is not allowlisted: {channel}")
    # Deliberately restrict scipy to neural fields. Behavior fields are never loaded in this lane.
    data = loadmat(path, variable_names=["ephys_SR", channel], squeeze_me=True)
    if "ephys_SR" not in data or channel not in data:
        raise ValueError(f"{path.name} is missing {channel} or ephys_SR")
    fs = float(np.asarray(data["ephys_SR"]).reshape(-1)[0])
    trace = np.asarray(data[channel], dtype=np.float64).reshape(-1)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"invalid ephys sampling rate in {path.name}")
    if trace.size == 0 or not np.all(np.isfinite(trace)):
        raise ValueError(f"{path.name}:{channel} contains empty or non-finite voltage data")
    return fs, trace


def _prominence_summary(trace: np.ndarray, fs: float) -> dict[str, Any]:
    # MATLAB Figure 3B/C methods specify relative prominence. No behavior signal enters this step.
    peaks, properties = find_peaks(trace, prominence=(0.0, None))
    prominences = np.asarray(properties["prominences"], dtype=np.float64)
    positive = prominences[prominences > 0]
    if positive.size == 0:
        raise ValueError("no positive-prominence local maxima found")

    q_values = np.quantile(positive, _QUANTILES)
    candidates = []
    duration_s = trace.size / fs
    for q, threshold in zip(_QUANTILES, q_values, strict=True):
        count = int(np.count_nonzero(positive >= threshold))
        candidates.append(
            {
                "prominence_quantile": q,
                "threshold": float(threshold),
                "events_at_or_above": count,
                "event_rate_hz": float(count / duration_s),
            }
        )

    lo = float(np.min(positive))
    hi = float(np.max(positive))
    if hi > lo:
        edges = np.geomspace(max(lo, np.finfo(float).tiny), hi, 65)
        hist, edges = np.histogram(positive, bins=edges)
        histogram = {"edges": edges.tolist(), "counts": hist.astype(int).tolist()}
    else:
        histogram = {"edges": [lo, hi], "counts": [int(positive.size)]}

    return {
        "sample_count": int(trace.size),
        "duration_s": float(duration_s),
        "local_maxima_count": int(peaks.size),
        "positive_prominence_count": int(positive.size),
        "prominence_quantiles": {
            f"q{q:g}": float(value) for q, value in zip(_QUANTILES, q_values, strict=True)
        },
        "candidate_sweep": candidates,
        "prominence_histogram": histogram,
    }


def audit_sources(
    source_paths: list[str | Path],
    *,
    output_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT,
    evidence_path: str | Path = DEFAULT_EVIDENCE,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if contract.status != READY:
        raise ValueError("DNa02 source contract is not READY_FOR_EXTRACTION")
    evidence = _load_byte_evidence(Path(evidence_path))
    evidence_files = evidence.get("files")
    if not isinstance(evidence_files, list) or len(evidence_files) != 4:
        raise ValueError("DNa02 byte evidence must contain exactly four files")
    evidence_by_name = {str(item["filename"]): item for item in evidence_files}
    contract_by_name = {ref.filename: ref for ref in contract.data_file_map}
    if set(evidence_by_name) != set(contract_by_name):
        raise ValueError("source contract and byte evidence filename sets differ")
    local = _source_map(source_paths, set(contract_by_name))

    reports: list[dict[str, Any]] = []
    for filename in sorted(contract_by_name):
        ref = contract_by_name[filename]
        evidence_ref = evidence_by_name[filename]
        path = local[filename]
        size, md5, sha256 = _hash_file(path)
        if size != int(evidence_ref["byte_count"]):
            raise ValueError(f"byte-count mismatch for {filename}")
        if md5 != str(evidence_ref["observed_md5"]):
            raise ValueError(f"MD5 mismatch for {filename}")
        if sha256 != ref.sha256 or sha256 != str(evidence_ref["observed_sha256"]):
            raise ValueError(f"SHA-256 mismatch for {filename}")

        channel_reports = []
        observed_fs: float | None = None
        for channel, soma_side in _CHANNELS:
            fs, trace = _load_neural_channel(path, channel)
            if observed_fs is None:
                observed_fs = fs
            elif fs != observed_fs:
                raise ValueError(f"sampling-rate disagreement within {filename}")
            summary = _prominence_summary(trace, fs)
            channel_reports.append(
                {
                    "source_field": channel,
                    "soma_side": soma_side,
                    **summary,
                }
            )
            del trace

        reports.append(
            {
                "fly_alias": ref.fly_alias,
                "filename": filename,
                "sha256": sha256,
                "ephys_sampling_rate_hz": observed_fs,
                "channels": channel_reports,
            }
        )

    payload: dict[str, Any] = {
        "schema": "fly-sniff-dna02-prominence-audit-v1",
        "status": "PROMINENCE_AUDITED_PENDING_THRESHOLD_ADJUDICATION",
        "source_contract_sha256": contract.sha256,
        "source_byte_evidence_sha256": evidence["evidence_sha256"],
        "input_fields_allowlist": list(_ALLOWED_MAT_FIELDS),
        "behavior_fields_loaded": False,
        "yaw_loaded": False,
        "navigation_performance_used": False,
        "figure3c_statistic_computed": False,
        "thresholds_frozen": False,
        "threshold_selection_rule": (
            "No threshold is selected automatically. Quantiles and threshold sweeps summarize only "
            "relative-prominence structure for independent electrophysiology review."
        ),
        "files": reports,
        "next_allowed_action": (
            "Adjudicate one prominence threshold per fly/channel using neural-only QC, freeze the eight "
            "thresholds with provenance, then run the deterministic Figure 3C extractor."
        ),
        "forbidden_interpretation": [
            "a prominence quantile is automatically the publication's original hand-tuned threshold",
            "the audit reproduces Figure 3C before thresholds are independently frozen",
            "behavior or navigation performance may be used to choose a threshold",
        ],
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit DNa02 spike-prominence structure without loading behavior or choosing thresholds"
    )
    parser.add_argument("sources", nargs="+", help="The four exact authenticated DNa02 MAT files")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = audit_sources(
        args.sources,
        output_path=args.out,
        contract_path=args.contract,
        evidence_path=args.evidence,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
