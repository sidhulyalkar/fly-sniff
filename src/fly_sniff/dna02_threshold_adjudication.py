from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

from .dna02_source import READY, load_contract
from .dna02_source_inspect import DEFAULT_EVIDENCE, _hash_file, _load_byte_evidence
from .dna02_threshold_audit import DEFAULT_CONTRACT
from .freeze import canonical_sha256

_CHANNELS = (("ephys_A", "L"), ("ephys_B", "R"))
_DEFAULT_CANDIDATE_QUANTILES = (0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999)
_ALLOWED_SELECTION_BASIS = (
    "raw_trace_review",
    "waveform_stereotypy",
    "refractory_violations",
    "blockwise_stability",
    "bilateral_artifact_coincidence",
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _validated_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "fly-sniff-dna02-prominence-audit-v1":
        raise ValueError("unexpected prominence-audit schema")
    observed = str(payload.get("audit_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("audit_sha256", None)
    expected = canonical_sha256(unhashed)
    if observed != expected:
        raise ValueError("prominence-audit hash mismatch")
    if payload.get("behavior_fields_loaded") is not False:
        raise ValueError("prominence audit is not behavior-blind")
    if payload.get("yaw_loaded") is not False:
        raise ValueError("prominence audit loaded yaw")
    if payload.get("navigation_performance_used") is not False:
        raise ValueError("prominence audit used navigation performance")
    if payload.get("figure3c_statistic_computed") is not False:
        raise ValueError("prominence audit already computed Figure 3C")
    if payload.get("thresholds_frozen") is not False:
        raise ValueError("prominence audit unexpectedly reports frozen thresholds")
    return payload


def _audit_channel_map(audit: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    mapped: dict[tuple[str, str], dict[str, Any]] = {}
    for file_item in audit.get("files", []):
        alias = str(file_item["fly_alias"])
        for channel in file_item.get("channels", []):
            key = (alias, str(channel["soma_side"]))
            if key in mapped:
                raise ValueError(f"duplicate audit channel: {key}")
            mapped[key] = {
                **channel,
                "filename": str(file_item["filename"]),
                "source_sha256": str(file_item["sha256"]),
                "ephys_sampling_rate_hz": float(file_item["ephys_sampling_rate_hz"]),
            }
    if len(mapped) != 8:
        raise ValueError("prominence audit must contain exactly eight fly/side channels")
    return mapped


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


def _load_channel(path: Path, channel: str) -> tuple[float, np.ndarray]:
    if channel not in {item[0] for item in _CHANNELS}:
        raise ValueError(f"channel is not allowlisted: {channel}")
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


def _candidate_thresholds(
    audit_channel: dict[str, Any], quantiles: tuple[float, ...]
) -> list[tuple[float, float]]:
    sweep = {
        float(item["prominence_quantile"]): float(item["threshold"])
        for item in audit_channel["candidate_sweep"]
    }
    result: list[tuple[float, float]] = []
    for quantile in quantiles:
        if quantile not in sweep:
            raise ValueError(f"audit is missing requested prominence quantile {quantile:g}")
        threshold = sweep[quantile]
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"invalid candidate threshold for q={quantile:g}")
        result.append((quantile, threshold))
    if len({threshold for _, threshold in result}) != len(result):
        raise ValueError("candidate thresholds must be unique")
    return sorted(result, key=lambda item: item[1])


def _deterministic_subsample(
    peaks: np.ndarray, *, cap: int, seed_text: str, margin: int, n_samples: int
) -> np.ndarray:
    eligible = peaks[(peaks >= margin) & (peaks < n_samples - margin)]
    if eligible.size <= cap:
        return eligible
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=cap, replace=False))


def _fwhm_ms(median_waveform: np.ndarray, peak_offset: int, fs: float) -> float | None:
    peak_value = float(median_waveform[peak_offset])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return None
    half = 0.5 * peak_value
    left = np.flatnonzero(median_waveform[: peak_offset + 1] <= half)
    right = np.flatnonzero(median_waveform[peak_offset:] <= half)
    if left.size == 0 or right.size == 0:
        return None
    left_idx = int(left[-1])
    right_idx = int(peak_offset + right[0])
    if right_idx <= left_idx:
        return None
    return float((right_idx - left_idx) / fs * 1000.0)


def _waveform_summary(
    trace: np.ndarray,
    peaks: np.ndarray,
    *,
    fs: float,
    seed_text: str,
    cap: int = 2000,
    pre_ms: float = 1.0,
    post_ms: float = 2.0,
) -> dict[str, Any]:
    pre = max(1, int(round(pre_ms * fs / 1000.0)))
    post = max(1, int(round(post_ms * fs / 1000.0)))
    margin = max(pre, post) + 1
    sampled = _deterministic_subsample(
        peaks, cap=cap, seed_text=seed_text, margin=margin, n_samples=trace.size
    )
    if sampled.size == 0:
        return {
            "sampled_waveform_count": 0,
            "pre_ms": pre_ms,
            "post_ms": post_ms,
            "time_ms": [],
            "median_waveform": [],
            "mad_waveform": [],
            "median_peak_amplitude": None,
            "mad_peak_amplitude": None,
            "median_template_correlation": None,
            "fwhm_ms": None,
        }

    offsets = np.arange(-pre, post + 1)
    waves = trace[sampled[:, None] + offsets[None, :]].astype(np.float64, copy=False)
    baseline_width = max(1, pre // 2)
    baselines = np.median(waves[:, :baseline_width], axis=1, keepdims=True)
    waves = waves - baselines
    median_waveform = np.median(waves, axis=0)
    mad_waveform = np.median(np.abs(waves - median_waveform[None, :]), axis=0)
    peak_amplitudes = waves[:, pre]

    centered_template = median_waveform - np.mean(median_waveform)
    template_norm = float(np.linalg.norm(centered_template))
    correlations: list[float] = []
    if template_norm > 0:
        centered_waves = waves - np.mean(waves, axis=1, keepdims=True)
        norms = np.linalg.norm(centered_waves, axis=1)
        valid = norms > 0
        if np.any(valid):
            correlations = (
                centered_waves[valid] @ centered_template / (norms[valid] * template_norm)
            ).tolist()

    return {
        "sampled_waveform_count": int(sampled.size),
        "pre_ms": float(pre_ms),
        "post_ms": float(post_ms),
        "time_ms": (offsets / fs * 1000.0).tolist(),
        "median_waveform": median_waveform.tolist(),
        "mad_waveform": mad_waveform.tolist(),
        "median_peak_amplitude": float(np.median(peak_amplitudes)),
        "mad_peak_amplitude": float(
            np.median(np.abs(peak_amplitudes - np.median(peak_amplitudes)))
        ),
        "median_template_correlation": (
            float(np.median(correlations)) if correlations else None
        ),
        "fwhm_ms": _fwhm_ms(median_waveform, pre, fs),
    }


def _block_rates(peaks: np.ndarray, *, n_samples: int, fs: float, blocks: int = 20) -> list[float]:
    edges = np.linspace(0, n_samples, blocks + 1, dtype=np.int64)
    counts, _ = np.histogram(peaks, bins=edges)
    durations = np.diff(edges) / fs
    return (counts / durations).astype(float).tolist()


def _candidate_qc(
    trace: np.ndarray,
    peaks: np.ndarray,
    prominences: np.ndarray,
    *,
    fs: float,
    quantile: float,
    threshold: float,
    seed_text: str,
) -> tuple[dict[str, Any], np.ndarray]:
    selected = np.asarray(peaks[prominences >= threshold], dtype=np.int64)
    duration_s = trace.size / fs
    isi = np.diff(selected) / fs if selected.size >= 2 else np.array([], dtype=float)
    rates = _block_rates(selected, n_samples=trace.size, fs=fs)
    rate_mean = float(np.mean(rates))
    rate_cv = float(np.std(rates) / rate_mean) if rate_mean > 0 else None

    return (
        {
            "prominence_quantile": float(quantile),
            "threshold": float(threshold),
            "event_count": int(selected.size),
            "event_rate_hz": float(selected.size / duration_s),
            "isi_median_ms": float(np.median(isi) * 1000.0) if isi.size else None,
            "isi_p01_ms": float(np.quantile(isi, 0.01) * 1000.0) if isi.size else None,
            "refractory_violation_fraction_lt_1ms": (
                float(np.mean(isi < 0.001)) if isi.size else None
            ),
            "short_isi_fraction_lt_2ms": float(np.mean(isi < 0.002)) if isi.size else None,
            "block_rate_hz": rates,
            "block_rate_cv": rate_cv,
            "waveform": _waveform_summary(
                trace,
                selected,
                fs=fs,
                seed_text=f"{seed_text}|q={quantile:g}|threshold={threshold:.17g}",
            ),
        },
        selected,
    )


def _plot_raw_windows(
    trace: np.ndarray,
    *,
    fs: float,
    peak_sets: list[tuple[str, np.ndarray]],
    output_path: Path,
    title: str,
    window_s: float = 0.25,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = trace.size
    width = max(10, int(round(window_s * fs)))
    centers = (np.array([0.10, 0.30, 0.50, 0.70, 0.90]) * (n - 1)).astype(np.int64)
    fig, axes = plt.subplots(len(centers), 1, figsize=(12, 10), sharey=True)
    for axis, center in zip(np.atleast_1d(axes), centers, strict=True):
        start = max(0, int(center - width // 2))
        stop = min(n, start + width)
        x = np.arange(start, stop) / fs
        axis.plot(x, trace[start:stop], linewidth=0.8, label="raw ephys")
        for label, peaks in peak_sets:
            lo = int(np.searchsorted(peaks, start, side="left"))
            hi = int(np.searchsorted(peaks, stop, side="left"))
            local = peaks[lo:hi]
            if local.size:
                axis.scatter(
                    local / fs,
                    trace[local],
                    s=10,
                    alpha=0.75,
                    label=label,
                )
        axis.set_ylabel("voltage")
        axis.legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title + " | deterministic 10/30/50/70/90% trace windows")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_candidate_summary(
    candidates: list[dict[str, Any]], *, output_path: Path, title: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = np.array([item["threshold"] for item in candidates], dtype=float)
    rates = np.array([item["event_rate_hz"] for item in candidates], dtype=float)
    refractory = np.array(
        [
            np.nan
            if item["refractory_violation_fraction_lt_1ms"] is None
            else item["refractory_violation_fraction_lt_1ms"]
            for item in candidates
        ],
        dtype=float,
    )
    stability = np.array(
        [np.nan if item["block_rate_cv"] is None else item["block_rate_cv"] for item in candidates],
        dtype=float,
    )
    stereotypy = np.array(
        [
            np.nan
            if item["waveform"]["median_template_correlation"] is None
            else item["waveform"]["median_template_correlation"]
            for item in candidates
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(thresholds, rates, marker="o")
    axes[0, 0].set_ylabel("event rate (Hz, descriptive only)")
    axes[0, 1].plot(thresholds, refractory, marker="o")
    axes[0, 1].set_ylabel("ISI < 1 ms fraction")
    axes[1, 0].plot(thresholds, stability, marker="o")
    axes[1, 0].set_ylabel("20-block event-rate CV")
    axes[1, 1].plot(thresholds, stereotypy, marker="o")
    axes[1, 1].set_ylabel("median waveform/template corr")
    for axis in axes.flat:
        axis.set_xlabel("prominence threshold")
        axis.grid(alpha=0.2)
    fig.suptitle(title + " | no automatic threshold selection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _coincidence_fraction(
    reference: np.ndarray, other: np.ndarray, tolerance_samples: int
) -> float | None:
    if reference.size == 0:
        return None
    if other.size == 0:
        return 0.0
    idx = np.searchsorted(other, reference)
    left_idx = np.clip(idx - 1, 0, other.size - 1)
    right_idx = np.clip(idx, 0, other.size - 1)
    distance = np.minimum(np.abs(reference - other[left_idx]), np.abs(reference - other[right_idx]))
    return float(np.mean(distance <= tolerance_samples))


def _bilateral_coincidence(
    left: dict[float, np.ndarray],
    right: dict[float, np.ndarray],
    *,
    fs: float,
) -> list[dict[str, Any]]:
    result = []
    for quantile in sorted(set(left) & set(right)):
        l_peaks = left[quantile]
        r_peaks = right[quantile]
        item: dict[str, Any] = {"prominence_quantile": quantile}
        for tolerance_ms in (0.2, 0.5):
            tolerance_samples = max(1, int(round(tolerance_ms * fs / 1000.0)))
            l_to_r = _coincidence_fraction(l_peaks, r_peaks, tolerance_samples)
            r_to_l = _coincidence_fraction(r_peaks, l_peaks, tolerance_samples)
            item[f"left_fraction_with_right_within_{tolerance_ms:g}ms"] = l_to_r
            item[f"right_fraction_with_left_within_{tolerance_ms:g}ms"] = r_to_l
        result.append(item)
    return result


def adjudicate_sources(
    source_paths: list[str | Path],
    *,
    prominence_audit_path: str | Path,
    output_dir: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT,
    evidence_path: str | Path = DEFAULT_EVIDENCE,
    candidate_quantiles: tuple[float, ...] = _DEFAULT_CANDIDATE_QUANTILES,
    write_plots: bool = True,
) -> dict[str, Any]:
    audit_path = Path(prominence_audit_path)
    audit = _validated_receipt(audit_path)
    audit_channels = _audit_channel_map(audit)

    contract = load_contract(contract_path)
    if contract.status != READY:
        raise ValueError("DNa02 source contract is not READY_FOR_EXTRACTION")
    evidence = _load_byte_evidence(Path(evidence_path))
    evidence_files = evidence.get("files")
    if not isinstance(evidence_files, list) or len(evidence_files) != 4:
        raise ValueError("DNa02 byte evidence must contain exactly four files")
    evidence_by_name = {str(item["filename"]): item for item in evidence_files}
    contract_by_name = {ref.filename: ref for ref in contract.data_file_map}
    local = _source_map(source_paths, set(contract_by_name))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_reports: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for filename in sorted(contract_by_name):
        ref = contract_by_name[filename]
        path = local[filename]
        evidence_ref = evidence_by_name[filename]
        size, md5, sha256 = _hash_file(path)
        if size != int(evidence_ref["byte_count"]):
            raise ValueError(f"byte-count mismatch for {filename}")
        if md5 != str(evidence_ref["observed_md5"]):
            raise ValueError(f"MD5 mismatch for {filename}")
        if sha256 != ref.sha256 or sha256 != str(evidence_ref["observed_sha256"]):
            raise ValueError(f"SHA-256 mismatch for {filename}")

        channel_reports: list[dict[str, Any]] = []
        peak_sets_by_side: dict[str, dict[float, np.ndarray]] = {}
        observed_fs: float | None = None

        for channel, soma_side in _CHANNELS:
            audit_channel = audit_channels[(ref.fly_alias, soma_side)]
            if audit_channel["filename"] != filename or audit_channel["source_sha256"] != sha256:
                raise ValueError(f"audit/source identity mismatch for {ref.fly_alias}:{soma_side}")
            candidates = _candidate_thresholds(audit_channel, candidate_quantiles)
            fs, trace = _load_channel(path, channel)
            if observed_fs is None:
                observed_fs = fs
            elif fs != observed_fs:
                raise ValueError(f"sampling-rate disagreement within {filename}")
            if fs != float(audit_channel["ephys_sampling_rate_hz"]):
                raise ValueError(
                    f"audit/source sampling-rate mismatch for {ref.fly_alias}:{soma_side}"
                )

            min_threshold = min(threshold for _, threshold in candidates)
            peaks, properties = find_peaks(trace, prominence=(min_threshold, None))
            prominences = np.asarray(properties["prominences"], dtype=np.float64)

            qc_candidates: list[dict[str, Any]] = []
            peak_sets: dict[float, np.ndarray] = {}
            for quantile, threshold in candidates:
                qc, selected = _candidate_qc(
                    trace,
                    peaks,
                    prominences,
                    fs=fs,
                    quantile=quantile,
                    threshold=threshold,
                    seed_text=f"{sha256}|{channel}",
                )
                qc_candidates.append(qc)
                peak_sets[quantile] = selected

            if write_plots:
                anchor_indices = sorted({0, len(candidates) // 2, len(candidates) - 1})
                anchors = [
                    (
                        f"q={candidates[index][0]:g}, thr={candidates[index][1]:.4g}",
                        peak_sets[candidates[index][0]],
                    )
                    for index in anchor_indices
                ]
                stem = f"{ref.fly_alias}-{soma_side}"
                _plot_raw_windows(
                    trace,
                    fs=fs,
                    peak_sets=anchors,
                    output_path=out_dir / "plots" / f"{stem}-raw-windows.png",
                    title=f"{ref.fly_alias} {soma_side} ({channel})",
                )
                _plot_candidate_summary(
                    qc_candidates,
                    output_path=out_dir / "plots" / f"{stem}-candidate-qc.png",
                    title=f"{ref.fly_alias} {soma_side} ({channel})",
                )

            channel_reports.append(
                {
                    "source_field": channel,
                    "soma_side": soma_side,
                    "minimum_prominence_computed": float(min_threshold),
                    "candidate_qc": qc_candidates,
                }
            )
            peak_sets_by_side[soma_side] = peak_sets
            decisions.append(
                {
                    "fly_alias": ref.fly_alias,
                    "filename": filename,
                    "source_sha256": sha256,
                    "source_field": channel,
                    "soma_side": soma_side,
                    "candidate_thresholds": [
                        {"prominence_quantile": q, "threshold": threshold}
                        for q, threshold in candidates
                    ],
                    "selected_prominence_quantile": None,
                    "selected_threshold": None,
                    "selection_basis": [],
                    "rationale": "",
                }
            )
            del trace

        if observed_fs is None:
            raise RuntimeError("no ephys channels were analyzed")
        coincidence = _bilateral_coincidence(
            peak_sets_by_side["L"], peak_sets_by_side["R"], fs=observed_fs
        )
        file_reports.append(
            {
                "fly_alias": ref.fly_alias,
                "filename": filename,
                "source_sha256": sha256,
                "ephys_sampling_rate_hz": observed_fs,
                "channels": channel_reports,
                "bilateral_candidate_coincidence": coincidence,
            }
        )

    receipt: dict[str, Any] = {
        "schema": "fly-sniff-dna02-threshold-adjudication-qc-v1",
        "status": "NEURAL_QC_COMPLETE_PENDING_HUMAN_THRESHOLD_DECISIONS",
        "source_contract_sha256": contract.sha256,
        "source_byte_evidence_sha256": evidence["evidence_sha256"],
        "prominence_audit_sha256": audit["audit_sha256"],
        "candidate_quantiles": list(candidate_quantiles),
        "input_fields_allowlist": ["ephys_SR", "ephys_A", "ephys_B"],
        "behavior_fields_loaded": False,
        "yaw_loaded": False,
        "navigation_performance_used": False,
        "figure3c_statistic_computed": False,
        "thresholds_frozen": False,
        "automatic_threshold_selection": False,
        "selection_basis_allowlist": list(_ALLOWED_SELECTION_BASIS),
        "files": file_reports,
        "next_allowed_action": (
            "Review deterministic ephys-only plots/QC and fill exactly eight explicit threshold "
            "decisions. Do not inspect yaw, Figure 3C fit, or navigation performance while "
            "deciding."
        ),
        "forbidden_interpretation": [
            "event rate alone identifies the publication threshold",
            "a candidate with the cleanest downstream Figure 3C fit should be selected",
            "navigation performance may break an electrophysiology threshold tie",
        ],
    }
    receipt["qc_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(out_dir / "threshold-qc-v1.json", receipt)

    decision_template: dict[str, Any] = {
        "schema": "fly-sniff-dna02-threshold-decisions-v1",
        "qc_sha256": receipt["qc_sha256"],
        "prominence_audit_sha256": audit["audit_sha256"],
        "adjudication_method": "manual_ephys_only_review",
        "reviewer": "",
        "behavior_fields_reviewed": False,
        "yaw_reviewed": False,
        "navigation_performance_used": False,
        "figure3c_statistic_reviewed": False,
        "allowed_selection_basis": list(_ALLOWED_SELECTION_BASIS),
        "decisions": decisions,
    }
    _atomic_write_json(out_dir / "threshold-decisions-template.json", decision_template)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic neural-only DNa02 threshold QC without selecting thresholds "
            "or loading behavior"
        )
    )
    parser.add_argument("sources", nargs="+", help="The four exact authenticated DNa02 MAT files")
    parser.add_argument("--prominence-audit", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--candidate-quantiles",
        nargs="+",
        type=float,
        default=list(_DEFAULT_CANDIDATE_QUANTILES),
        help="Prominence quantiles already present in the audit receipt",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    quantiles = tuple(float(value) for value in args.candidate_quantiles)
    if len(quantiles) != len(set(quantiles)) or not quantiles:
        raise ValueError("candidate quantiles must be non-empty and unique")
    result = adjudicate_sources(
        args.sources,
        prominence_audit_path=args.prominence_audit,
        output_dir=args.out_dir,
        contract_path=args.contract,
        evidence_path=args.evidence,
        candidate_quantiles=quantiles,
        write_plots=not args.no_plots,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
