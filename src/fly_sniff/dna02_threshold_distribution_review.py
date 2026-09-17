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
from scipy.signal import find_peaks

from .dna02_threshold_adjudication import (
    _audit_channel_map,
    _candidate_thresholds,
    _deterministic_subsample,
    _load_channel,
    _source_map,
    _validated_receipt,
)
from .freeze import canonical_sha256

_DEFAULT_QUANTILES = (0.995, 0.999, 0.9995)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_waves(
    trace: np.ndarray,
    peaks: np.ndarray,
    *,
    fs: float,
    seed_text: str,
    cap: int = 5000,
    pre_ms: float = 1.0,
    post_ms: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = max(1, int(round(pre_ms * fs / 1000.0)))
    post = max(1, int(round(post_ms * fs / 1000.0)))
    margin = max(pre, post) + 1
    sampled = _deterministic_subsample(
        np.asarray(peaks, dtype=np.int64),
        cap=cap,
        seed_text=seed_text,
        margin=margin,
        n_samples=trace.size,
    )
    offsets = np.arange(-pre, post + 1, dtype=np.int64)
    if sampled.size == 0:
        return sampled, offsets, np.empty((0, offsets.size), dtype=float)
    waves = trace[sampled[:, None] + offsets[None, :]].astype(np.float64, copy=False)
    baseline_width = max(1, pre // 2)
    waves = waves - np.median(waves[:, :baseline_width], axis=1, keepdims=True)
    return sampled, offsets, waves


def _template_correlations(waves: np.ndarray, template: np.ndarray) -> np.ndarray:
    if waves.ndim != 2 or template.ndim != 1 or waves.shape[1] != template.size:
        raise ValueError("waveforms and template have incompatible shapes")
    if waves.shape[0] == 0:
        return np.array([], dtype=float)
    centered_template = template - np.mean(template)
    template_norm = float(np.linalg.norm(centered_template))
    if not np.isfinite(template_norm) or template_norm <= 0:
        raise ValueError("anchor template has zero or invalid norm")
    centered = waves - np.mean(waves, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    correlations = np.full(waves.shape[0], np.nan, dtype=float)
    valid = norms > 0
    correlations[valid] = (
        centered[valid] @ centered_template / (norms[valid] * template_norm)
    )
    return correlations


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: None for key in ("p05", "p10", "p25", "p50", "p75", "p90", "p95")}
    levels = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    return {
        key: float(np.quantile(finite, level))
        for key, level in zip(
            ("p05", "p10", "p25", "p50", "p75", "p90", "p95"), levels, strict=True
        )
    }


def _band_masks(
    prominences: np.ndarray, thresholds: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, mid, high = thresholds
    if not (0 < low < mid < high):
        raise ValueError("nested prominence thresholds must be positive and strictly increasing")
    prominences = np.asarray(prominences, dtype=float)
    return (
        (prominences >= low) & (prominences < mid),
        (prominences >= mid) & (prominences < high),
        prominences >= high,
    )


def _band_summary(
    waves: np.ndarray,
    correlations: np.ndarray,
    *,
    peak_offset: int,
    total_event_count: int,
) -> dict[str, Any]:
    amplitudes = waves[:, peak_offset] if waves.shape[0] else np.array([], dtype=float)
    finite_corr = correlations[np.isfinite(correlations)]
    return {
        "total_event_count": int(total_event_count),
        "sampled_event_count": int(waves.shape[0]),
        "anchor_correlation_quantiles": _quantiles(finite_corr),
        "anchor_correlation_fraction_ge_0p80": (
            float(np.mean(finite_corr >= 0.80)) if finite_corr.size else None
        ),
        "anchor_correlation_fraction_ge_0p90": (
            float(np.mean(finite_corr >= 0.90)) if finite_corr.size else None
        ),
        "anchor_correlation_fraction_ge_0p95": (
            float(np.mean(finite_corr >= 0.95)) if finite_corr.size else None
        ),
        "peak_amplitude_quantiles": _quantiles(amplitudes),
        "positive_peak_amplitude_fraction": (
            float(np.mean(amplitudes > 0)) if amplitudes.size else None
        ),
    }


def _shape_normalized(waves: np.ndarray, peak_offset: int) -> np.ndarray:
    if waves.shape[0] == 0:
        return waves.copy()
    scales = np.abs(waves[:, peak_offset])
    valid = np.isfinite(scales) & (scales > 1e-12)
    result = np.full_like(waves, np.nan, dtype=float)
    result[valid] = waves[valid] / scales[valid, None]
    return result


def _plot_distribution_review(
    *,
    time_ms: np.ndarray,
    bands: list[dict[str, Any]],
    anchor_template: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    all_prominence: list[np.ndarray] = []
    all_correlation: list[np.ndarray] = []
    for index, band in enumerate(bands):
        waves = np.asarray(band["waves"], dtype=float)
        corr = np.asarray(band["correlations"], dtype=float)
        prom = np.asarray(band["sampled_prominences"], dtype=float)
        normalized = _shape_normalized(waves, int(band["peak_offset"]))
        if normalized.shape[0]:
            draw_count = min(120, normalized.shape[0])
            chosen = np.linspace(0, normalized.shape[0] - 1, draw_count, dtype=int)
            for row in normalized[chosen]:
                if np.all(np.isfinite(row)):
                    axes[0, index].plot(time_ms, row, linewidth=0.5, alpha=0.08)
        anchor_scale = abs(float(anchor_template[int(band["peak_offset"])]))
        if anchor_scale > 0:
            axes[0, index].plot(
                time_ms,
                anchor_template / anchor_scale,
                linewidth=2.0,
                label="q0.9995+ anchor",
            )
        axes[0, index].set_title(
            f"{band['label']} | n={band['total_event_count']}"
        )
        axes[0, index].set_xlabel("time from peak (ms)")
        axes[0, index].set_ylabel("waveform / |peak|")
        axes[0, index].axvline(0.0, linewidth=0.8, alpha=0.4)
        axes[0, index].grid(alpha=0.2)
        axes[0, index].legend(fontsize=8)
        if corr.size:
            finite = corr[np.isfinite(corr)]
            if finite.size:
                axes[1, 0].hist(finite, bins=40, alpha=0.35, density=True, label=band["label"])
        if waves.shape[0]:
            amplitudes = waves[:, int(band["peak_offset"])]
            finite_amp = amplitudes[np.isfinite(amplitudes)]
            if finite_amp.size:
                axes[1, 1].hist(
                    finite_amp,
                    bins=40,
                    alpha=0.35,
                    density=True,
                    label=band["label"],
                )
        if prom.size and corr.size:
            valid = np.isfinite(prom) & np.isfinite(corr)
            axes[1, 2].scatter(
                prom[valid], corr[valid], s=5, alpha=0.20, label=band["label"]
            )
        all_prominence.append(prom)
        all_correlation.append(corr)

    axes[1, 0].set_title("individual waveform correlation to high-confidence anchor")
    axes[1, 0].set_xlabel("Pearson correlation")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].set_title("baseline-subtracted peak amplitude")
    axes[1, 1].set_xlabel("peak amplitude")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)

    axes[1, 2].set_title("prominence vs waveform correlation")
    axes[1, 2].set_xlabel("prominence")
    axes[1, 2].set_ylabel("correlation to anchor")
    axes[1, 2].set_ylim(-1.05, 1.05)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.2)

    fig.suptitle(title + " | ephys-only individual-event distribution review")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_distribution_review(
    source_paths: list[str | Path],
    *,
    prominence_audit_path: str | Path,
    output_dir: str | Path,
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES,
) -> dict[str, Any]:
    if quantiles != _DEFAULT_QUANTILES:
        raise ValueError(
            "v1 distribution review is frozen to q0.995/q0.999/q0.9995"
        )
    audit = _validated_receipt(Path(prominence_audit_path))
    audit_channels = _audit_channel_map(audit)
    expected_names = {item["filename"] for item in audit_channels.values()}
    sources = _source_map(source_paths, expected_names)
    out_dir = Path(output_dir)
    file_results: dict[str, dict[str, Any]] = {}

    for (alias, side), audit_channel in sorted(audit_channels.items()):
        filename = str(audit_channel["filename"])
        path = sources[filename]
        observed_sha = _sha256_file(path)
        if observed_sha != str(audit_channel["source_sha256"]):
            raise ValueError(f"source SHA-256 mismatch for {filename}")
        field = "ephys_A" if side == "L" else "ephys_B"
        fs, trace = _load_channel(path, field)
        candidates = _candidate_thresholds(audit_channel, quantiles)
        thresholds = tuple(float(item[1]) for item in candidates)
        base_threshold = thresholds[0]
        peaks, properties = find_peaks(trace, prominence=(base_threshold, None))
        prominences = np.asarray(properties["prominences"], dtype=float)
        masks = _band_masks(prominences, thresholds)

        high_peaks = np.asarray(peaks[masks[2]], dtype=np.int64)
        _, offsets, anchor_waves = _extract_waves(
            trace,
            high_peaks,
            fs=fs,
            seed_text=f"{alias}|{side}|anchor|q0.9995",
            cap=2000,
        )
        if anchor_waves.shape[0] < 50:
            raise ValueError(f"insufficient q0.9995+ anchor events for {alias} {side}")
        anchor_template = np.median(anchor_waves, axis=0)
        peak_offset = int(round(1.0 * fs / 1000.0))
        time_ms = offsets / fs * 1000.0

        labels = ("q0.995–q0.999", "q0.999–q0.9995", "q0.9995+")
        bands_for_plot: list[dict[str, Any]] = []
        band_payloads: list[dict[str, Any]] = []
        for band_index, (label, mask) in enumerate(zip(labels, masks, strict=True)):
            band_peaks_all = np.asarray(peaks[mask], dtype=np.int64)
            band_prom_all = np.asarray(prominences[mask], dtype=float)
            sampled_peaks, _, waves = _extract_waves(
                trace,
                band_peaks_all,
                fs=fs,
                seed_text=f"{alias}|{side}|band={band_index}|{label}",
                cap=5000,
            )
            if sampled_peaks.size:
                positions = np.searchsorted(band_peaks_all, sampled_peaks)
                sampled_prom = band_prom_all[positions]
            else:
                sampled_prom = np.array([], dtype=float)
            correlations = _template_correlations(waves, anchor_template)
            summary = _band_summary(
                waves,
                correlations,
                peak_offset=peak_offset,
                total_event_count=int(band_peaks_all.size),
            )
            band_payloads.append(
                {
                    "label": label,
                    "prominence_lower_inclusive": thresholds[band_index],
                    "prominence_upper_exclusive": (
                        thresholds[band_index + 1] if band_index < 2 else None
                    ),
                    **summary,
                }
            )
            bands_for_plot.append(
                {
                    "label": label,
                    "total_event_count": int(band_peaks_all.size),
                    "waves": waves,
                    "correlations": correlations,
                    "sampled_prominences": sampled_prom,
                    "peak_offset": peak_offset,
                }
            )

        plot_path = out_dir / "plots" / f"{alias}-{side}-waveform-distributions.png"
        _plot_distribution_review(
            time_ms=time_ms,
            bands=bands_for_plot,
            anchor_template=anchor_template,
            output_path=plot_path,
            title=f"{alias} {side}",
        )

        file_entry = file_results.setdefault(
            alias,
            {
                "fly_alias": alias,
                "filename": filename,
                "source_sha256": observed_sha,
                "ephys_sampling_rate_hz": float(fs),
                "channels": [],
            },
        )
        file_entry["channels"].append(
            {
                "soma_side": side,
                "source_field": field,
                "candidate_quantiles": list(quantiles),
                "candidate_thresholds": list(thresholds),
                "anchor_quantile": 0.9995,
                "anchor_sampled_event_count": int(anchor_waves.shape[0]),
                "bands": band_payloads,
                "plot": str(plot_path.relative_to(out_dir)),
            }
        )

    payload: dict[str, Any] = {
        "schema": "fly-sniff-dna02-threshold-distribution-review-v1",
        "status": "INDIVIDUAL_WAVEFORM_DISTRIBUTIONS_RENDERED_NO_SELECTION",
        "prominence_audit_sha256": audit["audit_sha256"],
        "candidate_quantiles": list(quantiles),
        "anchor_rule": (
            "Within each fly/side, build the reference waveform only from q0.9995+ events, "
            "then score deterministic individual-event samples from each nested prominence band."
        ),
        "behavior_fields_loaded": False,
        "yaw_loaded": False,
        "figure3c_statistic_computed": False,
        "navigation_performance_used": False,
        "automatic_threshold_selection": False,
        "thresholds_frozen": False,
        "files": [file_results[key] for key in sorted(file_results)],
        "next_allowed_action": (
            "Review whether lower-prominence bands retain the high-confidence spike waveform. "
            "Freeze thresholds only if individual-event ephys evidence is defensible; otherwise "
            "keep the channel blocked or refine the neural-only candidate grid."
        ),
    }
    payload["review_sha256"] = canonical_sha256(payload)
    _atomic_write_json(out_dir / "threshold-distribution-review-v1.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Review individual DNa02 waveform distributions across nested prominence bands "
            "without loading behavior or selecting thresholds"
        )
    )
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--prominence-audit", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    result = run_distribution_review(
        args.sources,
        prominence_audit_path=args.prominence_audit,
        output_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
