from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .freeze import canonical_sha256

_DEFAULT_QUANTILES = (0.995, 0.999, 0.9995, 0.9999)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _validated_qc(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "fly-sniff-dna02-threshold-adjudication-qc-v1":
        raise ValueError("unexpected DNa02 threshold-QC schema")
    observed = str(payload.get("qc_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("qc_sha256", None)
    if observed != canonical_sha256(unhashed):
        raise ValueError("threshold-QC hash mismatch")
    for field in (
        "behavior_fields_loaded",
        "yaw_loaded",
        "navigation_performance_used",
        "figure3c_statistic_computed",
        "automatic_threshold_selection",
        "thresholds_frozen",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"threshold-QC violates sealed review boundary: {field}")
    return payload


def _channel_map(qc: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
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


def _candidate_map(channel: dict[str, Any]) -> dict[float, dict[str, Any]]:
    candidates = {
        float(item["prominence_quantile"]): item for item in channel.get("candidate_qc", [])
    }
    if len(candidates) != len(channel.get("candidate_qc", [])):
        raise ValueError("duplicate prominence quantile in channel QC")
    return candidates


def _select_candidates(
    channel: dict[str, Any], quantiles: tuple[float, ...]
) -> list[dict[str, Any]]:
    candidates = _candidate_map(channel)
    selected = []
    for quantile in quantiles:
        if quantile not in candidates:
            raise ValueError(f"channel QC missing q={quantile:g}")
        selected.append(candidates[quantile])
    return selected


def _plot_waveforms(
    candidates: list[dict[str, Any]], *, output_path: Path, title: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for candidate in candidates:
        quantile = float(candidate["prominence_quantile"])
        threshold = float(candidate["threshold"])
        waveform = candidate["waveform"]
        time_ms = np.asarray(waveform["time_ms"], dtype=float)
        median = np.asarray(waveform["median_waveform"], dtype=float)
        mad = np.asarray(waveform["mad_waveform"], dtype=float)
        if time_ms.size == 0 or median.size != time_ms.size or mad.size != time_ms.size:
            raise ValueError(f"incomplete waveform summary for q={quantile:g}")
        label = (
            f"q={quantile:g}, thr={threshold:.4g}, "
            f"corr={waveform['median_template_correlation']:.3f}"
        )
        line = axes[0].plot(time_ms, median, label=label)[0]
        axes[0].fill_between(time_ms, median - mad, median + mad, alpha=0.12)

        peak_amplitude = float(waveform["median_peak_amplitude"])
        scale = abs(peak_amplitude)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid median peak amplitude for q={quantile:g}")
        normalized = median / scale
        normalized_mad = mad / scale
        axes[1].plot(time_ms, normalized, label=label, color=line.get_color())
        axes[1].fill_between(
            time_ms,
            normalized - normalized_mad,
            normalized + normalized_mad,
            alpha=0.12,
            color=line.get_color(),
        )

    axes[0].set_ylabel("baseline-subtracted voltage")
    axes[0].set_title("absolute median waveform ± MAD")
    axes[1].set_ylabel("median / |median peak|")
    axes[1].set_xlabel("time from detected peak (ms)")
    axes[1].set_title("shape-normalized median waveform ± MAD")
    for axis in axes:
        axis.axvline(0.0, linewidth=0.8, alpha=0.4)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle(title + " | ephys-only high-threshold waveform review")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def render_waveform_review(
    qc_path: str | Path,
    *,
    output_dir: str | Path,
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES,
) -> dict[str, Any]:
    if len(quantiles) < 2 or len(set(quantiles)) != len(quantiles):
        raise ValueError("waveform review requires at least two unique candidate quantiles")
    qc = _validated_qc(qc_path)
    channels = _channel_map(qc)
    out_dir = Path(output_dir)
    plots: list[dict[str, Any]] = []

    for (alias, side), channel in sorted(channels.items()):
        selected = _select_candidates(channel, quantiles)
        path = out_dir / "waveforms" / f"{alias}-{side}-high-threshold-waveforms.png"
        _plot_waveforms(selected, output_path=path, title=f"{alias} {side}")
        plots.append(
            {
                "fly_alias": alias,
                "soma_side": side,
                "source_field": channel["source_field"],
                "source_sha256": channel["source_sha256"],
                "plot": str(path.relative_to(out_dir)),
                "candidate_quantiles": list(quantiles),
            }
        )

    payload: dict[str, Any] = {
        "schema": "fly-sniff-dna02-threshold-waveform-review-v1",
        "status": "HIGH_THRESHOLD_WAVEFORMS_RENDERED_NO_SELECTION",
        "qc_sha256": qc["qc_sha256"],
        "prominence_audit_sha256": qc["prominence_audit_sha256"],
        "candidate_quantiles": list(quantiles),
        "behavior_fields_loaded": False,
        "yaw_loaded": False,
        "figure3c_statistic_computed": False,
        "navigation_performance_used": False,
        "automatic_threshold_selection": False,
        "thresholds_frozen": False,
        "plots": plots,
        "next_allowed_action": (
            "Review these waveform-family panels together with exact global raw-window overlays at "
            "q0.995/q0.999/q0.9995. Freeze a threshold only if ephys-only evidence is defensible."
        ),
    }
    payload["review_sha256"] = canonical_sha256(payload)
    _atomic_write_json(out_dir / "threshold-waveform-review-v1.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render DNa02 high-threshold waveform families from frozen ephys-only QC"
    )
    parser.add_argument("--qc", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--candidate-quantiles",
        type=float,
        nargs="+",
        default=list(_DEFAULT_QUANTILES),
    )
    args = parser.parse_args(argv)
    result = render_waveform_review(
        args.qc,
        output_dir=args.out_dir,
        quantiles=tuple(args.candidate_quantiles),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
