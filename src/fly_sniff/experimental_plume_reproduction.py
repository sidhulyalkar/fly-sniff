from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .experimental_plume import file_sha256, load_experimental_plume_config
from .experimental_plume_archive import inspect_experimental_plume_archive
from .experimental_plume_cues import crosswind_gradient, published_motion_cue

REPRODUCTION_PROTOCOL = "experimental-plume-cue-reproduction-v3"


class _RunningMoments:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
        sample = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = sample - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (sample - self.mean)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("cannot finish empty running moments")
        variance = self.m2 / self.count
        return self.mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def compute_cue_moments(
    frames: h5py.Dataset,
    *,
    start: int = 0,
    stop: int | None = None,
    intensity_scale: float = 1.0,
) -> dict[str, np.ndarray | int]:
    if frames.ndim != 3:
        raise ValueError(f"expected (time,y,x) plume frames, got shape {frames.shape}")
    stop = int(frames.shape[0]) if stop is None else int(stop)
    start = int(start)
    if start < 0 or stop > frames.shape[0] or stop - start < 3:
        raise ValueError("cue reproduction requires at least three in-range frames")

    def frame(index: int) -> np.ndarray:
        return np.asarray(frames[index], dtype=np.float32) * np.float32(intensity_scale)

    first = frame(start)
    gradient_moments = _RunningMoments(first.shape)
    motion_moments = _RunningMoments(first.shape)
    previous = first
    current = frame(start + 1)
    gradient_moments.update(crosswind_gradient(previous))
    gradient_moments.update(crosswind_gradient(current))
    for index in range(start + 2, stop):
        following = frame(index)
        gradient_moments.update(crosswind_gradient(following))
        motion_moments.update(published_motion_cue(previous, current, following))
        previous, current = current, following
    gradient_mean, gradient_std = gradient_moments.finish()
    motion_mean, motion_std = motion_moments.finish()
    return {
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "gradient_samples": gradient_moments.count,
        "motion_samples": motion_moments.count,
    }


def reproduce_complex_cues(
    source_file: str | Path,
    output_npz: str | Path,
    receipt_json: str | Path,
    document: dict | None = None,
) -> dict:
    document = load_experimental_plume_config() if document is None else document
    archive_report = inspect_experimental_plume_archive(source_file, "complex", document)
    source = document["complex"]
    dataset_path = source["dataset"]
    with h5py.File(source_file, "r") as archive:
        summary = compute_cue_moments(
            archive[dataset_path],
            start=0,
            stop=int(source["expected_shape"][0]),
            intensity_scale=float(source["intensity_scale"]),
        )

    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        gradient_mean=summary["gradient_mean"],
        gradient_std=summary["gradient_std"],
        motion_mean=summary["motion_mean"],
        motion_std=summary["motion_std"],
    )
    receipt = {
        "schema_version": 1,
        "protocol": REPRODUCTION_PROTOCOL,
        "plume": "complex",
        "status": "computed_pending_reference_comparison",
        "source_sha256": archive_report["source_sha256"],
        "archive_report_sha256": archive_report["report_sha256"],
        "archive_status": archive_report["status"],
        "native_time_basis": archive_report["native_time_basis"],
        "dataset_path": dataset_path,
        "dataset_shape": archive_report["source_dataset"]["shape"],
        "frame_range": [0, int(source["expected_shape"][0])],
        "intensity_scale": float(source["intensity_scale"]),
        "gradient_samples": int(summary["gradient_samples"]),
        "motion_samples": int(summary["motion_samples"]),
        "gradient_equation": "centered dI/dy with one-sided y boundaries",
        "motion_equation": "-(dI/dy)(dI/dt) with centered one-frame temporal difference",
        "map_spatial_smoothing_sigma_px": 0.0,
        "output_npz_sha256": file_sha256(output_path),
        "reference_comparison_status": "not_run_no_reference_summary_supplied",
        "controller_access": False,
        "navigation_performance_used": False,
        "sensory_timing_claim_allowed": bool(archive_report["native_sensory_timing_allowed"]),
        "claim_boundary": (
            "This artifact reproduces published plume cue fields from archived frames. "
            "It is not yet proof of numerical agreement with the authors' generated summary and is not a navigation result."
        ),
    }
    receipt_path = Path(receipt_json)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute published complex-plume cue moments")
    parser.add_argument("--config")
    parser.add_argument("--plume", choices=("complex",), default="complex")
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    document = load_experimental_plume_config(args.config)
    receipt = reproduce_complex_cues(args.source_file, args.output, args.receipt, document)
    print(args.output)
    print(args.receipt)
    print(receipt["status"])


if __name__ == "__main__":
    main()
