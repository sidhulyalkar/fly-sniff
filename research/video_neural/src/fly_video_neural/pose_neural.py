from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

POSE_ROOT_GROUPS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (10, 11, 12, 13, 14, 15, 16, 17, 18),
    (19, 20, 21, 22, 23),
    (24, 25, 26, 27, 28),
    (29, 30, 31, 32, 33, 34, 35, 36, 37),
)
FEATURE_STATISTICS = (
    "mean",
    "std",
    "endpoint_delta",
    "mean_abs_velocity",
    "velocity_std",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pose3d(path: str | Path) -> np.ndarray:
    pose = np.load(path, mmap_mode="r", allow_pickle=False)
    if pose.ndim != 3 or pose.shape[1:] != (38, 3):
        raise ValueError(f"pose3d must have shape [frames, 38, 3], got {tuple(pose.shape)}")
    if not np.issubdtype(pose.dtype, np.number) or not np.isfinite(pose).all():
        raise ValueError("pose3d must contain finite numeric values")
    return pose


def root_relative_pose(pose: np.ndarray) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (38, 3):
        raise ValueError("pose must have shape [frames, 38, 3]")
    rooted = values.copy()
    for group in POSE_ROOT_GROUPS:
        rooted[:, group] -= values[:, [group[0]]]
    return rooted


def summarize_pose_window(pose_window: np.ndarray) -> np.ndarray:
    rooted = root_relative_pose(pose_window)
    if rooted.shape[0] < 2:
        raise ValueError("pose feature window requires at least two behavior frames")
    flat = rooted.reshape(rooted.shape[0], -1)
    velocity = np.diff(flat, axis=0)
    feature = np.concatenate(
        (
            flat.mean(axis=0),
            flat.std(axis=0),
            flat[-1] - flat[0],
            np.abs(velocity).mean(axis=0),
            velocity.std(axis=0),
        )
    )
    expected = 38 * 3 * len(FEATURE_STATISTICS)
    if feature.shape != (expected,):
        raise RuntimeError("unexpected pose feature dimension")
    return np.asarray(feature, dtype=np.float32)


def open_dff_memmap(path: str | Path, *, side: int) -> np.memmap:
    if side < 1:
        raise ValueError("dff side must be positive")
    dff_path = Path(path)
    frame_bytes = side * side * np.dtype("float32").itemsize
    size = dff_path.stat().st_size
    if size == 0 or size % frame_bytes:
        raise ValueError(
            f"dF/F file size {size} is not an exact number of float32 {side}x{side} frames"
        )
    frames = size // frame_bytes
    return np.memmap(dff_path, dtype="float32", mode="r", shape=(frames, side, side))


def _load_windows(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if document.get("dataset_id") != "mc2p_v1":
        raise ValueError("window manifest must be mc2p_v1")
    windows = document.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("window manifest must contain prediction windows")
    return document


def build_session_pose_neural_batch(
    windows_path: str | Path,
    pose3d_path: str | Path,
    dff_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path,
    *,
    dff_side: int,
) -> dict[str, Any]:
    document = _load_windows(windows_path)
    pose = load_pose3d(pose3d_path)
    dff = open_dff_memmap(dff_path, side=dff_side)

    sample_ids: list[str] = []
    animal_ids: list[str] = []
    session_ids: list[str] = []
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    observed_session: str | None = None
    observed_animal: str | None = None

    for row in document["windows"]:
        sample = row["sample"]
        session_id = str(sample["session_id"])
        animal_id = str(sample["animal_id"])
        observed_session = observed_session or session_id
        observed_animal = observed_animal or animal_id
        if session_id != observed_session or animal_id != observed_animal:
            raise ValueError("session batch manifest must contain exactly one animal/session")
        input_start, input_end = map(int, row["input_behavior_frames"])
        if input_start < 0 or input_end > len(pose) or input_end <= input_start:
            raise ValueError(f"pose input frames out of bounds for {sample['sample_id']}")
        neural_indices = np.asarray(row["target_neural_indices"], dtype=np.int64)
        if neural_indices.ndim != 1 or neural_indices.size == 0:
            raise ValueError(f"target neural indices missing for {sample['sample_id']}")
        if (neural_indices < 0).any() or int(neural_indices.max()) >= len(dff):
            raise ValueError(f"target neural indices out of bounds for {sample['sample_id']}")
        sample_ids.append(str(sample["sample_id"]))
        animal_ids.append(animal_id)
        session_ids.append(session_id)
        features.append(summarize_pose_window(pose[input_start:input_end]))
        target = np.asarray(dff[neural_indices].mean(axis=0), dtype=np.float32).reshape(-1)
        targets.append(target)

    x = np.stack(features)
    y = np.stack(targets)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=np.asarray(sample_ids),
        animal_ids=np.asarray(animal_ids),
        session_ids=np.asarray(session_ids),
        features=x,
        targets=y,
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mc2p-pose-neural-session-batch-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "dataset_id": "mc2p_v1",
        "animal_id": observed_animal,
        "session_id": observed_session,
        "sample_count": len(sample_ids),
        "windows_sha256": _sha256_file(windows_path),
        "pose3d_sha256": _sha256_file(pose3d_path),
        "dff_sha256": _sha256_file(dff_path),
        "output_sha256": _sha256_file(output),
        "feature_shape": list(x.shape),
        "target_shape": list(y.shape),
        "pose_feature_contract": {
            "root_groups": [list(group) for group in POSE_ROOT_GROUPS],
            "statistics": list(FEATURE_STATISTICS),
            "uses_input_behavior_frames_only": True,
        },
        "target_contract": {
            "operation": "mean_measured_dff_over_unique_future_neural_indices",
            "dff_side": dff_side,
            "evidence_class": "measured_neural_activity",
        },
        "claim_boundary": (
            "This artifact contains deterministic pose features and measured future dF/F targets "
            "for one animal/session. It is not a decoded result."
        ),
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    receipt_file = Path(receipt_path)
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    receipt_file.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
