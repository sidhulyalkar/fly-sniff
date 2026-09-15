from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import load_safe_alignment, materialize_session_windows
from .mc2p import MC2PSession, build_manifest, discover_sessions
from .mc2p_legacy import convert_legacy_pickle, sha256_file
from .pose_neural import build_session_pose_neural_batch, load_pose3d
from .session_benchmark import build_session_split_lock, load_session_batches


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fresh_output_dir(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty preparation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _copy_safe_alignment(source: Path, destination: Path) -> dict[str, Any]:
    alignment = load_safe_alignment(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, alignment, allow_pickle=False)
    return {
        "schema_version": 1,
        "protocol": "mc2p-safe-array-copy-v1",
        "dataset_id": "mc2p_v1",
        "kind": "alignment",
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output_path": str(destination.resolve()),
        "output_sha256": sha256_file(destination),
        "output_shape": list(alignment.shape),
        "output_dtype": str(alignment.dtype),
    }


def _copy_safe_pose(source: Path, destination: Path) -> dict[str, Any]:
    pose = load_pose3d(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.asarray(pose, dtype=np.float32), allow_pickle=False)
    return {
        "schema_version": 1,
        "protocol": "mc2p-safe-array-copy-v1",
        "dataset_id": "mc2p_v1",
        "kind": "pose3d",
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output_path": str(destination.resolve()),
        "output_sha256": sha256_file(destination),
        "output_shape": list(pose.shape),
        "output_dtype": str(pose.dtype),
    }


def _prepare_alignment(
    session: MC2PSession,
    directory: Path,
    *,
    trust_upstream_pickle: bool,
) -> tuple[Path, dict[str, Any]]:
    source = Path(session.synchronization)
    destination = directory / "alignment.npy"
    receipt_path = directory / "alignment-conversion.json"
    if source.suffix.lower() == ".npy":
        receipt = _copy_safe_alignment(source, destination)
        receipt["receipt_sha256"] = _sha(receipt)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return destination, receipt
    if source.suffix.lower() in {".pkl", ".pickle"}:
        receipt = convert_legacy_pickle(
            source,
            destination,
            receipt_path,
            kind="alignment",
            trust_upstream_pickle=trust_upstream_pickle,
        )
        return destination, receipt
    raise ValueError(f"unsupported synchronization format for {session.session_id}: {source}")


def _prepare_pose(
    session: MC2PSession,
    directory: Path,
    *,
    trust_upstream_pickle: bool,
) -> tuple[Path, dict[str, Any]]:
    if session.pose is None:
        raise ValueError(f"session {session.session_id} has no pose artifact")
    source = Path(session.pose)
    destination = directory / "pose3d.npy"
    receipt_path = directory / "pose-conversion.json"
    if source.suffix.lower() == ".npy":
        receipt = _copy_safe_pose(source, destination)
        receipt["receipt_sha256"] = _sha(receipt)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return destination, receipt
    if source.suffix.lower() in {".pkl", ".pickle"}:
        receipt = convert_legacy_pickle(
            source,
            destination,
            receipt_path,
            kind="pose3d",
            trust_upstream_pickle=trust_upstream_pickle,
        )
        return destination, receipt
    raise ValueError(
        f"session {session.session_id} pose is not an auditable pose3d .npy or trusted pickle: {source}"
    )


def _select_dff(session: MC2PSession) -> tuple[Path, int, str]:
    if session.neural_dff_resized is not None:
        return Path(session.neural_dff_resized), 64, "resized_measured_dff_64x64"
    return Path(session.neural_dff), 128, "measured_dff_128x128"


def prepare_mc2p_v1(
    root: str | Path,
    output_dir: str | Path,
    *,
    trust_upstream_pickle: bool = False,
) -> dict[str, Any]:
    output = _fresh_output_dir(output_dir)
    manifest = build_manifest(root)
    manifest_path = output / "mc2p-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    batch_paths: list[Path] = []
    session_receipts: list[dict[str, Any]] = []
    for session in discover_sessions(root):
        session_dir = output / "sessions" / session.session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        alignment_path, alignment_receipt = _prepare_alignment(
            session,
            session_dir,
            trust_upstream_pickle=trust_upstream_pickle,
        )
        pose_path, pose_receipt = _prepare_pose(
            session,
            session_dir,
            trust_upstream_pickle=trust_upstream_pickle,
        )
        windows = materialize_session_windows(session.path, alignment_path)
        if windows["window_count"] < 2:
            raise ValueError(
                f"session {session.session_id} has only {windows['window_count']} v1 prediction windows"
            )
        windows_path = session_dir / "windows.json"
        windows_path.write_text(json.dumps(windows, indent=2, sort_keys=True) + "\n")
        dff_path, dff_side, dff_kind = _select_dff(session)
        batch_path = session_dir / "pose-neural-batch.npz"
        batch_receipt_path = session_dir / "pose-neural-batch.json"
        batch_receipt = build_session_pose_neural_batch(
            windows_path,
            pose_path,
            dff_path,
            batch_path,
            batch_receipt_path,
            dff_side=dff_side,
        )
        batch_paths.append(batch_path)
        session_receipts.append(
            {
                "animal_id": session.animal_id,
                "session_id": session.session_id,
                "alignment_conversion_sha256": alignment_receipt["receipt_sha256"],
                "pose_conversion_sha256": pose_receipt["receipt_sha256"],
                "windows_sha256": sha256_file(windows_path),
                "batch_receipt_sha256": batch_receipt["receipt_sha256"],
                "batch_sha256": sha256_file(batch_path),
                "dff_source_sha256": sha256_file(dff_path),
                "dff_target_kind": dff_kind,
            }
        )

    batch, source_batches = load_session_batches([str(path) for path in batch_paths])
    split_lock = build_session_split_lock(batch, source_batches=source_batches)
    split_path = output / "session-split-lock.json"
    split_path.write_text(json.dumps(split_lock, indent=2, sort_keys=True) + "\n")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "mc2p-v1-preparation-v1",
        "benchmark_id": "mc2p_future_neural_v1",
        "dataset_id": "mc2p_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "animal_count": manifest["animal_count"],
        "session_count": manifest["session_count"],
        "split_lock_sha256": split_lock["split_lock_sha256"],
        "sessions": sorted(session_receipts, key=lambda row: row["session_id"]),
        "models_fit": False,
        "test_data_consumed": False,
        "claim_boundary": (
            "This preparation run converts and packages measured MC2P data only. "
            "It fits no decoder and reports no performance result."
        ),
    }
    receipt["receipt_sha256"] = _sha(receipt)
    (output / "preparation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MC2P v1 measurement batches without fitting a model")
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-upstream-pickle", action="store_true")
    args = parser.parse_args()
    report = prepare_mc2p_v1(
        args.root,
        args.output,
        trust_upstream_pickle=args.trust_upstream_pickle,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "animals": report["animal_count"],
                "sessions": report["session_count"],
                "sha256": report["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
