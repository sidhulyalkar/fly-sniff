from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fly_video_neural.development_protocol import run_development_protocol
from fly_video_neural.session_benchmark import SessionBenchmarkBatch, build_session_split_lock


def _write_batches(root: Path) -> tuple[list[str], dict]:
    paths: list[str] = []
    chunks: list[SessionBenchmarkBatch] = []
    for animal_index in range(4):
        animal = f"fly{animal_index}"
        for session_index in range(3):
            session = f"{animal}_{session_index + 1:03d}"
            sample_ids = []
            features = []
            targets = []
            for index in range(8):
                sample_ids.append(f"{session}:{index * 50}-{index * 50 + 350}")
                features.append([index, index * index, animal_index + session_index * 0.01])
                targets.append([2 * index + animal_index, -index + session_index * 0.02])
            path = root / f"{session}.npz"
            np.savez_compressed(
                path,
                sample_ids=np.asarray(sample_ids),
                animal_ids=np.asarray([animal] * 8),
                session_ids=np.asarray([session] * 8),
                features=np.asarray(features, dtype=float),
                targets=np.asarray(targets, dtype=float),
            )
            paths.append(str(path))
            chunks.append(
                SessionBenchmarkBatch(
                    sample_ids=np.asarray(sample_ids),
                    animal_ids=np.asarray([animal] * 8),
                    session_ids=np.asarray([session] * 8),
                    features=np.asarray(features, dtype=float),
                    targets=np.asarray(targets, dtype=float),
                )
            )
    combined = SessionBenchmarkBatch(
        sample_ids=np.concatenate([chunk.sample_ids for chunk in chunks]),
        animal_ids=np.concatenate([chunk.animal_ids for chunk in chunks]),
        session_ids=np.concatenate([chunk.session_ids for chunk in chunks]),
        features=np.concatenate([chunk.features for chunk in chunks]),
        targets=np.concatenate([chunk.targets for chunk in chunks]),
    )
    from fly_video_neural.session_benchmark import load_session_batches

    _, sources = load_session_batches(paths)
    lock = build_session_split_lock(combined, source_batches=sources)
    return paths, lock


def test_development_protocol_never_emits_test_metrics(tmp_path: Path):
    batches, lock = _write_batches(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(lock))
    package = Path(__file__).resolve().parents[1]
    report = run_development_protocol(
        split,
        batches,
        tmp_path / "development",
        qc_config_path=package / "configs" / "data_qc_v1.json",
        acceptance_config_path=package / "configs" / "validation_acceptance_v1.json",
    )
    assert report["test_consumption_capability"] is False
    assert report["test_metrics_present"] is False
    aligned = json.loads((tmp_path / "development" / "aligned-ridge-development.json").read_text())
    null = json.loads((tmp_path / "development" / "temporal-null-development.json").read_text())
    assert aligned["test_status"] == "locked_not_consumed"
    assert null["test_status"] == "locked_not_consumed"
    assert all(row["test_metrics"] is None for row in aligned["animals"])
    assert all(row["test_metrics"] is None for row in null["animals"])
