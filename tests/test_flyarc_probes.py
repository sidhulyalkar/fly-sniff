from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd

from fly_sniff.flyarc_marathon import MarathonConfig, build_schedule, projection_seed
from fly_sniff.flyarc_probes import (
    ProbeConfig,
    load_shared_trajectory,
    run_probe,
)
from fly_sniff.graph import GraphBundle


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle() -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "target": [2, 3, 4, 5, 6, 1, 4, 5, 6, 1, 2, 3],
            "weight": [9, 8, 7, 6, 5, 4, 7, 7, 6, 6, 5, 5],
            "sign": [1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1],
        }
    )
    return GraphBundle(nodes=nodes, edges=edges, roles={}, manifest=None)


def _make_shared_run(tmp_path):
    root = tmp_path / "source"
    intact = root / "intact"
    rewire = root / "rewire"
    intact.mkdir(parents=True)
    rewire.mkdir(parents=True)

    frames = []
    shapes = []
    for index in range(40):
        frame = np.zeros((4, 4), dtype=np.uint8)
        frame[index % 4, (index // 2) % 4] = (index % 5) + 1
        frames.append(frame)
        shapes.append(frame.shape)
    packed = np.stack(frames)
    shapes_array = np.asarray(shapes, dtype=np.int16)
    np.savez_compressed(intact / "frames.npz", frames=packed, shapes=shapes_array)
    shutil.copy2(intact / "frames.npz", rewire / "frames.npz")

    # Frame 20 is a reset observation with no action row.
    rows = []
    step = 0
    for observation_index in list(range(1, 20)) + list(range(21, 40)):
        state = "GAME_OVER" if observation_index == 19 else "NOT_FINISHED"
        rows.append(
            {
                "step": step,
                "observation_index": observation_index,
                "state": state,
            }
        )
        step += 1

    for variant in ("intact", "rewire"):
        with (root / variant / "steps.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    receipt = {
        "experiment": "flyarc-v1",
        "paired_version_check": True,
        "files": {
            "intact/frames.npz": _sha(intact / "frames.npz"),
            "intact/steps.jsonl": _sha(intact / "steps.jsonl"),
            "rewire/frames.npz": _sha(rewire / "frames.npz"),
            "rewire/steps.jsonl": _sha(rewire / "steps.jsonl"),
        },
    }
    (root / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return root


def test_shared_trajectory_reconstructs_reset_segments(tmp_path):
    root = _make_shared_run(tmp_path)
    trajectory = load_shared_trajectory(root)

    assert trajectory.reset_indices == (0, 20)
    assert trajectory.segment_ids.tolist()[:3] == [0, 0, 0]
    assert trajectory.segment_ids.tolist()[19:22] == [0, 1, 1]
    assert trajectory.source_variant == "intact"


def test_probe_memory_outputs_are_finite_for_small_fixture(tmp_path):
    root = _make_shared_run(tmp_path)
    trajectory = load_shared_trajectory(root)
    config = ProbeConfig(
        max_nodes=6,
        pool=2,
        input_fanout=2,
        target_sketch_dim=8,
        ridge_alpha=0.2,
        lags=(1, 2),
        health_neurons=4,
        blocked_folds=2,
        purge=1,
    )

    leak = run_probe(_bundle(), trajectory, variant="leak_only", config=config)
    stateless = run_probe(_bundle(), trajectory, variant="stateless", config=config)

    assert leak["trajectory_segments"] == 2
    assert stateless["trajectory_segments"] == 2
    assert np.isfinite(leak["memory"]["1"]["reservoir_r2"])
    assert np.isfinite(leak["memory"]["1"]["current_input_r2"])
    assert np.isfinite(stateless["future_prediction"]["reservoir_r2"])
    assert leak["health"]["health_neuron_subsample"] == 4


def test_marathon_schedule_balances_projection_blocks_before_null_depth():
    config = MarathonConfig(
        hours=1,
        projection_count=2,
        rewires_per_projection=3,
        randoms_per_projection=2,
    )
    schedule = build_schedule(config)

    assert len(schedule) == 16
    assert [job.variant for job in schedule[:6]] == [
        "intact",
        "leak_only",
        "stateless",
        "intact",
        "leak_only",
        "stateless",
    ]
    assert schedule[0].projection_seed == projection_seed(config, 0)
    assert schedule[3].projection_seed == projection_seed(config, 1)

    rewires = [job for job in schedule if job.variant == "rewire"]
    randoms = [job for job in schedule if job.variant == "random"]
    assert len(rewires) == 6
    assert len(randoms) == 4
    assert len({job.job_id for job in schedule}) == len(schedule)
