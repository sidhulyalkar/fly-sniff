import copy
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fly_sniff.cinematic_showcase import render_cinematic_showcase, validate_scene_binding
from fly_sniff.freeze import make_seed_split
from fly_sniff.graph import GraphBundle
from fly_sniff.neural_scene import build_neural_scene, write_neural_scene
from fly_sniff.plume import concentration_from_snapshot
from fly_sniff.recorded_showcase import reveal_text
from fly_sniff.recording import build_recording, load_recording, write_recording
from fly_sniff.scientific_view import density_grid, trace_arrays
from fly_sniff.staged_trace import run_staged_trace


def test_seed_sampling_avoids_full_population_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("allocating full seed population")
    monkeypatch.setattr(np, "arange", forbidden)
    heldout, ood = make_seed_split(7, 1000, 400)
    assert (heldout, ood) == make_seed_split(7, 1000, 400)
    assert len(set(heldout + ood)) == 1400
    assert min(heldout + ood) >= 1 and max(heldout + ood) < 2_000_000_000


def test_integer_sampling_matches_legacy_array_indexing():
    # Same NumPy sampling method, compared on a tractable population.
    for seed in (1, 7, 13013):
        legacy = np.random.default_rng(seed).choice(np.arange(1, 10000), 1400, replace=False)
        indices = np.random.default_rng(seed).choice(9999, 1400, replace=False) + 1
        np.testing.assert_array_equal(legacy, indices)


def test_density_raster_matches_model_at_grid_points():
    snapshot = np.array([[0, 0, .1], [.3, .8, .5], [1, 1, .01]])
    xs, ys = np.linspace(-1, 2, 15), np.linspace(-1, 2, 13)
    grid = density_grid(snapshot, 1.7, xs, ys)
    expected = [[concentration_from_snapshot(snapshot, 1.7, x, y) for x in xs] for y in ys]
    np.testing.assert_allclose(grid, expected, atol=1e-12, rtol=1e-12)


def test_stopped_sensors_follow_plume_but_commands_stop():
    payload = build_recording(seed=13013, sim_seconds=20)["recording"]
    states = [f["agents"][0] for f in payload["frames"]]
    stopped = [s for s in states if s["done"]]
    assert len(stopped) > 5
    assert len({(s["x"], s["y"]) for s in stopped}) == 1
    assert len({s["observation"]["left_odor"] for s in stopped}) > 1
    assert all(not s["decision_valid"] and s["action"]["speed"] == 0 for s in stopped)
    trace = trace_arrays(payload, payload["controllers"][0]["label"])
    assert np.isnan(trace["turn"][[i for i, s in enumerate(states) if s["done"]]]).all()


def test_timed_reveal_never_claims_success():
    assert reveal_text({"found": False}, {"found": False}, reveal=True) == "SOURCE REVEAL • NOT FOUND"


def test_staged_trace_rejects_empty_config(tmp_path):
    with pytest.raises(ValueError, match="at least one stage"):
        run_staged_trace(pd.DataFrame(), pd.DataFrame(), {}, tmp_path)


def candidate_bundle():
    # Explicit synthetic fixture, never a public MaleCNS result.
    return GraphBundle(
        pd.DataFrame({"bodyId": [1, 2], "somaLocation": [[0, 0, 0], [10, 0, 1]]}),
        pd.DataFrame({"source": [1], "target": [2], "weight": [5.], "sign": [1]}),
        {"odor_left": [1], "steer_left": [2]},
        {"dataset": "synthetic-test", "qualification_status": "candidate"},
    )


def test_candidate_recording_binds_scene_and_never_promotes_claim(tmp_path):
    graph = candidate_bundle()
    bundle = build_recording(sim_seconds=.15, controller_names=("random",), candidate_graph=graph)
    payload = bundle["recording"]
    scene = build_neural_scene(graph)
    validate_scene_binding(payload, scene)
    assert "NOT A QUALIFIED" in payload["claim_boundary"]
    assert payload["frames"][0]["agents"][0]["neural_activity"]["cells"]
    # Uniform projection preserves the 10:1 anatomical aspect ratio.
    a, b = scene["nodes"]
    assert abs((b["screen_x"]-a["screen_x"])/(b["screen_y"]-a["screen_y"])) == pytest.approx(10)
    bad = copy.deepcopy(scene)
    bad["graph_sha256"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_scene_binding(payload, bad)
    path = write_recording(tmp_path / "run.json", bundle)
    scene_path = write_neural_scene(tmp_path / "scene.json", scene)
    render_cinematic_showcase(path, scene_path, tmp_path / "frame.png", seconds=1, fps=1)
    assert load_recording(path) == bundle  # rendering is read-only


def test_proxy_scientific_render_does_not_require_fake_anatomy(tmp_path, monkeypatch):
    bundle = build_recording(sim_seconds=.1)
    assert all("neural_activity" not in a for f in bundle["recording"]["frames"] for a in f["agents"])
    path = write_recording(tmp_path / "run.json", bundle)
    # Rendering must not advance simulator state.
    def forbidden(*args, **kwargs):
        raise AssertionError("renderer ran a simulation")
    monkeypatch.setattr("fly_sniff.env.FlySniffEnv.step", forbidden)
    render_cinematic_showcase(path, None, tmp_path / "frame.png", seconds=1, fps=1)
    assert load_recording(path) == bundle


@pytest.mark.parametrize("command", ["fly-sniff-record", "fly-sniff-staged-trace", "fly-sniff-cinematic"])
def test_installed_console_commands(command, tmp_path):
    # Run outside the checkout so pytest's source-path override cannot hide packaging errors.
    from pathlib import Path
    executable = Path(sys.executable).parent / command
    result = subprocess.run([str(executable), "--help"], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_staged_cli_processes_independent_stages_and_strict_failure(tmp_path):
    import json
    from pathlib import Path
    annotations = tmp_path / "annotations.feather"
    weights = tmp_path / "weights.feather"
    pd.DataFrame({"bodyId": [1, 2], "type": ["ORN_demo", "DNa02_demo"]}).to_feather(annotations)
    pd.DataFrame({"source": [1], "target": [2], "weight": [10]}).to_feather(weights)
    stage = {"name": "route", "source": ["^ORN"], "target": ["^DNa02"],
             "max_hops": 2, "min_weight": 5, "fanout": 10}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"dataset": "synthetic-test", "stages": [stage]}))
    output = tmp_path / "trace"
    executable = Path(sys.executable).parent / "fly-sniff-staged-trace"
    command = [str(executable), str(annotations), str(weights), "--config", str(config),
               "--output", str(output), "--strict"]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert (output / "all_stage_edges.parquet").exists()
    stage["source"] = ["^MISSING"]
    config.write_text(json.dumps({"dataset": "synthetic-test", "stages": [stage]}))
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert second.returncode == 2, second.stderr
    assert not (output / "all_stage_edges.parquet").exists()
    summary = json.loads((output / "staged_trace_report.json").read_text())
    assert summary["stages"][0]["status"] == "empty_source_seed"


def test_scene_integrity_rejects_modified_geometry(tmp_path):
    import json

    from fly_sniff.cinematic_showcase import load_neural_scene
    path = write_neural_scene(tmp_path / "scene.json", build_neural_scene(candidate_bundle()))
    scene = json.loads(path.read_text())
    scene["nodes"][0]["screen_x"] += .1
    path.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="SHA-256"):
        load_neural_scene(path)
