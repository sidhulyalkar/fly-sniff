from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from fly_video_neural.benchmark import load_benchmark, validate_benchmark


def test_shipped_benchmark_freezes_future_neural_task():
    path = Path(__file__).resolve().parents[1] / "configs" / "benchmark_v0.json"
    document = load_benchmark(path)
    assert document["input_history_s"] == 3.0
    assert document["prediction_horizon_s"] == 0.5
    assert document["split_unit"] == "animal_id"
    assert document["connectome_prior_status"] == "future_separately_versioned_lane"


def test_benchmark_rejects_test_tuning():
    path = Path(__file__).resolve().parents[1] / "configs" / "benchmark_v0.json"
    document = json.loads(path.read_text())
    document["test_animals_for_hyperparameter_selection"] = True
    with pytest.raises(ValueError, match="test animals"):
        validate_benchmark(document)


def test_research_package_does_not_import_fly_sniff():
    source = Path(__file__).resolve().parents[1] / "src" / "fly_video_neural"
    violations: list[str] = []
    for path in source.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(alias.name for alias in node.names if alias.name.startswith("fly_sniff"))
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fly_sniff"):
                violations.append(node.module)
    assert not violations, f"parallel research lane imported fly_sniff: {violations}"


def test_active_v1_generic_cli_has_no_direct_scoring_shortcut():
    cli = Path(__file__).resolve().parents[1] / "src" / "fly_video_neural" / "cli.py"
    source = cli.read_text()
    assert '"within-animal-ridge"' not in source
    assert "Historical v0 animal-held-out ridge utility" in source
