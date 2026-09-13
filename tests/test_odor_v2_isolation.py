from __future__ import annotations

import ast
from pathlib import Path

SEALED_V1_MODULES = (
    "controllers.py",
    "graph.py",
    "training.py",
    "reproducible_training.py",
    "sealed_experiment.py",
    "trained_qualification.py",
    "trained_final.py",
    "qualified_eval.py",
)
V2_MODULES = {
    "fly_sniff.odor_motion",
    "fly_sniff.odor_events",
    "fly_sniff.odor_motion_visual",
    "fly_sniff.odor_motion_showcase",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            module = node.module
            if node.level:
                module = f"fly_sniff.{module}"
            imports.add(module)
    return imports


def test_sealed_v1_execution_modules_do_not_import_v2_sensory_analysis():
    source = Path(__file__).resolve().parents[1] / "src" / "fly_sniff"
    violations: dict[str, list[str]] = {}

    for filename in SEALED_V1_MODULES:
        path = source / filename
        assert path.is_file(), f"sealed-v1 boundary file disappeared: {filename}"
        forbidden = sorted(_imports(path) & V2_MODULES)
        if forbidden:
            violations[filename] = forbidden

    assert not violations, (
        "sealed v1 imported the v2 odor-timing analysis lane; create a separately versioned "
        f"functional protocol instead of silently changing v1: {violations}"
    )
