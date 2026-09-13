from __future__ import annotations

import ast
from pathlib import Path

SEALED = {
    "controllers.py", "graph.py", "training.py", "reproducible_training.py",
    "sealed_experiment.py", "trained_qualification.py", "trained_final.py", "qualified_eval.py",
}
NEW_ANALYSIS = {
    "fly_sniff.odor_motion", "fly_sniff.odor_events", "fly_sniff.odor_event_cli",
    "fly_sniff.odor_sensitivity", "fly_sniff.odor_motion_visual",
    "fly_sniff.odor_motion_showcase", "fly_sniff.odor_motion_edge_assay",
    "fly_sniff.odor_motion_plume_assay", "fly_sniff.odor_motion_qualification",
    "fly_sniff.odor_motion_qualification_receipt", "fly_sniff.experimental_plume",
    "fly_sniff.experimental_plume_archive", "fly_sniff.experimental_plume_cues",
    "fly_sniff.experimental_plume_reproduction", "fly_sniff.olfactory_motion_audit",
    "fly_sniff.olfactory_motion_connectivity",
}


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(f"fly_sniff.{node.module}" if node.level else node.module)
    return found


def test_sealed_v1_does_not_import_new_analysis_or_qualification():
    source = Path(__file__).resolve().parents[1] / "src" / "fly_sniff"
    violations = {
        name: sorted(_imports(source / name) & NEW_ANALYSIS)
        for name in SEALED
        if _imports(source / name) & NEW_ANALYSIS
    }
    assert not violations, f"sealed v1 imported new analysis/qualification modules: {violations}"
