from __future__ import annotations

import pandas as pd

from fly_sniff.graph import GraphBundle
from fly_sniff.steering_probe import probe_steering_scaffold


def _bundle(*, add_odor_alias: bool = False) -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6, 7, 8]})
    rows = [
        (1, 4, 20.0, "PFL3->DNa02"),
        (1, 2, 12.0, "PFL3->DNa03"),
        (1, 3, 10.0, "PFL3->LAL010"),
        (2, 4, 30.0, "DNa03->DNa02"),
        (3, 4, 25.0, "LAL010->DNa02"),
        (6, 5, 20.0, "PFL3->DNa02"),
        (6, 7, 12.0, "PFL3->DNa03"),
        (6, 8, 10.0, "PFL3->LAL010"),
        (7, 5, 30.0, "DNa03->DNa02"),
        (8, 5, 25.0, "LAL010->DNa02"),
    ]
    edges = pd.DataFrame(
        [
            {
                "source": source,
                "target": target,
                "weight": weight,
                "sign": 1,
                "edge_family": family,
            }
            for source, target, weight, family in rows
        ]
    )
    roles = {
        "turn_drive_left": [1],
        "turn_drive_right": [6],
        "steer_left": [4],
        "steer_right": [5],
    }
    if add_odor_alias:
        roles["odor_left"] = [1]
    return GraphBundle(
        nodes,
        edges,
        roles,
        {"dataset": "male-cns:v1.0", "qualification_status": "candidate"},
    )


def test_steering_probe_passes_symmetric_excitation_scaffold() -> None:
    report = probe_steering_scaffold(_bundle(), seed=7, steps=24)
    assert report["passed"]
    assert report["probe"]["left_turn"] > 0.0
    assert report["probe"]["right_turn"] < 0.0
    assert report["probe"]["deterministic_error"] == 0.0
    assert report["probe"]["pfl3_cut_peak_turn"] == 0.0
    assert report["descriptive_lesions"]["direct_only"]["left_turn"] > 0.0
    assert report["descriptive_lesions"]["no_direct_PFL3_DNa02"]["left_turn"] > 0.0


def test_steering_probe_rejects_sensory_role_aliasing() -> None:
    report = probe_steering_scaffold(_bundle(add_odor_alias=True), seed=7, steps=24)
    gates = {gate["name"]: gate for gate in report["gates"]}
    assert not report["passed"]
    assert not gates["no_sensory_role_aliasing"]["passed"]
