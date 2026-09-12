import numpy as np
import pandas as pd

from fly_sniff.graph import GraphBundle, MaleCNSRateController


def _bundle():
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        [
            {"source": 1, "target": 6, "weight": 1.0, "sign": 1},
        ]
    )
    roles = {
        "wind_basis_left": [1, 2],
        "wind_basis_right": [3, 4, 5],
        "steer_left": [6],
    }
    return GraphBundle(nodes, edges, roles, {"qualification_status": "candidate"})


def test_task_sensory_roles_receive_equal_total_drive_despite_population_size():
    controller = MaleCNSRateController(_bundle(), require_qualified=False)
    left = np.zeros(6, dtype=float)
    right = np.zeros(6, dtype=float)
    controller._inject("wind_basis_left", 0.9, left)
    controller._inject("wind_basis_right", 0.9, right)
    assert np.isclose(left.sum(), 0.9)
    assert np.isclose(right.sum(), 0.9)
    assert np.allclose(left[:2], [0.45, 0.45])
    assert np.allclose(right[2:5], [0.3, 0.3, 0.3])


def test_non_task_role_keeps_legacy_per_neuron_semantics():
    controller = MaleCNSRateController(_bundle(), require_qualified=False)
    drive = np.zeros(6, dtype=float)
    controller._inject("steer_left", 0.7, drive)
    assert np.isclose(drive[5], 0.7)
