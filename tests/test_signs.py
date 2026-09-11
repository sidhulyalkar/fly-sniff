import pandas as pd

from fly_sniff.signs import attach_presynaptic_signs


def test_conservative_sign_policy_does_not_invent_glutamate_sign():
    nodes = pd.DataFrame(
        {
            "bodyId": [1, 2, 3],
            "predictedNt": ["ACh", "GABA", "glutamate"],
        }
    )
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3],
            "target": [2, 3, 1],
            "weight": [5, 6, 7],
        }
    )
    signed, stats = attach_presynaptic_signs(nodes, edges)
    assert list(signed.sign) == [1, -1, 0]
    assert stats["signed_fraction"] == 2 / 3
