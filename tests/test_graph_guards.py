import pandas as pd
import pytest

from fly_sniff.graph import GraphBundle, MaleCNSRateController


def _bundle(*, signed: bool, qualified: bool) -> GraphBundle:
    nodes = pd.DataFrame({"bodyId": [1, 2]})
    edges = pd.DataFrame({"source": [1], "target": [2], "weight": [7]})
    if signed:
        edges["sign"] = [1]
    manifest = {"qualification_status": "qualified"} if qualified else None
    roles = {"odor_left": [1], "steer_right": [2], "steer_left": [1]}
    return GraphBundle(nodes, edges, roles, manifest)


def test_qualified_controller_rejects_unsigned_graph():
    with pytest.raises(ValueError, match="sign"):
        MaleCNSRateController(_bundle(signed=False, qualified=True))


def test_qualified_controller_rejects_candidate_manifest():
    with pytest.raises(ValueError, match="not sealed"):
        MaleCNSRateController(_bundle(signed=True, qualified=False))


def test_development_override_is_explicit():
    controller = MaleCNSRateController(_bundle(signed=True, qualified=False), require_qualified=False)
    assert controller.name == "malecns-rate-v0"
