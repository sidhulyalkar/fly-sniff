from types import SimpleNamespace

from fly_sniff.route_review import infer_side_evidence

SIDE_CONFIG = {
    "priority": ["somaSide", "rootSide", "instance"],
    "left_values": ["L", "left", "Left"],
    "right_values": ["R", "right", "Right"],
    "left_pattern": r"(?:^|_)L(?:\d|_|$)",
    "right_pattern": r"(?:^|_)R(?:\d|_|$)",
}


def test_soma_side_has_priority_over_instance_fallback():
    row = SimpleNamespace(somaSide="L", rootSide="R", instance="PFNp_c_R3")
    assert infer_side_evidence(row, SIDE_CONFIG) == ("L", "somaSide")


def test_root_side_is_used_when_soma_side_is_missing():
    row = SimpleNamespace(somaSide=None, rootSide="R", instance="PFNp_c")
    assert infer_side_evidence(row, SIDE_CONFIG) == ("R", "rootSide")


def test_instance_is_only_a_fallback():
    row = SimpleNamespace(somaSide=None, rootSide=None, instance="PFNa_L4_C2")
    assert infer_side_evidence(row, SIDE_CONFIG) == ("L", "instance")


def test_unresolved_side_remains_unresolved():
    row = SimpleNamespace(somaSide=None, rootSide=None, instance="PFNp_c")
    assert infer_side_evidence(row, SIDE_CONFIG) == (None, None)
