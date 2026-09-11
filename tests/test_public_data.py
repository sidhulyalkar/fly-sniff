import pandas as pd

from fly_sniff.public_data import discover_from_annotations


def test_annotation_discovery_assigns_candidate_families():
    frame = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["DNa02_L", "PFL3_R", "Ir75a_ORN", "unrelated"],
            "instance": ["", "", "", ""],
        }
    )
    out = discover_from_annotations(frame)
    pairs = set(zip(out.bodyId, out.candidate_family, strict=True))
    assert (1, "descending") in pairs
    assert (2, "navigation") in pairs
    assert (3, "olfactory_candidate") in pairs
    assert 4 not in set(out.bodyId)
