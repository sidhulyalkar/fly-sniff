from pathlib import Path

import pandas as pd
import pytest

from fly_sniff.r002_authority import build_r002_authority


def _page(side: str, lc4_count: int, lplc2_count: int, dn_id: int) -> bytes:
    lc4_rep = 17478 if side == "L" else 22847
    lplc2_rep = 31563 if side == "L" else 20182
    return f'''<!doctype html>
<h1 id=top>DNp01<span class=soma-side>({side})</span></h1>
<p>AKA: GF (<a href=x>paper</a>) ,</p>
<script>const config = {{ visibleNeurons: ["{dn_id}"] }};</script>
<table id=upstream-table><tbody>
<tr id=u0><td class=p-c data-body-ids=[{lc4_rep}]><a href=LC4_{side}.html#s-c>LC4 ({side})</a><td>{lc4_count}<td><abbr title=acetylcholine>ACh</abbr><td title="∑ connections: 120">120<td class=ug title=20.0%>20.0%<td>0.2
<tr id=u1><td class=p-c data-body-ids=[{lplc2_rep}]><a href=LPLC2_{side}.html#s-c>LPLC2 ({side})</a><td>{lplc2_count}<td><abbr title=acetylcholine>ACh</abbr><td title="∑ connections: 90">90<td class=ug title=15.0%>15.0%<td>0.4
</tbody></table>
<table id=downstream-table><tbody></tbody></table>'''.encode()


def _write_inputs(tmp_path: Path, *, bad_count: bool = False):
    rows = []
    body_id = 1
    counts = {"LPLC2": 3, "LC4": 2, "DNp01": 1}
    for type_name, count in counts.items():
        for side in ("L", "R"):
            for _ in range(count):
                rows.append({"bodyId": body_id, "type": type_name, "somaSide": side})
                body_id += 1
    annotations = tmp_path / "annotations.feather"
    pd.DataFrame(rows).to_feather(annotations)

    left = tmp_path / "DNp01_L.html"
    right = tmp_path / "DNp01_R.html"
    left.write_bytes(_page("L", 99 if bad_count else 2, 3, 6))
    right.write_bytes(_page("R", 2, 3, 7))
    return annotations, {"L": left, "R": right}


def test_builder_uses_annotations_for_complete_membership(tmp_path):
    annotations, pages = _write_inputs(tmp_path)
    authority = build_r002_authority(annotations, pages)

    assert authority["qualification_status"] == "candidate"
    assert authority["scientific_claim_allowed"] is False
    assert len(authority["populations"]["LPLC2_L"]) == 3
    assert len(authority["populations"]["LC4_R"]) == 2
    assert len(authority["populations"]["DNp01_L"]) == 1
    assert authority["execution_ready"] is False

    lplc2_left = next(
        row
        for row in authority["structural_summaries"]
        if row["source_type"] == "LPLC2" and row["source_side"] == "L"
    )
    assert lplc2_left["source_population_count"] == 3
    assert lplc2_left["page_representative_body_ids"] == [31563]
    assert lplc2_left["page_representative_ids_are_membership_authority"] is False


def test_builder_refuses_page_annotation_population_mismatch(tmp_path):
    annotations, pages = _write_inputs(tmp_path, bad_count=True)
    with pytest.raises(ValueError, match="population mismatch for LC4_L"):
        build_r002_authority(annotations, pages)
