import pytest

from fly_sniff.type_page_authority import find_partner, parse_type_page

PAGE = '''<!doctype html>
<h1 id=top>DNp01<span class=soma-side>(L)</span></h1>
<p>AKA: GF (<a href=x>paper</a>) ,</p>
<script>const config = { visibleNeurons: ["10010"] };</script>
<table id=upstream-table><tbody>
<tr id=u0><td class=p-c data-body-ids=[17478]><a href=LC4_L.html#s-c>LC4 (L)</a><td>71<td><abbr title=acetylcholine>ACh</abbr><td title="∑ connections: 3,782">3,782<td class=ug title=23.6%>23.6%<td>0.2
<tr id=u1><td class=p-c data-body-ids=[31563]><a href=LPLC2_L.html#s-c>LPLC2 (L)</a><td>94<td><abbr title=acetylcholine>ACh</abbr><td title="∑ connections: 2,642">2,642<td class=ug title=16.5%>16.5%<td>0.5
</tbody></table>
<table id=downstream-table><tbody>
<tr id=d0><td class=p-c data-body-ids=[800146]><a href=TTMn_L.html#s-c>TTMn (L)</a><td>1<td><abbr title=glutamate>Glu</abbr><td title="∑ connections: 81">81<td class=dg title=6.1%>6.1%<td>0.0
</tbody></table>'''.encode()


def test_representative_body_ids_are_not_population_membership():
    evidence = parse_type_page(
        PAGE,
        source_path="types/DNp01_L.html",
        upstream_commit="abc123",
    )
    assert evidence.type_name == "DNp01"
    assert evidence.side == "L"
    assert evidence.display_representatives == (10010,)

    lplc2 = find_partner(
        evidence,
        partner_type="LPLC2",
        side="L",
        direction="upstream",
    )
    assert lplc2.population_count == 94
    assert lplc2.aggregate_connections == 2642
    assert lplc2.representative_body_ids == (31563,)
    assert len(lplc2.representative_body_ids) != lplc2.population_count


def test_parser_preserves_type_level_structural_summary():
    evidence = parse_type_page(
        PAGE,
        source_path="types/DNp01_L.html",
        upstream_commit="abc123",
    )
    lc4 = find_partner(evidence, partner_type="LC4", side="L", direction="upstream")
    assert lc4.population_count == 71
    assert lc4.aggregate_connections == 3782
    assert lc4.neurotransmitter == "acetylcholine"
    assert lc4.percent_of_direction == 23.6
    assert lc4.partner_page == "LC4_L.html"


def test_parser_rejects_non_numeric_representative_id():
    bad = PAGE.replace(b"data-body-ids=[31563]", b"data-body-ids=[oops]")
    with pytest.raises(ValueError, match="non-numeric body ID"):
        parse_type_page(bad, source_path="bad.html", upstream_commit="abc123")
