from __future__ import annotations

import hashlib
import html as html_lib
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PartnerSummary:
    partner_type: str
    side: str | None
    direction: str
    population_count: int
    aggregate_connections: int
    neurotransmitter: str | None
    percent_of_direction: float | None
    representative_body_ids: tuple[int, ...]
    partner_page: str | None


@dataclass(frozen=True)
class TypePageEvidence:
    type_name: str
    side: str | None
    aliases: tuple[str, ...]
    display_representatives: tuple[int, ...]
    partner_summaries: tuple[PartnerSummary, ...]
    source_sha256: str
    source_path: str
    upstream_commit: str

    def as_dict(self) -> dict:
        return asdict(self)


_BODY_IDS_RE = re.compile(r"data-body-ids=\[([^\]]*)\]", re.IGNORECASE)
_VISIBLE_RE = re.compile(r"visibleNeurons\s*:\s*\[([^\]]*)\]", re.IGNORECASE)
_H1_RE = re.compile(
    r"<h1[^>]*id=top[^>]*>(?P<name>[^<]+)"
    r"(?:<span[^>]*class=[\"']?soma-side[\"']?[^>]*>\((?P<side>[LRM])\)</span>)?",
    re.IGNORECASE,
)
_ALIAS_RE = re.compile(r"AKA:\s*([^<]+)", re.IGNORECASE)
_ROW_RE = re.compile(r"<tr\s+id=[^>]+>(.*?)(?=<tr\s+id=|</tbody>|</table>)", re.IGNORECASE | re.DOTALL)
_PARTNER_RE = re.compile(
    r"<td[^>]*class=[\"']?p-c[\"']?[^>]*data-body-ids=\[[^\]]*\][^>]*>"
    r"<a\s+href=(?P<href>[^\s>]+)[^>]*>(?P<label>.*?)</a>"
    r"<td>(?P<count>[\d,]+)"
    r"<td>(?P<nt>.*?)"
    r"<td[^>]*title=[\"']?∑\s*connections:\s*(?P<connections>[\d,]+)[\"']?[^>]*>"
    r"(?P<connections_text>[\d,]+)"
    r"<td[^>]*title=(?P<percent>[\d.]+)%",
    re.IGNORECASE | re.DOTALL,
)
_NT_RE = re.compile(r"<abbr[^>]*title=[\"']?([^\"' >]+)[\"']?", re.IGNORECASE)
_SIDE_RE = re.compile(r"\s*\(([LRM])\)\s*$")


def _parse_ints(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if not raw:
        return ()
    values: list[int] = []
    for token in raw.split(","):
        cleaned = token.strip().strip('"\'')
        if not cleaned:
            continue
        if not cleaned.isdigit():
            raise ValueError(f"non-numeric body ID in MaleCNS page: {cleaned!r}")
        value = int(cleaned)
        if value <= 0:
            raise ValueError(f"body ID must be positive: {value}")
        values.append(value)
    return tuple(dict.fromkeys(values))


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    return " ".join(html_lib.unescape(text).split())


def _parse_partner_label(label_html: str) -> tuple[str, str | None]:
    label = _strip_tags(label_html)
    match = _SIDE_RE.search(label)
    side = match.group(1) if match else None
    if match:
        label = label[: match.start()].strip()
    return label, side


def _parse_table(section: str, direction: str) -> list[PartnerSummary]:
    rows: list[PartnerSummary] = []
    for row_match in _ROW_RE.finditer(section):
        row = row_match.group(1)
        partner = _PARTNER_RE.search(row)
        if not partner:
            continue
        body_match = _BODY_IDS_RE.search(row)
        if body_match is None:
            raise ValueError("partner row is missing data-body-ids")
        partner_type, side = _parse_partner_label(partner.group("label"))
        nt_match = _NT_RE.search(partner.group("nt"))
        href = html_lib.unescape(partner.group("href")).split("#", 1)[0].strip('"\'')
        percent = float(partner.group("percent"))
        rows.append(
            PartnerSummary(
                partner_type=partner_type,
                side=side,
                direction=direction,
                population_count=int(partner.group("count").replace(",", "")),
                aggregate_connections=int(
                    partner.group("connections").replace(",", "")
                ),
                neurotransmitter=nt_match.group(1) if nt_match else None,
                percent_of_direction=percent,
                representative_body_ids=_parse_ints(body_match.group(1)),
                partner_page=href or None,
            )
        )
    return rows


def _table_section(page: str, table_id: str) -> str:
    marker = re.search(
        rf"<table[^>]*id=[\"']?{re.escape(table_id)}[\"']?[^>]*>",
        page,
        re.IGNORECASE,
    )
    if marker is None:
        return ""
    end = page.find("</table>", marker.end())
    if end < 0:
        raise ValueError(f"unterminated {table_id} table")
    return page[marker.end() : end]


def parse_type_page(
    page: str | bytes,
    *,
    source_path: str,
    upstream_commit: str,
) -> TypePageEvidence:
    """Parse type-level evidence without promoting representative IDs to membership.

    The explorer's ``visibleNeurons`` and connectivity-row ``data-body-ids`` are
    visualization conveniences. For multi-neuron cell types they can contain only
    one or a few representatives. This parser therefore records them explicitly as
    representatives and never infers a complete population from them.
    """

    raw = page if isinstance(page, bytes) else page.encode()
    text = raw.decode("utf-8")
    h1 = _H1_RE.search(text)
    if h1 is None:
        raise ValueError("could not parse type name from MaleCNS type page")
    type_name = html_lib.unescape(h1.group("name")).strip()
    side = h1.group("side") or None

    alias_match = _ALIAS_RE.search(text)
    aliases: tuple[str, ...] = ()
    if alias_match:
        alias_text = _strip_tags(alias_match.group(1)).strip(" ,")
        if alias_text:
            aliases = tuple(part.strip() for part in alias_text.split(",") if part.strip())

    visible_match = _VISIBLE_RE.search(text)
    visible = _parse_ints(visible_match.group(1)) if visible_match else ()

    partners = [
        *_parse_table(_table_section(text, "upstream-table"), "upstream"),
        *_parse_table(_table_section(text, "downstream-table"), "downstream"),
    ]
    return TypePageEvidence(
        type_name=type_name,
        side=side,
        aliases=aliases,
        display_representatives=visible,
        partner_summaries=tuple(partners),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_path=source_path,
        upstream_commit=upstream_commit,
    )


def find_partner(
    evidence: TypePageEvidence,
    *,
    partner_type: str,
    side: str | None,
    direction: str,
) -> PartnerSummary:
    matches = [
        row
        for row in evidence.partner_summaries
        if row.partner_type == partner_type and row.side == side and row.direction == direction
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {direction} {partner_type}_{side or 'NA'} row on "
            f"{evidence.source_path}; found {len(matches)}"
        )
    return matches[0]
