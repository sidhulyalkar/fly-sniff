from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .public_data import ANNOTATIONS_URL, download_file, sha256_file
from .type_page_authority import TypePageEvidence, find_partner, parse_type_page

DATASET = "male-cns:v1.0"
EXPLORER_REPOSITORY = "reiserlab/celltype-explorer-drosophila-male-cns"
EXPLORER_COMMIT = "789cc6c105798ce2fd70ba85dab394f90899616b"
RAW_EXPLORER_ROOT = (
    "https://raw.githubusercontent.com/"
    f"{EXPLORER_REPOSITORY}/{EXPLORER_COMMIT}/types"
)
DN_PAGE_NAMES = ("DNp01_L.html", "DNp01_R.html")


def _members(frame: pd.DataFrame, type_name: str, side: str) -> tuple[int, ...]:
    required = {"bodyId", "type", "somaSide"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MaleCNS annotations missing required columns: {sorted(missing)}")
    mask = frame["type"].fillna("").astype(str).eq(type_name)
    mask &= frame["somaSide"].fillna("").astype(str).str.upper().eq(side)
    values = tuple(sorted(frame.loc[mask, "bodyId"].astype(int).tolist()))
    if not values:
        raise ValueError(f"no MaleCNS members found for {type_name}_{side}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate body IDs found for {type_name}_{side}")
    return values


def _load_page(path: Path) -> TypePageEvidence:
    return parse_type_page(
        path.read_bytes(),
        source_path=f"types/{path.name}",
        upstream_commit=EXPLORER_COMMIT,
    )


def build_r002_authority(
    annotations_path: str | Path,
    page_paths: dict[str, str | Path],
) -> dict:
    """Build a candidate looming-circuit authority from two independent public sources.

    Complete body membership comes from the released annotation Feather. The pinned
    Cell Type Explorer pages contribute release-specific type-to-type connectivity
    summaries. Page display IDs are preserved only as representatives and never used
    to infer population membership.
    """

    annotations_path = Path(annotations_path)
    frame = pd.read_feather(annotations_path)
    memberships = {
        f"{type_name}_{side}": _members(frame, type_name, side)
        for type_name in ("LPLC2", "LC4", "DNp01")
        for side in ("L", "R")
    }
    for side in ("L", "R"):
        if len(memberships[f"DNp01_{side}"]) != 1:
            raise ValueError(
                f"expected singleton DNp01_{side}; found {len(memberships[f'DNp01_{side}'])}"
            )

    pages = {side: _load_page(Path(page_paths[side])) for side in ("L", "R")}
    structural: list[dict] = []
    for side in ("L", "R"):
        evidence = pages[side]
        if evidence.type_name != "DNp01" or evidence.side != side:
            raise ValueError(
                f"expected DNp01_{side} type page, observed {evidence.type_name}_{evidence.side}"
            )
        for upstream_type in ("LPLC2", "LC4"):
            row = find_partner(
                evidence,
                partner_type=upstream_type,
                side=side,
                direction="upstream",
            )
            observed_count = len(memberships[f"{upstream_type}_{side}"])
            if row.population_count != observed_count:
                raise ValueError(
                    f"population mismatch for {upstream_type}_{side}: "
                    f"explorer={row.population_count} annotations={observed_count}"
                )
            if row.aggregate_connections <= 0:
                raise ValueError(f"no structural connections for {upstream_type}_{side} -> DNp01")
            structural.append(
                {
                    "source_type": upstream_type,
                    "source_side": side,
                    "target_type": "DNp01",
                    "target_side": side,
                    "source_population_count": row.population_count,
                    "aggregate_connections": row.aggregate_connections,
                    "neurotransmitter": row.neurotransmitter,
                    "percent_of_dnp01_input": row.percent_of_direction,
                    "page_representative_body_ids": list(row.representative_body_ids),
                    "page_representative_ids_are_membership_authority": False,
                }
            )

    return {
        "schema": "fly-sniff-r002-authority-v1",
        "dataset": DATASET,
        "qualification_status": "candidate",
        "scientific_claim_allowed": False,
        "authority": {
            "annotations": {
                "url": ANNOTATIONS_URL,
                "path": str(annotations_path),
                "sha256": sha256_file(annotations_path),
                "membership_authority": True,
            },
            "cell_type_explorer": {
                "repository": EXPLORER_REPOSITORY,
                "commit": EXPLORER_COMMIT,
                "pages": {side: pages[side].as_dict() for side in ("L", "R")},
                "membership_authority": False,
                "structural_summary_authority": True,
            },
        },
        "populations": {key: list(value) for key, value in memberships.items()},
        "role_hypotheses": {
            "loom_size_left": {
                "population": "LPLC2_L",
                "semantic_status": "literature-motivated-hypothesis",
            },
            "loom_size_right": {
                "population": "LPLC2_R",
                "semantic_status": "literature-motivated-hypothesis",
            },
            "loom_velocity_left": {
                "population": "LC4_L",
                "semantic_status": "literature-motivated-hypothesis",
            },
            "loom_velocity_right": {
                "population": "LC4_R",
                "semantic_status": "literature-motivated-hypothesis",
            },
            "escape_left": {
                "population": "DNp01_L",
                "semantic_status": "type-alias-and-literature-supported",
                "note": "DNp01 is labelled AKA GF by the pinned MaleCNS explorer.",
            },
            "escape_right": {
                "population": "DNp01_R",
                "semantic_status": "type-alias-and-literature-supported",
                "note": "DNp01 is labelled AKA GF by the pinned MaleCNS explorer.",
            },
        },
        "structural_summaries": structural,
        "execution_ready": False,
        "execution_blockers": [
            "extract body-level weighted edges for the selected populations/pathway",
            "resolve edge signs from released transmitter evidence",
            "choose and justify behavioral readout semantics; DNp01/GF is not silently a turn command",
            "seal an R002 GraphBundle and pass circuit-sanity/lesion tests",
        ],
    }


def _download_inputs(cache_dir: Path) -> tuple[Path, dict[str, Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    annotations = cache_dir / "body-annotations-male-cns-v1.0.feather"
    if not annotations.exists():
        download_file(ANNOTATIONS_URL, annotations)
    pages: dict[str, Path] = {}
    for side, filename in zip(("L", "R"), DN_PAGE_NAMES, strict=True):
        path = cache_dir / filename
        if not path.exists():
            download_file(f"{RAW_EXPLORER_ROOT}/{filename}", path)
        pages[side] = path
    return annotations, pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate R002 MaleCNS authority")
    parser.add_argument("--annotations", help="public MaleCNS v1.0 body annotation Feather")
    parser.add_argument("--left-page", help="pinned DNp01_L HTML page")
    parser.add_argument("--right-page", help="pinned DNp01_R HTML page")
    parser.add_argument("--download", action="store_true", help="download pinned public inputs")
    parser.add_argument("--cache-dir", default="data/raw/r002")
    parser.add_argument("--output", default="artifacts/r002/authority.json")
    args = parser.parse_args()

    if args.download:
        annotations, pages = _download_inputs(Path(args.cache_dir))
    else:
        if not args.annotations or not args.left_page or not args.right_page:
            parser.error("provide all three input paths or pass --download")
        annotations = Path(args.annotations)
        pages = {"L": Path(args.left_page), "R": Path(args.right_page)}

    authority = build_r002_authority(annotations, pages)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(authority, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "qualification_status": authority["qualification_status"],
                "population_counts": {
                    key: len(value) for key, value in authority["populations"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
