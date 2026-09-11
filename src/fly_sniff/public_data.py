from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

ANNOTATIONS_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
CONNECTOME_WEIGHTS_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

DEFAULT_DISCOVERY_PATTERNS = {
    "navigation": [r"^hDelta", r"^PFN", r"^PFL2", r"^PFL3"],
    "descending": [r"^DNa02"],
    "olfactory_candidate": [r"ORN", r"^Or", r"^Ir"],
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "fly-sniff/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return destination


def discover_from_annotations(
    frame: pd.DataFrame,
    patterns: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    patterns = patterns or DEFAULT_DISCOVERY_PATTERNS
    text_cols = [col for col in ["type", "instance", "class", "subclass"] if col in frame.columns]
    if not text_cols:
        raise ValueError(f"no expected annotation text columns found; columns={list(frame.columns)}")
    searchable = frame[text_cols].fillna("").astype(str).agg(" | ".join, axis=1)
    pieces: list[pd.DataFrame] = []
    for family, regexes in patterns.items():
        union = "(?:" + ")|(?:".join(regexes) + ")"
        mask = searchable.str.contains(re.compile(union, re.IGNORECASE), regex=True)
        if mask.any():
            part = frame.loc[mask].copy()
            part["candidate_family"] = family
            pieces.append(part)
    if not pieces:
        return frame.head(0).assign(candidate_family=pd.Series(dtype=str))
    body_col = "bodyId" if "bodyId" in frame.columns else None
    out = pd.concat(pieces, ignore_index=True)
    subset = [body_col, "candidate_family"] if body_col else None
    return out.drop_duplicates(subset=subset)


def download_annotations_main() -> None:
    parser = argparse.ArgumentParser(description="Download the public MaleCNS v1.0 annotation table")
    parser.add_argument("--output", default="data/raw/body-annotations-male-cns-v1.0.feather")
    args = parser.parse_args()
    path = download_file(ANNOTATIONS_URL, args.output)
    print(json.dumps({"path": str(path), "sha256": sha256_file(path), "url": ANNOTATIONS_URL}, indent=2))


def offline_discover_main() -> None:
    parser = argparse.ArgumentParser(description="Discover candidate MaleCNS cell types from public annotations")
    parser.add_argument("annotations")
    parser.add_argument("--patterns", help="optional JSON family -> regex-list mapping")
    parser.add_argument("--output", default="data/cache/public-discovery-v0.csv")
    args = parser.parse_args()
    patterns = DEFAULT_DISCOVERY_PATTERNS
    if args.patterns:
        patterns = json.loads(Path(args.patterns).read_text())
    frame = pd.read_feather(args.annotations)
    candidates = discover_from_annotations(frame, patterns)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out, index=False)
    print(f"wrote {len(candidates)} candidate rows to {out}")
