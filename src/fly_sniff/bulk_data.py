from __future__ import annotations

import argparse
import json
from pathlib import Path

from .public_data import CONNECTOME_WEIGHTS_URL, download_file, sha256_file

DEFAULT_WEIGHTS_PATH = Path("data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather")


def require_large_download_confirmation(confirmed: bool) -> None:
    if not confirmed:
        raise SystemExit(
            "The MaleCNS v1.0 full connection-weight table is roughly 1.1 GB. "
            "Re-run with --yes-large-download to fetch it."
        )


def download_weights_main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public MaleCNS v1.0 full connection-weight table"
    )
    parser.add_argument("--output", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument(
        "--yes-large-download",
        action="store_true",
        help="confirm the roughly 1.1 GB download",
    )
    args = parser.parse_args()
    require_large_download_confirmation(args.yes_large_download)
    path = download_file(CONNECTOME_WEIGHTS_URL, args.output)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "url": CONNECTOME_WEIGHTS_URL,
                "dataset": "male-cns:v1.0",
                "authority": "Janelia MaleCNS v1.0 public flat connectome",
            },
            indent=2,
        )
    )
