from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import PlumeConfig, default_config_dict


def canonical_sha256(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def current_git_ref() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def make_seed_split(seed: int, n_id: int, n_ood: int) -> tuple[list[int], list[int]]:
    if n_id < 0 or n_ood < 0 or n_id + n_ood > 1_999_999_999:
        raise ValueError("split sizes must be nonnegative and fit the seed population")
    rng = np.random.default_rng(seed)
    # Sampling integer indices is identical to indexing arange, without 16 GB allocation.
    values = rng.choice(
        1_999_999_999,
        size=n_id + n_ood,
        replace=False,
    ) + 1
    return [int(x) for x in values[:n_id]], [int(x) for x in values[n_id:]]


def default_ood_plume() -> dict:
    # Frozen distribution shift: faster, more intermittent, more crosswind wandering.
    return asdict(
        PlumeConfig(
            wind_speed=0.90,
            emission_rate_hz=6.0,
            crosswind_noise=0.22,
            meander_amplitude=0.30,
        )
    )


def build_manifest(
    *,
    seed: int,
    n_id: int,
    n_ood: int,
    code_ref: str,
    circuit_sha256: str,
) -> dict:
    id_seeds, ood_seeds = make_seed_split(seed, n_id, n_ood)
    payload = {
        "schema": "fly-sniff-final-v1",
        "status": "sealed",
        "code_ref": code_ref,
        "circuit_sha256": circuit_sha256,
        "split_seed": int(seed),
        "heldout_seeds": id_seeds,
        "ood_seeds": ood_seeds,
        "config": default_config_dict(),
        "ood_plume": default_ood_plume(),
        "rewire": {"seed": 913013, "swaps_per_edge": 8},
        "gold": {
            "success_rate_min": 0.70,
            "spl_delta_vs_rewire_min": 0.10,
            "paired_ci95_must_exclude_zero": True,
            "ood_success_rate_min": 0.60,
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal a one-way FlyBrain Plume Hunt final-test manifest"
    )
    parser.add_argument("--output", default="manifests/final-sealed-v1.json")
    parser.add_argument("--seed", type=int, default=48151623)
    parser.add_argument("--heldout", type=int, default=1000)
    parser.add_argument("--ood", type=int, default=400)
    parser.add_argument("--circuit-sha256", required=True)
    parser.add_argument("--code-ref", default=None)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite sealed manifest: {output}")
    manifest = build_manifest(
        seed=args.seed,
        n_id=args.heldout,
        n_ood=args.ood,
        code_ref=args.code_ref or current_git_ref(),
        circuit_sha256=args.circuit_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"sealed {output}: {manifest['manifest_sha256']}")
