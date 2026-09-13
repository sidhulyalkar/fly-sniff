from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .freeze import make_rewire_seeds, make_seed_split
from .trained_final import (
    TRAINED_FINAL_HELDOUT_EPISODES,
    TRAINED_FINAL_OOD_EPISODES,
    TRAINED_FINAL_SPLIT_SEED,
)
from .training import (
    FINAL_TEST_MAX_SEED,
    FROZEN_V1_REWIRE_COUNT,
    FROZEN_V1_SWAPS_PER_EDGE,
    canonical_sha256,
    load_training_config,
    make_training_seed_split,
)

PROTOCOL = "sealed-navigation-seed-plan-v1"


def build_seed_plan(config: dict[str, Any]) -> dict[str, Any]:
    train_seeds, validation_seeds = make_training_seed_split(config)
    heldout_seeds, ood_seeds = make_seed_split(
        TRAINED_FINAL_SPLIT_SEED,
        TRAINED_FINAL_HELDOUT_EPISODES,
        TRAINED_FINAL_OOD_EPISODES,
    )
    rewire_seeds = make_rewire_seeds(n=FROZEN_V1_REWIRE_COUNT)

    development = train_seeds + validation_seeds
    final = heldout_seeds + ood_seeds
    if set(train_seeds) & set(validation_seeds):
        raise RuntimeError("train and validation seeds overlap")
    if set(heldout_seeds) & set(ood_seeds):
        raise RuntimeError("held-out and OOD final seeds overlap")
    if set(development) & set(final):
        raise RuntimeError("development and final seed populations overlap")
    if min(development) <= FINAL_TEST_MAX_SEED:
        raise RuntimeError("development seed entered the final-test namespace")
    if max(final) > FINAL_TEST_MAX_SEED:
        raise RuntimeError("final seed escaped the frozen final-test namespace")
    if len(set(rewire_seeds)) != FROZEN_V1_REWIRE_COUNT:
        raise RuntimeError("rewire seed plan contains duplicates")

    payload = {
        "protocol": PROTOCOL,
        "status": "SEALED_BEFORE_NAVIGATION_PERFORMANCE",
        "training_config_sha256": canonical_sha256(config),
        "development": {
            "train_seeds": train_seeds,
            "train_seed_sha256": canonical_sha256(train_seeds),
            "validation_seeds": validation_seeds,
            "validation_seed_sha256": canonical_sha256(validation_seeds),
            "namespace_rule": "all development episode seeds are > 1,999,999,999",
        },
        "rewire": {
            "seeds": rewire_seeds,
            "seeds_sha256": canonical_sha256(rewire_seeds),
            "count": FROZEN_V1_REWIRE_COUNT,
            "swaps_per_edge": FROZEN_V1_SWAPS_PER_EDGE,
        },
        "final": {
            "split_seed": TRAINED_FINAL_SPLIT_SEED,
            "heldout_seeds": heldout_seeds,
            "heldout_seed_sha256": canonical_sha256(heldout_seeds),
            "ood_seeds": ood_seeds,
            "ood_seed_sha256": canonical_sha256(ood_seeds),
            "heldout_count": TRAINED_FINAL_HELDOUT_EPISODES,
            "ood_count": TRAINED_FINAL_OOD_EPISODES,
            "namespace_rule": "all final held-out and OOD seeds are in 1..1,999,999,999",
        },
        "selection_rule": (
            "These exact seed lists are materialized and hash-bound before any navigation "
            "objective is evaluated. Observed model performance cannot change them."
        ),
    }
    payload["seed_plan_sha256"] = canonical_sha256(payload)
    return payload


def verify_seed_plan(plan: dict[str, Any], config: dict[str, Any]) -> None:
    stored = plan.get("seed_plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("seed_plan_sha256", None)
    if not stored or stored != canonical_sha256(unsigned):
        raise ValueError("seed-plan hash mismatch")
    expected = build_seed_plan(config)
    if plan != expected:
        raise ValueError("seed plan differs from the frozen v1 generator/config contract")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize all development, rewire, held-out, and OOD seeds before performance"
    )
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument("--output", default="manifests/navigation-seed-plan-v1.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite sealed seed plan: {output}")
    config = load_training_config(args.config)
    plan = build_seed_plan(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "seed_plan_sha256": plan["seed_plan_sha256"],
                "train_count": len(plan["development"]["train_seeds"]),
                "validation_count": len(plan["development"]["validation_seeds"]),
                "rewire_count": len(plan["rewire"]["seeds"]),
                "heldout_count": len(plan["final"]["heldout_seeds"]),
                "ood_count": len(plan["final"]["ood_seeds"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
