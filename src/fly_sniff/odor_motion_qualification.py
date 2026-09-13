from __future__ import annotations

import argparse

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .odor_motion import load_odor_motion_config
from .odor_motion_edge_assay import run_edge_assay
from .odor_motion_plume_assay import run_plume_assay
from .odor_motion_qualification_receipt import build_receipt, write_qualification


def build_qualification_bundle(document, motion_config):
    q = document.get("qualification")
    if not isinstance(q, dict) or q.get("status") != "prefunctional_required":
        raise ValueError("qualification must remain prefunctional_required")
    arena, plume, sensors = ArenaConfig(), PlumeConfig(), SensorConfig()
    edge = run_edge_assay(q, motion_config, arena, sensors)
    fixed = run_plume_assay(q, motion_config, arena, plume, sensors)
    return build_receipt(q, motion_config, arena, plume, sensors, edge, fixed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controller-free odor-motion qualification")
    parser.add_argument("--config")
    parser.add_argument("--output", default="artifacts/showcase/odor-motion-qualification-v2.json")
    args = parser.parse_args()
    document, motion_config = load_odor_motion_config(args.config)
    bundle = build_qualification_bundle(document, motion_config)
    print(write_qualification(args.output, bundle))
    print(bundle["qualification_sha256"])
    print(bundle["qualification"]["traveling_edge_assay"]["status"])
    print(bundle["qualification"]["fixed_plume_probe_assay"]["status"])


if __name__ == "__main__":
    main()
