from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .graph import GraphBundle
from .r002_probe import run_escape_probe
from .rewire import degree_preserving_rewire, save_bundle

QUALIFICATION_PROTOCOL = "R002-qualification-v1"
DEFAULT_REWIRE_SEEDS = (24018, 24019, 24020, 24021, 24022)


def _lesion_visual_outputs(bundle: GraphBundle) -> GraphBundle:
    lesioned = bundle.edges.copy()
    visual_ids = set(
        bundle.roles.get("loom_size_left", [])
        + bundle.roles.get("loom_size_right", [])
        + bundle.roles.get("loom_velocity_left", [])
        + bundle.roles.get("loom_velocity_right", [])
    )
    lesioned.loc[lesioned.source.astype(int).isin(visual_ids), "sign"] = 0
    manifest = copy.deepcopy(bundle.manifest) if bundle.manifest else {}
    manifest["graph_role"] = "combined-LPLC2-LC4-output-lesion"
    return GraphBundle(
        nodes=bundle.nodes.copy(),
        edges=lesioned,
        roles={key: list(value) for key, value in bundle.roles.items()},
        manifest=manifest,
    )


def evaluate_qualification(
    *,
    structural_exact: bool,
    intact: dict,
    rewires: list[dict],
    lesion: dict,
) -> dict:
    """Evaluate the predeclared R002 gate without access to hidden thresholds."""

    if len(rewires) < 5:
        raise ValueError("R002 qualification requires at least five fixed rewires")
    intact_sep = float(intact["mean_direct_minus_near_target"])
    intact_lat = float(intact["mean_direct_lateralization"])
    intact_rate = float(intact["direct_stronger_than_near_rate"])
    rewire_sep_wins = sum(
        intact_sep > float(report["mean_direct_minus_near_target"]) for report in rewires
    )
    rewire_lat_wins = sum(
        intact_lat > float(report["mean_direct_lateralization"]) for report in rewires
    )

    intact_direct = float(
        np.mean([row["direct_peak_target"] for row in intact["trials_detail"]])
    )
    lesion_direct = float(
        np.mean([row["direct_peak_target"] for row in lesion["trials_detail"]])
    )
    lesion_fraction = lesion_direct / max(abs(intact_direct), 1e-12)

    gates = {
        "structural_exact": bool(structural_exact),
        "direct_stronger_than_near_rate_gte_0_95": intact_rate >= 0.95,
        "mean_direct_minus_near_positive": intact_sep > 0.0,
        "mean_direct_lateralization_positive": intact_lat > 0.0,
        "intact_beats_at_least_4_of_5_rewires_on_separation": rewire_sep_wins >= 4,
        "intact_beats_at_least_4_of_5_rewires_on_lateralization": rewire_lat_wins >= 4,
        "combined_visual_lesion_reduces_direct_escape_below_5pct": lesion_fraction < 0.05,
    }
    return {
        "protocol": QUALIFICATION_PROTOCOL,
        "gates": gates,
        "pass": all(gates.values()),
        "summary": {
            "intact_direct_stronger_than_near_rate": intact_rate,
            "intact_mean_direct_minus_near_target": intact_sep,
            "intact_mean_direct_lateralization": intact_lat,
            "rewire_separation_wins": rewire_sep_wins,
            "rewire_lateralization_wins": rewire_lat_wins,
            "rewire_count": len(rewires),
            "intact_mean_direct_peak_target": intact_direct,
            "lesion_mean_direct_peak_target": lesion_direct,
            "lesion_fraction_of_intact": lesion_fraction,
        },
    }


def qualify_r002(
    bundle: GraphBundle,
    *,
    trials: int = 24,
    seed: int = 24017,
    rewire_seeds: tuple[int, ...] = DEFAULT_REWIRE_SEEDS,
) -> tuple[GraphBundle | None, dict]:
    if not bundle.manifest or bundle.manifest.get("qualification_status") != "candidate":
        raise ValueError("R002 qualifier requires a candidate graph bundle")
    structural_rows = bundle.manifest.get("direct_edge_validation", [])
    structural_exact = len(structural_rows) == 4 and all(
        bool(row.get("exact_match")) for row in structural_rows
    )

    intact = run_escape_probe(
        bundle,
        trials=trials,
        seed=seed,
        require_qualified=False,
    )
    rewires: list[dict] = []
    for rewire_seed in rewire_seeds:
        rewired = degree_preserving_rewire(bundle, seed=rewire_seed)
        report = run_escape_probe(
            rewired,
            trials=trials,
            seed=seed,
            require_qualified=False,
        )
        report["rewire_seed"] = int(rewire_seed)
        rewires.append(report)

    lesion_bundle = _lesion_visual_outputs(bundle)
    lesion = run_escape_probe(
        lesion_bundle,
        trials=trials,
        seed=seed,
        require_qualified=False,
    )
    decision = evaluate_qualification(
        structural_exact=structural_exact,
        intact=intact,
        rewires=rewires,
        lesion=lesion,
    )
    report = {
        **decision,
        "dataset": bundle.manifest.get("dataset"),
        "trials": trials,
        "seed": seed,
        "rewire_seeds": list(rewire_seeds),
        "intact": intact,
        "rewires": rewires,
        "combined_visual_output_lesion": lesion,
        "claim_scope": (
            "A qualified pass supports only a modeled MaleCNS LPLC2/LC4-to-DNp01/GF "
            "collision-sensitive escape readout under the explicit R002 synthetic looming adapter. "
            "It does not establish perception, measured neural dynamics, steering, or a complete "
            "biological escape response."
        ),
    }
    if not decision["pass"]:
        return None, report

    manifest = copy.deepcopy(bundle.manifest)
    manifest["qualification_status"] = "qualified"
    manifest["scientific_claim_allowed"] = True
    manifest["qualification"] = {
        "protocol": QUALIFICATION_PROTOCOL,
        "trials": trials,
        "seed": seed,
        "rewire_seeds": list(rewire_seeds),
        "gates": decision["gates"],
        "summary": decision["summary"],
        "claim_scope": report["claim_scope"],
    }
    qualified = GraphBundle(
        nodes=bundle.nodes.copy(),
        edges=bundle.edges.copy(),
        roles={key: list(value) for key, value in bundle.roles.items()},
        manifest=manifest,
    )
    qualified.validate(require_sign=True, require_qualified=True)
    return qualified, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen R002 functional qualification gate")
    parser.add_argument("circuit")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=24017)
    parser.add_argument("--output", default="artifacts/r002/qualification.json")
    parser.add_argument("--qualified-circuit", default="artifacts/r002/qualified-graph")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="exit non-zero on a scientific qualification miss; default preserves negative results",
    )
    args = parser.parse_args()

    bundle = GraphBundle.load(args.circuit)
    qualified, report = qualify_r002(bundle, trials=args.trials, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": report["pass"], **report["summary"]}, indent=2, sort_keys=True))
    if qualified is not None:
        save_bundle(qualified, args.qualified_circuit)
    elif args.require_pass:
        raise SystemExit("R002 functional qualification did not pass; candidate artifacts are preserved")


if __name__ == "__main__":
    main()
