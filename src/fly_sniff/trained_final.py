from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .controllers import CastSurgeController, Controller
from .evaluate import evaluate, manifest_digest, write_receipt
from .freeze import build_manifest, current_git_ref, make_rewire_seeds
from .graph import GraphBundle
from .qualified_eval import _gold_report, circuit_digest, verify_code_ref
from .rewire import degree_preserving_rewire, lesion_incoming_to_roles
from .trained_qualification import parameters_from_training_report
from .training import (
    FROZEN_V1_REWIRE_COUNT,
    FROZEN_V1_SWAPS_PER_EDGE,
    DynamicsParameters,
    TaskOptimizedMaleCNSController,
    _assert_rewire_matches_original,
    _require_identical_optimizer_budgets,
    canonical_sha256,
    load_training_config,
)

TRAINED_FINAL_SCHEMA = "fly-sniff-trained-final-v1"
TRAINED_FINAL_PROTOCOL = "trained-matched-task-optimization-final-v1"
TRAINED_FINAL_SPLIT_SEED = 26091217
TRAINED_FINAL_HELDOUT_EPISODES = 1000
TRAINED_FINAL_OOD_EPISODES = 400


def _verify_e002(
    report: dict[str, Any],
    *,
    bundle: GraphBundle,
    config: dict[str, Any],
    intact_training_report: dict[str, Any],
) -> None:
    if report.get("protocol") != config["trained_e002"]["protocol"]:
        raise ValueError("trained E002 protocol does not match frozen training config")
    if report.get("passed") is not True:
        raise ValueError("trained final cannot be frozen before intact trained E002 passes")
    if report.get("graph_sha256") != bundle.replay_fingerprint():
        raise ValueError("trained E002 graph fingerprint does not match the intact circuit")
    if report.get("training_config_sha256") != canonical_sha256(config):
        raise ValueError("trained E002 config hash does not match the frozen training config")
    if report.get("trained_parameter_sha256") != intact_training_report.get(
        "trained_parameter_sha256"
    ):
        raise ValueError("trained E002 parameter hash does not match intact trained parameters")
    if report.get("training_audit_receipt_sha256") != intact_training_report.get(
        "audit_receipt_sha256"
    ):
        raise ValueError("trained E002 is not bound to the intact training audit receipt")


def validate_matched_training_artifact(
    bundle: GraphBundle,
    config: dict[str, Any],
    matched_report: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct every matched topology and verify its own training receipt."""
    bundle.validate(require_sign=True, require_qualified=True)
    if matched_report.get("protocol") != "matched-task-optimization-controls-v1":
        raise ValueError("trained final requires matched-task-optimization-controls-v1")
    if matched_report.get("training_config_sha256") != canonical_sha256(config):
        raise ValueError("matched training report config hash mismatch")
    if matched_report.get("intact_graph_sha256") != bundle.replay_fingerprint():
        raise ValueError("matched training report intact graph mismatch")
    if int(matched_report.get("rewire_count", -1)) != FROZEN_V1_REWIRE_COUNT:
        raise ValueError("trained final requires the frozen eight-rewire matched cohort")
    if int(matched_report.get("swaps_per_edge", -1)) != FROZEN_V1_SWAPS_PER_EDGE:
        raise ValueError("trained final requires the frozen 8-swaps/edge null budget")

    expected_rewire_seeds = make_rewire_seeds(n=FROZEN_V1_REWIRE_COUNT)
    observed_rewire_seeds = [int(seed) for seed in matched_report.get("rewire_seeds", [])]
    if observed_rewire_seeds != expected_rewire_seeds:
        raise ValueError("matched training report rewire seeds differ from the frozen v1 seeds")

    results = matched_report.get("results", {})
    intact_report = results.get("intact")
    lesion_report = results.get("lesion")
    rewire_reports = results.get("rewires")
    if not isinstance(intact_report, dict):
        raise TypeError("matched training report is missing intact training evidence")
    if not isinstance(lesion_report, dict):
        raise TypeError("matched training report is missing lesion training evidence")
    if not isinstance(rewire_reports, dict):
        raise TypeError("matched training report is missing rewire training evidence")

    intact_parameters = parameters_from_training_report(intact_report, bundle, config)
    verified_rewires: list[dict[str, Any]] = []
    nested_reports: list[dict[str, Any]] = [intact_report]
    for seed in expected_rewire_seeds:
        rewired = degree_preserving_rewire(
            bundle,
            seed=seed,
            swaps_per_edge=FROZEN_V1_SWAPS_PER_EDGE,
        )
        _assert_rewire_matches_original(bundle, rewired)
        report = rewire_reports.get(str(seed))
        if not isinstance(report, dict):
            raise TypeError(f"matched training report is missing rewire seed {seed}")
        if int(report.get("rewire_seed", -1)) != seed:
            raise ValueError(f"rewire training receipt has wrong seed for {seed}")
        parameters = parameters_from_training_report(report, rewired, config)
        verified_rewires.append(
            {
                "seed": seed,
                "bundle": rewired,
                "parameters": parameters,
                "report": report,
            }
        )
        nested_reports.append(report)

    lesion_roles = ["steer_left", "steer_right"]
    lesioned = lesion_incoming_to_roles(bundle, lesion_roles)
    lesion_parameters = parameters_from_training_report(lesion_report, lesioned, config)
    nested_reports.append(lesion_report)

    budget_sha256 = _require_identical_optimizer_budgets(nested_reports)
    if matched_report.get("optimizer_budget_sha256") != budget_sha256:
        raise ValueError("matched report top-level optimizer budget hash is inconsistent")

    return {
        "intact": {
            "bundle": bundle,
            "parameters": intact_parameters,
            "report": intact_report,
        },
        "rewires": verified_rewires,
        "lesion": {
            "bundle": lesioned,
            "parameters": lesion_parameters,
            "report": lesion_report,
        },
        "optimizer_budget_sha256": budget_sha256,
    }


def build_trained_final_manifest(
    *,
    bundle: GraphBundle,
    circuit_sha256: str,
    config: dict[str, Any],
    matched_report: dict[str, Any],
    trained_e002: dict[str, Any],
    code_ref: str,
) -> dict[str, Any]:
    """Freeze a new final protocol only after all trained artifacts are immutable."""
    if not code_ref or code_ref == "UNKNOWN":
        raise ValueError("trained final manifest requires an exact git commit")
    verified = validate_matched_training_artifact(bundle, config, matched_report)
    intact_report = verified["intact"]["report"]
    _verify_e002(
        trained_e002,
        bundle=bundle,
        config=config,
        intact_training_report=intact_report,
    )

    payload = build_manifest(
        seed=TRAINED_FINAL_SPLIT_SEED,
        n_id=TRAINED_FINAL_HELDOUT_EPISODES,
        n_ood=TRAINED_FINAL_OOD_EPISODES,
        code_ref=code_ref,
        circuit_sha256=circuit_sha256,
    )
    payload.pop("manifest_sha256", None)
    payload["schema"] = TRAINED_FINAL_SCHEMA
    payload["trained_protocol"] = TRAINED_FINAL_PROTOCOL
    payload["circuit_replay_sha256"] = bundle.replay_fingerprint()
    payload["training_config"] = config
    payload["training_config_sha256"] = canonical_sha256(config)
    payload["matched_training_report_sha256"] = canonical_sha256(matched_report)
    payload["trained_e002_sha256"] = canonical_sha256(trained_e002)
    payload["training_audit_receipt_sha256"] = intact_report["audit_receipt_sha256"]
    payload["optimizer_budget_sha256"] = verified["optimizer_budget_sha256"]
    payload["model_contract"] = {
        "controller": TaskOptimizedMaleCNSController.name,
        "comparison": (
            "intact, each degree-preserving rewire, and the steering-input lesion are evaluated "
            "with the separately optimized parameter set frozen for that exact topology"
        ),
        "intact_parameter_sha256": intact_report["trained_parameter_sha256"],
        "rewire_parameter_sha256_by_seed": {
            str(item["seed"]): item["report"]["trained_parameter_sha256"]
            for item in verified["rewires"]
        },
        "lesion_parameter_sha256": verified["lesion"]["report"]["trained_parameter_sha256"],
        "rewire_seeds": [item["seed"] for item in verified["rewires"]],
        "swaps_per_edge": FROZEN_V1_SWAPS_PER_EDGE,
    }
    payload["final_test_policy"] = (
        "This manifest is generated only after the reviewed circuit, frozen task-optimization "
        "config, complete matched-training report, and passing intact trained-E002 artifact are "
        "supplied and hash-bound. Any subsequent model/config/role/parameter change invalidates "
        "this manifest and requires a new protocol version rather than another v1 seed draw."
    )
    payload["manifest_sha256"] = manifest_digest(payload)
    return payload


def verify_trained_final_manifest(manifest: dict[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not expected:
        raise ValueError("trained final manifest is missing manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    actual = manifest_digest(unsigned)
    if expected != actual:
        raise ValueError("trained final manifest hash mismatch")
    if manifest.get("status") != "sealed":
        raise ValueError("trained final evaluation requires a sealed manifest")
    if manifest.get("schema") != TRAINED_FINAL_SCHEMA:
        raise ValueError(f"trained final requires schema={TRAINED_FINAL_SCHEMA!r}")
    if manifest.get("trained_protocol") != TRAINED_FINAL_PROTOCOL:
        raise ValueError("trained final protocol mismatch")
    if int(manifest.get("split_seed", -1)) != TRAINED_FINAL_SPLIT_SEED:
        raise ValueError("trained final split seed differs from frozen v1 protocol")
    if len(manifest.get("heldout_seeds", [])) != TRAINED_FINAL_HELDOUT_EPISODES:
        raise ValueError("trained final held-out episode count differs from frozen v1 protocol")
    if len(manifest.get("ood_seeds", [])) != TRAINED_FINAL_OOD_EPISODES:
        raise ValueError("trained final OOD episode count differs from frozen v1 protocol")


def _controller_factory(
    bundle: GraphBundle,
    parameters: DynamicsParameters,
    *,
    model_dt_s: float,
) -> Callable[[], Controller]:
    return lambda: TaskOptimizedMaleCNSController(
        bundle,
        parameters,
        model_dt_s=model_dt_s,
        require_qualified=True,
    )


def build_trained_factories(
    verified: dict[str, Any],
    *,
    model_dt_s: float,
) -> tuple[dict[str, Callable[[], Controller]], list[str]]:
    factories: dict[str, Callable[[], Controller]] = {
        "malecns": _controller_factory(
            verified["intact"]["bundle"],
            verified["intact"]["parameters"],
            model_dt_s=model_dt_s,
        ),
        "lesion": _controller_factory(
            verified["lesion"]["bundle"],
            verified["lesion"]["parameters"],
            model_dt_s=model_dt_s,
        ),
        "classical": CastSurgeController,
    }
    null_labels: list[str] = []
    for index, item in enumerate(verified["rewires"]):
        label = "rewire" if index == 0 else f"rewire_{index:02d}"
        null_labels.append(label)
        factories[label] = _controller_factory(
            item["bundle"],
            item["parameters"],
            model_dt_s=model_dt_s,
        )
    return factories, null_labels


def freeze_main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the one-way trained matched-control final manifest"
    )
    parser.add_argument("circuit", help="qualified reviewed GraphBundle directory")
    parser.add_argument("matched_training_report", help="complete matched-training report JSON")
    parser.add_argument("trained_e002", help="passing intact trained-E002 report JSON")
    parser.add_argument("--config", default="configs/task_optimization_v1.json")
    parser.add_argument("--output", default="manifests/trained-final-v1.json")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite trained final manifest: {output}")
    bundle = GraphBundle.load(args.circuit)
    config = load_training_config(args.config)
    matched_report = json.loads(Path(args.matched_training_report).read_text())
    trained_e002 = json.loads(Path(args.trained_e002).read_text())
    manifest = build_trained_final_manifest(
        bundle=bundle,
        circuit_sha256=circuit_digest(args.circuit),
        config=config,
        matched_report=matched_report,
        trained_e002=trained_e002,
        code_ref=current_git_ref(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"sealed {output}: {manifest['manifest_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sealed final benchmark with separately trained matched controls"
    )
    parser.add_argument("circuit", help="qualified reviewed GraphBundle directory")
    parser.add_argument("manifest", help="sealed trained-final-v1 manifest JSON")
    parser.add_argument("matched_training_report", help="hash-bound matched-training report JSON")
    parser.add_argument("trained_e002", help="hash-bound passing intact trained-E002 report JSON")
    parser.add_argument("--output", default="results/trained-final-v1")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite trained final evaluation directory: {out}")

    manifest = json.loads(Path(args.manifest).read_text())
    verify_trained_final_manifest(manifest)
    verified_code_ref = verify_code_ref(manifest)
    observed_circuit_sha256 = circuit_digest(args.circuit)
    if observed_circuit_sha256 != manifest["circuit_sha256"]:
        raise ValueError("trained final circuit file digest does not match sealed manifest")

    bundle = GraphBundle.load(args.circuit)
    bundle.validate(require_sign=True, require_qualified=True)
    if bundle.replay_fingerprint() != manifest["circuit_replay_sha256"]:
        raise ValueError("trained final circuit replay fingerprint does not match sealed manifest")

    config = manifest["training_config"]
    if canonical_sha256(config) != manifest["training_config_sha256"]:
        raise ValueError("embedded trained-final config hash mismatch")
    matched_report = json.loads(Path(args.matched_training_report).read_text())
    trained_e002 = json.loads(Path(args.trained_e002).read_text())
    if canonical_sha256(matched_report) != manifest["matched_training_report_sha256"]:
        raise ValueError("matched training report does not match the sealed trained-final artifact")
    if canonical_sha256(trained_e002) != manifest["trained_e002_sha256"]:
        raise ValueError("trained E002 report does not match the sealed trained-final artifact")

    verified = validate_matched_training_artifact(bundle, config, matched_report)
    _verify_e002(
        trained_e002,
        bundle=bundle,
        config=config,
        intact_training_report=verified["intact"]["report"],
    )
    model_contract = manifest["model_contract"]
    if model_contract["intact_parameter_sha256"] != verified["intact"]["report"][
        "trained_parameter_sha256"
    ]:
        raise ValueError("sealed intact trained-parameter hash mismatch")
    for item in verified["rewires"]:
        expected = model_contract["rewire_parameter_sha256_by_seed"].get(str(item["seed"]))
        if expected != item["report"]["trained_parameter_sha256"]:
            raise ValueError(f"sealed trained-parameter hash mismatch for rewire {item['seed']}")
    if model_contract["lesion_parameter_sha256"] != verified["lesion"]["report"][
        "trained_parameter_sha256"
    ]:
        raise ValueError("sealed lesion trained-parameter hash mismatch")

    benchmark = manifest["config"]
    arena = ArenaConfig(**benchmark["arena"])
    id_plume = PlumeConfig(**benchmark["plume"])
    sensors = SensorConfig(**benchmark["sensor"])
    ood_plume = PlumeConfig(**manifest["ood_plume"])
    factories, null_labels = build_trained_factories(verified, model_dt_s=arena.dt)

    id_frame = evaluate(
        factories,
        [int(seed) for seed in manifest["heldout_seeds"]],
        arena=arena,
        plume=id_plume,
        sensors=sensors,
    )
    ood_frame = evaluate(
        factories,
        [int(seed) for seed in manifest["ood_seeds"]],
        arena=arena,
        plume=ood_plume,
        sensors=sensors,
    )
    report = _gold_report(id_frame, ood_frame, manifest, null_labels)
    report["trained_model_contract"] = {
        "controller": TaskOptimizedMaleCNSController.name,
        "matched_training_report_sha256": manifest["matched_training_report_sha256"],
        "trained_e002_sha256": manifest["trained_e002_sha256"],
        "intact_parameter_sha256": model_contract["intact_parameter_sha256"],
        "rewire_parameter_sha256_by_seed": model_contract[
            "rewire_parameter_sha256_by_seed"
        ],
        "lesion_parameter_sha256": model_contract["lesion_parameter_sha256"],
        "interpretation": (
            "Every neural topology is evaluated using the separately optimized parameter set "
            "hash-bound for that exact graph. The classical controller remains untrained."
        ),
    }

    out.mkdir(parents=True, exist_ok=False)
    id_frame.to_parquet(out / "heldout_episodes.parquet", index=False)
    ood_frame.to_parquet(out / "ood_episodes.parquet", index=False)
    (out / "gold_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_receipt(
        out / "receipt.json",
        manifest,
        {
            "code_ref": verified_code_ref,
            "circuit_sha256": observed_circuit_sha256,
            "circuit_replay_sha256": bundle.replay_fingerprint(),
            "matched_training_report_sha256": canonical_sha256(matched_report),
            "trained_e002_sha256": canonical_sha256(trained_e002),
            "gold_report": report,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
