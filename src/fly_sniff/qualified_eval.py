from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import ArenaConfig, PlumeConfig, SensorConfig
from .controllers import CastSurgeController
from .evaluate import (
    evaluate,
    manifest_digest,
    paired_spl_report,
    paired_success_report,
    summarize,
    write_receipt,
)
from .freeze import current_git_ref
from .graph import GraphBundle, MaleCNSRateController
from .metrics import paired_bootstrap_delta
from .rewire import degree_preserving_rewire, lesion_incoming_to_roles


def _file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def circuit_digest(directory: str | Path) -> str:
    root = Path(directory)
    return _file_digest([root / "nodes.parquet", root / "edges.parquet", root / "roles.json"])


def verify_sealed_manifest(manifest: dict) -> None:
    expected = manifest.get("manifest_sha256")
    if not expected:
        raise ValueError("sealed manifest is missing manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    actual = manifest_digest(unsigned)
    if actual != expected:
        raise ValueError(f"manifest hash mismatch: expected {expected}, recomputed {actual}")
    if manifest.get("status") != "sealed":
        raise ValueError("final evaluation requires status='sealed'")
    if manifest.get("schema") != "fly-sniff-final-v2":
        raise ValueError("final evaluation requires schema='fly-sniff-final-v2'")


def verify_code_ref(manifest: dict) -> str:
    expected = str(manifest.get("code_ref", ""))
    actual = current_git_ref()
    if not expected or expected == "UNKNOWN":
        raise ValueError("sealed final manifest must name an exact git commit")
    if actual == "UNKNOWN":
        raise ValueError(
            "cannot establish current git commit; run final evaluation from the repository checkout"
        )
    if actual != expected:
        raise ValueError(f"code-ref mismatch: manifest={expected} checkout={actual}")
    return actual


def _paired_against_null_mean(
    frame: pd.DataFrame,
    *,
    metric: str,
    intact_label: str,
    null_labels: list[str],
    bootstrap_seed: int = 13013,
    n_boot: int = 10000,
) -> dict[str, float | int | str]:
    required = [intact_label, *null_labels]
    pivot = frame.pivot(index="seed", columns="label", values=metric).dropna(subset=required)
    intact = pivot[intact_label].astype(float).to_numpy()
    null_mean = pivot[null_labels].astype(float).mean(axis=1).to_numpy()
    mean, lo, hi = paired_bootstrap_delta(
        intact,
        null_mean,
        seed=bootstrap_seed,
        n_boot=n_boot,
    )
    return {
        "metric": metric,
        "comparison": "intact minus per-environment mean across sealed rewire topologies",
        "mean_delta": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_environment_seeds": len(pivot),
        "n_null_topologies": len(null_labels),
    }


def _null_topology_summary(
    summary: pd.DataFrame,
    *,
    null_labels: list[str],
) -> dict[str, float | int | list[dict]]:
    indexed = summary.set_index("label")
    nulls = indexed.loc[null_labels]
    intact_spl = float(indexed.loc["malecns", "mean_spl"])
    null_spl = nulls.mean_spl.astype(float)
    return {
        "n_null_topologies": len(null_labels),
        "mean_of_null_mean_spl": float(null_spl.mean()),
        "std_of_null_mean_spl": float(null_spl.std(ddof=1)) if len(null_spl) > 1 else 0.0,
        "min_null_mean_spl": float(null_spl.min()),
        "max_null_mean_spl": float(null_spl.max()),
        "intact_mean_spl": intact_spl,
        "null_topologies_matching_or_beating_intact": int((null_spl >= intact_spl).sum()),
        "per_topology": nulls.reset_index().to_dict(orient="records"),
    }


def _gold_report(
    id_frame: pd.DataFrame,
    ood_frame: pd.DataFrame,
    manifest: dict,
    null_labels: list[str],
) -> dict:
    id_summary_frame = summarize(id_frame)
    ood_summary_frame = summarize(ood_frame)
    id_summary = id_summary_frame.set_index("label")
    ood_summary = ood_summary_frame.set_index("label")
    paired_spl = paired_spl_report(id_frame, "malecns", "rewire")
    paired_success = paired_success_report(id_frame, "malecns", "rewire")
    lesion_spl = paired_spl_report(id_frame, "malecns", "lesion")
    lesion_success = paired_success_report(id_frame, "malecns", "lesion")
    ensemble_spl = _paired_against_null_mean(
        id_frame,
        metric="spl",
        intact_label="malecns",
        null_labels=null_labels,
    )
    ensemble_success = _paired_against_null_mean(
        id_frame,
        metric="success",
        intact_label="malecns",
        null_labels=null_labels,
    )
    ood_ensemble_spl = _paired_against_null_mean(
        ood_frame,
        metric="spl",
        intact_label="malecns",
        null_labels=null_labels,
    )
    gold = manifest["gold"]
    sr = float(id_summary.loc["malecns", "success_rate"])
    ood_sr = float(ood_summary.loc["malecns", "success_rate"])
    checks = {
        "id_success": sr >= float(gold["success_rate_min"]),
        "primary_rewire_spl_delta": paired_spl["mean_delta"]
        >= float(gold["spl_delta_vs_rewire_min"]),
        "primary_rewire_ci_excludes_zero": paired_spl["ci95_low"] > 0.0,
        "ood_success": ood_sr >= float(gold["ood_success_rate_min"]),
        "ensemble_spl_delta": ensemble_spl["mean_delta"]
        >= float(gold["ensemble_spl_delta_min"]),
        "ensemble_ci_excludes_zero": ensemble_spl["ci95_low"] > 0.0,
    }
    return {
        "flynav_gold": bool(all(checks.values())),
        "checks": checks,
        "id_success_rate": sr,
        "ood_success_rate": ood_sr,
        "malecns_vs_primary_rewire_spl": paired_spl,
        "malecns_vs_primary_rewire_success": paired_success,
        "malecns_vs_rewire_ensemble_spl": ensemble_spl,
        "malecns_vs_rewire_ensemble_success": ensemble_success,
        "ood_malecns_vs_rewire_ensemble_spl": ood_ensemble_spl,
        "rewire_ensemble_topology_summary": _null_topology_summary(
            id_summary_frame,
            null_labels=null_labels,
        ),
        "malecns_vs_lesion_spl": lesion_spl,
        "malecns_vs_lesion_success": lesion_success,
        "id_summary": id_summary.reset_index().to_dict(orient="records"),
        "ood_summary": ood_summary.reset_index().to_dict(orient="records"),
        "control_contract": {
            "primary_rewire": manifest["rewire"],
            "rewire_ensemble": manifest["rewire_ensemble"],
            "lesion": manifest["lesion"],
            "paired_environment": (
                "malecns, every rewire topology, lesion, and classical controller are evaluated on "
                "the same seed lists and plume configuration within each cohort"
            ),
        },
        "statistical_note": (
            "Final-v2 retains the designated primary rewire for backwards-comparable reporting, "
            "but the wiring claim additionally requires paired SPL superiority over the per-seed "
            "mean of the sealed degree-preserving rewire ensemble. The matched lesion and success "
            "effects are prespecified dependency/control evidence rather than substitute gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sealed qualified MaleCNS benchmark")
    parser.add_argument("circuit", help="qualified GraphBundle directory")
    parser.add_argument("manifest", help="sealed final-test JSON")
    parser.add_argument("--output", default="results/final-v2")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite final evaluation directory: {out}")

    manifest = json.loads(Path(args.manifest).read_text())
    verify_sealed_manifest(manifest)
    verified_code_ref = verify_code_ref(manifest)
    observed_circuit_sha = circuit_digest(args.circuit)
    if observed_circuit_sha != manifest["circuit_sha256"]:
        raise SystemExit(
            "circuit hash mismatch; refusing to run sealed evaluation: "
            f"manifest={manifest['circuit_sha256']} observed={observed_circuit_sha}"
        )

    biological = GraphBundle.load(args.circuit)
    biological.validate(require_sign=True, require_qualified=True)

    ensemble_cfg = manifest["rewire_ensemble"]
    rewire_seeds = [int(seed) for seed in ensemble_cfg["seeds"]]
    if len(rewire_seeds) != int(ensemble_cfg["count"]) or len(set(rewire_seeds)) != len(
        rewire_seeds
    ):
        raise ValueError("sealed rewire ensemble count/seeds are inconsistent")
    if int(manifest["rewire"]["seed"]) != rewire_seeds[0]:
        raise ValueError("designated primary rewire must be the first sealed ensemble seed")
    swaps_per_edge = int(ensemble_cfg["swaps_per_edge"])
    rewired_bundles: list[GraphBundle] = []
    for rewire_seed in rewire_seeds:
        rewired = degree_preserving_rewire(
            biological,
            seed=rewire_seed,
            swaps_per_edge=swaps_per_edge,
        )
        rewired.validate(require_sign=True, require_qualified=True)
        rewire_manifest = (rewired.manifest or {}).get("rewire", {})
        if not rewire_manifest.get("mixing_complete", False):
            raise ValueError(
                "degree-preserving null failed to complete its sealed swap target: "
                f"seed={rewire_seed} accepted={rewire_manifest.get('accepted_swaps')} "
                f"target={rewire_manifest.get('target_swaps')}"
            )
        rewired_bundles.append(rewired)

    lesion_cfg = manifest["lesion"]
    if lesion_cfg.get("kind") != "remove-incoming-edges-to-roles":
        raise ValueError(f"unsupported sealed lesion kind: {lesion_cfg.get('kind')!r}")
    lesioned = lesion_incoming_to_roles(biological, lesion_cfg["roles"])
    lesioned.validate(require_sign=True, require_qualified=True)

    cfg = manifest["config"]
    arena = ArenaConfig(**cfg["arena"])
    id_plume = PlumeConfig(**cfg["plume"])
    sensors = SensorConfig(**cfg["sensor"])
    ood_plume = PlumeConfig(**manifest["ood_plume"])

    # All graph controls and the classical baseline see the identical environment
    # seed on each paired episode. The neural model and environment share one
    # sealed simulation clock.
    factories = {
        "malecns": lambda: MaleCNSRateController(biological, model_dt_s=arena.dt),
        "lesion": lambda: MaleCNSRateController(lesioned, model_dt_s=arena.dt),
        "classical": CastSurgeController,
    }
    null_labels: list[str] = []
    for index, rewired in enumerate(rewired_bundles):
        label = "rewire" if index == 0 else f"rewire_{index:02d}"
        null_labels.append(label)
        factories[label] = (
            lambda graph=rewired: MaleCNSRateController(graph, model_dt_s=arena.dt)
        )

    id_frame = evaluate(
        factories,
        [int(x) for x in manifest["heldout_seeds"]],
        arena=arena,
        plume=id_plume,
        sensors=sensors,
    )
    ood_frame = evaluate(
        factories,
        [int(x) for x in manifest["ood_seeds"]],
        arena=arena,
        plume=ood_plume,
        sensors=sensors,
    )
    report = _gold_report(id_frame, ood_frame, manifest, null_labels)

    out.mkdir(parents=True, exist_ok=False)
    id_frame.to_parquet(out / "heldout_episodes.parquet", index=False)
    ood_frame.to_parquet(out / "ood_episodes.parquet", index=False)
    (out / "gold_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_receipt(
        out / "receipt.json",
        manifest,
        {
            "code_ref": verified_code_ref,
            "circuit_sha256": observed_circuit_sha,
            "rewire_manifests": [bundle.manifest for bundle in rewired_bundles],
            "lesion_manifest": lesioned.manifest,
            "gold_report": report,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
