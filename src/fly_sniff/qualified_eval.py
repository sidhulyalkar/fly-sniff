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


def verify_code_ref(manifest: dict) -> str:
    expected = str(manifest.get("code_ref", ""))
    actual = current_git_ref()
    if not expected or expected == "UNKNOWN":
        raise ValueError("sealed final manifest must name an exact git commit")
    if actual == "UNKNOWN":
        raise ValueError("cannot establish current git commit; run final evaluation from the repository checkout")
    if actual != expected:
        raise ValueError(f"code-ref mismatch: manifest={expected} checkout={actual}")
    return actual


def _gold_report(id_frame: pd.DataFrame, ood_frame: pd.DataFrame, manifest: dict) -> dict:
    id_summary = summarize(id_frame).set_index("label")
    ood_summary = summarize(ood_frame).set_index("label")
    paired_spl = paired_spl_report(id_frame, "malecns", "rewire")
    paired_success = paired_success_report(id_frame, "malecns", "rewire")
    lesion_spl = paired_spl_report(id_frame, "malecns", "lesion")
    lesion_success = paired_success_report(id_frame, "malecns", "lesion")
    gold = manifest["gold"]
    sr = float(id_summary.loc["malecns", "success_rate"])
    ood_sr = float(ood_summary.loc["malecns", "success_rate"])
    checks = {
        "id_success": sr >= float(gold["success_rate_min"]),
        "spl_delta": paired_spl["mean_delta"] >= float(gold["spl_delta_vs_rewire_min"]),
        "ci_excludes_zero": paired_spl["ci95_low"] > 0.0,
        "ood_success": ood_sr >= float(gold["ood_success_rate_min"]),
    }
    return {
        "flynav_gold": bool(all(checks.values())),
        "checks": checks,
        "id_success_rate": sr,
        "ood_success_rate": ood_sr,
        "malecns_vs_rewire_spl": paired_spl,
        "malecns_vs_rewire_success": paired_success,
        "malecns_vs_lesion_spl": lesion_spl,
        "malecns_vs_lesion_success": lesion_success,
        "id_summary": id_summary.reset_index().to_dict(orient="records"),
        "ood_summary": ood_summary.reset_index().to_dict(orient="records"),
        "control_contract": {
            "rewire": manifest["rewire"],
            "lesion": manifest["lesion"],
            "paired_environment": (
                "malecns, rewire, lesion, and classical controllers are evaluated on the same "
                "seed lists and plume configuration within each cohort"
            ),
        },
        "statistical_note": (
            "Gold gating remains preregistered on absolute held-out/OOD success and paired SPL "
            "against the degree-preserving rewire. The matched lesion and paired success reports "
            "are additional prespecified dependency/control evidence, not post-hoc replacement gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sealed qualified MaleCNS benchmark")
    parser.add_argument("circuit", help="qualified GraphBundle directory")
    parser.add_argument("manifest", help="sealed final-test JSON")
    parser.add_argument("--output", default="results/final-v1")
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
    rew_cfg = manifest["rewire"]
    rewired = degree_preserving_rewire(
        biological,
        seed=int(rew_cfg["seed"]),
        swaps_per_edge=int(rew_cfg["swaps_per_edge"]),
    )
    rewired.validate(require_sign=True, require_qualified=True)

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
        "rewire": lambda: MaleCNSRateController(rewired, model_dt_s=arena.dt),
        "lesion": lambda: MaleCNSRateController(lesioned, model_dt_s=arena.dt),
        "classical": CastSurgeController,
    }
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
    report = _gold_report(id_frame, ood_frame, manifest)

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
            "rewire_manifest": rewired.manifest,
            "lesion_manifest": lesioned.manifest,
            "gold_report": report,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
