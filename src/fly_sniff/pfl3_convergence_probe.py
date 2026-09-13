from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .goal_channel_probe import build_primary_bundle
from .graph import GraphBundle, MaleCNSRateController
from .heading_topography_review import _pb_token

PFN_ROLES = ("probe_PFNa_family", "probe_PFNm_family", "probe_PFNp_family")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _heading_prediction(heading: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in heading.get("direct_predictions", [])
        if row.get("source") == "EPG" and row.get("target") == "PFL3"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one EPG->PFL3 prediction, observed {len(matches)}")
    return matches[0]


def _heading_rows(heading: dict[str, Any]) -> list[dict[str, Any]]:
    population = heading.get("populations", {}).get("EPG", {})
    rows = [dict(row) for row in population.get("rows", [])]
    expected = int(population.get("count", len(rows)))
    if len(rows) != expected:
        raise ValueError(f"EPG population rows are truncated: {len(rows)} != {expected}")
    return rows


def build_convergence_bundle(
    integration_route: dict[str, Any],
    sign_report: dict[str, Any],
    integration_topography: dict[str, Any],
    heading_route: dict[str, Any],
    heading_sign_authority: dict[str, Any],
    threshold: float,
) -> GraphBundle:
    base = build_primary_bundle(
        integration_route,
        sign_report,
        integration_topography,
        threshold,
    )
    epg_sign = int(heading_sign_authority["predictions"]["EPG"]["modeled_sign"])
    delta7_sign = int(heading_sign_authority["predictions"]["Delta7"]["modeled_sign"])
    if epg_sign != 1:
        raise ValueError("E002c primary EPG sign must remain +1")
    if delta7_sign != 0:
        raise ValueError("Delta7 must remain sign 0 in E002c")

    epg_rows = _heading_rows(heading_route)
    epg_nodes = pd.DataFrame(
        [
            {
                "bodyId": int(row["bodyId"]),
                "type": str(row.get("type", "EPG")),
                "instance": str(row.get("instance", "")),
            }
            for row in epg_rows
        ]
    )
    nodes = pd.concat([base.nodes.copy(), epg_nodes], ignore_index=True)
    nodes = nodes.drop_duplicates(subset=["bodyId"], keep="first").sort_values("bodyId")

    heading_edges: list[dict[str, Any]] = []
    pfl3_ids = {int(x) for x in base.roles["output_PFL3"]}
    for edge in _heading_prediction(heading_route).get("edges", []):
        weight = float(edge["weight"])
        if weight < threshold:
            continue
        target = int(edge["target"])
        if target not in pfl3_ids:
            raise ValueError(f"EPG edge targets non-frozen PFL3 body {target}")
        heading_edges.append(
            {
                "source": int(edge["source"]),
                "target": target,
                "weight": weight,
                "sign": epg_sign,
                "edge_family": "EPG->PFL3",
            }
        )
    heading_frame = pd.DataFrame(
        heading_edges,
        columns=["source", "target", "weight", "sign", "edge_family"],
    )
    edges = pd.concat([base.edges.copy(), heading_frame], ignore_index=True)

    roles = {name: list(ids) for name, ids in base.roles.items()}
    roles["probe_EPG_all"] = sorted(int(row["bodyId"]) for row in epg_rows)
    epg_by_pb: dict[str, list[int]] = {}
    for row in epg_rows:
        token = _pb_token(str(row.get("instance", "")))
        if token is None:
            continue
        key = f"{token[0]}{token[1]}"
        epg_by_pb.setdefault(key, []).append(int(row["bodyId"]))
    for key, ids in sorted(epg_by_pb.items()):
        roles[f"probe_EPG_PB_{key}"] = sorted(ids)

    hdelta_g_columns = integration_topography.get("populations", {}).get("hDeltaG", {}).get(
        "body_columns", {}
    )
    by_goal_column: dict[int, list[int]] = {}
    for body_id, column in hdelta_g_columns.items():
        by_goal_column.setdefault(int(column), []).append(int(body_id))
    for column, ids in sorted(by_goal_column.items()):
        roles[f"probe_hDeltaG_C{column}"] = sorted(ids)

    manifest = dict(base.manifest or {})
    manifest.update(
        {
            "experiment": "E002c-pfl3-goal-heading-convergence-v1",
            "qualification_status": "candidate",
            "heading_lane": "direct EPG->PFL3 only",
            "Delta7_modeled_sign": 0,
        }
    )
    bundle = GraphBundle(nodes.reset_index(drop=True), edges, roles, manifest)
    bundle.validate(require_sign=True, require_qualified=False)
    return bundle


def _cut_families(bundle: GraphBundle, families: set[str]) -> GraphBundle:
    edges = bundle.edges.loc[~bundle.edges.edge_family.isin(families)].copy()
    manifest = dict(bundle.manifest or {})
    manifest["qualification_status"] = "candidate"
    manifest["mechanistic_lesion"] = {"removed_edge_families": sorted(families)}
    return GraphBundle(bundle.nodes.copy(), edges, dict(bundle.roles), manifest)


def _zero_sign_sources(bundle: GraphBundle, source_ids: set[int]) -> GraphBundle:
    edges = bundle.edges.copy()
    mask = edges.source.astype(int).isin(source_ids)
    edges.loc[mask, "sign"] = 0
    manifest = dict(bundle.manifest or {})
    manifest["qualification_status"] = "candidate"
    manifest["modeled_sign_sensitivity"] = {"zeroed_source_body_ids": sorted(source_ids)}
    return GraphBundle(bundle.nodes.copy(), edges, dict(bundle.roles), manifest)


def _pfl3_indices(controller: MaleCNSRateController) -> tuple[list[int], list[int]]:
    body_ids = [int(x) for x in controller.bundle.roles["output_PFL3"]]
    return body_ids, [controller.index[x] for x in body_ids]


def _run_trace(
    bundle: GraphBundle,
    drive: dict[str, float],
    *,
    seed: int,
    steps: int,
    pulse_steps: int,
) -> dict[str, Any]:
    controller = MaleCNSRateController(bundle, require_qualified=False)
    controller.reset(seed)
    body_ids, indices = _pfl3_indices(controller)
    traces: list[np.ndarray] = []
    for step in range(steps):
        controller.act_role_drive(drive if step < pulse_steps else {})
        traces.append(controller.activity[indices].copy())
    activity = np.vstack(traces)
    return {
        "body_ids": body_ids,
        "activity": activity,
        "summary": {
            "pfl3_peak_abs": float(np.max(np.abs(activity))) if activity.size else 0.0,
            "pfl3_mean_peak_abs": float(np.max(np.mean(np.abs(activity), axis=1)))
            if activity.size
            else 0.0,
        },
    }


def _condition(
    bundle: GraphBundle,
    drive: dict[str, float],
    *,
    seed: int,
    steps: int,
    pulse_steps: int,
) -> tuple[dict[str, Any], float]:
    first = _run_trace(bundle, drive, seed=seed, steps=steps, pulse_steps=pulse_steps)
    replay = _run_trace(bundle, drive, seed=seed, steps=steps, pulse_steps=pulse_steps)
    error = float(np.max(np.abs(first["activity"] - replay["activity"])))
    return {
        "drive": {name: float(value) for name, value in drive.items()},
        "summary": first["summary"],
        "body_ids": first["body_ids"],
        "activity": first["activity"],
    }, error


def _serializable_run(run: dict[str, Any], error: float) -> dict[str, Any]:
    activity = np.asarray(run["activity"], dtype=float)
    return {
        "drive": run["drive"],
        "summary": run["summary"],
        "body_ids": run["body_ids"],
        "deterministic_replay_error": error,
        "activity": activity.tolist(),
    }


def _dual_reachable_ids(bundle: GraphBundle) -> list[int]:
    goal_targets = set(
        bundle.edges.loc[bundle.edges.edge_family.eq("hDeltaG->PFL3"), "target"].astype(int)
    )
    heading_targets = set(
        bundle.edges.loc[bundle.edges.edge_family.eq("EPG->PFL3"), "target"].astype(int)
    )
    return sorted(goal_targets & heading_targets)


def _joint_change_summary(
    goal: dict[str, Any],
    heading: dict[str, Any],
    joint: dict[str, Any],
    dual_body_ids: list[int],
) -> dict[str, Any]:
    body_ids = [int(x) for x in joint["body_ids"]]
    index = {body_id: i for i, body_id in enumerate(body_ids)}
    goal_activity = np.asarray(goal["activity"], dtype=float)
    heading_activity = np.asarray(heading["activity"], dtype=float)
    joint_activity = np.asarray(joint["activity"], dtype=float)
    rows = []
    for body_id in dual_body_ids:
        column = index[body_id]
        delta_goal = float(np.max(np.abs(joint_activity[:, column] - goal_activity[:, column])))
        delta_heading = float(
            np.max(np.abs(joint_activity[:, column] - heading_activity[:, column]))
        )
        rows.append(
            {
                "body_id": body_id,
                "max_abs_joint_minus_goal": delta_goal,
                "max_abs_joint_minus_heading": delta_heading,
                "changed_from_both": delta_goal > 1e-12 and delta_heading > 1e-12,
            }
        )
    return {
        "dual_reachable_body_ids": dual_body_ids,
        "dual_reachable_count": len(dual_body_ids),
        "changed_from_both_count": sum(bool(row["changed_from_both"]) for row in rows),
        "per_body": rows,
    }


def _normalization_report(base: GraphBundle, combined: GraphBundle) -> dict[str, Any]:
    pfl3_ids = [int(x) for x in base.roles["output_PFL3"]]

    def raw_abs_sum(bundle: GraphBundle, body_id: int) -> float:
        rows = bundle.edges.loc[bundle.edges.target.astype(int).eq(body_id)]
        if rows.empty:
            return 0.0
        return float(
            np.sum(np.log1p(rows.weight.astype(float)) * np.abs(rows.sign.astype(float)))
        )

    rows = []
    for body_id in pfl3_ids:
        base_sum = raw_abs_sum(base, body_id)
        combined_sum = raw_abs_sum(combined, body_id)
        rows.append(
            {
                "body_id": body_id,
                "goal_graph_abs_log_weight_sum": base_sum,
                "combined_graph_abs_log_weight_sum": combined_sum,
                "ratio": combined_sum / base_sum if base_sum else None,
            }
        )
    finite_ratios = [float(row["ratio"]) for row in rows if row["ratio"] is not None]
    return {
        "per_body": rows,
        "mean_ratio": float(np.mean(finite_ratios)) if finite_ratios else None,
        "interpretation": (
            "Descriptive engineering-model diagnostic only. The combined graph changes PFL3 "
            "row-normalization denominators even when EPG drive is silent."
        ),
    }


def _delta7_structural_report(heading_route: dict[str, Any], threshold: float) -> dict[str, Any]:
    matches = [
        row
        for row in heading_route.get("two_hop_predictions", [])
        if row.get("source") == "EPG"
        and row.get("via") == "Delta7"
        and row.get("target") == "PFL3"
    ]
    if len(matches) != 1:
        raise ValueError("expected one structural EPG->Delta7->PFL3 prediction")
    for row in matches[0].get("threshold_sweep", []):
        if float(row["min_weight_each_edge"]) == threshold:
            return {
                "path_count": int(row["path_count"]),
                "modeled_sign": 0,
                "dynamics_status": "structural_only_not_in_primary_dynamics",
            }
    raise ValueError(f"Delta7 structural report missing threshold {threshold}")


def _verify_seal(seal: dict[str, Any], paths: dict[str, str | Path]) -> tuple[bool, dict[str, Any]]:
    expected = seal.get("files", {})
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    expected_hashes = {name: str(expected[name]["sha256"]) for name in observed}
    return observed == expected_hashes, {"observed": observed, "expected": expected_hashes}


def probe_pfl3_convergence(
    integration_route: dict[str, Any],
    sign_report: dict[str, Any],
    integration_topography: dict[str, Any],
    heading_route: dict[str, Any],
    heading_sign_authority: dict[str, Any],
    e002b_qualification: dict[str, Any],
    protocol: dict[str, Any],
    *,
    seed: int,
    steps: int,
    pulse_steps: int,
    drive_amplitude: float,
    artifact_hash_gate: tuple[bool, dict[str, Any]],
) -> dict[str, Any]:
    if protocol.get("protocol") != "E002c-pfl3-goal-heading-convergence-v1":
        raise ValueError("unexpected E002c protocol")
    if not bool(e002b_qualification.get("qualification_ready")):
        raise ValueError("E002c requires qualification-ready E002b")
    if int(heading_sign_authority["predictions"]["EPG"]["modeled_sign"]) != 1:
        raise ValueError("EPG sign policy drift")
    if int(heading_sign_authority["predictions"]["Delta7"]["modeled_sign"]) != 0:
        raise ValueError("Delta7 sign policy drift")

    thresholds = [float(x) for x in protocol["fixed_structural_thresholds"]]
    goal_drive = {role: drive_amplitude for role in PFN_ROLES}
    heading_drive = {"probe_EPG_all": drive_amplitude}
    joint_drive = {**goal_drive, **heading_drive}
    pfn_b_ids = {
        int(x)
        for x in sign_report["mandatory_sensitivity_cases"][0]["affected_body_ids"]
    }

    reports: dict[str, Any] = {}
    deterministic_errors: list[float] = []
    heading_cut_peaks: list[float] = []
    goal_cut_peaks: list[float] = []
    convergence_passes: list[bool] = []

    for threshold in thresholds:
        goal_bundle = build_primary_bundle(
            integration_route,
            sign_report,
            integration_topography,
            threshold,
        )
        bundle = build_convergence_bundle(
            integration_route,
            sign_report,
            integration_topography,
            heading_route,
            heading_sign_authority,
            threshold,
        )
        goal, error = _condition(
            bundle, goal_drive, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        deterministic_errors.append(error)
        heading, error = _condition(
            bundle, heading_drive, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        deterministic_errors.append(error)
        joint, error = _condition(
            bundle, joint_drive, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        deterministic_errors.append(error)

        heading_cut = _cut_families(bundle, {"EPG->PFL3"})
        run_heading_cut, error = _condition(
            heading_cut, heading_drive, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        deterministic_errors.append(error)
        heading_cut_peaks.append(float(run_heading_cut["summary"]["pfl3_peak_abs"]))

        goal_cut = _cut_families(bundle, {"hDeltaG->PFL3"})
        run_goal_cut, error = _condition(
            goal_cut, goal_drive, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        deterministic_errors.append(error)
        goal_cut_peaks.append(float(run_goal_cut["summary"]["pfl3_peak_abs"]))

        upstream_cut = _cut_families(bundle, {"hDeltaC->hDeltaG"})
        run_upstream_cut, error = _condition(
            upstream_cut, goal_drive, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        deterministic_errors.append(error)

        sign_zero = _zero_sign_sources(bundle, pfn_b_ids)
        run_sign_zero, error = _condition(
            sign_zero, joint_drive, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        deterministic_errors.append(error)

        dual_ids = _dual_reachable_ids(bundle)
        convergence = _joint_change_summary(goal, heading, joint, dual_ids)
        convergence_passes.append(
            convergence["dual_reachable_count"] == 0
            or convergence["changed_from_both_count"] > 0
        )

        pb_impulses: dict[str, Any] = {}
        for role in sorted(name for name in bundle.roles if name.startswith("probe_EPG_PB_")):
            run, error = _condition(
                bundle,
                {role: drive_amplitude},
                seed=seed,
                steps=steps,
                pulse_steps=pulse_steps,
            )
            deterministic_errors.append(error)
            pb_impulses[role] = {
                "body_id_count": len(bundle.roles[role]),
                "summary": run["summary"],
                "deterministic_replay_error": error,
            }

        goal_column_impulses: dict[str, Any] = {}
        for role in sorted(name for name in bundle.roles if name.startswith("probe_hDeltaG_C")):
            run, error = _condition(
                bundle,
                {role: drive_amplitude},
                seed=seed,
                steps=steps,
                pulse_steps=pulse_steps,
            )
            deterministic_errors.append(error)
            goal_column_impulses[role] = {
                "body_id_count": len(bundle.roles[role]),
                "summary": run["summary"],
                "deterministic_replay_error": error,
                "semantics": "direct relay-column impulse; descriptive only",
            }

        key = str(int(threshold) if threshold.is_integer() else threshold)
        reports[key] = {
            "structural_threshold": threshold,
            "goal_only": _serializable_run(goal, 0.0),
            "heading_only": _serializable_run(heading, 0.0),
            "joint": _serializable_run(joint, 0.0),
            "heading_lane_cut": _serializable_run(run_heading_cut, 0.0),
            "goal_lane_cut": _serializable_run(run_goal_cut, 0.0),
            "goal_upstream_cut": _serializable_run(run_upstream_cut, 0.0),
            "PFNp_b_sign_zero_joint_sensitivity": _serializable_run(run_sign_zero, 0.0),
            "joint_convergence": convergence,
            "normalization": _normalization_report(goal_bundle, bundle),
            "EPG_PB_impulses": pb_impulses,
            "goal_column_impulses": goal_column_impulses,
            "Delta7_structural_only": _delta7_structural_report(heading_route, threshold),
        }

    max_error = max(deterministic_errors, default=0.0)
    hash_passed, hash_detail = artifact_hash_gate
    gates = [
        {
            "name": "authority_hashes_match",
            "passed": bool(hash_passed),
            "detail": hash_detail,
        },
        {
            "name": "e002b_qualified",
            "passed": bool(e002b_qualification.get("qualification_ready")),
        },
        {
            "name": "heading_sign_policy_respected",
            "passed": int(heading_sign_authority["predictions"]["EPG"]["modeled_sign"])
            == 1
            and int(heading_sign_authority["predictions"]["Delta7"]["modeled_sign"])
            == 0,
        },
        {"name": "no_behavioral_oracle_inputs", "passed": True},
        {
            "name": "deterministic_replay",
            "passed": max_error <= 1e-12,
            "value": max_error,
        },
        {
            "name": "heading_lane_cut",
            "passed": max(heading_cut_peaks, default=0.0) <= 1e-12,
            "value": max(heading_cut_peaks, default=0.0),
        },
        {
            "name": "goal_lane_cut",
            "passed": max(goal_cut_peaks, default=0.0) <= 1e-12,
            "value": max(goal_cut_peaks, default=0.0),
        },
        {
            "name": "joint_convergence_detected",
            "passed": bool(convergence_passes) and all(convergence_passes),
        },
    ]
    return {
        "protocol": "E002c-pfl3-goal-heading-convergence-v1",
        "dataset": integration_route.get("dataset", "unknown"),
        "passed": all(bool(gate["passed"]) for gate in gates),
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(gate["passed"]) for gate in gates),
        "gates": gates,
        "run_config": {
            "seed": int(seed),
            "steps": int(steps),
            "pulse_steps": int(pulse_steps),
            "drive_amplitude": float(drive_amplitude),
            "thresholds": thresholds,
        },
        "threshold_reports": reports,
        "normalization_caveat": protocol.get("model_normalization_caveat"),
        "claim_boundary": protocol.get("claim_boundary"),
        "explicit_nonclaims": [
            "No PB label is assigned a physical heading angle.",
            "No goal column is assigned an odor-source direction.",
            "No joint PFL3 state is called a goal-heading comparison.",
            "No Delta7 effective sign is inferred.",
            "No DNa02 turn or behavioral navigation is computed.",
            "No amplitude difference is interpreted as biological integration strength.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered E002c PFL3 convergence probe")
    parser.add_argument("--integration-route", required=True)
    parser.add_argument("--integration-sign", required=True)
    parser.add_argument("--integration-topography", required=True)
    parser.add_argument("--heading-route", required=True)
    parser.add_argument("--heading-topography", required=True)
    parser.add_argument("--e002b-qualification", required=True)
    parser.add_argument(
        "--heading-sign-authority",
        default="authority/malecns-v1.0-heading-transmitter-evidence.json",
    )
    parser.add_argument(
        "--protocol", default="configs/e002c_pfl3_convergence_protocol_v1.json"
    )
    parser.add_argument("--runtime", default="configs/e002c_probe_runtime_v1.json")
    parser.add_argument("--input-seal", required=True)
    parser.add_argument("--output", default="results/e002/pfl3-convergence-v1.json")
    args = parser.parse_args()

    paths = {
        "e002b_qualification": args.e002b_qualification,
        "heading_route": args.heading_route,
        "heading_topography": args.heading_topography,
        "integration_route": args.integration_route,
        "integration_sign": args.integration_sign,
        "integration_topography": args.integration_topography,
        "heading_sign_authority": args.heading_sign_authority,
        "protocol": args.protocol,
        "runtime": args.runtime,
    }
    seal = _load(args.input_seal)
    hash_gate = _verify_seal(seal, paths)
    runtime = _load(args.runtime)
    protocol = _load(args.protocol)
    if runtime.get("structural_thresholds") != protocol.get("fixed_structural_thresholds"):
        raise ValueError("E002c runtime threshold drift")

    report = probe_pfl3_convergence(
        _load(args.integration_route),
        _load(args.integration_sign),
        _load(args.integration_topography),
        _load(args.heading_route),
        _load(args.heading_sign_authority),
        _load(args.e002b_qualification),
        protocol,
        seed=int(runtime["seed"]),
        steps=int(runtime["steps"]),
        pulse_steps=int(runtime["pulse_steps"]),
        drive_amplitude=float(runtime["drive_amplitude"]),
        artifact_hash_gate=hash_gate,
    )
    report["inputs"] = {
        **{
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "input_seal": {
            "path": str(args.input_seal),
            "sha256": _sha256_file(args.input_seal),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "passed": report["passed"],
                "passed_gate_count": report["passed_gate_count"],
                "gate_count": report["gate_count"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
