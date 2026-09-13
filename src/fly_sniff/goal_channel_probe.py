from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph import GraphBundle, MaleCNSRateController
from .literature_route_audit import _normalize_edges


PRIMARY_DIRECT_FAMILIES = (
    ("FB5AB", "hDeltaC"),
    ("PFNa_family", "hDeltaC"),
    ("PFNm_family", "hDeltaC"),
    ("PFNp_family", "hDeltaC"),
    ("hDeltaC", "hDeltaG"),
    ("hDeltaG", "PFL3"),
    ("hDeltaG", "PFL2"),
)
PFN_POPULATIONS = ("PFNa_family", "PFNm_family", "PFNp_family")
PFN_EDGE_FAMILIES = {f"{name}->hDeltaC" for name in PFN_POPULATIONS}
PRIMARY_OUTPUT_FAMILIES = {"hDeltaG->PFL3", "hDeltaG->PFL2"}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _population_rows(route: dict[str, Any], name: str) -> list[dict[str, Any]]:
    try:
        rows = route["populations"][name]["rows"]
    except KeyError as exc:
        raise ValueError(f"route audit missing population {name!r}") from exc
    return [dict(row) for row in rows]


def _population_ids(route: dict[str, Any], name: str) -> list[int]:
    return sorted(int(row["bodyId"]) for row in _population_rows(route, name))


def _body_type_map(route: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for population in route.get("populations", {}).values():
        for row in population.get("rows", []):
            body_id = int(row["bodyId"])
            cell_type = str(row.get("type", "")).strip()
            if body_id in mapping and mapping[body_id] != cell_type:
                raise ValueError(f"body ID {body_id} has inconsistent type labels")
            mapping[body_id] = cell_type
    return mapping


def _direct_prediction(route: dict[str, Any], source: str, target: str) -> dict[str, Any]:
    matches = [
        row
        for row in route.get("direct_predictions", [])
        if str(row.get("source")) == source and str(row.get("target")) == target
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one direct prediction {source}->{target}; observed {len(matches)}"
        )
    return matches[0]


def _population_sign_lookup(
    sign_report: dict[str, Any], route: dict[str, Any]
) -> dict[int, int]:
    body_types = _body_type_map(route)
    lookup: dict[int, int] = {}
    for population_name, population in sign_report.get("population_signs", {}).items():
        evidence = population.get("type_evidence", {})
        for body_id in population.get("body_ids", []):
            body_id = int(body_id)
            cell_type = body_types.get(body_id)
            if cell_type is None:
                raise ValueError(
                    f"sign report population {population_name!r} body {body_id} "
                    "is absent from route population metadata"
                )
            row = evidence.get(cell_type)
            if row is None:
                raise ValueError(
                    f"sign report has no type evidence for {cell_type!r} body {body_id}"
                )
            lookup[body_id] = int(row.get("modeled_sign", 0))
    return lookup


def _primary_nodes(route: dict[str, Any]) -> pd.DataFrame:
    rows: dict[int, dict[str, Any]] = {}
    for name in (
        "FB5AB",
        "PFNa_family",
        "PFNm_family",
        "PFNp_family",
        "hDeltaC",
        "hDeltaG",
        "PFL3",
        "PFL2",
    ):
        for row in _population_rows(route, name):
            body_id = int(row["bodyId"])
            rows[body_id] = {
                "bodyId": body_id,
                "type": str(row.get("type", "")),
                "instance": str(row.get("instance", "")),
            }
    return pd.DataFrame([rows[k] for k in sorted(rows)])


def _primary_roles(
    route: dict[str, Any], topography: dict[str, Any]
) -> dict[str, list[int]]:
    roles = {
        "probe_PFNa_family": _population_ids(route, "PFNa_family"),
        "probe_PFNm_family": _population_ids(route, "PFNm_family"),
        "probe_PFNp_family": _population_ids(route, "PFNp_family"),
        "odor_context_FB5AB": _population_ids(route, "FB5AB"),
        "integrator_hDeltaC": _population_ids(route, "hDeltaC"),
        "relay_hDeltaG": _population_ids(route, "hDeltaG"),
        "output_PFL3": _population_ids(route, "PFL3"),
        "output_PFL2": _population_ids(route, "PFL2"),
    }
    topo_pops = topography.get("populations", {})
    for population in ("PFNa_family", "PFNm_family"):
        body_columns = topo_pops.get(population, {}).get("body_columns", {})
        by_column: dict[int, list[int]] = {}
        for body_id_text, column in body_columns.items():
            by_column.setdefault(int(column), []).append(int(body_id_text))
        for column, ids in sorted(by_column.items()):
            roles[f"probe_{population}_C{column}"] = sorted(ids)
    return roles


def _primary_edges(
    route: dict[str, Any], sign_report: dict[str, Any], threshold: float
) -> pd.DataFrame:
    sign_lookup = _population_sign_lookup(sign_report, route)
    records: list[dict[str, Any]] = []
    for source_name, target_name in PRIMARY_DIRECT_FAMILIES:
        prediction = _direct_prediction(route, source_name, target_name)
        family = f"{source_name}->{target_name}"
        for edge in prediction.get("edges", []):
            weight = float(edge["weight"])
            if weight < threshold:
                continue
            source = int(edge["source"])
            target = int(edge["target"])
            modeled_sign = sign_lookup.get(source)
            if modeled_sign is None:
                raise ValueError(f"no modeled sign for source body {source} on {family}")
            records.append(
                {
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "sign": int(modeled_sign),
                    "edge_family": family,
                }
            )
    if not records:
        return pd.DataFrame(columns=["source", "target", "weight", "sign", "edge_family"])
    frame = pd.DataFrame(records)
    return (
        frame.groupby(["source", "target", "sign", "edge_family"], as_index=False)
        .weight.sum()
        .sort_values(["source", "target", "edge_family"])
        .reset_index(drop=True)
    )


def build_primary_bundle(
    route: dict[str, Any],
    sign_report: dict[str, Any],
    topography: dict[str, Any],
    threshold: float,
) -> GraphBundle:
    nodes = _primary_nodes(route)
    edges = _primary_edges(route, sign_report, threshold)
    roles = _primary_roles(route, topography)
    manifest = {
        "dataset": route.get("dataset", "unknown"),
        "qualification_status": "candidate",
        "experiment": "E002b-goal-channel-propagation-v1",
        "structural_threshold": float(threshold),
        "scope": "restricted goal-channel candidate; no heading or motor-side semantics",
    }
    bundle = GraphBundle(nodes, edges, roles, manifest)
    bundle.validate(require_sign=True, require_qualified=False)
    return bundle


def _cut_families(bundle: GraphBundle, families: set[str]) -> GraphBundle:
    if "edge_family" not in bundle.edges.columns:
        raise ValueError("goal-channel edges require edge_family provenance")
    edges = bundle.edges.loc[~bundle.edges.edge_family.isin(families)].copy()
    manifest = dict(bundle.manifest or {})
    manifest["qualification_status"] = "candidate"
    manifest["mechanistic_lesion"] = {"removed_edge_families": sorted(families)}
    return GraphBundle(bundle.nodes.copy(), edges, dict(bundle.roles), manifest)


def _zero_sign_for_sources(bundle: GraphBundle, source_ids: set[int]) -> GraphBundle:
    edges = bundle.edges.copy()
    mask = edges.source.astype(int).isin(source_ids)
    edges.loc[mask, "sign"] = 0
    manifest = dict(bundle.manifest or {})
    manifest["qualification_status"] = "candidate"
    manifest["modeled_sign_sensitivity"] = {
        "zeroed_source_body_ids": sorted(source_ids),
        "affected_edge_count": int(mask.sum()),
    }
    return GraphBundle(bundle.nodes.copy(), edges, dict(bundle.roles), manifest)


def _role_mean(controller: MaleCNSRateController, role: str) -> float:
    idx = [
        controller.index[body_id]
        for body_id in controller.bundle.roles.get(role, [])
        if body_id in controller.index
    ]
    return float(controller.activity[idx].mean()) if idx else 0.0


def _run_trace(
    bundle: GraphBundle,
    role_drive: dict[str, float],
    *,
    seed: int,
    steps: int,
    pulse_steps: int,
) -> dict[str, Any]:
    controller = MaleCNSRateController(bundle, require_qualified=False)
    controller.reset(seed)
    activity_trace: list[np.ndarray] = []
    population_trace: list[dict[str, float]] = []
    for step in range(steps):
        active = role_drive if step < pulse_steps else {}
        controller.act_role_drive(active)
        activity_trace.append(controller.activity.copy())
        population_trace.append(
            {
                "hDeltaC": _role_mean(controller, "integrator_hDeltaC"),
                "hDeltaG": _role_mean(controller, "relay_hDeltaG"),
                "PFL3": _role_mean(controller, "output_PFL3"),
                "PFL2": _role_mean(controller, "output_PFL2"),
            }
        )
    activity = np.vstack(activity_trace)
    pfl3 = np.asarray([row["PFL3"] for row in population_trace], dtype=float)
    pfl2 = np.asarray([row["PFL2"] for row in population_trace], dtype=float)
    tail = max(4, steps // 4)
    return {
        "activity": activity,
        "population_trace": population_trace,
        "summary": {
            "pfl3_peak_abs": float(np.max(np.abs(pfl3))),
            "pfl2_peak_abs": float(np.max(np.abs(pfl2))),
            "pfl3_tail_mean_abs": float(np.mean(np.abs(pfl3[-tail:]))),
            "pfl2_tail_mean_abs": float(np.mean(np.abs(pfl2[-tail:]))),
            "hDeltaC_peak_abs": float(max(abs(row["hDeltaC"]) for row in population_trace)),
            "hDeltaG_peak_abs": float(max(abs(row["hDeltaG"]) for row in population_trace)),
        },
    }


def _max_output_peak(run: dict[str, Any]) -> float:
    summary = run["summary"]
    return float(max(summary["pfl3_peak_abs"], summary["pfl2_peak_abs"]))


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
    report = {
        "drive": {str(k): float(v) for k, v in drive.items()},
        "summary": first["summary"],
        "deterministic_replay_error": error,
    }
    return report, error


def _artifact_hashes_match(
    seal: dict[str, Any],
    *,
    route_path: str | Path,
    sign_path: str | Path,
    topography_path: str | Path,
) -> tuple[bool, dict[str, Any]]:
    observed = {
        "integration_route_audit_v1": _sha256_file(route_path),
        "integration_sign_audit_v2": _sha256_file(sign_path),
        "integration_topography_audit_v1": _sha256_file(topography_path),
    }
    expected = {key: str(seal["artifacts"][key]["sha256"]) for key in observed}
    return observed == expected, {"observed": observed, "expected": expected}


def _column_probe_reports(
    bundle: GraphBundle, *, seed: int, steps: int, pulse_steps: int
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for role in sorted(bundle.roles):
        if not (
            role.startswith("probe_PFNa_family_C")
            or role.startswith("probe_PFNm_family_C")
        ):
            continue
        run, error = _condition(
            bundle, {role: 1.0}, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        reports[role] = {
            "body_id_count": len(bundle.roles[role]),
            "summary": run["summary"],
            "deterministic_replay_error": error,
            "directional_role_status": "unresolved",
        }
    return reports


def _hdelta_m_comparator_bundle(
    primary: GraphBundle,
    raw_weights: pd.DataFrame,
    goal_relay_authority: dict[str, Any],
    hdelta_m_sign_authority: dict[str, Any],
    threshold: float,
) -> GraphBundle:
    normalized = _normalize_edges(raw_weights)
    body_ids = goal_relay_authority["population_body_ids"]
    hdc = {int(x) for x in body_ids["hDeltaC"]}
    hdm = {int(x) for x in body_ids["hDeltaM"]}
    pfl3 = {int(x) for x in body_ids["PFL3"]}
    pfl2 = {int(x) for x in body_ids["PFL2"]}
    hdm_sign = int(hdelta_m_sign_authority["prediction"]["modeled_sign"])
    if hdm_sign == 0:
        raise ValueError("hDeltaM comparator authority has unresolved modeled sign")

    keep_primary = primary.edges.loc[
        primary.edges.edge_family.isin(PFN_EDGE_FAMILIES | {"FB5AB->hDeltaC"})
    ].copy()

    def select(
        source_ids: set[int], target_ids: set[int], family: str, sign: int
    ) -> pd.DataFrame:
        rows = normalized.loc[
            normalized.source.astype(int).isin(source_ids)
            & normalized.target.astype(int).isin(target_ids)
            & (normalized.weight.astype(float) >= threshold),
            ["source", "target", "weight"],
        ].copy()
        if rows.empty:
            return pd.DataFrame(columns=["source", "target", "weight", "sign", "edge_family"])
        rows["sign"] = int(sign)
        rows["edge_family"] = family
        return rows

    hdc_to_hdm = select(hdc, hdm, "hDeltaC->hDeltaM", 1)
    hdm_to_pfl3 = select(hdm, pfl3, "hDeltaM->PFL3", hdm_sign)
    hdm_to_pfl2 = select(hdm, pfl2, "hDeltaM->PFL2", hdm_sign)
    edges = pd.concat([keep_primary, hdc_to_hdm, hdm_to_pfl3, hdm_to_pfl2], ignore_index=True)

    nodes = primary.nodes.copy()
    existing = set(nodes.bodyId.astype(int))
    extra = [
        {"bodyId": body_id, "type": "hDeltaM", "instance": ""}
        for body_id in sorted(hdm - existing)
    ]
    if extra:
        nodes = pd.concat([nodes, pd.DataFrame(extra)], ignore_index=True)
    roles = dict(primary.roles)
    roles.pop("relay_hDeltaG", None)
    roles["relay_hDeltaM"] = sorted(hdm)
    manifest = dict(primary.manifest or {})
    manifest["relay_comparator"] = "hDeltaM"
    manifest["qualification_status"] = "candidate"
    bundle = GraphBundle(nodes, edges, roles, manifest)
    bundle.validate(require_sign=True, require_qualified=False)
    return bundle


def probe_goal_channel(
    route: dict[str, Any],
    sign_report: dict[str, Any],
    topography: dict[str, Any],
    protocol: dict[str, Any],
    seal: dict[str, Any],
    *,
    seed: int,
    steps: int,
    pulse_steps: int,
    drive_amplitude: float,
    artifact_hash_gate: tuple[bool, dict[str, Any]] | None = None,
    raw_weights: pd.DataFrame | None = None,
    goal_relay_authority: dict[str, Any] | None = None,
    hdelta_m_sign_authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if steps < 4:
        raise ValueError("steps must be >= 4")
    if not 1 <= pulse_steps <= steps:
        raise ValueError("pulse_steps must be in [1, steps]")
    if not np.isfinite(drive_amplitude) or drive_amplitude <= 0.0:
        raise ValueError("drive_amplitude must be finite and > 0")
    if str(protocol.get("protocol")) != "E002b-goal-channel-propagation-v1":
        raise ValueError("unexpected E002b protocol")
    if str(topography.get("directional_role_status")) != "unresolved":
        raise ValueError("topography audit must keep directional roles unresolved")
    if not bool(sign_report.get("ready_for_modeled_sign_probe")):
        raise ValueError("sign-v2 report is not ready for modeled sign probe")

    thresholds = [float(x) for x in protocol["fixed_structural_thresholds"]]
    probe_roles = {
        "PFNa_family": "probe_PFNa_family",
        "PFNm_family": "probe_PFNm_family",
        "PFNp_family": "probe_PFNp_family",
    }
    threshold_reports: dict[str, Any] = {}
    deterministic_errors: list[float] = []
    gate_off_peaks: list[float] = []
    hdc_cut_peaks: list[float] = []
    hdg_cut_peaks: list[float] = []

    sensitivity_case = sign_report["mandatory_sensitivity_cases"][0]
    pfn_b_ids = {int(x) for x in sensitivity_case["affected_body_ids"]}

    for threshold in thresholds:
        bundle = build_primary_bundle(route, sign_report, topography, threshold)
        intact: dict[str, Any] = {}
        gate_off: dict[str, Any] = {}
        hdc_cut: dict[str, Any] = {}
        hdg_cut: dict[str, Any] = {}
        additive: dict[str, Any] = {}

        gated_bundle = _cut_families(bundle, PFN_EDGE_FAMILIES)
        hdc_cut_bundle = _cut_families(bundle, {"hDeltaC->hDeltaG"})
        hdg_cut_bundle = _cut_families(bundle, PRIMARY_OUTPUT_FAMILIES)

        for name, role in probe_roles.items():
            drive = {role: drive_amplitude}
            intact_run, error = _condition(
                bundle, drive, seed=seed, steps=steps, pulse_steps=pulse_steps
            )
            deterministic_errors.append(error)
            intact[name] = intact_run

            run, error = _condition(
                gated_bundle, drive, seed=seed, steps=steps, pulse_steps=pulse_steps
            )
            deterministic_errors.append(error)
            gate_off[name] = run
            gate_off_peaks.append(_max_output_peak(run))

            run, error = _condition(
                hdc_cut_bundle, drive, seed=seed, steps=steps, pulse_steps=pulse_steps
            )
            deterministic_errors.append(error)
            hdc_cut[name] = run
            hdc_cut_peaks.append(_max_output_peak(run))

            run, error = _condition(
                hdg_cut_bundle, drive, seed=seed, steps=steps, pulse_steps=pulse_steps
            )
            deterministic_errors.append(error)
            hdg_cut[name] = run
            hdg_cut_peaks.append(_max_output_peak(run))

            additive_drive = {
                role: drive_amplitude,
                "odor_context_FB5AB": drive_amplitude,
            }
            run, error = _condition(
                bundle,
                additive_drive,
                seed=seed,
                steps=steps,
                pulse_steps=pulse_steps,
            )
            deterministic_errors.append(error)
            additive[name] = run

        all_pfn_drive = {role: drive_amplitude for role in probe_roles.values()}
        combined, error = _condition(
            bundle, all_pfn_drive, seed=seed, steps=steps, pulse_steps=pulse_steps
        )
        deterministic_errors.append(error)
        partial_lesions: dict[str, Any] = {}
        for population, family in (
            ("PFNa_family", "PFNa_family->hDeltaC"),
            ("PFNm_family", "PFNm_family->hDeltaC"),
            ("PFNp_family", "PFNp_family->hDeltaC"),
        ):
            lesioned = _cut_families(bundle, {family})
            run, error = _condition(
                lesioned,
                all_pfn_drive,
                seed=seed,
                steps=steps,
                pulse_steps=pulse_steps,
            )
            deterministic_errors.append(error)
            partial_lesions[f"without_{population}"] = run

        sign_zero = _zero_sign_for_sources(bundle, pfn_b_ids)
        pfn_p_sensitivity, error = _condition(
            sign_zero,
            {"probe_PFNp_family": drive_amplitude},
            seed=seed,
            steps=steps,
            pulse_steps=pulse_steps,
        )
        deterministic_errors.append(error)

        comparator: dict[str, Any]
        if (
            raw_weights is not None
            and goal_relay_authority is not None
            and hdelta_m_sign_authority is not None
        ):
            comp_bundle = _hdelta_m_comparator_bundle(
                bundle,
                raw_weights,
                goal_relay_authority,
                hdelta_m_sign_authority,
                threshold,
            )
            comp_runs: dict[str, Any] = {}
            for name, role in probe_roles.items():
                run, error = _condition(
                    comp_bundle,
                    {role: drive_amplitude},
                    seed=seed,
                    steps=steps,
                    pulse_steps=pulse_steps,
                )
                deterministic_errors.append(error)
                comp_runs[name] = run
            comparator = {
                "status": "modeled",
                "relay": "hDeltaM",
                "runs": comp_runs,
                "selection_status": "frozen_comparator_not_eligible_for_promotion",
            }
        else:
            comparator = {
                "status": "not_run_missing_raw_comparator_inputs",
                "relay": "hDeltaM",
                "selection_status": "frozen_comparator_not_eligible_for_promotion",
            }

        key = str(int(threshold) if threshold.is_integer() else threshold)
        threshold_reports[key] = {
            "structural_threshold": threshold,
            "intact_gate_model": intact,
            "odor_gate_off": gate_off,
            "hDeltaC_output_cut": hdc_cut,
            "hDeltaG_output_cut": hdg_cut,
            "additive_FB5AB_ACh_comparator": additive,
            "combined_PFN_drive": combined,
            "partial_input_family_lesions": partial_lesions,
            "PFNp_b_sign_zero_sensitivity": {
                "affected_body_id_count": len(pfn_b_ids),
                "run": pfn_p_sensitivity,
            },
            "hDeltaM_relay_comparator": comparator,
            "column_impulses": _column_probe_reports(
                bundle, seed=seed, steps=steps, pulse_steps=pulse_steps
            ),
        }

    if artifact_hash_gate is None:
        hash_passed, hash_detail = True, {"status": "not-evaluated-in-memory"}
    else:
        hash_passed, hash_detail = artifact_hash_gate

    max_deterministic_error = max(deterministic_errors, default=0.0)
    max_gate_off = max(gate_off_peaks, default=0.0)
    max_hdc_cut = max(hdc_cut_peaks, default=0.0)
    max_hdg_cut = max(hdg_cut_peaks, default=0.0)

    gates = [
        {
            "name": "authority_hashes_match",
            "passed": bool(hash_passed),
            "detail": hash_detail,
            "criterion": "sealed route/sign/topography artifact SHA-256 values match",
        },
        {
            "name": "sign_v2_ready",
            "passed": bool(sign_report.get("ready_for_modeled_sign_probe")),
            "criterion": "integration-sign-audit-v2 is ready for modeled sign probe",
        },
        {
            "name": "no_behavioral_oracle_inputs",
            "passed": True,
            "accepted_inputs": [
                "sealed structural route",
                "sealed type-level modeled signs",
                "descriptive anatomical columns",
                "abstract named-role perturbations",
            ],
            "criterion": (
                "probe API accepts no source coordinates, culprit identity, distance, "
                "plume image, reward, or behavioral success signal"
            ),
        },
        {
            "name": "deterministic_replay",
            "passed": max_deterministic_error <= 1e-12,
            "value": max_deterministic_error,
            "criterion": "all repeated modeled traces reproduce with max error <= 1e-12",
        },
        {
            "name": "primary_route_hDeltaC_cut",
            "passed": max_hdc_cut <= 1e-12,
            "value": max_hdc_cut,
            "criterion": "cutting hDeltaC->hDeltaG abolishes PFL propagation from PFN probes",
        },
        {
            "name": "primary_route_hDeltaG_cut",
            "passed": max_hdg_cut <= 1e-12,
            "value": max_hdg_cut,
            "criterion": "cutting hDeltaG->PFL3/PFL2 abolishes PFL propagation",
        },
        {
            "name": "odor_gate_negative_control",
            "passed": max_gate_off <= 1e-12,
            "value": max_gate_off,
            "criterion": (
                "under the primary gate abstraction, removing PFN->hDeltaC transmission "
                "when odor context is off suppresses downstream PFL propagation"
            ),
        },
    ]

    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "protocol": "E002b-goal-channel-propagation-v1",
        "dataset": route.get("dataset", "unknown"),
        "passed": passed,
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(gate["passed"]) for gate in gates),
        "gates": gates,
        "run_config": {
            "seed": int(seed),
            "steps": int(steps),
            "pulse_steps": int(pulse_steps),
            "drive_amplitude": float(drive_amplitude),
            "thresholds": thresholds,
            "model_semantics": "FB5AB odor-context gate on PFN->hDeltaC transmission",
        },
        "threshold_reports": threshold_reports,
        "mandatory_sensitivity_cases": sign_report.get("mandatory_sensitivity_cases", []),
        "topography_status": {
            "directional_role_status": topography.get("directional_role_status"),
            "PFNp_column_parse_fraction": topography.get("populations", {})
            .get("PFNp_family", {})
            .get("column_parse_fraction"),
        },
        "claim_boundary": str(protocol.get("claim_boundary", "")),
        "explicit_nonclaims": [
            "No physical wind angle is assigned to an anatomical column.",
            "No PFL activity is labeled a steering command.",
            "No DNa02 turn is computed.",
            "No modeled state is described as measured neural firing.",
            "No odor-source navigation or intact-versus-rewire claim is tested.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered E002b restricted MaleCNS goal-channel probe"
    )
    parser.add_argument("route_audit")
    parser.add_argument("sign_audit")
    parser.add_argument("topography_audit")
    parser.add_argument("--protocol", default="configs/e002b_goal_channel_protocol_v1.json")
    parser.add_argument("--input-seal", default="authority/malecns-v1.0-e002b-input-seal.json")
    parser.add_argument(
        "--weights",
        default="data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    )
    parser.add_argument(
        "--goal-relay-authority",
        default="authority/malecns-v1.0-goal-relay-evidence.json",
    )
    parser.add_argument(
        "--hdelta-m-sign-authority",
        default="authority/malecns-v1.0-hDeltaM-transmitter-evidence.json",
    )
    parser.add_argument("--output", default="results/e002/goal-channel-v1.json")
    parser.add_argument("--seed", type=int, default=22002)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--pulse-steps", type=int, default=8)
    parser.add_argument("--drive-amplitude", type=float, default=1.0)
    args = parser.parse_args()

    route = _load_json(args.route_audit)
    sign_report = _load_json(args.sign_audit)
    topography = _load_json(args.topography_audit)
    protocol = _load_json(args.protocol)
    seal = _load_json(args.input_seal)
    hash_gate = _artifact_hashes_match(
        seal,
        route_path=args.route_audit,
        sign_path=args.sign_audit,
        topography_path=args.topography_audit,
    )

    weights_path = Path(args.weights)
    expected_weights_sha = str(seal["weights_sha256"])
    observed_weights_sha = _sha256_file(weights_path)
    if observed_weights_sha != expected_weights_sha:
        raise ValueError(
            f"raw weights SHA-256 mismatch: {observed_weights_sha} != {expected_weights_sha}"
        )
    raw_weights = pd.read_feather(weights_path)
    goal_relay_authority = _load_json(args.goal_relay_authority)
    hdelta_m_sign_authority = _load_json(args.hdelta_m_sign_authority)

    report = probe_goal_channel(
        route,
        sign_report,
        topography,
        protocol,
        seal,
        seed=args.seed,
        steps=args.steps,
        pulse_steps=args.pulse_steps,
        drive_amplitude=args.drive_amplitude,
        artifact_hash_gate=hash_gate,
        raw_weights=raw_weights,
        goal_relay_authority=goal_relay_authority,
        hdelta_m_sign_authority=hdelta_m_sign_authority,
    )
    report["inputs"] = {
        "route_audit": {"path": str(args.route_audit), "sha256": _sha256_file(args.route_audit)},
        "sign_audit": {"path": str(args.sign_audit), "sha256": _sha256_file(args.sign_audit)},
        "topography_audit": {
            "path": str(args.topography_audit),
            "sha256": _sha256_file(args.topography_audit),
        },
        "weights": {"path": str(weights_path), "sha256": observed_weights_sha},
        "protocol": {"path": str(args.protocol), "sha256": _sha256_file(args.protocol)},
        "input_seal": {"path": str(args.input_seal), "sha256": _sha256_file(args.input_seal)},
        "goal_relay_authority": {
            "path": str(args.goal_relay_authority),
            "sha256": _sha256_file(args.goal_relay_authority),
        },
        "hdelta_m_sign_authority": {
            "path": str(args.hdelta_m_sign_authority),
            "sha256": _sha256_file(args.hdelta_m_sign_authority),
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
