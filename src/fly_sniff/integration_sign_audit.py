from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .literature_route_audit import _git_output, _normalize_edges, _sha256_file
from .signs import infer_nt_column

DEFAULT_CONFIG = Path("configs/integration_sign_audit_v1.json")


def _population_body_ids(integration_audit: dict[str, Any], name: str) -> list[int]:
    populations = integration_audit.get("populations", {})
    if name not in populations:
        raise ValueError(f"integration audit missing population {name!r}")
    rows = populations[name].get("rows", [])
    body_ids = sorted({int(row["bodyId"]) for row in rows if row.get("bodyId") is not None})
    expected_count = int(populations[name].get("count", len(body_ids)))
    if len(body_ids) != expected_count:
        raise ValueError(
            f"integration audit population {name!r} body-ID count mismatch: "
            f"rows={len(body_ids)} expected={expected_count}"
        )
    return body_ids


def _population_sign_summary(
    annotations: pd.DataFrame,
    body_ids: list[int],
    *,
    nt_column: str,
    policy: dict[str, int],
) -> dict[str, Any]:
    rows = annotations.loc[annotations.bodyId.astype(int).isin(body_ids), ["bodyId", nt_column]].copy()
    rows["bodyId"] = rows.bodyId.astype(int)
    rows = rows.drop_duplicates("bodyId")
    found_ids = set(rows.bodyId.astype(int))
    missing_ids = sorted(set(body_ids) - found_ids)

    rows["nt_raw"] = rows[nt_column].fillna("").astype(str).str.strip()
    rows["nt_norm"] = rows.nt_raw.str.lower()
    rows["modeled_sign"] = rows.nt_norm.map(policy).fillna(0).astype(int)

    sign_counts = Counter(int(value) for value in rows.modeled_sign)
    nt_counts = Counter(str(value) if value else "<missing>" for value in rows.nt_raw)
    unresolved_rows = rows.loc[rows.modeled_sign.eq(0), ["bodyId", "nt_raw"]]
    unresolved = [
        {"bodyId": int(row.bodyId), "transmitter": str(row.nt_raw) or None}
        for row in unresolved_rows.itertuples(index=False)
    ]
    unresolved.extend({"bodyId": int(body_id), "transmitter": None} for body_id in missing_ids)

    return {
        "body_id_count": len(body_ids),
        "annotation_rows_found": len(rows),
        "missing_body_ids": missing_ids,
        "transmitter_counts": dict(sorted(nt_counts.items())),
        "modeled_sign_counts": {str(key): int(value) for key, value in sorted(sign_counts.items())},
        "resolved_nonzero_count": int(rows.modeled_sign.ne(0).sum()),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "ready": len(unresolved) == 0,
    }


def _edge_family_summary(
    edges: pd.DataFrame,
    annotations: pd.DataFrame,
    source_ids: list[int],
    target_ids: list[int],
    *,
    nt_column: str,
    policy: dict[str, int],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    detail = edges.loc[
        edges.source.astype(int).isin(source_ids) & edges.target.astype(int).isin(target_ids)
    ].copy()
    lookup = annotations[["bodyId", nt_column]].drop_duplicates("bodyId").copy()
    lookup["bodyId"] = lookup.bodyId.astype(int)
    lookup["nt_norm"] = lookup[nt_column].fillna("").astype(str).str.strip().str.lower()
    lookup["modeled_sign"] = lookup.nt_norm.map(policy).fillna(0).astype(int)
    detail = detail.merge(
        lookup[["bodyId", "modeled_sign"]],
        left_on="source",
        right_on="bodyId",
        how="left",
    ).drop(columns=["bodyId"])
    detail["modeled_sign"] = detail.modeled_sign.fillna(0).astype(int)

    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        retained = detail.loc[detail.weight.astype(float).ge(threshold)]
        resolved = retained.modeled_sign.ne(0)
        sweep.append(
            {
                "min_weight": float(threshold),
                "edge_pairs": len(retained),
                "resolved_signed_edges": int(resolved.sum()),
                "unresolved_edges": int((~resolved).sum()),
                "signed_fraction": float(resolved.mean()) if len(retained) else 0.0,
            }
        )

    return {
        "edge_pairs": len(detail),
        "weight_sum": float(detail.weight.sum()) if len(detail) else 0.0,
        "threshold_sweep": sweep,
    }


def build_integration_sign_audit(
    annotations: pd.DataFrame,
    weights: pd.DataFrame,
    integration_audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if "bodyId" not in annotations.columns:
        raise ValueError("annotations require bodyId")
    annotations = annotations.copy()
    annotations["bodyId"] = annotations.bodyId.astype(int)
    edges = _normalize_edges(weights)

    try:
        nt_column = infer_nt_column(annotations)
    except ValueError as exc:
        return {
            "protocol": str(config.get("protocol", "malecns-integration-sign-audit-v1")),
            "dataset": str(config.get("dataset", "male-cns:v1.0")),
            "ready_for_modeled_sign_probe": False,
            "blocking_reason": "missing_neurotransmitter_annotation_column",
            "blocking_detail": str(exc),
            "annotation_columns": list(annotations.columns),
            "claim_boundary": str(config.get("claim_boundary", "Model sign audit only.")),
        }

    policy = {str(key).lower(): int(value) for key, value in config["model_sign_policy"].items()}
    required = [str(name) for name in config.get("required_presynaptic_populations", [])]
    thresholds = tuple(float(value) for value in config.get("thresholds", [1, 3, 5, 10]))

    population_ids: dict[str, list[int]] = {}
    for name in integration_audit.get("populations", {}):
        population_ids[str(name)] = _population_body_ids(integration_audit, str(name))

    population_signs = {
        name: _population_sign_summary(
            annotations,
            population_ids[name],
            nt_column=nt_column,
            policy=policy,
        )
        for name in required
    }

    edge_families: dict[str, Any] = {}
    for family in config.get("edge_families", []):
        source = str(family["source"])
        target = str(family["target"])
        key = f"{source}->{target}"
        edge_families[key] = _edge_family_summary(
            edges,
            annotations,
            population_ids[source],
            population_ids[target],
            nt_column=nt_column,
            policy=policy,
            thresholds=thresholds,
        )

    ready = all(population_signs[name]["ready"] for name in required)
    return {
        "protocol": str(config.get("protocol", "malecns-integration-sign-audit-v1")),
        "dataset": str(config.get("dataset", "male-cns:v1.0")),
        "neurotransmitter_column": nt_column,
        "model_sign_policy": policy,
        "required_presynaptic_populations": required,
        "population_signs": population_signs,
        "edge_families": edge_families,
        "thresholds": list(thresholds),
        "ready_for_modeled_sign_probe": ready,
        "readiness_rule": str(config.get("readiness_rule", "All required signs resolved.")),
        "claim_boundary": str(config.get("claim_boundary", "Model sign audit only.")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit presynaptic transmitter-derived modeled signs for the integration route"
    )
    parser.add_argument("annotations")
    parser.add_argument("weights")
    parser.add_argument("integration_audit")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="results/route/integration-sign-audit-v1.json")
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    weights_path = Path(args.weights)
    audit_path = Path(args.integration_audit)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())

    observed_audit_sha = _sha256_file(audit_path)
    expected_audit_sha = str(config["expected_integration_audit_sha256"])
    if observed_audit_sha != expected_audit_sha:
        raise SystemExit(
            "integration audit SHA-256 mismatch: "
            f"observed={observed_audit_sha} expected={expected_audit_sha}"
        )

    annotations = pd.read_feather(annotations_path)
    weights = pd.read_feather(weights_path)
    integration_audit = json.loads(audit_path.read_text())
    report = build_integration_sign_audit(annotations, weights, integration_audit, config)
    report["runtime"] = {
        "git_branch": _git_output("branch", "--show-current"),
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_dirty_paths": (_git_output("status", "--porcelain") or "").splitlines(),
    }
    report["inputs"] = {
        "annotations": {"path": str(annotations_path), "sha256": _sha256_file(annotations_path)},
        "weights": {"path": str(weights_path), "sha256": _sha256_file(weights_path)},
        "integration_audit": {"path": str(audit_path), "sha256": observed_audit_sha},
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

    if not report.get("ready_for_modeled_sign_probe", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
