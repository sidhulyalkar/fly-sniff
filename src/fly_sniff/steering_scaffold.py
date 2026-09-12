from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .graph import GraphBundle
from .rewire import save_bundle

DEFAULT_CONFIG = Path("configs/steering_scaffold_candidate_v1.json")
DEFAULT_SIGN_AUTHORITY = Path("authority/malecns-v1.0-steering-sign-evidence.json")

EDGE_FAMILIES = (
    ("PFL3", "DNa02"),
    ("PFL3", "DNa03"),
    ("PFL3", "LAL010"),
    ("DNa03", "DNa02"),
    ("LAL010", "DNa02"),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction(audit: dict[str, Any], source: str, target: str) -> dict[str, Any]:
    hits = [
        row
        for row in audit.get("direct_predictions", [])
        if row.get("source") == source and row.get("target") == target
    ]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one direct prediction {source}->{target}, found {len(hits)}")
    return hits[0]


def _population_ids(audit: dict[str, Any], name: str) -> list[int]:
    pop = audit.get("populations", {}).get(name)
    if not pop:
        raise ValueError(f"missing population {name!r} in route audit")
    return [int(row["bodyId"]) for row in pop.get("rows", [])]


def _load_type_signs(path: str | Path) -> dict[str, int]:
    authority = json.loads(Path(path).read_text())
    rule = authority.get("model_sign_rule", {})
    type_rows = authority.get("types", {})
    out: dict[str, int] = {}
    for name, row in type_rows.items():
        nt = row.get("predicted_neurotransmitter")
        sign = int(rule.get(nt, rule.get("other_or_unclear", 0)))
        out[name] = sign
    return out


def build_steering_scaffold(
    audit: dict[str, Any],
    config: dict[str, Any],
    *,
    type_signs: dict[str, int],
    source_audit_sha256: str | None = None,
    sign_authority_path: str | Path | None = None,
    sign_authority_sha256: str | None = None,
) -> GraphBundle:
    """Build the restricted excitatory steering scaffold from an audited route artifact.

    This intentionally excludes odor-sensory roles, hDeltaC, PFL2, and the
    contralateral inhibitory arm. The resulting graph is a *candidate* model
    scaffold and cannot be promoted to qualified status by this function.
    """
    if audit.get("protocol") != "malecns-literature-route-audit-v1":
        raise ValueError("unexpected route-audit protocol")
    if audit.get("dataset") != config.get("dataset"):
        raise ValueError("route-audit dataset does not match scaffold config")

    expected_sha = config.get("evidence_audit_sha256")
    if expected_sha and source_audit_sha256 and expected_sha != source_audit_sha256:
        raise ValueError("route-audit SHA-256 does not match the sealed scaffold evidence")

    population_names = ("PFL3", "DNa03", "LAL010", "DNa02")
    node_rows: list[dict[str, Any]] = []
    for name in population_names:
        observed = _population_ids(audit, name)
        expected = [int(x) for x in config["populations"][name]]
        if observed != expected:
            raise ValueError(f"population body IDs changed for {name}")
        node_rows.extend(audit["populations"][name]["rows"])

    nodes = pd.DataFrame(node_rows).drop_duplicates(subset=["bodyId"]).reset_index(drop=True)
    edge_rows: list[dict[str, Any]] = []
    for source, target in EDGE_FAMILIES:
        pred = _prediction(audit, source, target)
        sign = int(type_signs.get(source, 0))
        if sign == 0:
            raise ValueError(f"presynaptic sign unresolved for {source}; refusing silent excitation")
        for edge in pred.get("edges", []):
            edge_rows.append(
                {
                    "source": int(edge["source"]),
                    "target": int(edge["target"]),
                    "weight": float(edge["weight"]),
                    "sign": sign,
                    "source_type": source,
                    "target_type": target,
                    "edge_family": f"{source}->{target}",
                    "sign_provenance": "male-cns:v1.0 type-level predicted neurotransmitter",
                }
            )

    edges = pd.DataFrame(edge_rows)
    roles = {key: [int(x) for x in values] for key, values in config["candidate_roles"].items()}
    authority_path = Path(sign_authority_path) if sign_authority_path is not None else None
    manifest = {
        "dataset": audit["dataset"],
        "protocol": config["protocol"],
        "qualification_status": "candidate",
        "graph_role": "restricted-excitatory-steering-scaffold",
        "source_route_audit_sha256": source_audit_sha256,
        "evidence_authority": config.get("evidence_authority"),
        "sign_authority": {
            "path": str(authority_path) if authority_path is not None else None,
            "sha256": sign_authority_sha256,
        },
        "included_edge_families": [f"{a}->{b}" for a, b in EDGE_FAMILIES],
        "excluded": config.get("excluded_from_this_scaffold", {}),
        "claim_boundary": (
            "Body-ID-resolved structural steering scaffold with explicit type-level sign assumptions. "
            "Not a qualified odor-navigation circuit, not measured activity, and not the complete see-saw circuit."
        ),
    }
    bundle = GraphBundle(nodes=nodes, edges=edges, roles=roles, manifest=manifest)
    bundle.validate(require_sign=True, require_qualified=False)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the restricted MaleCNS steering scaffold from literature-route-audit-v1"
    )
    parser.add_argument("audit")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sign-authority", default=str(DEFAULT_SIGN_AUTHORITY))
    parser.add_argument("--output", default="data/cache/steering-scaffold-v1")
    args = parser.parse_args()

    audit_path = Path(args.audit)
    sign_authority_path = Path(args.sign_authority)
    audit = json.loads(audit_path.read_text())
    config = json.loads(Path(args.config).read_text())
    signs = _load_type_signs(sign_authority_path)
    audit_sha = sha256_file(audit_path)
    sign_authority_sha = sha256_file(sign_authority_path)
    bundle = build_steering_scaffold(
        audit,
        config,
        type_signs=signs,
        source_audit_sha256=audit_sha,
        sign_authority_path=sign_authority_path,
        sign_authority_sha256=sign_authority_sha,
    )
    save_bundle(bundle, args.output)
    summary = {
        "output": str(args.output),
        "nodes": int(len(bundle.nodes)),
        "edges": int(len(bundle.edges)),
        "signed_edge_fraction": float(bundle.edges.sign.ne(0).mean()) if len(bundle.edges) else 0.0,
        "qualification_status": bundle.manifest["qualification_status"],
        "source_route_audit_sha256": audit_sha,
        "sign_authority_sha256": sign_authority_sha,
        "roles": bundle.roles,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
