from __future__ import annotations

import pandas as pd

# These are modeling assumptions, not universal receptor-level truth. In
# particular, glutamate is intentionally left unresolved rather than silently
# treated as inhibitory. Unknown/ambiguous transmitters receive sign 0 and are
# reported as unresolved in qualification statistics.
DEFAULT_SIGN_POLICY = {
    "acetylcholine": 1,
    "ach": 1,
    "gaba": -1,
    "glutamate": 0,
    "glu": 0,
    "dopamine": 0,
    "serotonin": 0,
    "octopamine": 0,
}


def infer_nt_column(nodes: pd.DataFrame) -> str:
    candidates = [
        "predictedNt",
        "predicted_nt",
        "cell_type_nt",
        "nt_type",
        "neurotransmitter",
    ]
    for col in candidates:
        if col in nodes.columns:
            return col
    raise ValueError(
        "no recognized neurotransmitter annotation column found; pass an explicit column "
        f"after inspecting the MaleCNS annotation schema: {list(nodes.columns)}"
    )


def attach_presynaptic_signs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    nt_column: str | None = None,
    policy: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach a conservative presynaptic transmitter-derived sign to each edge.

    A sign of 0 means unresolved and contributes no signed drive in the v0 rate
    model. This deliberately sacrifices coverage rather than inventing physiology.
    """
    policy = {k.lower(): int(v) for k, v in (policy or DEFAULT_SIGN_POLICY).items()}
    nt_column = nt_column or infer_nt_column(nodes)
    if "bodyId" not in nodes.columns:
        raise ValueError("nodes require bodyId")
    if "source" not in edges.columns:
        raise ValueError("edges require source")

    lookup = nodes[["bodyId", nt_column]].drop_duplicates("bodyId").copy()
    lookup["_nt_norm"] = lookup[nt_column].fillna("").astype(str).str.strip().str.lower()
    lookup["sign"] = lookup["_nt_norm"].map(policy).fillna(0).astype(int)
    signed = edges.merge(lookup[["bodyId", nt_column, "sign"]], left_on="source", right_on="bodyId", how="left")
    signed = signed.drop(columns=["bodyId"])
    signed["sign"] = signed["sign"].fillna(0).astype(int)
    unresolved = signed.sign.eq(0)
    stats = {
        "edges": float(len(signed)),
        "signed_edges": float((~unresolved).sum()),
        "unresolved_edges": float(unresolved.sum()),
        "signed_fraction": float((~unresolved).mean()) if len(signed) else 0.0,
    }
    return signed, stats
