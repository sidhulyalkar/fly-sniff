from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .freeze import canonical_sha256


@dataclass(frozen=True)
class GraphEvaluation:
    """One graph-level statistic evaluated on an exact paired condition set."""

    graph_id: str
    graph_sha256: str
    statistic_name: str
    statistic: float
    condition_ids: tuple[str, ...]
    null_family: str | None = None

    def validate(self) -> None:
        if not self.graph_id.strip():
            raise ValueError("graph_id must be non-empty")
        if len(self.graph_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.graph_sha256
        ):
            raise ValueError("graph_sha256 must be a lowercase SHA-256 digest")
        if not self.statistic_name.strip():
            raise ValueError("statistic_name must be non-empty")
        if not math.isfinite(self.statistic):
            raise ValueError("graph statistic must be finite")
        if not self.condition_ids:
            raise ValueError("graph evaluation requires at least one paired condition")
        if len(self.condition_ids) != len(set(self.condition_ids)):
            raise ValueError("condition_ids must be unique")
        if any(not condition.strip() for condition in self.condition_ids):
            raise ValueError("condition_ids must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "graph_id": self.graph_id,
            "graph_sha256": self.graph_sha256,
            "statistic_name": self.statistic_name,
            "statistic": self.statistic,
            "condition_ids": list(self.condition_ids),
        }
        if self.null_family is not None:
            payload["null_family"] = self.null_family
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphEvaluation:
        evaluation = cls(
            graph_id=str(payload["graph_id"]),
            graph_sha256=str(payload["graph_sha256"]),
            statistic_name=str(payload["statistic_name"]),
            statistic=float(payload["statistic"]),
            condition_ids=tuple(str(value) for value in payload["condition_ids"]),
            null_family=(str(payload["null_family"]) if payload.get("null_family") else None),
        )
        evaluation.validate()
        return evaluation


@dataclass(frozen=True)
class TopologyInferenceReceipt:
    """Randomization inference where graph realizations are the independent null units."""

    intact: GraphEvaluation
    nulls: tuple[GraphEvaluation, ...]
    higher_is_better: bool
    minimum_null_count: int
    null_family: str
    empirical_p: float
    extreme_or_tied_null_count: int
    intact_rank_best_is_one: int
    intact_percentile: float
    signed_effect_vs_null_mean: float
    signed_effect_vs_null_median: float
    schema: str = "fly-sniff-topology-inference-v1"

    def validate(self) -> None:
        if self.schema != "fly-sniff-topology-inference-v1":
            raise ValueError(f"unsupported topology inference schema: {self.schema}")
        self.intact.validate()
        if self.minimum_null_count < 1:
            raise ValueError("minimum_null_count must be positive")
        if len(self.nulls) < self.minimum_null_count:
            raise ValueError(
                f"topology inference requires at least {self.minimum_null_count} null graphs; "
                f"received {len(self.nulls)}"
            )
        if not self.null_family.strip():
            raise ValueError("null_family must be non-empty")
        graph_ids: set[str] = set()
        graph_hashes: set[str] = set()
        for null in self.nulls:
            null.validate()
            if null.statistic_name != self.intact.statistic_name:
                raise ValueError("all graph evaluations must use the same statistic_name")
            if null.condition_ids != self.intact.condition_ids:
                raise ValueError("all topology nulls must use the exact paired condition_ids")
            if null.null_family != self.null_family:
                raise ValueError("one inference receipt may contain only one null family")
            if null.graph_id in graph_ids or null.graph_sha256 in graph_hashes:
                raise ValueError("null graph realizations must be unique")
            graph_ids.add(null.graph_id)
            graph_hashes.add(null.graph_sha256)
        if self.intact.graph_sha256 in graph_hashes:
            raise ValueError("intact graph cannot also appear in the null ensemble")
        if not 0.0 < self.empirical_p <= 1.0:
            raise ValueError("empirical_p must lie in (0, 1]")
        if not 0.0 <= self.intact_percentile <= 1.0:
            raise ValueError("intact_percentile must lie in [0, 1]")
        if not 1 <= self.intact_rank_best_is_one <= len(self.nulls) + 1:
            raise ValueError("intact rank is outside the graph ensemble")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "schema": self.schema,
            "experimental_unit": "graph_realization",
            "intact": self.intact.to_dict(),
            "nulls": [null.to_dict() for null in self.nulls],
            "higher_is_better": self.higher_is_better,
            "minimum_null_count": self.minimum_null_count,
            "null_count": len(self.nulls),
            "null_family": self.null_family,
            "empirical_p": self.empirical_p,
            "extreme_or_tied_null_count": self.extreme_or_tied_null_count,
            "intact_rank_best_is_one": self.intact_rank_best_is_one,
            "intact_percentile": self.intact_percentile,
            "signed_effect_vs_null_mean": self.signed_effect_vs_null_mean,
            "signed_effect_vs_null_median": self.signed_effect_vs_null_median,
            "claim_boundary": (
                "episode-level replication does not increase the independent topology sample size; "
                "report each prespecified null family separately"
            ),
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        return payload


def topology_randomization_inference(
    intact: GraphEvaluation,
    nulls: tuple[GraphEvaluation, ...],
    *,
    higher_is_better: bool = True,
    minimum_null_count: int = 31,
) -> TopologyInferenceReceipt:
    """Compute one-family empirical randomization inference over graph realizations.

    Ties count against the intact graph in the empirical tail probability. For a
    higher-is-better statistic, ``p = (1 + count(null >= intact)) / (N + 1)``.
    The lower-is-better case is symmetric.
    """

    intact.validate()
    if intact.null_family is not None:
        raise ValueError("intact evaluation must not be labeled as a null family")
    if minimum_null_count < 1:
        raise ValueError("minimum_null_count must be positive")
    if len(nulls) < minimum_null_count:
        raise ValueError(
            f"topology inference requires at least {minimum_null_count} null graphs; "
            f"received {len(nulls)}"
        )
    families = {null.null_family for null in nulls}
    if None in families or len(families) != 1:
        raise ValueError("null ensemble must declare exactly one common null_family")
    null_family = next(iter(families))
    assert null_family is not None

    null_values: list[float] = []
    graph_ids: set[str] = set()
    graph_hashes: set[str] = set()
    for null in nulls:
        null.validate()
        if null.statistic_name != intact.statistic_name:
            raise ValueError("all graph evaluations must use the same statistic_name")
        if null.condition_ids != intact.condition_ids:
            raise ValueError("all topology nulls must use the exact paired condition_ids")
        if null.graph_id in graph_ids or null.graph_sha256 in graph_hashes:
            raise ValueError("null graph realizations must be unique")
        if null.graph_sha256 == intact.graph_sha256:
            raise ValueError("intact graph cannot also appear in the null ensemble")
        graph_ids.add(null.graph_id)
        graph_hashes.add(null.graph_sha256)
        null_values.append(null.statistic)

    if higher_is_better:
        extreme = sum(value >= intact.statistic for value in null_values)
        better = sum(value > intact.statistic for value in null_values)
        percentile = sum(value <= intact.statistic for value in null_values) / len(null_values)
        mean_effect = intact.statistic - statistics.fmean(null_values)
        median_effect = intact.statistic - statistics.median(null_values)
    else:
        extreme = sum(value <= intact.statistic for value in null_values)
        better = sum(value < intact.statistic for value in null_values)
        percentile = sum(value >= intact.statistic for value in null_values) / len(null_values)
        mean_effect = statistics.fmean(null_values) - intact.statistic
        median_effect = statistics.median(null_values) - intact.statistic

    receipt = TopologyInferenceReceipt(
        intact=intact,
        nulls=nulls,
        higher_is_better=higher_is_better,
        minimum_null_count=minimum_null_count,
        null_family=null_family,
        empirical_p=(1 + extreme) / (len(null_values) + 1),
        extreme_or_tied_null_count=extreme,
        intact_rank_best_is_one=1 + better,
        intact_percentile=float(percentile),
        signed_effect_vs_null_mean=float(mean_effect),
        signed_effect_vs_null_median=float(median_effect),
    )
    receipt.validate()
    return receipt


def _load_evaluation(path: str | Path) -> GraphEvaluation:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return GraphEvaluation.from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute graph-level empirical randomization inference for one null family."
    )
    parser.add_argument("--intact", required=True, help="Intact GraphEvaluation JSON")
    parser.add_argument("--null", action="append", required=True, help="Null GraphEvaluation JSON")
    parser.add_argument("--minimum-nulls", type=int, default=31)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    receipt = topology_randomization_inference(
        _load_evaluation(args.intact),
        tuple(_load_evaluation(path) for path in args.null),
        higher_is_better=not args.lower_is_better,
        minimum_null_count=args.minimum_nulls,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing inference receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())