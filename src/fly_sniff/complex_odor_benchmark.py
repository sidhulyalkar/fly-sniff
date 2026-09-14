from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .odor_authority import validate_authority

FAST_PROTOCOL = "E003b-fast-preflight-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class Source:
    odorant: str
    x: float
    y: float
    emission: float
    is_target: bool


def _antennae(x: float, y: float, heading: float, separation: float) -> tuple[tuple[float, float], tuple[float, float]]:
    half = 0.5 * separation
    left = (x - np.sin(heading) * half, y + np.cos(heading) * half)
    right = (x + np.sin(heading) * half, y - np.cos(heading) * half)
    return left, right


def _concentration(sensor: tuple[float, float], source: Source, wind_angle: float, cfg: dict[str, Any]) -> float:
    dx = float(sensor[0] - source.x)
    dy = float(sensor[1] - source.y)
    c = float(np.cos(wind_angle))
    s = float(np.sin(wind_angle))
    downwind = c * dx + s * dy
    crosswind = -s * dx + c * dy
    sigma_cross = float(cfg["sigma_cross"])
    sigma_upwind = float(cfg["sigma_upwind"])
    decay = float(cfg["downwind_decay"])
    # A deterministic, analytical engineering plume used only for fast sensory
    # preflight. It is intentionally not presented as CFD or real plume physics.
    axial = np.exp(-max(downwind, 0.0) / decay)
    if downwind < 0.0:
        axial *= np.exp(-0.5 * (downwind / sigma_upwind) ** 2)
    lateral = np.exp(-0.5 * (crosswind / sigma_cross) ** 2)
    return float(source.emission * axial * lateral)


def _mixture_response(
    concentrations: dict[str, float],
    odor_vectors: dict[str, np.ndarray],
    model: str,
) -> np.ndarray:
    if not concentrations:
        n = len(next(iter(odor_vectors.values())))
        return np.zeros(n, dtype=float)
    weighted = np.zeros_like(next(iter(odor_vectors.values())), dtype=float)
    total_concentration = 0.0
    for odorant, concentration in concentrations.items():
        c = max(0.0, float(concentration))
        weighted += c * odor_vectors[odorant]
        total_concentration += c
    if model == "independent-saturating-receptors":
        return 1.0 - np.exp(-weighted)
    if model == "competitive-binding-sensitivity":
        return weighted / (1.0 + total_concentration + 1e-12)
    raise ValueError(f"unsupported mixture model: {model}")


def _cosine_score(response: np.ndarray, template: np.ndarray) -> float:
    denom = float(np.linalg.norm(response) * np.linalg.norm(template))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(response, template) / denom)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float(np.sum(precision[y == 1]) / total_pos)


def _trial_sources(rng: np.random.Generator, config: dict[str, Any], target_present: bool) -> list[Source]:
    arena = config["arena"]
    odorants = config["odorants"]
    sources: list[Source] = []
    if target_present:
        sources.append(
            Source(
                odorant=str(odorants["target"]),
                x=float(rng.uniform(0.5, float(arena["width"]) - 0.5)),
                y=float(rng.uniform(0.5, float(arena["height"]) - 0.5)),
                emission=float(rng.uniform(*map(float, config["emission_range"]))),
                is_target=True,
            )
        )
    distractors = list(map(str, odorants["distractors"]))
    count = int(rng.integers(int(config["distractor_count_range"][0]), int(config["distractor_count_range"][1]) + 1))
    for _ in range(count):
        sources.append(
            Source(
                odorant=str(rng.choice(distractors)),
                x=float(rng.uniform(0.5, float(arena["width"]) - 0.5)),
                y=float(rng.uniform(0.5, float(arena["height"]) - 0.5)),
                emission=float(rng.uniform(*map(float, config["emission_range"]))),
                is_target=False,
            )
        )
    return sources


def run_fast_preflight(config: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != FAST_PROTOCOL:
        raise ValueError("unexpected fast-preflight protocol")
    validate_authority(authority)
    target = str(config["odorants"]["target"])
    distractors = list(map(str, config["odorants"]["distractors"]))
    required = {target, *distractors}
    missing = sorted(required - set(authority["odorants"]))
    if missing:
        raise ValueError("odor authority missing configured odorants: " + ", ".join(missing))

    odor_vectors = {name: np.asarray(authority["odorants"][name], dtype=float) for name in required}
    target_template = odor_vectors[target]
    n_trials = int(config["trials"])
    models = list(map(str, config["mixture_models"]))
    seed = int(config["seed"])
    arena = config["arena"]
    antenna_sep = float(config["antenna_separation"])

    model_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    for model in models:
        rng = np.random.default_rng(seed)
        for i in range(n_trials):
            target_present = bool(i % 2 == 0)
            x = float(rng.uniform(0.8, float(arena["width"]) - 0.8))
            y = float(rng.uniform(0.8, float(arena["height"]) - 0.8))
            heading = float(rng.uniform(-np.pi, np.pi))
            wind_angle = float(rng.uniform(-np.pi, np.pi))
            sources = _trial_sources(rng, config, target_present)
            left_pos, right_pos = _antennae(x, y, heading, antenna_sep)

            sensor_scores: list[float] = []
            for sensor in (left_pos, right_pos):
                concentrations: dict[str, float] = {}
                for source in sources:
                    concentrations[source.odorant] = concentrations.get(source.odorant, 0.0) + _concentration(
                        sensor, source, wind_angle, config["plume"]
                    )
                response = _mixture_response(concentrations, odor_vectors, model)
                sensor_scores.append(_cosine_score(response, target_template))

            mean_score = float(np.mean(sensor_scores))
            direction_correct: bool | None = None
            true_lateral_sign: float | None = None
            target_sources = [source for source in sources if source.is_target]
            if target_sources:
                source = target_sources[0]
                bearing_body = _wrap(np.arctan2(source.y - y, source.x - x) - heading)
                true_lateral_sign = float(np.sign(np.sin(bearing_body)))
                predicted_lateral_sign = float(np.sign(sensor_scores[0] - sensor_scores[1]))
                if true_lateral_sign != 0.0 and predicted_lateral_sign != 0.0:
                    direction_correct = bool(true_lateral_sign == predicted_lateral_sign)

            model_rows[model].append(
                {
                    "trial": i,
                    "target_present": target_present,
                    "target_score": mean_score,
                    "left_target_score": float(sensor_scores[0]),
                    "right_target_score": float(sensor_scores[1]),
                    "direction_correct": direction_correct,
                    "true_lateral_sign": true_lateral_sign,
                    "wind_angle_rad": wind_angle,
                }
            )

    reports: dict[str, Any] = {}
    for model, rows in model_rows.items():
        labels = np.asarray([int(row["target_present"]) for row in rows], dtype=int)
        scores = np.asarray([float(row["target_score"]) for row in rows], dtype=float)
        direction = [row["direction_correct"] for row in rows if row["direction_correct"] is not None]
        reports[model] = {
            "trials": len(rows),
            "AUROC_external_template_readout": _auroc(labels, scores),
            "AUPRC_external_template_readout": _average_precision(labels, scores),
            "bilateral_direction_sign_accuracy": float(np.mean(direction)) if direction else None,
            "direction_evaluable_trials": len(direction),
            "rows": rows,
        }

    return {
        "protocol": FAST_PROTOCOL,
        "status": "engineering-preflight-not-E003b-qualification",
        "dataset": config["dataset"],
        "authority_source": authority["source"],
        "target": target,
        "distractors": distractors,
        "seed": seed,
        "model_reports": reports,
        "claim_boundary": (
            "Fast analytical-plume sensory preflight only. The cosine target score is an external fixed decoder, "
            "not a MaleCNS neuron or learned navigation policy. Passing does not qualify E003b, closed-loop behavior, "
            "or intact-vs-rewire performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E003b vector-odor fast sensory preflight")
    parser.add_argument("--config", default="configs/e003b_fast_preflight_v1.json")
    parser.add_argument("--authority", default="authority/door-response-authority-v1.json")
    parser.add_argument("--output", default="results/e003/complex-odor-fast-preflight-v1.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    authority_path = Path(args.authority)
    config = json.loads(config_path.read_text())
    authority = json.loads(authority_path.read_text())
    report = run_fast_preflight(config, authority)
    report["input_sha256"] = {
        "config": _sha256(config_path),
        "authority": _sha256(authority_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    for name, row in report["model_reports"].items():
        print(
            f"{name}: AUROC={row['AUROC_external_template_readout']:.3f} "
            f"AUPRC={row['AUPRC_external_template_readout']:.3f} "
            f"dir_acc={row['bilateral_direction_sign_accuracy']}"
        )


if __name__ == "__main__":
    main()
