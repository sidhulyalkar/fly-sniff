from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from .schema import SampleWindow

SPLITS = ("train", "validation", "test")


def make_animal_disjoint_splits(
    windows: Iterable[SampleWindow],
    *,
    seed: int = 1701,
    train_fraction: float = 0.625,
    validation_fraction: float = 0.125,
) -> dict[str, str]:
    rows = list(windows)
    animals = sorted({row.animal_id for row in rows})
    if len(animals) < 3:
        raise ValueError("at least three animals are required for train/validation/test splitting")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave a non-empty test fraction")
    rng = random.Random(seed)
    rng.shuffle(animals)
    n = len(animals)
    n_train = max(1, round(n * train_fraction))
    n_val = max(1, round(n * validation_fraction))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    assignment: dict[str, str] = {}
    for index, animal in enumerate(animals):
        if index < n_train:
            assignment[animal] = "train"
        elif index < n_train + n_val:
            assignment[animal] = "validation"
        else:
            assignment[animal] = "test"
    validate_animal_disjoint_splits(rows, assignment)
    return assignment


def validate_animal_disjoint_splits(
    windows: Iterable[SampleWindow], animal_to_split: dict[str, str]
) -> None:
    rows = list(windows)
    observed = {row.animal_id for row in rows}
    missing = sorted(observed - set(animal_to_split))
    if missing:
        raise ValueError(f"animals missing split assignments: {missing}")
    unknown_labels = sorted(set(animal_to_split.values()) - set(SPLITS))
    if unknown_labels:
        raise ValueError(f"unsupported split labels: {unknown_labels}")
    present = {animal_to_split[animal] for animal in observed}
    if present != set(SPLITS):
        raise ValueError(f"observed samples must populate all splits; got {sorted(present)}")
    sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        sessions[(row.dataset_id, row.session_id)].add(animal_to_split[row.animal_id])
    leaked = sorted(key for key, labels in sessions.items() if len(labels) > 1)
    if leaked:
        raise ValueError(f"session leakage across splits: {leaked[:10]}")


def sample_split_map(
    windows: Iterable[SampleWindow], animal_to_split: dict[str, str]
) -> dict[str, str]:
    rows = list(windows)
    validate_animal_disjoint_splits(rows, animal_to_split)
    return {row.sample_id: animal_to_split[row.animal_id] for row in rows}
