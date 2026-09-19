from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .graph import GraphBundle

VARIANTS = ("intact", "rewire", "random", "stateless")
SELECTION_POLICY = "top-sqrt-in-out-log1p-synapse-strength-v1"
ENCODER_ID = "categorical-spatial-8x8-plus-change-v1"
REWARD_ID = "arc-progress-novelty-v1"
MODEL_ID = "flyarc-rate-reservoir-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _state_name(observation: Any) -> str:
    value = getattr(observation, "state", "")
    return getattr(value, "name", str(value))


def _action_name(action: Any) -> str:
    return getattr(action, "name", str(action))


def _extract_frame(observation: Any) -> np.ndarray:
    raw = getattr(observation, "frame", None)
    if raw is None:
        raise ValueError("ARC observation does not contain frame data")
    frame = np.asarray(raw)
    if frame.ndim == 3:
        if frame.shape[0] < 1:
            raise ValueError("ARC observation contains no frame planes")
        frame = frame[-1]
    if frame.ndim != 2:
        raise ValueError(f"expected a 2-D ARC frame, observed shape={frame.shape}")
    return frame


def _frame_hash(frame: np.ndarray) -> str:
    array = np.ascontiguousarray(frame.astype(np.uint8, copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


class ARCFrameEncoder:
    """Deterministic categorical encoder for ARC-AGI-3 frame grids.

    Color IDs remain categorical. Each spatial bin stores a normalized histogram
    over the 16 ARC cell values plus one fraction-changed feature.
    """

    def __init__(self, *, pool: int = 8, colors: int = 16):
        if pool <= 0 or colors <= 1:
            raise ValueError("pool must be positive and colors must be greater than one")
        self.pool = int(pool)
        self.colors = int(colors)
        self.previous: np.ndarray | None = None

    @property
    def feature_dim(self) -> int:
        return self.pool * self.pool * (self.colors + 1)

    def reset(self) -> None:
        self.previous = None

    def _bounds(self, length: int) -> np.ndarray:
        return np.linspace(0, length, self.pool + 1, dtype=int)

    def encode(self, frame: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("ARC frame must be a non-empty 2-D array")
        if not np.issubdtype(array.dtype, np.integer):
            if not np.all(np.equal(array, np.floor(array))):
                raise ValueError("ARC frame values must be integer color IDs")
            array = array.astype(int)
        else:
            array = array.astype(int, copy=False)
        if array.min() < 0 or array.max() >= self.colors:
            raise ValueError(f"ARC frame values must be in [0, {self.colors - 1}]")

        previous = (
            self.previous
            if self.previous is not None and self.previous.shape == array.shape
            else None
        )
        changed = np.zeros_like(array, dtype=float)
        if previous is not None:
            changed = (array != previous).astype(float)

        y_bounds = self._bounds(array.shape[0])
        x_bounds = self._bounds(array.shape[1])
        hist = np.zeros((self.pool, self.pool, self.colors), dtype=np.float32)
        delta = np.zeros((self.pool, self.pool), dtype=np.float32)

        for yi in range(self.pool):
            y0, y1 = int(y_bounds[yi]), int(y_bounds[yi + 1])
            for xi in range(self.pool):
                x0, x1 = int(x_bounds[xi]), int(x_bounds[xi + 1])
                block = array[y0:y1, x0:x1]
                if block.size == 0:
                    continue
                hist[yi, xi] = (
                    np.bincount(block.ravel(), minlength=self.colors).astype(np.float32)
                    / float(block.size)
                )
                delta[yi, xi] = float(changed[y0:y1, x0:x1].mean())

        self.previous = array.copy()
        return np.concatenate((hist.ravel(), delta.ravel())).astype(
            np.float32, copy=False
        )


@dataclass(frozen=True)
class CoreSelection:
    body_ids: tuple[int, ...]
    nodes: pd.DataFrame
    edges: pd.DataFrame
    policy: str


def select_structural_core(bundle: GraphBundle, *, max_nodes: int) -> CoreSelection:
    """Select a recurrently connected core without inspecting any ARC task data."""

    if max_nodes <= 1:
        raise ValueError("max_nodes must be greater than one")
    bundle.validate(require_sign=True, require_qualified=False)

    edges = bundle.edges.loc[:, ["source", "target", "weight", "sign"]].copy()
    edges["source"] = edges["source"].astype(int)
    edges["target"] = edges["target"].astype(int)
    if edges.duplicated(["source", "target"]).any():
        raise ValueError("FlyARC v1 requires one aggregated edge per directed neuron pair")
    if (edges["weight"].astype(float) <= 0).any():
        raise ValueError("edge weights must be positive synapse counts")

    transformed = np.log1p(edges["weight"].astype(float))
    incoming = transformed.groupby(edges["target"]).sum()
    outgoing = transformed.groupby(edges["source"]).sum()

    scored = bundle.nodes.loc[:, ["bodyId"]].copy()
    scored["bodyId"] = scored["bodyId"].astype(int)
    scored["in_strength"] = scored["bodyId"].map(incoming).fillna(0.0)
    scored["out_strength"] = scored["bodyId"].map(outgoing).fillna(0.0)
    scored["score"] = np.sqrt(scored["in_strength"] * scored["out_strength"])
    scored = scored.sort_values(
        ["score", "bodyId"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected_ids = tuple(scored.head(min(max_nodes, len(scored)))["bodyId"].tolist())
    selected_set = set(selected_ids)
    induced = edges[
        edges["source"].isin(selected_set) & edges["target"].isin(selected_set)
    ].reset_index(drop=True)
    if induced.empty:
        raise ValueError("structural core selection produced no recurrent edges")

    nodes = bundle.nodes[bundle.nodes["bodyId"].astype(int).isin(selected_set)].copy()
    order = {body_id: index for index, body_id in enumerate(selected_ids)}
    nodes["_flyarc_order"] = nodes["bodyId"].astype(int).map(order)
    nodes = (
        nodes.sort_values("_flyarc_order")
        .drop(columns="_flyarc_order")
        .reset_index(drop=True)
    )
    return CoreSelection(
        body_ids=selected_ids,
        nodes=nodes,
        edges=induced,
        policy=SELECTION_POLICY,
    )


def degree_preserving_rewire(
    edges: pd.DataFrame,
    *,
    seed: int,
    swaps_per_edge: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Directed target-swap null preserving exact in/out degree counts."""

    if swaps_per_edge < 1:
        raise ValueError("swaps_per_edge must be at least one")
    rewired = edges.copy().reset_index(drop=True)
    sources = rewired["source"].astype(int).to_numpy(copy=True)
    targets = rewired["target"].astype(int).to_numpy(copy=True)
    if len(sources) < 2:
        raise ValueError("at least two edges are required for rewiring")

    pairs = set(zip(sources.tolist(), targets.tolist(), strict=True))
    if len(pairs) != len(sources):
        raise ValueError("rewiring requires unique directed edges")

    rng = np.random.default_rng(seed)
    target_swaps = swaps_per_edge * len(sources)
    max_attempts = max(1000, target_swaps * 30)
    completed = 0

    for _ in range(max_attempts):
        if completed >= target_swaps:
            break
        i, j = rng.integers(0, len(sources), size=2)
        i, j = int(i), int(j)
        if i == j:
            continue
        a, b = int(sources[i]), int(targets[i])
        c, d = int(sources[j]), int(targets[j])
        if a == c or b == d:
            continue
        p1 = (a, d)
        p2 = (c, b)
        if a == d or c == b or p1 == p2:
            continue

        pairs.remove((a, b))
        pairs.remove((c, d))
        if p1 in pairs or p2 in pairs:
            pairs.add((a, b))
            pairs.add((c, d))
            continue

        targets[i], targets[j] = d, b
        pairs.add(p1)
        pairs.add(p2)
        completed += 1

    minimum = min(target_swaps, max(1, len(sources) // 2))
    if completed < minimum:
        raise RuntimeError(
            f"rewire mixing failed: completed {completed} target swaps, required {minimum}"
        )

    rewired["target"] = targets
    return rewired, completed


def random_edges_like(
    edges: pd.DataFrame,
    *,
    body_ids: Sequence[int],
    seed: int,
) -> pd.DataFrame:
    """Random directed null with matched node/edge counts and edge-attribute multiset."""

    ids = np.asarray(body_ids, dtype=np.int64)
    n = len(ids)
    m = len(edges)
    if n < 2 or m < 1:
        raise ValueError("random reservoir requires at least two nodes and one edge")
    total = n * (n - 1)
    if m > total:
        raise ValueError("edge count exceeds number of unique non-self directed pairs")

    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    while len(chosen) < m:
        need = m - len(chosen)
        batch = max(1024, min(need * 3, 100_000))
        for value in rng.integers(0, total, size=batch):
            chosen.add(int(value))
            if len(chosen) == m:
                break

    keys = np.fromiter(chosen, dtype=np.int64, count=m)
    rng.shuffle(keys)
    source_idx = keys // (n - 1)
    target_reduced = keys % (n - 1)
    target_idx = target_reduced + (target_reduced >= source_idx)

    perm = rng.permutation(m)
    return pd.DataFrame(
        {
            "source": ids[source_idx],
            "target": ids[target_idx],
            "weight": edges["weight"].to_numpy()[perm],
            "sign": edges["sign"].to_numpy()[perm],
        }
    )


def _normalized_weight_matrix(
    body_ids: Sequence[int],
    edges: pd.DataFrame,
) -> sparse.csr_matrix:
    index = {int(body_id): i for i, body_id in enumerate(body_ids)}
    src = edges["source"].astype(int).map(index).to_numpy()
    dst = edges["target"].astype(int).map(index).to_numpy()
    if pd.isna(src).any() or pd.isna(dst).any():
        raise ValueError("reservoir edges reference neurons outside the selected core")
    values = (
        np.log1p(edges["weight"].astype(float).to_numpy())
        * edges["sign"].astype(float).to_numpy()
    )
    matrix = sparse.coo_matrix(
        (values, (dst.astype(int), src.astype(int))),
        shape=(len(body_ids), len(body_ids)),
    ).tocsr()
    row_norm = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    inv = np.divide(1.0, row_norm, out=np.ones_like(row_norm), where=row_norm > 0)
    return (sparse.diags(inv) @ matrix).tocsr()


def _fixed_input_projection(
    *,
    state_dim: int,
    feature_dim: int,
    fanout: int,
    seed: int,
) -> sparse.csr_matrix:
    if fanout < 1 or fanout > state_dim:
        raise ValueError("input fanout must be within [1, state_dim]")
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    scale = 1.0 / np.sqrt(float(fanout))
    for feature in range(feature_dim):
        targets = rng.choice(state_dim, size=fanout, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=fanout)
        rows.extend(int(x) for x in targets)
        cols.extend([feature] * fanout)
        values.extend(float(x * scale) for x in signs)
    return sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(state_dim, feature_dim),
    ).tocsr()


class FlyARCReservoir:
    """Fixed recurrent state system whose topology is the experimental variable."""

    def __init__(
        self,
        core: CoreSelection,
        *,
        variant: str,
        feature_dim: int,
        projection_seed: int,
        topology_seed: int,
        input_fanout: int = 4,
        leak: float = 0.82,
        gain: float = 1.6,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown FlyARC variant {variant!r}")
        self.variant = variant
        self.body_ids = core.body_ids
        self.leak = float(leak)
        self.gain = float(gain)
        self.activity = np.zeros(len(self.body_ids), dtype=np.float32)
        self.projection = _fixed_input_projection(
            state_dim=len(self.body_ids),
            feature_dim=feature_dim,
            fanout=input_fanout,
            seed=projection_seed,
        )

        edges = core.edges
        rewire_swaps = 0
        if variant == "rewire":
            edges, rewire_swaps = degree_preserving_rewire(
                edges,
                seed=topology_seed,
            )
        elif variant == "random":
            edges = random_edges_like(
                edges,
                body_ids=self.body_ids,
                seed=topology_seed,
            )

        self.edge_count = len(edges)
        self.rewire_swaps = int(rewire_swaps)
        self.w = (
            None
            if variant == "stateless"
            else _normalized_weight_matrix(self.body_ids, edges)
        )

    def reset(self) -> None:
        self.activity.fill(0.0)

    def step(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 1 or values.shape[0] != self.projection.shape[1]:
            raise ValueError(
                f"feature vector must have shape ({self.projection.shape[1]},)"
            )
        if not np.isfinite(values).all():
            raise ValueError("FlyARC features must be finite")
        drive = np.asarray(self.projection @ values).ravel()
        if self.w is None:
            proposal = np.tanh(self.gain * drive)
            self.activity = proposal.astype(np.float32, copy=False)
        else:
            recurrent = self.w @ self.activity
            proposal = np.tanh(self.gain * (recurrent + drive))
            updated = self.leak * self.activity + (1.0 - self.leak) * proposal
            self.activity = updated.astype(np.float32, copy=False)
        return self.activity.copy()

    def top_activity(self, *, k: int = 12) -> list[dict[str, float | int]]:
        if k <= 0:
            return []
        count = min(k, len(self.activity))
        indices = np.argpartition(np.abs(self.activity), -count)[-count:]
        indices = indices[np.argsort(-np.abs(self.activity[indices]))]
        return [
            {
                "body_id": int(self.body_ids[int(index)]),
                "activity": float(self.activity[int(index)]),
            }
            for index in indices
        ]


class LinearQPolicy:
    """Tiny online TD head. This is engineered learning, not a biological readout."""

    def __init__(
        self,
        action_count: int,
        state_dim: int,
        *,
        seed: int,
        learning_rate: float = 0.08,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        td_clip: float = 3.0,
        l2: float = 1e-5,
    ):
        if action_count < 1 or state_dim < 1:
            raise ValueError("action_count and state_dim must be positive")
        self.weights = np.zeros((action_count, state_dim), dtype=np.float32)
        self.bias = np.zeros(action_count, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.td_clip = float(td_clip)
        self.l2 = float(l2)

    def _phi(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32) / np.sqrt(float(len(state)))

    def q_values(self, state: np.ndarray) -> np.ndarray:
        phi = self._phi(state)
        return self.weights @ phi + self.bias

    def choose(self, state: np.ndarray, valid: Sequence[int]) -> tuple[int, np.ndarray]:
        if not valid:
            raise ValueError("no valid ARC actions")
        q = self.q_values(state)
        valid_array = np.asarray(valid, dtype=int)
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_array)), q
        values = q[valid_array]
        best = valid_array[np.flatnonzero(values == values.max())]
        return int(self.rng.choice(best)), q

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        *,
        done: bool,
        valid_next: Sequence[int],
    ) -> float:
        phi = self._phi(state)
        q = self.q_values(state)
        next_q = self.q_values(next_state)
        bootstrap = (
            0.0
            if done
            else float(np.max(next_q[np.asarray(valid_next, dtype=int)]))
        )
        td = float(reward) + self.gamma * bootstrap - float(q[action])
        td = float(np.clip(td, -self.td_clip, self.td_clip))
        self.weights[action] *= 1.0 - self.learning_rate * self.l2
        self.weights[action] += self.learning_rate * td * phi
        self.bias[action] += self.learning_rate * td
        return td


@dataclass(frozen=True)
class RunConfig:
    game: str = "ls20"
    max_steps: int = 500
    max_resets: int = 8
    max_nodes: int = 4096
    pool: int = 8
    colors: int = 16
    input_fanout: int = 4
    leak: float = 0.82
    gain: float = 1.6
    learning_rate: float = 0.08
    gamma: float = 0.95
    epsilon: float = 0.15
    step_cost: float = 0.001
    novelty_reward: float = 0.01
    level_reward: float = 1.0
    win_reward: float = 2.0
    game_over_penalty: float = 1.0
    arc_seed: int = 7
    projection_seed: int = 1701
    topology_seed: int = 2903
    policy_seed: int = 4109


def shaped_reward(
    *,
    previous_levels: int,
    next_levels: int,
    state_name: str,
    novel_frame: bool,
    config: RunConfig,
) -> float:
    reward = -config.step_cost
    reward += config.level_reward * max(0, next_levels - previous_levels)
    if novel_frame:
        reward += config.novelty_reward
    if state_name == "WIN":
        reward += config.win_reward
    elif state_name == "GAME_OVER":
        reward -= config.game_over_penalty
    return float(reward)


def _available_actions(env: Any) -> list[Any]:
    actions = list(getattr(env, "action_space", []))
    if not actions:
        raise RuntimeError("ARC environment exposed no available actions")
    for action in actions:
        checker = getattr(action, "is_complex", None)
        if checker is not None and checker():
            raise RuntimeError(
                "FlyARC v1 supports simple ARC actions only; complex coordinate actions "
                "require a separately frozen decoder"
            )
    return actions


def run_live_variant(
    bundle: GraphBundle,
    *,
    variant: str,
    output_dir: Path,
    config: RunConfig,
    allow_candidate: bool,
    render_mode: str | None,
) -> dict[str, Any]:
    try:
        import arc_agi
    except ImportError as exc:
        raise RuntimeError(
            "ARC toolkit is not installed; run pip install -e '.[arc]' under Python 3.12+"
        ) from exc

    bundle.validate(require_sign=True, require_qualified=not allow_candidate)
    core = select_structural_core(bundle, max_nodes=config.max_nodes)
    encoder = ARCFrameEncoder(pool=config.pool, colors=config.colors)
    reservoir = FlyARCReservoir(
        core,
        variant=variant,
        feature_dim=encoder.feature_dim,
        projection_seed=config.projection_seed,
        topology_seed=config.topology_seed,
        input_fanout=config.input_fanout,
        leak=config.leak,
        gain=config.gain,
    )

    arcade = arc_agi.Arcade()
    env = arcade.make(
        config.game,
        seed=config.arc_seed,
        render_mode=render_mode,
        save_recording=False,
        include_frame_data=True,
    )
    if env is None:
        raise RuntimeError(f"ARC toolkit could not create environment {config.game!r}")

    observation = env.reset()
    if observation is None:
        raise RuntimeError("ARC environment reset returned no observation")
    actions = _available_actions(env)
    action_names = [_action_name(action) for action in actions]
    policy = LinearQPolicy(
        len(actions),
        len(core.body_ids),
        seed=config.policy_seed,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    step_path = output_dir / "steps.jsonl"
    state_path = output_dir / "states.npz"
    seen_frames: set[str] = set()
    state_rows: list[np.ndarray] = []
    action_counts: Counter[str] = Counter()
    total_reward = 0.0
    td_values: list[float] = []
    game_overs = 0
    resets = 0
    max_levels = int(getattr(observation, "levels_completed", 0))
    observed_game_ids: set[str] = set()
    won = False

    encoder.reset()
    reservoir.reset()
    frame = _extract_frame(observation)
    seen_frames.add(_frame_hash(frame))
    state = reservoir.step(encoder.encode(frame))

    with step_path.open("x") as handle:
        for step in range(config.max_steps):
            current_actions = _available_actions(env)
            current_names = [_action_name(action) for action in current_actions]
            if current_names != action_names:
                raise RuntimeError(
                    "FlyARC v1 requires a fixed action set within a run; "
                    f"expected {action_names}, observed {current_names}"
                )
            action_index, q_values = policy.choose(state, range(len(actions)))
            action = actions[action_index]
            next_observation = env.step(action)
            if next_observation is None:
                raise RuntimeError("ARC environment step returned no observation")

            next_frame = _extract_frame(next_observation)
            next_hash = _frame_hash(next_frame)
            novel = next_hash not in seen_frames
            seen_frames.add(next_hash)
            next_state = reservoir.step(encoder.encode(next_frame))
            state_label = _state_name(next_observation)
            previous_levels = int(getattr(observation, "levels_completed", 0))
            next_levels = int(getattr(next_observation, "levels_completed", 0))
            max_levels = max(max_levels, next_levels)
            reward = shaped_reward(
                previous_levels=previous_levels,
                next_levels=next_levels,
                state_name=state_label,
                novel_frame=novel,
                config=config,
            )
            done = state_label in {"WIN", "GAME_OVER"}
            td = policy.update(
                state,
                action_index,
                reward,
                next_state,
                done=done,
                valid_next=range(len(actions)),
            )

            game_id = str(getattr(next_observation, "game_id", config.game))
            observed_game_ids.add(game_id)
            row = {
                "step": step,
                "game_id": game_id,
                "variant": variant,
                "action": action_names[action_index],
                "action_index": action_index,
                "reward": reward,
                "td_error": td,
                "state": state_label,
                "levels_completed": next_levels,
                "frame_sha256": next_hash,
                "novel_frame": novel,
                "activity_mean_abs": float(np.abs(next_state).mean()),
                "activity_max_abs": float(np.abs(next_state).max(initial=0.0)),
                "q_values": [float(x) for x in q_values],
                "top_activity": reservoir.top_activity(k=12),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            state_rows.append(next_state.astype(np.float32, copy=True))
            action_counts[action_names[action_index]] += 1
            total_reward += reward
            td_values.append(td)

            observation = next_observation
            state = next_state

            if state_label == "WIN":
                won = True
                break
            if state_label == "GAME_OVER":
                game_overs += 1
                if resets >= config.max_resets:
                    break
                reset_observation = env.reset()
                if reset_observation is None:
                    raise RuntimeError("ARC reset after GAME_OVER returned no observation")
                resets += 1
                observation = reset_observation
                encoder.reset()
                reservoir.reset()
                reset_frame = _extract_frame(observation)
                seen_frames.add(_frame_hash(reset_frame))
                state = reservoir.step(encoder.encode(reset_frame))

    states = (
        np.stack(state_rows).astype(np.float32, copy=False)
        if state_rows
        else np.zeros((0, len(core.body_ids)), dtype=np.float32)
    )
    np.savez_compressed(
        state_path,
        states=states,
        body_ids=np.asarray(core.body_ids, dtype=np.int64),
    )

    scorecard_payload = None
    try:
        scorecard = arcade.close_scorecard()
        scorecard_payload = _jsonable(scorecard) if scorecard is not None else None
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        scorecard_payload = {"error": type(exc).__name__, "message": str(exc)}

    metrics = {
        "variant": variant,
        "game": config.game,
        "observed_game_ids": sorted(observed_game_ids),
        "steps": len(state_rows),
        "won": won,
        "max_levels_completed": max_levels,
        "game_overs": game_overs,
        "resets": resets,
        "unique_frames": len(seen_frames),
        "total_shaped_reward": total_reward,
        "mean_abs_td_error": float(np.mean(np.abs(td_values))) if td_values else 0.0,
        "action_counts": dict(sorted(action_counts.items())),
        "state_dim": len(core.body_ids),
        "edge_count": reservoir.edge_count,
        "rewire_swaps": reservoir.rewire_swaps,
        "scorecard": scorecard_payload,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "steps_path": step_path,
        "states_path": state_path,
    }


def run_comparison(
    bundle: GraphBundle,
    *,
    variants: Sequence[str],
    output_dir: Path,
    config: RunConfig,
    allow_candidate: bool,
    render_mode: str | None,
) -> dict[str, Any]:
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown FlyARC variants: {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("FlyARC variants must be unique")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "experiment": "flyarc-v1",
        "claim_status": (
            "development-candidate" if allow_candidate else "qualified-graph-input"
        ),
        "scientific_question": (
            "Does fixed biological MaleCNS recurrent topology provide a more useful internal "
            "state for ARC-AGI-3 interaction than matched topology controls?"
        ),
        "model_id": MODEL_ID,
        "encoder_id": ENCODER_ID,
        "reward_id": REWARD_ID,
        "selection_policy": SELECTION_POLICY,
        "variants": list(variants),
        "config": asdict(config),
        "graph_manifest": bundle.manifest,
        "git_head": _git_head(),
        "claim_boundaries": [
            (
                "ARC inputs and action semantics are engineered interfaces, not natural fly "
                "senses or actions."
            ),
            "Reservoir activity is modeled state, not recorded neural activity.",
            "The linear TD policy is an engineered trainable readout.",
            "A game win does not establish biological understanding or reasoning.",
            (
                "Topology claims require matched controls and repeated seeds; a hero run is "
                "illustrative only."
            ),
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    runs: dict[str, Any] = {}
    observed_versions: set[str] = set()
    for variant in variants:
        result = run_live_variant(
            bundle,
            variant=variant,
            output_dir=output_dir / variant,
            config=config,
            allow_candidate=allow_candidate,
            render_mode=render_mode,
        )
        runs[variant] = result
        observed_versions.update(result["metrics"]["observed_game_ids"])

    comparison = {
        "experiment": "flyarc-v1",
        "paired_seed": config.arc_seed,
        "observed_game_ids": sorted(observed_versions),
        "paired_version_check": len(observed_versions) <= 1,
        "results": {variant: runs[variant]["metrics"] for variant in variants},
    }
    if len(observed_versions) > 1:
        comparison["warning"] = (
            "ARC variants observed different game IDs/versions; do not interpret this run "
            "as a paired topology comparison."
        )
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    receipt_files = [manifest_path, comparison_path]
    for result in runs.values():
        receipt_files.extend(
            [result["metrics_path"], result["steps_path"], result["states_path"]]
        )
    receipt = {
        "experiment": "flyarc-v1",
        "files": {
            str(path.relative_to(output_dir)): _sha256(path) for path in receipt_files
        },
        "paired_version_check": comparison["paired_version_check"],
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled MaleCNS reservoir comparison on an ARC-AGI-3 environment"
        )
    )
    parser.add_argument("graph", help="signed GraphBundle directory")
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--output", default="artifacts/flyarc-ls20-v1")
    parser.add_argument(
        "--variant",
        action="append",
        choices=VARIANTS,
        help="repeatable; defaults to intact, rewire, random, stateless",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-resets", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--arc-seed", type=int, default=7)
    parser.add_argument("--topology-seed", type=int, default=2903)
    parser.add_argument("--projection-seed", type=int, default=1701)
    parser.add_argument("--policy-seed", type=int, default=4109)
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="development only: allow an unqualified graph bundle",
    )
    parser.add_argument(
        "--render",
        choices=("terminal", "terminal-fast", "human"),
        default=None,
    )
    args = parser.parse_args()

    config = RunConfig(
        game=args.game,
        max_steps=args.steps,
        max_resets=args.max_resets,
        max_nodes=args.max_nodes,
        arc_seed=args.arc_seed,
        topology_seed=args.topology_seed,
        projection_seed=args.projection_seed,
        policy_seed=args.policy_seed,
    )
    variants = tuple(args.variant or VARIANTS)
    bundle = GraphBundle.load(args.graph)
    comparison = run_comparison(
        bundle,
        variants=variants,
        output_dir=Path(args.output),
        config=config,
        allow_candidate=args.allow_candidate,
        render_mode=args.render,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
