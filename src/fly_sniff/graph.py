from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .controllers import Action, Controller
from .env import Observation


@dataclass(frozen=True)
class GraphBundle:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    roles: dict[str, list[int]]
    manifest: dict | None = None

    @classmethod
    def load(cls, directory: str | Path) -> GraphBundle:
        root = Path(directory)
        nodes = pd.read_parquet(root / "nodes.parquet")
        edges = pd.read_parquet(root / "edges.parquet")
        roles = json.loads((root / "roles.json").read_text())
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        return cls(
            nodes=nodes,
            edges=edges,
            roles={k: [int(x) for x in v] for k, v in roles.items()},
            manifest=manifest,
        )

    def validate(self, require_sign: bool = False, require_qualified: bool = False) -> None:
        required_nodes = {"bodyId"}
        required_edges = {"source", "target", "weight"}
        if not required_nodes.issubset(self.nodes.columns):
            raise ValueError(f"nodes missing columns: {required_nodes - set(self.nodes.columns)}")
        if not required_edges.issubset(self.edges.columns):
            raise ValueError(f"edges missing columns: {required_edges - set(self.edges.columns)}")
        if require_sign and "sign" not in self.edges.columns:
            raise ValueError(
                "qualified neural dynamics require an explicit edge sign column; "
                "unsigned synapse counts cannot silently become excitatory weights"
            )
        if "sign" in self.edges.columns:
            signs = set(pd.Series(self.edges.sign).dropna().astype(int).unique())
            if not signs.issubset({-1, 0, 1}):
                raise ValueError(f"sign must be in -1/0/+1; observed {sorted(signs)}")
        if require_qualified and (
            not self.manifest or self.manifest.get("qualification_status") != "qualified"
        ):
            raise ValueError(
                "graph is not sealed as qualification_status='qualified'; "
                "candidate graphs may be explored but not labelled as a MaleCNS result"
            )
        ids = set(self.nodes.bodyId.astype(int))
        missing = (set(self.edges.source.astype(int)) | set(self.edges.target.astype(int))) - ids
        if missing:
            raise ValueError(f"edges reference {len(missing)} unknown body IDs")
        for role, body_ids in self.roles.items():
            unknown = set(body_ids) - ids
            if unknown:
                raise ValueError(f"role {role!r} references unknown IDs: {sorted(unknown)[:5]}")


class MaleCNSRateController(Controller):
    """Explicit rate-model assumption over an extracted MaleCNS topology.

    Structural connectivity comes from MaleCNS. The state update is a modeled
    dynamical assumption, not a biological recording. Qualified mode refuses
    unsigned or unreviewed graph bundles.
    """

    name = "malecns-rate-v0"

    def __init__(
        self,
        bundle: GraphBundle,
        leak: float = 0.82,
        gain: float = 1.6,
        *,
        require_qualified: bool = True,
    ):
        bundle.validate(require_sign=True, require_qualified=require_qualified)
        self.bundle = bundle
        self.leak = float(leak)
        self.gain = float(gain)
        ids = bundle.nodes.bodyId.astype(int).tolist()
        self.ids = ids
        self.index = {body_id: i for i, body_id in enumerate(ids)}
        src = bundle.edges.source.astype(int).map(self.index).to_numpy()
        dst = bundle.edges.target.astype(int).map(self.index).to_numpy()
        raw = np.log1p(bundle.edges.weight.astype(float).to_numpy())
        sign = bundle.edges.sign.astype(float).to_numpy()
        values = raw * sign
        mat = sparse.coo_matrix((values, (dst, src)), shape=(len(ids), len(ids))).tocsr()
        row_norm = np.asarray(np.abs(mat).sum(axis=1)).ravel()
        inv = np.divide(1.0, row_norm, out=np.ones_like(row_norm), where=row_norm > 0)
        self.w = sparse.diags(inv) @ mat
        self.activity = np.zeros(len(ids), dtype=float)
        self._diag: dict[str, float] = {}

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.activity.fill(0.0)
        self._diag = {}

    def _inject(self, role: str, value: float, drive: np.ndarray) -> None:
        for body_id in self.bundle.roles.get(role, []):
            if body_id in self.index:
                drive[self.index[body_id]] += value

    def _role_mean(self, role: str) -> float:
        idx = [self.index[x] for x in self.bundle.roles.get(role, []) if x in self.index]
        return float(self.activity[idx].mean()) if idx else 0.0

    def act(self, obs: Observation) -> Action:
        drive = np.zeros_like(self.activity)
        self._inject("odor_left", obs.left_odor, drive)
        self._inject("odor_right", obs.right_odor, drive)
        self._inject("wind_forward", max(obs.wind_x_body, 0.0), drive)
        self._inject("wind_backward", max(-obs.wind_x_body, 0.0), drive)
        self._inject("wind_left", max(obs.wind_y_body, 0.0), drive)
        self._inject("wind_right", max(-obs.wind_y_body, 0.0), drive)
        recurrent = self.w @ self.activity
        proposal = np.tanh(self.gain * (recurrent + drive))
        self.activity = self.leak * self.activity + (1.0 - self.leak) * proposal
        left = self._role_mean("steer_left")
        right = self._role_mean("steer_right")
        turn = float(np.tanh(2.4 * (right - left)))
        self._diag = {
            "dn_left": left,
            "dn_right": right,
            "activity_mean": float(np.abs(self.activity).mean()),
            "modeled_dynamics": 1.0,
        }
        return Action(turn=turn, speed=1.0)

    def diagnostics(self) -> dict[str, float]:
        return self._diag.copy()

    def activity_snapshot(self, *, limit: int = 256) -> dict[str, Any] | None:
        """Return the strongest modeled neuron activities without inventing spikes.

        The values are rate-model state variables in [-1, 1], not measured
        electrophysiology. Sparse top-|activity| storage keeps social recordings
        compact while preserving exact body IDs for an anatomical replay.
        """
        if limit < 1 or not len(self.activity):
            return None
        count = min(int(limit), len(self.activity))
        if count == len(self.activity):
            indices = np.arange(len(self.activity), dtype=int)
        else:
            indices = np.argpartition(np.abs(self.activity), -count)[-count:]
        indices = indices[np.argsort(np.abs(self.activity[indices]))[::-1]]
        cells = [
            {
                "body_id": int(self.ids[int(index)]),
                "activity": float(self.activity[int(index)]),
            }
            for index in indices
            if abs(float(self.activity[int(index)])) > 1e-9
        ]
        status = (self.bundle.manifest or {}).get("qualification_status", "candidate")
        return {
            "model": self.name,
            "claim_status": str(status),
            "signal_kind": "modeled_rate_state",
            "cells": cells,
        }
