from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from .graph import GraphBundle


@dataclass(frozen=True)
class RuntimeSnapshot:
    """One modeled connectome step with explicitly named interface channels."""

    step: int
    inputs: dict[str, float]
    readouts: dict[str, float]
    activity_mean: float
    activity_max: float


class ConnectomeRuntime:
    """Generic rate-model runtime over a signed ``GraphBundle``.

    The runtime deliberately knows nothing about odor, vision, games, or motor
    semantics. Adapters inject named role channels and request named role
    readouts. This keeps biological wiring separate from engineered interfaces.

    Structural connectivity can come from MaleCNS. The rate dynamics, channel
    encoding, and output decoding remain explicit modeling assumptions.
    """

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
        self.ids = bundle.nodes.bodyId.astype(int).tolist()
        self.index = {body_id: i for i, body_id in enumerate(self.ids)}

        src = bundle.edges.source.astype(int).map(self.index).to_numpy()
        dst = bundle.edges.target.astype(int).map(self.index).to_numpy()
        raw = np.log1p(bundle.edges.weight.astype(float).to_numpy())
        sign = bundle.edges.sign.astype(float).to_numpy()
        values = raw * sign
        mat = sparse.coo_matrix(
            (values, (dst, src)), shape=(len(self.ids), len(self.ids))
        ).tocsr()
        row_norm = np.asarray(np.abs(mat).sum(axis=1)).ravel()
        inv = np.divide(1.0, row_norm, out=np.ones_like(row_norm), where=row_norm > 0)
        self.w = sparse.diags(inv) @ mat
        self.activity = np.zeros(len(self.ids), dtype=float)
        self.step_index = 0

    def reset(self) -> None:
        self.activity.fill(0.0)
        self.step_index = 0

    def _role_indices(self, role: str) -> list[int]:
        return [self.index[x] for x in self.bundle.roles.get(role, []) if x in self.index]

    def role_mean(self, role: str) -> float:
        idx = self._role_indices(role)
        return float(self.activity[idx].mean()) if idx else 0.0

    def step(
        self,
        inputs: Mapping[str, float],
        *,
        readouts: Iterable[str] = (),
    ) -> RuntimeSnapshot:
        """Advance one modeled neural step.

        ``inputs`` maps adapter channel names to GraphBundle roles. Unknown roles
        are rejected instead of silently dropping sensory data. ``readouts`` may
        include an empty role, which deterministically reports zero.
        """

        unknown = sorted(set(inputs) - set(self.bundle.roles))
        if unknown:
            raise ValueError(f"runtime inputs reference unknown roles: {unknown}")

        drive = np.zeros_like(self.activity)
        clean_inputs: dict[str, float] = {}
        for role, raw_value in inputs.items():
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"runtime input {role!r} must be finite")
            clean_inputs[role] = value
            for idx in self._role_indices(role):
                drive[idx] += value

        recurrent = self.w @ self.activity
        proposal = np.tanh(self.gain * (recurrent + drive))
        self.activity = self.leak * self.activity + (1.0 - self.leak) * proposal
        self.step_index += 1

        named_readouts = {role: self.role_mean(role) for role in readouts}
        return RuntimeSnapshot(
            step=self.step_index,
            inputs=clean_inputs,
            readouts=named_readouts,
            activity_mean=float(np.abs(self.activity).mean()),
            activity_max=float(np.abs(self.activity).max(initial=0.0)),
        )
