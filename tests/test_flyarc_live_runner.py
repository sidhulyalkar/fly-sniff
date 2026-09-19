import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fly_sniff.flyarc import RunConfig, run_live_variant
from fly_sniff.graph import GraphBundle


class _Action:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def is_complex():
        return False


class _Observation:
    def __init__(self, value, *, state="NOT_FINISHED", levels=0):
        self.frame = np.array([[[value, 0], [0, value]]], dtype=np.uint8)
        self.state = SimpleNamespace(name=state)
        self.levels_completed = levels
        self.game_id = "ls20-fixture-v1"


class _Env:
    def __init__(self):
        self.action_space = [_Action("ACTION1"), _Action("ACTION2")]
        self.counter = 0

    def reset(self):
        self.counter = 0
        return _Observation(1)

    def step(self, action):
        assert action in self.action_space
        self.counter += 1
        if self.counter == 2:
            return _Observation(3, state="GAME_OVER")
        return _Observation(2 + self.counter)


class _Arcade:
    def __init__(self):
        self.env = None

    def make(self, game, **kwargs):
        assert game == "ls20"
        assert kwargs["include_frame_data"] is True
        self.env = _Env()
        return self.env

    @staticmethod
    def close_scorecard():
        return {"fixture": True}


def _bundle():
    nodes = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame(
        {
            "source": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "target": [2, 3, 4, 5, 6, 1, 4, 5, 6, 1, 2, 3],
            "weight": [9, 8, 7, 6, 5, 4, 7, 7, 6, 6, 5, 5],
            "sign": [1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1],
        }
    )
    return GraphBundle(nodes=nodes, edges=edges, roles={}, manifest=None)


def test_live_runner_preserves_reset_observations(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace(Arcade=_Arcade))
    output = tmp_path / "intact"
    result = run_live_variant(
        _bundle(),
        variant="intact",
        output_dir=output,
        config=RunConfig(
            max_steps=3,
            max_resets=1,
            max_nodes=6,
            pool=2,
            input_fanout=2,
        ),
        allow_candidate=True,
        render_mode=None,
    )

    with np.load(output / "frames.npz") as archive:
        frames = archive["frames"]
        shapes = archive["shapes"]
    with np.load(output / "states.npz") as archive:
        states = archive["states"]

    rows = [
        json.loads(line)
        for line in (output / "steps.jsonl").read_text().splitlines()
        if line
    ]

    assert result["metrics"]["steps"] == 3
    assert result["metrics"]["game_overs"] == 1
    assert result["metrics"]["resets"] == 1
    assert frames.shape[0] == 5
    assert shapes.shape == (5, 2)
    assert states.shape == (3, 6)
    assert [row["observation_index"] for row in rows] == [1, 2, 4]
    assert result["metrics"]["observed_game_ids"] == ["ls20-fixture-v1"]
    assert result["metrics"]["scorecard"] == {"fixture": True}
