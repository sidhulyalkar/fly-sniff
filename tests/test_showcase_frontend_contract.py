from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "web" / "showcase"


def test_showcase_v2_frontend_keeps_synchronized_visual_contract() -> None:
    html = (SHOWCASE / "index.html").read_text()
    app = (SHOWCASE / "app.js").read_text()

    for element_id in (
        "arenaCanvas",
        "connectomeCanvas",
        "signalReadout",
        "signalSensory",
        "signalBilateral",
        "signalOutput",
        "leftOdorValue",
        "rightOdorValue",
        "turnValue",
        "readoutValue",
    ):
        assert f'id="{element_id}"' in html

    for view in ("front", "oblique", "side"):
        assert f'data-view="{view}"' in html

    assert "function drawFly(" in app
    assert "function updateSignalReadout(" in app
    assert "function setViewPreset(" in app
    assert "browser replays frozen state" in html


def test_showcase_v2_does_not_claim_named_odor_drives_behavior() -> None:
    html = (SHOWCASE / "index.html").read_text()

    assert "This selector does <em>not</em> drive the movement replay yet." in html
    assert "presentation replays frozen state" in html
