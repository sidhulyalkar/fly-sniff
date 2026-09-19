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
        "eventStrip",
        "worldHud",
        "compareSummary",
        "compareCards",
        "o002OdorCount",
        "o002UnitCount",
        "o002ClassCount",
    ):
        assert f'id="{element_id}"' in html

    for view in ("front", "oblique", "side"):
        assert f'data-view="{view}"' in html

    for mode in ("watch", "compare", "prove"):
        assert f'data-mode="{mode}"' in html

    for function_name in (
        "drawFly",
        "drawDevelopmentController",
        "updateSignalReadout",
        "setViewPreset",
        "deriveEvents",
        "setupCompareCards",
        "setMode",
    ):
        assert f"function {function_name}(" in app

    assert "browser replays frozen state" in html
    assert 'qs("o002Image")' not in app


def test_showcase_v2_development_mode_is_visually_complete_without_fake_anatomy() -> None:
    html = (SHOWCASE / "index.html").read_text()
    app = (SHOWCASE / "app.js").read_text()

    assert "CIRCUIT / CONTROLLER VIEW" in html
    assert "NOT MALECNS ACTIVITY" in app
    assert "DEVELOPMENT CONTROLLER • NOT MALECNS ACTIVITY" in app
    assert "visualized from replayed controller inputs/outputs only" in app
    assert 'qs("viewPresets").hidden = !hasMeasuredAnatomy' in app


def test_showcase_v2_does_not_claim_named_odor_drives_behavior() -> None:
    html = (SHOWCASE / "index.html").read_text()

    assert "This selector does <em>not</em> drive the movement replay yet." in html
    assert "presentation replays frozen state" in html
    assert "Measured • development only".casefold() in html.casefold()


def test_showcase_v2_separates_watch_compare_and_prove() -> None:
    html = (SHOWCASE / "index.html").read_text()
    css = (SHOWCASE / "styles.css").read_text()

    assert "<b>WATCH</b>" in html
    assert "<b>COMPARE</b>" in html
    assert "<b>PROVE</b>" in html
    assert 'body[data-view-mode="watch"]' in css
    assert 'body[data-view-mode="compare"]' in css
    assert 'body[data-view-mode="prove"]' in css
