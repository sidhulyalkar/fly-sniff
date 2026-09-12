from fly_sniff.r002_qualify import evaluate_qualification


def _report(*, rate: float, separation: float, lateralization: float, direct_peak: float) -> dict:
    return {
        "direct_stronger_than_near_rate": rate,
        "mean_direct_minus_near_target": separation,
        "mean_direct_lateralization": lateralization,
        "trials_detail": [{"direct_peak_target": direct_peak} for _ in range(4)],
    }


def test_r002_qualification_passes_only_when_all_frozen_gates_pass():
    intact = _report(rate=1.0, separation=0.20, lateralization=0.30, direct_peak=0.50)
    rewires = [
        _report(rate=0.6, separation=0.05 + index * 0.01, lateralization=0.10, direct_peak=0.2)
        for index in range(5)
    ]
    lesion = _report(rate=0.5, separation=0.0, lateralization=0.0, direct_peak=0.01)
    decision = evaluate_qualification(
        structural_exact=True,
        intact=intact,
        rewires=rewires,
        lesion=lesion,
    )
    assert decision["pass"] is True
    assert all(decision["gates"].values())


def test_r002_qualification_fails_topology_gate_when_rewires_match_intact():
    intact = _report(rate=1.0, separation=0.20, lateralization=0.30, direct_peak=0.50)
    rewires = [
        _report(rate=1.0, separation=0.25, lateralization=0.35, direct_peak=0.50)
        for _ in range(5)
    ]
    lesion = _report(rate=0.5, separation=0.0, lateralization=0.0, direct_peak=0.01)
    decision = evaluate_qualification(
        structural_exact=True,
        intact=intact,
        rewires=rewires,
        lesion=lesion,
    )
    assert decision["pass"] is False
    assert not decision["gates"]["intact_beats_at_least_4_of_5_rewires_on_separation"]
    assert not decision["gates"]["intact_beats_at_least_4_of_5_rewires_on_lateralization"]


def test_r002_qualification_requires_combined_visual_lesion_to_collapse_escape():
    intact = _report(rate=1.0, separation=0.20, lateralization=0.30, direct_peak=0.50)
    rewires = [
        _report(rate=0.6, separation=0.05, lateralization=0.10, direct_peak=0.20)
        for _ in range(5)
    ]
    lesion = _report(rate=0.8, separation=0.10, lateralization=0.10, direct_peak=0.10)
    decision = evaluate_qualification(
        structural_exact=True,
        intact=intact,
        rewires=rewires,
        lesion=lesion,
    )
    assert decision["pass"] is False
    assert not decision["gates"]["combined_visual_lesion_reduces_direct_escape_below_5pct"]
