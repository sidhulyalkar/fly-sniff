from __future__ import annotations

import copy

import pytest

from fly_sniff.recorded_showcase_v3 import _threshold_summary, _trace_payload, _validate_inputs


def _config() -> dict:
    return {
        "protocol": "who-farted-showcase-evidence-v3",
        "modeled_activity": {"behavior_sync_allowed": False},
        "behavioral_state": {"malecns_behavior_claim_allowed": False},
        "comparison": {"real_vs_rewire_headline_allowed": False},
        "mechanism_probe": {
            "e002b_required": True,
            "e002c_protocol": "E002c-pfl3-goal-heading-convergence-v1",
            "e002c_must_pass": True,
            "fc2_protocol": "malecns-fc2-goal-interface-audit-v1",
            "display_threshold": 5,
            "all_thresholds_must_be_reported": [1, 3, 5, 10],
            "phase_mapping_must_remain_unresolved": True,
        },
    }


def _activity(scale: float) -> list[list[float]]:
    return [[scale * step for _ in range(24)] for step in range(4)]


def _e002c() -> dict:
    reports = {}
    for threshold, changed, reachable in ((1, 24, 24), (3, 24, 24), (5, 23, 24), (10, 2, 21)):
        reports[str(threshold)] = {
            "goal_only": {"activity": _activity(0.1)},
            "heading_only": {"activity": _activity(0.2)},
            "joint": {"activity": _activity(0.3)},
            "joint_convergence": {
                "changed_from_both_count": changed,
                "dual_reachable_count": reachable,
            },
        }
    return {
        "protocol": "E002c-pfl3-goal-heading-convergence-v1",
        "passed": True,
        "gates": [{"name": "e002b_qualified", "passed": True}],
        "threshold_reports": reports,
        "run_config": {"steps": 4},
    }


def _fc2() -> dict:
    populations = {}
    interfaces = {}
    for family in ("FC2A", "FC2B", "FC2C"):
        populations[family] = {"instance_columns": {"parse_fraction": 1.0}}
        interfaces[family] = {
            "threshold_reports": {"10": {"target_coverage": "24/24"}}
        }
    return {
        "protocol": "malecns-fc2-goal-interface-audit-v1",
        "phase_mapping_status": "unresolved",
        "populations": populations,
        "interfaces": interfaces,
    }


def test_v3_accepts_sealed_mechanism_reports() -> None:
    _validate_inputs(_config(), _e002c(), _fc2())


def test_v3_rejects_behavior_sync_promotion() -> None:
    config = _config()
    config["modeled_activity"]["behavior_sync_allowed"] = True
    with pytest.raises(ValueError, match="separate from chase timeline"):
        _validate_inputs(config, _e002c(), _fc2())


def test_v3_rejects_unqualified_e002c() -> None:
    report = _e002c()
    report["passed"] = False
    with pytest.raises(ValueError, match="passed E002c"):
        _validate_inputs(_config(), report, _fc2())


def test_v3_rejects_fc2_phase_promotion() -> None:
    fc2 = _fc2()
    fc2["phase_mapping_status"] = "assigned"
    with pytest.raises(ValueError, match="goal angles"):
        _validate_inputs(_config(), _e002c(), fc2)


def test_v3_rejects_missing_fc2_target_coverage() -> None:
    fc2 = copy.deepcopy(_fc2())
    fc2["interfaces"]["FC2B"]["threshold_reports"]["10"]["target_coverage"] = "23/24"
    with pytest.raises(ValueError, match="FC2B->PFL3"):
        _validate_inputs(_config(), _e002c(), fc2)


def test_v3_trace_and_threshold_summary_preserve_all_preregistered_thresholds() -> None:
    report = _e002c()
    traces = _trace_payload(report, 5)
    assert traces["joint"].shape == (4,)
    assert traces["joint"][-1] > traces["heading_only"][-1] > traces["goal_only"][-1]
    assert _threshold_summary(report) == [
        (1, 24, 24),
        (3, 24, 24),
        (5, 23, 24),
        (10, 2, 21),
    ]
