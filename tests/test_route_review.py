import json
from pathlib import Path

import pandas as pd

from fly_sniff.route_review import build_review, infer_instance_side


PRIMARY = [
    "fb5ab_to_hDeltaC",
    "wind_to_hDeltaC",
    "pfl3_to_dna02",
]


def _write_stage(root: Path, name: str, source_rows, target_rows):
    stage = root / name
    stage.mkdir(parents=True)
    source = pd.DataFrame(source_rows)
    target = pd.DataFrame(target_rows)
    source.to_csv(stage / "source_seeds.csv", index=False)
    target.to_csv(stage / "target_seeds.csv", index=False)
    retained_ids = source.bodyId.astype(int).tolist() + target.bodyId.astype(int).tolist()
    provenance = pd.DataFrame(
        {
            "bodyId": retained_ids,
            "is_source_seed": [True] * len(source) + [False] * len(target),
            "is_target_seed": [False] * len(source) + [True] * len(target),
        }
    )
    provenance.to_csv(stage / "path_provenance.csv", index=False)
    nodes = pd.concat([source, target], ignore_index=True)
    nodes = pd.concat(
        [
            nodes,
            pd.DataFrame(
                [
                    {
                        "bodyId": max(retained_ids) + 1000,
                        "type": "INTERMEDIATE",
                        "instance": "INTERMEDIATE_1",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    nodes.to_parquet(stage / "nodes.parquet", index=False)
    edges = pd.DataFrame(
        {
            "source": [retained_ids[0]],
            "target": [retained_ids[-1]],
            "weight": [7],
        }
    )
    edges.to_parquet(stage / "edges.parquet", index=False)


def _policy():
    return {
        "protocol": "e001-role-review-v1",
        "input_protocol": "staged-structural-discovery-v3",
        "claim_boundary": "candidate only",
        "side_inference": {
            "left_pattern": r"(?:^|_)L(?:\d|_|$)",
            "right_pattern": r"(?:^|_)R(?:\d|_|$)",
        },
        "roles": {
            "odor_context_left": {
                "stage": "fb5ab_to_hDeltaC",
                "seed_role": "source",
                "type_regex": "^FB5AB$",
                "required_side": "L",
                "minimum_count": 1,
            },
            "odor_context_right": {
                "stage": "fb5ab_to_hDeltaC",
                "seed_role": "source",
                "type_regex": "^FB5AB$",
                "required_side": "R",
                "minimum_count": 1,
            },
            "wind_basis_left": {
                "stage": "wind_to_hDeltaC",
                "seed_role": "source",
                "type_regex": "^(?:PFNa|PFNm(?:_a|_b)?)$",
                "required_side": "L",
                "minimum_count": 1,
            },
            "wind_basis_right": {
                "stage": "wind_to_hDeltaC",
                "seed_role": "source",
                "type_regex": "^(?:PFNa|PFNm(?:_a|_b)?)$",
                "required_side": "R",
                "minimum_count": 1,
            },
            "steer_left": {
                "stage": "pfl3_to_dna02",
                "seed_role": "target",
                "type_regex": "^DNa02$",
                "required_side": "L",
                "minimum_count": 1,
                "maximum_count": 1,
            },
            "steer_right": {
                "stage": "pfl3_to_dna02",
                "seed_role": "target",
                "type_regex": "^DNa02$",
                "required_side": "R",
                "minimum_count": 1,
                "maximum_count": 1,
            },
        },
        "structural_context_roles": {
            "integration_hdelta_c": {
                "handoff": "hDeltaC_odor_wind_convergence",
                "source": "shared_body_ids",
            },
            "pfl3_output": {
                "handoff": "pfl3_body_id_continuity",
                "source": "shared_body_ids",
            },
        },
        "wind_unresolved_policy": {"include_in_training_role": False},
    }


def _artifact(tmp_path: Path):
    root = tmp_path / "staged"
    root.mkdir()
    _write_stage(
        root,
        "fb5ab_to_hDeltaC",
        [
            {"bodyId": 10, "type": "FB5AB", "instance": "FB5AB_L"},
            {"bodyId": 11, "type": "FB5AB", "instance": "FB5AB_R"},
        ],
        [{"bodyId": 20, "type": "hDeltaC", "instance": "hDeltaC_1"}],
    )
    _write_stage(
        root,
        "wind_to_hDeltaC",
        [
            {"bodyId": 30, "type": "PFNa", "instance": "PFNa_L3_C1"},
            {"bodyId": 31, "type": "PFNm_b", "instance": "PFNm_b_R4_C2"},
            {"bodyId": 32, "type": "PFNp_c", "instance": "PFNp_c"},
        ],
        [{"bodyId": 20, "type": "hDeltaC", "instance": "hDeltaC_1"}],
    )
    _write_stage(
        root,
        "pfl3_to_dna02",
        [{"bodyId": 40, "type": "PFL3", "instance": "PFL3_L1_C1"}],
        [
            {"bodyId": 50, "type": "DNa02", "instance": "DNa02_L"},
            {"bodyId": 51, "type": "DNa02", "instance": "DNa02_R"},
        ],
    )
    staged = {
        "protocol": "staged-structural-discovery-v3",
        "dataset": "male-cns:v1.0",
        "primary_structural_hypothesis_passed": True,
        "input_provenance": {"annotations": {"sha256": "a"}, "weights": {"sha256": "b"}},
        "stages": [
            {
                "name": name,
                "required_for_primary_hypothesis": True,
                "status": "candidate_corridor",
                "structural_audit_passed": True,
            }
            for name in PRIMARY
        ],
    }
    (root / "staged_trace_report.json").write_text(json.dumps(staged))
    handoffs = {
        "handoffs": [
            {
                "name": "hDeltaC_odor_wind_convergence",
                "shared_body_ids": [20],
                "passed": True,
            },
            {
                "name": "pfl3_body_id_continuity",
                "shared_body_ids": [40],
                "passed": True,
            },
        ]
    }
    (root / "handoff_audit.json").write_text(json.dumps(handoffs))
    return root


def test_instance_side_inference_is_explicit_and_ambiguous_safe():
    left = r"(?:^|_)L(?:\d|_|$)"
    right = r"(?:^|_)R(?:\d|_|$)"
    assert infer_instance_side("FB5AB_L", left_pattern=left, right_pattern=right) == "L"
    assert infer_instance_side("PFNa_R3_C2", left_pattern=left, right_pattern=right) == "R"
    assert infer_instance_side("PFNp_c", left_pattern=left, right_pattern=right) is None


def test_review_derives_roles_without_assigning_side_unresolved_pfn(tmp_path):
    root = _artifact(tmp_path)
    report = build_review(root, _policy())
    roles = report["role_draft"]
    assert roles["odor_context_left"] == [10]
    assert roles["odor_context_right"] == [11]
    assert roles["wind_basis_left"] == [30]
    assert roles["wind_basis_right"] == [31]
    assert roles["steer_left"] == [50]
    assert roles["steer_right"] == [51]
    assert roles["integration_hdelta_c"] == [20]
    assert roles["pfl3_output"] == [40]
    assert 32 not in roles["wind_basis_left"] + roles["wind_basis_right"]
    assert report["wind_resolution"]["side_unresolved_count"] == 1
    assert report["all_model_role_rules_pass"]
    assert report["all_disjoint_checks_pass"]
    assert report["review_status"] == "candidate_needs_human_review"
    assert report["automatic_qualification"] is False


def test_review_reports_intermediate_composition(tmp_path):
    root = _artifact(tmp_path)
    report = build_review(root, _policy())
    stages = {item["name"]: item for item in report["primary_stage_summaries"]}
    assert stages["fb5ab_to_hDeltaC"]["intermediate_node_count"] == 1
    assert stages["fb5ab_to_hDeltaC"]["top_intermediate_types"] == [
        {"type": "INTERMEDIATE", "count": 1}
    ]
