# Task-optimization scientific red-team v1

This lane is intentionally adversarial. Its job is not to improve navigation performance. Its job is to find ways the task-optimization result could look stronger than the evidence warrants.

Target: PR #16, `feat/task-optimized-connectome-v1`.

## Non-negotiable rule

Do not relax biological, statistical, development, or final-test thresholds because a model or test fails. A red sentry means the implementation or claim boundary must be repaired, or the claim must be weakened.

## Current positive findings

The implementation already has several strong protections:

- development seeds occupy a namespace above 2.1e9 while the final benchmark samples only from 1..1,999,999,999;
- CEM candidate ranking uses training seeds, not development-validation seeds;
- common random numbers are used within each generation;
- the task-optimized controller has no direct x/y, source-coordinate, distance-to-source, or wall-distance input;
- the controller ignores world heading when body-frame odor and airflow are held fixed;
- intact, rewired, and lesioned topologies all call the same optimizer implementation with the same config object;
- the graph fingerprint binds nodes, weighted signed edges, role membership, and dataset identity.

These protections are worth keeping. The red-team suite includes positive sentries so future cleanup cannot accidentally remove them.

## Vulnerabilities found

### RT-01: final-test non-peeking is partly self-asserted

`optimize_dynamics()` writes `final_test_namespace_touched: false`, and trained qualification currently trusts that boolean. The report also contains train/validation seed hashes, but qualification does not recompute the frozen split and verify those receipts.

Risk: a forged, stale, or hand-edited report can claim that the final namespace stayed unopened while retaining a valid graph/config/parameter hash.

Required repair: qualification must recompute the frozen train/validation split, verify seed counts and hashes, verify every persisted CEM seed batch is a subset of the training split, and refuse contradictory receipts. This still cannot prove that an untrusted external process never peeked; it only makes the artifact internally auditable.

Sentry: `test_training_report_seed_receipts_are_recomputed_not_trusted`.

### RT-02: development gate is trusted as a boolean

Trained qualification reads `development_gate_passed` without recomputing it from the persisted validation deltas and frozen gate thresholds.

Risk: a report can flip the boolean without changing parameters or hashes.

Required repair: recompute the gate from sealed deltas/summaries and reject contradictions.

Sentry: `test_training_report_development_gate_is_recomputed_not_trusted`.

### RT-03: frozen eight-rewire contract is not actually enforced

The v1 config says the matched topology control is eight independently seeded rewires at 8 swaps per edge. The public function/CLI accepts any `rewire_count >= 2`, and `swaps_per_edge` remains a free function argument.

Risk: a cheaper or luckier null ensemble can be run and still look like the v1 matched-control protocol.

Required repair: encode `rewire_count` and `swaps_per_edge` as structured frozen config values and require exact equality for protocol v1. A different budget must use a different protocol/version.

Sentries:

- `test_v1_matched_control_contract_refuses_fewer_than_eight_rewires`
- `test_v1_matched_control_contract_refuses_changed_swap_budget`

### RT-04: rewire completion trusts a mutable manifest flag

`_require_complete_rewire()` currently checks only `rewire.mixing_complete`.

Risk: contradictory metadata such as `mixing_complete=true`, `accepted_swaps=0`, `target_swaps=100` is accepted. More importantly, reaching a requested number of accepted double-edge swaps is a heuristic amount of rewiring, not a mathematical proof that the Markov chain has mixed.

Required repair: independently verify the swap receipt, exact in/out-degree preservation, node/role identity, edge count, absence of self-loops/duplicates, and changed-topology diagnostics. Rename any language that implies proven Markov-chain mixing unless such a proof/diagnostic is actually supplied.

Sentry: `test_rewire_completion_cannot_be_asserted_by_boolean_only`.

### RT-05: graph identity is fingerprinted after training but not guarded during training

`GraphBundle` is a frozen dataclass, but its DataFrames, role dictionary, and role lists remain mutable objects. `optimize_dynamics()` does not currently seal the starting fingerprint and check it after evaluation calls.

Risk: an accidental or malicious evaluator can mutate role membership or topology during optimization, after which the report simply fingerprints the mutated graph.

Required repair: capture the graph fingerprint before any baseline/training evaluation, assert it before and after every optimization phase, and fail closed on mutation.

Sentry: `test_optimizer_rejects_role_or_topology_mutation_during_training`.

### RT-06: sensory/steering role overlap can bypass the lesion

The controller requires all six modeled roles to be non-empty, but does not require the sensory-drive roles to be disjoint from the steering roles.

If a body ID belongs to both a PFN/odor role and a DNa02 steering role, `lesion_incoming_to_roles()` removes structural incoming edges but `TaskOptimizedMaleCNSController.act()` can still inject sensory drive directly into that steering neuron. Optimization can then compensate through unrelated sensory/readout gains, making the lesion uninterpretable.

Required repair: enforce role-disjointness needed by the causal claim before training and in trained E002. At minimum, all four directly injected sensory roles must be disjoint from both steering roles; left/right steering must remain mutually disjoint.

Sentry: `test_sensory_and_steering_roles_must_be_disjoint_to_make_lesion_interpretable`.

### RT-07: objective component contracts are assumed, not checked

The optimizer assumes success is boolean, SPL is finite and in [0,1], and terminal progress is finite and in [-1,1]. `_episode_objective()` itself accepts invalid or non-finite components.

Risk: a simulator/metric regression can silently create an exploitable objective rather than failing at the scientific boundary.

Required repair: validate component finiteness/ranges before combining them. This changes no thresholds and no model power.

Sentry: `test_objective_rejects_nonfinite_or_out_of_contract_components`.

### RT-08: CEM tie selection has no canonical tie-break contract

The implementation uses `np.argsort(scores)[::-1]`. Equal-score candidates are therefore resolved by sort behavior rather than an explicit scientific rule. The first population member is deliberately the current mean, but a complete tie need not select it as the best candidate.

Risk: exact ties can alter elite membership across implementations/versions and make replay less robust than the deterministic claim suggests.

Required repair: use an explicit deterministic secondary key, for example candidate index ascending, and seal that tie-break rule in the protocol/report.

Sentry: `test_cem_ties_have_a_canonical_candidate_order`.

### RT-09: equal compute budget is implemented by shared code path but not auditable from the report

Every topology currently calls `optimize_dynamics()` with the same config, which is good. The resulting report does not include a normalized compute-budget receipt such as candidate evaluations and candidate episodes.

Risk: future early stopping, retries, topology-specific shortcuts, or config plumbing changes can create unequal budgets while leaving a superficially similar report.

Required repair: persist the frozen optimizer budget and actual evaluation/episode counts. Matched-control assembly must verify equality before reporting a comparison.

Sentry: `test_optimizer_report_seals_an_auditable_compute_budget`.

## Candidate-selection leakage audit

The current CEM implementation does **not** use validation seeds to score candidates. Validation is evaluated for the untrained default and final trained mean only. That is a positive result.

However, development validation is still human-visible. Repeatedly editing the frozen config, optimizer seed, graph/roles, or parameterization after seeing validation would convert the validation set into a de facto training set. The code cannot cryptographically prove that a human never did this.

Operational requirement: only the exact frozen config hash and reviewed graph/role artifact may advance. Any new hyperparameter/config/role choice after validation exposure requires a fresh development-validation namespace or an explicitly exploratory protocol version.

Sentry: `test_candidate_selection_never_uses_validation_inside_cem`.

## Source-location and simulator-boundary leakage audit

No direct source coordinates, agent coordinates, distance-to-source, or wall distances are exposed through `Observation`. The task-optimized controller uses mean odor and body-frame airflow; it does not use the `heading` field. Those are positive boundaries.

Residual scientific risk remains indirect rather than API-level: the source is fixed at the upwind side of the arena, so airflow plus wall dynamics may support useful behavior even with weak odor dependence. That is not a code leak, but it can inflate an informal "smell found the source" interpretation.

Before a public odor-navigation claim, report negative-control performance with odor drive clamped or permuted and with arena/source geometry perturbations chosen before looking at results. Do not invent a pass threshold after seeing those controls; initially report them as diagnostics unless a threshold is preregistered.

Sentries:

- `test_observation_api_exposes_no_direct_position_source_or_wall_state`
- `test_controller_does_not_use_world_heading_when_body_sensory_inputs_match`

## Lesion interpretation rule

A lesion result is only causal evidence for the removed pathway if unrelated free parameters cannot route the same information around the lesion. The v1 steering-input lesion therefore needs both:

1. structural verification that all incoming edges to the steering roles were removed; and
2. role-interface verification that no directly injected sensory role overlaps a steering role.

If either fails, the lesion is an implementation control, not causal evidence.

## Status of the red-team branch

`redteam/task-optimization-validity-v1` is expected to be red against the current PR #16 head. That is intentional. The failing sentries define the smallest scientific hardening tranche. Positive sentries should remain green throughout repair.

No scientific threshold is changed by this red-team lane, and no additional model freedom is introduced.
