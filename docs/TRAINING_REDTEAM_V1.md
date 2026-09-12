# Task-optimization scientific red-team v1

This lane is intentionally adversarial. Its job is not to improve navigation performance. Its job is to find ways the task-optimization result could look stronger than the evidence warrants, then make those failure modes executable wherever possible.

Target: PR #16, `feat/task-optimized-connectome-v1`.

## Non-negotiable rule

Do not relax biological, statistical, development, or final-test thresholds because a model or test fails. A red sentry means the implementation or claim boundary must be repaired, or the claim must be weakened.

No red-team repair may add controller inputs, trainable parameters, edge-level freedom, topology mutations, role mutations, or extra optimization budget.

## Current status

The first red-team tranche found nine implementation-level validity holes. They are now hardened on `redteam/task-optimization-validity-v1` and covered by regression sentries. The ordinary CI suite passed after this hardening before the shortcut-diagnostic tranche was added.

The repaired guarantees are about **internal artifact integrity**. They do not prove that an external human/process never inspected final outcomes, that accepted double-edge swaps establish Markov-chain stationarity, or that a high navigation score necessarily depends on odor. Those are kept as explicit residual risks rather than silently promoted to guarantees.

## Positive protections retained

The implementation retains the following strong boundaries:

- development seeds occupy a namespace above 2.1e9 while final benchmark seeds are restricted to `1..1,999,999,999`;
- CEM candidate ranking uses training seeds, not development-validation seeds;
- common random numbers are mandatory within each generation;
- the controller has no direct x/y, source-coordinate, distance-to-source, or wall-distance input;
- world heading does not affect the task controller when body-frame sensory inputs are held fixed;
- graph fingerprints bind body IDs, weighted signed edges, role membership, and dataset identity;
- the model remains limited to the same six global dynamics parameters.

These are positive sentries, not claims about living-fly physiology.

## Repaired implementation vulnerabilities

### RT-01: final-test non-peeking was partly self-asserted

Previous state: qualification trusted `final_test_namespace_touched=false` and did not reconstruct the frozen seed split.

Repair:

- reports now persist exact train and validation seed lists, counts, and hashes;
- qualification recomputes the split from the frozen config and requires exact equality;
- optimizer/evaluation entry points reject final-test-namespace seeds;
- a compound audit receipt binds graph, config, seed, parameter, and budget identities.

Sentry: `test_training_report_seed_receipts_are_recomputed_not_trusted`.

Boundary: this establishes internal consistency of the supplied run. It cannot prove an external person or process never peeked at a final result.

### RT-02: development gate was trusted as mutable metadata

Previous state: qualification trusted `development_gate_passed`.

Repair: qualification recomputes objective and success deltas from persisted baseline/trained validation summaries and applies the unchanged frozen thresholds. Contradictory booleans or deltas are rejected.

Sentry: `test_training_report_development_gate_is_recomputed_not_trusted`.

### RT-03: the frozen null budget was configurable at runtime

Previous state: the v1 prose required eight rewires at 8 swaps/edge, but the function accepted smaller ensembles and arbitrary swap counts.

Repair: v1 now requires exactly eight independently seeded rewires and exactly 8 accepted swaps per edge. A different design must use a different protocol version rather than masquerading as v1.

Sentries:

- `test_v1_matched_control_contract_refuses_fewer_than_eight_rewires`
- `test_v1_matched_control_contract_refuses_changed_swap_budget`

### RT-04: rewire completion trusted one boolean

Previous state: `mixing_complete=true` was sufficient.

Repair: the training path now verifies the swap receipt, edge count, node/body-ID identity, role identity, and exact directed in/out-degree preservation. Contradictory accepted/target/attempted counts fail closed.

Sentry: `test_rewire_completion_cannot_be_asserted_by_boolean_only`.

Important limitation: reaching eight accepted double-edge swaps per edge is **rewire-budget completion**, not proof that a degree-preserving Markov chain has mixed to stationarity. Public wording must not call this a proved mixed null. Additional topology-distance diagnostics should be reported for the actual eight rewires.

### RT-05: mutable graph objects were not guarded throughout optimization

Previous state: `GraphBundle` is a frozen dataclass whose DataFrames/dicts/lists are still mutable.

Repair: optimization seals the initial replay fingerprint and checks it before and after parameter evaluations and matched-control runs. Mutation of topology, weights, signs, body IDs, or roles kills the run.

Sentry: `test_optimizer_rejects_role_or_topology_mutation_during_training`.

### RT-06: sensory/steering role overlap could bypass the lesion

Previous state: a body ID could theoretically be both a directly injected sensory role and a steering role. Removing structural input edges would not prevent direct modeled drive from entering that neuron.

Repair: all six modeled role populations are required to be pairwise disjoint before task optimization. The lesion therefore cannot be bypassed through direct role injection.

Sentries:

- `test_sensory_and_steering_roles_must_be_disjoint_to_make_lesion_interpretable`
- `test_steering_lesion_cannot_be_rescued_by_global_gain_extremes`

The second sentry explicitly drives every trainable global parameter to its allowed maximum after the steering-input lesion and verifies that steering remains zero in a connected synthetic graph.

### RT-07: invalid objective components could enter CEM

Repair: SPL must be finite in `[0,1]`; terminal progress must be finite in `[-1,1]`; aggregate objective values and evaluation summaries must be finite and internally consistent. Simulator path/distance outputs fail closed if invalid.

Sentry: `test_objective_rejects_nonfinite_or_out_of_contract_components`.

### RT-08: exact CEM ties lacked a frozen secondary key

Repair: candidates are ordered by descending objective then ascending candidate index. The tie rule is written into generation history.

Sentry: `test_cem_ties_have_a_canonical_candidate_order`.

### RT-09: equal compute budget was implicit rather than auditable

Repair: every optimization report now contains a normalized budget receipt with population, generations, episodes/candidate, candidate evaluations, candidate episodes, full-pool evaluations, and total episode evaluations. Matched controls require identical budget hashes before a comparison is emitted.

Sentries:

- `test_optimizer_report_seals_an_auditable_compute_budget`
- `test_matched_controls_require_identical_optimizer_budget_receipts`

## Candidate-selection leakage boundary

CEM candidate ranking does not consume validation seeds. Validation remains a post-optimization development measurement. That is enforced by `test_candidate_selection_never_uses_validation_inside_cem`.

However, development validation is human-visible. Repeatedly changing the config, optimizer seed, role policy, graph construction, or parameterization after inspecting validation would convert that cohort into a de facto training set. Software cannot cryptographically prove that this never happened.

Operational rule: once development validation is exposed for a frozen v1 artifact, any subsequent model-selection change requires a new explicitly versioned development protocol and a fresh validation namespace. The final benchmark remains one-way and separate.

## Source-location and simulator shortcut battery

No direct source coordinates, agent coordinates, source distance, or wall distances are exposed through `Observation`. The task controller uses mean odor and body-frame airflow and does not consume `heading`.

That does not eliminate an indirect shortcut: in the default benchmark the source is always upwind. A policy that mainly turns upwind and exploits arena geometry could appear competent while depending weakly on odor.

`fly-sniff-redteam-training` therefore evaluates the frozen trained parameters on a separate diagnostic seed namespace under seven counterfactuals:

1. normal sensory input;
2. odor clamped to zero at the controller boundary;
3. wind clamped to zero;
4. odor and wind both clamped;
5. source shifted to the lower crosswind quarter of the arena;
6. source shifted to the upper crosswind quarter;
7. mirrored source/start positions with reversed world-frame wind.

The same frozen parameters are used in every case. There is no retraining and no diagnostic result feeds CEM.

These outputs are deliberately `diagnostic_only_not_a_gate`. No pass threshold is invented after looking at the result. The correct interpretation is comparative: if odor-clamped or geometry-counterfactual performance remains close to normal, the public wording must be weaker than “the connectome smelled out the source,” even if the primary benchmark score is high.

Sentries:

- `test_observation_api_exposes_no_direct_position_source_or_wall_state`
- `test_controller_does_not_use_world_heading_when_body_sensory_inputs_match`
- `test_diagnostic_seed_namespace_is_separate_from_training_validation_and_final`
- `test_sensory_masks_change_only_explicit_controller_channels`
- `test_shortcut_battery_is_diagnostic_only_and_contains_geometry_attacks`

## Rewire-null interpretation

The current directed double-edge-swap null preserves exact directed in/out degree, retains the same node/body-ID set and role assignments, and carries presynaptic edge attributes through target swaps. This is a useful topology-disruption control, not a universal null for every graph statistic.

Before using the null in a public topology claim, the actual eight rewires should additionally report, without post-hoc pass thresholds:

- intact-vs-rewire directed edge overlap;
- pairwise rewire edge overlap;
- unique replay fingerprints for all eight rewires;
- self-loop and duplicate-edge counts;
- degree equality checks;
- sign/weight distribution summaries globally and by presynaptic role/type where interpretable.

If the rewires remain unusually close to intact or to one another, the correct response is to qualify the null design, not to increase swap counts after seeing navigation results under the same protocol label.

## What this lane can and cannot certify

It can certify, for the implementation and artifacts it verifies:

- train/validation/final namespace separation inside the training path;
- fixed model freedom;
- fixed graph/role identity during optimization;
- deterministic CEM behavior under the frozen software stack;
- exact matched optimizer budgets;
- exact v1 rewire count and swap budget;
- lesion interface integrity;
- internal report/config/hash consistency;
- diagnostic measurement of sensory and geometry shortcuts.

It cannot certify by itself:

- that nobody externally peeked at final outcomes;
- that the rewire Markov chain reached stationarity;
- that modeled neural states equal measured physiology;
- that the selected body-ID roles are biologically correct without the independent literature/role-review lane;
- that successful navigation is odor-dependent until shortcut diagnostics are actually run on the frozen trained candidate;
- that a result is publication-grade merely because CI is green.

## Claim rule

A strong public result requires all lanes to agree:

1. reviewed body-ID-resolved biological role evidence;
2. frozen task-optimization contract and internally auditable training receipt;
3. trained E002 internal-consistency qualification;
4. matched intact/rewire/lesion training under equal budgets;
5. shortcut diagnostics disclosed beside the main result;
6. one-way final held-out/OOD evaluation after all preceding artifacts are frozen.

If one of those is missing, wording must stop at the strongest completed rung.
