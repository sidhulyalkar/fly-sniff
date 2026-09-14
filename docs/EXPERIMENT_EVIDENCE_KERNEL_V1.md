# Experiment + Evidence Kernel v1

This document defines the product-core contract that future `fly-sniff` scientific lanes should target.
It is intentionally independent of odor-navigation performance and can be merged without promoting any
candidate MaleCNS circuit into a functional claim.

Project coordination and lane ownership are tracked in issue #24.

## Why this exists

A connectome is not an executable animal. Every executable experiment adds assumptions about sensory
transduction, neural dynamics, output mappings, fitting, null models, and evaluation. Those assumptions
must be explicit scientific objects rather than invisible glue.

The core execution chain is:

```text
EvidenceLedger
     |
     v
ExperimentSpec
     |
   validate
     |
     v
ExperimentLock  -- exact code/runtime binding
     |
     v
    run
     |
     v
RunReceipt      -- content-addressed outputs
```

A renderer or paper figure should ultimately consume a `RunReceipt`, never recompute claim-bearing neural
dynamics behind the viewer's back.

## Epistemic types

`fly_sniff.evidence.EvidenceClass` makes the project's claim boundary machine readable:

- `measured_structure`
- `measured_physiology`
- `predicted_annotation`
- `cross_dataset_prior`
- `model_assumption`
- `fitted_parameter`
- `modeled_state`
- `behavioral_output`

These labels are deliberately not interchangeable. A structural path does not support a measured
physiology claim. Simulated activity does not become recorded neural activity because it is rendered on a
real neuron morphology.

Each `EvidenceRecord` contains a stable record ID, subject, predicate, value, authority, optional dataset,
confidence, and provenance reference. Modeled states, fitted parameters, and behavioral outputs require a
provenance reference to the run or artifact that generated them.

`EvidenceLedger` sorts records by stable ID before hashing, so two independently assembled ledgers with the
same statements receive the same digest.

## Narrow claim support

`fly_sniff.claims.support_claim` currently supports only direct claim classes:

- structural
- measured physiology
- predicted annotation
- modeled state
- behavioral output

It requires direct evidence of the matching epistemic class for every requested subject. It does not infer
higher-order functional claims from connectivity.

Topology dependence, causal pathway dependence, and biological generalization will require validated
experiment receipts in later kernel versions.

## Three scientific programs

`ExperimentSpec` requires each experiment to declare one program.

### Program A: `latent_wiring`

Primary scientific question:

> How much task-relevant computation is already latent in biological wiring before task-specific learning?

The validator enforces:

- no navigation reward;
- no topology-specific fitting;
- no whole-graph backpropagation;
- at least one independent calibration target.

This is the primary future odor-navigation claim lane.

### Program B: `topology_inductive_bias`

Question:

> Does biological topology learn the task better or faster than matched controls?

Topology-specific fitting is permitted only when optimization budgets are identical across compared
topologies. Whole-graph backprop remains outside the credible v1 contract.

The existing sealed matched task-optimization work belongs here.

### Program C: `biological_learning`

Question:

> Can prespecified biologically supported plasticity mechanisms modify behavior?

The validator requires an explicit plasticity scope and forbids whole-graph backpropagation. A future
mushroom-body experiment could therefore permit KC→MBON plasticity while keeping unrelated circuitry
frozen.

## Confirmatory topology claims

For a confirmatory `ExperimentSpec` that requests a topology claim, v1 requires:

- at least 31 frozen topology nulls;
- paired episode conditions;
- hidden final entropy before model freeze;
- a one-way final evaluation;
- retention of negative results;
- at least one declared null family.

Thirty-one is a minimum credibility gate, not an assertion that 31 is sufficient for every future study.
For the flagship odor experiment, 63 nulls remain the preferred target where compute permits.

Development experiments may use smaller null cohorts, but cannot silently promote themselves into a
confirmatory topology claim.

## Content-addressed artifacts

Every scientific input or output referenced by the kernel uses `ArtifactRef`:

```text
name
kind
sha256
uri
```

`ExperimentLock` binds the validated spec to an exact code reference and runtime digest.

`RunReceipt` binds completed, blocked, or failed execution to one exact lock. Completed runs require at
least one content-addressed result artifact. A receipt is not equivalent to scientific success: `blocked`
and `failed` are first-class outcomes, and a `completed` run may still be a null biological result.

## Example Program A shape

```python
spec = ExperimentSpec(
    experiment_id="odor-zero-shot-v1",
    program=ExperimentProgram.LATENT_WIRING,
    phase=ExperimentPhase.DEVELOPMENT,
    scientific_question="Does intact MaleCNS topology improve unseen plume navigation?",
    evidence_ledger_sha256=ledger.sha256,
    artifacts=(connectome, sensory_recording, dynamics_contract),
    null_families=("degree_preserving",),
    interventions=("intact", "pfn_acute_lesion", "steering_acute_lesion"),
    primary_metrics=("spl",),
    training_policy=TrainingPolicy(
        navigation_reward_allowed=False,
        topology_specific_fit_allowed=False,
        equal_budget_across_topologies=True,
        whole_graph_backprop_allowed=False,
        calibration_targets=("pfn_airflow_tuning", "dna02_turn_relationship"),
    ),
    final_policy=FinalPolicy(
        topology_claim=False,
        topology_null_count=8,
        paired_episodes_required=True,
        hidden_final_entropy_required=False,
        negative_results_retained=True,
        one_way_final=False,
    ),
)
```

The example is a development shape only. It does not freeze the eventual physiology targets, circuit
membership, null hierarchy, sensor model, or final metric thresholds.

## Integration rules

1. Scientific branches should export evidence and artifacts into this contract rather than importing
   performance logic into the kernel.
2. PR #22 remains a sensory/data-validation lane until its physical calibration gates pass.
3. PR #23 remains structure-only.
4. Existing task optimization is Program B and must not be used to claim Program A latent computation.
5. Negative/blocked artifacts remain valid evidence and should never be overwritten by a later successful
   protocol version.
6. UI work should consume ledger classes and receipts so `MEASURED`, `PREDICTED`, and `MODELED` status is
   visible rather than encoded only in prose.

## Next kernel tranche

After this base lands:

1. add file serialization helpers and a validation CLI;
2. add `ExperimentLock` environment/package fingerprinting;
3. add a higher-order claim engine that requires receipts for topology/causal claims;
4. connect experimental-plume and structural-audit artifacts without granting either lane navigation
   authority;
5. add commit-reveal final entropy tooling;
6. build the null-family interface and topology-level inference layer.
