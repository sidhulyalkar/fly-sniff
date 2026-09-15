# Program A preflight v1

Program A asks a narrower question than the task-optimized navigation lane:

> With dynamics constrained independently of navigation performance, does intact MaleCNS-derived wiring outperform prespecified topology controls on an unseen odor-navigation benchmark?

The preflight layer exists to make it difficult to accidentally answer a different question.

## Fail closed

The Program A integration surface has only two readiness states:

- `READY_TO_LOCK`
- `BLOCKED`

There is no warning-only state for a required scientific dependency. A file being present is not evidence that its scientific gate passed.

Every dependency therefore has both:

1. an exact `ArtifactRef` that must be present byte/science-hash-equivalently in the `ExperimentSpec`; and
2. an explicit `pass` or `blocked` scientific gate state.

A blocked artifact remains useful evidence, but it cannot unlock the confirmatory experiment.

## Required dependency families

The v1 preflight requires hash-bound references for:

- evidence ledger;
- exact MaleCNS graph / body-role membership;
- role, sign and laterality authority;
- physiology calibration protocol;
- frozen calibrated dynamics or a prespecified sensitivity-family artifact;
- plume source/preprocessing/time contract;
- physical bilateral sensor geometry;
- topology-null protocol;
- intervention/lesion protocol;
- environment + OOD contract;
- hidden final-entropy commitment.

This list is intentionally stricter than the generic `ExperimentSpec` kernel.

## Program A restrictions

The preflight additionally requires:

- `program = latent_wiring`;
- `phase = confirmatory`;
- exactly one headline primary graph-level statistic;
- no navigation reward in calibration;
- no topology-specific fitting;
- no whole-graph backpropagation;
- at least one independently defined calibration target;
- an explicit topology claim;
- at least 31 frozen null graphs **per declared confirmatory null family**;
- paired episode/environment conditions;
- hidden final entropy;
- one-way final evaluation;
- retention of negative results.

`topology_null_count` in the v1 Program A spec is interpreted as the number of graph realizations generated for each declared confirmatory null family. Families are analyzed separately by the topology-inference layer and may not be pooled into one opaque scramble distribution.

## Lock assembly

A readiness report is a deterministic function of the exact `ExperimentSpec` and exact dependency manifest. Its hash changes when either changes.

`fly-sniff-program-a lock` recomputes readiness from those inputs and refuses to create an `ExperimentLock` unless:

- the supplied readiness report exactly matches the recomputed report; and
- the recomputed report is `READY_TO_LOCK`.

The resulting generic `ExperimentLock` binds the exact spec, code reference and runtime hash. A dependency mutation requires a new spec, readiness report and lock.

## What this does not establish

A green preflight does not establish that the biological hypothesis is true. It establishes only that the experiment about to run matches the preregistered Program A question and that every required dependency claims a reviewed PASS state under an exact content identity.

The evaluator remains separately responsible for:

- generating the frozen number of null graphs for every declared family;
- verifying each null receipt/invariant;
- evaluating identical paired conditions;
- using the frozen headline statistic;
- consuming final entropy once;
- retaining negative or null outcomes.

The preflight never imports or calls navigation training/evaluation code.
