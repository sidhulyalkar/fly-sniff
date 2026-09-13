# Bilateral odor-motion v2

## Why this is a separate experiment

The sealed navigation v1 contract intentionally uses mean bilateral odor only as a nondirectional
context signal. Direction comes from the body-frame airflow/PFN-basis interface. The v1 config
explicitly forbids adding instantaneous left-right odor direction after performance is observed.

Therefore temporal bilateral odor information is **not** being patched into v1. It is versioned as
a separate experiment so v1 remains an untouched comparator.

## Phase A: causal sensory sidecar

`fly-sniff-odor-motion` reads a hash-verified canonical recording and derives a visualization-ready
causal timing estimate from the already recorded modeled antenna responses.

For each fixed delay `d`:

```text
E_d(t) = [L(t-d) R(t) - R(t-d) L(t)]
         / [|L(t-d) R(t)| + |R(t-d) L(t)| + eps]
```

Positive evidence means the left antenna led the right antenna at that delay. Negative evidence
means the right antenna led the left antenna. Multiple preregistered delays are combined with a
decaying weight and a signal-strength gate.

The estimator is causal by construction. It cannot read future samples. It does not read source
coordinates, distance to source, plume particles, culprit identity, or ground-truth odor-flow
direction.

The output is a SHA-256-bound JSON sidecar containing, per frame:

- left and right antenna response;
- signed odor-motion evidence;
- confidence;
- categorical direction;
- dominant tested delay;
- all per-delay evidence values.

This is an engineered temporal comparator. It is **not** a measured neural signal and does not
assert that a particular MaleCNS population implements the same equation.

## Frozen Phase A controls

Before interpreting real plume traces, the implementation must pass four invariants:

1. left-leading-right synthetic input gives positive evidence;
2. mirroring the antenna histories reverses evidence sign without changing confidence;
3. simultaneous bilateral pulses give approximately zero direction evidence;
4. extending a sequence with future samples cannot change any estimate in the existing prefix.

The sidecar must also preserve the exact source recording SHA-256 and reject tampered recordings.

## Phase B: visualization

The first public visualization should add a compact timing strip beneath the existing sensory HUD:

```text
LEFT ANTENNA   ▁▂▅█▃▁▁▄██▆▂
RIGHT ANTENNA  ▁▁▂▅█▃▁▁▄██▆
ODOR MOTION                  LEFT -> RIGHT
CONFIDENCE                         0.71
```

The display label is `MODELED BILATERAL ODOR TIMING`. It must never be labelled measured neural
activity. The plume remains audience-only.

## Phase C: information ablations

Before any temporal feature becomes a controller input, freeze matched replay diagnostics that
separate information from performance:

- intact bilateral timing;
- left/right mirror;
- zero temporal evidence while preserving instantaneous antenna samples;
- deterministic within-episode temporal misalignment that preserves each antenna's marginal
  signal distribution but destroys bilateral timing alignment.

The temporal-misalignment transform is analysis-only unless a separate causal online controller
contract is preregistered.

## Phase D: functional v2

Only after Phase A-C are reviewed should a new navigation protocol be frozen. It must have a new
protocol name, config hash, seed namespace, optimizer receipt, final-test namespace, and one-way
final lock. It must not reuse v1 final outcomes for model selection.

The primary functional question should be:

> Does adding causal bilateral odor timing improve held-out plume reacquisition and source finding,
> and does that benefit remain more dependent on intact MaleCNS-derived topology than on equally
> optimized degree-preserving rewires?

At minimum the matched cohort should contain:

- intact topology;
- the same number of independently seeded degree-preserving rewires as the frozen protocol;
- the prespecified steering-path lesion;
- a temporal-information ablation.

Every topology receives the same optimizer, episode budget, seed split, sensory adapter, and
parameter bounds.

## Metrics to add in v2

Navigation success and SPL remain useful, but temporal olfaction needs event-level measurements:

- time from odor loss to reacquisition;
- probability of reacquisition within a fixed horizon;
- crosswind casting amplitude after loss;
- heading change after a high-confidence timing event;
- source success on unseen plume intermittency/meander regimes;
- performance under bilateral timing ablation;
- intact-minus-rewire effect on each prespecified metric.

These should be frozen before the first real v2 optimization result.

## Public claim ladder

Phase A supports only:

> We can extract a causal, auditable bilateral timing signal from the same modeled antenna samples
> already present in the benchmark recording.

A future qualified v2 result may support a stronger topology statement only after matched training,
lesion/ablation controls, held-out evaluation, and the existing claim gates pass.
