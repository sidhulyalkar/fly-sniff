# Bilateral odor-motion v2

## Why this is a separate experiment

The sealed navigation v1 contract intentionally uses mean bilateral odor only as a nondirectional
context signal. Direction comes from the body-frame airflow/PFN-basis interface. The v1 contract
forbids adding new directional odor information after performance is observed.

Temporal bilateral odor information is therefore **not** patched into v1. It is versioned as a
separate experiment so v1 remains an untouched comparator.

## Current implementation status

The repository now contains three analysis-only layers that run downstream of a canonical hashed
recording:

1. causal bilateral timing estimation;
2. hysteretic odor-event and behavioral-response diagnostics;
3. temporal counterfactual controls and receipt-bound visualization.

None of these layers modifies the sealed v1 controller.

## Phase A: causal sensory sidecar — implemented

`fly-sniff-odor-motion` reads a hash-verified canonical recording and derives a visualization-ready
causal timing estimate from the already recorded modeled antenna responses.

For each fixed delay `d`:

```text
E_d(t) = [L(t-d) R(t) - R(t-d) L(t)]
         / [|L(t-d) R(t)| + |R(t-d) L(t)| + eps]
```

Positive evidence means the left antenna led the right antenna at that delay. Negative evidence
means the right antenna led the left antenna. The frozen lags are 50, 100, 150, and 200 ms. They are
combined using delay decay and a modeled-signal gate.

The estimator is causal by construction. It cannot read future samples. It does not read source
coordinates, distance to source, plume particles, culprit identity, or ground-truth odor-flow
direction.

The SHA-256-bound output contains, per frame:

- left and right antenna response;
- signed odor-motion evidence;
- confidence;
- categorical direction;
- dominant tested delay;
- all per-delay evidence values.

This is an engineered temporal comparator. It is **not** a measured neural signal and does not
assert that a particular MaleCNS population implements the same equation.

### Frozen causal controls

The implementation tests that:

1. left-leading-right synthetic input gives positive evidence;
2. mirroring the antenna histories reverses evidence sign without changing confidence;
3. simultaneous bilateral pulses give approximately zero direction evidence;
4. extending a sequence with future samples cannot change an existing prefix;
5. tampering with either recording or analysis receipt is rejected.

## Phase B: event diagnostics — implemented

`fly-sniff-odor-events` consumes the canonical recording plus its odor-motion sidecar.

Odor presence is defined from `max(left_response, right_response)` with frozen hysteresis:

- odor-on threshold: `0.16`;
- odor-off threshold: `0.08`;
- minimum state duration: `0.10 s`.

The two-threshold rule prevents near-threshold noise from repeatedly manufacturing loss and
reacquisition events.

The event stream distinguishes:

- first encounter;
- odor loss;
- reacquisition;
- onset of a high-confidence bilateral timing event.

For each event, the report binds the exact recorded controller behavior and records:

- mean signed turn command over the frozen response horizon;
- mean and peak absolute turn command;
- wrapped heading change;
- for loss events, whether odor was reacquired within the frozen horizon and the reacquisition
  latency when present.

The report also summarizes per agent:

- encounter count;
- loss count;
- reacquisition count;
- fraction of losses reacquired within the horizon;
- median reacquisition latency;
- high-confidence timing-event count;
- mean absolute heading response to those timing events.

These are replay diagnostics, not evidence that the original controller used the timing signal.

## Phase C: information counterfactuals — implemented for replay diagnostics

The event report now includes four timing conditions:

### Baseline

The original causal timing sidecar.

### Left/right mirror

Antenna identities are swapped and the estimator is rerun causally. The signed evidence must reverse
while confidence remains symmetric. The report records the maximum sign-reversal error.

### Zero temporal evidence

Timing evidence and confidence are replaced with zero while the original antenna samples remain
unchanged. This is an analysis-only information ablation.

### Right-channel circular shift

The right antenna history is deterministically circularly shifted by the frozen offset while keeping
its exact multiset of response values. This destroys temporal alignment without changing that
channel's marginal response distribution.

This transform is deliberately labelled **analysis-only and noncausal** because circular wrapping
can pair an early sample with a later recorded value. It must never be presented as an online sensor
or fed into a controller without a separately preregistered causal transform.

## Visualization — implemented for the scientific diagnostic

`fly-sniff-plot-odor-motion` can now consume the event receipt using `--events`.

The figure shows:

```text
LEFT ANTENNA     continuous modeled response
RIGHT ANTENNA    continuous modeled response
ENCOUNTER        event marker
LOSS             event marker
REACQUISITION    event marker
ODOR MOTION      signed causal timing evidence
CONFIDENCE       causal estimator confidence
```

The footer displays shortened hashes for the recording, timing analysis, and event receipt. A
mismatched event report is rejected before rendering.

The display label remains `MODELED BILATERAL ODOR TIMING`. It must never be labelled measured neural
activity. The plume remains audience-only.

## Reproducible command chain

```bash
fly-sniff-record \
  --output artifacts/showcase/episode.json

fly-sniff-odor-motion \
  artifacts/showcase/episode.json \
  --output artifacts/showcase/odor-motion-v2.json

fly-sniff-odor-events \
  artifacts/showcase/episode.json \
  artifacts/showcase/odor-motion-v2.json \
  --output artifacts/showcase/odor-events-v2.json

fly-sniff-plot-odor-motion \
  artifacts/showcase/episode.json \
  artifacts/showcase/odor-motion-v2.json \
  --events artifacts/showcase/odor-events-v2.json \
  --output artifacts/showcase/odor-motion-v2.png
```

CI builds the wheel and exercises this chain outside the repository checkout so package-resource or
entrypoint mistakes cannot hide behind the source tree.

## Phase D: functional v2 — intentionally not started

Only after the replay diagnostics are reviewed should a new navigation protocol be frozen. It must
have a new protocol name, config hash, seed namespace, optimizer receipt, final-test namespace, and
one-way final lock. It must not reuse v1 final outcomes for model selection.

The primary functional question is:

> Does adding causal bilateral odor timing improve held-out plume reacquisition and source finding,
> and does that benefit remain more dependent on intact MaleCNS-derived topology than on equally
> optimized degree-preserving rewires?

At minimum the matched cohort must contain:

- intact topology;
- the same number of independently seeded degree-preserving rewires as the frozen protocol;
- the prespecified steering-path lesion;
- a temporal-information ablation.

Every topology must receive the same optimizer, episode budget, seed split, sensory adapter, and
parameter bounds.

## Metrics to freeze before functional v2 optimization

Navigation success and SPL remain useful, but temporal olfaction needs event-level outcomes:

- time from odor loss to reacquisition;
- probability of reacquisition within a fixed horizon;
- post-loss turning activity under a prespecified horizon;
- heading change after a high-confidence timing event;
- source success on unseen plume intermittency and meander regimes;
- performance under causal bilateral-timing ablation;
- intact-minus-rewire effect on each prespecified metric.

The current replay layer computes the first four quantities descriptively. The functional-v2
protocol must freeze their aggregation and statistical comparison before optimization begins.

## Public claim ladder

The current implementation supports only:

> We can extract a causal, auditable bilateral timing signal from the same modeled antenna samples
> already present in the benchmark recording, identify frozen loss/reacquisition events, and inspect
> temporal counterfactuals without modifying the sealed v1 controller.

A future qualified v2 result may support a stronger topology statement only after matched training,
lesion and timing-ablation controls, held-out evaluation, and the existing claim gates pass.
