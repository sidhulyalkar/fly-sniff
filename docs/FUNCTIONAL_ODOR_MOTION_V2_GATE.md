# Functional bilateral odor-motion v2: pre-performance gate

This document defines what must be true before bilateral odor timing may influence navigation.
It is deliberately **not** a runnable training protocol. The current `odor_motion_v2` lane remains
post-recording analysis only.

## Why there is a gate

The sealed v1 experiment answers a specific question with a fixed sensory contract: bilateral odor
is nondirectional context and direction comes from the PFN airflow basis. The v2 timing analysis
adds a new engineered sensory quantity. Allowing that quantity to affect steering changes the
experiment, even if the graph is unchanged.

A new result therefore requires a new protocol, code seal, seed namespace, matched optimization,
qualification, and final-test lock. It must not inherit v1 final outcomes as model-selection data.

## Gate 0: analysis integrity

Before functional work starts, the analysis lane must continue to satisfy all of the following:

- canonical recording SHA verifies;
- timing sidecar SHA verifies and is bound to that recording;
- event receipt SHA verifies and is bound to both recording and timing sidecar;
- timing-estimator config embedded in the sidecar exactly matches any counterfactual replay config;
- threshold-sensitivity receipt is bound to the same receipt chain;
- installed-wheel `record -> timing -> events -> sensitivity -> render` smoke test passes outside the
  repository checkout;
- sealed-v1 execution modules do not import v2 timing, event, sensitivity, or visualization modules.

A failure here is an implementation failure, not a reason to relax a scientific threshold.

## Gate 1: sensory robustness, not threshold shopping

Primary analysis settings remain:

- causal lags: 50, 100, 150, and 200 ms;
- odor-on threshold: 0.16;
- odor-off threshold: 0.08;
- minimum odor-state dwell: 0.10 s;
- high-confidence timing threshold: 0.35;
- reacquisition horizon: 2.0 s.

The frozen secondary sensitivity grid may quantify fragility around these engineering choices, but it
may not choose a replacement threshold based on event count, navigation performance, visual appeal,
or a stronger intact-vs-rewire effect.

If the primary timing signal is rare or event conclusions are unstable over nearby thresholds, that
is evidence against escalating the present adapter. A more biologically or statistically motivated
adapter would require a new analysis protocol rather than post-hoc retuning.

## Gate 2: no connectome bypass

The first functional implementation must not add a direct term such as

```text
turn += timing_gain * odor_motion_evidence
```

outside the connectome dynamics and then attribute improved navigation to biological topology. Such a
path would allow the engineered adapter to solve part of the task independently of the graph.

Similarly, the current FB5AB roles must not silently become left/right odor-direction channels. v1
uses them as a tied nondirectional odor-context abstraction, and the present literature audit does not
establish that the selected MaleCNS FB5AB bodies encode the exact bilateral temporal comparator used
here.

Before timing can be injected into graph dynamics, one of two claim paths must be chosen and frozen:

### Path A: body-ID-qualified sensory recipient

Independently identify and review an anatomically plausible MaleCNS recipient population for the new
bilateral temporal drive. Freeze exact body IDs, laterality authority, normalization, sign semantics,
and literature/anatomy uncertainty before navigation optimization.

This path can support a future topology-dependent circuit claim if the rest of the matched-control
experiment succeeds.

### Path B: explicit engineering adapter

Treat temporal timing as a task-level modeled sensory adapter without claiming a specific biological
recipient mechanism. The adapter may enter the graph only through an explicitly declared, symmetric,
non-bypass interface shared identically by intact and every control topology.

This path permits a narrower statement about a MaleCNS-derived dynamical model under an engineered
sensory interface. It must not be described as discovering the fly's temporal olfactory circuit.

If neither path can be made auditable, functional v2 does not start.

## Gate 3: fresh experimental namespace

Functional v2 must create new immutable artifacts rather than mutating v1:

- `task-optimized-connectome-dynamics-v2` protocol;
- new config hash;
- new code ref;
- new development seed namespace and split seed;
- new optimizer receipt;
- new trained-qualification protocol;
- new final manifest;
- new one-way final-consumption lock.

The exact v1 candidate graph may be reused only by binding its immutable candidate-manifest identity
as an input to the new v2 seal. Reusing the graph does not permit reusing v1 final outcomes for v2
selection.

## Gate 4: matched cohort

At minimum the new experiment must optimize separately under identical budgets:

1. intact sealed topology + temporal adapter;
2. eight independently seeded degree-preserving rewires + the identical temporal adapter;
3. prespecified steering-input lesion + the identical temporal adapter;
4. intact topology + temporal-information ablation.

A stronger design also includes the same temporal-information ablation for every rewire, allowing the
analysis to ask whether timing benefit and topology benefit interact rather than conflating them.

All cohorts must share:

- train/validation seed split;
- optimizer algorithm and seed policy;
- population size and generation count;
- episodes per candidate;
- parameter bounds;
- sensory normalization;
- environment/plume distribution;
- stopping rules;
- compute budget;
- qualification thresholds.

No control may remain untrained while the intact graph is optimized.

## Gate 5: frozen endpoints before optimization

### Primary behavioral endpoints

- source success rate;
- SPL/path efficiency;
- loss-to-reacquisition probability within the frozen horizon;
- loss-to-reacquisition latency, with censoring/failure handling specified before evaluation.

### Prespecified mechanistic/behavioral diagnostics

- heading change following a high-confidence timing event;
- post-loss casting/turn activity;
- timing-ablation effect on reacquisition;
- intact-minus-rewire paired effect;
- interaction between topology status and timing availability, if the factorial control is used.

### OOD conditions

Freeze at least one plume intermittency shift and one meander/turbulence shift before optimization.
OOD parameters may not be selected after observing which condition produces the largest intact
advantage.

## Gate 6: statistical comparison

Episodes must be paired by environment seed wherever the comparison permits. Report effect sizes and
uncertainty, not only pass/fail counts.

The final analysis must distinguish:

- benefit of temporal information itself;
- benefit of intact topology itself;
- any interaction between timing information and intact topology.

A useful temporal adapter with no intact-vs-rewire advantage is not evidence that biological wiring
implements the computation. Conversely, an intact-topology advantage without a timing-ablation
benefit is not evidence that temporal odor timing caused the advantage.

## Stop conditions

Do **not** escalate to functional v2 merely because the social visualization is compelling.
Functional work should stop or return to sensory-model review if any of the following occurs:

- the causal timing signal is essentially absent across representative development episodes;
- apparent timing events are dominated by the offline misalignment control;
- event conclusions change qualitatively across the frozen nearby-threshold sensitivity grid;
- the proposed graph injection creates a direct steering bypass;
- biological recipient-role assignment would need to be chosen from navigation performance;
- a new adapter requires changing the sealed v1 claim after final outcomes are known.

A negative result at this gate is useful. It says the current temporal comparator is not yet a strong
enough sensory hypothesis to justify a much more expensive connectome experiment.

## Unlock condition

Functional implementation can begin only after the chosen Path A or Path B interface, seed namespace,
matched cohort, trainable parameters, endpoints, OOD distributions, qualification gates, and final
lock are written into a new immutable v2 protocol **before** the first v2 navigation optimization is
run.
