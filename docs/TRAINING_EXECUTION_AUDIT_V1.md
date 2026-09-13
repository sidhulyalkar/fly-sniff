# Training execution and simulator-shortcut audit v1

This document records two red-team controls that sit between development optimization and the one-way trained final benchmark.

Neither control increases model capacity, changes role membership, changes graph topology, adds controller inputs, or relaxes a scientific threshold.

## 1. Expected budget is not executed budget

`optimizer_budget_receipt()` defines the work that the frozen v1 protocol is supposed to perform. That is necessary, but by itself it is only a formula derived from configuration.

A future implementation could accidentally early-stop, skip a candidate, retry only one topology, evaluate a different seed batch, or change one full-pool evaluation while still writing the old expected receipt.

`fly-sniff-redteam-budget` therefore reconstructs represented optimizer work from the report itself:

- generation count must exactly equal the frozen CEM generation count;
- generation indices must be complete and ordered;
- every generation seed batch must have the frozen size;
- every generation seed must belong to the frozen training split;
- every seed-batch hash must recompute exactly;
- every generation must contain exactly the frozen population of candidate receipts;
- candidate indices must be complete and ordered;
- candidate parameter hashes must be valid SHA-256 values;
- candidate objectives must be finite;
- candidate evaluations and episodes are counted from those receipts rather than copied from configuration;
- baseline-train, baseline-validation, trained-train, and trained-validation episode counts are recomputed from their persisted summaries;
- the reconstructed totals must exactly equal the frozen expected budget;
- intact, all eight rewires, and the lesion must reconstruct to the same budget hash.

The trained-final manifest binds the SHA-256 of this reconstructed matched-cohort execution audit before opening final seeds.

This still cannot prove that a person or unrelated process performed no hidden experiments outside the persisted artifact. That remains an operational provenance limitation.

## 2. Wall reflection is a simulator cue

`FlySniffEnv` reflects an agent that crosses an arena boundary. Wall distance is not exposed through `Observation`, which is an important protection, but reflected motion can still influence future body-frame sensory state and therefore behavior.

A model that succeeds mostly after repeated reflections could look better at source navigation than its sensory computation warrants.

The shortcut battery therefore records wall-reflection telemetry without changing the controller API. Before each physical simulator step, the diagnostic reconstructs the commanded heading and displacement and asks whether that step would cross an x or y boundary before reflection is applied.

For every episode it records:

- total simulator steps;
- wall-contact steps;
- x-wall-contact steps;
- y-wall-contact steps;
- fraction of steps that invoke reflection;
- first wall-contact step.

For every diagnostic scenario it reports:

- fraction of episodes with any wall contact;
- mean wall-contact count and step fraction;
- success rate among episodes with no wall contact;
- success rate among episodes with wall contact;
- fraction of successful episodes that used at least one reflection.

These values are diagnostic only. They do not become controller inputs and they are not post-hoc pass/fail gates.

## 3. Sensory and geometry shortcut battery

The frozen trained parameters are evaluated with no retraining on a separate diagnostic seed namespace under:

1. normal sensory input;
2. odor clamped to zero;
3. airflow clamped to zero;
4. odor and airflow both clamped;
5. source shifted toward the low crosswind side;
6. source shifted toward the high crosswind side;
7. source/start geometry mirrored with world wind reversed.

The wall telemetry is reported for each of these conditions. This lets us distinguish, for example, a controller that loses performance when odor is removed from one that retains performance but begins relying heavily on boundary reflections.

No threshold is invented after observing these results. If a future protocol wants wall dependence to become a formal gate, that threshold must be preregistered under a new protocol version before its outcomes are opened.

## 4. One-way final boundary

`fly-sniff-freeze-trained-final` now requires a complete matched-training artifact whose persisted execution history reconstructs to the frozen budget. The final manifest hash-binds:

- reviewed qualified circuit identity;
- training configuration;
- matched training artifact;
- passing intact trained-E002 artifact;
- shared optimizer budget;
- reconstructed optimizer-execution audit;
- topology-specific trained parameter hashes;
- frozen final seed split.

`fly-sniff-trained-final` recomputes those bindings before running the final benchmark.

The historical untrained final protocol remains untouched. Trained task-optimization evidence uses a separate final protocol so an older sealed benchmark is not silently redefined.

## Claim boundary

Passing these software and simulator audits means that the represented computation is internally consistent with the frozen protocol. It does **not** establish that the selected MaleCNS roles are physiologically correct, that rate dynamics reproduce living-fly physiology, that the rewire Markov chain reached stationarity, or that no external human peeking occurred.

Those questions remain separate evidence lanes rather than assumptions hidden inside the demo.
