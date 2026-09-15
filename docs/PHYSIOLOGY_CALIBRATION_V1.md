# Physiology calibration v1

Program A asks whether useful computation is latent in biological wiring **without navigation training**. That question is only interpretable if missing dynamical parameters are constrained independently of source-finding performance.

This document defines the calibration firewall. It intentionally does not choose numerical PFN, PFL3, DNa02, odor-gating, or time-constant targets yet. Those values belong in a later evidence-review tranche after their source, preparation, uncertainty and transferability have been audited.

## Evidence before fitting

Every calibration target references exact records in an `EvidenceLedger`.

Two target strengths exist:

- `quantitative_fit`: requires at least one `MEASURED_PHYSIOLOGY` evidence record;
- `qualitative_sensitivity`: may use measured physiology or an explicitly labeled `CROSS_DATASET_PRIOR`.

Measured structure by itself cannot calibrate a neural timescale, gain, tuning curve or functional sign. Predicted annotations and modeled activity likewise cannot bootstrap themselves into calibration authority.

## Allowed parameter scope

The v1 contract permits only prespecified low-dimensional parameter classes:

- global dynamics;
- sensory gain;
- population gain;
- population timescale.

The contract deliberately has no individual-edge-weight parameter class.

## Forbidden freedoms

A valid Program A calibration protocol hard-fails if it enables any of:

- navigation reward;
- topology-specific fitting;
- whole-graph backpropagation;
- navigation-performance-based model selection;
- individual edge-weight fitting;
- body-membership fitting;
- role-assignment fitting.

The model-family set must also be declared in the protocol. Sensitivity across multiple prespecified families is acceptable. Selecting whichever family later navigates best is not.

## Candidate evidence families

The policy file names target families that merit review without presuming they survive review:

- PFN airflow tuning/laterality;
- PFL3 steering relationships;
- DNa02 lateralized turning relationships;
- odor-context gating;
- response timescales;
- independent intervention directionality.

For every retained target, the later evidence tranche must record the biological entity, preparation/dataset, measurement, model observable, objective, uncertainty policy, allowed parameter scopes, allowed wording and forbidden wording.

## Important biological caveats

### hDeltaC / hDeltaK

A hDeltaC-specific functional calibration target may not silently ignore the published hDeltaC/hDeltaK attribution confound. Odor-context gating remains a candidate sensitivity family until an unconfounded quantitative target is justified.

### Neurotransmitter prediction

Predicted transmitter identity is not measured receptor-specific postsynaptic sign. It may inform an explicitly labeled model assumption or sensitivity analysis, but cannot be promoted into measured physiology.

### Cross-dataset transfer

A measurement from another preparation or connectome remains a cross-dataset prior unless exact equivalence has been independently established. Its uncertainty and transfer assumptions must stay visible in the evidence record.

## Selection firewall

Before any navigation performance is inspected, an external reviewer should be able to reconstruct why every target, model family, parameter scope, uncertainty range and acceptance rule exists.

The critical question is:

> Could this choice have been made because it improved odor-source navigation?

For a valid Program A calibration contract, the answer must be **no**.

## What calibration can establish

A successful calibration can support a statement such as:

> This frozen model family was constrained to reproduce specified independent physiological observations within the prespecified uncertainty policy.

It cannot establish source-finding ability, a topology advantage, a complete biological circuit, or uniquely correct neural dynamics. Those are separate experiments downstream of the frozen calibration artifact.
