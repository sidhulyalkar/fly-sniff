# Program A evaluation contract v1

This contract freezes **what is evaluated** before Program A performance exists. It intentionally does not choose the final physical ranges while the sensory substrate is still under review.

## Separation of concerns

The contract is downstream of two independently reviewed sensory artifacts:

- `plume_contract_sha256`
- `bilateral_sensor_geometry_sha256`

Until those artifacts exist with scientifically reviewed PASS status, a real Program A evaluation contract remains blocked even though this schema can be tested synthetically.

## Condition families

Every final condition belongs to a prespecified family labeled `id` or `ood`. Each family freezes:

- scientific rationale;
- source-position rule;
- start-position/heading rule;
- wind rule;
- plume rule;
- sensor-sampling rule;
- episode rule;
- success rule;
- a dedicated hidden-final-seed namespace.

Program A v1 requires at least one ID and one OOD family. Families cannot share a hidden-seed namespace and cannot use development namespaces.

OOD families are scientific perturbations, not adversarial search results. They may not be added or selected because intact MaleCNS happened to outperform a null there.

## Condition identities

The contract does not contain materialized final seeds. Those arise only from the separate commit-reveal final-entropy protocol after the experiment is frozen.

After reveal, a condition ID is deterministically derived from:

- evaluation-contract SHA-256;
- frozen family ID;
- revealed final seed;
- replicate index.

The same exact ordered condition IDs must be consumed by intact wiring, every topology null realization, and every acute intervention. The topology-inference layer treats these as paired nuisance conditions nested inside graph realization.

## Controller information boundary

Source coordinates and other evaluator metadata may exist to compute outcomes, but `source_coordinates_visible_to_controller` is hard-frozen `false`.

The evaluation contract must never become a backdoor observation API.

## Acute interventions

Program A interventions are acute tests of a frozen model. Every intervention has a prespecified transformation, rationale and affected entities.

The validator rejects interventions that permit:

- parameter refitting;
- topology reselection;
- role reassignment.

The top-level contract also forbids retraining after intervention. A lesion followed by retraining is a different adaptive experiment and belongs outside this Program A confirmatory contract.

## Frozen anti-selection rules

A valid contract requires all of the following to remain false:

- materializing final seeds before entropy reveal;
- topology-specific condition selection;
- outcome-driven OOD selection;
- source-coordinate controller visibility;
- acute-intervention retraining.

Changing any source/start/wind/plume/sensor/episode/success rule changes the contract hash. Changing an intervention changes the contract hash.

## What remains blocked today

This schema does not certify a real environment. The final physical values and ranges should be frozen only after the experimental-plume lane has produced a usable plume contract and bilateral sensor-geometry artifact.

For Program A v1, prefer the smallest end-to-end physically qualified sensory substrate. Bilateral odor-motion timing is not required if the first experiment uses nondirectional odor context plus independently calibrated PFN airflow direction.
