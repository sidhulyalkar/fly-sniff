# Sealed Graph Experiment v1

This lane begins **after E001 structural discovery** and after a candidate GraphBundle has been chosen without using navigation performance.

Its purpose is to make the next scientific claim auditable:

> Does preserving this fixed MaleCNS-derived topology help odor-navigation performance relative to matched degree-preserving rewires and a prespecified steering-input lesion when every topology receives the same optimization budget?

## Branch

Run this lane from:

```bash
git fetch origin
git switch feat/sealed-graph-experiment-v1
git pull --ff-only origin feat/sealed-graph-experiment-v1
```

Do not continue on a dirty checkout.

## What is frozen before performance

The candidate seal binds all of the following before any task-optimization episode is allowed:

- exact GraphBundle `nodes.parquet`, `edges.parquet`, `roles.json`, and existing `manifest.json` hashes;
- exact sorted body-ID set and canonical `(source, target, weight, sign)` edge identity;
- exact E001 role-review populations;
- exact required E001 staged-route and handoff evidence hashes;
- the candidate inclusion policy;
- source-body transmitter/sign authority and signed-edge coverage;
- the task-optimization config;
- the sensory normalization rule;
- the code commit.

Any mismatch causes the sealed experiment command to stop. It does not repair the graph.

## Unique-edge accounting

A MaleCNS edge may appear in more than one staged E001 corridor. Evidence reporting now treats `(source, target)` as the biological edge identity. Repeated stage appearances are collapsed to one edge while `stages_seen` preserves route provenance. If two stage artifacts report different structural weights for the same pair, the audit fails.

This change affects evidence bookkeeping only. It does not alter the original E001 stage or handoff pass/fail result.

## Transmitter and sign authority

Run:

```bash
fly-sniff-sign-authority /path/to/sealed-graphbundle \
  --authority authority/malecns-v1.0-steering-sign-evidence.json \
  --authority authority/malecns-v1.0-body-sign-overrides-v1.json \
  --output results/candidate/sign-authority-v1
```

Every unique presynaptic body ID receives a record containing:

```text
source_body_id
cell type
transmitter
confidence
evidence level
authority file + hash
source reference
model sign
sign status
```

Resolution order is exact body-ID authority, then type-level authority, then unknown. Unknown remains:

```text
transmitter = null
confidence = null
sign = 0
```

A nonzero graph sign without authority is a hard mismatch. Type-level transmitter prediction is explicitly a modeling evidence source, not measured receptor-specific synaptic physiology.

## Exact graph inclusion rule

`configs/candidate_graph_v1.json` freezes the membership rule.

The **exact supplied pre-performance GraphBundle is authoritative membership**. The seal does not automatically union every neuron seen during staged search. It verifies instead that:

1. every sealed node occurs in at least one of the five required primary E001 stages;
2. every sealed edge occurs in the unique union of those five stages;
3. structural weights are unchanged;
4. source-target edges are unique;
5. six model-role populations exactly match the hash-bound E001 role review;
6. optional hDeltaK/PFG memory-stage evidence cannot add nodes or edges;
7. the complete node set, intermediate-node set, file hashes, and canonical edge hash are frozen in the candidate manifest.

No intermediate type can be added, whitelisted, or pruned after seeing navigation performance.

## PFN sensory normalization

The task model uses `role_total_l1` for:

```text
odor_context_left
odor_context_right
wind_basis_left
wind_basis_right
```

For a role with `N` exact sealed body IDs, a role-level modeled sensory value `x` assigns `x/N` to each present body ID. Thus total absolute direct drive is `|x|`, independent of `N`.

This specifically prevents the current 120-left versus 127-right PFN population sizes from generating a directional advantage merely because one side contains more retained cells. Membership itself is unchanged.

## Mac development run

Use the same Python 3.11 environment prepared for E001. Point `BUNDLE` at the already sealed GraphBundle and `E001_RUN` at the exact passing Mac E001 artifact directory:

```bash
BUNDLE=/absolute/path/to/graphbundle \
E001_RUN=/absolute/path/to/passing-e001-run \
bash scripts/run_sealed_experiment_mac.sh
```

The script performs, in order:

1. rebuilds the E001 evidence summary using unique-edge accounting, without rerunning structural discovery;
2. audits exact source-body transmitter/sign authority;
3. creates and immediately verifies the immutable candidate manifest;
4. runs matched optimization for intact, exactly eight frozen degree-preserving rewires, and the steering-input lesion;
5. reconstructs actual optimizer budget receipts;
6. deterministically replays CEM mechanics;
7. regenerates and audits the frozen rewire ensemble;
8. runs diagnostic-only shortcut attacks on the trained intact controller;
9. runs trained E002;
10. freezes the final held-out/OOD manifest;
11. writes `READY_FOR_FINAL.md` and stops.

The development invocation does **not** evaluate final held-out or OOD episodes.

## One-way final

After reviewing every development artifact and deciding to proceed, consume the already frozen run exactly once:

```bash
RUN_FINAL=1 \
FINAL_RUN_DIR=results/sealed-experiment/<frozen-run-id> \
BUNDLE=/absolute/path/to/graphbundle \
bash scripts/run_sealed_experiment_mac.sh
```

Final mode performs no structural discovery, graph editing, training, requalification, or seed regeneration. It verifies the existing candidate seal and then creates `final-run-consumed-v1.json` **before** evaluating final seeds.

If the process fails after that lock appears, v1 remains consumed. A rerun requires a new protocol version rather than a new output directory.

## What can stop the lane before training

A sealed GraphBundle is not automatically training-ready. Expected hard stops include:

- a nonzero edge sign with no matching source-body/type transmitter authority;
- signed-edge coverage below the frozen trained-E002 minimum;
- bundle roles that differ from the exact E001 role review;
- an edge or node outside the five required E001 primary stages;
- a structural weight mismatch;
- a task config that does not use `role_total_l1`;
- graph qualification status other than the existing required `qualified` state.

Do not repair these failures based on navigation performance. Resolve provenance or create a new protocol before training.

## Final interpretation

Whatever the final result is, the cinematic must replay it rather than select a favorable episode family.

If intact beats matched rewires under the frozen gold criteria, report that result with the exact confidence intervals and OOD behavior. If the nulls tie or beat intact, report that. If trained E002 or any pre-final audit fails, do not consume the final namespace merely to see what would have happened.

The visual labels should continue to distinguish:

```text
MEASURED ANATOMY      MaleCNS body IDs and structural edges
LITERATURE PRIOR      cross-dataset cell-type/function evidence
MODEL INPUT           imposed odor/airflow interface
MODEL STATE           simulated rate dynamics
BEHAVIOR              simulated fly trajectory
```

Do not label modeled state as connectome "firing".
