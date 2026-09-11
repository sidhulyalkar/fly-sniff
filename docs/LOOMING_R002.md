# R002 — looming / escape

## Public question

> **A virtual object is flying at the fruit fly. Can the connectome turn away before impact?**

R002 is the first non-olfactory showcase built on the generic connectome runtime. It is
intended to reproduce a recognizable class of Drosophila looming/escape experiments while
keeping the world-to-neuron interface explicit and auditable.

## Why looming

Looming is a strong showcase target because the stimulus, neural computation and behavior
are all visually legible. Drosophila work has identified looming-sensitive visual projection
neurons including LPLC2 and LC4, with escape pathways involving the Giant Fiber and other
descending neurons. LPLC2 is associated with angular-size / radial-expansion evidence, while
LC4 contributes strong expansion-velocity evidence.

A recent 2026 NeuroGraphBench preprint independently demonstrates an executable
LPLC2→Giant-Fiber collision-detection abstraction and compares direct-hit and near-miss
trajectories. `fly-sniff` does **not** copy that model or its code. We use the same broad
scientific question as a replication target while retaining our own graph runtime, evidence
receipts and topology controls.

## Adapter contract

The current synthetic adapter emits four explicit modeled channels:

- `loom_size_left`
- `loom_size_right`
- `loom_velocity_left`
- `loom_velocity_right`

The size channel is normalized angular diameter. The velocity channel is normalized positive
angular expansion speed. These names describe modeled sensory features, **not yet qualified
MaleCNS cell identities**.

Candidate biological mappings to LPLC2-, LC4-, Giant-Fiber- and other descending-neuron
populations must be traced and frozen from MaleCNS before a run is allowed to claim that the
actual MaleCNS looming pathway was used.

Output roles are:

- `steer_left`
- `steer_right`
- optional `escape`

As with the odor benchmark, output semantics must be attached to reviewed body IDs rather
than inferred from a convenient sign convention.

## Geometry

A spherical object of radius `r` approaches at constant axial velocity. The projected angular
diameter is

```text
theta(t) = 2 atan(r / d(t))
```

where `d(t)` is Euclidean distance from the fly. Two matched trajectory classes are used:

1. **direct hit** — zero lateral offset;
2. **near miss** — fixed non-zero clearance.

Before evaluation, object radius, initial distance, speed and miss offset receive small
seeded perturbations. Direct-hit and near-miss trials with the same seed share those physical
parameters so the comparison is paired.

## Metrics

R002 reports:

- direct-hit turn-away accuracy;
- mean signed turn-away margin;
- optional `escape` readout difference between direct hits and matched near misses;
- intact minus degree-preserving-rewire accuracy;
- intact minus degree-preserving-rewire turn margin.

Chance directional accuracy is 50%.

The topology comparison uses the same directed degree-preserving rewire implementation as the
odor benchmark. The same sensory frames are replayed through intact and rewired graphs.

## Run it

Emit a deterministic stimulus without any connectome graph:

```bash
fly-sniff-loom \
  --emit-stimulus artifacts/loom-left-direct.jsonl \
  --side left \
  --trajectory direct-hit
```

Run a candidate graph during circuit development:

```bash
fly-sniff-loom \
  --circuit artifacts/r002-candidate \
  --allow-candidate \
  --trials 40 \
  --output artifacts/r002-development.json
```

Candidate output is development-only. A public MaleCNS claim requires a graph whose manifest
has `qualification_status: qualified` and whose sensory/readout roles are backed by reviewed
MaleCNS authority.

## Qualification ladder

### R002.0 — environment

PASS when direct-hit and near-miss geometry, seeded perturbations and replay output are
deterministic and tested.

### R002.1 — structural authority

Resolve exact MaleCNS v1.0 body IDs for the selected looming-sensitive visual populations and
descending readouts. Preserve evidence source, dataset version and hashes.

### R002.2 — circuit sanity

Show that lateralized looming input produces reproducible lateralized modeled responses and
that relevant lesions alter those responses in the expected direction.

### R002.3 — causal topology

Freeze the trial set, dynamics and rewire seed. Compare intact MaleCNS topology against the
matched degree-preserving rewire.

### R002.4 — behavioral anchor

Only after the structural and topology results are stable should the model be compared with
published fly response timing/direction or a matched behavioral dataset.

## Public visualization target

The social render should be understandable without a caption:

```text
                INCOMING OBJECT
                      ↓
       ┌────────────────────────────┐
       │ REAL WIRING | SCRAMBLED    │
       │       🪰     |     🪰       │
       │        ↘     |      ?       │
       └────────────────────────────┘

        DIRECT HIT / NEAR MISS
        neural bars + turn trace
```

A black expanding disc is enough for the first version. The audience can see the object. The
controller receives only the declared adapter channels. If a development graph is used, the
render must say **DEVELOPMENT / NOT A MALECNS RESULT**.

## References guiding the design

- Klapoetke et al., *Ultra-selective looming detection from radial motion opponency*.
- Ache et al. / related work on LPLC2, LC4 and Giant-Fiber looming escape.
- Dombrovski et al., *Synaptic gradients transform object location to action*, Nature 2023.
- Lazar, Shukla & Zhou, *NeuroGraphBench: Interacting with Drosophila Connectomes at Scale
  for Exploring the Functional Logic of Neural Circuits*, bioRxiv 2026.

These references motivate hypotheses and adapter features. They do not substitute for MaleCNS
v1.0 body-level authority in this repository.
