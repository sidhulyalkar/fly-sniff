# Evidence-gated neural-routing storyboard

The final public visualization should make the fly's control problem legible without
turning a structural connectome into fictional neural activity.

## The visual story

The audience should be able to understand the causal chain without reading a caption:

```text
ODOR PUFF
   |
   v
LEFT / RIGHT ANTENNA SIGNALS
   |
   +----------------------+ 
   |                      |
   |                 AIRFLOW / WIND
   |                      |
   v                      v
          ODOR-GATED NAVIGATION GOAL
                    |
                    v
            LEFT / RIGHT STEERING
                    |
                    v
               FLY MOVEMENT
```

For the qualified MaleCNS version, this simple behavioral story can expand into a
small neural-routing panel. The intended literature-informed hypothesis is:

```text
olfactory pathways ----------------------\
                                          > hDeltaC / FB goal circuitry
wind pathway -> PFN populations ---------/             |
                                                        v
                                                PFL3 / PFL2 outputs
                                                        |
                                                        v
                                              premotor / descending network
                                                        |
                                                        v
                                                   DNa02 + peers
```

This diagram is a **candidate storyboard**, not a predeclared MaleCNS wiring result.
Every edge shown in the final neural panel must first be resolved against the
MaleCNS v1.0 body-level graph.

## Evidence classes

Every visual edge or node belongs to exactly one class:

1. **MEASURED STRUCTURE**
   - exact MaleCNS v1.0 body IDs;
   - exact directed synaptic edges and weights;
   - cell-type annotations;
   - transmitter/sign provenance when available.

2. **MODELED STATE**
   - activity produced by our explicit rate/controller model on the sealed graph;
   - never presented as recorded physiology;
   - replayed from the evaluation recording, not recomputed by the renderer.

3. **BEHAVIORAL STATE**
   - modeled antenna signals;
   - modeled body-frame airflow;
   - controller turn/speed command;
   - position, heading, source-finding state.

The renderer should use visibly different labels for these classes. In particular,
`MODELED ACTIVITY` must never be visually conflated with `MEASURED CONNECTIVITY`.

## What current literature lets us say before E001

Existing Drosophila work supports several useful hypotheses for discovery:

- odor-sensitive fan-shaped-body inputs can converge with wind-sensitive PFN inputs
  onto hDeltaC neurons;
- hDeltaC activity can represent an odor-gated directional goal;
- PFL3 populations transform goal/heading relationships into left/right steering;
- PFL2 populations are associated with forward-speed control;
- central-complex output reaches descending steering systems, including pathways
  converging on DNa02.

These are search priors only. MaleCNS v1.0 decides which exact cells and edges are
eligible for this project's final circuit.

## MaleCNS promotion rule

A population may appear by biological name in the final public HUD only after:

- exact v1.0 body IDs are sealed;
- all displayed edges exist in the extracted graph;
- left/right roles are reviewed;
- unresolved transmitter/sign information stays visibly unresolved;
- sensory perturbation reaches the expected steering outputs;
- mirrored perturbations produce appropriately opposed steering;
- steering-output lesions collapse the modeled steering response;
- deterministic replay reproduces the same activity and action trace.

Before those gates pass, the public HUD must use generic labels such as
`ANTENNA SIGNAL`, `AIRFLOW`, `ODOR-GATED GOAL`, and `STEERING COMMAND`.

## Recommended 15-second social sequence

### 0.0-1.5 s: Hook

`WHO FARTED?`

Six suspects. One visible audience-only plume. No explanatory paragraph.

### 1.5-8.5 s: Search

One large room. The fly trajectory grows in real time. The lower HUD shows left and
right antenna values, sensed airflow, and the resulting turn command. When odor is
lost, the HUD visibly switches to `CROSSWIND CAST`; on odor contact it switches to
`ODOR-GATED UPWIND`.

### 8.5-11.5 s: Neural zoom, qualified version only

The behavioral HUD compresses and an evidence-gated circuit panel lights up. Synapse
lines represent measured MaleCNS structure. Population glow represents modeled
activity from the immutable episode recording.

### 11.5-13.5 s: Culprit reveal

Circle the true source suspect. Keep the actual path on screen so the reveal cannot
be mistaken for a hand-authored ending.

### 13.5-15.0 s: Causal comparison

Qualified release: `REAL WIRING` versus `DEGREE-PRESERVING SCRAMBLE` on the same
sealed plume episode, followed by the cohort statistic rather than a single hero run.

Development release: retain `DEVELOPMENT PROXY • NOT A MALECNS RESULT` and compare
against the transparent random or cast/surge control.

## Things we should deliberately not fake

- no 3-D room airflow unless we actually simulate obstacle-aware fluid dynamics;
- no glowing named neuron population until exact MaleCNS identities are sealed;
- no successful culprit ending when the controller failed;
- no single-run claim that substitutes for the preregistered cohort metrics;
- no source-distance or culprit signal fed into the controller;
- no hidden renderer dynamics that can diverge from the recorded scientific run.

This document is the presentation contract. The scientific qualification documents
remain authoritative for whether a MaleCNS result is eligible to replace the proxy.
