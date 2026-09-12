# R002: MaleCNS looming-to-escape

## Public question

> **A virtual object is flying at the fruit fly. Does the real MaleCNS wiring preserve a
> collision-sensitive escape signal when matched scrambled wiring does not?**

R002 is the first non-olfactory experiment on the generic `fly-sniff` connectome runtime.
It deliberately tests a native Drosophila visuomotor pathway rather than mapping the
connectome onto an arbitrary game action.

The public causal comparison is simple:

```text
same looming stimulus + same modeled dynamics
                  |
          -------------------
          |                 |
     MaleCNS wiring     degree-preserving rewire
          |                 |
       DNp01/GF           DNp01/GF
     escape readout     escape readout
```

The result is an **open-loop modeled neural readout**. R002 does not claim that the virtual
fly sees an object, performs a biological jump, or turns away.

## What is biological and what is modeled

Structural authority comes from `male-cns:v1.0`. The bounded R002 graph contains the released
MaleCNS neurons and synapse counts for LPLC2, LC4 and DNp01/Giant-Fiber populations.

The following components remain explicit modeling assumptions:

- synthetic looming geometry;
- angular-size and positive-expansion sensory encoding;
- mapping those feature channels onto LPLC2 and LC4 populations;
- transmitter-to-fast-receptor sign convention;
- generic leaky rate dynamics;
- interpreting modeled DNp01/GF activity as an escape-circuit readout.

That separation is intentional. A connectome supplies anatomy, not measured membrane dynamics
or an end-to-end behavioral policy.

## Frozen MaleCNS authority

The committed candidate authority is `authority/r002_malecns_v1_candidate.json`.
Complete population membership comes from the public MaleCNS annotation Feather, not from
Cell Type Explorer display IDs. The Explorer HTML is pinned separately and used for type-level
structural summaries.

Resolved populations:

| population | bodies |
| --- | ---: |
| `LPLC2_L` | 94 |
| `LPLC2_R` | 91 |
| `LC4_L` | 71 |
| `LC4_R` | 55 |
| `DNp01_L` / GF | 1 |
| `DNp01_R` / GF | 1 |

The body-level weights table independently reproduces all four Explorer aggregate synapse
counts exactly:

| pathway | Explorer | body-level weights |
| --- | ---: | ---: |
| `LPLC2_L -> DNp01_L` | 2,642 | 2,642 |
| `LC4_L -> DNp01_L` | 3,782 | 3,782 |
| `LPLC2_R -> DNp01_R` | 2,220 | 2,220 |
| `LC4_R -> DNp01_R` | 2,580 | 2,580 |

The bounded induced graph contains **313 neurons and 20,607 edges**. This is not a whole-brain
simulation. It is a deliberately small, auditable pathway experiment.

## Sensory adapter

The synthetic adapter emits four named channels:

- `loom_size_left`
- `loom_size_right`
- `loom_velocity_left`
- `loom_velocity_right`

The size feature is normalized angular diameter. The velocity feature is normalized positive
angular expansion speed. Candidate mappings are:

```text
loom_size_left/right       -> LPLC2_L/R
loom_velocity_left/right   -> LC4_L/R
escape_left/right          <- DNp01_L/R (GF)
```

The feature-to-cell mapping is literature motivated but is not itself measured in the
MaleCNS connectome.

## Stimulus geometry

A spherical object of radius `r` approaches at constant axial velocity. Its angular diameter
is

```text
theta(t) = 2 atan(r / d(t))
```

Two paired conditions use the same seeded physical parameters:

1. `direct-hit`: zero lateral clearance;
2. `near-miss`: non-zero lateral clearance.

Object radius, initial distance, speed and near-miss offset receive small deterministic
seeded perturbations before each pair.

## Primary readouts

R002 does **not** score turn direction. For the stimulated side it records:

- peak target-side DNp01/GF activity;
- peak opposite-side DNp01/GF activity;
- direct-hit minus matched near-miss target activity;
- direct-hit target minus opposite activity, called lateralization;
- the fraction of paired trials where direct-hit target activity exceeds near-miss activity.

The topology control is an exact directed degree-preserving rewire. In/out degree is preserved,
roles remain on the same neuron identities, and edge attributes stay attached to their source
edge record while targets are swapped.

## Frozen qualification protocol

`R002-qualification-v1` is encoded in `src/fly_sniff/r002_qualify.py`. The thresholds were
frozen before inspecting the multi-rewire qualification result.

A candidate becomes `qualification_status=qualified` only when **all** gates pass:

1. all four cross-source structural synapse totals match exactly;
2. direct-hit target activity exceeds near-miss activity on at least 95% of paired trials;
3. mean direct-hit minus near-miss target activity is positive;
4. mean direct-hit lateralization is positive;
5. intact topology beats at least four of five fixed degree-preserving rewires on collision
   separation;
6. intact topology beats at least four of five fixed rewires on lateralization;
7. lesioning all modeled LPLC2/LC4 outputs reduces direct-hit DNp01/GF activity below 5% of
   intact.

Fixed qualification seeds:

```text
trial seed: 24017
rewire seeds: 24018, 24019, 24020, 24021, 24022
```

A scientific miss writes a complete `qualification.json` and remains a successful CI run.
Negative science is data; broken software is what should turn CI red.

## Reproduce the chain

Build the body-level graph from the public MaleCNS release:

```bash
fly-sniff-r002-graph \
  --authority authority/r002_malecns_v1_candidate.json \
  --download \
  --output artifacts/r002/graph
```

Apply the frozen qualifier:

```bash
fly-sniff-r002-qualify artifacts/r002/graph \
  --trials 24 \
  --seed 24017 \
  --output artifacts/r002/qualification.json \
  --qualified-circuit artifacts/r002/qualified-graph
```

A qualified graph can then be independently probed without candidate overrides:

```bash
fly-sniff-r002-probe artifacts/r002/qualified-graph \
  --trials 40 \
  --seed 77777 \
  --rewire-seed 77778 \
  --output artifacts/r002/qualified-probe.json
```

Render the causal comparison:

```bash
fly-sniff-r002-render artifacts/r002/qualified-graph \
  --rewire-seed 24018 \
  --output-dir artifacts/r002/render
```

The renderer emits a summary PNG, animated GIF, frame-level trace JSON and hashed render
receipt. Candidate rendering is possible only with `--allow-candidate` and remains visibly
labelled `CANDIDATE`.

## Claim boundary

If R002 qualifies, the strongest allowed statement is:

> Under the explicit R002 synthetic looming adapter and generic modeled rate dynamics, the
> bounded MaleCNS LPLC2/LC4-to-DNp01/GF topology produces a collision-sensitive, lateralized
> escape-circuit readout that survives the frozen qualification controls better than matched
> degree-preserving rewires.

Do not shorten that into “the connectome sees the object,” “the fly brain jumps,” or “we
simulated a conscious fly.”

## Public visualization

The first shareable render uses one shared expanding object and two neural panels:

```text
       SAME LOOMING INPUT
              |
    -----------------------
    |                     |
MALECNS WIRING       SCRAMBLED WIRING
DNp01 target         DNp01 target
DNp01 opposite       DNp01 opposite
```

The headline is:

> **Same looming input. Same modeled dynamics. Only topology changed.**

That makes the causal manipulation legible without pretending that the animation is a
recording of a living fly.

## References guiding the hypotheses

- Klapoetke et al., *Ultra-selective looming detection from radial motion opponency*.
- Work on LPLC2, LC4 and Giant-Fiber-mediated escape circuitry.
- Dombrovski et al., *Synaptic gradients transform object location to action*, Nature 2023.
- Lazar, Shukla & Zhou, *NeuroGraphBench: Interacting with Drosophila Connectomes at Scale
  for Exploring the Functional Logic of Neural Circuits*, bioRxiv 2026.

These references motivate the adapter and role hypotheses. They do not replace the frozen
MaleCNS v1.0 body-level authority used by this repository.
