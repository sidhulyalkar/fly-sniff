# MaleCNS replication and showcase ladder

The goal is not to collect disconnected viral demos. The goal is to build one auditable
MaleCNS execution stack that can reproduce several recognizable experiment archetypes,
then ask which behaviors genuinely depend on biological topology.

A connectome supplies structural connectivity. It does **not** supply a camera model,
membrane dynamics, receptor transfer functions, a Doom button mapping, reward semantics,
or the meaning of an arbitrary neural readout. Those interfaces must therefore be first-
class experiment artifacts rather than hidden glue.

## One core, many worlds

Every showcase should use the same separation:

```text
WORLD / RECORDED STIMULUS
          |
          v
  sensory adapter manifest
          |
          v
  named GraphBundle roles
          |
          v
  ConnectomeRuntime
          |
          v
  named neural readouts
          |
          v
   motor/output adapter
          |
          v
WORLD / VISUALIZATION
```

`ConnectomeRuntime` is intentionally modality-agnostic. It advances the same signed,
normalized rate model for any set of named input roles. The adapter owns the engineered
mapping between world variables and those roles.

## Replication ladder

### R000 - deterministic stimulus replay

**Analogue:** neural-activity movie demonstrations.

Feed a recorded JSONL stimulus through a frozen graph and save every requested readout.
This is the simplest way to make a Bad-Apple-style or camera-stimulus neural visualization
without a live game loop.

```bash
fly-sniff-replay path/to/graph stimulus.jsonl \
  --readout steer_left --readout steer_right \
  --output artifacts/replay.jsonl
```

The runner hashes the input stimulus and output trace and writes a receipt. Neural activity
under a movie stimulus is not evidence that the fly "recognizes" or "understands" the movie.

### R001 - native bilateral odor choice

**Analogue:** a tiny sensory-to-motor reflex experiment.

Already implemented as E002A. This is our lowest-cost causal sanity test because the same
controller interface is used by the plume benchmark.

```bash
fly-sniff-choice --controller malecns --graph path/to/qualified-graph --trials 100
```

### R002 - looming / escape arena

**Analogue:** visually obvious reflex demos.

Trace and qualify a MaleCNS looming pathway, expose `loom_left`, `loom_right`, and descending
escape/steering roles, then run an expanding-disc or incoming-object arena. Compare intact,
lesioned, and degree-preserving-rewired graphs on identical frozen stimulus sequences.

This is a particularly good public demo because the input, neural response, and output can
all be understood in a few seconds.

### R003 - open-loop connectome cinema

**Analogue:** movie-driven whole-connectome activity visualizations.

Create a reproducible retinal encoder whose output is bound to exact MaleCNS visual sensory
neurons. Replay short public-domain or generated clips. Render:

1. the stimulus;
2. anatomical neural activity;
3. selected population traces;
4. the exact sensory mapping and model label.

The first result is an activity visualization. A later scientific experiment can compare
intact versus rewired topology or test whether downstream readouts retain stimulus features.

### R004 - closed-loop fly arcade

**Analogue:** Doom, Mario, Minecraft, or other game-control demonstrations.

The world adapter must expose a small, explicit observation contract and a small, explicit
action contract. Start with a legally clean minimal arena before integrating a third-party
game engine.

Recommended progression:

```text
1-D target tracking
    -> 2-D corridor / light-dark arena
    -> obstacle + looming arena
    -> simple open-source arcade world
    -> Doom/Mario-style external adapter
```

Every control must be labeled either:

- biologically motivated descending-neuron readout;
- engineered fixed decoder;
- learned decoder.

Never describe an engineered button assignment as an established natural motor function.

### R005 - Who Farted? / turbulent plume hunt

**Analogue:** our native-behavior flagship rather than a game replica.

This remains the primary scientific result because odor and wind-guided navigation are
natural fly computations. Run intact MaleCNS topology, exact degree-preserving rewires, and
a transparent classical controller on identical frozen plume episodes. Render directly
from the resulting evaluation receipt.

### R006 - cross-world topology tournament

Once at least two adapters are qualified, run the same causal perturbations across worlds:

- intact biological topology;
- degree-preserving rewire;
- laterality scramble;
- role-specific lesions;
- matched artificial recurrent baseline where appropriate.

This is more interesting than asking whether a fly can merely move in a game. It asks:

> Which computations survive because of the biological wiring, and which are artifacts of
> the interface we designed around it?

## Run artifact contract

A showcase run should become a self-contained evidence directory:

```text
run/
  adapter_manifest.json
  stimulus.jsonl              # or a hash-addressed external source manifest
  neural_trace.jsonl
  receipt.json
  render.mp4                  # optional derivative, never primary evidence
```

The adapter manifest must record:

- MaleCNS dataset/version and GraphBundle digest;
- exact sensory roles and encoding rule;
- exact neural-dynamics parameters;
- exact output roles and decoding rule;
- world/stimulus version and seed;
- whether the graph is candidate or qualified;
- which claims the run is permitted to support.

## Claim ladder

### Level 0 - spectacle

A modeled connectome reacts to a stimulus or controls a world. This may be entertaining,
but it says little about biological computation by itself.

### Level 1 - structure-backed demonstration

The run uses pinned MaleCNS neurons/edges and reproducible interface manifests. Engineering
assumptions are explicit and the result can be replayed byte-for-byte.

### Level 2 - causal topology result

The intact graph beats or differs from matched topology controls under a frozen evaluation.
This is the level at which we can make an interesting claim about biological wiring.

### Level 3 - behaviorally anchored result

The modeled circuit additionally reproduces or predicts a measurable feature of real fly
behavior under comparable sensory conditions.

The public renderer should always show the attained level rather than encouraging the viewer
to infer a stronger one.

## What we should build next

The fastest high-value sequence is:

1. qualify the current odor/steering graph and earn R001;
2. land the generic replay runtime and produce a development-only R000 visualization;
3. trace a looming pathway and build R002;
4. create a retinal adapter for R003;
5. use the same adapter contract for a minimal closed-loop arcade arena before connecting a
   heavyweight game;
6. finish R005 and use R006 to turn several demos into one causal study of connectome
   topology.

This ordering gives us something visually new at every step while keeping the scientific
center of gravity on the experiments that can actually teach us something about the wiring.
