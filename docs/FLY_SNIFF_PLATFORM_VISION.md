# Fly-Sniff Platform Vision

## Mission

Build an auditable experimental platform for asking how a whole Drosophila CNS transforms realistic sensory scenes into action.

The public `WHO FARTED?` clip is one deliberately funny experiment preset. The underlying product is a reusable scientific system:

```text
chemical / visual / mechanosensory world
                 |
                 v
        peripheral transduction
                 |
                 v
      MaleCNS connectome dynamics
                 |
                 v
       descending / VNC outputs
                 |
                 v
       physics-simulated fly body
                 |
                 +-------------------- feedback --------------------+
```

Every run must preserve the distinction between measured structure, modeled activity, simulator state, and behavioral output.

## Why MaleCNS is the right substrate

MaleCNS v1.0 spans the brain, optic lobes, and ventral nerve cord in one connectome. That gives the project a longer-term path beyond a brain-only descending-neuron interface: first use known descending neurons as a conservative high-level bridge to biomechanics, then progressively test whether MaleCNS VNC circuitry can replace pieces of the handcrafted motor bridge.

Do not attempt the full motor hierarchy in the first embodied release. The first embodied controller should use a narrow, auditable interface from qualified descending neurons into a standard locomotor controller.

## Four execution layers

### 1. World / embodiment

Use MuJoCo + NeuroMechFly/FlyGym for the production embodied lane.

The current 2-D simulator remains useful for fast mechanistic experiments and exact replay, but it is not the final public embodiment.

World state may contain privileged variables such as odor-source coordinates. Those variables must be excluded from the neural/controller observation contract.

### 2. Sensory encoding

A complex odor world is a vector field, not one scalar plume.

For odorants `k = 1..K`, the simulator produces concentration fields `c_k(x,y,z,t)`. The antennae sample only their local concentration vectors. A peripheral encoder maps those concentrations to receptor/ORN activity using a pinned response authority such as DoOR, with explicit concentration response, saturation, adaptation, noise and mixture-model sensitivities.

The CNS never receives odor identity labels or target-presence booleans in the identity benchmark.

### 3. Brain dynamics

Support two backends behind one interface:

- `rate`: current fast connectome-constrained model for qualification, lesions, rewires and large cohorts;
- `lif`: a whole-CNS spiking backend using the same sealed MaleCNS graph/sign authority for high-fidelity short probes and visual replays.

Both backends must emit the same event schema:

```text
time
body_id
modeled_state
spike/event (optional)
population/type
provenance
```

Never label either backend as measured firing.

### 4. Body control

Stage A uses qualified descending outputs as a low-dimensional interface into a pretrained locomotor body controller.

Stage B progressively replaces that bridge with MaleCNS VNC premotor/motor circuitry only when exact body-ID routes and signs are independently qualified.

## Complex scent benchmark

The first serious olfactory task should separate identity from localization.

### Identity question

> Is target odorant T present in this mixture?

The model receives receptor/ORN activity only.

Vary:

- target concentration;
- distractor identity;
- mixture ratio;
- number of distractors;
- background concentration;
- receptor noise/adaptation;
- target-present / target-absent balance.

Primary identity metrics:

- AUROC / AUPRC;
- false-positive rate at frozen threshold;
- detection latency;
- calibration;
- concentration invariance;
- held-out-mixture generalization.

### Localization question

> Given target identity evidence, can the embodied system locate the target source?

Multiple odor sources may coexist. The controller must localize the target-specific source rather than simply follow the strongest total odor concentration.

Primary localization metrics:

- success;
- SPL/path efficiency;
- time to target;
- wrong-source visits;
- target-loss recovery;
- OOD wind/source layouts.

## What 'training the connectome' is allowed to mean

Keep three modes distinct.

### Fixed-connectome inference

No learning. All weights are structural/model-derived. This is the cleanest test of whether wiring itself supports a computation.

### Biologically constrained plasticity

Learning is permitted only at preregistered plastic loci with a stated rule. For learned odor value, the mushroom-body KC->MBON system with dopaminergic modulation is the natural first target. Do not optimize the full connectome.

### Engineering readout training

A learned external decoder/readout may be useful as a benchmark, but it must be labeled `EXTERNAL LEARNED READOUT`, never as the fly brain learning.

Any intact-vs-rewire comparison must grant identical plasticity/readout capacity and optimization compute to every graph variant.

## Whole-connectome visualization

The production visualizer should be WebGL/Three.js, not a static plotting layout.

Use progressive level of detail:

1. faint whole-CNS context using aggressively simplified MaleCNS centerline geometry;
2. higher-resolution morphology for currently selected/qualified populations;
3. exact high-resolution skeletons for a small active route;
4. activity color/opacity driven only by actual modeled state from the replay.

A full-resolution rendering of 166k neuron meshes is neither necessary nor desirable for every frame.

### Production layout

For the scientific 16:9 view:

```text
+---------------------------+--------------------------+
| embodied fly / odor world | whole MaleCNS brain+VNC |
|                           |                          |
| target plume + distractors| active route highlighted|
| antenna samples           | actual modeled state    |
| body motion               | selected body IDs/types |
+---------------------------+--------------------------+
| shared causal timeline / interventions / readouts     |
+-------------------------------------------------------+
```

For the 4:5 social view, crop the same scene into a cinematic two-column composition rather than rebuilding a separate dashboard.

### Visual evidence semantics

- whole anatomy: neutral, low-opacity structural geometry;
- exact qualified route: saturated structural highlight;
- modeled subthreshold state: smooth emissive intensity;
- modeled spikes/events: brief localized flashes/pulses along the exact neuron's own morphology;
- lesion: geometry remains visible but desaturated/crossed;
- rewire: use a separate graph view, never morph biological skeletons into invented anatomical arbors.

Do not animate generic traveling pulses and call them neural activity.

## Software surface for outside users

A user should be able to define an experiment declaratively:

```yaml
world:
  body: flygym-v2
  odor_sources:
    - odorant: target
      source: [x, y, z]
    - odorant: distractor_a
      source: [x, y, z]
  wind: turbulent

sensory:
  olfaction: door-v2
  mixture_model: receptor_competition_sensitivity

brain:
  dataset: male-cns:v1.0
  backend: rate
  graph_threshold: 5

plasticity:
  mode: fixed

controls:
  rewires: [24018, 24019, 24020, 24021, 24022]
  lesions: [FC2_goal_lane, EPG_heading_lane, PFL3_output]

evaluation:
  train_seeds: [...]
  validation_seeds: [...]
  final_seeds: [...]
```

Target CLI:

```bash
fly-sniff run experiments/target-odor-mixtures.yaml
fly-sniff replay runs/<run-id>
fly-sniff compare runs/<intact> runs/<rewires...>
fly-sniff render runs/<run-id> --view scientific
```

Every run directory should contain:

```text
manifest.json
resolved_config.json
input_hashes.json
neural_events.parquet
body_state.parquet
sensory_state.parquet
metrics.json
video.mp4
```

## Qualification ladder from current state

1. E002e: PFL3 relative phase -> descending steering.
2. E003b: multi-seed, variable-wind sensory-goal benchmark.
3. E003c: receptor-level multi-odor identity encoder using pinned odor-response data.
4. E003d: target-identity evidence gates goal/navigation state.
5. E005: closed-loop embodied controller with fixed parameters.
6. E006: matched intact / rewire / lesion held-out cohort.
7. Public hero video only after E006.

## Public claim target

The strongest eventual headline should remain empirical:

> Same sensory world. Same body. Same compute. We kept the fly's measured wiring intact, then scrambled it while preserving basic graph statistics. Which controller finds the target odor source more reliably?

Do not claim consciousness, a complete physiological brain reproduction, or measured neural firing.
