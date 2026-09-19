# FlyARC v1 — MaleCNS recurrent topology on ARC-AGI-3

## Question

> Does fixed biological MaleCNS recurrent topology provide a more useful internal state for
> ARC-AGI-3 interaction than matched topology controls?

This is deliberately **not** a claim that a fruit fly understands ARC. ARC observations and
actions are engineered interfaces, the neural state is modeled, and the trainable policy head
is an artificial linear TD learner.

The value of the experiment is the controlled comparison.

## Why ARC-AGI-3

ARC-AGI-3 is interactive rather than static. An agent sees a sequence of categorical grid
frames, acts, sees the consequence, and must retain useful state while discovering a new
world. That makes it a substantially better probe of recurrent memory and state representation
than a one-shot classifier.

The v1 integration uses the official `arc-agi` Python toolkit. It starts with `ls20` and
refuses games exposing complex coordinate actions because a click decoder is a separate
experimental choice that should be frozen independently.

## Frozen v1 interface

```text
ARC frame
  ↓
8×8 spatial bins
  ↓
16-way categorical histogram per bin
+ fraction changed since previous frame
  ↓
fixed seeded sparse projection
  ↓
4096-neuron structurally selected MaleCNS reservoir
  ↓
linear Q readout
  ↓
simple ARC action
```

The encoder never treats ARC color IDs as an ordinal intensity and contains no learned visual
front end.

The default 4096-neuron core is selected *before* ARC is inspected using only connectome
structure:

```text
score(neuron) = sqrt(in_log_synapse_strength × out_log_synapse_strength)
```

Ties are resolved by `bodyId`. This is intentionally simple and preregistered. Searching many
subgraphs and publishing the best ARC performer would turn the experiment into architecture
search.

## Controls

Every default comparison includes:

1. **intact** — selected biological topology;
2. **rewire** — target-swap null preserving exact directed in/out degree counts;
3. **random** — same nodes, edge count, weight multiset and sign multiset with random topology;
4. **stateless** — same state dimension and input projection, recurrence removed.

All four receive the same ARC seed, frame encoder, sparse input projection seed, policy
initialization, policy size, optimization hyperparameters, and action budget.

The closed-loop trajectories will naturally diverge after the policies choose different
actions. That is why v2 should also add an open-loop frozen-trajectory probe for memory and
state decodability.

## Development reward

The tiny policy head receives no game-specific semantic hints.

```text
-0.001    each action
+0.01     first time a frame hash is seen
+1.0      each newly completed level
+2.0      win
-1.0      game over
```

This reward is engineering glue and is always labeled as such.

## Install

The ARC toolkit currently requires Python 3.12 for the intended local/competition workflow.

```bash
git switch feat/flyarc-v1
python3.12 -m venv .venv-flyarc
source .venv-flyarc/bin/activate
pip install -e '.[arc,dev]'
```

An `ARC_API_KEY` is optional in the official toolkit, but a registered key gives access to
more environments.

## First run

Use a signed GraphBundle. A candidate graph is development-only:

```bash
fly-sniff-arc /path/to/graph \
  --game ls20 \
  --steps 500 \
  --max-nodes 4096 \
  --allow-candidate \
  --output artifacts/flyarc-ls20-v1
```

For an already qualified graph, omit `--allow-candidate`.

A faster smoke test is:

```bash
fly-sniff-arc /path/to/graph \
  --game ls20 \
  --steps 40 \
  --max-nodes 512 \
  --allow-candidate \
  --variant intact \
  --variant rewire \
  --output artifacts/flyarc-smoke
```

## Artifacts

```text
flyarc-ls20-v1/
  manifest.json
  comparison.json
  receipt.json
  intact/
    steps.jsonl
    states.npz
    frames.npz
    metrics.json
  rewire/
    ...
  random/
    ...
  stateless/
    ...
```

`states.npz` contains the full float32 modeled reservoir state and selected MaleCNS body IDs
for every executed action. `frames.npz` contains the actual categorical ARC observations,
including reset observations, in a padded uint8 tensor with original shapes. `steps.jsonl`
binds each action to the exact post-action observation index and contains the action, reward,
frame hash, game state, Q values, activity summaries, and the most active body IDs.

The public renderer is strictly downstream of those hashed outputs:

```bash
fly-sniff-arc-render artifacts/flyarc-ls20-v1 \
  --output artifacts/flyarc-ls20-v1.gif \
  --fps 8
```

For MP4, install `ffmpeg` and use an `.mp4` output path. The renderer first validates every
file named in `receipt.json`; a missing or modified scientific artifact aborts rendering.
The reservoir panel is explicitly labelled as a structural-rank layout rather than anatomy.

## Interpretation

Interesting first-light outcomes include:

- intact learns faster than a distribution of rewires;
- intact and rewires behave similarly but both beat stateless;
- all recurrent controls are equivalent;
- no condition solves the game, but their state representations differ on later frozen probes.

Only the first case even suggests a biological-topology advantage, and it still requires
replication over independently generated rewires, ARC seeds, and games.

## Next tranche

After first light:

1. add a deterministic frozen-trajectory replay probe;
2. fit linear probes for previous action, frame history, next-frame change, and level progress;
3. increase topology-null count;
4. freeze a multi-seed / multi-game evaluation manifest;
5. export a browser replay showing ARC world, MaleCNS activity, state embedding, and controls;
6. only then evaluate whether biological topology carries a reproducible advantage.
