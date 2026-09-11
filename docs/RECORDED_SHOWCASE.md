# Recorded showcase contract

The social renderer must never be the scientific execution path.

The authoritative development flow is:

```text
seed + plume model + controller
        |
        v
  simulation episode
        |
        v
 hashed JSON recording
        |
        +--> tests / audit
        |
        v
  GIF / MP4 replay
```

## What each recorded frame contains

Each frame stores the exogenous plume snapshot plus, for every controller:

- position and heading;
- left and right modeled antenna signals;
- odor mean and bilateral difference;
- airflow vector in the fly's body frame;
- the exact turn and speed command generated from that observation;
- controller diagnostics;
- distance to source and success state.

The episode bundle is serialized canonically and SHA-256 hashed. The replay loader
rejects a recording if any field changes without a matching hash.

## Paired-plume rule

All paired controllers receive the same deterministic plume realization. If one
controller reaches the source early, that fly stops moving but its private copy
of the exogenous plume continues to advance. The recorder checks that paired
plume snapshots remain bit-identical at every stored step.

This prevents a subtle visualization error where one agent finishing could freeze
its plume while the comparison agent continued in a later plume state.

## Sensory routing shown to the audience

The development HUD is deliberately limited to quantities the simulator really
uses:

```text
left antenna ----\
                  +--> odor-gated search --> steering command
right antenna ---/           ^
                              |
                    body-frame airflow
```

The proxy uses odor to gate search behavior, sensed airflow to orient upwind or
crosswind, and a smaller bilateral odor-difference term to bias steering. It does
not receive source coordinates, distance to source, culprit identity, or the
viewer-visible plume image.

This panel is a modeled sensory-to-steering trace, not a claim that these values
are measured MaleCNS neural activities.

## Why the proxy now uses body-frame wind

Earlier development code derived the upwind target from world heading while the
environment already supplied body-frame airflow. The revised controller removes
that privileged world-frame shortcut. Upwind and crosswind bearings are computed
from `wind_x_body` and `wind_y_body` only.

## Running the recorded showcase

```bash
fly-sniff-record \
  --output artifacts/showcase/who-farted-run.json \
  --seed 13013 \
  --sim-seconds 45

fly-sniff-replay-showcase \
  artifacts/showcase/who-farted-run.json \
  --output artifacts/showcase/who-farted-4x5.mp4 \
  --seconds 15 \
  --fps 30
```

The development recording remains permanently labeled:

`DEVELOPMENT PROXY • NOT A MALECNS RESULT`

## Next biological upgrade

Once E001/E002 produces a reviewed and sealed MaleCNS circuit, the same recording
schema can carry connectome-controller diagnostics. The renderer should only add
named neural populations after those exact body IDs, edges, signs, and roles have
passed the repository qualification gates.

A future odor-motion model should likewise be added as an explicit controller or
ablation with its own tests. It should not be hidden inside the current proxy.
