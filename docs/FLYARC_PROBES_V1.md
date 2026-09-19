# FlyARC representation probes v1

## Why this tranche exists

The first 500-step FlyARC run produced distinct modeled reservoir states but an identical ARC
trajectory for intact MaleCNS, a degree-preserving rewire, a random recurrent null, and the
stateless control. That makes the first-light dataset more valuable as an **identical-stimulus
representation experiment** than as a closed-loop game benchmark.

This branch starts from the frozen first-light checkpoint:

`4ed40fa205f54c276cace7150ab6a685e59b447c`

The closed-loop policy and its null result are not tuned here.

## Primary question

> Given exactly the same recorded ARC sensory history, does biological MaleCNS recurrent
> topology preserve temporal information differently from matched controls?

The primary developmental endpoint is **excess memory R²**.

For each lag in:

`1, 2, 4, 8, 16, 32`

we train the same cross-validated linear ridge probe to reconstruct a fixed 128-dimensional
random signed sketch of the past categorical ARC frame features.

Two predictors are evaluated:

1. the modeled reservoir state at time t;
2. the current encoded ARC frame at time t.

The metric is:

`excess_memory_R2 = reservoir_R2 - current_frame_R2`

The subtraction matters. ARC screens often evolve slowly, so the previous frame can be
predictable from the current frame even when a system has no internal memory.

The primary marathon summary is the mean excess R² across the six frozen lags.

## Controls

The replay probe adds one control that the closed-loop v1 did not contain:

- **intact** — biological MaleCNS topology;
- **rewire** — exact directed in/out-degree-preserving target swaps;
- **random** — matched edge/sign/weight multiset with randomized topology;
- **leak_only** — the same input projection, gain, and leak but recurrent W=0;
- **stateless** — instantaneous input response with both recurrence and temporal leak removed.

This separates:

`recurrent topology`

from:

`single-unit leaky memory`

from:

`no temporal state`.

## Source-trajectory gate

The probe refuses to run unless:

- the source FlyARC receipt validates;
- at least two source conditions contain hashed frame streams;
- all condition `frames.npz` SHA-256 values are identical;
- reset observations can be reconstructed consistently from the recorded steps.

The probe therefore cannot quietly compare reservoirs on different sensory histories.

## Cross-validation

When the trajectory contains at least two reset-delimited segments, each usable segment is
held out in turn. This is stronger than randomly mixing adjacent timepoints between train and
test.

For a single long segment, deterministic blocked folds are used with a 32-step temporal purge.

The ridge regularization, lag set, target sketch, target-sketch seed, and primary endpoint are
all frozen in:

`authority/flyarc-representation-probes-v1.json`

## Secondary measurements

Every topology also reports:

- next-frame prediction and excess over the current-frame baseline;
- current-frame representation fidelity;
- state RMS and mean absolute activity;
- saturation fraction;
- state-change RMS;
- lag-1 state cosine;
- approximate effective dimensionality;
- approximate 95% variance rank.

Dimensionality is computed on the same deterministic 256-neuron body-ID-hash subsample for
every topology so the multi-hour run remains practical.

## Projection robustness

A single random ARC→neuron interface could accidentally favor a topology.

The marathon therefore uses multiple frozen input-projection blocks. Within one block every
topology receives the **same** input projection. Between blocks the projection changes by a
deterministic seed schedule.

The default development schedule is:

- 4 projection blocks;
- intact + leak-only + stateless anchor controls in every block;
- 16 independently rewired graphs per block;
- 8 random recurrent graphs per block.

The overnight script uses a deeper null schedule but the same frozen probe definition.

## Resumability

Each topology/projection combination is written atomically to:

`results/<job-id>.json`

The output directory contains a frozen `manifest.json`. Re-running with the exact same
contract skips valid completed jobs. Changing any manifest field requires a new output
directory.

Ctrl-C is handled gracefully: completed checkpoints are retained and the aggregate report is
rebuilt before exit.

## Outputs

The marathon produces:

```text
manifest.json
progress.json
results/*.json
summary.csv
aggregate.json
report.md
memory_curve.png
primary_metric_distribution.png
receipt.json
```

The report is intentionally descriptive. An intact result near the edge of a rewire
distribution is hypothesis-generating, not a confirmatory biological-topology claim.

## Quick developmental run

```bash
fly-sniff-arc-marathon \
  ~/fly-sniff-data/malecns-v1.0/flyarc-core-4096 \
  ~/fly-sniff-data/artifacts/flyarc-ls20-firstlight-4ed40fa \
  --output ~/fly-sniff-data/artifacts/flyarc-probes-quick \
  --hours 0.5 \
  --projection-count 2 \
  --rewires-per-projection 8 \
  --randoms-per-projection 4
```

## Multi-hour run

Use the Mac helper:

```bash
FLYARC_HOURS=8 bash scripts/run_flyarc_probe_marathon_mac.sh
```

The helper uses macOS `caffeinate` when available, runs the focused probe tests first, and
prints the resulting report when the marathon stops or exhausts its frozen schedule.
