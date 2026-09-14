# Fly Video → Neural State research lane

This directory is an intentionally isolated research prototype. It is **not** part of the current `fly-sniff` odor-navigation controller or any sealed MaleCNS experiment.

## Concrete first goal

> Given recent Drosophila behavior, predict future **measured** neural population activity.

The first paired benchmark is MC2P. Large behavior-only datasets are registered for representation pretraining, not treated as neural ground truth.

## v1 correction before scoring

The original `benchmark_v0.json` proposed raw neural prediction in held-out animals. It is preserved rather than edited.

Before any real benchmark score was consumed, we identified a confound: raw two-photon image coordinates differ across animals, so held-out-animal raw-pixel regression mixes neural-dynamics prediction with unseen anatomy. `benchmark_v1.json` therefore makes the primary task **within-animal, held-out-session** future-neural prediction. The unseen-animal task remains a secondary lane that is blocked until a subject-invariant measured-neural representation is frozen.

## MC2P ingress and safe conversion

`inspect-mc2p` discovers trial directories, groups them by animal identity, and binds behavior video, synchronization, raw/resized measured dF/F, optional pose/kinematics, rest masks, and optional ROI traces. Discovery never deserializes upstream pickle files.

Convert only a trusted upstream pickle explicitly:

```bash
fly-video-neural convert-mc2p-legacy SESSION/sync_indices.pkl \
  --kind alignment --output SESSION/indices.npy \
  --receipt SESSION/indices.conversion.json --trust-upstream-pickle
fly-video-neural convert-mc2p-legacy SESSION/pose_result.pkl \
  --kind pose3d --output SESSION/pose3d.npy \
  --receipt SESSION/pose3d.conversion.json --trust-upstream-pickle
```

The trust flag is deliberately noisy because Python pickle may execute code while loading. Conversion receipts bind source/output hashes and normalized array structure but do not establish biological correctness.

## Deterministic pose → measured-neural session batch

```bash
fly-video-neural build-mc2p-session-batch SESSION/windows.json SESSION/pose3d.npy \
  SESSION/2p_dff_resized.mm --dff-side 64 \
  --output SESSION/pose-neural-v1.npz --receipt SESSION/pose-neural-v1.json
```

The pose transform mirrors MC2P's group-root preprocessing and summarizes only the input behavior window. The target is the mean of measured dF/F frames mapped into the future interval.

## Freeze v1 session assignments

Build every eligible session batch first, then freeze the exact batch bytes and within-animal session assignments:

```bash
fly-video-neural make-v1-session-split data/batches/*.npz --output data/v1-session-split.json
```

For every animal with at least three sessions, seed `2701` assigns one validation session, one test session, and all remaining sessions to training. The split lock contains the SHA-256 of every source batch.

## Run the low-complexity baseline

Development mode never reads test metrics:

```bash
fly-video-neural within-animal-ridge data/v1-session-split.json data/batches/*.npz \
  --output data/ridge-development.json
```

Only after the feature/target/split/model-selection contract is frozen should test be explicitly consumed:

```bash
fly-video-neural within-animal-ridge data/v1-session-split.json data/batches/*.npz \
  --output data/ridge-final.json --consume-test
```

A separate ridge model is selected and fit inside each animal. Aggregate metrics are summaries of per-animal test results, never one cross-animal raw-pixel model.

## Current gate

Once the public MC2P bytes are locally available, the code path is now complete from trusted conversion → synchronized windows → deterministic session batches → frozen within-animal splits → ridge development evaluation. A video foundation model is intentionally deferred until that baseline produces an auditable real-data result.
