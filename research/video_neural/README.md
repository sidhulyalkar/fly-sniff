# Fly Video → Neural State research lane

This directory is an intentionally isolated research prototype. It is **not** part of the current `fly-sniff` odor-navigation controller or any sealed MaleCNS experiment.

## Concrete first goal

> Given recent Drosophila behavior, predict future **measured** neural population activity.

The first paired benchmark is MC2P. Large behavior-only datasets are registered for representation pretraining, not treated as neural ground truth.

## v1 correction before scoring

The original `benchmark_v0.json` proposed raw neural prediction in held-out animals. It is preserved rather than edited.

Before any real benchmark score was consumed, we identified a confound: raw two-photon image coordinates differ across animals, so held-out-animal raw-pixel regression mixes neural-dynamics prediction with unseen anatomy. `benchmark_v1.json` therefore makes the primary task **within-animal, held-out-session** future-neural prediction. The unseen-animal task remains a secondary lane that is blocked until a subject-invariant measured-neural representation is frozen.

## Local validation

```bash
python -m pip install -e '.[dev]'
fly-video-neural validate-registry configs/datasets_v1.json
fly-video-neural validate-benchmark configs/benchmark_v0.json
fly-video-neural validate-benchmark configs/benchmark_v1.json
fly-video-neural inspect-mc2p /path/to/extracted/MC2P --output mc2p-manifest.json
pytest -q
```

No public dataset is downloaded in CI.

## MC2P ingress

`inspect-mc2p` discovers trial directories, groups them by animal identity, and binds behavior video, synchronization, measured dF/F, optional pose/kinematics, rest masks, and optional ROI traces. Discovery never deserializes upstream pickle files.

The public release contains legacy pickle synchronization and pose artifacts. Convert only a trusted upstream file explicitly:

```bash
fly-video-neural convert-mc2p-legacy SESSION/sync_indices.pkl \
  --kind alignment \
  --output SESSION/indices.npy \
  --receipt SESSION/indices.conversion.json \
  --trust-upstream-pickle

fly-video-neural convert-mc2p-legacy SESSION/pose_result.pkl \
  --kind pose3d \
  --output SESSION/pose3d.npy \
  --receipt SESSION/pose3d.conversion.json \
  --trust-upstream-pickle
```

The trust flag is deliberately noisy because Python pickle may execute code while loading. Conversion receipts bind source/output hashes and normalized array structure but do not establish biological correctness.

## Baseline protocol

The next data-gated milestone is deterministic pose-feature extraction and a within-animal ridge baseline on held-out sessions. A video foundation model comes only after that low-complexity baseline exists.
