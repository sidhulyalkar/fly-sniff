# Fly Video → Neural State research lane

This directory is an intentionally isolated research prototype. It is **not** part of the current `fly-sniff` odor-navigation controller or any sealed MaleCNS experiment.

## Concrete first goal

> Given recent Drosophila behavior video, predict future **measured** neural population activity in a held-out animal.

The first paired benchmark is MC2P. Large behavior-only datasets are registered for representation pretraining, not treated as neural ground truth.

## Why the boundary matters

Behavior does not uniquely identify the complete neural state. Missing sensory variables, internal state, neuromodulation, and many-to-one motor mappings make full activity reconstruction underdetermined. Future whole-connectome outputs therefore represent a modeled posterior over latent state, never measured firing.

## Local validation

```bash
python -m pip install -e '.[dev]'
fly-video-neural validate-registry configs/datasets_v1.json
fly-video-neural validate-benchmark configs/benchmark_v0.json
fly-video-neural inspect-mc2p /path/to/extracted/MC2P --output mc2p-manifest.json
pytest -q
```

No public dataset is downloaded in CI.

## MC2P ingress

`inspect-mc2p` discovers trial directories, groups them by animal identity, and binds the behavior video, synchronization file, measured dF/F, optional pose/kinematics, rest mask, and optional ROI traces into a hash-addressed manifest. Discovery never deserializes upstream pickle files.

## Baseline protocol

The first executable decoder is a NumPy ridge baseline. Hyperparameters are selected on the locked validation animal only. Test animals remain unread unless `--consume-test` is passed explicitly. This is a procedural guard for v0; stronger commit-reveal test locking can be added before a headline result.
