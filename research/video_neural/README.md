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
pytest -q
```

No public dataset is downloaded in CI.
