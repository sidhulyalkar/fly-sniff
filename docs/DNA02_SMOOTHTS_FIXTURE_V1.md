# Synthetic MATLAB `smoothts` fixture v1

## Goal

Resolve the remaining numerical-conformance uncertainty in the Figure 3C exponential smoother without reading any DNa02 recording, behavior, Figure 3C statistic, model output, or navigation result.

The version-of-record paper specifies 10 ms spike-count bins smoothed with MATLAB `smoothts` using the exponential method and a 30 ms window. Legacy documentation maps a period length of three bins to `alpha = 2/(3+1) = 0.5`.

A secondary historical worked example also supports the candidate recurrence:

```text
s(1) = x(1)
s(t) = alpha*x(t) + (1-alpha)*s(t-1)
```

The fixture tests that candidate directly against the installed MATLAB implementation instead of accepting it by analogy.

## What the fixture runs

Fixed synthetic row vectors only:

- constant sequence;
- ramp;
- impulse at the first sample;
- impulse in the middle;
- step;
- sparse count sequence.

For each vector it records:

- `smoothts(x, 'e', 3)`;
- `smoothts(x, 'e', 0.5)`;
- the explicit first-observation recurrence above;
- maximum absolute differences;
- the raw `smoothts(...,'e',3) / 1.5` values for inspection only.

It also records the MATLAB release, Financial Toolbox version, and resolved `smoothts` path.

The `/1.5` values are exported because the pinned historical DNa02 helper applied that integral correction. Exporting them does **not** declare that correction part of the final Figure 3C production pipeline.

## Run on macOS

If `matlab` is on `PATH`:

```bash
bash scripts/run_dna02_smoothts_fixture_mac.sh
```

The shareable result is:

```text
data/cache/dna02-smoothts-fixture-v1-share.zip
```

If MATLAB is installed but its executable is not on `PATH`, open MATLAB in the repository root and run:

```matlab
addpath('scripts')
dna02_smoothts_fixture('data/cache/dna02-smoothts-fixture-v1/smoothts-fixture-v1.json')
```

The shell runner is preferred because it also records the Git commit and file SHA-256.

## Boundary

This probe contains no fly data. It cannot select spike thresholds, inspect yaw, compute Figure 3C, calibrate MaleCNS, or evaluate navigation.
