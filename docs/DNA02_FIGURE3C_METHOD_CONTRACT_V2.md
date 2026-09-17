# DNa02 Figure 3C method contract v2

## Purpose

This tranche freezes the Figure 3B-C transformation contract **after** the DNa02 prominence thresholds were frozen in PR #73 and **before** any numeric `yaw` sample is inspected.

The contract intentionally separates three evidence tiers:

1. the eLife version-of-record Methods and published author clarification;
2. the pinned secondary analysis code operating on the released MAT schema;
3. historical MATLAB code, retained only as contextual or predeclared sensitivity evidence when it conflicts with the version of record.

The canonical contract is:

`authority/program-a-dna02-figure3c-method-contract-v2.json`

with canonical SHA-256:

`b1760e495e9be7029222320223b5e25d67e2b559ee7eeb6e004418e2b70ddfff`

## Threshold one-way door

This method contract is bound to the canonical threshold freeze already on `main`:

- threshold freeze receipt: `a308eb18cef1d5e524243457e5cae3d3caf0803855b43ae722e643548f11aad8`
- generated threshold manifest: `1367113491adc3625d52a9fdc7214820677d19495df8c56066aacd991711d082`
- canonical merge commit: `7faf77ecfc9808c8520c2e3fcf4b3c074b840a42`

No Figure 3 result may reopen those eight neural-only decisions.

## Version-of-record neural transform

For Figure 3B-C, the published Methods specify:

- relative-prominence spike detection using MATLAB `findpeaks`;
- 10 ms non-overlapping spike-count bins;
- exponential MATLAB `smoothts` smoothing with a 30 ms window;
- neural data shifted forward by 150 ms before comparison to behavior;
- both neural and behavioral signals averaged in 50 ms non-overlapping windows.

At 10 kHz ephys, a 10 ms bin is exactly 100 samples. At the expected 100 Hz ball rate, 150 ms is 15 samples and 50 ms is five samples.

The predictor is frozen as:

`right DNa02 firing rate - left DNa02 firing rate`.

The laboratory's published sign convention defines clockwise/rightward body rotation as positive. Therefore a positive R-L predictor is expected to correspond to positive/rightward rotational velocity. This sign may not be flipped after behavior is opened.

## Exponential smoothing conformance

Legacy `smoothts` documentation specifies, for exponential mode with period length `n > 1`:

`alpha = 2 / (n + 1)`.

A 30 ms window over 10 ms bins therefore gives:

`n = 3`, `alpha = 0.5`.

The frozen recurrence is:

`y[0] = x[0]`

`y[t] = 0.5*x[t] + 0.5*y[t-1]`.

A behavior-free numerical fixture is frozen as:

`[1,2,3,4,5] -> [1,1.5,2.25,3.125,4.0625]`.

The historical helper `calculate_psth_A2.m` additionally divides the `smoothts` output by 1.5 and calls this an integral correction. That helper also uses a derivative-threshold spike detector that conflicts with the final Methods. The primary v2 reconstruction therefore does **not** silently inherit `/1.5`. The operation is frozen as a predeclared firing-rate-scale sensitivity and may not be promoted because it improves the downstream relationship.

## Kinematic preprocessing evidence

The version of record says treadmill variables used in these analyses receive offset correction, 50 ms Gaussian smoothing, and Gaussian-random-walk MAP smoothing.

The pinned secondary code at commit:

`7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8`

provides the concrete released-MAT processing path:

- raw `yaw` scale: `raw_yaw * 0.000395 / 0.000436`;
- 20 ball-sample interpolated padding;
- 5 ball-sample Gaussian kernel;
- 10x upsample, forward/reverse convolution average, downsample;
- `gp_smooth(..., smooth_ratio=0.2)`;
- baseline estimated from samples whose absolute first difference is below 0.025, then one global median offset is subtracted.

The original environment is Python 3.7.3 with NumPy 1.16.3, SciPy 1.2.1, PyMC3 3.6, and Theano 1.0.4.

The PyMC3/Theano random-walk stage remains the only transformation that is not yet authorized for a modern reimplementation. Before numeric yaw is opened, we must either execute the pinned environment or qualify a replacement numerically against that implementation.

## Transform-only qualification

`fly-sniff-dna02-figure3c-method-v2` validates the content-addressed contract without behavior.

Unit tests use synthetic data to qualify:

- exact 10 ms spike-bin boundaries;
- count-to-Hz conversion;
- the exponential recurrence;
- bilateral R-L construction;
- 150 ms future-behavior alignment;
- 50 ms non-overlapping aggregation and trailing-window handling.

No MAT behavior field is loaded by these tests.

## Remaining gate

Behavior stays sealed until:

- the PyMC3/Theano smoothing stage is numerically qualified or executed from the pinned environment;
- portable prominence semantics are qualified against MATLAB-style synthetic fixtures;
- the extractor implementation is sealed to an exact code commit.

Opening yaw will then consume a one-way gate. A failed or partial Figure 3C reconstruction is an allowed scientific result and does not reopen thresholds, signs, smoothing choices, or the four-fly cohort.
