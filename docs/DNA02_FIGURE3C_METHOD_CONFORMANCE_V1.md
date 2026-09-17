# DNa02 Figure 3C method conformance v1

## Purpose

This tranche freezes the parts of the Figure 3C preprocessing path that are independently supported before any Figure 3C behavior is opened. It deliberately does **not** implement numerical smoothing yet.

The version-of-record Methods state that spikes for Figures 3B-C were detected with a relative-prominence metric in MATLAB `findpeaks`, counted in 10 ms non-overlapping bins, and smoothed with MATLAB `smoothts` using an exponential filter with a 30 ms window.

Legacy `smoothts` documentation states that in exponential mode an argument `n > 1` is a period/window length and corresponds to:

```text
alpha = 2 / (n + 1)
```

Therefore a 30 ms smoothing window on 10 ms spike-count bins resolves to:

```text
period_bins = 3
alpha = 0.5
```

These parameter values can be frozen without choosing a spike threshold, opening yaw, computing Figure 3C, or looking at navigation.

## Historical MATLAB implementation evidence

The pinned historical DNa02 helper at:

```text
SashaRayshubskiy/eLife_102230_analysis_code
commit 55e30c19b1a18f601df1803295f0c401aec3c167
calculate_psth_A2.m
```

contains:

```matlab
INTEGRAL_CORRECTION = 1.5;
cur_psth = smoothts(cur_psth, 'e', 3) / INTEGRAL_CORRECTION;
```

and describes `1.5` as an integral correction for the exponential smoothing window.

This is useful implementation evidence, but it is not automatically promoted into the final Figure 3C reproduction contract. The same historical helper belongs to an analysis path whose spike-detection implementation differs from the version-of-record Methods. We therefore preserve the `/1.5` operation as a conformance question rather than silently assuming it survived into the final production analysis.

## What remains unresolved

Before a Python numerical smoother is allowed, we still need to establish:

1. exact `smoothts` exponential recurrence initialization;
2. exact beginning-of-series edge behavior;
3. whether the historical `/1.5` integral correction is part of the final Figure 3C production path;
4. at least one short numerical fixture produced by authoritative MATLAB `smoothts`, or source-equivalent evidence sufficient to prove byte-level/numerical equivalence within a frozen tolerance.

Until those conditions are satisfied, the method contract reports:

```text
PARAMETERS_RESOLVED_EDGE_SEMANTICS_PENDING
```

and `numeric_smoothing_implementation_allowed` remains false.

## Scientific boundary

Threshold adjudication remains upstream of this method gate. Figure 3C behavior remains sealed. Neither Figure 3C fit quality nor odor-navigation performance may be used to choose smoothing initialization, edge handling, integral correction, or any other preprocessing detail.

The 150 ms neural-to-behavior alignment is retained only as a publication-reproduction operation. It is not interpreted as a universal biological delay or a fixed delay to impose on the MaleCNS navigation model.
