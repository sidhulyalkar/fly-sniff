# Odor-motion physical-validity audit

This note records why the current bilateral timing adapter is **not** eligible for functional-v2 navigation yet, even though its causal implementation passes synthetic direction tests.

## Frozen qualification result

The controller-free `odor-motion-qualification-v2` assay separates two questions that were previously conflated:

1. **Can the engineered comparator recover known bilateral latency when timing information is present?**
2. **Does the current fly-sniff stochastic-puff plume actually deliver resolvable timing information to the modeled antennae?**

On the frozen PR #21 qualification grid:

- all eight traveling-edge conditions passed: four realized lags (50, 100, 150, 200 ms) × two directions;
- every qualified traveling-edge estimate had the correct sign;
- 216 immobilized probes sampled the current stochastic-puff plume across eight seeds, three downwind positions, three crosswind offsets, and three antenna orientations;
- **zero of those fixed-plume samples produced a qualified directional timing estimate**;
- the qualification receipt therefore reports `blocked_no_resolvable_plume_timing` and `functional_v2_allowed = false`.

This is a negative result about the current **plume + sensor + temporal-resolution stack**, not evidence that odor motion is biologically unimportant and not a reason to lower thresholds after seeing the outcome.

## Temporal-resolution mismatch

The current arena runs at `dt = 0.05 s` (20 Hz), so the earliest distinct lag representable by the estimator is 50 ms.

Kadakia et al. (Nature, 2022), *Odour motion sensing enhances navigation of complex plumes*, reported fly odor-direction behavior at inter-antennal latencies on the order of 15 ms and modeled a correlation detector with a roughly 15 ms delay timescale. The published work therefore motivates a sensory assay at substantially finer temporal resolution than the current navigation simulator.

We must not describe 50--200 ms as a biologically validated Drosophila odor-motion timescale. It is the resolution of this engineering adapter.

Reference: https://doi.org/10.1038/s41586-022-05423-4

## Plume-statistics mismatch

The present `TurbulentPlume` is a convenient stochastic 2-D Gaussian-puff benchmark. It was never claimed to reproduce experimental smoke-plume statistics.

Brudner et al. (PNAS, 2026), *Fly navigational responses exploit plume-specific odor motion and gradient cues*, show that cue usefulness depends on plume statistics: bilateral gradient information and odor-motion information do not have equal predictive value in smooth and complex plumes.

That makes the current qualification failure mechanistically informative. A motion detector can be correct in isolation while receiving almost no motion information from a particular simulated plume.

Reference: https://github.com/emonetlab/GradientMotionMultiplePlumes

## Public experimental substrates for the next lane

The published reproduction repository provides explicit provenance and loaders for experimental plume data:

### Smooth plume

- Dryad DOI: `10.5061/dryad.g27mq71`
- file: `10302017_10cms_bounded_2.h5`
- dataset: `/dataset2`
- raw shape: `(3600, 406, 216)`
- source rate: 15 Hz
- repository provides both a notebook-compatible and corrected 60 Hz temporal interpolation profile.

### Complex plume

- DANDI dandiset: `001871`, version `0.260630.1657`
- public Figure-S1 NWB asset contains the background-subtracted 1,500-frame analysis window under
  `/scratch/smoke_1a_orig_figs1_background_subtracted_window_frames`;
- those rows correspond to full-video frames 300--1799 and are already scaled to the legacy-like 0--255 range.

Upstream reproduction contract:
https://github.com/emonetlab/GradientMotionMultiplePlumes/blob/main/REPRODUCE_FIGURES_1_3.md

## Required architecture for an experimental-plume successor

The replacement lane must be a new protocol rather than an edit that makes this v2 qualification pass.

It should:

1. ingest archived plume frames read-only and record upstream URL, version, file hash, dataset path, frame range, native rate, interpolation policy, intensity transform, and coordinate transform;
2. preserve native/high-rate sensory time independently from the slower navigation/control timestep;
3. first reproduce published gradient and motion cue maps or bilateral-history statistics from the archived data;
4. qualify any new bilateral timing estimator before it can affect connectome state;
5. compare smooth and complex plume conditions rather than choosing whichever produces a favorable result;
6. keep source coordinates, centerline labels, full plume pixels, and future frames out of controller inputs;
7. expose any interpolation or spatial resampling as an explicit engineering transformation;
8. keep this PR's `blocked_no_resolvable_plume_timing` receipt immutable as the result of the original stochastic-puff protocol.

## Claim boundary

A future result using these datasets may support statements about a **MaleCNS-derived model evaluated under experimentally grounded plume statistics**.

It still would not by itself show that a living fly implements our engineered temporal comparator, that a selected MaleCNS population computes odor motion, or that connectome topology is causally beneficial. Those stronger statements require the separately frozen biological-role and intact-vs-rewire experiments described in the functional-v2 gate.
