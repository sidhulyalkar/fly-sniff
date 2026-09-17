# DNa02 Figure 3C method authority

## Decision

For Program A, the version-of-record Methods for **Figure 3B-C** remain the controlling authority for the production reproduction contract:

- spikes are detected with MATLAB `findpeaks` using relative prominence;
- prominence is selected independently for each experiment/neuron and manually scrutinized;
- spikes are counted in 10 ms non-overlapping bins;
- firing rate is smoothed with MATLAB `smoothts` using an exponential 30 ms window;
- the Figure 3 relationship is summarized in 50 ms bins;
- the neural-to-behavior comparison uses the publication's 150 ms shift.

The exact per-experiment/per-neuron prominence values are not exposed by the version-of-record text we have identified. They therefore remain an independent source-reproduction blocker. Navigation performance, Figure 3C fit quality, and expected firing rate may not resolve that blocker.

## Historical MATLAB evidence

Pinned primary repository:

- repository: `SashaRayshubskiy/eLife_102230_analysis_code`
- commit: `55e30c19b1a18f601df1803295f0c401aec3c167`
- `calculate_psth_A2.m`
- `A2_dual_patch_analysis_v2.m`

The historical `calculate_psth_A2.m` path is materially different from the version-of-record Figure 3B-C Methods. It:

1. median-filters voltage over 80 ms;
2. subtracts the filtered baseline;
3. differentiates the residual;
4. thresholds that derivative;
5. calls `findpeaks` with a 5 ms minimum peak distance;
6. bins the resulting spikes;
7. applies `smoothts(..., 'e', 3)` and an integral correction.

`A2_dual_patch_analysis_v2.m` passes `SPIKE_THRESHOLD_LAL_DN = 0.25` to that function for both channels of its `180410..._08` session.

This is useful historical evidence, especially for reconstructing old analysis behavior, but it is **not authority to replace the final Figure 3B-C relative-prominence method**. The script is session-specific and its spike detector conflicts with the final Methods description.

## Secondary Python evidence

Pinned secondary repository:

- repository: `wilson-lab/rayshubskiy_elife_102230_secondary_analysis_code`
- commit: `7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8`
- `import_preprocess_data.ipynb`

That notebook contains a global A2 prominence setting (`thresh_a2 = 15.0`) with a comment that roughly 7.5-11 typically works, plus a 10 s prominence window, 12.5 ms bins, and a Gaussian firing-rate estimate. This is a later secondary preprocessing path and also does not match the version-of-record Figure 3B-C contract.

It may be used as QC context only. It must not be silently substituted as the Figure 3C production pipeline or used to retroactively choose the eight Program A thresholds.

## First-light adjudication consequence

The first neural-only QC receipt is content-addressed by:

- prominence audit: `4c421c803a696c5193b645412f674875c0746509043b835b11b8501d49fc2c9a`
- threshold QC: `da7b8d837c5c8116847a9f58b281516500bf7f7bbb775d5a8f0aea783c864958`

It remains behavior-blind and does not freeze thresholds. Multiple channels exhibit a substantial high-prominence waveform-family transition, while the first raw-window plots show only the first, middle, and last of the seven candidates. Therefore the next permitted step is a neural-only visual comparison of q0.995, q0.999, and q0.9995, together with high-threshold waveform-family overlays.

## Hard boundary

Until all eight thresholds are defensibly frozen:

- do not load `yaw`, `fwd`, or `lat` for threshold selection;
- do not compute Figure 3C and then revise thresholds;
- do not use navigation performance;
- do not infer thresholds from desired firing rates;
- do not treat the historical derivative threshold or secondary Python threshold as the missing publication prominence values.

A channel that remains ambiguous after the strengthened ephys-only review remains ambiguous. The correct v1 result can be `BLOCKED`, rather than a threshold chosen to make downstream results work.
