# DNa02 prominence audit v1

This lane exists to adjudicate the only remaining Figure 3C extraction ambiguity before numeric reproduction: the relative-prominence threshold used to call spikes in each bilateral DNa02 channel.

The version-of-record methods specify MATLAB `findpeaks` relative prominence, 10 ms non-overlapping spike bins, exponential `smoothts` smoothing with a 30 ms window, a 150 ms neural-to-behavior comparison alignment, and 50 ms averaging for the neuron-behavior relationship. The publication does not expose the exact hand-tuned prominence value for each of the eight Figure 3C channels.

## Scientific firewall

`fly-sniff-dna02-prominence-audit` re-authenticates all four frozen source files and then loads only:

- `ephys_SR`
- `ephys_A`
- `ephys_B`

It never loads `yaw`, `fwd`, `lat`, treadmill time series, navigation results, or model outputs. The output contains only prominence-distribution summaries and candidate threshold sweeps. It does not automatically choose or freeze a threshold and does not compute Figure 3C.

## Local run

From current project environment:

```bash
mkdir -p data/cache/dna02-threshold-audit

fly-sniff-dna02-prominence-audit \
  data/raw/dna02/180410_gfp_3G_ss730_dual_08_data_for_SH_with_lat_vel.mat \
  data/raw/dna02/180430_gfp_3G_ss730_dual_12_data_for_SH_with_lat_vel.mat \
  data/raw/dna02/180501_gfp_3G_ss730_dual_13_data_for_SH_with_lat_vel.mat \
  data/raw/dna02/180517_gfp_3G_ss730_dual_14_data_for_SH_with_lat_vel.mat \
  --out data/cache/dna02-threshold-audit/prominence-audit-v1.json
```

Return only:

`data/cache/dna02-threshold-audit/prominence-audit-v1.json`

Do not upload the raw MAT files.

## Interpretation

The candidate sweep is deliberately descriptive. Quantile-derived prominence values are not assumed to equal the publication's original manually scrutinized thresholds. The next review must choose an electrophysiology-only adjudication policy or preserve a transparent reproduction deviation if the original exact thresholds cannot be recovered.
