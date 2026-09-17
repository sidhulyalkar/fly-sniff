# DNa02 individual-waveform distribution review v1

## Why this gate exists

The first-light and high-threshold neural-only reviews established that raising the prominence threshold often produces a cleaner median waveform. That observation alone is not enough to freeze the Figure 3C spike-detection thresholds.

A higher threshold can look cleaner simply because it retains only the largest action potentials. Deterministic raw windows from the real recordings also contain lower-prominence events that remain visually compatible with spikes. Freezing q0.999 or q0.9995 solely because the median waveform is prettier would therefore risk an over-stringency bias.

The v1 distribution review asks a narrower question before behavior is opened:

> Do the individual events that would be discarded by a higher threshold retain the same waveform shape as a high-confidence within-channel spike population?

## Frozen inputs

The command reads only:

- `ephys_SR`
- `ephys_A`
- `ephys_B`

It reuses the authenticated prominence-audit receipt and the exact four source-file SHA-256 identities. It does not read `yaw`, `fwd`, `lat`, behavioral timebases, Figure 3C statistics, navigation results, or model outputs.

## Nested event bands

For every fly and side, v1 detects peaks once at the q0.995 prominence threshold and partitions those detections into three non-overlapping sets:

1. q0.995 <= prominence < q0.999
2. q0.999 <= prominence < q0.9995
3. prominence >= q0.9995

The q0.9995+ set supplies a high-confidence within-channel anchor waveform. This anchor is not declared ground truth and does not automatically select a threshold. It is used only to score waveform-shape consistency in the lower-prominence bands.

## Diagnostics

For deterministic samples in each band, the review reports:

- individual waveform correlation to the q0.9995+ anchor;
- correlation quantiles;
- fractions with correlation >= 0.80, >= 0.90, and >= 0.95;
- baseline-subtracted peak-amplitude quantiles;
- fraction of positive peak amplitudes;
- shape-normalized individual-waveform overlays;
- correlation histograms;
- amplitude histograms;
- prominence-versus-anchor-correlation scatter.

Event rate remains excluded as a threshold-selection basis.

## Decision logic

This tool does not freeze thresholds. Human adjudication follows after the receipt and plots are reviewed.

A lower threshold becomes more defensible when the extra events it admits form the same spike-like waveform family as the high-confidence anchor across the individual-event distribution, not merely at the median.

A higher threshold becomes more defensible when the discarded lower-prominence band has broad or low anchor correlations, incompatible waveform shapes, or other neural-only evidence of contamination.

If the distributions remain mixed or ambiguous, the channel stays blocked. The next permitted response is a finer neural-only candidate grid or a more targeted waveform diagnostic, never behavior or navigation performance.

## Local run

From repository `main` or the active review branch:

```bash
git fetch origin
git switch feat/dna02-waveform-distribution-review-v1
git pull --ff-only

source .venv/bin/activate 2>/dev/null || true
bash scripts/run_dna02_threshold_distribution_review_mac.sh
```

The default output shared back for review is:

```text
data/cache/dna02-threshold-distribution-review-v1-share.zip
```

Do not upload the raw MAT files.
