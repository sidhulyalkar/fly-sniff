# DNa02 threshold adjudication v1

This tranche converts the navigation-blind prominence audit into an auditable, fail-closed
electrophysiology review without opening behavior.

## Scientific boundary

The adjudication lane may read only:

- `ephys_SR`
- `ephys_A`
- `ephys_B`

It must not load or inspect `yaw`, `fwd`, `lat`, behavioral timebases, Figure 3C fit, odor-navigation
results, model outputs, or task score while choosing thresholds.

The output is **not** an automatic threshold estimate. It is a deterministic review packet for a
human electrophysiology decision.

## Why the prominence quantiles are not enough

The prominence audit summarizes the full positive-prominence distribution, but quantiles do not
identify the authors' original hand-tuned thresholds. Event rate at a given quantile is also partly
determined by the quantile itself, so firing-rate plausibility is descriptive context only and is
not an allowed selection basis.

The v1 adjudicator therefore evaluates a frozen upper-tail candidate set:

`q = 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999`

All candidates are derived from the already authenticated prominence-audit receipt. The tool runs
`find_peaks` once at the lowest candidate threshold and obtains every stricter candidate by filtering
that same peak/prominence set. This makes the candidate sets exactly nested and avoids repeated
threshold-dependent detection passes.

## QC generated for every candidate

For each of the eight fly/channel combinations the receipt records:

- event count and event rate, explicitly marked descriptive;
- median and 1st-percentile inter-spike interval;
- fraction of intervals shorter than 1 ms;
- fraction of intervals shorter than 2 ms;
- event rate in 20 equal-duration recording blocks and block-rate CV;
- up to 2,000 deterministic spike waveforms;
- median waveform and median absolute deviation envelope;
- median peak amplitude and amplitude MAD;
- median waveform-to-template correlation;
- approximate median-waveform FWHM.

For each bilateral recording it also records near-synchronous L/R event fractions at 0.2 ms and
0.5 ms tolerances. These are artifact diagnostics, not proof that synchronous events are artifacts.

## Deterministic visual review

Two PNGs are produced per channel.

`*-raw-windows.png` shows 250 ms raw voltage windows centered at fixed 10%, 30%, 50%, 70%, and 90%
positions through the recording. The locations are not selected by a reviewer. Three candidate
thresholds are overlaid: the lowest candidate, the midpoint candidate, and the highest candidate.

`*-candidate-qc.png` shows candidate-level event rate, refractory violations, block stability, and
waveform stereotypy. Event rate is labeled descriptive only.

## Run on the Mac

From the repository root:

```bash
git fetch origin
git switch feat/dna02-threshold-adjudication-v1
git pull --ff-only

bash scripts/run_dna02_threshold_adjudication_mac.sh
```

If the MAT files live somewhere other than `data/raw/dna02`, pass the directory:

```bash
bash scripts/run_dna02_threshold_adjudication_mac.sh \
  /absolute/path/to/dna02 \
  data/cache/dna02-threshold-adjudication-v1
```

The script regenerates the prominence audit from the authenticated MAT files, runs neural-only QC,
records the exact Git commit, and creates:

```text
data/cache/dna02-threshold-adjudication-v1-share.zip
```

Upload only that ZIP for review. Do not upload the raw MAT files.

## Decision template

`threshold-decisions-template.json` contains exactly eight blank decisions. For each channel a
reviewer must explicitly fill:

- `selected_prominence_quantile`
- `selected_threshold`
- `selection_basis`
- `rationale`

Allowed selection bases are:

- `raw_trace_review`
- `waveform_stereotypy`
- `refractory_violations`
- `blockwise_stability`
- `bilateral_artifact_coincidence`

`event_rate` is deliberately not an allowed selection basis.

The selected threshold must exactly match one candidate already present in the QC receipt. Arbitrary
new values are rejected in v1. If the original publication threshold is later recovered from an
authoritative source, that should become a separately versioned authority lane rather than silently
changing this manual-reproduction contract.

## Freeze gate

After all eight neural-only decisions have been reviewed:

```bash
cp \
  data/cache/dna02-threshold-adjudication-v1/threshold-decisions-template.json \
  data/cache/dna02-threshold-adjudication-v1/threshold-decisions-v1.json

# Edit only the eight decisions plus reviewer. Do not inspect behavior.

fly-sniff-dna02-threshold-freeze \
  --qc data/cache/dna02-threshold-adjudication-v1/threshold-qc-v1.json \
  --decisions data/cache/dna02-threshold-adjudication-v1/threshold-decisions-v1.json \
  --code-ref "$(git rev-parse HEAD)" \
  --out data/cache/dna02-threshold-adjudication-v1/threshold-manifest-v1.json
```

The freezer rejects:

- fewer or more than eight decisions;
- duplicate fly/side entries;
- thresholds not present in the audited candidate set;
- empty rationales;
- unrecognized selection bases;
- a mismatched QC hash or source identity;
- any declaration that behavior, yaw, Figure 3C, or navigation performance was reviewed.

A successful manifest has status:

`THRESHOLDS_FROZEN_BEFORE_BEHAVIOR`

Only after that artifact exists may the Figure 3C reproduction lane load `yaw`.

## What this does not establish

This tranche does not reproduce Figure 3C, validate the 150 ms alignment as a universal neural delay,
derive an absolute DNa02-to-turn gain for MaleCNS, or show that intact connectome wiring improves odor
navigation. It only freezes spike-detection thresholds independently of those downstream outcomes.
