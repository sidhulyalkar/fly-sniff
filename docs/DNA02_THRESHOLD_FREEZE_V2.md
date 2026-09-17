# DNa02 threshold freeze v2

## Status

`THRESHOLDS_FROZEN_BEFORE_BEHAVIOR_V2`

The Figure 3C DNa02 prominence thresholds are frozen from electrophysiology-only evidence before `yaw`, the Figure 3C statistic, model output, or navigation performance is opened.

The sealed authority receipt is:

`authority/program-a-dna02-threshold-freeze-receipt-v2.json`

Its canonical receipt SHA-256 is:

`f21ea2f0f514b0da55a05a99c6fd62fd641c72cfc40883c8064281692ce52607`

The complete deterministic manifest generated from the frozen evidence has SHA-256:

`b26a5d9b156a907eb036afd95643080e90b7dbb9b89dd9b2cbf4b56dc14ff6dc`

The executable freezer authority referenced by that receipt is commit:

`cd859f49844076ec22e107bddb296fe965997aae`

## Primary thresholds

| Fly | Side | Prominence quantile | Threshold |
|---|---|---:|---:|
| `a2_d_08` | L | 0.999 | 6.637119885669186 |
| `a2_d_08` | R | 0.999 | 6.765934524457742 |
| `a2_d_12` | L | 0.995 | 5.413398798564633 |
| `a2_d_12` | R | 0.995 | 5.661359850533993 |
| `a2_d_13` | L | 0.9995 | 5.32161236591439 |
| `a2_d_13` | R | 0.999 | 8.736763473032937 |
| `a2_d_14` | L | 0.999 | 5.497124009234522 |
| `a2_d_14` | R | 0.999 | 5.620147750325609 |

These values are not recovered author thresholds. They are the v2 project thresholds chosen by manual electrophysiology-only adjudication using the frozen prominence audit, raw-window/waveform QC, and individual-waveform distribution review.

## Why the distribution gate mattered

The high-threshold median-waveform review alone could bias the analysis toward overly stringent thresholds because retaining only the largest action potentials naturally produces cleaner median waveforms.

The final distribution gate therefore compared individual events in three nested prominence bands against the within-channel q0.9995+ anchor waveform:

1. q0.995 <= prominence < q0.999
2. q0.999 <= prominence < q0.9995
3. q0.9995+

The lower band for `a2_d_12` L/R overwhelmingly retained the same waveform family as the anchor, supporting q0.995. In contrast, `a2_d_13` L remained heterogeneous through q0.999 and showed a sharp waveform/amplitude transition only at q0.9995. The remaining channels supported q0.999 as the least stringent defensible separator.

Event rate was descriptive only and was not a selection criterion.

## Evidence chain

The v2 freeze binds to all of the following:

- prominence audit SHA-256: `4c421c803a696c5193b645412f674875c0746509043b835b11b8501d49fc2c9a`
- threshold-QC SHA-256: `da7b8d837c5c8116847a9f58b281516500bf7f7bbb775d5a8f0aea783c864958`
- distribution-review SHA-256: `a451d9166248a73ba7607b6c0cb78eab241a470ac0bfbdb83a21db5682a61686`
- compact threshold-freeze evidence SHA-256: `c0372a777e23298cb703091815bcea2a04de56d6b9ad43087b0e8d84023adc12`
- compact distribution-freeze evidence SHA-256: `591974bcd0b726f072c54b58cba3720faeff4c6cdff53a7c96373a71b2514732`
- threshold decisions SHA-256: `a8f89c2599a05a4be29776b32c7c5637b39b3d2d297767b8e119809f2ff3a612`

The compact repository evidence is derived from the larger local receipts. The larger receipts are identified by their parent hashes rather than duplicated into Git history.

## Prespecified sensitivity profiles

Before behavior was opened, v2 also froze three threshold profiles:

- primary;
- one candidate step lower;
- one candidate step higher.

These profiles are stored in the authority receipt. They may later quantify whether the Figure 3C reproduction is robust to a nearby neural-only threshold choice. A sensitivity profile may never replace the primary profile because it produces a more favorable downstream result.

## One-way door

The next allowed action is a separately qualified Figure 3C reproduction pipeline.

Once that pipeline reads `yaw` for v2:

- primary thresholds may not change;
- the four-fly cohort may not change because a downstream result is inconvenient;
- a sensitivity profile may not be promoted to primary;
- navigation performance may not alter any physiology decision.

A failed or partial Figure 3C reproduction is itself a valid result. Failure does not reopen threshold adjudication.

## Remaining method qualification before behavior

Threshold freezing does **not** by itself authorize an exact reproduction claim. Before the first v2 behavior opening, the extractor must still qualify or explicitly bound:

- MATLAB `findpeaks` relative-prominence semantics versus the portable implementation;
- exact 10 ms non-overlapping spike-count binning;
- MATLAB `smoothts(..., 'e', ...)` exponential-smoothing semantics corresponding to the published 30 ms window;
- 150 ms neural-to-behavior alignment convention;
- 50 ms relationship averaging;
- raw `yaw` units, sign, and calibration provenance.

Only after those transformation contracts are fixed should the pipeline open `yaw` and compute the Figure 3C relationship.

## Executable authority

The v2 manifest generator is frozen to commit `696668d3f24d8d0caa2208c8ba22a3cb5348601e`. Later documentation, tests, or receipt commits do not change this executable authority reference.

