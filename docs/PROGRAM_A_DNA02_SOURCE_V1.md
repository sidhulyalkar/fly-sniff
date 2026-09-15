# Program A DNa02 source contract v1

Program A needs a navigation-independent physiological constraint on the steering readout. Rayshubskiy et al. provide unusually strong bilateral DNa02 evidence, but reproducing that evidence requires more than copying a sentence from the paper.

This contract freezes the **source-ingress problem** before any DNa02 number can become a calibration target.

## Published target

The version of record is:

- Rayshubskiy et al., *Neural circuit mechanisms for steering control in walking Drosophila*;
- DOI `10.7554/eLife.102230.3`;
- Figure 3C.

Figure 3C reports mean rotational velocity as a function of the **right-minus-left DNa02 firing-rate difference**, one line per fly, `n=4` paired flies. The text describes this relation as essentially linear across the observed dynamic range and consistent across paired recordings.

For Program A this supports a candidate **normalized relational** calibration target. It does not justify transferring the absolute firing-rate-to-turn gain from this preparation to exact MaleCNS body IDs.

## Published Figure 3 timing/preprocessing

The version-of-record Methods describe the Figure 3B-C firing-rate path as:

- spike detection using relative peak prominence;
- 10 ms non-overlapping firing-rate bins;
- exponential smoothing with a 30 ms window;
- neural activity shifted by 150 ms for the Figure 3 firing-rate/behavior comparison;
- neural and behavioral variables then averaged into 50 ms non-overlapping windows.

The 150 ms shift is preserved as a **publication reproduction alignment**. It is explicitly not promoted to a universal biological delay because the paper states that the lag may reflect biological delay and/or inertia of the spherical treadmill.

Program A must not fit a delay to navigation performance to make this relationship look better.

## Immutable analysis-code authorities

The source contract pins both public analysis repositories.

Primary MATLAB repository:

- `SashaRayshubskiy/eLife_102230_analysis_code`
- commit `55e30c19b1a18f601df1803295f0c401aec3c167`
- `A2_dual_patch_analysis_v2.m`
- git blob `3dd3d22135bf399d70dc3c41172eef965e1abbf7`

Secondary Python repository:

- `wilson-lab/rayshubskiy_elife_102230_secondary_analysis_code`
- commit `7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8`
- `import_preprocess_data.ipynb`, blob `0da2089b468c172f700881b714bfdda99a6fe424`
- `kv_linrel_scatters.ipynb`, blob `693bbdd8b60083744239ad049dd39f4d988418a4`
- `physiology_quant_analysis_figures.ipynb`, blob `384f5d342e089e101600955820fa46306b1c2593`

The secondary import notebook identifies the source fields needed by a portable extractor:

- `ephys_A`: left DNa02 recording;
- `ephys_B`: right DNa02 recording;
- `yaw`: rotational velocity;
- `fwd`, `lat`: other ball velocities;
- `stim`: optogenetic stimulus;
- `t_ephys`, `t_ball`: separate neural and ball timebases.

Machine-local paths in the historical notebooks are not authority.

## Why the Figure 3C cohort remains blocked

General secondary-analysis dictionaries contain four bilateral A2 aliases:

- `a2_d_08`
- `a2_d_12`
- `a2_d_13`
- `a2_d_14`

The paper reports `n=4` for Figure 3C, so these are an obvious candidate set. However, `physiology_quant_analysis_figures.ipynb` separately comments out `a2_d_14` as `LOW SNR, REJECT!` for a different physiology-quantification analysis.

That comment is neither evidence that `_14` was excluded from Figure 3C nor evidence that it was included. Selecting four aliases merely because the paper says `n=4` would be circular bookkeeping.

Therefore the real authority artifact remains:

`BLOCKED_FIGURE3C_COHORT_UNRESOLVED`

until panel-level provenance or exact source data identifies the four traces used for Figure 3C.

## Why the Dataverse map remains blocked

The paper identifies Harvard Dataverse DOI:

`10.7910/DVN/0NCLP1`

The current audit has not yet resolved an exact Dataverse file ID + filename + SHA-256 for every Figure 3C fly. Local Dropbox paths and historical machine directories are not acceptable substitutes.

Therefore the second blocker is:

`BLOCKED_DATAVERSE_FILE_MAP_UNRESOLVED`

A resolved source contract requires exactly one unique, content-addressed Dataverse input for every frozen Figure 3C fly alias.

## Promotion rule

`dna02_source.py` deliberately uses a two-key gate:

1. exact four-fly Figure 3C cohort + explicit cohort authority;
2. exact one-to-one Dataverse file map for that cohort.

Only after both resolve does the source artifact become:

`READY_FOR_EXTRACTION`

That state still does **not** mean a calibration target exists. It only authorizes deterministic extraction of the prespecified bilateral steering statistic from the verified source bytes.

## Evidence boundary

The contract cites the existing EvidenceLedger records:

- `rayshubskiy2025-dna02-bilateral-steering`
- `rayshubskiy2025-dna02-temporal-precedence`

Allowed interpretation:

- bilateral DNa02 activity has a reproducible lateralized relationship with rotational velocity in the published preparation;
- this relationship is a strong candidate for a normalized, navigation-independent Program A calibration constraint.

Forbidden interpretation includes:

- treating the 150 ms alignment as a universal model delay;
- transferring absolute gain unchanged to MaleCNS;
- choosing/excluding flies because doing so improves later odor navigation;
- replacing unmapped Dataverse inputs with guessed local files.

No navigation performance is used anywhere in this source contract.
