# Program A PFN source ingress v1

This tranche freezes the quantitative **definition** of the ventral P-FN airflow target before any Program A navigation result exists. It does not claim that the source data have been extracted yet.

## Published target

The authority is Currier, Matheson & Nagel, eLife 2021 (`10.7554/eLife.61510`) and its Dryad release (`10.5061/dryad.vq83bk3rh`).

Program A v1 uses the Figure 4 P-F2N3 whole-cell electrophysiology experiment because it directly tests the conclusion we need: airflow tuning is organized primarily by **cell-body hemisphere**, not by fan-shaped-body column.

The publication reports:

- airflow only at 25 cm/s;
- eight directions: `-135, -90, -45, 0, 45, 90, 135, 180` degrees;
- five repetitions per direction, 40 trials per recording;
- 12 recordings, all of which completed the 40-trial protocol;
- a 1 s baseline window ending 500 ms before stimulus onset;
- a 1 s response window beginning 500 ms after stimulus onset;
- per-trial response = mean response-window firing rate minus mean baseline firing rate;
- per-direction response = mean across repetitions;
- preferred direction = angle of the mean of eight signed response vectors whose angles are airflow directions and whose coefficients are the corresponding mean responses.

Negative baseline-subtracted responses are therefore retained. They are not rectified before the vector calculation.

The publication describes preferred tuning near approximately 45 degrees ipsilateral to the soma. Its text establishes the angular convention used here: 0 degrees is frontal, +90 degrees is left, and -90 degrees is right. Program A folds right-soma angles by sign inversion so positive folded angle always means ipsilateral.

## Program A normalized statistic

The frozen target is intentionally geometric rather than an absolute-rate transfer:

1. reproduce the published preferred-direction angle for every frozen Figure 4 recording;
2. fold each angle by cell-body hemisphere so ipsilateral is positive;
3. aggregate with the circular mean;
4. report a 95% percentile bootstrap interval over recordings using exactly 10,000 resamples and seed 0.

Absolute firing-rate gain and the heterogeneous transient/tonic response time course are not transferred in v1.

This bootstrap rule is a **Program A analysis choice**, not a claim that the source paper used bootstrapping. It is frozen now, before source extraction or navigation performance, so uncertainty handling cannot be selected post hoc.

## Why source extraction is still blocked

The Dryad deposit is approximately 117.49 GB and is stored as a multipart archive (`Currier2020.z01`, `Currier2020.zip.001` through `.013`) plus a 2.55 KB `Currier2020README.rtf`.

The public repository/mirror records the README as Dryad file-stream `536042` with MD5 `91e5213503788fcde11c0f5aa3e91f43`. We have not yet independently retrieved those bytes and computed SHA-256, so the committed contract remains `BLOCKED_README_SHA256_UNVERIFIED`.

The README/archive directory map has also not yet been content-addressed, so we do not know which minimal archive members contain the 12 Figure 4 recordings. This is `BLOCKED_ARCHIVE_MEMBER_MAP_UNRESOLVED`.

Finally, the publication gives the count and inclusion rule but the exact archived recording/session identifiers have not yet been mapped to the Figure 4 set. This is `BLOCKED_FIGURE4_RECORDING_SET_UNRESOLVED`.

The default next step is therefore **not** to download the full 117.49 GB release. It is to retrieve and hash the tiny README, resolve the archive member map, identify the exact 12 Figure 4 recording assets, and download only the minimum byte set that can reproduce their directional responses. If the split-archive format makes selective extraction impossible, that limitation must be demonstrated and recorded rather than assumed.

## Transfer boundary

The physiology is direct evidence for the published ventral P-FN preparation. It is not exact-body physiology for MaleCNS v1.0 PFNa/PFNm/PFNp cells.

The allowed Program A statement is that the published ventral P-FN physiology supplies a **cross-dataset, type-level directional tuning prior**. The structural/type mapping into MaleCNS remains a separate evidence layer.

Navigation performance may not select archive members, recordings, normalization, uncertainty, hemisphere convention, or target statistic.
