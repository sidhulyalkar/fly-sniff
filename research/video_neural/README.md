# Fly Video → Neural State research lane

This directory is an intentionally isolated research prototype. It is **not** part of the active `fly-sniff` odor-navigation controller or any sealed MaleCNS experiment.

## Concrete first goal

> Given 3.0 s of recent Drosophila behavior, can a low-complexity pose decoder predict the next 0.5 s of **measured dF/F** in the same animal on a held-out session better than prespecified temporal-misalignment controls?

The first paired benchmark is the eight-animal public MC2P release. Large behavior-only datasets are registered for possible representation pretraining; they are never treated as neural ground truth.

## Why v1 is within animal

`benchmark_v0.json` preserves the original proposal to predict raw future neural pixels in unseen animals. Before any real benchmark score was inspected, that design was rejected as the active primary task because raw two-photon image coordinates are subject-specific. An unseen-animal raw-pixel model would confound neural-dynamics prediction with unobserved anatomy.

`benchmark_v1.json` therefore freezes the active primary task as **within-animal, held-out-session** prediction. The unseen-animal question remains blocked until a measured, subject-invariant neural representation is frozen independently of held-out task performance.

## Canonical v1 workflow

The scored v1 workflow has four operational stages:

1. **PREPARE** deterministic measured-data artifacts and freeze the split.
2. **PREFLIGHT** verify the complete prepared provenance chain, exact sample/session split, strict-future window metadata, and artifact hashes without deserializing any pose-neural NPZ batch.
3. **DEVELOP** deserialize train/validation batches only, run QC, select ridge α, run the temporal-null ensemble, freeze the confirmatory animal population, and decide whether the one-way final evaluation is allowed.
4. **FINAL** verify all frozen evidence, exactly replay train+validation development evidence without opening test batches, write the irreversible consumption marker, reopen the prepared held-out batches, and evaluate once.

Low-level conversion, window, batch, and split commands remain useful for inspection and tests, but they are not an alternative scored v1 workflow.

### Phase 1: PREPARE

```bash
MC2P_ROOT=/path/to/MC2P
RUN=/path/to/mc2p-v1-run

fly-video-neural-prepare "$MC2P_ROOT" \
  --output "$RUN/prepared" \
  --trust-upstream-pickle
```

PREPARE requires the complete eight-animal release, discovers every session, converts trusted legacy synchronization and 3D-pose pickle artifacts to safe NumPy arrays, checks behavior-frame agreement, and creates deterministic 3.0 s input / 0.5 s target windows. Because nearest-timestamp alignment can map adjacent behavior frames to the same slower neural frame, every target neural index is required to be **strictly later than the neural index aligned to the final input behavior frame**.

The pose representation uses MC2P's published six group-root joint partition as its grouping convention. Applying that grouping coordinate-wise to released 3D pose and summarizing each input window with mean, standard deviation, first/last displacement, and mean velocity is a `fly-sniff` engineering transform, not an upstream MC2P method.

The measured neural target is the mean dF/F image over the strictly future mapped neural frames. PREPARE prefers the released 64×64 resized measured dF/F when available, otherwise the 128×128 measured dF/F. It fits no model and reports no score.

PREPARE also writes the exact batch hashes, split-lock bytes, per-session provenance, strict-future boundary policy, and ingestion-code fingerprint. A non-empty destination is rejected.

### Phase 2: PREFLIGHT

```bash
fly-video-neural-preflight "$RUN/prepared" \
  --output "$RUN/preflight.json"
```

PREFLIGHT is a provenance-only audit performed before DEVELOPMENT. It verifies the preparation, manifest, split-lock, conversion, window, and session-batch receipt chains; exact file SHA-256 values; the complete eight-animal/session inventory; per-animal train/validation/test partition structure; exact sample-to-split identities reconstructed from `windows.json`; frozen 3.0 s / 0.5 s / 0.5 s timing; input/target behavior-frame boundaries; strict-future target neural indices; input-only pose features; and the measured dF/F target geometry.

PREFLIGHT deliberately does **not** deserialize `pose-neural-batch.npz`, fit a model, inspect a model metric, or consume a held-out test result. Its report is self-hashed and records the auditor source SHA-256. Adversarial tests cover raw byte tampering plus rehashed split, window, conversion, and modeling-claim tampering.

If PREFLIGHT fails, stop. Do not run DEVELOPMENT on that prepared tree. Repairing a software/provenance defect is allowed only by producing a new clean PREPARE directory; do not mutate a partially audited prepared run in place.

### Phase 3: DEVELOP

```bash
fly-video-neural-develop \
  "$RUN/prepared/session-split-lock.json" \
  "$RUN"/prepared/sessions/*/pose-neural-batch.npz \
  --preparation-receipt "$RUN/prepared/preparation-receipt.json" \
  --qc-config configs/data_qc_v1.json \
  --acceptance-config configs/validation_acceptance_v1.json \
  --output "$RUN/development"
```

The split seed is `2701`. Each animal contributes exactly one validation session and one test session; all remaining sessions train. At least three sessions are required per animal.

After the split is frozen, DEVELOPMENT SHA-authenticates every prepared session batch but **does not deserialize the held-out test-session NPZ arrays**. It loads exactly the train+validation projection. Tests fail if even one test sample contaminates that projection, and a monkeypatched regression fails if NumPy attempts to open any frozen test batch during DEVELOPMENT.

DEVELOPMENT independently rechecks the PREPARE receipt, split bytes, strict-future policy, ingestion fingerprint, and exact supplied batch hashes. PREFLIGHT is therefore an additional provenance audit, not a substitute for the development firewall.

The aligned model is ridge regression with α in `[0.01, 0.1, 1, 10, 100]`, selected independently per animal using validation median Pearson correlation. The primary score is deliberately simple: median per-pixel Pearson correlation over the measured dF/F image. No neural-support mask is introduced in v1, so background or low-information pixels may reduce sensitivity. That limitation is frozen rather than repaired after seeing scores.

The temporal control is an ensemble of deterministic within-session circular pose shifts at **25%, 50%, and 75%** of each session. Each shift receives the same α grid. For each animal, the null fraction with the strongest validation median Pearson correlation is frozen as that animal's primary final comparator. FINAL may not choose a comparator from test performance.

The validation unlock requires all structural/QC contracts to pass, at least **six eligible animals**, a strict majority of positive aligned-minus-selected-null validation effects, and a positive median paired validation effect. A non-computable validation metric makes that animal ineligible rather than assigning it an artificial score.

Crucially, `validation-unlock.json` freezes the **exact validation-eligible animal IDs** as the final confirmatory population before any test batch is reopened. Validation-ineligible animals can later be reported descriptively but can never enter the confirmatory statistic.

Stop unless `development/validation-unlock.json` reports `unlocked_for_single_test_consumption`.

### Phase 4: FINAL

```bash
fly-video-neural-final \
  "$RUN/prepared/session-split-lock.json" \
  "$RUN/development" \
  "$RUN"/prepared/sessions/*/pose-neural-batch.npz \
  --acceptance-config configs/validation_acceptance_v1.json \
  --output "$RUN/final"
```

Before reopening any prepared held-out batch arrays, FINAL verifies the development receipt, QC and validation-unlock self-hashes, exact source-batch hashes, split-lock bytes, acceptance-config bytes, critical implementation fingerprint, and Python/NumPy runtime. It then loads **only the authenticated train+validation session batches**, reruns the aligned decoder and temporal-null development analyses, and requires exact reproduction of the saved development evidence. A replay mismatch fails before the final namespace is consumed.

Only after all test-free checks succeed does FINAL create `FINAL_TEST_CONSUMED.json` using exclusive-create semantics, with reopening after failure forbidden. Only after that marker does the supported workflow reopen the full prepared batch set and evaluate held-out sessions.

PREPARE necessarily reads the raw data to deterministically create every session batch before the split is evaluated. The stronger post-split claim is therefore precise: PREFLIGHT never opens NPZ arrays, DEVELOPMENT does not reopen held-out target arrays, and FINAL does not reopen them until the pre-consumption development replay has succeeded and the one-way marker has been written.

## Prespecified final inference

The biological inferential unit is the **animal**, not the 4,096 image pixels and not the overlapping time windows. `configs/validation_acceptance_v1.json` is supplied to DEVELOPMENT and hash-bound into its receipt before validation results are available. It freezes the final rule:

- the confirmatory population is exactly the animal IDs that were validation-eligible at unlock time;
- at least 6 confirmatory animals are required;
- every validation-eligible animal must have a computable paired held-out effect for a supportive result;
- effect = aligned test median Pearson r minus the validation-selected null's test median Pearson r;
- validation-ineligible animals are descriptive-only and cannot enter the final statistic;
- missing or non-computable confirmatory effects cannot shrink the denominator;
- zero effects count as non-positive;
- median paired test effect must be positive;
- an exact one-sided sign test against positive-effect probability 0.5 must satisfy `p <= 0.05`.

For a complete eight-animal confirmatory population, this requires at least 7/8 positive effects (`p = 9/256 ≈ 0.0352`). Six of eight is not close enough (`p = 37/256 ≈ 0.1445`). For a frozen six-animal confirmatory population, all 6/6 must be positive (`p = 1/64 ≈ 0.0156`). If an eight-animal confirmatory population later has only six computable test effects, it is **incomplete**, not a new 6/6 experiment.

A valid run can therefore finish as `supports_prespecified_predictive_generalization`, `does_not_meet_prespecified_support_rule`, `insufficient_prespecified_test_population`, or `incomplete_prespecified_test_population` without changing any threshold after test inspection.

## Claim boundary

A supportive v1 result would mean that this prespecified pose decoder predicts held-out-session measured dF/F better than its validation-selected temporal-misalignment control with consistent direction across the frozen confirmatory animals under the prespecified rule.

It would **not** establish behavior causing the neural activity, spike-level prediction, whole-brain reconstruction, an unseen-animal raw-pixel decoder, or a connectome mechanism. A negative, insufficient, or incomplete result is preserved as a scientific result, not repaired by changing the metric, null, split, horizon, support mask, population, or significance rule.

No real MC2P benchmark result has been consumed yet. A video foundation model remains intentionally deferred until this measurement-grounded baseline has produced an auditable outcome.
