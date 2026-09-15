# Video-to-neural scientific contract

## v0 is preserved

`benchmark_v0.json` records the original animal-held-out raw-neural-pixel proposal. It remains immutable historical context rather than being silently rewritten.

Before any real benchmark score was inspected, the design was red-teamed against the MC2P measurement geometry. Raw two-photon images are subject-specific: an unseen animal changes both neural dynamics and the anatomical image coordinate system. Directly regressing raw pixels in an unseen animal therefore mixes two questions.

`benchmark_v1.json` records the correction explicitly as `pre_data_scoring_correction` with `revision_was_informed_by_benchmark_scores = false`.

## v1 primary question

Can 3 s of recent observed behavior predict the next 0.5 s of **measured dF/F** in the same fly on a held-out session, better than prespecified within-session temporal-misalignment controls?

The unit of model fitting is the animal. The unit of held-out generalization is session within animal. The unit of final statistical inference is the animal.

At least three sessions are required for an animal to participate. Exactly one session is validation and one is final test; all remaining sessions train. The session split seed is `2701`. Test sessions never select features, architecture, α, temporal-null fraction, metric, or final inferential criterion.

The raw/resized dF/F image is allowed as a target because anatomical coordinates are stable within the same animal. No raw neural-pixel correspondence is assumed across animals.

## Strict future-target boundary

Behavior is sampled faster than two-photon imaging. Under closest-timestamp alignment, the first nominal target behavior frame can map to the same neural frame as the final input behavior frame even when the behavior-frame slices do not overlap.

For v1, every target neural index must therefore be **strictly greater than the neural index aligned to the final input behavior frame**. Windows that contain no neural frame satisfying that rule are dropped. The boundary policy is recorded in PREPARE artifacts and bound into the scored provenance chain.

## Pose representation boundary

MC2P's upstream preprocessing defines six group-root joint groups. v1 reuses that joint partition as the grouping convention.

Applying those groups coordinate-wise to released 3D `points3d`, then summarizing each 3 s input window with per-coordinate mean, standard deviation, endpoint displacement, and mean velocity, is a deterministic `fly-sniff` engineering representation. It must not be described as an upstream MC2P feature method.

Only input behavior frames contribute to pose features. No neural target value enters the pose representation.

## Measured-neural target and primary metric

Each target is the mean measured dF/F image over the strictly future mapped neural frames. PREPARE prefers the released 64×64 resized dF/F when present and otherwise uses measured 128×128 dF/F.

The v1 primary model-selection metric is median Pearson correlation across image pixels. Correlations that are not computable are represented as JSON `null`, never IEEE `NaN` or a favorable replacement value.

No neural-support/ROI mask is introduced in v1. Because the primary metric spans the full measured image, background or low-information pixels can reduce sensitivity. This is a frozen limitation, not a reason to redefine the metric after observing development or final scores.

## Temporal-misalignment control

The primary control is `circular_quartile_session_feature_shift_ensemble`: pose features are circularly shifted within each session by 25%, 50%, and 75% of the available prediction windows. Each shift preserves the session-level feature multiset while breaking contemporaneous behavior-neural pairing.

Every null fraction receives the identical ridge α grid `[0.01, 0.1, 1, 10, 100]`. For each animal, the fraction with the strongest validation median Pearson correlation is frozen as that animal's primary final comparator. Test data may never choose the null fraction.

The aligned model and every null are intentionally low complexity. More powerful video or multimodal models belong to later, separately versioned work only after the v1 measurement-grounded result exists.

## PREPARE → DEVELOP → FINAL provenance boundary

PREPARE deterministically discovers the complete eight-animal public release, converts explicitly trusted legacy pickle artifacts to safe NumPy arrays, materializes synchronized windows and session batches, freezes the session split, and hashes the derived artifacts. PREPARE fits no model and reports no performance.

PREPARE necessarily reads raw neural data from all sessions in order to create deterministic session batches before the split is evaluated. Therefore the scientifically accurate post-split firewall claim is not that test values were never physically read.

After the split is frozen, DEVELOPMENT:

- verifies the PREPARE receipt, ingestion-code fingerprint, split-lock bytes, and every supplied batch SHA;
- classifies session batches into train/validation/test using authenticated PREPARE metadata plus the frozen split lock;
- deserializes exactly the train+validation session batches;
- authenticates held-out test batch bytes without reopening their NPZ arrays;
- rejects any development projection that differs from the exact frozen train+validation sample set or contains even one held-out sample.

FINAL is the only supported v1 scoring path that reopens prepared held-out arrays after the split. Before doing so, it verifies the complete development authorization, exact source bytes, acceptance-config bytes, implementation fingerprint, and Python/NumPy runtime, then creates `FINAL_TEST_CONSUMED.json` with exclusive-create semantics. A crash after this marker does not permit reopening the same final namespace.

This is an operational/provenance barrier. It does not cryptographically prove that a human or unrelated process could never copy or inspect files outside the tooling.

## Validation unlock

`configs/validation_acceptance_v1.json` is frozen before development scoring. FINAL is not opened unless:

- development QC passes without test-array deserialization;
- at least **6 animals** have computable aligned and validation-selected-null validation metrics;
- aligned-minus-selected-null validation effect is positive in a strict majority of eligible animals;
- the median paired validation effect is positive;
- aligned and null reports remain test-locked and bind the same split.

Animals with non-computable validation median Pearson correlation are ineligible rather than assigned an artificial score. The unlock grants permission for one final evaluation; it is not itself evidence of held-out generalization.

## Prespecified final inference

The final confirmatory comparison is animal-level. Pixels and overlapping windows contribute to each animal's test score but are not independent biological replicates.

The same acceptance config supplied to DEVELOPMENT and hash-bound in its receipt freezes the final rule before validation scores can be used to rewrite it:

- at least **6 scorable animals** are required;
- per-animal effect = aligned test median Pearson r minus the test median Pearson r of that animal's validation-selected null fraction;
- positive effect means strictly `> 0`; zero counts as non-positive;
- the median paired effect must be `> 0`;
- an exact one-sided sign test against positive-effect probability 0.5 must satisfy `p <= 0.05`.

The exact small-sample consequences are intentional. With 8 scorable animals, 7/8 positive effects gives `p = 9/256 ≈ 0.0352`, while 6/8 gives `37/256 ≈ 0.1445`. With 7 scorable animals, 7/7 is required; 6/7 gives `p = 0.0625`. With 6 scorable animals, 6/6 gives `p = 1/64 ≈ 0.0156`.

A completed final run therefore receives one of three prespecified interpretations: `supports_prespecified_predictive_generalization`, `does_not_meet_prespecified_support_rule`, or `insufficient_scorable_animals`. A valid negative result is not a software failure and must not trigger post-test threshold, metric, null, mask, split, or horizon changes inside v1.

## v1 secondary question

Can behavior predict a **subject-invariant measured-neural representation** in an entirely unseen animal?

This lane remains `blocked_pending_neural_representation_contract`. Raw neural pixels are forbidden as its target. Unlocking it requires freezing a cross-animal comparable measured-neural representation without using held-out decoding performance to choose that representation.

## Measurement and synchronization contract

The MC2P acquisition basis is frozen as 100 Hz behavior video and nominal 16 Hz two-photon imaging. The synchronization authority is the behavior frame associated with each neural frame by closest timestamp. Converted index arrays are execution artifacts of that authority, not new measurements.

The public release includes legacy Python pickle artifacts. Discovery never deserializes them. Trusted conversion requires explicit `--trust-upstream-pickle`, hashes source and output bytes, emits conversion receipts, and writes NumPy arrays subsequently loaded with `allow_pickle=False`. Opcode inspection is provenance only and is not presented as a security guarantee.

## Epistemic classes and allowed claim

`MEASURED_VIDEO`, `MEASURED_POSE`, `MEASURED_NEURAL_ACTIVITY`, `MEASURED_CONNECTIVITY`, `INFERRED_BEHAVIOR_STATE`, `MODELED_LATENT_NEURAL_STATE`, and `VIEWER_ONLY` remain distinct. Renderers and reports must never label modeled latent state as measured, recorded, or actual neural activity.

If and only if the final support rule passes, v1 may support wording of the form:

> Under the prespecified v1 protocol, recent pose predicted held-out-session measured dF/F better than validation-selected temporal-misalignment controls with consistent direction across animals.

It does not support causal influence of behavior on neural activity, spike-level prediction, reconstruction of an unseen fly's brain, whole-connectome firing inference, or a connectome mechanism.

No real MC2P benchmark score has been consumed at the time this contract is frozen.
