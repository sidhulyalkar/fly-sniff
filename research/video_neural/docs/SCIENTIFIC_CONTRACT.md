# Video-to-neural scientific contract

## v0 is preserved

`benchmark_v0.json` records the original animal-held-out proposal. It remains immutable historical context rather than being silently rewritten.

Before any real benchmark score was inspected, the design was red-teamed against the MC2P measurement geometry. Raw two-photon images are subject-specific: an unseen animal changes both dynamics and the anatomical image coordinate system. Directly regressing raw pixels in an unseen animal therefore mixes two questions.

`benchmark_v1.json` records that correction explicitly as `pre_data_scoring_correction` with `revision_was_informed_by_benchmark_scores = false`.

## v1 primary question

Can 3 s of recent observed behavior predict the next 0.5 s of **measured neural activity** in the same fly on a held-out session?

- The unit of independence is session within animal.
- At least three sessions are required for an animal to participate in train/validation/test evaluation.
- Input and future target windows never overlap.
- Test sessions never select features, architecture, or hyperparameters.
- The raw dF/F image is an allowed target because the anatomical coordinate system is fixed within that animal.
- Models are fit/selected within animal and only then aggregated across animals.

## v1 secondary question

Can behavior predict a **subject-invariant measured-neural representation** in an entirely unseen animal?

This lane remains `blocked_pending_neural_representation_contract`. Raw neural pixels are forbidden as its target. Unlocking it requires freezing a cross-animal comparable measured-neural representation without using held-out decoding performance to choose that representation.

## Measurement and synchronization contract

The MC2P acquisition basis is frozen as 100 Hz behavior video and nominal 16 Hz two-photon imaging. The synchronization authority is the behavior frame associated with each neural frame by closest timestamp. Converted index arrays are an execution artifact of that authority, not a new measurement.

The public release includes legacy Python pickle artifacts. Discovery never deserializes them. `convert-mc2p-legacy` requires explicit `--trust-upstream-pickle`, hashes source and output bytes, emits a conversion receipt, and writes NumPy arrays that are subsequently loaded with `allow_pickle=False`. Opcode inspection is recorded for provenance only and is not presented as a security guarantee.

## Epistemic classes

`MEASURED_VIDEO`, `MEASURED_POSE`, `MEASURED_NEURAL_ACTIVITY`, `MEASURED_CONNECTIVITY`, `INFERRED_BEHAVIOR_STATE`, `MODELED_LATENT_NEURAL_STATE`, and `VIEWER_ONLY` remain distinct. Renderers and reports must never label `MODELED_LATENT_NEURAL_STATE` as measured, recorded, or actual neural activity.

## Model progression

1. deterministic pose features → measured neural target;
2. video representation → measured neural target;
3. video + pose → measured neural target;
4. video + pose + measured sensory/context channels when available;
5. separately versioned connectome-prior model only after the measured-neural baselines are established.

The connectome prior may not change target definitions or evaluation splits.

## Claim boundary

A successful primary result may support:

> Behavior predicts future measured neural activity within a fly on held-out sessions.

It does not support:

> We reconstructed an unseen fly's complete brain from video.

Whole-connectome visualization, if added later, is a modeled posterior constrained by observations and structural priors, not ground-truth firing.
