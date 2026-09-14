# Video-to-neural scientific contract v0

## Primary question

Can recent observed fly behavior predict **measured neural population activity** in a held-out animal, and do richer behavioral representations improve that prediction?

Milestone 0 deliberately stops there. It does not claim that behavior uniquely determines the full-brain state or that a complete MaleCNS activity trajectory has been reconstructed.

## Frozen first benchmark

- Primary paired dataset: MC2P.
- Unit of independence: animal, not frame or sliding window.
- Default history target: up to 3 s observed behavior predicting the next 0.5 s measured neural population state.
- Train/validation/test assignment must be animal-disjoint.
- A session may not straddle splits.
- Input and target intervals may not overlap.
- Hyperparameters may use train and validation animals only.
- Test animals are consumed only for the final model comparison.

## Epistemic classes

`MEASURED_VIDEO`, `MEASURED_POSE`, `MEASURED_NEURAL_ACTIVITY`, `MEASURED_CONNECTIVITY`, `INFERRED_BEHAVIOR_STATE`, `MODELED_LATENT_NEURAL_STATE`, and `VIEWER_ONLY` are distinct. Renderers and reports must never label `MODELED_LATENT_NEURAL_STATE` as measured, recorded, or actual neural activity.

## Milestone 0 model comparison

The intended progression is:

1. pose-only representation → measured neural target;
2. video-only representation → measured neural target;
3. video + pose → measured neural target;
4. video + pose + measured sensory/context channels when available;
5. only after 1–4 are established, a separately versioned connectome-prior model.

The connectome prior is not allowed to change data splits or target definitions.

## Metrics

Report target-wise Pearson correlation and R² on held-out animals, with aggregate mean/median values. Preserve per-target results and stratify later by behavior state. A constant train-mean predictor is the minimum baseline.

## Claims

Allowed after a successful Milestone 0 result:

> A model predicted measured neural population activity from held-out fly behavior video.

Not allowed:

> We reconstructed the fly's complete brain from video.

Whole-connectome visualization, when added later, must be labeled as a posterior/model state constrained by observations and structural priors, not ground-truth firing.
