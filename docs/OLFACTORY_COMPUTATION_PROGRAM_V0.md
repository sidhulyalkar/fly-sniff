# Olfactory Computation Program v0

## North star

**Which computations let Drosophila transform unreliable chemical evidence into robust odor detection, behavioral meaning, and source-directed action, and which of those computations transfer to engineered smell sensors?**

This lane is a publication-oriented research program, not a new demo controller. It consumes validated structural, physiological, behavioral, plume, and sensor artifacts through the repository's Experiment + Evidence Kernel. It must remain informative if the biological topology has no measurable advantage.

## Central thesis

The connectome specifies possible information-flow paths, not in-vivo dynamics. The study therefore separates four ingredients:

1. **receptor evidence** — measured odor/receptor response functions;
2. **wiring** — independently identified cell types and synaptic topology;
3. **dynamics** — measured or explicitly modeled synaptic/cellular dynamics;
4. **behavior** — held-out detection, valence, conflict, and navigation outcomes.

The main question is not whether a connectome simulation can be trained to perform an odor task. It is whether independently constrained biological wiring and dynamics explain or improve held-out olfactory computation relative to matched controls.

## Five aims

### O001 — geosmin specialist-channel calibration

Reconstruct the independently established Or56a → DA2 geosmin channel, audit its connectomic continuation, and verify that the model reproduces calibration phenomena without fitting to the headline behavioral benchmark.

**O001 is not a topology-advantage claim.** Because geosmin is detected by an unusually selective receptor channel, receptor-only models are expected to be strong. Failure to beat a receptor-only detector on simple geosmin presence/absence is not a failure of the program.

Calibration targets include receptor selectivity/dose response, DA2 pathway identity, measured response dynamics where available, and qualitative perturbation direction from published experiments.

### O002 — multi-odor representation benchmark

Build a provenance-resolved odor panel spanning attractive, aversive, and neutral/control odorants and mixtures. Quantify how odor identity, concentration, and valence become decodable across receptor, glomerular/PN, lateral-horn, and later circuit representations.

This aim distinguishes simple receptor separability from circuit-derived computation.

### O003 — odor-conflict and temporal-computation benchmark

The first candidate flagship mechanistic task is **behavioral conflict**, not easy single-odor detection. Test whether independently constrained real wiring plus physiology predicts how an aversive geosmin channel modifies responses to otherwise attractive odors, and whether target-specific transient/sustained dynamics are necessary to explain time-resolved behavior.

Primary comparisons must include receptor-only, topology-only/static, wiring+physiology, generic capacity-matched models, and matched topology nulls.

### O004 — plume evidence-to-action benchmark

Use validated plume recordings and directional odor cues to ask how temporal evidence, bilateral gradients/odor motion, persistence, and steering interact. Structural and physiology artifacts from existing fly-sniff lanes may be consumed only after their own gates pass.

Navigation performance is an evaluation target, never a calibration target for Program A latent-wiring claims.

### O005 — engineered sensor translation

Feed measured receptor-cell or electronic sensor signals into frozen transformations discovered in O001–O004. Evaluate detection, discrimination, mixtures, concentration shift, interferents, sensor drift, and plume/domain shift against conventional baselines.

O005 can establish engineering utility. It cannot retroactively validate a biological mechanism.

## Publication logic

A strong paper should establish a chain rather than a single leaderboard:

1. independently reproduce known olfactory physiology/circuit facts;
2. freeze a biological model without headline-task reward;
3. evaluate held-out odors/mixtures/temporal conditions;
4. compare intact topology to prespecified null families;
5. perform prespecified acute lesions;
6. quantify topology-level uncertainty across many independently generated null graphs;
7. generate novel predictions;
8. prospectively validate at least one nontrivial prediction experimentally when pursuing a top neuroscience journal;
9. separately test whether the resulting computation improves engineered sensing.

## Evidence hierarchy

Every claim-bearing parameter must be labeled as one of the repository kernel evidence classes. In particular:

- a FlyWire edge is **measured structure**, not measured physiology;
- transmitter prediction is not automatically synaptic sign/function;
- a published response from another fly/dataset is a **cross-dataset prior** unless directly matched;
- fitted time constants are **fitted parameters**;
- simulated activity is **modeled state**;
- assay outcomes are **behavioral output**.

## Required null hierarchy

Development may use small cohorts. A confirmatory topology claim requires at least 63 frozen null graphs where compute permits, never fewer than the kernel minimum of 31. Planned families are:

- directed degree-preserving rewires;
- degree + transmitter/sign-preserving rewires where sign evidence is defensible;
- hemisphere-constrained rewires;
- cell-type-constrained rewires;
- glomerulus-output identity shuffles;
- generic capacity-matched random/reservoir controls.

Nulls must preserve nuisance statistics tightly enough that performance differences cannot be explained by trivial graph density, degree, or parameter-budget differences.

## Primary anti-leakage rules

- no behavioral outcome or navigation reward in Program A physiology calibration;
- no topology-specific fitting for a latent-wiring claim;
- identical model capacity and fitting budget for topology comparisons;
- held-out odor identities/mixtures/concentrations declared before confirmatory evaluation;
- hidden final entropy and one-way final execution for confirmatory topology claims;
- final topology inference is over topology replicates, not over episodes treated as independent graphs;
- negative/blocked results remain first-class artifacts;
- no selector broadening or cell-role reassignment after performance inspection.

## Current state

This branch establishes the study contract and orchestration only. Confirmatory execution is intentionally **blocked** until every required evidence authority in `authority/olfactory-evidence-requirements-v0.json` is resolved and content-addressed.

## Existing-lane boundaries

- PR #22 remains the sensory/plume validation authority.
- PR #23 remains structure-only for its olfactory-motion candidate audit.
- Program B task optimization cannot be used as evidence for zero-shot biological computation.
- PR #26 remains the independent MC2P behavior→neural benchmark.
- This lane may consume qualified receipts from those lanes but must not import their provisional performance choices.
