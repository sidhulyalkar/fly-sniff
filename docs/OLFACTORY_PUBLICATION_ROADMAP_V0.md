# Publication Roadmap — Olfactory Computation Program v0

## Intended paper-level contribution

The target paper should not be framed as "a fly connectome simulation detects odors." The stronger contribution is:

> **A quantitatively constrained test of which components of Drosophila olfactory computation arise from receptor tuning, anatomical wiring, target-specific neural dynamics, and temporal evidence integration, with matched topology controls and prospective experimental validation.**

A second, explicitly separate contribution asks whether the discovered computation improves engineered odor sensing under distribution shift.

## Claim tiers

### Tier 0 — resource/methods

A reproducible evidence ledger, connectome subgraph, physiology contracts, null hierarchy, and benchmark suite. Valuable infrastructure, but not the headline biological result.

### Tier 1 — retrospective computational biology

Real topology + independently calibrated physiology predicts held-out published behavior better than receptor-only, static-topology, generic, and rewired controls. This can support a strong computational neuroscience paper if the datasets and controls are sufficiently broad.

### Tier 2 — mechanistic prediction

The model generates a prespecified prediction about a mixture, timing manipulation, pathway lesion, or cell-type intervention that is not used in calibration.

### Tier 3 — prospective experimental validation

A collaborator tests the frozen prediction in new animals/experiments. The preregistered primary endpoint agrees or disagrees with the model. A negative prospective result is retained.

Tier 3 is the preferred standard for submission to a top general/neuroscience journal.

## Central hypotheses

### H1 — specialist-channel calibration

The Or56a/DA2 pathway provides a high-specificity geosmin calibration case. Because receptor selectivity is already strong, simple geosmin presence/absence detection is expected to be solvable largely from receptor evidence and is **not** the flagship topology hypothesis.

### H2 — topology contributes to conflict computation

For odor combinations in which aversive and attractive drives compete, biological downstream wiring carries predictive information beyond receptor activation alone.

Primary test: paired held-out behavioral predictive performance of intact biological topology versus prespecified matched rewires under identical dynamics and parameter budgets.

### H3 — dynamics contribute beyond topology

Target-specific synaptic/cellular dynamics improve prediction of time-resolved responses beyond a static weighted graph with the same topology.

Primary test: wiring+physiology versus topology-only/static under held-out temporal waveforms and odor transitions.

### H4 — biological architecture generalizes under sensory shift

A biologically constrained model retains more useful performance than capacity-matched alternatives under prespecified concentration, mixture, interferent, and temporal/plume shifts.

This hypothesis must distinguish architectural advantage from extra biological supervision.

### H5 — transferable computation

Frozen transformations identified in biological experiments improve an independent engineered-sensor benchmark. This is an engineering claim and must be reported separately from H1–H4.

## Experimental units and pseudoreplication

The unit of biological replication must follow the source experiment: animal, trial nested in animal, or independently prepared sensor cartridge/batch as appropriate. Frames, pixels, synapses, repeated odor samples, and episodes must not be treated as independent biological replicates when they share an animal/topology/sensor preparation.

Topology inference is performed across independently generated topology-null graphs. Episode uncertainty is nested within topology, not substituted for topology replication.

## Model comparison ladder

Every flagship task should contain, where the available data support it:

1. receptor-only baseline;
2. receptor + generic linear/nonlinear baseline;
3. static biological topology;
4. biological topology + independently measured dynamics;
5. capacity-matched generic network;
6. biological topology with acute prespecified lesions;
7. multiple families of matched topology nulls.

If biological topology receives additional fitted degrees of freedom, the generic/null comparator must receive an equivalent budget or the comparison is not interpretable.

## Null design

Confirmatory topology inference should prefer 63 frozen null graphs. Required families include directed degree-preserving, cell-type-constrained, hemisphere-constrained, and glomerulus-output identity-shuffle controls. Sign/transmitter-preserving nulls are added only where sign evidence is sufficiently strong.

For each null family, report nuisance-statistic matching before reporting task performance: node count, edge count, in/out degree distributions, weight/strength distributions, hemisphere composition, relevant cell-type counts, and any spatial/neuropil constraints.

## Data splits

Development, validation/model-selection, and confirmatory conditions must be split by the highest-level independent unit available. The confirmatory condition set should include prespecified OOD dimensions such as unseen odor identities, unseen mixtures, unseen concentrations, different temporal statistics, or different plume regimes.

No confirmatory condition may be moved back into development after performance inspection.

## Prospective experiment candidates

The final choice must be made before confirmatory model evaluation. Candidate classes include:

- mixture-conflict threshold prediction: concentration/timing at which geosmin suppresses attraction;
- pulse-duration or inter-pulse interval prediction separating transient and sustained LHN pathways;
- acute lesion prediction identifying a downstream cell type/path expected to selectively alter transient versus sustained behavior;
- plume-regime prediction specifying when gradient versus odor-motion circuitry should dominate.

The prediction must include intervention, stimulus, primary endpoint, direction/magnitude target where justified, sample-size rationale, exclusion criteria, and frozen analysis.

## Statistics

Effect sizes and uncertainty are primary. Exact statistical tests depend on the data-generating experiment and must be frozen before confirmatory evaluation. Avoid treating enormous numbers of frames/synapses as n. Use hierarchical/bootstrap or mixed-effects models when repeated trials are nested within animals, and randomization/permutation inference across topology nulls for topology claims.

Multiple primary hypotheses require prespecified family-wise/FDR handling or a single designated primary endpoint with secondary endpoints labeled exploratory.

## Robustness analyses

Predeclare sensitivity to:

- connectome snapshot / edge threshold;
- uncertain cell-type membership;
- synaptic-weight transformation;
- transmitter/sign uncertainty;
- physiology parameter uncertainty;
- concentration normalization;
- temporal binning;
- lesion implementation;
- null-generation family;
- behavioral preprocessing.

A result that exists only under one arbitrary preprocessing or graph threshold is not a strong mechanistic result.

## Figure plan

1. **Biological question and evidence map**: odor → receptors → glomeruli/PN → LH/MB/navigation → behavior, with evidence class overlays.
2. **O001 calibration**: geosmin pathway identity, receptor/PN dynamics, calibration without headline-task fitting.
3. **Representation**: odor/concentration/valence geometry across stages.
4. **Flagship O003 result**: intact versus receptor-only/static/generic/matched rewires on held-out conflict/temporal behavior.
5. **Mechanism**: lesions and transient/sustained pathway predictions.
6. **Generalization/O004**: plume or sensory-statistics shift.
7. **Prospective validation**: frozen prediction versus new experiment.
8. **Translation/O005**: independent sensor benchmark, clearly separated from biological validation.

## Stop rules

- If evidence authorities cannot be resolved cleanly, retain the block.
- If O001 calibration fails, do not tune downstream topology to compensate.
- If intact topology does not beat matched nulls, report the null result and investigate representation/dynamics separately.
- If a result requires post-hoc selector broadening, new null families, changed thresholds, or new primary endpoints, version a new experiment rather than rewriting the old one.
- If prospective validation fails, preserve the failed prediction and update the model only in a new version.

## Journal-readiness checklist

A submission should not be called top-journal ready until it has: independent evidence provenance; exact reproducibility; topology-matched nulls; OOD evaluation; acute causal predictions; uncertainty/sensitivity analysis; a one-way confirmatory run; clear negative-result retention; and, preferably, prospective experimental validation of a novel prediction.
