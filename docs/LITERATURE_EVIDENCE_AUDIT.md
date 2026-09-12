# Literature evidence audit

This document is the scientific claim ledger for the current odor-navigation route. It separates release-specific MaleCNS anatomy, cross-dataset functional evidence, and explicit model assumptions so that one evidence class cannot silently substitute for another.

The audit is intentionally conservative. A failed or ambiguous biological claim is retained as a constraint rather than repaired by changing search thresholds, optimizer freedom, or the benchmark.

## Evidence classes

- **MaleCNS direct anatomy**: exact or aggregate observations from `male-cns:v1.0`.
- **Cross-dataset structural prior**: connectivity observed in a different connectome release or animal and used only to motivate a search.
- **Cross-animal functional prior**: physiology or perturbation evidence from other flies.
- **Model abstraction**: an engineered variable, transfer function, gain, sign rule, or role assignment used by this repository.
- **Insufficient/confounded**: evidence does not uniquely support the cell type or mechanism named in the claim.

## Current claim ledger

| Component | Best-supported interpretation | Evidence class | Main uncertainty | Allowed wording | Forbidden wording |
| --- | --- | --- | --- | --- | --- |
| FB5AB | Fan-shaped-body tangential input linked to olfactory-navigation pathways; earlier work found direct/indirect MBON and LHON routes converging onto FB5AB and activation promoted upwind orientation. | cross-animal functional + cross-dataset structural prior; MaleCNS type exists | exact MaleCNS sensory response is unmeasured | `literature-anchored olfactory-context input candidate` | `the MaleCNS FB5AB neurons detect odor` |
| Bilateral FB5AB model drive | Same nondirectional odor-context scalar is injected into left/right model roles. | model abstraction | biological amplitude, timing, adaptation and exact bilateral equality are unknown | `modeled bilateral nondirectional odor-context drive` | `measured FB5AB odor activity` |
| PFNa/PFNm/PFNp | Ventral PFNs encode airflow direction; direct recordings show approximately 45-degree ipsilateral tuning tied to cell-body hemisphere. | cross-animal functional prior | physiology is not measured for every MaleCNS PFN subtype/body ID | `side-resolved ventral-PFN airflow basis candidate` | `every selected PFN has identical wind tuning` |
| PFN laterality | `somaSide` is the strongest available side authority; `rootSide` and instance are fallbacks. | cross-animal functional prior + MaleCNS annotation | fallback side inference is anatomical, not physiological | `soma-side anchored laterality; fallback inferred when necessary` | `rootSide proves airflow preference` |
| hDeltaC | Earlier connectomics report direct FB5AB and PFN input, making hDeltaC a plausible structural convergence population. MaleCNS also contains corresponding type-level convergence. | cross-dataset structural prior + MaleCNS direct anatomy | cell-type-specific functional physiology is confounded | `hDeltaC structural convergence candidate` | `hDeltaC is experimentally proven to compute odor-gated wind direction` |
| hDeltaC functional response | The original `VT062617` sensory and perturbation results cannot be uniquely assigned to hDeltaC because the line also or perhaps predominantly labels hDeltaK. | insufficient/confounded | exact contribution of hDeltaC vs hDeltaK in those experiments | `original functional attribution is confounded by hDeltaK labeling` | citing the 2022 VT062617 result as hDeltaC-specific physiology without the addendum |
| hDeltaK/PFG persistence | 2026 work supports hDeltaK-dominated odor-evidence integration and persistence and motivates a separate recurrent memory hypothesis. | cross-animal functional + structural prior | exact MaleCNS body-ID recurrence and signs remain unqualified | `separate odor-blank persistence hypothesis` | using hDeltaK persistence to rescue a failed hDeltaC steering corridor |
| hDeltaC to PFL3 | Earlier central-complex work and the Matheson model support an indirect local-FB-to-PFL output route; hDeltaA/G/H/M were explicitly discussed as downstream local populations with strong PFL3 projections. | cross-dataset structural/model prior | which exact MaleCNS intermediates carry the audited route is an empirical E001 result | `bounded hDeltaC-to-PFL3 structural corridor with explicit intermediates` | `hDeltaC directly drives PFL3` unless exact qualified edges justify that narrower statement |
| PFL3 | Steering-related CX output population that combines heading and goal-related inputs and projects laterally toward descending steering circuitry. | cross-animal functional/connectomic prior + MaleCNS direct anatomy | the goal signal used by this model is not measured physiology | `steering-related PFL3 output candidate` | `PFL3 implements our exact steering equation in vivo` |
| PFL3 to DNa02 | MaleCNS contains a strong aggregate type-level structural anchor; independent work supports direct and indirect PFL3 influence on DNa02. | MaleCNS direct anatomy + cross-animal functional/connectomic prior | exact frozen body-ID edges still require candidate-graph extraction | `MaleCNS contains a type-level PFL3-to-DNa02 structural projection` | `every PFL3 synapse causally causes turning` |
| DNa02 | Bilateral high-gain steering readout candidate; direct recordings show ipsilateral DNa02 activity preceding/covarying with ipsilateral turning and bilateral activity difference tracking rotational velocity. | cross-animal direct physiology + MaleCNS direct anatomy | DNa02 is part of a larger motor network | `literature-supported bilateral steering readout` | `DNa02 alone controls turning` |
| Transmitter-derived sign | Presynaptic transmitter predictions provide a model sign convention where explicitly configured. | model abstraction anchored to MaleCNS annotation | receptor identity, compartment and effective postsynaptic sign are not measured per edge | `transmitter-derived modeled sign` | `ACh prediction proves an excitatory physiological edge` |

## Literature corrections that materially affect the project

### 1. The 2024 hDeltaC addendum is a hard claim boundary

Matheson et al. (2024) report that `VT062617`, used for the original odor/wind physiology and behavioral perturbations, also or perhaps predominantly labels hDeltaK. Therefore the original functional response cannot be treated as hDeltaC-specific. This does **not** erase the direct FB5AB/PFNa to hDeltaC connectomic observation.

Repository consequence: hDeltaC remains in the frozen structural search, but only as a structural convergence hypothesis. No search threshold, role membership, or benchmark requirement was changed because of this correction.

### 2. PFN side assignment has unusually good physiological support

Currier et al. directly recorded ventral PFNs and found tuning around 45 degrees ipsilateral, tied to the hemisphere containing the cell body. This supports the existing decision to prioritize `somaSide` over `rootSide` and instance-derived side.

Repository consequence: unresolved PFNs remain excluded from side-specific training roles rather than having their side invented by the optimizer.

### 3. A multi-hop hDeltaC to PFL3 stage is more defensible than a direct-edge assumption

Matheson et al. explicitly modeled local FB populations downstream of hDeltaC before PFL output and named hDeltaA, hDeltaG, hDeltaH, and hDeltaM as examples with strong PFL3 projections in the earlier central-complex connectome. The E001 stage therefore correctly allows bounded intermediates and must report them rather than collapsing them into a fictional direct hDeltaC to PFL3 edge.

This literature expectation is **not** a post-hoc whitelist. The frozen E001 search must not be retuned to force those types to appear.

## Highest-priority unresolved literature/anatomy questions

1. **Exact MaleCNS body-ID role audit.** Review the generated role artifact for each FB5AB, PFN and DNa02 assignment, including `somaSide`, fallback evidence source, neurotransmitter prediction/confidence and membership continuity.
2. **Exact hDeltaC-to-PFL3 intermediate composition.** Compare retained MaleCNS intermediates against prior local-FB motifs without changing the preregistered search plan. Unexpected intermediates are results, not automatic failures or invitations to tune.
3. **Upstream olfactory specificity.** The Matheson route supports MBON12, MBON13, MBON19 and AD1b2 as literature priors, but the first E001 stage must still show whether the exact MaleCNS release contains a compact body-ID-resolved route to FB5AB.
4. **Effective sign uncertainty.** Quantify how much of the candidate GraphBundle depends on transmitter-derived sign assumptions versus unresolved edges. Do not translate transmitter confidence directly into physiological sign confidence.
5. **PFL3 laterality semantics.** Keep PFL3 anatomical side conventions separate from the DNa02 turn-sign convention. PFL3 naming by LAL projection and bridge/FB innervation can differ, so the final steering sign must be verified by perturbation rather than inferred from a label alone.
6. **Memory separation.** hDeltaK/PFG recurrence should remain a separate hypothesis with its own body-ID continuity, signs, lesion logic and blank-duration sensitivity.

## Primary references

- Currier TA, Matheson AMM, Nagel KI. *Encoding and control of orientation to airflow by a set of Drosophila fan-shaped body neurons.* eLife 10:e61510 (2021). https://doi.org/10.7554/eLife.61510
- Hulse BK et al. *A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection.* eLife 10:e66039 (2021). https://doi.org/10.7554/eLife.66039
- Matheson AMM et al. *A neural circuit for wind-guided olfactory navigation.* Nature Communications 13, 4613 (2022). https://doi.org/10.1038/s41467-022-32247-7
- Matheson AMM et al. *Addendum: A neural circuit for wind-guided olfactory navigation.* Nature Communications 15, 1903 (2024). https://doi.org/10.1038/s41467-024-46225-8
- Westeinde EA et al. *Converting an allocentric goal into an egocentric steering signal.* Nature (2024). https://doi.org/10.1038/s41586-023-07006-3
- Rayshubskiy A et al. *Neural circuit mechanisms for steering control in walking Drosophila.* eLife 13:RP102230 (2025). https://doi.org/10.7554/eLife.102230.3
- Kathman ND et al. *Neural dynamics for working memory and evidence integration during olfactory navigation in Drosophila.* Nature Communications 17, 9082 (2026). https://doi.org/10.1038/s41467-026-75945-2
- MaleCNS v1.0 project and Cell Type Explorer: https://male-cns.janelia.org/ and https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/
