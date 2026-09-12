# E001/E002 protocol: from public MaleCNS data to a runnable candidate circuit

This protocol prevents a type name, a pretty path, or a literature diagram from silently becoming a benchmark controller.

## E001: structural authority

### E001-A — release lock

Use only Janelia MaleCNS `v1.0` for the primary graph. Record SHA-256 hashes of the annotation table and connection-weight table before discovery.

### E001-B — release-specific steering anchor

The current MaleCNS Cell Type Explorer establishes a type-level PFL3 → DNa02 structural anchor and bilateral DNa02 output population. The checked-in file `authority/malecns-v1.0-type-evidence.json` records this evidence. It is **not** a body-ID manifest.

### E001-C — sensory and navigation discovery

Discover exact v1.0 body IDs for candidate populations. Literature constrains what to look for, but does not authorize copying IDs or edges from another connectome.

The initial search families are:

- olfactory sensory / antennal-lobe candidates;
- MB/LH and fan-shaped-body tangential olfactory-pathway candidates;
- wind-sensitive ventral-PFN families, especially MaleCNS types corresponding to `PFNa`, `PFNp`, and `PFNm` functional pathways;
- `hDeltaC` as a **structural convergence hypothesis** because earlier connectomics report direct FB5AB and PFN input;
- additional local FB populations only when supported by v1.0 connectivity;
- `PFL3` and bilateral `DNa02` as steering-output candidates.

Do **not** interpret the hDeltaC search role as evidence that hDeltaC itself has been functionally shown to compute odor-gated wind direction. The 2024 addendum to Matheson et al. reports that the `VT062617` line used for the original sensory-response and behavioral experiments also or perhaps predominantly labels `hDeltaK`. Those functional results therefore cannot be assigned specifically to hDeltaC. The direct FB5AB/PFNa → hDeltaC connectomic observation remains a valid structural prior.

Direct ventral-PFN recordings support a side-resolved airflow basis: tuning is approximately 45° ipsilateral and depends on the hemisphere containing the cell body. Consequently, `somaSide` is the primary laterality authority for PFN role review. `rootSide` and `instance` are weaker anatomical fallbacks and must remain explicitly identified as such.

The 2026 evidence-integration work introduces a separate **memory hypothesis** centered on hDeltaK-dominated local-FB activity and recurrence with PFG. This hypothesis can motivate a distinct odor-blank persistence analysis, but it cannot rescue a missing hDeltaC structural corridor or be silently folded into the steering hypothesis.

### E001-D — corridor extraction

Run bounded forward/reverse tracing from reviewed sensory candidates to reviewed steering targets. Preserve:

- exact body IDs;
- source and target type/instance labels;
- side labels and the evidence source used for side inference;
- raw edge weights;
- forward and reverse hop depth;
- extraction thresholds;
- upstream file hashes.

Every retained intermediate population must be reviewable by name. An anonymous multi-hop subgraph does not qualify.

### E001-E — sign provenance

Attach presynaptic transmitter-derived signs conservatively. ACh is modeled excitatory and GABA inhibitory in v0. Glutamate, dopamine, serotonin, octopamine, and unresolved transmitter annotations remain sign `0` until a stronger receptor-level assumption is justified. Report signed and unresolved fractions.

This is a model convention, not a claim that transmitter prediction proves the postsynaptic effect of every MaleCNS edge. Predicted transmitter identity and edge-specific physiological sign are different evidence classes.

## E002: modeled circuit sanity

E002 asks whether our explicit rate-model assumption can propagate biologically anchored inputs through the extracted topology. It does **not** claim that the rates reproduce in-vivo physiology.

Run:

```bash
fly-sniff-qualify data/cache/candidate-v0 \
  --output results/e002/qualification.json
```

### Core steering qualification

The candidate steering circuit must pass all ten core gates:

1. non-empty bilateral odor and steering roles;
2. non-overlapping left/right role sets;
3. complete body-ID closure;
4. a preregistered minimum resolved-sign fraction;
5. structural reachability from sensory/wind seeds to the left steering output;
6. structural reachability from sensory/wind seeds to the right steering output;
7. distinguishable mirrored odor perturbations;
8. opposite steering signs under mirrored perturbations;
9. near-zero steering after bilateral steering-input lesion;
10. exact-seed deterministic replay.

Passing these gates qualifies only **modeled steering propagation sanity**. It does not establish biological dynamics, prove hDeltaC-specific physiology, or prove the functional identity of a candidate sensory pathway.

### Separate memory hypothesis

`fly-sniff-qualify` also reports `blank_bridge_memory_hypothesis`, which measures modeled directional persistence through a short odor blank. This result is deliberately **nonblocking** for core steering qualification.

The separation is important. 2026 work supports several-second odor-evidence integration and persistent local fan-shaped-body dynamics during turbulent plume navigation, with hDeltaK-dominated activity implicated in this process. A candidate can therefore pass E002 steering while failing the memory hypothesis. Conversely, a persistent candidate does not become a valid steering circuit merely because it bridges a blank.

The v0 blank threshold is a model sanity floor, not a fit to or claim about the biological time constant. Persistent-circuit body IDs, recurrence, signs, and lesion effects require their own structural qualification before any memory claim.

## Promotion rule

`fly-sniff-qualify` never changes `qualification_status` itself. Human review must inspect the report, exact body IDs, type labels, signs, unresolved edges, pathway composition, and literature-evidence class. Only then may a separate sealed manifest be written with:

```json
{"qualification_status": "qualified"}
```

A candidate that fails is retained as a negative result. Do not tune the frozen final benchmark to rescue it.

## References

- Currier, Matheson & Nagel, *Encoding and control of orientation to airflow by a set of Drosophila fan-shaped body neurons*, eLife 10:e61510 (2021). https://doi.org/10.7554/eLife.61510
- Matheson et al., *A neural circuit for wind-guided olfactory navigation*, Nature Communications 13, 4613 (2022). https://doi.org/10.1038/s41467-022-32247-7
- Matheson et al., *Addendum: A neural circuit for wind-guided olfactory navigation*, Nature Communications 15, 1903 (2024). https://doi.org/10.1038/s41467-024-46225-8
- Rayshubskiy et al., *Neural circuit mechanisms for steering control in walking Drosophila*, eLife 13:RP102230 (2025). https://doi.org/10.7554/eLife.102230.3
- Kathman et al., *Neural dynamics for working memory and evidence integration during olfactory navigation in Drosophila*, Nature Communications 17, 9082 (2026). https://doi.org/10.1038/s41467-026-75945-2
- MaleCNS Cell Type Explorer: https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/
