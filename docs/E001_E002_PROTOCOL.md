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
- MB/LH and fan-shaped-body tangential odor-pathway candidates;
- wind-sensitive PFN families, especially MaleCNS homologues corresponding to PFNa/PFNp/PFNm-like functional pathways;
- `hDeltaC` as the primary odor × wind integration hypothesis;
- additional local FB populations only when supported by v1.0 connectivity;
- `PFL3` and bilateral `DNa02` as steering-output candidates.

The 2026 evidence-integration work introduces a separate **memory hypothesis**: persistent local-FB circuitry can bridge odor blanks over several seconds. Candidate recurrence associated with this computation should be traced and tested separately rather than silently folded into the hDeltaC steering hypothesis.

### E001-D — corridor extraction

Run bounded forward/reverse tracing from reviewed sensory candidates to reviewed steering targets. Preserve:

- exact body IDs;
- source and target type/instance labels;
- side labels;
- raw edge weights;
- forward and reverse hop depth;
- extraction thresholds;
- upstream file hashes.

Every retained intermediate population must be reviewable by name. An anonymous multi-hop subgraph does not qualify.

### E001-E — sign provenance

Attach presynaptic transmitter-derived signs conservatively. ACh is modeled excitatory and GABA inhibitory in v0. Glutamate, dopamine, serotonin, octopamine, and unresolved transmitter annotations remain sign `0` until a stronger receptor-level assumption is justified. Report signed and unresolved fractions.

## E002: modeled circuit sanity

E002 asks whether our explicit rate-model assumption can propagate biologically anchored inputs through the extracted topology. It does **not** claim that the rates reproduce in-vivo physiology.

Run:

```bash
fly-sniff-qualify data/cache/candidate-v0 \
  --output results/e002/qualification.json
```

The v0 report requires all of the following gates:

1. non-empty bilateral odor and steering roles;
2. non-overlapping left/right role sets;
3. complete body-ID closure;
4. a preregistered minimum resolved-sign fraction;
5. structural reachability from sensory/wind seeds to both steering outputs;
6. distinguishable mirrored odor perturbations;
7. opposite steering signs under mirrored perturbations;
8. near-zero steering after bilateral steering-input lesion;
9. exact-seed deterministic replay;
10. measurable persistence through a short odor blank.

The blank test is motivated by 2026 evidence that local fan-shaped-body activity integrates odor encounters over several seconds and can persist after odor loss during plume navigation. The E002 threshold is only a **model sanity floor**; it is not fitted to or presented as the biological time constant.

## Promotion rule

`fly-sniff-qualify` never changes `qualification_status` itself. Human review must inspect the report, exact body IDs, type labels, signs, unresolved edges, and pathway composition. Only then may a separate sealed manifest be written with:

```json
{"qualification_status": "qualified"}
```

A candidate that fails is retained as a negative result. Do not tune the frozen final benchmark to rescue it.

## References

- Matheson et al., *A neural circuit for wind-guided olfactory navigation*, Nature Communications 13, 4613 (2022). https://doi.org/10.1038/s41467-022-32247-7
- Kathman et al., *Neural dynamics for working memory and evidence integration during olfactory navigation in Drosophila*, Nature Communications 17, 9082 (2026). https://doi.org/10.1038/s41467-026-75945-2
- MaleCNS Cell Type Explorer: https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/
