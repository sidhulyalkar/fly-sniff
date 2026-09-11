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

Passing these gates qualifies only **modeled steering propagation sanity**. It does not establish biological dynamics or prove the functional identity of a candidate sensory pathway.

### Separate memory hypothesis

`fly-sniff-qualify` also reports `blank_bridge_memory_hypothesis`, which measures modeled directional persistence through a short odor blank. This result is deliberately **nonblocking** for core steering qualification.

The separation is important. 2026 work supports several-second odor-evidence integration and persistent local fan-shaped-body dynamics during turbulent plume navigation, but the responsible recurrent population need not be identical to the hDeltaC/PFL3 steering pathway. A candidate can therefore pass E002 steering while failing the memory hypothesis. Conversely, a persistent candidate does not become a valid steering circuit merely because it bridges a blank.

The v0 blank threshold is a model sanity floor, not a fit to or claim about the biological time constant. Persistent-circuit body IDs, recurrence, signs, and lesion effects require their own structural qualification before any memory claim.

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
