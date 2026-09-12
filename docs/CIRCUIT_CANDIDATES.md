# Circuit candidates: what is known vs what still needs tracing

This file is deliberately a **candidate ledger**, not a sealed circuit manifest.

## Confirmed steering anchor: DNa02

The current MaleCNS v1.0 Cell Type Explorer reports:

- `DNa02` contains exactly **2 neurons**, one left and one right;
- the aggregate type is predicted **ACh (acetylcholine), 94.6% confidence**;
- the aggregate DNa02 population receives **368 connections from 24 PFL3 neurons**;
- the left DNa02 page specifically shows input from contralateral `PFL3 (R)`;
- DNa02 projects into VNC circuitry associated with steering-related leg control.

Sources:

- https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/DNa02.html
- https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/DNa02_L.html

These values are useful **release-specific structural anchors**, not a complete functional model. They make DNa02 a defensible bilateral steering readout candidate, but they do not by themselves prove that odor evidence reaches it through the particular path we want to model.

## Science-v1 broad-corridor result

The corrected local MaleCNS trace at checkout `c1af7604ee50611bf6003f4dfeee0777fbfca040` used `max_hops=6`, `min_weight=5`, and `fanout_per_node=30` from broad ORN candidates to bilateral DNa02. The structural audit passed after preserving four unannotated body IDs.

Authoritative counts from the handoff artifact:

- 2,635 input source candidates;
- 2,228 retained source seeds;
- 2 retained DNa02 targets;
- 9,336 structural corridor nodes;
- 9,332 annotated corridor nodes;
- 4 structurally present but unannotated body IDs;
- 222,031 saved edges satisfying the corrected threshold and bounded-path policy.

This supersedes the earlier provisional 641,962-edge count.

### Source-selection finding

The legacy source patterns were `ORN`, `^Or`, and `^Ir`. In MaleCNS annotations used by the handoff:

- `ORN` matched 2,635 neurons;
- `^Or` matched the same 2,635 neurons because matching is case-insensitive and `ORN_*` begins with `Or`;
- `^Ir` matched zero neurons because the free-text matcher searches type/instance/class/subclass, not `receptorType`.

Therefore this source specification is effectively one broad ORN set, not three independently defined receptor populations. Do not use the legacy regex trio as a sealed sensory manifest.

### Named candidate occupancy in the broad corridor

The handoff found:

- `FB5AB`: 2 / 2 neurons retained;
- `PFL3`: 4 / 24 neurons retained;
- `DNa02`: 2 / 2 neurons retained;
- `PFNa`: 0 / 58 retained;
- `PFNm` family: 0 / 47 retained;
- `PFNp` family: 0 / 291 retained;
- `hDeltaC`: 0 / 20 retained.

`FB5AB` and the retained `PFL3` cells sit on six-hop bounded paths. Their presence does not define a mechanistic chain. Conversely, absence of `hDeltaC` or PFNs from this particular corridor is not evidence against the published wind-guided navigation mechanism because the search is optimized for short strong source-to-DNa02 paths and applies local fanout pruning.

## Central-complex literature constraints

Matheson et al. 2022 reports that attractive odor reaches fan-shaped-body tangential inputs including FB5AB, hDeltaC integrates odor-pathway and wind-sensitive PFN input, and hDeltaC output contributes to PFL2/PFL3 steering computations. The study used hemibrain connectomics plus functional experiments and therefore supplies a biological prior, not MaleCNS body-ID proof.

Recent steering work reports direct PFL3 input to DNa02 plus indirect PFL3 routes through DNa03 and LAL010. These route motifs are useful preregistered structural predictions to test in MaleCNS without shortest-path or fanout pruning.

References:

- Matheson et al. 2022, Nature Communications, DOI 10.1038/s41467-022-32247-7
- Rayshubskiy et al. / eLife 102230, steering-control circuit analysis

## Literature-guided route audit v1

`configs/literature_route_audit_v1.json` freezes the next descriptive structural test before inspecting its output. It audits exact named populations and raw body-ID connectivity for:

- `FB5AB -> hDeltaC`;
- `hDeltaC -> PFL3`;
- `PFL3 -> DNa02`;
- `PFL3 -> DNa03`;
- `PFL3 -> LAL010`;
- `DNa03 -> DNa02`;
- `LAL010 -> DNa02`;
- `PFL2 -> DNa03`;
- `PFL2 -> LAL010`;
- `MBON32 -> DNa02`.

It also audits the two explicit indirect motifs:

- `PFL3 -> DNa03 -> DNa02`;
- `PFL3 -> LAL010 -> DNa02`.

The tool reports exact body-ID edges and fixed descriptive threshold sweeps at 1, 3, 5, and 10 synapses. These thresholds are not qualification gates and must not be changed after seeing the result to rescue a preferred route.

Run locally with:

```bash
bash scripts/run_literature_route_audit.sh
```

The output is `results/route/literature-route-audit-v1.json`.

## The hardest part: sensory-to-navigation tracing

The project must **not** pretend there is a simple ORN -> hDeltaC -> PFL3 -> DNa02 chain unless MaleCNS connectivity supports the relevant staged motifs. Olfactory receptor neurons feed antennal-lobe circuitry; odor-guided navigation signals can reach steering systems through multiple parallel routes. Therefore:

1. resolve a non-overlapping sensory source manifest from actual MaleCNS annotation fields;
2. audit literature-defined odor projection-neuron populations and wind-sensitive PFN populations separately;
3. test staged structural motifs directly rather than relying on one shortest source-to-motor corridor;
4. inspect exact left/right body-ID connectivity into PFL3, DNa03, LAL010, and DNa02;
5. acquire transmitter/sign provenance from the same release before a signed dynamics graph is sealed;
6. perturb candidate roles and compare matched intact versus rewired graphs under the frozen benchmark.

If a literature-predicted motif is absent in MaleCNS, report that result instead of substituting another release or relaxing thresholds post hoc.

## Qualification rule

A candidate becomes a benchmark role only when the body IDs, side labels, upstream dataset, edge extraction, transmitter/sign provenance, and file hashes are sealed in a reviewed manifest. Until then, all figures say `candidate` or `proxy`.
