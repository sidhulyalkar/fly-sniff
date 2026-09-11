# Circuit candidates: what is known vs what still needs tracing

This file is deliberately a **candidate ledger**, not a sealed circuit manifest.

## Confirmed steering anchor: DNa02

The MaleCNS Cell Type Explorer currently reports:

- `DNa02` contains two neurons, one left and one right;
- the aggregate type is predicted cholinergic with high confidence;
- `PFL3` is an upstream partner of DNa02;
- the left DNa02 page specifically shows input from contralateral `PFL3 (R)`;
- DNa02 projects into VNC circuitry associated with steering-related leg control.

Sources:

- https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/DNa02.html
- https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/DNa02_L.html

This makes DNa02 a defensible **output readout candidate**, but it does not by itself prove that odor evidence reaches it through the particular path we want to model.

## Central-complex candidates

The discovery set includes `hDelta*`, `PFN*`, `PFL2`, and `PFL3` families because prior Drosophila navigation work implicates fan-shaped-body and central-complex populations in wind/odor integration and steering. The exact MaleCNS body IDs and the relevant subtypes must be selected from the v1.0 annotations and connectivity, not copied from another connectome release.

## The hardest part: sensory-to-navigation tracing

The project must **not** pretend there is a simple ORN → hDeltaC → PFL3 → DNa02 chain unless MaleCNS connectivity supports it. Olfactory receptor neurons feed antennal-lobe circuitry; odor-guided navigation signals can reach central-complex circuitry through multiple intermediate populations. Therefore:

1. identify annotated olfactory sensory/projection populations in MaleCNS;
2. identify the wind-direction input populations used by the relevant fan-shaped-body circuit;
3. trace weighted multi-hop connectivity toward the selected central-complex populations;
4. trace from those populations through `PFL3`/related outputs toward bilateral `DNa02`;
5. report every included intermediate population and every threshold used to retain it.

If no compact structural path supports the proposed functional chain, that is a result and the circuit hypothesis must change.

## Qualification rule

A candidate becomes a benchmark role only when the body IDs, side labels, upstream dataset, edge extraction, and file hashes are sealed in a reviewed manifest. Until then, all figures say `candidate` or `proxy`.
