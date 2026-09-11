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

## Central-complex candidates

The discovery set includes `hDelta*`, `PFN*`, `PFL2`, and `PFL3` families because prior Drosophila navigation work implicates fan-shaped-body and central-complex populations in wind/odor integration and steering. `PFL3 → DNa02` connectivity is directly visible in MaleCNS v1.0, but the exact PFL3 subtypes/body IDs and upstream odor/wind corridor still require review from the same release.

The project must not substitute a circuit reconstructed from another fly connectome release for MaleCNS body IDs simply because the type names look familiar.

## The hardest part: sensory-to-navigation tracing

The project must **not** pretend there is a simple ORN → hDeltaC → PFL3 → DNa02 chain unless MaleCNS connectivity supports it. Olfactory receptor neurons feed antennal-lobe circuitry; odor-guided navigation signals can reach central-complex circuitry through multiple intermediate populations. Therefore:

1. identify annotated olfactory sensory/projection populations in MaleCNS;
2. identify the wind-direction input populations used by the relevant fan-shaped-body circuit;
3. trace weighted multi-hop connectivity toward the selected central-complex populations;
4. trace from those populations through `PFL3`/related outputs toward bilateral `DNa02`;
5. report every included intermediate population and every threshold used to retain it;
6. distinguish a structural corridor from evidence of physiological influence.

If no compact structural path supports the proposed functional chain, that is a result and the circuit hypothesis must change.

## Qualification rule

A candidate becomes a benchmark role only when the body IDs, side labels, upstream dataset, edge extraction, transmitter/sign provenance, and file hashes are sealed in a reviewed manifest. Until then, all figures say `candidate` or `proxy`.
