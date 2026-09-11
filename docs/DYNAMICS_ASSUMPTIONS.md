# Neural-dynamics assumptions

MaleCNS is a structural connectome. `fly-sniff` therefore separates **measured/reconstructed structure** from **modeled dynamics**.

## v0 signed rate model

The first graph controller uses:

- MaleCNS nodes and weighted directed edges as structural authority;
- `log(1 + synapse_count)` as a bounded structural weight transform;
- an explicit presynaptic transmitter-derived sign policy;
- a single global leak and gain;
- `tanh` state nonlinearity;
- role-based sensory drive and bilateral descending readout.

None of these state values are labelled spikes, membrane voltage, calcium, or measured neural activity.

## Conservative sign policy

For the first sensitivity analysis:

- acetylcholine / ACh: `+1`;
- GABA: `-1`;
- glutamate: `0` (unresolved in v0 rather than assumed inhibitory);
- dopamine, serotonin, octopamine, unknown: `0`.

A zero sign means that edge is structurally present but contributes no drive in this particular signed-dynamics model. The signed-edge coverage fraction must be reported with every experiment.

This policy is an **assumption**, not a claim that transmitter identity alone determines every receptor-level postsynaptic effect. Later experiments should test alternative receptor-informed or sensitivity-analysis policies.

## Qualification guard

`MaleCNSRateController` refuses to initialize in qualified mode unless:

1. `edges.parquet` contains an explicit `sign` column constrained to `{-1, 0, +1}`;
2. the graph bundle has a `manifest.json` whose `qualification_status` is exactly `qualified`;
3. every role body ID exists in the graph.

Candidate graphs can still be explored by explicitly disabling the qualification guard in development code, but they cannot silently become the public benchmark controller.
