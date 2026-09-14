# Topology null inference v1

This tranche implements the graph-level control machinery required before any strong statement that biological wiring outperforms scrambled wiring.

## Experimental unit

The independent unit for the topology claim is the **graph realization**, not the episode. Running 1,000 plume episodes on one intact graph does not create 1,000 independent topology samples.

Episodes are paired nuisance/measurement variation nested inside each graph realization. Every intact/null comparison must use the exact same frozen `condition_ids`.

## Null hierarchy

The generic null engine performs directed double-edge target swaps. Because the edge row remains attached to its original presynaptic source, all source-row attributes such as structural weight and sign remain attached to that source.

Supported families are:

- `directed_degree`: exact directed in/out degree preservation;
- `source_attribute`: same transformation with explicit source weight/sign invariant auditing;
- `hemisphere`: swaps only inside source-side × target-side blocks;
- `cell_type`: swaps only inside source-type × target-type blocks;
- `constrained`: arbitrary predeclared node metadata columns, useful later for neuropil/spatial bins.

Constrained families preserve exact edge counts in every source-metadata → target-metadata block in addition to directed degree.

Every output records source/output graph fingerprints, RNG seed, swap budget, accepted/attempted swaps, overlap with intact wiring, exact degree checks, source weight/sign multiset checks, constraint-block checks and whether the requested swap budget completed.

A fixed accepted-swap budget is a disruption protocol. It is **not evidence that a Markov chain mixed to stationarity**.

## Confirmatory inference

The v1 confirmatory minimum is 31 independently seeded null graphs **per prespecified null family**. Sixty-three is preferred for the flagship experiment when practical.

For a higher-is-better statistic:

```text
p = (1 + count(null_stat >= intact_stat)) / (N_null + 1)
```

Ties count against the intact graph. Lower-is-better metrics use the symmetric tail.

The receipt reports:

- the intact graph-level statistic;
- the complete null distribution;
- empirical tail probability;
- intact rank and percentile;
- signed effect relative to null mean and median;
- exact paired condition IDs;
- one declared null family.

Different null families are analyzed separately. We do not pool a permissive degree null and a stricter cell-type/hemisphere null into one opaque scramble distribution.

## Development versus confirmatory use

The existing eight rewires remain useful development controls. They do not satisfy the confirmatory topology-null requirement and may not be relabeled after observing results.

The family, count, transformation budget, graph source, condition set and statistic must be frozen before confirmatory performance is inspected. Nulls cannot be dropped because their outcome is inconvenient.

## What this does not establish

This infrastructure does not establish that any connectome-derived dynamics are biologically correct, that the odor circuit hypothesis is correct, or that MaleCNS wiring helps navigation. It only makes the topology-control and statistical layer explicit and auditable.
